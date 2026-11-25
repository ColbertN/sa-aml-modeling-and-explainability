"""Post-hoc account error analysis for the frozen 100-event GRU test run.

Simulator scenario fields are used only for diagnosis here; they are never
passed to the model.  The outputs identify error concentration and split
representation so future modelling changes can be justified without leakage.
"""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/SA_SAML_D.csv"
EXPERIMENT = os.environ.get("AML_SEQUENCE_EXPERIMENT", "sequence")
PRED = ROOT / f"results/{EXPERIMENT}/test_predictions.csv"
OUT = ROOT / f"results/{EXPERIMENT}/error_analysis"
OUT.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(PRED)
pred["Account_id"] = pred["Account_id"].astype(str)
raw = pd.read_csv(RAW, low_memory=False)
raw["Account_id"] = raw["Account_id"].astype(str)
raw["Amount"] = pd.to_numeric(raw["Amount"], errors="coerce").fillna(0)
raw["currency_mismatch"] = (raw["Payment_currency"].astype(str) != raw["Received_currency"].astype(str)).astype(int)

def mode_value(x):
    x = x.dropna().astype(str)
    return x.value_counts().index[0] if len(x) else "None"

def type_string(x):
    x = x.dropna().astype(str)
    if not len(x): return "None"
    return "; ".join(f"{k} ({v})" for k, v in x.value_counts().items())

account = raw.groupby("Account_id", sort=False).agg(
    transaction_count=("Transaction_id", "size"),
    laundering_event_count=("Is_laundering", "sum"),
    laundering_event_rate=("Is_laundering", "mean"),
    mean_amount=("Amount", "mean"), max_amount=("Amount", "max"), total_amount=("Amount", "sum"),
    cross_border_rate=("Cross_border_transaction", "mean"), sadc_rate=("SADC_corridor_transaction", "mean"),
    currency_mismatch_rate=("currency_mismatch", "mean"),
    dominant_payment_type=("Payment_type", mode_value), dominant_jurisdiction=("Jurisdiction_risk_tier", mode_value),
    dominant_counterparty_country=("Counterparty_country", mode_value), dominant_customer_type=("Customer_type", mode_value),
    laundering_types=("Laundering_type", type_string),
).reset_index()

for col in ["Laundering_type", "Scenario_detail", "AML_scenario_category", "AML_scenario_code"]:
    if col in raw:
        account[col.lower()] = raw.groupby("Account_id", sort=False)[col].agg(type_string).reindex(account.Account_id).fillna("None").to_numpy()

errors = pred.merge(account, on="Account_id", how="left")
errors["error_type"] = np.select([
    (errors.y_true == 0) & (errors.prediction == 1),
    (errors.y_true == 1) & (errors.prediction == 0),
    (errors.y_true == 1) & (errors.prediction == 1),
    (errors.y_true == 0) & (errors.prediction == 0),
], ["false_positive", "false_negative", "true_positive", "true_negative"], default="unknown")
errors[errors.error_type == "false_positive"].sort_values("probability", ascending=False).to_csv(OUT / "false_positive_accounts.csv", index=False)
errors[errors.error_type == "false_negative"].sort_values("probability").to_csv(OUT / "false_negative_accounts.csv", index=False)

positive = errors[errors.y_true == 1].copy()
# A laundering account can also contain many legitimate transactions.  The
# primary type must therefore be derived only from laundering-labelled rows,
# never from the modal type across the complete account history.
laundering_type_by_account = raw[raw.Is_laundering == 1].groupby("Account_id")["Laundering_type"].agg(mode_value)
positive["primary_laundering_type"] = positive.Account_id.map(laundering_type_by_account).fillna("Unknown")
type_rows = []
for typ, g in positive.groupby("primary_laundering_type", dropna=False):
    type_rows.append({"primary_laundering_type": typ, "test_accounts": len(g), "false_negatives": int((g.prediction == 0).sum()), "true_positives": int((g.prediction == 1).sum()), "recall": float((g.prediction == 1).mean()), "mean_transactions": float(g.transaction_count.mean()), "median_transactions": float(g.transaction_count.median()), "mean_probability": float(g.probability.mean())})
type_results = pd.DataFrame(type_rows).sort_values(["recall", "test_accounts"])
type_results.to_csv(OUT / "laundering_type_error_rates.csv", index=False)

cohorts = {name: pd.read_csv(ROOT / f"data/sequence/{name}_accounts.csv", usecols=["Account_id", "account_target"]) for name in ("train", "validation", "test")}
raw_pos = raw[raw.Is_laundering == 1].copy()
raw_pos["Account_id"] = raw_pos.Account_id.astype(str)
raw_pos["primary_type"] = raw_pos.groupby("Account_id")["Laundering_type"].transform(mode_value)
split_rows = []
for split, cohort in cohorts.items():
    ids = set(cohort.Account_id.astype(str))
    c = cohort[cohort.account_target == 1]
    type_map = raw_pos[raw_pos.Account_id.isin(ids)].groupby("Account_id")["Laundering_type"].agg(mode_value)
    for typ, n in type_map.value_counts().items():
        split_rows.append({"split": split, "primary_laundering_type": typ, "positive_accounts": int(n), "share_of_positive_accounts": float(n / max(len(c), 1))})
    split_rows.append({"split": split, "primary_laundering_type": "ALL_POSITIVE", "positive_accounts": int(len(c)), "share_of_positive_accounts": 1.0})
representation = pd.DataFrame(split_rows)
representation.to_csv(OUT / "laundering_type_representation_by_split.csv", index=False)

fp = errors[errors.error_type == "false_positive"]
fn = errors[errors.error_type == "false_negative"]
summary = {
    "test_accounts": int(len(errors)), "false_positive_accounts": int(len(fp)), "false_negative_accounts": int(len(fn)),
    "false_positive_rate_among_normal": float(len(fp) / max((errors.y_true == 0).sum(), 1)),
    "false_negative_rate_among_laundering": float(len(fn) / max((errors.y_true == 1).sum(), 1)),
    "false_positive_mean_transactions": float(fp.transaction_count.mean()), "false_positive_median_transactions": float(fp.transaction_count.median()),
    "false_negative_mean_transactions": float(fn.transaction_count.mean()), "false_negative_median_transactions": float(fn.transaction_count.median()),
    "false_positive_dominant_payment_types": fp.dominant_payment_type.value_counts().head(10).to_dict(),
    "false_negative_type_error_rates_file": "laundering_type_error_rates.csv",
    "split_representation_file": "laundering_type_representation_by_split.csv",
    "interpretation": "Laundering type and simulator scenario columns are diagnostic metadata only and were not model inputs."
}
(OUT / "error_analysis_summary.json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
print(json.dumps(summary, indent=2, default=int))
print("\\nLaundering-type error rates:")
print(type_results.to_string(index=False))
