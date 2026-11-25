"""Statistical, temporal, and structural validation for the synthetic AML data.

This stage is deliberately independent of model training.  It checks whether
the generated population is internally coherent and whether an optional
reference or sampled population is distributionally comparable.  It writes
CSV/JSON summaries and one figure per diagnostic under
``results/validation``.

Examples
--------
python scripts/run_all.py
python src/validate_simulation.py --reference path/to/reference.csv
python src/validate_simulation.py --sample path/to/negative_sample.csv

The raw CSV is not committed to Git.  Set ``AML_RAW_DATA`` or pass ``--raw``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw" / "SA_SAML_D.csv"
DEFAULT_OUT = ROOT / "results" / "validation"
TARGET = "Is_laundering"
ACCOUNT = "Account_id"
CATEGORY_COLUMNS = [
    "Direction", "Payment_type", "Payment_currency", "Received_currency",
    "Sender_bank_location", "Receiver_bank_location", "Counterparty_country",
    "Counterparty_type", "Jurisdiction_risk_tier", "Customer_type",
    "Employment_status", "Province", "Social_grant_type",
    "Financial_inclusion_segment", "Channel_preference",
]
NUMERIC_COLUMNS = [
    "Amount", "Age", "Monthly_income_estimate", "Account_age_days",
    "Business_age_years", "Connected_party_count",
]


def cramers_v(table: pd.DataFrame) -> float:
    """Return Cramer's V for a contingency table."""
    if table.empty or table.to_numpy().sum() == 0:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    r, c = table.shape
    denominator = n * max(1, min(r - 1, c - 1))
    return float(np.sqrt(chi2 / denominator))


def safe_datetime(frame: pd.DataFrame) -> pd.Series:
    if "fulldatetime" in frame:
        return pd.to_datetime(frame["fulldatetime"], errors="coerce")
    date = frame.get("Date", pd.Series(index=frame.index, dtype="object")).astype(str)
    time = frame.get("Time", pd.Series(index=frame.index, dtype="object")).astype(str)
    return pd.to_datetime(date + " " + time, errors="coerce")


def account_labels(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby(ACCOUNT, dropna=False)[TARGET].max().astype(int)


def summary(frame: pd.DataFrame) -> dict:
    labels = account_labels(frame)
    return {
        "transactions": int(len(frame)),
        "accounts": int(labels.size),
        "laundering_accounts": int(labels.sum()),
        "normal_accounts": int((labels == 0).sum()),
        "laundering_account_rate": float(labels.mean()),
        "laundering_transaction_rate": float(pd.to_numeric(frame[TARGET], errors="coerce").mean()),
        "missing_cells": int(frame.isna().sum().sum()),
        "columns": int(frame.shape[1]),
    }


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in NUMERIC_COLUMNS:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append({
            "feature": column,
            "n": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "q01": float(values.quantile(.01)),
            "q25": float(values.quantile(.25)),
            "median": float(values.median()),
            "q75": float(values.quantile(.75)),
            "q99": float(values.quantile(.99)),
            "min": float(values.min()),
            "max": float(values.max()),
        })
    return pd.DataFrame(rows)


def categorical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = frame[TARGET].astype(int) if TARGET in frame else None
    for column in CATEGORY_COLUMNS:
        if column not in frame:
            continue
        values = frame[column].fillna("__MISSING__").astype(str)
        counts = values.value_counts(dropna=False)
        suspicious_only = []
        normal_only = []
        if labels is not None:
            for category in counts.index:
                mask = values.eq(category)
                if labels[mask].sum() == 0:
                    normal_only.append(category)
                if labels[mask].sum() == mask.sum():
                    suspicious_only.append(category)
        rows.append({
            "feature": column,
            "unique_values": int(values.nunique(dropna=False)),
            "most_common_value": str(counts.index[0]),
            "most_common_share": float(counts.iloc[0] / len(values)),
            "normal_only_categories": ", ".join(normal_only[:20]),
            "suspicious_only_categories": ", ".join(suspicious_only[:20]),
        })
    return pd.DataFrame(rows)


def temporal_summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[[ACCOUNT]].copy()
    work["event_time"] = safe_datetime(frame).to_numpy()
    work = work.dropna().sort_values([ACCOUNT, "event_time"])
    gaps = work.groupby(ACCOUNT)["event_time"].diff().dt.total_seconds().dropna()
    if gaps.empty:
        return pd.DataFrame()
    grouped = gaps.groupby(work.loc[gaps.index, ACCOUNT])
    burstiness = grouped.agg(lambda x: (x.std() - x.mean()) / (x.std() + x.mean()) if (x.std() + x.mean()) else 0)
    return pd.DataFrame([{
        "interarrival_n": int(gaps.size),
        "interarrival_mean_seconds": float(gaps.mean()),
        "interarrival_median_seconds": float(gaps.median()),
        "interarrival_q90_seconds": float(gaps.quantile(.90)),
        "interarrival_q99_seconds": float(gaps.quantile(.99)),
        "burstiness_mean": float(burstiness.mean()),
        "burstiness_median": float(burstiness.median()),
    }])


def compare_distributions(current: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_rows = []
    for column in NUMERIC_COLUMNS:
        if column not in current or column not in reference:
            continue
        a = pd.to_numeric(current[column], errors="coerce").dropna()
        b = pd.to_numeric(reference[column], errors="coerce").dropna()
        if a.empty or b.empty:
            continue
        ks = ks_2samp(a, b)
        numeric_rows.append({
            "feature": column, "current_n": int(len(a)), "reference_n": int(len(b)),
            "ks_statistic": float(ks.statistic), "ks_p_value": float(ks.pvalue),
            "wasserstein_distance": float(wasserstein_distance(a, b)),
            "interpretation": "PASS" if ks.pvalue >= .05 else "REVIEW",
        })
    categorical_rows = []
    for column in CATEGORY_COLUMNS:
        if column not in current or column not in reference:
            continue
        a = current[column].fillna("__MISSING__").astype(str).value_counts()
        b = reference[column].fillna("__MISSING__").astype(str).value_counts()
        table = pd.DataFrame({"current": a, "reference": b}).fillna(0)
        if table.shape[0] < 2:
            continue
        chi2, p, _, _ = chi2_contingency(table.to_numpy())
        categorical_rows.append({
            "feature": column, "chi2": float(chi2), "p_value": float(p),
            "cramers_v": cramers_v(table), "categories_compared": int(len(table)),
            "interpretation": "PASS" if p >= .05 else "REVIEW",
        })
    return pd.DataFrame(numeric_rows), pd.DataFrame(categorical_rows)


def sampling_comparison(full: pd.DataFrame, sampled: pd.DataFrame) -> pd.DataFrame:
    """Compare negative account populations when a sampled file is supplied."""
    full_labels = account_labels(full)
    sampled_labels = account_labels(sampled)
    full_negative = full[full[ACCOUNT].isin(full_labels[full_labels == 0].index)]
    sampled_negative = sampled[sampled[ACCOUNT].isin(sampled_labels[sampled_labels == 0].index)]
    rows = []
    numeric = [c for c in NUMERIC_COLUMNS if c in full_negative and c in sampled_negative]
    for column in numeric:
        a = pd.to_numeric(full_negative[column], errors="coerce").dropna()
        b = pd.to_numeric(sampled_negative[column], errors="coerce").dropna()
        if a.empty or b.empty:
            continue
        test = ks_2samp(a, b)
        rows.append({"feature": column, "test": "KS", "statistic": float(test.statistic), "p_value": float(test.pvalue), "interpretation": "PASS" if test.pvalue >= .05 else "REVIEW"})
    for column in [c for c in CATEGORY_COLUMNS if c in full_negative and c in sampled_negative]:
        a = full_negative[column].fillna("__MISSING__").astype(str).value_counts()
        b = sampled_negative[column].fillna("__MISSING__").astype(str).value_counts()
        table = pd.DataFrame({"full_negative": a, "sampled_negative": b}).fillna(0)
        if table.shape[0] < 2:
            continue
        chi2, p, _, _ = chi2_contingency(table.to_numpy())
        rows.append({"feature": column, "test": "chi-square", "statistic": float(chi2), "p_value": float(p), "interpretation": "PASS" if p >= .05 else "REVIEW"})
    return pd.DataFrame(rows)


def labelled_bar(values: pd.Series, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(values.index.astype(str), values.values, color="#2b6cb0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    for bar, value in zip(bars, values.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    raw_path = Path(args.raw or os.environ.get("AML_RAW_DATA", DEFAULT_RAW))
    out = Path(args.output or os.environ.get("AML_VALIDATION_OUTPUT", DEFAULT_OUT))
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(raw_path, low_memory=False)
    result = {"raw_path": str(raw_path), "summary": summary(frame), "validation": {}}
    population_table = pd.DataFrame([result["summary"]])
    numeric_table = numeric_summary(frame)
    categorical_table = categorical_summary(frame)
    temporal_table = temporal_summary(frame)
    population_table.to_csv(out / "population_summary.csv", index=False)
    numeric_table.to_csv(out / "numeric_summary.csv", index=False)
    categorical_table.to_csv(out / "categorical_summary.csv", index=False)
    temporal_table.to_csv(out / "temporal_summary.csv", index=False)
    try:
        with pd.ExcelWriter(out / "validation_results.xlsx", engine="openpyxl") as writer:
            population_table.to_excel(writer, sheet_name="population", index=False)
            numeric_table.to_excel(writer, sheet_name="numeric", index=False)
            categorical_table.to_excel(writer, sheet_name="categorical", index=False)
            temporal_table.to_excel(writer, sheet_name="temporal", index=False)
    except ImportError:
        result["excel_warning"] = "openpyxl is not installed; CSV outputs remain available."

    labels = account_labels(frame)
    labelled_bar(labels.value_counts().sort_index(), "Account-level class prevalence", "Number of accounts", fig_dir / "account_class_prevalence.png")
    account_sizes = frame.groupby(ACCOUNT).size().clip(upper=100).value_counts().sort_index().head(20)
    labelled_bar(account_sizes, "Account history length (first 20 bins)", "Number of accounts", fig_dir / "account_history_lengths.png")

    if args.reference:
        reference = pd.read_csv(args.reference, low_memory=False)
        n, c = compare_distributions(frame, reference)
        n.to_csv(out / "reference_numeric_comparison.csv", index=False)
        c.to_csv(out / "reference_categorical_comparison.csv", index=False)
        result["validation"]["reference"] = {"numeric_tests": int(len(n)), "categorical_tests": int(len(c)), "pass_rule": "p >= 0.05; inspect effect sizes and Wasserstein distance"}
    if args.sample:
        sampled = pd.read_csv(args.sample, low_memory=False)
        s = sampling_comparison(frame, sampled)
        s.to_csv(out / "negative_sampling_comparison.csv", index=False)
        result["validation"]["negative_sampling"] = {"tests": int(len(s)), "pass_rule": "p >= 0.05; inspect effect sizes and category coverage"}

    (out / "validation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", help="Raw simulator CSV; defaults to AML_RAW_DATA or data/raw/SA_SAML_D.csv")
    parser.add_argument("--reference", help="Optional reference CSV for KS/Wasserstein/chi-square comparisons")
    parser.add_argument("--sample", help="Optional sampled CSV for negative-population representativeness checks")
    parser.add_argument("--output", help="Output directory; defaults to results/validation")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
