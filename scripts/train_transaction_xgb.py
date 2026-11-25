"""Leakage-aware transaction-level XGBoost experiment.

The target is the transaction label ``Is_laundering``.  Account profiles are
split before rows are sampled, so a customer's transactions cannot appear in
more than one cohort.  Only negative training rows are undersampled; validation
and test retain their natural transaction distributions.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "SA_SAML_D.csv"
DATA = ROOT / "data" / "transaction_xgb"
RESULTS = ROOT / "results" / "transaction_xgb"
MODEL_DIR = ROOT / "models" / "transaction_xgb"
TARGET = "Is_laundering"
SEED = 42
NEGATIVE_RATIO = 10
TRIALS = int(os.environ.get("XGB_OPTUNA_TRIALS", "4"))

LEAK = {
    "Is_laundering", "Account_is_laundering", "Laundering_type",
    "AML_scenario_category", "AML_scenario_code", "AML_predicate_offence",
    "AML_stage", "AML_risk_sector", "AML_scenario_source_keys",
    "Scenario_reference_key", "Scenario_detail", "Scenario_label",
    "Suspicion_indicator", "Behaviour_component", "Typology_stage",
    "Legitimate_behaviour", "Normal_behavior_category", "Normal_behavior_source_keys",
    "Opened_for_laundering_flag", "Scenario_forced_coverage_flag",
    "Account_laundering_probability", "First_observed_transaction_date",
    "Opened_during_observation_flag", "Active_observation_days",
    "Account_activity_multiplier", "Expected_transaction_count",
    "Generated_transaction_count", "Counterparty_type", "KYC_strength",
    "Beneficial_ownership_opacity", "Transaction_purpose_category",
}
IDS = {"Transaction_id", "Account_id", "Sender_account", "Receiver_account", "Counterparty_account"}

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
    df = df.dropna(subset=["event_time", "Account_id"]).sort_values(["Account_id", "event_time", "Transaction_id"], kind="stable").reset_index(drop=True)
    g = df.groupby("Account_id", sort=False)
    df["hour"] = df.event_time.dt.hour; df["weekday"] = df.event_time.dt.dayofweek; df["month"] = df.event_time.dt.month
    df["hour_sin"] = np.sin(2*np.pi*df.hour/24); df["hour_cos"] = np.cos(2*np.pi*df.hour/24)
    df["weekday_sin"] = np.sin(2*np.pi*df.weekday/7); df["weekday_cos"] = np.cos(2*np.pi*df.weekday/7)
    df["month_sin"] = np.sin(2*np.pi*df.month/12); df["month_cos"] = np.cos(2*np.pi*df.month/12)
    df["amount_log"] = np.log1p(df["Amount"].clip(lower=0))
    df["seconds_since_previous"] = g.event_time.diff().dt.total_seconds().fillna(0).clip(lower=0)
    df["previous_amount"] = g.Amount.shift(1).fillna(0); df["amount_change"] = df.Amount - df.previous_amount
    df["rolling_amount_mean"] = g.Amount.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
    df["rolling_amount_std"] = g.Amount.transform(lambda x: x.shift(1).rolling(5, min_periods=1).std()).fillna(0)
    df["txn_index"] = g.cumcount(); df["currency_mismatch"] = (df.Payment_currency.astype(str) != df.Received_currency.astype(str)).astype("int8")
    return df

def split_accounts(df):
    labels = df.groupby("Account_id")[TARGET].max().reset_index(name="account_target")
    dev, test = train_test_split(labels, test_size=.20, stratify=labels.account_target, random_state=SEED)
    train, val = train_test_split(dev, test_size=.20, stratify=dev.account_target, random_state=SEED)
    return train, val, test

def sample_training_rows(train_df):
    pos = train_df[train_df[TARGET] == 1]
    neg = train_df[train_df[TARGET] == 0]
    n = min(len(neg), NEGATIVE_RATIO * len(pos))
    key = (neg.get("Payment_type", "").astype(str) + "|" + neg.get("Receiver_bank_location", "").astype(str))
    sampled_neg = train_test_split(neg, train_size=n, stratify=key if key.nunique() > 1 else None, random_state=SEED)[0]
    return pd.concat([pos, sampled_neg], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

def make_features(train_df, val_df, test_df):
    excluded = LEAK | IDS | {"Date", "Time", "event_time"}
    candidates = [c for c in train_df.columns if c not in excluded and not c.startswith("Unnamed")]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(train_df[c])]
    categorical = [c for c in candidates if c not in numeric and train_df[c].nunique(dropna=False) <= 100]
    # High-cardinality free text and arbitrary identifiers are intentionally not
    # sent to one-hot encoding.
    use = numeric + categorical
    prep = ColumnTransformer([("num", RobustScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32), categorical)], remainder="drop")
    Xtr = prep.fit_transform(train_df[use].copy()); Xv = prep.transform(val_df[use].copy()); Xt = prep.transform(test_df[use].copy())
    names = prep.get_feature_names_out().tolist()
    joblib.dump({"pipeline": prep, "numeric": numeric, "categorical": categorical, "feature_names": names, "excluded": sorted(excluded)}, MODEL_DIR / "preprocessing.joblib")
    return Xtr.tocsr(), Xv.tocsr(), Xt.tocsr(), names

def fit_model(Xtr, ytr, Xv, yv):
    def objective(trial):
        params = {"n_estimators": trial.suggest_int("n_estimators", 120, 320), "max_depth": trial.suggest_int("max_depth", 3, 7), "learning_rate": trial.suggest_float("learning_rate", .03, .20, log=True), "subsample": trial.suggest_float("subsample", .70, 1.0), "colsample_bytree": trial.suggest_float("colsample_bytree", .60, 1.0), "min_child_weight": trial.suggest_int("min_child_weight", 1, 15), "gamma": trial.suggest_float("gamma", 0, 2), "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10, log=True), "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 20)}
        model = XGBClassifier(**params, objective="binary:logistic", eval_metric="aucpr", tree_method="hist", n_jobs=8, random_state=SEED, early_stopping_rounds=35)
        model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        return average_precision_score(yv, model.predict_proba(Xv)[:, 1])
    if TRIALS <= 0:
        # Reusable fallback from the first completed Optuna trial.  This makes
        # reruns possible without repeating the expensive Windows CPU search.
        best = {"n_estimators":80,"max_depth":5,"learning_rate":0.0802866588,"subsample":0.90,"colsample_bytree":0.75,"min_child_weight":3,"gamma":0.1161672243,"reg_lambda":3.9676050771,"scale_pos_weight":12.4211852231,"max_bin":64}
        study = None
    else:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=TRIALS, show_progress_bar=False)
        best = dict(study.best_params)
    best.update({"objective":"binary:logistic", "eval_metric":"aucpr", "tree_method":"hist", "n_jobs":8, "random_state":SEED, "early_stopping_rounds":45})
    model = XGBClassifier(**best); model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    return model, study

def main():
    for p in [DATA, RESULTS, MODEL_DIR]: p.mkdir(parents=True, exist_ok=True)
    df = engineer(pd.read_csv(RAW, low_memory=False)); train_acc, val_acc, test_acc = split_accounts(df)
    train_df = df[df.Account_id.isin(train_acc.Account_id)].copy(); val_df = df[df.Account_id.isin(val_acc.Account_id)].copy(); test_df = df[df.Account_id.isin(test_acc.Account_id)].copy(); sampled = sample_training_rows(train_df)
    for name, frame in [("train_accounts", train_acc), ("validation_accounts", val_acc), ("test_accounts", test_acc)]: frame.to_csv(DATA/f"{name}.csv", index=False)
    sampled[["Transaction_id","Account_id",TARGET]].to_csv(DATA/"sampled_train_rows.csv", index=False)
    Xtr, Xv, Xt, names = make_features(sampled, val_df, test_df); ytr=sampled[TARGET].to_numpy(int); yv=val_df[TARGET].to_numpy(int); yt=test_df[TARGET].to_numpy(int)
    sp.save_npz(DATA/"X_train.npz", Xtr); sp.save_npz(DATA/"X_validation.npz", Xv); sp.save_npz(DATA/"X_test.npz", Xt); np.save(DATA/"y_train.npy",ytr); np.save(DATA/"y_validation.npy",yv); np.save(DATA/"y_test.npy",yt)
    model, study = fit_model(Xtr,ytr,Xv,yv); model.save_model(MODEL_DIR/"model.json")
    if study is not None: study.trials_dataframe().to_csv(RESULTS/"optuna_trials.csv",index=False)
    pv=model.predict_proba(Xv)[:,1]; pt=model.predict_proba(Xt)[:,1]; precision,recall,thresholds=precision_recall_curve(yv,pv); rows=[(t,p,r,2*p*r/max(p+r,1e-9)) for p,r,t in zip(precision[:-1],recall[:-1],thresholds)]; cand=pd.DataFrame(rows,columns=["threshold","precision","recall","f1"])
    # Select an operating point with a minimum precision target.  This is
    # appropriate for AML alerting, where false positives consume investigator
    # capacity.  If no point reaches the target, fall back to maximum F1.
    feasible=cand[cand.precision>=.8]
    chosen=(feasible.sort_values(["recall","f1"],ascending=False).iloc[0] if len(feasible) else cand.sort_values("f1",ascending=False).iloc[0]); threshold=float(chosen.threshold); pred=(pt>=threshold).astype(int)
    report=pd.DataFrame(classification_report(yt,pred,labels=[0,1],target_names=["Normal","Laundering"],output_dict=True,zero_division=0)).T; report.to_csv(RESULTS/"classification_report.csv"); tn,fp,fn,tp=confusion_matrix(yt,pred,labels=[0,1]).ravel(); metrics={"unit":"transaction","group_split":"Account_id","threshold":threshold,"roc_auc":roc_auc_score(yt,pt),"average_precision":average_precision_score(yt,pt),"true_negative":int(tn),"false_positive":int(fp),"false_negative":int(fn),"true_positive":int(tp),"precision_positive":float(report.loc["Laundering","precision"]),"recall_positive":float(report.loc["Laundering","recall"]),"train_rows":int(len(sampled)),"validation_rows":int(len(val_df)),"test_rows":int(len(test_df))}
    pd.DataFrame([metrics]).to_csv(RESULTS/"test_metrics.csv",index=False); cand.to_csv(RESULTS/"threshold_candidates.csv",index=False); pd.DataFrame({"Transaction_id":test_df.Transaction_id.to_numpy(),"Account_id":test_df.Account_id.to_numpy(),"y_true":yt,"probability":pt,"prediction":pred}).to_csv(RESULTS/"test_predictions.csv",index=False)
    (MODEL_DIR/"model_manifest.json").write_text(json.dumps({"unit":"transaction","group_key":"Account_id","target":TARGET,"negative_sampling_ratio":NEGATIVE_RATIO,"feature_names":names,"metrics":metrics,"leakage_excluded":sorted(LEAK|IDS)},indent=2),encoding="utf-8"); print(json.dumps(metrics,indent=2)); print(report.to_string())

if __name__ == "__main__": main()
