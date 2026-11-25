"""Score a new transaction CSV with the persisted transaction XGBoost model.

Usage: py -3.11 scripts/predict_transaction_xgb.py input.csv output.csv
"""
from pathlib import Path
import sys
import joblib
import pandas as pd
from xgboost import XGBClassifier
from train_transaction_xgb import engineer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "transaction_xgb"


def main(input_path, output_path):
    raw = pd.read_csv(input_path, low_memory=False)
    engineered = engineer(raw)
    bundle = joblib.load(MODEL_DIR / "preprocessing.joblib")
    pipeline = bundle["pipeline"]
    features = bundle["numeric"] + bundle["categorical"]
    X = pipeline.transform(engineered[features])
    model = XGBClassifier()
    model.load_model(MODEL_DIR / "model.json")
    probability = model.predict_proba(X)[:, 1]
    manifest = __import__("json").loads((MODEL_DIR / "model_manifest.json").read_text(encoding="utf-8"))
    threshold = float(manifest["metrics"]["threshold"])
    out = engineered[[c for c in ["Transaction_id", "Account_id", "event_time"] if c in engineered]].copy()
    out["probability_laundering"] = probability
    out["prediction"] = (probability >= threshold).astype(int)
    out.to_csv(output_path, index=False)
    print(f"Saved {len(out):,} transaction predictions to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: py -3.11 scripts/predict_transaction_xgb.py input.csv output.csv")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
