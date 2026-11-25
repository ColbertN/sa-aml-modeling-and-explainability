"""Run the complete transaction-level XGBoost workflow in order."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "train_transaction_xgb.py",
    "transaction_xgb_figures.py",
    "transaction_xgb_calibration.py",
    "transaction_xgb_explainability.py",
]

for step in STEPS:
    print(f"\n===== {step} =====", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / step)], cwd=ROOT, check=True)
print("\nTransaction XGBoost workflow completed.")
