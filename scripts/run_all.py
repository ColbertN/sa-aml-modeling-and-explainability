"""Run simulation validation, account GRU modelling, and XGBoost benchmarking."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("Mathematical and statistical simulation validation", "validate_simulation.py"),
    ("Account-level sequential GRU workflow", "run_sequence_pipeline.py"),
    ("Transaction-level XGBoost benchmark", "run_transaction_xgb.py"),
]


def main() -> None:
    for title, filename in STAGES:
        print(f"\n===== {title} =====", flush=True)
        subprocess.run([sys.executable, str(ROOT / "src" / filename)], cwd=ROOT, check=True)
    print("\nComplete validation and modelling workflow finished.", flush=True)


if __name__ == "__main__":
    main()
