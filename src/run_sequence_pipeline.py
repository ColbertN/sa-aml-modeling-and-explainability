"""Run the complete Account_id GRU workflow and all independent artefacts."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for name in [
    "train_sequence_from_scratch.py",
    "make_sequence_figures.py",
    "calibration_explainability.py",
    "make_explainability_style.py",
    "make_event_contribution_views.py",
]:
    print(f"\n===== {name} =====", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src" / name)], cwd=ROOT, check=True)
print("\nAccount-level sequence workflow completed.")
