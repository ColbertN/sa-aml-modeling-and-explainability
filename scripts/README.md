# Scripts

The scripts are intentionally organised around two reproducible entry points:

* `run_sequence_pipeline.py` runs account-level sequence preparation, GRU training, evaluation, calibration, and explainability.
* `run_transaction_xgb.py` runs the transaction-level XGBoost benchmark, evaluation, calibration, and explainability.
* `validate_simulation.py` performs the pre-modelling statistical, temporal, structural, and optional sampling/reference validation.

The remaining scripts are the stages called by those entry points. They can be run individually when debugging or regenerating one artefact family. All scripts resolve paths relative to the repository root so that the workflow is portable across machines.

Before running, verify the input data path and simulator release. Do not change the test set after threshold selection, and do not fit preprocessing or calibration objects on test data.

Run the simulator validation before training:

```bash
python scripts/validate_simulation.py
```

Optional comparisons can be supplied with `--reference path/to/reference.csv` and `--sample path/to/negative_sample.csv`. Outputs are written to `results/simulation_validation/` as independent figures, CSV tables, and `validation_summary.json`.
