# Public entry point

The repository deliberately keeps the public command surface small:

```bash
python scripts/run_all.py
```

This runs the complete workflow in order:

1. Mathematical and statistical simulation validation.
2. Account-level sequence preprocessing, GRU training, evaluation, calibration, and explainability.
3. Transaction-level XGBoost benchmarking, evaluation, calibration, and explainability.

The implementation modules and individual workflow stages are stored in [`src/`](../src/). They can be run separately for debugging, but `run_all.py` is the recommended reproducible entry point. Set `AML_RAW_DATA` when the raw CSV is stored outside `data/raw/SA_SAML_D.csv`.
