# Results

This directory contains compact, reader-facing reference outputs. Large prediction files, threshold grids, raw explanation arrays, and model binaries are excluded by `.gitignore`.

* `figures/sequence/` contains account-level GRU performance, calibration, and event-feature explanation figures.
* `figures/transaction_xgb/` contains transaction-level XGBoost benchmark figures.
* `tables/` contains small CSV snapshots of the current test metrics and full class reports.

To regenerate outputs, run the corresponding entry-point script from the repository root. Results should always be interpreted with their prediction unit: account for the GRU and transaction for XGBoost.
