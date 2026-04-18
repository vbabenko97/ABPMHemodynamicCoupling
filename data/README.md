# Data

Patient monitoring data is **not** version-controlled in this repository. The
directory is intentionally gitignored so that CSV, XLSX, and derived
participant-level files cannot be committed by accident.

To run the pipeline locally, obtain the datasets from the authors and place
them here:

- `data/monitoring_data.csv` — time-series ABPM data
- `data/aggregated_data.csv` — subject-level aggregates (optional)
- `data/aggregated_data_clf.csv` — responder classifier inputs (optional)

Expected columns and validation rules are documented in
[`src/abpm_hemodynamic_coupling/data_processing.py`](../src/abpm_hemodynamic_coupling/data_processing.py).
