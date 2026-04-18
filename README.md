<p align="center">
  <img src="assets/logo/abpm_analysis.png" alt="ABPM Analysis Logo" width="200"/>
</p>

# ABPM Hemodynamic Coupling

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/vbabenko97/ABPMHemodynamicCoupling/actions/workflows/ci.yml/badge.svg)](https://github.com/vbabenko97/ABPMHemodynamicCoupling/actions/workflows/ci.yml)

ABPM Hemodynamic Coupling is a Python research pipeline for analyzing stress-linked blood pressure relationship shifts in ambulatory blood pressure monitoring (ABPM) data. The repository supports the IEEE ELNANO 2026 project context and reproduces the subject-level modeling, cohort-level summaries, publication figures, and Streamlit review workflow used in the study.

## Authors

- Vitalii Babenko
- Alyona Tymchak

## Highlights

- Subject-level DBP modeling with Brandon selection, Lasso, RFE, and OLS baselines
- Leakage-aware cross-validation with fixed random seeds for reproducible runs
- Cohort-level bootstrap confidence intervals, Wilcoxon testing, and FDR correction
- Publication-ready figure generation and conference presentation support
- Interactive Streamlit MVP for uploading ABPM CSV data and reviewing results in-browser
- Ukrainian UI labels for summary tables, figures, and subject-level metrics

## Repository Layout

```text
ABPMHemodynamicCoupling/
├── src/                         # Reusable analysis package
├── analysis/                    # Study-specific analysis code and notebooks
│   ├── thesis/                  # Thesis-track stats helpers and EDA notebooks
│   └── hemodynamic_reactivity/  # Hemodynamic-reactivity sub-analysis
├── app.py                       # Streamlit app entry point
├── pages/                       # Streamlit multipage views
├── run_pipeline.py              # Batch analysis entry point
├── scripts/                     # Repeatable helper scripts
├── tests/                       # Regression and smoke tests
├── data/                        # Local input datasets (gitignored)
├── results/                     # Committed research outputs
├── artifacts/                   # Larger generated outputs (gitignored)
├── docs/                        # Papers, drafts, and conference material
│   ├── ieee_elnano/             # IEEE ELNANO 2026 paper and presentation
│   └── thesis/                  # Ukrainian thesis drafts, figures, tables
├── assets/                      # Logo and static assets
├── .github/                     # CI, issue templates, ownership metadata
└── pyproject.toml               # Package and tool configuration
```

See [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md) for the operating conventions behind this split.

## Installation

### Editable install

```bash
git clone https://github.com/vbabenko97/ABPMHemodynamicCoupling.git
cd ABPMHemodynamicCoupling
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

### Requirements

- Python 3.10 or newer
- input CSV or XLSX files placed under `data/`

## Usage

### Run the analysis pipeline

```bash
python run_pipeline.py
```

### Run the Streamlit web app

```bash
streamlit run app.py
```

The web app provides an interactive MVP workflow for:

- uploading `monitoring_data.csv`
- sanitizing and validating incoming rows before analysis
- viewing cohort summaries in tabular form
- reviewing localized subject metrics and figures inside the browser

The pipeline expects:

| File | Required | Description |
|------|----------|-------------|
| `data/monitoring_data.csv` | Yes | Time-series ABPM data with `participant_id`, `datetime`, `SBP`, `DBP`, `HR`, and context columns |
| `data/aggregated_data.csv` | No | Subject-level aggregates for follow-on association analyses |
| `data/aggregated_data_clf.csv` | No | Subject-level classifier inputs for responder modeling |

### Outputs

Core outputs are written to `results/`:

- `per_subject_metrics.csv`
- `results_summary.txt`
- `cross_condition_analysis.txt`
- `pairwise_tests.txt`
- `demographics.png`
- `dotplots.png`
- `obs_vs_pred.png`
- `timeseries_residuals.png`

In the Streamlit app, the same analysis is exposed through three tabs:

- `Підсумок` with demographics, model counts, baseline MAE, condition comparisons, and subgroup analysis
- `Метрики учасників` with localized per-subject results and CSV export
- `Графіки` with the generated analysis figures

## Development

Install the development extras and run the standard checks:

```bash
python -m pip install -e .[dev]
python -m pytest
ruff check .
```

Use `uv.lock` as the canonical lockfile for reproducible local environments. `requirements.txt` and `requirements-thesis.txt` remain lightweight compatibility lists.

## Reproducibility

- Core configuration lives in `src/config.py`.
- The repository fixes `RANDOM_SEED = 42`, `FDR_ALPHA = 0.1`, and `BOOTSTRAP_ITERATIONS = 10000`.
- Input validation checks required columns, numeric dtypes, and conservative SBP/DBP/HR ranges before analysis starts.
- CI runs linting and tests on every push and pull request.

## Citation

If you use this repository in academic work, cite the project metadata in `CITATION.cff`.

```bibtex
@software{babenko_tymchak_abpm_hemodynamic_coupling,
  title = {ABPM Hemodynamic Coupling},
  author = {Babenko, Vitalii and Tymchak, Alyona},
  year = {2026},
  note = {IEEE ELNANO 2026 project repository},
  url = {https://github.com/vbabenko97/ABPMHemodynamicCoupling}
}
```

## Contributing

Bug reports and focused pull requests are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) for the local workflow and development expectations.

## Security

For security-sensitive reports or accidental data exposure, use the private reporting path described in [SECURITY.md](SECURITY.md).

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) for the full text.
