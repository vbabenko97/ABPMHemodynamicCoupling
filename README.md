# ABPM Hemodynamic Uncoupling Analysis

A modular Python pipeline for analyzing hemodynamic coupling patterns in ambulatory blood pressure monitoring (ABPM) data.

**This repository accompanies a project on hemodynamic uncoupling during cognitive stress.**

## Project Structure

```
ABPMHemodynamicCoupling/
├── src/                     # Main analysis package
│   ├── __init__.py
│   ├── config.py            # Configuration and constants
│   ├── models.py            # Data models
│   ├── data_processing.py   # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature extraction
│   ├── modeling.py          # Model training and evaluation
│   ├── statistics.py        # Statistical analysis
│   ├── utils.py             # Utility functions
│   └── visualization.py     # Figure generation
├── data/                    # Data files (CSV/Excel)
├── results/                 # Output directory
├── run_pipeline.py          # Main orchestration script
└── requirements.txt         # Python dependencies
```

## Installation

### Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

### Setup

```bash
# Clone the repository
git clone https://github.com/vbabenko97/ABPMHemodynamicCoupling.git
cd ABPMHemodynamicCoupling

# Install dependencies
python3 -m pip install -r requirements.txt
```

## Usage

### Running the Pipeline

```bash
python run_pipeline.py
```

This will:
1. Load and preprocess monitoring data from `data/monitoring_data.csv`
2. Run subject-level hemodynamic uncoupling analysis
3. Compute cohort-level statistics with FDR correction
4. Generate publication-quality figures

### Input Data Format

Place the following files in the **root directory** or `data/`:

| File | Description |
|------|-------------|
| `monitoring_data.csv` | Time-series hemodynamic readings (SBP, DBP, HR, timestamps, context labels) |
| `aggregated_data.csv` | Subject-level aggregated data (optional, for correlation analysis) |

### Output Files

The pipeline generates the following in `results/`:

| File | Description |
|------|-------------|
| `per_subject_metrics.csv` | Subject-level uncoupling metrics |
| `results_summary.txt` | Statistical summary with FDR-corrected p-values |
| `dotplots.png` | **Figure 1**: MAE inflation and bias distributions |
| `obs_vs_pred.png` | **Figure 2**: Observed vs predicted DBP for case studies |
| `timeseries_residuals.png` | **Figure 3**: Time-series with residual analysis |
| `demographics.png` | Demographics table visualization |

## Features

### Modeling
- **Brandon's Feature Selection**: Multiplicative ratio-based feature selection
- **Model Comparison**: Lasso, RFE, OLS baselines
- **Leakage-free CV**: Strict cross-validation with inner scaling
- **DBP Prediction**: 6D feature space (SBP, HR, and derived features)

### Statistical Analysis
- Bootstrap confidence intervals
- Wilcoxon signed-rank tests
- Mann-Whitney U tests
- Spearman correlations
- FDR correction (Benjamini-Hochberg, α=0.1)

### Visualization
- High-resolution figures (400 DPI)
- Screen-Positive vs Screen-Negative marker distinction
- Thesis-ready styling with clear labels and legends

## Author

**Vitalii Babenko**, 2025

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
