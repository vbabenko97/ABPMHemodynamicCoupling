# ABPM Hemodynamic Uncoupling Analysis

A modular Python pipeline for analyzing hemodynamic coupling patterns in ambulatory blood pressure monitoring (ABPM) data.

## Project Structure

```
ABPMHemodynamicCoupling/
├── src/abpm_analysis/       # Main analysis package
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
# Install dependencies
python3 -m pip install -r requirements.txt
```

## Usage

### Run the Complete Pipeline

```bash
python3 run_pipeline.py
```

### Input Files

Place the following files in the `data/` directory:
- `monitoring_data.csv` - Time-series hemodynamic data
- `aggregated_data.csv` - Subject-level aggregated data (optional, for correlations)
- `aggregated_data_clf.xlsx` - Subject-level data for classifier (optional)

### Output Files

The pipeline generates the following in `results/`:

**Tables:**
- `table1_final.csv` - Demographics by condition
- `per_subject_metrics.csv` - Subject-level metrics

**Statistical Summaries:**
- `results_summary.txt` - Main statistical results
- `cross_condition_analysis.txt` - Correlation analysis
- `pairwise_tests.txt` - Pairwise comparisons

**Figures (400 DPI):**
- `figure_2_dotplots.png` - MAE inflation and bias distributions
- `figure_3_obs_vs_pred.png` - Observed vs predicted for case studies
- `figure_4_timeseries_residuals.png` - Time-series with residuals

## Features

### Modeling
- **Brandon's Feature Selection**: Multiplicative ratio-based feature selection
- **Model Comparison**: Lasso, RFE, OLS baselines
- **Leakage-free CV**: Strict cross-validation with inner scaling
- **DBP Prediction**: 6D feature space (SBP, HR, and derived features)
- **PP Prediction**: 4D feature space (HR-only derived features)

### Statistical Analysis
- Bootstrap confidence intervals
- Wilcoxon signed-rank tests
- Mann-Whitney U tests
- Spearman correlations
- FDR correction (Benjamini-Hochberg)

### Responder Classification
- Logistic regression classifier
- Grid search with stratified CV
- Comprehensive performance metrics
- Feature importance analysis

## Development

### Code Structure

The codebase follows clean architecture principles:
- **Separation of Concerns**: Each module has a single responsibility
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Docstrings for all public functions
- **Error Handling**: Graceful degradation with informative messages

### Testing

To verify the refactored pipeline produces identical results to the original:

```bash
# Run refactored pipeline
python3 run_pipeline.py

# Compare outputs with original results
diff results/per_subject_metrics.csv original_results/per_subject_metrics.csv
```

## Author

**Vitalii Babenko**  
Refactored: December 2025

## License

[Specify license]
