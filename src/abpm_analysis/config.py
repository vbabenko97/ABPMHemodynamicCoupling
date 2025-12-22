"""
Configuration and Constants
============================

Centralized configuration for all pipeline parameters, file paths, and constants.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass
class Config:
    """Main configuration class for the ABPM analysis pipeline."""
    
    # Random seed for reproducibility
    RANDOM_SEED: int = 42
    
    # Numerical stability
    EPSILON: float = 1e-9
    
    # Cross-validation settings
    LASSO_CV_FOLDS: int = 3
    N_CV_SPLITS: int = 3
    
    # Statistical settings
    FDR_ALPHA: float = 0.1
    BOOTSTRAP_ITERATIONS: int = 10000
    
    # Responder thresholds
    RESPONDER_ANOMALY_THRESHOLD: float = 50.0  # %
    RESPONDER_BIAS_THRESHOLD: float = 2.0  # mmHg
    
    # Minimum data requirements
    MIN_BASELINE_SAMPLES: int = 15
    MIN_CV_SAMPLES_MULTIPLIER: int = 2  # min samples = n_splits * multiplier
    
    # Directory paths (relative to project root)
    DATA_DIR: ClassVar[Path] = Path("data")
    RESULTS_DIR: ClassVar[Path] = Path("results")
    
    # Input file names
    MONITORING_FILE: str = "monitoring_data.csv"
    AGGREGATED_FILE: str = "aggregated_data.csv"
    AGGREGATED_CLF_FILE: str = "aggregated_data_clf.csv"

    # Output file names
    SUBJECT_METRICS_OUTPUT: str = "per_subject_metrics.csv"
    SUMMARY_OUTPUT: str = "results_summary.txt"
    CROSS_CONDITION_OUTPUT: str = "cross_condition_analysis.txt"
    PAIRWISE_OUTPUT: str = "pairwise_tests.txt"
    DEMOGRAPHICS_FIGURE: str = "demographics.png"

    # Figure file names
    FIGURE_2_OUTPUT: str = "dotplots.png"
    FIGURE_3_OUTPUT: str = "obs_vs_pred.png"
    FIGURE_4_OUTPUT: str = "timeseries_residuals.png"
    
    # Figure settings
    FIGURE_DPI: int = 400
    
    def __post_init__(self):
        """Create results directory if it doesn't exist."""
        self.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    def get_data_path(self, filename: str) -> Path:
        """Get full path to a data file."""
        return self.DATA_DIR / filename
    
    def get_results_path(self, filename: str) -> Path:
        """Get full path to a results file."""
        return self.RESULTS_DIR / filename


class Columns:
    """Column name constants for data frames."""
    
    # Participant and time
    PAT_ID = "participant_id"
    TIME = "datetime"
    
    # Hemodynamic measurements
    SBP = "SBP"
    DBP = "DBP"
    HR = "HR"
    
    # Context indicators
    STATE = "state"
    ALERT = "alert_window"
    BEFORE = "before_testing_period"
    BREAK = "break_period"
    TRIGGER = "trigger_type"
    
    # Task type indicators
    IS_COG = "is_cog"
    IS_PHYS = "is_phys"
    
    # Task phase indicators
    PRE = "pre_task"
    DURING = "during_task"
    POST = "post_task"
    
    # Derived columns
    LABEL = "label"
    
    # Task labels
    LABEL_AIR_ALERT = "Air Alert"
    LABEL_BEFORE_TESTING = "Before Testing"
    LABEL_COGNITIVE_PRE = "Cognitive Pre"
    LABEL_COGNITIVE_TASK = "Cognitive Task"
    LABEL_PHYSICAL_PRE = "Physical Pre"
    LABEL_PHYSICAL_TASK = "Physical Task"
    LABEL_SLEEP = "Sleep"
    LABEL_MANUAL_TRIGGER = "Manual Trigger"
    LABEL_BREAK = "Break"
    LABEL_BASELINE = "Baseline"
