"""
Data Processing and Loading
=============================

Handles data loading, validation, preprocessing, and hierarchical labeling.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .config import Columns, Config


class Labeler:
    """Applies hierarchical context labeling to hemodynamic data."""
    
    @staticmethod
    def apply_hierarchy(row: pd.Series) -> str:
        """
        Apply mutually exclusive state assignment with strict priority.
        
        Priority order:
        1. Air Alert
        2. Before Testing
        3. Cognitive Pre/Task
        4. Physical Pre/Task
        5. Sleep
        6. Manual Trigger
        7. Break
        8. Baseline (default)
        
        Args:
            row: DataFrame row with context indicator columns
            
        Returns:
            Label string for the context
        """
        if row.get(Columns.ALERT, 0) == 1:
            return Columns.LABEL_AIR_ALERT
        
        if row.get(Columns.BEFORE, 0) == 1:
            return Columns.LABEL_BEFORE_TESTING
        
        # Cognitive tasks
        if row.get(Columns.IS_COG, 0) == 1:
            if row.get(Columns.PRE, 0) == 1:
                return Columns.LABEL_COGNITIVE_PRE
            if row.get(Columns.DURING, 0) == 1 or row.get(Columns.POST, 0) == 1:
                return Columns.LABEL_COGNITIVE_TASK
            return Columns.LABEL_COGNITIVE_TASK  # Fallback
        
        # Physical tasks
        if row.get(Columns.IS_PHYS, 0) == 1:
            if row.get(Columns.PRE, 0) == 1:
                return Columns.LABEL_PHYSICAL_PRE
            if row.get(Columns.DURING, 0) == 1 or row.get(Columns.POST, 0) == 1:
                return Columns.LABEL_PHYSICAL_TASK
            return Columns.LABEL_PHYSICAL_TASK  # Fallback
        
        # Sleep
        if row.get(Columns.STATE, 0) == 1:
            return Columns.LABEL_SLEEP
        
        # Manual trigger
        if row.get(Columns.TRIGGER, 0) == 1:
            return Columns.LABEL_MANUAL_TRIGGER
        
        # Break
        if row.get(Columns.BREAK, 0) == 1:
            return Columns.LABEL_BREAK
        
        # Default
        return Columns.LABEL_BASELINE


class DataValidator:
    """Validates data quality and completeness."""

    @staticmethod
    def _validate_physiological_range(
        df: pd.DataFrame,
        column: str,
        lower: float,
        upper: float,
    ) -> None:
        """Validate that non-missing values fall within a conservative range."""
        valid_values = df[column].dropna()
        if not np.isfinite(valid_values).all():
            raise ValueError(f"Column {column} contains non-finite values")

        out_of_range = valid_values[(valid_values < lower) | (valid_values > upper)]
        if not out_of_range.empty:
            raise ValueError(
                f"Column {column} contains values outside [{lower}, {upper}] "
                f"(n={len(out_of_range)})"
            )
    
    @staticmethod
    def validate_monitoring_data(df: pd.DataFrame) -> None:
        """
        Validate monitoring data has required columns and valid values.
        
        Args:
            df: Monitoring data DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        required_cols = [Columns.PAT_ID, Columns.TIME, Columns.SBP, Columns.DBP, Columns.HR]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for any data
        if df.empty:
            raise ValueError("Data frame is empty")
        
        # Check numeric columns are numeric
        for col in [Columns.SBP, Columns.DBP, Columns.HR]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must be numeric")

        DataValidator._validate_physiological_range(
            df,
            Columns.SBP,
            Config.MIN_SBP,
            Config.MAX_SBP,
        )
        DataValidator._validate_physiological_range(
            df,
            Columns.DBP,
            Config.MIN_DBP,
            Config.MAX_DBP,
        )
        DataValidator._validate_physiological_range(
            df,
            Columns.HR,
            Config.MIN_HR,
            Config.MAX_HR,
        )
    
    @staticmethod
    def check_subject_data_quality(
        df_subject: pd.DataFrame,
        min_samples: int,
        subject_id: int
    ) -> bool:
        """
        Check if subject has sufficient quality data.
        
        Args:
            df_subject: Subject's data
            min_samples: Minimum required samples
            subject_id: Subject ID for logging
            
        Returns:
            True if data quality is sufficient
        """
        if len(df_subject) < min_samples:
            return False
        
        # Check for excessive missing values
        missing_pct = df_subject[[Columns.SBP, Columns.DBP, Columns.HR]].isnull().mean().mean()
        if missing_pct > 0.5:
            return False
        
        return True

    @staticmethod
    def sanitize_monitoring_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Drop rows with non-finite or out-of-range hemodynamic values."""
        sanitized = df.copy()
        initial_len = len(sanitized)

        bounds = [
            (Columns.SBP, Config.MIN_SBP, Config.MAX_SBP),
            (Columns.DBP, Config.MIN_DBP, Config.MAX_DBP),
            (Columns.HR, Config.MIN_HR, Config.MAX_HR),
        ]

        for column, lower, upper in bounds:
            sanitized = sanitized[np.isfinite(sanitized[column])]
            sanitized = sanitized[(sanitized[column] >= lower) & (sanitized[column] <= upper)]

        return sanitized, initial_len - len(sanitized)


class DataLoader:
    """Handles loading data from various sources."""
    
    def __init__(self, config: Config):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def load_monitoring_data(self) -> pd.DataFrame:
        """
        Load and preprocess monitoring data.
        
        Returns:
            Preprocessed monitoring DataFrame
        """
        filepath = self.config.get_data_path(self.config.MONITORING_FILE)
        
        print(f"Loading monitoring data from {filepath}...")
        df = pd.read_csv(filepath)

        required_cols = [Columns.PAT_ID, Columns.TIME, Columns.SBP, Columns.DBP, Columns.HR]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        for col in [Columns.SBP, Columns.DBP, Columns.HR]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must be numeric")

        df, removed_n = DataValidator.sanitize_monitoring_data(df)
        if removed_n:
            print(f"Dropped {removed_n} out-of-range monitoring rows before analysis.")
        
        # Validate
        DataValidator.validate_monitoring_data(df)
        
        # Convert datetime
        df[Columns.TIME] = pd.to_datetime(df[Columns.TIME])
        
        # Sort by participant and time
        df = df.sort_values([Columns.PAT_ID, Columns.TIME])
        
        # Drop rows with missing hemodynamic values
        df = df.dropna(subset=[Columns.SBP, Columns.DBP, Columns.HR])
        
        # Apply hierarchical labeling
        print("Applying hierarchical labeling...")
        df[Columns.LABEL] = df.apply(Labeler.apply_hierarchy, axis=1)
        
        print(f"Loaded {len(df)} records for {df[Columns.PAT_ID].nunique()} subjects")
        
        return df
    
    def load_aggregated_data(self) -> Optional[pd.DataFrame]:
        """
        Load aggregated subject-level data for correlations.
        
        Returns:
            Aggregated DataFrame or None if file not found
        """
        filepath = self.config.get_data_path(self.config.AGGREGATED_FILE)
        
        try:
            print(f"Loading aggregated data from {filepath}...")
            df = pd.read_csv(filepath)
            print(f"Loaded aggregated data: {len(df)} subjects")
            return df
        except FileNotFoundError:
            print(f"Aggregated data file not found: {filepath}")
            return None
    
    def load_aggregated_classifier_data(self) -> Optional[pd.DataFrame]:
        """
        Load aggregated data for classifier training.
        
        Returns:
            Aggregated DataFrame or None if file not found
        """
        filepath = self.config.get_data_path(self.config.AGGREGATED_CLF_FILE)
        
        try:
            print(f"Loading classifier data from {filepath}...")
            df = pd.read_csv(filepath)
            print(f"Loaded classifier data: {len(df)} subjects")
            return df
        except FileNotFoundError:
            print(f"Classifier data file not found: {filepath}")
            return None
