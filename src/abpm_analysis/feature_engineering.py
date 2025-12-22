"""
Feature Engineering
====================

Feature extraction for DBP and PP prediction models.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from .config import Config, Columns


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    def __init__(self, config: Config):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    @abstractmethod
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features from DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> list:
        """
        Get feature names.
        
        Returns:
            List of feature names
        """
        pass


class DBPFeatureExtractor(FeatureExtractor):
    """
    DBP feature extractor.
    
    Generates 6D feature space for DBP prediction from SBP and HR:
    - f1: SBP
    - f2: HR
    - f3: 1/SBP
    - f4: 1/HR
    - f5: SBP*HR
    - f6: 1/(SBP*HR)
    """
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract DBP features.
        
        Args:
            df: DataFrame with SBP and HR columns
            
        Returns:
            Feature matrix (n_samples, 6)
        """
        sbp = df[Columns.SBP].values.astype(float)
        hr = df[Columns.HR].values.astype(float)
        epsilon = self.config.EPSILON
        
        f1 = sbp
        f2 = hr
        f3 = 1.0 / (sbp + epsilon)
        f4 = 1.0 / (hr + epsilon)
        prod = sbp * hr
        f5 = prod
        f6 = 1.0 / (prod + epsilon)
        
        return np.column_stack([f1, f2, f3, f4, f5, f6])
    
    def get_feature_names(self) -> list:
        """Get DBP feature names."""
        return ["SBP", "HR", "1/SBP", "1/HR", "SBP*HR", "1/(SBP*HR)"]


class PPFeatureExtractor(FeatureExtractor):
    """
    Pulse Pressure (PP) feature extractor.
    
    Generates 4D feature space for PP prediction from HR only:
    - f1: HR
    - f2: 1/HR
    - f3: HR^2
    - f4: log(HR)
    """
    
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract PP features.
        
        Args:
            df: DataFrame with HR column
            
        Returns:
            Feature matrix (n_samples, 4)
        """
        hr = df[Columns.HR].values.astype(float)
        epsilon = self.config.EPSILON
        
        f1 = hr
        f2 = 1.0 / (hr + epsilon)
        f3 = hr ** 2
        f4 = np.log(hr + epsilon)
        
        return np.column_stack([f1, f2, f3, f4])
    
    def get_feature_names(self) -> list:
        """Get PP feature names."""
        return ["HR", "1/HR", "HR^2", "log(HR)"]
