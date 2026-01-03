"""
Statistical Analysis
=====================

Bootstrap confidence intervals, distribution stats, correlation analysis,
and multiple testing correction.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr, chi2_contingency
from statsmodels.stats.multitest import fdrcorrection

from config import Config
from models import StatisticalResult


class BootstrapAnalyzer:
    """Performs bootstrap analysis for confidence intervals."""
    
    def __init__(self, config: Config):
        """
        Initialize bootstrap analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def median_ci(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for median.
        
        Args:
            data: Data array
            
        Returns:
            Tuple of (lower CI, upper CI)
        """
        if len(data) < 2:
            return np.nan, np.nan
        
        rng = np.random.default_rng(self.config.RANDOM_SEED)
        medians = [
            np.median(rng.choice(data, size=len(data), replace=True))
            for _ in range(self.config.BOOTSTRAP_ITERATIONS)
        ]
        
        return np.percentile(medians, 2.5), np.percentile(medians, 97.5)


class DistributionAnalyzer:
    """Analyzes distributions and computes summary statistics."""
    
    def __init__(self, config: Config):
        """
        Initialize distribution analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.bootstrap = BootstrapAnalyzer(config)
    
    def compute_stats(self, data: np.ndarray, name: str) -> Optional[StatisticalResult]:
        """
        Compute comprehensive distribution statistics.
        
        Args:
            data: Data array
            name: Name of the data/metric
            
        Returns:
            StatisticalResult object or None if insufficient data
        """
        if len(data) == 0:
            return None
        
        # Bootstrap CI
        ci_low, ci_high = self.bootstrap.median_ci(data)
        
        # Wilcoxon signed-rank test (against zero)
        try:
            w_stat, w_p = wilcoxon(data)
        except:
            w_stat, w_p = np.nan, 1.0
        
        return StatisticalResult(
            name=name,
            n=len(data),
            median=np.median(data),
            q25=np.percentile(data, 25),
            q75=np.percentile(data, 75),
            ci_low=ci_low,
            ci_high=ci_high,
            test_statistic=w_stat,
            p_value=w_p
        )


class CorrelationAnalyzer:
    """Performs correlation and association analysis."""
    
    def __init__(self, config: Config):
        """
        Initialize correlation analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def spearman_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute Spearman correlation.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        if len(x) < 3:
            return np.nan, 1.0
        
        rho, p = spearmanr(x, y)
        return rho, p
    
    def mannwhitney_test(
        self,
        group0: np.ndarray,
        group1: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Mann-Whitney U test.
        
        Args:
            group0: First group
            group1: Second group
            
        Returns:
            Tuple of (U statistic, p-value)
        """
        if len(group0) < 2 or len(group1) < 2:
            return np.nan, 1.0
        
        u, p = mannwhitneyu(group0, group1)
        return u, p
    
    def chi2_test(
        self,
        contingency_table: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Perform Chi-square test of independence.
        
        Args:
            contingency_table: Contingency table (2x2)
            
        Returns:
            Tuple of (chi2 statistic, p-value)
        """
        if contingency_table.shape != (2, 2):
            return np.nan, 1.0
        
        chi2, p, _, _ = chi2_contingency(contingency_table)
        return chi2, p


class MultipleTestingCorrector:
    """Handles multiple testing correction."""
    
    def __init__(self, config: Config):
        """
        Initialize multiple testing corrector.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def apply_fdr(self, p_values: List[float]) -> List[float]:
        """
        Apply FDR (Benjamini-Hochberg) correction.
        
        Args:
            p_values: List of p-values
            
        Returns:
            List of q-values (FDR-corrected p-values)
        """
        if not p_values:
            return []
        
        _, q_values = fdrcorrection(p_values, alpha=self.config.FDR_ALPHA)
        return q_values.tolist()
    
    def create_lookup(
        self,
        keys: List[tuple],
        p_values: List[float]
    ) -> Dict[tuple, float]:
        """
        Create lookup dictionary for q-values.
        
        Args:
            keys: List of keys (e.g., (condition, metric) tuples)
            p_values: Corresponding p-values
            
        Returns:
            Dictionary mapping keys to q-values
        """
        q_values = self.apply_fdr(p_values)
        return {key: q for key, q in zip(keys, q_values)}
