"""
Data Models and Domain Objects
================================

Dataclasses and type definitions for structured data throughout the pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class ModelPerformance:
    """Cross-validation performance scores for different models."""
    
    brandon: float
    lasso: float
    rfe: float
    ols_sbp: float
    ols_sbp_hr: float
    
    def get_winner(self) -> str:
        """Determine the best-performing model."""
        scores = {
            "Brandon": self.brandon,
            "Lasso": self.lasso,
            "RFE": self.rfe,
            "OLS(SBP)": self.ols_sbp,
            "OLS(SBP,HR)": self.ols_sbp_hr,
        }
        return min(scores, key=scores.get)
    
    def get_best_score(self) -> float:
        """Get the score of the best-performing model."""
        return min([self.brandon, self.lasso, self.rfe, self.ols_sbp, self.ols_sbp_hr])


@dataclass
class ConditionMetrics:
    """Metrics for a specific condition (e.g., Cognitive Task)."""
    
    n: int
    mae: float
    delta_bias: float
    anomaly: float  # MAE inflation %
    

@dataclass
class SubjectResult:
    """Complete results for a single subject."""
    
    participant_id: int
    train_n: int
    
    # DBP modeling
    dbp_winner: str
    dbp_ref_mae: float
    dbp_cognitive_task: Optional[ConditionMetrics] = None
    dbp_physical_task: Optional[ConditionMetrics] = None
    dbp_air_alert: Optional[ConditionMetrics] = None
    
    # PP modeling
    pp_winner: Optional[str] = None
    pp_ref_mae: Optional[float] = None
    pp_cognitive_task: Optional[ConditionMetrics] = None
    pp_physical_task: Optional[ConditionMetrics] = None
    pp_air_alert: Optional[ConditionMetrics] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for DataFrame creation."""
        result = {
            "participant_id": self.participant_id,
            "Train_N": self.train_n,
            "DBP_Winner": self.dbp_winner,
            "DBP_Ref_MAE": self.dbp_ref_mae,
        }
        
        # Add DBP condition metrics
        for cond_name, cond_metrics in [
            ("Cognitive Task", self.dbp_cognitive_task),
            ("Physical Task", self.dbp_physical_task),
            ("Air Alert", self.dbp_air_alert),
        ]:
            if cond_metrics:
                result[f"DBP_{cond_name}_N"] = cond_metrics.n
                result[f"DBP_{cond_name}_MAE"] = cond_metrics.mae
                result[f"DBP_{cond_name}_DeltaBias"] = cond_metrics.delta_bias
                result[f"DBP_{cond_name}_Anomaly"] = cond_metrics.anomaly
            else:
                result[f"DBP_{cond_name}_N"] = 0
                result[f"DBP_{cond_name}_MAE"] = np.nan
                result[f"DBP_{cond_name}_DeltaBias"] = np.nan
                result[f"DBP_{cond_name}_Anomaly"] = np.nan
        
        # Add PP metrics if available
        if self.pp_winner:
            result["PP_Winner"] = self.pp_winner
            result["PP_Ref_MAE"] = self.pp_ref_mae
            
            for cond_name, cond_metrics in [
                ("Cognitive Task", self.pp_cognitive_task),
                ("Physical Task", self.pp_physical_task),
                ("Air Alert", self.pp_air_alert),
            ]:
                if cond_metrics:
                    result[f"PP_{cond_name}_N"] = cond_metrics.n
                    result[f"PP_{cond_name}_MAE"] = cond_metrics.mae
                    result[f"PP_{cond_name}_DeltaBias"] = cond_metrics.delta_bias
                    result[f"PP_{cond_name}_Anomaly"] = cond_metrics.anomaly
                else:
                    result[f"PP_{cond_name}_N"] = 0
                    result[f"PP_{cond_name}_MAE"] = np.nan
                    result[f"PP_{cond_name}_DeltaBias"] = np.nan
                    result[f"PP_{cond_name}_Anomaly"] = np.nan
        
        return result


@dataclass
class StatisticalResult:
    """Result of a statistical test."""
    
    name: str
    n: int
    median: float
    q25: float
    q75: float
    ci_low: float
    ci_high: float
    test_statistic: float
    p_value: float
    q_value: Optional[float] = None
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is significant at given alpha level."""
        return self.p_value < alpha
    
    def is_significant_fdr(self, alpha: float = 0.1) -> bool:
        """Check if result is significant after FDR correction."""
        return self.q_value is not None and self.q_value < alpha


@dataclass
class ClassifierMetrics:
    """Metrics for classifier performance."""
    
    accuracy: float
    precision: float
    recall: float
    f1: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class ResponderClassificationResult:
    """Complete results from responder classification."""
    
    n: int
    best_params: Dict[str, Any]
    train_metrics: ClassifierMetrics
    cv_metrics_mean: ClassifierMetrics
    cv_metrics_median: ClassifierMetrics
    top_positive: Dict[str, float] = field(default_factory=dict)
    top_negative: Dict[str, float] = field(default_factory=dict)
