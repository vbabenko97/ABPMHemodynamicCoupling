"""
Modeling and Model Selection
==============================

Model training, cross-validation, and responder classification.
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV, LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, precision_score, 
    recall_score, f1_score
)
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV

from .config import Config
from .models import ModelPerformance, ClassifierMetrics, ResponderClassificationResult


class BrandonSelector:
    """
    Brandon's multiplicative ratio-based feature selection.
    
    Iteratively selects features based on correlation with residual ratios.
    """
    
    def __init__(self, config: Config, max_features: int = 3):
        """
        Initialize Brandon selector.
        
        Args:
            config: Configuration object
            max_features: Maximum number of features to select
        """
        self.config = config
        self.max_features = max_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, List[int]]:
        """
        Fit Brandon's feature selection and train model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            Tuple of (fitted model, list of selected feature indices)
        """
        mean_y = np.mean(y)
        if abs(mean_y) < self.config.EPSILON:
            mean_y = 1.0
        
        r_curr = y / mean_y
        selected = []
        n_features = X.shape[1]
        
        for _ in range(self.max_features):
            best_idx = -1
            best_corr = -1.0
            
            # Find best feature
            for j in range(n_features):
                if j in selected:
                    continue
                if np.std(X[:, j]) < self.config.EPSILON:
                    continue
                
                corr = abs(np.corrcoef(X[:, j], r_curr)[0, 1])
                if np.isnan(corr):
                    corr = 0
                
                if corr > best_corr:
                    best_corr = corr
                    best_idx = j
            
            # Stop if no good feature found
            if best_idx == -1 or best_corr < 0.05:
                break
            
            selected.append(best_idx)
            
            # Update residuals
            A = np.column_stack([X[:, selected], np.ones(len(r_curr))])
            try:
                coeffs = np.linalg.lstsq(A, r_curr, rcond=None)[0]
                pred = A @ coeffs
            except:
                pred = np.mean(r_curr) * np.ones_like(r_curr)
            
            # Avoid division by zero
            denom = np.where(
                np.abs(pred) < self.config.EPSILON,
                np.sign(pred) * self.config.EPSILON + (pred == 0) * self.config.EPSILON,
                pred
            )
            r_curr = r_curr / denom
        
        # Fallback to at least 2 features
        if not selected:
            selected = [0, 1]
        
        # Train final model
        model = LinearRegression()
        model.fit(X[:, selected], y)
        
        return model, selected


class CrossValidator:
    """Performs leakage-free cross-validation for model selection."""
    
    def __init__(self, config: Config):
        """
        Initialize cross-validator.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def evaluate_models(
        self, 
        X_raw: np.ndarray, 
        y: np.ndarray,
        n_splits: Optional[int] = None
    ) -> Optional[ModelPerformance]:
        """
        Evaluate all candidate models using cross-validation.
        
        Args:
            X_raw: Raw feature matrix (unscaled)
            y: Target values
            n_splits: Number of CV folds (default from config)
            
        Returns:
            ModelPerformance object or None if insufficient data
        """
        if n_splits is None:
            n_splits = self.config.N_CV_SPLITS
        
        # Check minimum data requirement
        if len(y) < n_splits * self.config.MIN_CV_SAMPLES_MULTIPLIER:
            return None
        
        kf = KFold(n_splits=n_splits, shuffle=False)
        scores = {
            "Brandon": [],
            "Lasso": [],
            "RFE": [],
            "OLS(SBP)": [],
            "OLS(SBP,HR)": []
        }
        
        for train_ix, val_ix in kf.split(X_raw):
            X_train_raw, X_val_raw = X_raw[train_ix], X_raw[val_ix]
            y_train, y_val = y[train_ix], y[val_ix]
            
            # Inner scaling (no leakage)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)
            
            # Brandon
            try:
                selector = BrandonSelector(self.config)
                model, idx = selector.fit(X_train, y_train)
                pred = model.predict(X_val[:, idx])
                scores["Brandon"].append(mean_absolute_error(y_val, pred))
            except:
                scores["Brandon"].append(np.nan)
            
            # Lasso
            try:
                model = LassoCV(
                    cv=self.config.LASSO_CV_FOLDS,
                    random_state=self.config.RANDOM_SEED
                ).fit(X_train, y_train)
                pred = model.predict(X_val)
                scores["Lasso"].append(mean_absolute_error(y_val, pred))
            except:
                scores["Lasso"].append(np.nan)
            
            # RFE
            try:
                n_feat = len(idx) if 'idx' in locals() else 3
                rfe = RFE(
                    LinearRegression(),
                    n_features_to_select=n_feat
                ).fit(X_train, y_train)
                pred = rfe.predict(X_val)
                scores["RFE"].append(mean_absolute_error(y_val, pred))
            except:
                scores["RFE"].append(np.nan)
            
            # OLS Baseline 1: DBP = f(SBP)
            try:
                model = LinearRegression().fit(X_train[:, [0]], y_train)
                pred = model.predict(X_val[:, [0]])
                scores["OLS(SBP)"].append(mean_absolute_error(y_val, pred))
            except:
                scores["OLS(SBP)"].append(np.nan)
            
            # OLS Baseline 2: DBP = f(SBP, HR)
            try:
                model = LinearRegression().fit(X_train[:, [0, 1]], y_train)
                pred = model.predict(X_val[:, [0, 1]])
                scores["OLS(SBP,HR)"].append(mean_absolute_error(y_val, pred))
            except:
                scores["OLS(SBP,HR)"].append(np.nan)
        
        # Compute average scores
        avg_scores = {}
        for name, vals in scores.items():
            valid_vals = [v for v in vals if not np.isnan(v)]
            avg_scores[name] = np.mean(valid_vals) if valid_vals else np.inf
        
        return ModelPerformance(
            brandon=avg_scores["Brandon"],
            lasso=avg_scores["Lasso"],
            rfe=avg_scores["RFE"],
            ols_sbp=avg_scores["OLS(SBP)"],
            ols_sbp_hr=avg_scores["OLS(SBP,HR)"]
        )


class ModelTrainer:
    """Trains final models on full training data."""
    
    def __init__(self, config: Config):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def train(
        self,
        X_raw: np.ndarray,
        y: np.ndarray,
        winner: str
    ) -> Tuple[Any, List[int], StandardScaler]:
        """
        Train final model using winning algorithm.
        
        Args:
            X_raw: Raw feature matrix (unscaled)
            y: Target values
            winner: Name of winning model
            
        Returns:
            Tuple of (fitted model, selected feature indices, fitted scaler)
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        
        if winner == "Brandon":
            selector = BrandonSelector(self.config)
            model, idx = selector.fit(X_scaled, y)
        elif winner == "Lasso":
            model = LassoCV(
                cv=self.config.LASSO_CV_FOLDS,
                random_state=self.config.RANDOM_SEED
            ).fit(X_scaled, y)
            idx = list(range(X_scaled.shape[1]))
        elif winner == "OLS(SBP)":
            model = LinearRegression().fit(X_scaled[:, [0]], y)
            idx = [0]
        elif winner == "OLS(SBP,HR)":
            model = LinearRegression().fit(X_scaled[:, [0, 1]], y)
            idx = [0, 1]
        else:  # RFE
            selector = BrandonSelector(self.config)
            _, idx_temp = selector.fit(X_scaled, y)
            rfe = RFE(
                LinearRegression(),
                n_features_to_select=len(idx_temp)
            ).fit(X_scaled, y)
            model = rfe
            idx = list(range(X_scaled.shape[1]))
        
        return model, idx, scaler


class ResponderClassifier:
    """Trains logistic regression classifier to predict responder status."""
    
    def __init__(self, config: Config):
        """
        Initialize responder classifier.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def train(
        self,
        merged_df: pd.DataFrame,
        numeric_cols: List[str],
        binary_cols: List[str],
        target_col: str = "Is_Responder"
    ) -> Optional[ResponderClassificationResult]:
        """
        Train classifier to predict responder status.
        
        Args:
            merged_df: DataFrame with features and target
            numeric_cols: List of numeric feature column names
            binary_cols: List of binary feature column names
            target_col: Name of target column
            
        Returns:
            ResponderClassificationResult or None if insufficient data
        """
        if target_col not in merged_df.columns:
            return None
        
        features = numeric_cols + binary_cols
        print(f"Classifier candidate features: {len(features)}")
        
        sub = merged_df[features + [target_col]].dropna()
        if len(sub) < 10:
            return None
        
        X = sub[features]
        y = sub[target_col]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cross-validation setup
        skf = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=self.config.RANDOM_SEED
        )
        
        # Grid search for best regularization
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear']
        }
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
        
        grid = GridSearchCV(
            LogisticRegression(random_state=self.config.RANDOM_SEED),
            param_grid,
            cv=skf,
            scoring=scoring,
            refit='f1',
            return_train_score=True
        )
        
        try:
            grid.fit(X_scaled, y)
        except ValueError:
            # Classes too rare for stratified CV
            return None
        
        best_model = grid.best_estimator_
        best_idx = grid.best_index_
        
        # Metrics on full training set
        y_pred = best_model.predict(X_scaled)
        train_metrics = ClassifierMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, zero_division=0),
            recall=recall_score(y, y_pred, zero_division=0),
            f1=f1_score(y, y_pred, zero_division=0)
        )
        
        # CV metrics (mean)
        cv_metrics_mean = ClassifierMetrics(
            accuracy=grid.cv_results_['mean_test_accuracy'][best_idx],
            precision=grid.cv_results_['mean_test_precision'][best_idx],
            recall=grid.cv_results_['mean_test_recall'][best_idx],
            f1=grid.cv_results_['mean_test_f1'][best_idx]
        )
        
        # CV metrics (median)
        cv_metrics_median = ClassifierMetrics(
            accuracy=np.median([grid.cv_results_[f'split{i}_test_accuracy'][best_idx] for i in range(3)]),
            precision=np.median([grid.cv_results_[f'split{i}_test_precision'][best_idx] for i in range(3)]),
            recall=np.median([grid.cv_results_[f'split{i}_test_recall'][best_idx] for i in range(3)]),
            f1=np.median([grid.cv_results_[f'split{i}_test_f1'][best_idx] for i in range(3)])
        )
        
        # Extract top coefficients
        coefs = pd.Series(best_model.coef_[0], index=features)
        pos_coefs = coefs[coefs > 0].sort_values(ascending=False).head(5)
        neg_coefs = coefs[coefs < 0].sort_values(ascending=True).head(5)
        
        return ResponderClassificationResult(
            n=len(sub),
            best_params=grid.best_params_,
            train_metrics=train_metrics,
            cv_metrics_mean=cv_metrics_mean,
            cv_metrics_median=cv_metrics_median,
            top_positive=pos_coefs.to_dict(),
            top_negative=neg_coefs.to_dict()
        )
