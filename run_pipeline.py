#!/usr/bin/env python3
"""
ABPM Hemodynamic Uncoupling Analysis Pipeline
==============================================

Main orchestration script for the refactored analysis pipeline.

Author: Vitalii Babenko
Refactored Date: 2025-12-22

Usage:
    python run_pipeline.py
"""

import sys
import warnings
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency

from abpm_analysis.config import Config, Columns
from abpm_analysis.models import SubjectResult, ConditionMetrics
from abpm_analysis.data_processing import DataLoader
from abpm_analysis.feature_engineering import DBPFeatureExtractor, PPFeatureExtractor
from abpm_analysis.modeling import CrossValidator, ModelTrainer, ResponderClassifier
from abpm_analysis.statistics import (
    DistributionAnalyzer, CorrelationAnalyzer, MultipleTestingCorrector
)
from abpm_analysis.visualization import VisualizationManager
from abpm_analysis.utils import setup_logging, ProgressTracker

# Suppress warnings
warnings.filterwarnings('ignore')


class SubjectAnalyzer:
    """Analyzes individual subjects."""
    
    def __init__(self, config: Config):
        self.config = config
        self.cv_validator = CrossValidator(config)
        self.trainer = ModelTrainer(config)
        self.dbp_extractor = DBPFeatureExtractor(config)
        self.pp_extractor = PPFeatureExtractor(config)
    
    def analyze_subject(
        self,
        subject_id: int,
        df_subject: pd.DataFrame
    ) -> SubjectResult:
        """
        Analyze a single subject.
        
        Args:
            subject_id: Subject ID
            df_subject: Subject's data
            
        Returns:
            SubjectResult object
        """
        # Get baseline data
        df_base = df_subject[df_subject[Columns.LABEL] == Columns.LABEL_BASELINE]
        
        if len(df_base) < self.config.MIN_BASELINE_SAMPLES:
            # Insufficient baseline data
            return SubjectResult(
                participant_id=subject_id,
                train_n=len(df_base),
                dbp_winner="NA",
                dbp_ref_mae=np.nan
            )
        
        # Analyze DBP
        dbp_result = self._analyze_target(
            df_subject, df_base,
            Columns.DBP, self.dbp_extractor
        )
        
        # Analyze PP
        pp_result = self._analyze_target(
            df_subject, df_base,
            Columns.PP, self.pp_extractor
        )
        
        # Combine results
        result = SubjectResult(
            participant_id=subject_id,
            train_n=len(df_base),
            dbp_winner=dbp_result['winner'],
            dbp_ref_mae=dbp_result['ref_mae'],
            dbp_cognitive_task=dbp_result.get('cognitive_task'),
            dbp_physical_task=dbp_result.get('physical_task'),
            dbp_air_alert=dbp_result.get('air_alert'),
            pp_winner=pp_result.get('winner'),
            pp_ref_mae=pp_result.get('ref_mae'),
            pp_cognitive_task=pp_result.get('cognitive_task'),
            pp_physical_task=pp_result.get('physical_task'),
            pp_air_alert=pp_result.get('air_alert'),
        )
        
        return result
    
    def _analyze_target(self, df_subject, df_base, target_col, feature_extractor):
        """Analyze a specific target (DBP or PP)."""
        # Extract features
        X_raw = feature_extractor.extract(df_base)
        y = df_base[target_col].values
        
        # Cross-validation for model selection
        cv_perf = self.cv_validator.evaluate_models(X_raw, y)
        
        if cv_perf is None:
            return {'winner': 'NA', 'ref_mae': np.nan}
        
        winner = cv_perf.get_winner()
        ref_mae = cv_perf.get_best_score()
        
        # Train final model
        model, idxs, scaler = self.trainer.train(X_raw, y, winner)
        
        # Evaluate on different conditions
        result = {
            'winner': winner,
            'ref_mae': ref_mae
        }
        
        for cond_name, label in [
            ('cognitive_task', Columns.LABEL_COGNITIVE_TASK),
            ('physical_task', Columns.LABEL_PHYSICAL_TASK),
            ('air_alert', Columns.LABEL_AIR_ALERT)
        ]:
            df_cond = df_subject[df_subject[Columns.LABEL] == label]
            
            if df_cond.empty:
                continue
            
            X_cond = feature_extractor.extract(df_cond)
            X_cond_scaled = scaler.transform(X_cond)
            y_true = df_cond[target_col].values
            y_pred = model.predict(X_cond_scaled[:, idxs])
            
            mae = np.mean(np.abs(y_true - y_pred))
            bias = np.median(y_true - y_pred)
            anomaly = 100 * (mae - ref_mae) / (ref_mae + self.config.EPSILON)
            
            result[cond_name] = ConditionMetrics(
                n=len(df_cond),
                mae=mae,
                delta_bias=bias,
                anomaly=anomaly
            )
        
        return result


class CohortAnalyzer:
    """Analyzes cohort-level statistics."""
    
    def __init__(self, config: Config):
        self.config = config
        self.dist_analyzer = DistributionAnalyzer(config)
        self.corr_analyzer = CorrelationAnalyzer(config)
        self.mtc = MultipleTestingCorrector(config)
    
    def generate_summary(self, res_df: pd.DataFrame, output_file: Path):
        """Generate results summary file."""
        with open(output_file, 'w') as f:
            f.write("RESULTS SUMMARY\\n")
            f.write("=" * 40 + "\\n\\n")
            
            # Model counts for DBP
            if "DBP_Winner" in res_df.columns:
                f.write("DBP Model Counts:\\n")
                f.write(f"{res_df['DBP_Winner'].value_counts()}\\n\\n")
            
            # Baseline MAE distribution
            self._write_baseline_stats(f, res_df)
            
            # Task analysis with FDR correction
            self._write_task_analysis(f, res_df)
            
            # Subgroup analysis
            self._write_subgroup_analysis(f, res_df)
    
    def _write_baseline_stats(self, f, res_df):
        """Write baseline performance statistics."""
        if "DBP_Ref_MAE" not in res_df.columns:
            return
        
        b_maes = res_df["DBP_Ref_MAE"].dropna()
        if b_maes.empty:
            return
        
        f.write(f"Baseline (Train) MAE Distribution (N={len(b_maes)}):\\n")
        f.write(f"  Median [IQR]: {np.median(b_maes):.2f} [{np.percentile(b_maes, 25):.2f}, {np.percentile(b_maes, 75):.2f}] mmHg\\n")
        f.write(f"  Range: [{b_maes.min():.2f}, {b_maes.max():.2f}] mmHg\\n\\n")
    
    def _write_task_analysis(self, f, res_df):
        """Write task condition analysis with FDR correction."""
        # Collect p-values for FDR correction
        p_map = {}
        
        for cond in [Columns.LABEL_COGNITIVE_TASK, Columns.LABEL_PHYSICAL_TASK, Columns.LABEL_AIR_ALERT]:
            for metric, suffix in [("Anomaly", "Anomaly"), ("Delta Bias", "DeltaBias")]:
                col = f"DBP_{cond}_{suffix}"
                if col not in res_df.columns:
                    continue
                
                valid = res_df[res_df[f"DBP_{cond}_N"] > 0]
                if valid.empty:
                    continue
                
                stats = self.dist_analyzer.compute_stats(valid[col].values, f"{cond} {metric}")
                if stats:
                    p_map[(cond, metric)] = stats.p_value
        
        # Apply FDR correction
        p_items = list(p_map.items())
        p_values = [p for _, p in p_items]
        q_values = self.mtc.apply_fdr(p_values)
        q_lookup = {p_items[i][0]: q_values[i] for i in range(len(p_items))}
        
        # Write results
        for cond in [Columns.LABEL_COGNITIVE_TASK, Columns.LABEL_PHYSICAL_TASK, Columns.LABEL_AIR_ALERT]:
            col_anom = f"DBP_{cond}_Anomaly"
            col_bias = f"DBP_{cond}_DeltaBias"
            
            if col_anom not in res_df.columns:
                continue
            
            valid = res_df[res_df[f"DBP_{cond}_N"] > 0]
            if valid.empty:
                continue
            
            stats_anom = self.dist_analyzer.compute_stats(valid[col_anom].values, f"DBP {cond} Anomaly")
            stats_bias = self.dist_analyzer.compute_stats(valid[col_bias].values, f"DBP {cond} Bias")
            
            if not stats_anom or not stats_bias:
                continue
            
            q_anom = q_lookup.get((cond, "Anomaly"), 1.0)
            q_bias = q_lookup.get((cond, "Delta Bias"), 1.0)
            
            f.write(f"\\nDBP {cond} (N={len(valid)})\\n")
            f.write(f"  Median MAE inflation = {stats_anom.median:.2f}%; Wilcoxon p={stats_anom.p_value:.4f} (q={q_anom:.4f})\\n")
            f.write(f"  Median delta_bias = {stats_bias.median:.2f} mmHg; Wilcoxon p={stats_bias.p_value:.4f} (q={q_bias:.4f})\\n")
            
            # Responder count for cognitive task
            if cond == Columns.LABEL_COGNITIVE_TASK:
                n_resp = ((valid[col_anom] > self.config.RESPONDER_ANOMALY_THRESHOLD) |
                         (valid[col_bias] > self.config.RESPONDER_BIAS_THRESHOLD)).sum()
                f.write(f"  DBP Responders: {n_resp}/{len(valid)} ({100*n_resp/len(valid):.1f}%)\\n")
    
    def _write_subgroup_analysis(self, f, res_df):
        """Write responder vs non-responder analysis."""
        f.write("\\n" + "=" * 40 + "\\n")
        f.write("SUBGROUP ANALYSIS: RESPONDERS VS NON-RESPONDERS\\n")
        f.write("=" * 40 + "\\n")
        
        # Define responders
        res_df["Is_Responder"] = (
            (res_df["DBP_Cognitive Task_Anomaly"] > self.config.RESPONDER_ANOMALY_THRESHOLD) |
            (res_df["DBP_Cognitive Task_DeltaBias"] > self.config.RESPONDER_BIAS_THRESHOLD)
        ).astype(int)
        
        valid_task = res_df[res_df["DBP_Cognitive Task_N"] > 0]
        resp = valid_task[valid_task["Is_Responder"] == 1]
        non_resp = valid_task[valid_task["Is_Responder"] == 0]
        
        f.write(f"N Responders: {len(resp)}, N Non-Responders: {len(non_resp)}\\n\\n")
        
        # Compare metrics
        metrics = [
            ("DBP_Cognitive Task_MAE", "mmHg"),
            ("DBP_Physical Task_MAE", "mmHg"),
            ("DBP_Cognitive Task_DeltaBias", "mmHg"),
            ("DBP_Physical Task_DeltaBias", "mmHg")
        ]
        
        for m, unit in metrics:
            if m not in valid_task.columns:
                continue
            
            g_resp = resp[m].dropna()
            g_non = non_resp[m].dropna()
            
            if len(g_resp) < 2 or len(g_non) < 2:
                continue
            
            u, p = mannwhitneyu(g_resp, g_non)
            
            f.write(f"{m}:\\n")
            f.write(f"  Responders: {np.median(g_resp):.2f} [{np.percentile(g_resp, 25):.2f}, {np.percentile(g_resp, 75):.2f}] {unit}\\n")
            f.write(f"  Non-Responders: {np.median(g_non):.2f} [{np.percentile(g_non, 25):.2f}, {np.percentile(g_non, 75):.2f}] {unit}\\n")
            f.write(f"  MW-U Test: U={u:.1f}, p={p:.4f}\\n\\n")


def main():
    """Main pipeline execution."""
    print("=" * 80)
    print("ABPM HEMODYNAMIC PIPELINE (REFACTORED)")
    print("=" * 80)
    
    # Initialize configuration
    config = Config()
    np.random.seed(config.RANDOM_SEED)
    
    # Load data
    loader = DataLoader(config)
    df = loader.load_monitoring_data()
    
    # Generate Table 1: Demographics
    print("\\nGenerating demographics...")
    gb = df.groupby(Columns.LABEL)
    table1 = gb.agg({
        Columns.PAT_ID: 'nunique',
        Columns.SBP: ['count', 'median'],
        Columns.DBP: ['median'],
        Columns.HR: ['median']
    })
    table1.to_csv(config.get_results_path(config.TABLE1_OUTPUT))
    print(f"Table 1 saved: {config.get_results_path(config.TABLE1_OUTPUT)}")
    
    # Subject-level analysis
    print("\\nStarting subject analysis...")
    subjects = df[Columns.PAT_ID].unique()
    analyzer = SubjectAnalyzer(config)
    tracker = ProgressTracker(len(subjects), "Subject Analysis")
    
    results = []
    for subject_id in subjects:
        df_subject = df[df[Columns.PAT_ID] == subject_id]
        result = analyzer.analyze_subject(subject_id, df_subject)
        results.append(result.to_dict())
        tracker.update()
    
    # Create results DataFrame
    res_df = pd.DataFrame(results)
    res_df.to_csv(config.get_results_path(config.SUBJECT_METRICS_OUTPUT), index=False)
    print(f"\\nSubject metrics saved: {config.get_results_path(config.SUBJECT_METRICS_OUTPUT)}")
    
    # Cohort-level statistics
    print("\\nComputing cohort statistics...")
    cohort_analyzer = CohortAnalyzer(config)
    cohort_analyzer.generate_summary(
        res_df,
        config.get_results_path(config.SUMMARY_OUTPUT)
    )
    print(f"Summary saved: {config.get_results_path(config.SUMMARY_OUTPUT)}")
    
    # TODO: Add correlation analysis if aggregated data available
    # TODO: Add classifier training if classifier data available
    
    # Generate figures
    viz_manager = VisualizationManager(config)
    viz_manager.generate_all(df, res_df)
    
    print("\\nPipeline complete! Results in results/ directory.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\\nERROR: Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
