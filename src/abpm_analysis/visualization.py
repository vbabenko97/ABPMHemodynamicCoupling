"""
Visualization and Figure Generation
=====================================

Generates all research figures with consistent styling.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

from .config import Config, Columns
from .feature_engineering import DBPFeatureExtractor
from .modeling import ModelTrainer


class FigureGenerator(ABC):
    """Abstract base class for figure generators."""
    
    def __init__(self, config: Config):
        """
        Initialize figure generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    @abstractmethod
    def generate(self, df: pd.DataFrame, res_df: pd.DataFrame) -> None:
        """
        Generate and save figure.
        
        Args:
            df: Monitoring data DataFrame
            res_df: Per-subject results DataFrame
        """
        pass


class Figure2Generator(FigureGenerator):
    """Generates Figure 2: Dot plots for Anomaly and DeltaBias."""
    
    def generate(self, df: pd.DataFrame, res_df: pd.DataFrame) -> None:
        """Generate Figure 2: MAE inflation and signed residual bias dot plots."""
        # Use only subjects with valid task data
        valid_df = res_df[res_df["DBP_Cognitive Task_N"] > 0].copy()
        if valid_df.empty:
            print("Skipping Figure 2 (no valid task data)")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Fig 2A: MAE inflation %
        sns.stripplot(
            data=valid_df,
            y="DBP_Cognitive Task_Anomaly",
            ax=ax1,
            color="#1f77b4",
            size=9,
            alpha=0.7,
            jitter=0.2
        )
        med_anom = valid_df["DBP_Cognitive Task_Anomaly"].median()
        ax1.axhline(med_anom, color='red', linestyle='--', label=f'Median: {med_anom:.1f}%')
        ax1.set_title(r"A: MAE inflation ($A_{i,cog}$)", fontsize=16, fontweight='bold', pad=15)
        ax1.set_ylabel("MAE inflation (%)", fontsize=14)
        ax1.set_xlabel(f"N={len(valid_df)} Participants", fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.legend(fontsize=12)
        
        # Fig 2B: Signed residual bias
        sns.stripplot(
            data=valid_df,
            y="DBP_Cognitive Task_DeltaBias",
            ax=ax2,
            color="#ff7f0e",
            size=9,
            alpha=0.7,
            jitter=0.2
        )
        med_bias = valid_df["DBP_Cognitive Task_DeltaBias"].median()
        ax2.axhline(med_bias, color='red', linestyle='--', label=f'Median: {med_bias:.2f} mmHg')
        ax2.set_title("B: Signed residual bias (ΔBias_i,cog)", fontsize=16, fontweight='bold', pad=15)
        ax2.set_ylabel("Signed residual bias (mmHg)", fontsize=14)
        ax2.set_xlabel(f"N={len(valid_df)} Participants", fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.legend(fontsize=12)
        
        plt.tight_layout()
        save_path = self.config.get_results_path(self.config.FIGURE_2_OUTPUT)
        plt.savefig(save_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"Figure 2 saved: {save_path}")


class Figure3Generator(FigureGenerator):
    """Generates Figure 3: Observed vs Predicted DBP for case studies."""
    
    def generate(self, df: pd.DataFrame, res_df: pd.DataFrame) -> None:
        """Generate Figure 3: Observed vs Predicted DBP scatter plots."""
        subject_ids = [36, 35]
        titles = [
            "Subject 36 (Low coupling deviation)",
            "Subject 35 (Highest residual-bias example)"
        ]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Feature extractor and trainer
        feature_extractor = DBPFeatureExtractor(self.config)
        trainer = ModelTrainer(self.config)
        
        for ax, pid, title in zip(axes, subject_ids, titles):
            df_p = df[df[Columns.PAT_ID] == pid]
            row_res = res_df[res_df[Columns.PAT_ID] == pid]
            if row_res.empty:
                continue
            
            winner = row_res["DBP_Winner"].iloc[0]
            if winner == "NA":
                continue
            
            # Train model on baseline
            df_base = df_p[df_p[Columns.LABEL] == Columns.LABEL_BASELINE]
            X_base = feature_extractor.extract(df_base)
            y_base = df_base[Columns.DBP].values
            
            model, idxs, scaler = trainer.train(X_base, y_base, winner)
            
            # Scatter points for each condition
            styles = [
                (Columns.LABEL_BASELINE, "o", "#1f77b4", 0.6),
                (Columns.LABEL_COGNITIVE_TASK, "x", "#ff7f0e", 0.9),
                (Columns.LABEL_PHYSICAL_TASK, "^", "#2ca02c", 0.9)
            ]
            
            all_y_true, all_y_pred = [], []
            for lbl, mkr, clr, alph in styles:
                sub = df_p[df_p[Columns.LABEL] == lbl]
                if sub.empty:
                    continue
                
                X_raw = feature_extractor.extract(sub)
                X_s = scaler.transform(X_raw)
                y_true = sub[Columns.DBP].values
                y_pred = model.predict(X_s[:, idxs])
                
                ax.scatter(
                    y_true, y_pred,
                    marker=mkr, color=clr, alpha=alph, s=70,
                    label=f"{lbl} (n={len(sub)})"
                )
                all_y_true.extend(y_true.tolist())
                all_y_pred.extend(y_pred.tolist())
            
            if not all_y_true:
                continue
            
            # Diagonal line
            lim_min = min(min(all_y_true), min(all_y_pred)) - 5
            lim_max = max(max(all_y_true), max(all_y_pred)) + 5
            ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='gray', alpha=0.5, label="Perfect Fit")
            
            # Annotations
            anom = row_res["DBP_Cognitive Task_Anomaly"].iloc[0]
            bias = row_res["DBP_Cognitive Task_DeltaBias"].iloc[0]
            desc = "(DBP over predicted)" if anom < 0 else "(DBP under predicted)"
            textstr = '\\n'.join((
                r'$A_{i,cog} = %.1f\%%$' % (anom, ),
                r'$\Delta Bias_{i,cog} = %.2f$ mmHg' % (bias, ),
                desc
            ))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("Observed DBP (mmHg)", fontsize=12)
            ax.set_ylabel("Predicted DBP (mmHg)", fontsize=12)
            ax.legend(fontsize=11, loc='lower right')
            ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        save_path = self.config.get_results_path(self.config.FIGURE_3_OUTPUT)
        plt.savefig(save_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"Figure 3 saved: {save_path}")


class Figure4Generator(FigureGenerator):
    """Generates Figure 4: Time-series residuals for case studies."""
    
    def generate(self, df: pd.DataFrame, res_df: pd.DataFrame) -> None:
        """Generate Figure 4: Time-series with residuals."""
        subject_ids = [37, 39]
        titles = [
            "Subject 37 (Short Alert Interval)",
            "Subject 39 (Most Alert Readings)"
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex='col')
        
        # Feature extractor and trainer
        feature_extractor = DBPFeatureExtractor(self.config)
        trainer = ModelTrainer(self.config)
        
        for i, pid in enumerate(subject_ids):
            df_p = df[df[Columns.PAT_ID] == pid].sort_values(Columns.TIME).copy()
            row_res = res_df[res_df[Columns.PAT_ID] == pid]
            if row_res.empty:
                continue
            
            winner = row_res["DBP_Winner"].iloc[0]
            if winner == "NA":
                continue
            
            # Train model on baseline
            df_base = df_p[df_p[Columns.LABEL] == Columns.LABEL_BASELINE]
            X_base = feature_extractor.extract(df_base)
            y_base = df_base[Columns.DBP].values
            model, idxs, scaler = trainer.train(X_base, y_base, winner)
            
            # Full predictions
            X_all = feature_extractor.extract(df_p)
            X_all_s = scaler.transform(X_all)
            all_preds = model.predict(X_all_s[:, idxs])
            all_resids = df_p[Columns.DBP].values - all_preds
            
            # Top Plot: DBP vs Pred
            ax_top = axes[0, i]
            ax_top.plot(df_p[Columns.TIME], df_p[Columns.DBP], 'o', markersize=6, label="Observed DBP", color="#1f77b4")
            ax_top.plot(df_p[Columns.TIME], all_preds, '-', label="Predicted DBP", color="#ff7f0e", linewidth=2.5)
            
            # Bottom Plot: Residuals
            ax_bot = axes[1, i]
            ax_bot.scatter(df_p[Columns.TIME], all_resids, color="#d62728", s=40, edgecolors='black', linewidth=0.5)
            ax_bot.axhline(0, color='black', linestyle='--', linewidth=1.5)
            
            # Shading for Sleep and Alert
            for ax in [ax_top, ax_bot]:
                # Create block identifier for consecutive labels
                df_p['label_shifted'] = df_p[Columns.LABEL].shift()
                df_p['block'] = (df_p[Columns.LABEL] != df_p['label_shifted']).cumsum()
                
                alert_labeled = False
                sleep_labeled = False
                
                for _, block_df in df_p.groupby('block'):
                    lbl = block_df[Columns.LABEL].iloc[0]
                    t_start = block_df[Columns.TIME].min()
                    t_end = block_df[Columns.TIME].max()
                    
                    if lbl == Columns.LABEL_AIR_ALERT:
                        label = 'Air Alert' if (i == 0 and ax == ax_top and not alert_labeled) else ""
                        ax.axvspan(t_start, t_end, color='red', alpha=0.3, label=label)
                        alert_labeled = True
                    elif "Sleep" in lbl:
                        label = 'Sleep' if (i == 0 and ax == ax_top and not sleep_labeled) else ""
                        ax.axvspan(t_start, t_end, color='lightblue', alpha=0.2, label=label)
                        sleep_labeled = True
            
            ax_top.set_title(titles[i], fontsize=15, fontweight='bold', pad=15)
            ax_top.set_ylabel("DBP (mmHg)", fontsize=13)
            ax_bot.set_ylabel("Residual (mmHg)", fontsize=13)
            ax_bot.set_xlabel("Time", fontsize=13)
            
            ax_bot.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax_top.grid(True, alpha=0.2)
            ax_bot.grid(True, alpha=0.2)
            
            if i == 0:
                ax_top.legend(fontsize=11, loc='upper left')
        
        plt.tight_layout()
        save_path = self.config.get_results_path(self.config.FIGURE_4_OUTPUT)
        plt.savefig(save_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"Figure 4 saved: {save_path}")


class VisualizationManager:
    """Manages generation of all figures."""
    
    def __init__(self, config: Config):
        """
        Initialize visualization manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.generators = [
            Figure2Generator(config),
            Figure3Generator(config),
            Figure4Generator(config)
        ]
    
    def generate_all(self, df: pd.DataFrame, res_df: pd.DataFrame) -> None:
        """
        Generate all figures.
        
        Args:
            df: Monitoring data DataFrame
            res_df: Per-subject results DataFrame
        """
        print("\\n" + "="*80)
        print("GENERATING FIGURES")
        print("="*80)
        
        for generator in self.generators:
            try:
                generator.generate(df, res_df)
            except Exception as e:
                print(f"Error generating {generator.__class__.__name__}: {e}")
