"""
Web Pipeline Adapter
====================

Wraps existing pipeline modules for the Streamlit web interface.
Keeps all scientific logic in src/ and run_pipeline.py untouched.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# SubjectAnalyzer and CohortAnalyzer live in run_pipeline.py
from run_pipeline import CohortAnalyzer, SubjectAnalyzer
from src.config import Columns, Config
from src.data_processing import DataValidator, Labeler
from src.visualization import VisualizationManager


@dataclass
class PipelineResults:
    """Container for all pipeline outputs."""

    subject_metrics: pd.DataFrame
    demographics_figure: Figure
    summary_text: str
    figures: dict[str, Figure]
    n_subjects: int
    n_records: int


@dataclass
class SanitizationReport:
    """Report of rows dropped during pre-validation sanitization."""

    dropped_rows: pd.DataFrame
    counts: dict[str, int]  # column name → number of invalid rows

    @property
    def total_dropped(self) -> int:
        return len(self.dropped_rows)

    @property
    def has_drops(self) -> bool:
        return self.total_dropped > 0


class WebPipeline:
    """Runs the ABPM analysis pipeline from in-memory data."""

    def __init__(self):
        self.config = Config()
        np.random.seed(self.config.RANDOM_SEED)

    def sanitize(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, SanitizationReport]:
        """Filter rows with out-of-range or non-finite physiological values.

        Web-only preprocessing step. The CLI path uses strict DataValidator.
        """
        ranges = {
            Columns.SBP: (self.config.MIN_SBP, self.config.MAX_SBP),
            Columns.DBP: (self.config.MIN_DBP, self.config.MAX_DBP),
            Columns.HR: (self.config.MIN_HR, self.config.MAX_HR),
        }

        mask_keep = pd.Series(True, index=df.index)
        counts: dict[str, int] = {}

        for col, (lo, hi) in ranges.items():
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            invalid = values.notna() & (~np.isfinite(values) | (values < lo) | (values > hi))
            n_bad = int(invalid.sum())
            if n_bad:
                counts[col] = n_bad
            mask_keep &= ~invalid

        dropped = df[~mask_keep].copy()
        cleaned = df[mask_keep].copy()
        return cleaned, SanitizationReport(dropped_rows=dropped, counts=counts)

    def validate_and_preprocess(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Validate and preprocess uploaded data.

        Mirrors DataLoader.load_monitoring_data() without file I/O.
        """
        DataValidator.validate_monitoring_data(df_raw)

        df = df_raw.copy()
        df[Columns.TIME] = pd.to_datetime(df[Columns.TIME])
        df = df.sort_values([Columns.PAT_ID, Columns.TIME])
        df = df.dropna(subset=[Columns.SBP, Columns.DBP, Columns.HR])
        df[Columns.LABEL] = df.apply(Labeler.apply_hierarchy, axis=1)
        return df

    def analyze_subjects(
        self, df: pd.DataFrame, progress_callback=None
    ) -> pd.DataFrame:
        """Run per-subject analysis and return metrics DataFrame."""
        subjects = df[Columns.PAT_ID].unique()
        analyzer = SubjectAnalyzer(self.config)

        results = []
        for i, subject_id in enumerate(subjects):
            df_subject = df[df[Columns.PAT_ID] == subject_id]
            result = analyzer.analyze_subject(subject_id, df_subject)
            results.append(result.to_dict())
            if progress_callback:
                progress_callback((i + 1) / len(subjects))

        return pd.DataFrame(results)

    def compute_statistics(self, res_df: pd.DataFrame) -> str:
        """Generate cohort summary and return as string."""
        cohort = CohortAnalyzer(self.config)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        cohort.generate_summary(res_df, tmp_path)
        text = tmp_path.read_text()
        tmp_path.unlink(missing_ok=True)
        return text

    def create_demographics_figure(self, df: pd.DataFrame) -> Figure:
        """Create demographics summary table as a matplotlib figure."""
        gb = df.groupby(Columns.LABEL)
        table_data = gb.agg(
            {
                Columns.PAT_ID: "nunique",
                Columns.SBP: ["count", "median"],
                Columns.DBP: ["median"],
                Columns.HR: ["median"],
            }
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.axis("tight")
        ax.table(
            cellText=np.round(table_data.values, 2),
            colLabels=[
                f"{c[0]} ({c[1]})" if isinstance(c, tuple) else c
                for c in table_data.columns
            ],
            rowLabels=table_data.index,
            loc="center",
        )
        fig.tight_layout()
        return fig

    def generate_figures(
        self, df: pd.DataFrame, res_df: pd.DataFrame
    ) -> dict[str, Figure]:
        """Generate all analysis figures without saving to disk."""
        viz = VisualizationManager(self.config)
        return viz.generate_all_figures(df, res_df)
