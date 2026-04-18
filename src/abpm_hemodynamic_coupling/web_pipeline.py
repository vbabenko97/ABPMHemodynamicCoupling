"""
Web Pipeline Adapter
====================

Wraps existing pipeline modules for the Streamlit web interface.
Keeps all scientific logic in src/ and run_pipeline.py untouched.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from abpm_hemodynamic_coupling.config import Columns, Config
from abpm_hemodynamic_coupling.data_processing import DataValidator, Labeler
from abpm_hemodynamic_coupling.stats_analysis import (
    CorrelationAnalyzer,
    DistributionAnalyzer,
    MultipleTestingCorrector,
)
from abpm_hemodynamic_coupling.visualization import VisualizationManager

# SubjectAnalyzer and CohortAnalyzer live in run_pipeline.py
from run_pipeline import CohortAnalyzer, SubjectAnalyzer

DISPLAY_LABELS = {
    Columns.LABEL_AIR_ALERT: "Повітряна тривога",
    Columns.LABEL_BEFORE_TESTING: "Перед тестуванням",
    Columns.LABEL_COGNITIVE_PRE: "Перед когнітивним навантаженням",
    Columns.LABEL_COGNITIVE_TASK: "Когнітивне навантаження",
    Columns.LABEL_PHYSICAL_PRE: "Перед фізичним навантаженням",
    Columns.LABEL_PHYSICAL_TASK: "Фізичне навантаження",
    Columns.LABEL_SLEEP: "Сон",
    Columns.LABEL_MANUAL_TRIGGER: "Ручна позначка",
    Columns.LABEL_BREAK: "Перерва",
    Columns.LABEL_BASELINE: "Базовий стан",
}

DISPLAY_COLUMN_LABELS = {
    Columns.PAT_ID: "Учасники",
    Columns.SBP: "САТ",
    Columns.DBP: "ДАТ",
    Columns.HR: "ЧСС",
}

DISPLAY_AGGREGATION_LABELS = {
    "nunique": "унік.",
    "count": "n",
    "median": "медіана",
}

DISPLAY_MODEL_LABELS = {
    "Brandon": "Brandon",
    "Lasso": "Lasso",
    "RFE": "RFE",
    "OLS(SBP)": "МНК (САТ)",
    "OLS(SBP,HR)": "МНК (САТ, ЧСС)",
    "NA": "Н/Д",
}

DISPLAY_RESULT_COLUMNS = {
    "participant_id": "ID учасника",
    "Train_N": "N у навчанні",
    "DBP_Winner": "Найкраща модель ДАТ",
    "DBP_Ref_MAE": "Референтний MAE ДАТ",
    "DBP_Cognitive Task_N": "ДАТ Когнітивне навантаження N",
    "DBP_Cognitive Task_MAE": "ДАТ Когнітивне навантаження MAE",
    "DBP_Cognitive Task_DeltaBias": "ДАТ Когнітивне навантаження Зміщення",
    "DBP_Cognitive Task_Anomaly": "ДАТ Когнітивне навантаження Аномалія",
    "DBP_Physical Task_N": "ДАТ Фізичне навантаження N",
    "DBP_Physical Task_MAE": "ДАТ Фізичне навантаження MAE",
    "DBP_Physical Task_DeltaBias": "ДАТ Фізичне навантаження Зміщення",
    "DBP_Physical Task_Anomaly": "ДАТ Фізичне навантаження Аномалія",
    "DBP_Air Alert_N": "ДАТ Повітряна тривога N",
    "DBP_Air Alert_MAE": "ДАТ Повітряна тривога MAE",
    "DBP_Air Alert_DeltaBias": "ДАТ Повітряна тривога Зміщення",
    "DBP_Air Alert_Anomaly": "ДАТ Повітряна тривога Аномалія",
}

SUMMARY_REPLACEMENTS = [
    ("RESULTS SUMMARY", "ПІДСУМОК РЕЗУЛЬТАТІВ"),
    ("DBP Model Counts:", "Кількість моделей для ДАТ:"),
    (
        "Baseline (Train) MAE Distribution",
        "Розподіл MAE на базовому інтервалі (навчання)",
    ),
    (
        "SUBGROUP ANALYSIS: RESPONDERS VS NON-RESPONDERS",
        "ПІДГРУПОВИЙ АНАЛІЗ: РЕСПОНДЕНТИ ПРОТИ НЕРЕСПОНДЕНТІВ",
    ),
    ("Median [IQR]:", "Медіана [IQR]:"),
    ("Range:", "Діапазон:"),
    ("Median MAE inflation =", "Медіанне зростання MAE ="),
    ("Median delta_bias =", "Медіанне зміщення залишків ="),
    ("DBP Responders:", "Респонденти за ДАТ:"),
    ("N Responders:", "К-сть респондентів:"),
    (", N Non-Responders:", ", к-сть нереспондентів:"),
    ("Responders:", "Респонденти:"),
    ("Non-Responders:", "Нереспонденти:"),
    ("MW-U Test:", "Тест Манна-Вітні:"),
    ("Wilcoxon p=", "p за Вілкоксоном="),
    ("DBP_Cognitive Task_MAE", "ДАТ_Когнітивне навантаження_MAE"),
    ("DBP_Physical Task_MAE", "ДАТ_Фізичне навантаження_MAE"),
    ("DBP_Cognitive Task_DeltaBias", "ДАТ_Когнітивне навантаження_Зміщення"),
    ("DBP_Physical Task_DeltaBias", "ДАТ_Фізичне навантаження_Зміщення"),
    ("Cognitive Task", "Когнітивне навантаження"),
    ("Physical Task", "Фізичне навантаження"),
    ("Air Alert", "Повітряна тривога"),
    ("Baseline", "Базовий стан"),
    ("DBP ", "ДАТ "),
    (" mmHg", " мм рт. ст."),
]

SUMMARY_CONDITIONS = [
    ("Cognitive Task", "Когнітивне навантаження"),
    ("Physical Task", "Фізичне навантаження"),
    ("Air Alert", "Повітряна тривога"),
]

SUBGROUP_METRIC_LABELS = {
    "DBP_Cognitive Task_MAE": "ДАТ: когнітивне навантаження — MAE",
    "DBP_Physical Task_MAE": "ДАТ: фізичне навантаження — MAE",
    "DBP_Cognitive Task_DeltaBias": "ДАТ: когнітивне навантаження — зміщення",
    "DBP_Physical Task_DeltaBias": "ДАТ: фізичне навантаження — зміщення",
}


@dataclass
class PipelineResults:
    """Container for all pipeline outputs."""

    subject_metrics: pd.DataFrame
    demographics_table: pd.DataFrame
    summary_view: "SummaryViewData"
    summary_text: str
    figures: dict[str, Figure]
    n_subjects: int
    n_records: int


@dataclass
class SummaryViewData:
    """Display-ready summary tables for the Streamlit interface."""

    model_counts: pd.DataFrame
    baseline_stats: dict[str, float | int]
    condition_stats: pd.DataFrame
    subgroup_stats: pd.DataFrame
    n_responders: int
    n_non_responders: int


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
        return self._localize_summary_text(text)

    def _localize_summary_text(self, text: str) -> str:
        """Translate the web summary output to Ukrainian."""
        localized = text
        for source, target in SUMMARY_REPLACEMENTS:
            localized = localized.replace(source, target)
        return localized

    def build_summary_view(self, res_df: pd.DataFrame) -> SummaryViewData:
        """Build structured summary tables for display in Streamlit."""
        model_counts = self._build_model_counts_table(res_df)
        baseline_stats = self._build_baseline_stats(res_df)
        condition_stats = self._build_condition_stats_table(res_df)
        subgroup_stats, n_responders, n_non_responders = self._build_subgroup_stats_table(res_df)

        return SummaryViewData(
            model_counts=model_counts,
            baseline_stats=baseline_stats,
            condition_stats=condition_stats,
            subgroup_stats=subgroup_stats,
            n_responders=n_responders,
            n_non_responders=n_non_responders,
        )

    def localize_subject_metrics(self, res_df: pd.DataFrame) -> pd.DataFrame:
        """Return a display-only Ukrainian version of the subject metrics table."""
        display_df = res_df.copy()

        if "DBP_Winner" in display_df.columns:
            display_df["DBP_Winner"] = display_df["DBP_Winner"].map(
                lambda value: DISPLAY_MODEL_LABELS.get(value, value)
            )

        return display_df.rename(
            columns={
                column: DISPLAY_RESULT_COLUMNS.get(column, column)
                for column in display_df.columns
            }
        )

    def create_demographics_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a display-ready demographics summary table."""
        gb = df.groupby(Columns.LABEL)
        table_data = gb.agg(
            {
                Columns.PAT_ID: "nunique",
                Columns.SBP: ["count", "median"],
                Columns.DBP: ["median"],
                Columns.HR: ["median"],
            }
        )

        def format_column_name(column: tuple[str, str] | str) -> str:
            if not isinstance(column, tuple):
                return DISPLAY_COLUMN_LABELS.get(column, column)
            if column[0] == Columns.PAT_ID and column[1] == "nunique":
                return "Учасники (унік.)"
            if column[0] == Columns.SBP and column[1] == "count":
                return "К-сть вимірювань (n)"
            return (
                f"{DISPLAY_COLUMN_LABELS.get(column[0], column[0])} "
                f"({DISPLAY_AGGREGATION_LABELS.get(column[1], column[1])})"
            )

        display_df = pd.DataFrame(
            table_data.values,
            columns=[format_column_name(column) for column in table_data.columns],
            index=[DISPLAY_LABELS.get(label, label) for label in table_data.index],
        ).reset_index(names="Стан")

        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        display_df[numeric_cols] = display_df[numeric_cols].round(2)
        return display_df

    def generate_figures(
        self, df: pd.DataFrame, res_df: pd.DataFrame
    ) -> dict[str, Figure]:
        """Generate all analysis figures without saving to disk."""
        viz = VisualizationManager(self.config)
        return viz.generate_all_figures(df, res_df)

    def _build_model_counts_table(self, res_df: pd.DataFrame) -> pd.DataFrame:
        """Build the winning-model counts table."""
        if "DBP_Winner" not in res_df.columns:
            return pd.DataFrame(columns=["Модель", "Кількість учасників"])

        counts = (
            res_df["DBP_Winner"]
            .value_counts()
            .rename_axis("Модель")
            .reset_index(name="Кількість учасників")
        )
        counts["Модель"] = counts["Модель"].map(
            lambda value: DISPLAY_MODEL_LABELS.get(value, value)
        )
        return counts

    def _build_baseline_stats(self, res_df: pd.DataFrame) -> dict[str, float | int]:
        """Build baseline MAE summary statistics."""
        if "DBP_Ref_MAE" not in res_df.columns:
            return {"n": 0, "median": np.nan, "q25": np.nan, "q75": np.nan, "min": np.nan, "max": np.nan}

        values = res_df["DBP_Ref_MAE"].dropna()
        if values.empty:
            return {"n": 0, "median": np.nan, "q25": np.nan, "q75": np.nan, "min": np.nan, "max": np.nan}

        return {
            "n": int(len(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    def _build_condition_stats_table(self, res_df: pd.DataFrame) -> pd.DataFrame:
        """Build the per-condition cohort summary table."""
        dist_analyzer = DistributionAnalyzer(self.config)
        mtc = MultipleTestingCorrector(self.config)

        p_map: dict[tuple[str, str], float] = {}
        stats_map: dict[tuple[str, str], object] = {}

        for condition_key, _ in SUMMARY_CONDITIONS:
            anomaly_col = f"DBP_{condition_key}_Anomaly"
            bias_col = f"DBP_{condition_key}_DeltaBias"
            n_col = f"DBP_{condition_key}_N"

            if anomaly_col not in res_df.columns or bias_col not in res_df.columns:
                continue

            valid = res_df[res_df[n_col] > 0]
            if valid.empty:
                continue

            anomaly_stats = dist_analyzer.compute_stats(
                valid[anomaly_col].values,
                f"DBP {condition_key} Anomaly",
            )
            bias_stats = dist_analyzer.compute_stats(
                valid[bias_col].values,
                f"DBP {condition_key} Bias",
            )

            if anomaly_stats is None or bias_stats is None:
                continue

            stats_map[(condition_key, "Anomaly")] = anomaly_stats
            stats_map[(condition_key, "DeltaBias")] = bias_stats
            p_map[(condition_key, "Anomaly")] = anomaly_stats.p_value
            p_map[(condition_key, "DeltaBias")] = bias_stats.p_value

        q_lookup = {
            key: q_value
            for key, q_value in zip(
                p_map.keys(),
                mtc.apply_fdr(list(p_map.values())),
                strict=False,
            )
        }

        rows = []
        for condition_key, condition_label in SUMMARY_CONDITIONS:
            anomaly_key = (condition_key, "Anomaly")
            bias_key = (condition_key, "DeltaBias")
            if anomaly_key not in stats_map or bias_key not in stats_map:
                continue

            n_valid = int((res_df[f"DBP_{condition_key}_N"] > 0).sum())
            responder_text = "—"
            if condition_key == "Cognitive Task":
                valid = res_df[res_df[f"DBP_{condition_key}_N"] > 0]
                n_responders = (
                    (valid[f"DBP_{condition_key}_Anomaly"] > self.config.RESPONDER_ANOMALY_THRESHOLD)
                    | (valid[f"DBP_{condition_key}_DeltaBias"] > self.config.RESPONDER_BIAS_THRESHOLD)
                ).sum()
                responder_text = f"{n_responders}/{len(valid)} ({100 * n_responders / len(valid):.1f}%)"

            rows.append(
                {
                    "Умова": condition_label,
                    "Учасники (n)": n_valid,
                    "Медіанне зростання MAE (%)": round(stats_map[anomaly_key].median, 2),
                    "p (MAE)": round(stats_map[anomaly_key].p_value, 4),
                    "q (MAE)": round(q_lookup.get(anomaly_key, 1.0), 4),
                    "Медіанне зміщення (мм рт. ст.)": round(stats_map[bias_key].median, 2),
                    "p (зміщення)": round(stats_map[bias_key].p_value, 4),
                    "q (зміщення)": round(q_lookup.get(bias_key, 1.0), 4),
                    "Респонденти за ДАТ": responder_text,
                }
            )

        return pd.DataFrame(rows)

    def _build_subgroup_stats_table(
        self, res_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, int, int]:
        """Build responder vs non-responder comparison table."""
        comparison = CorrelationAnalyzer(self.config)
        subgroup_df = res_df.copy()
        subgroup_df["Is_Responder"] = (
            (subgroup_df["DBP_Cognitive Task_Anomaly"] > self.config.RESPONDER_ANOMALY_THRESHOLD)
            | (subgroup_df["DBP_Cognitive Task_DeltaBias"] > self.config.RESPONDER_BIAS_THRESHOLD)
        ).astype(int)

        valid_task = subgroup_df[subgroup_df["DBP_Cognitive Task_N"] > 0]
        responders = valid_task[valid_task["Is_Responder"] == 1]
        non_responders = valid_task[valid_task["Is_Responder"] == 0]

        rows = []
        for metric, label in SUBGROUP_METRIC_LABELS.items():
            if metric not in valid_task.columns:
                continue

            responder_values = responders[metric].dropna().values
            non_responder_values = non_responders[metric].dropna().values

            if len(responder_values) < 2 or len(non_responder_values) < 2:
                continue

            u_stat, p_value = comparison.mannwhitney_test(
                responder_values,
                non_responder_values,
            )

            rows.append(
                {
                    "Показник": label,
                    "Респонденти": self._format_median_iqr(responder_values),
                    "Нереспонденти": self._format_median_iqr(non_responder_values),
                    "U": round(u_stat, 1),
                    "p": round(p_value, 4),
                }
            )

        return pd.DataFrame(rows), int(len(responders)), int(len(non_responders))

    def _format_median_iqr(self, values: np.ndarray) -> str:
        """Format median and IQR as a compact display string."""
        return (
            f"{np.median(values):.2f} "
            f"[{np.percentile(values, 25):.2f}, {np.percentile(values, 75):.2f}] "
            "мм рт. ст."
        )
