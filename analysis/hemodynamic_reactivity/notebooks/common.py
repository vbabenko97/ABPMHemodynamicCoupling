from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHART_DIR = ROOT / "charts"
CHART_SCRIPT_DIR = ROOT / "chart_scripts"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    mon = pd.read_csv(DATA_DIR / "monitoring_data.csv", parse_dates=["datetime"])
    agg = pd.read_csv(DATA_DIR / "aggregated_data.csv")
    if "avg_reaction_time_battey_2" in agg.columns:
        agg = agg.rename(
            columns={"avg_reaction_time_battey_2": "avg_reaction_time_battery_2"}
        )
    return mon, add_derived_metrics(agg)


def add_derived_metrics(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy()
    ratio_groups = {
        "cognitive_reactivity_mean": [
            "SBP_cog_1_ratio",
            "DBP_cog_1_ratio",
            "HR_cog_1_ratio",
            "SBP_cog_2_ratio",
            "DBP_cog_2_ratio",
            "HR_cog_2_ratio",
        ],
        "physical_reactivity_mean": [
            "SBP_phys_1_ratio",
            "DBP_phys_1_ratio",
            "HR_phys_1_ratio",
            "SBP_phys_2_ratio",
            "DBP_phys_2_ratio",
            "HR_phys_2_ratio",
        ],
        "sbp_reactivity_mean": [
            "SBP_cog_1_ratio",
            "SBP_cog_2_ratio",
            "SBP_phys_1_ratio",
            "SBP_phys_2_ratio",
        ],
        "dbp_reactivity_mean": [
            "DBP_cog_1_ratio",
            "DBP_cog_2_ratio",
            "DBP_phys_1_ratio",
            "DBP_phys_2_ratio",
        ],
        "hr_reactivity_mean": [
            "HR_cog_1_ratio",
            "HR_cog_2_ratio",
            "HR_phys_1_ratio",
            "HR_phys_2_ratio",
        ],
    }
    for metric, columns in ratio_groups.items():
        agg[metric] = agg[columns].mean(axis=1, skipna=True)

    for prefix, columns in {
        "cognitive": [
            "SBP_cog_1_ratio",
            "DBP_cog_1_ratio",
            "HR_cog_1_ratio",
            "SBP_cog_2_ratio",
            "DBP_cog_2_ratio",
            "HR_cog_2_ratio",
        ],
        "physical": [
            "SBP_phys_1_ratio",
            "DBP_phys_1_ratio",
            "HR_phys_1_ratio",
            "SBP_phys_2_ratio",
            "DBP_phys_2_ratio",
            "HR_phys_2_ratio",
        ],
    }.items():
        z = agg[columns].apply(lambda s: (s - s.mean()) / s.std(ddof=0))
        agg[f"{prefix}_reactivity_index"] = z.mean(axis=1, skipna=True)

    agg["task_reactivity_gap"] = (
        agg["physical_reactivity_mean"] - agg["cognitive_reactivity_mean"]
    )
    agg["bp_load_group"] = pd.qcut(agg["bp_load_%"], q=3, labels=["Low", "Mid", "High"])
    agg["sbp_dip_group"] = pd.qcut(
        agg["sbp_dip_%"], q=3, labels=["Low dip", "Mid dip", "High dip"]
    )
    return agg


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def pct_missing(frame: pd.DataFrame) -> pd.Series:
    return (frame.isna().mean() * 100).round(1)


def bool_label(value: bool) -> str:
    return "Yes" if bool(value) else "No"


def ordered_categorical_summary(series: pd.Series) -> list[dict]:
    counts = series.value_counts(dropna=False)
    total = len(series)
    return [
        {
            "label": "Missing" if pd.isna(label) else str(label),
            "count": int(count),
            "share": round(float(count / total), 4),
        }
        for label, count in counts.items()
    ]


def ensure_dirs() -> None:
    ROOT.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)
    CHART_SCRIPT_DIR.mkdir(exist_ok=True)


def describe_frame(frame: pd.DataFrame, name: str) -> dict:
    return {
        "name": name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {k: str(v) for k, v in frame.dtypes.items()},
        "missing_pct": {k: float(v) for k, v in pct_missing(frame).items()},
        "duplicates": int(frame.duplicated().sum()),
    }
