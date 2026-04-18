#!/usr/bin/env python3
"""Generate presentation-friendly figures from analysis outputs.

These figures are optimized for projected slides rather than paper layout.

Usage:
    python scripts/generate_presentation_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MPL_DIR = ROOT / "tmp" / "matplotlib"
XDG_CACHE_DIR = ROOT / "tmp" / "xdg-cache"

MPL_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

METRICS_CSV = RESULTS_DIR / "per_subject_metrics.csv"
TABLE1_CSV = RESULTS_DIR / "table1_final.csv"
THRESHOLDS_CSV = RESULTS_DIR / "sensitivity_screening_thresholds.csv"

FIG_CONTEXT = RESULTS_DIR / "presentation_context_breakdown.png"
FIG_COGNITIVE = RESULTS_DIR / "presentation_cognitive_patterns.png"
FIG_THRESHOLDS = RESULTS_DIR / "presentation_threshold_sensitivity.png"

NAVY = "#1B2A4A"
BLUE = "#2E86AB"
RED = "#E05263"
LIGHT_BLUE = "#E8F0FA"
LIGHT_RED = "#FDEEEF"
GRAY = "#6B7280"
LIGHT_GRAY = "#D6D9DE"
BG = "#F5F6F8"
DARK = "#222222"


def configure_style() -> None:
    """Set a deterministic presentation-oriented matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 20,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "axes.edgecolor": LIGHT_GRAY,
            "axes.linewidth": 1.0,
            "axes.titleweight": "bold",
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


def load_condition_summary() -> pd.DataFrame:
    """Load the compact condition summary table from the paper outputs."""
    table = pd.read_csv(TABLE1_CSV, header=[0, 1], index_col=0)
    table.columns = [
        "_".join(str(part).strip() for part in col if str(part) != "nan")
        for col in table.columns
    ]
    table = table.reset_index().rename(columns={"label": "condition"})
    table = table.rename(
        columns={
            "participant_id_nunique": "n_subjects",
            "SBP_count": "n_readings",
            "SBP_median": "sbp_median",
            "DBP_median": "dbp_median",
            "HR_median": "hr_median",
        }
    )
    return table


def build_context_breakdown() -> None:
    """Create a deck-friendly horizontal bar chart for labeled contexts."""
    table = load_condition_summary()
    display_rows = [
        {"group": "Baseline", "n_readings": 1238, "n_subjects": 28},
        {"group": "Sleep", "n_readings": 407, "n_subjects": 28},
        {"group": "Task", "n_readings": 98, "n_subjects": 27},
        {"group": "Air Alert", "n_readings": 117, "n_subjects": 11},
        {"group": "Other excluded", "n_readings": 304, "n_subjects": 27},
    ]
    grouped = pd.DataFrame(display_rows)

    order = ["Baseline", "Sleep", "Task", "Air Alert", "Other excluded"]
    grouped["group"] = pd.Categorical(grouped["group"], categories=order, ordered=True)
    grouped = grouped.sort_values("group")
    total = int(table["n_readings"].sum())
    grouped["pct"] = 100 * grouped["n_readings"] / total
    grouped["label"] = grouped.apply(
        lambda row: f"{int(row['n_readings']):,} readings  ({row['pct']:.1f}%)", axis=1
    )

    colors = {
        "Baseline": NAVY,
        "Sleep": BLUE,
        "Task": RED,
        "Air Alert": "#6A4C93",
        "Other excluded": "#9AA3AF",
    }

    fig, ax = plt.subplots(figsize=(11, 5.2))
    y = range(len(grouped))
    bars = ax.barh(
        y,
        grouped["n_readings"],
        color=[colors[group] for group in grouped["group"]],
        height=0.58,
    )

    for idx, (bar, (_, row)) in enumerate(zip(bars, grouped.iterrows(), strict=False)):
        ax.text(
            bar.get_width() + total * 0.015,
            idx,
            row["label"],
            va="center",
            ha="left",
            fontsize=13,
            color=DARK,
            fontweight="bold",
        )
        ax.text(
            total * 0.01,
            idx + 0.25,
            f"n subjects = {int(row['n_subjects'])}",
            va="center",
            ha="left",
            fontsize=10.5,
            color=GRAY,
        )

    ax.set_yticks(list(y), grouped["group"])
    ax.invert_yaxis()
    ax.set_xlim(0, total * 1.23)
    ax.set_xlabel("Valid ABPM readings")
    ax.set_title("Dataset composition by labeled context", loc="left")
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.savefig(FIG_CONTEXT, dpi=220)
    plt.close(fig)


def build_cognitive_patterns() -> None:
    """Create a presentation version of the cognitive participant patterns."""
    metrics = pd.read_csv(METRICS_CSV)
    chart = metrics[metrics["DBP_Cognitive Task_N"].fillna(0) > 0].copy()
    chart["flagged"] = (
        (chart["DBP_Cognitive Task_Anomaly"] > 50)
        | (chart["DBP_Cognitive Task_DeltaBias"] > 2)
    )
    chart = chart.sort_values("DBP_Cognitive Task_Anomaly").reset_index(drop=True)
    chart["rank"] = chart.index + 1

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(12.5, 5.8),
        gridspec_kw={"wspace": 0.24},
    )

    not_flagged = chart[~chart["flagged"]]
    flagged = chart[chart["flagged"]]

    for ax, value_col, title, ylabel in [
        (
            ax_left,
            "DBP_Cognitive Task_Anomaly",
            "A. Cognitive MAE inflation",
            "Inflation vs baseline MAE (%)",
        ),
        (
            ax_right,
            "DBP_Cognitive Task_DeltaBias",
            "B. Cognitive signed residual bias",
            "Observed DBP - predicted DBP (mmHg)",
        ),
    ]:
        ax.scatter(
            not_flagged["rank"],
            not_flagged[value_col],
            s=72,
            facecolor=BG,
            edgecolor=GRAY,
            linewidth=1.4,
            label="Not flagged",
            zorder=3,
        )
        ax.scatter(
            flagged["rank"],
            flagged[value_col],
            s=72,
            facecolor=BLUE,
            edgecolor=BLUE,
            linewidth=0.8,
            label="Flagged",
            zorder=3,
        )
        ax.set_title(title, loc="left", fontsize=18)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Participant rank")
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 27])

    ax_left.axhline(50, color=RED, linewidth=1.6, linestyle="--", alpha=0.9)
    ax_left.text(
        27.3,
        54,
        "flag threshold = 50%",
        ha="right",
        va="bottom",
        fontsize=11,
        color=RED,
        fontweight="bold",
    )

    ax_right.axhline(2, color=RED, linewidth=1.6, linestyle="--", alpha=0.9)
    ax_right.axhline(0, color=GRAY, linewidth=1.1, linestyle="-", alpha=0.7)
    ax_right.text(
        27.3,
        2.6,
        "flag threshold = +2 mmHg",
        ha="right",
        va="bottom",
        fontsize=11,
        color=RED,
        fontweight="bold",
    )

    median_anomaly = chart["DBP_Cognitive Task_Anomaly"].median()
    median_bias = chart["DBP_Cognitive Task_DeltaBias"].median()
    ax_left.axhline(median_anomaly, color=NAVY, linewidth=1.4, linestyle=":")
    ax_right.axhline(median_bias, color=NAVY, linewidth=1.4, linestyle=":")

    ax_left.text(
        1,
        median_anomaly + 5,
        f"median = {median_anomaly:.1f}%",
        color=NAVY,
        fontsize=11,
        fontweight="bold",
    )
    ax_right.text(
        1,
        median_bias + 0.7,
        f"median = {median_bias:+.2f} mmHg",
        color=NAVY,
        fontsize=11,
        fontweight="bold",
    )

    handles, labels = ax_right.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=12,
    )
    fig.savefig(FIG_COGNITIVE, dpi=220)
    plt.close(fig)


def build_threshold_sensitivity() -> None:
    """Create a compact backup figure for threshold sensitivity."""
    thresholds = pd.read_csv(THRESHOLDS_CSV)

    fig, ax_left = plt.subplots(figsize=(9.8, 5.3))
    ax_right = ax_left.twinx()

    ax_left.plot(
        thresholds["Threshold_Pct"],
        thresholds["Pct_Pos"],
        marker="o",
        markersize=8,
        linewidth=2.4,
        color=BLUE,
        label="Flagged participants (%)",
    )
    ax_right.plot(
        thresholds["Threshold_Pct"],
        thresholds["Median_Bias_Pos"],
        marker="s",
        markersize=7,
        linewidth=2.2,
        color=RED,
        label="Median bias among flagged (mmHg)",
    )

    for _, row in thresholds.iterrows():
        ax_left.text(
            row["Threshold_Pct"],
            row["Pct_Pos"] + 2.2,
            f"{row['N_Pos']:.0f}/{int(row['N_Pos'] + row['N_Neg'])}",
            ha="center",
            va="bottom",
            color=BLUE,
            fontsize=10.5,
            fontweight="bold",
        )

    ax_left.set_title("Threshold sensitivity of descriptive flagging rule", loc="left")
    ax_left.set_xlabel("Anomaly threshold (%)")
    ax_left.set_ylabel("Flagged participants (%)", color=BLUE)
    ax_right.set_ylabel("Median signed bias in flagged group (mmHg)", color=RED)
    ax_left.set_xticks(thresholds["Threshold_Pct"])
    ax_left.set_ylim(0, 100)
    ax_right.set_ylim(0, max(thresholds["Median_Bias_Pos"]) + 2)

    ax_left.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8, alpha=0.7)
    ax_left.spines["top"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    left_handles, left_labels = ax_left.get_legend_handles_labels()
    right_handles, right_labels = ax_right.get_legend_handles_labels()
    fig.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=11,
    )

    fig.savefig(FIG_THRESHOLDS, dpi=220)
    plt.close(fig)


def main() -> None:
    """Generate all presentation-friendly figures."""
    configure_style()
    build_context_breakdown()
    build_cognitive_patterns()
    build_threshold_sensitivity()
    print(f"Saved {FIG_CONTEXT}")
    print(f"Saved {FIG_COGNITIVE}")
    print(f"Saved {FIG_THRESHOLDS}")


if __name__ == "__main__":
    main()
