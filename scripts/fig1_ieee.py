#!/usr/bin/env python3
"""
fig1_ieee.py – IEEE camera-ready Figure 1
==========================================

Two-panel dot plot showing per-participant hemodynamic coupling
metrics during cognitive stress:
  (A) MAE inflation  A_{i,cog} (%)
  (B) Signed residual bias  ΔBias_{i,cog} (mmHg)

Usage
-----
    python fig1_ieee.py results/fig1_data.csv          # default output
    python fig1_ieee.py results/fig1_data.csv --outdir figs/

Required columns in the input CSV (one row per participant):
    participant_id    – int or string  (x-axis ordering)
    flagged           – "yes" | "no"  (rule-based threshold flag; see below)
    mae_inflation_pct – float  (A_{i,cog}, %)
    signed_bias_mmhg  – float  (ΔBias_{i,cog}, mmHg)

The ``flagged`` column is a rule-based visual marker derived from the
plotted metrics (A_{i,cog} > 50% OR ΔBias_{i,cog} > +2.0 mmHg) and is
used solely for visual stratification.  It does NOT represent an
independent patient grouping or external screening outcome.

Outputs
-------
    fig1.pdf   – vector (primary for LaTeX)
    fig1.png   – 600 dpi raster fallback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ── Required columns ─────────────────────────────────────────────────
REQUIRED = {
    "participant_id":    "Participant identifier (int/string)",
    "flagged":           'Threshold flag: "yes" or "no"',
    "mae_inflation_pct": "MAE inflation A_{i,cog} (%)",
    "signed_bias_mmhg":  "Signed residual bias ΔBias_{i,cog} (mmHg)",
}


# ── Deterministic styling ────────────────────────────────────────────
def configure_mpl() -> None:
    """Set global matplotlib parameters for IEEE figures."""
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":   "cm",
        "font.size":          8,
        "axes.titlesize":     10,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    7.5,
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "lines.linewidth":    0.8,
        "figure.dpi":         150,
        "savefig.dpi":        600,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,   # TrueType in PDF (IEEE requirement)
        "ps.fonttype":        42,
    })


# ── Marker specs (grayscale-safe) ────────────────────────────────────
# Not flagged: open circle; Flagged: ×
NEG_MARKER  = dict(marker="o", s=32, facecolors="none",
                   edgecolors="0.25", linewidths=0.7, zorder=3)
POS_MARKER  = dict(marker="x", s=28, color="0.10",
                   linewidths=0.9, zorder=3)

# Median line style
MED_LINE = dict(linewidth=0.9, linestyle="--", color="0.45", zorder=2)

# Zero reference
ZERO_LINE = dict(linewidth=0.6, linestyle="-", color="0.65", zorder=1)


# ── Core plotting ────────────────────────────────────────────────────
def make_figure(df: pd.DataFrame, outdir: Path) -> None:
    """Build the two-panel IEEE Figure 1 and save PDF + PNG."""

    # ---- sort participants by MAE inflation (ascending) ----
    df = df.sort_values("mae_inflation_pct").reset_index(drop=True)
    n = len(df)
    x = np.arange(n)

    neg = df["flagged"] == "no"
    pos = df["flagged"] == "yes"

    # ---- figure setup ----
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(7.0, 2.8),
        gridspec_kw={"bottom": 0.22, "top": 0.90, "left": 0.08,
                     "right": 0.98, "wspace": 0.32},
    )

    # ================================================================
    #  Panel A – MAE inflation
    # ================================================================
    ax_a.scatter(x[neg], df.loc[neg, "mae_inflation_pct"], **NEG_MARKER)
    ax_a.scatter(x[pos], df.loc[pos, "mae_inflation_pct"], **POS_MARKER)

    med_a = float(df["mae_inflation_pct"].median())
    ax_a.axhline(med_a, **MED_LINE)
    # Direct annotation for median (placed at left edge, above line)
    ax_a.annotate(
        f"median = {med_a:.1f}%",
        xy=(0, med_a), xytext=(2, 3),
        textcoords="offset points", fontsize=6.5,
        color="0.40", ha="left", va="bottom",
    )

    ax_a.set_ylabel(r"MAE inflation, $A_{i,\mathrm{cog}}$ (%)")
    ax_a.set_title(r"$\mathbf{A}$  MAE inflation", loc="left", pad=4)

    # ================================================================
    #  Panel B – Signed residual bias
    # ================================================================
    ax_b.scatter(x[neg], df.loc[neg, "signed_bias_mmhg"], **NEG_MARKER)
    ax_b.scatter(x[pos], df.loc[pos, "signed_bias_mmhg"], **POS_MARKER)

    # Zero reference
    ax_b.axhline(0, **ZERO_LINE)

    med_b = float(df["signed_bias_mmhg"].median())
    ax_b.axhline(med_b, **MED_LINE)
    ax_b.annotate(
        f"median = {med_b:+.1f} mmHg",
        xy=(n - 1, med_b), xytext=(-2, 3),
        textcoords="offset points", fontsize=6.5,
        color="0.40", ha="right", va="bottom",
    )

    ax_b.set_ylabel(r"Signed residual bias, $\Delta\mathrm{Bias}_{i,\mathrm{cog}}$ (mmHg)")
    ax_b.set_title(r"$\mathbf{B}$  Signed residual bias", loc="left", pad=4)

    # ================================================================
    #  Shared x-axis styling
    # ================================================================
    for ax in (ax_a, ax_b):
        ax.set_xlim(-1, n)
        ax.set_xlabel("Participant index (sorted by $A_{i,\\mathrm{cog}}$)",
                       fontsize=7.5)
        # Show ticks at every 5th participant index
        tick_pos = np.arange(0, n, 5)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_pos + 1)  # 1-based labels
        ax.tick_params(axis="x", length=2, pad=2)
        # Light horizontal grid only
        ax.grid(axis="y", linewidth=0.3, color="0.85", zorder=0)
        ax.set_axisbelow(True)

    # ================================================================
    #  Shared legend – outside, below panels
    # ================================================================
    n_neg = int(neg.sum())
    n_pos = int(pos.sum())
    legend_handles = [
        Line2D([], [], marker="o", color="0.25", markerfacecolor="none",
               markersize=5, linewidth=0,
               label=f"Not flagged (n = {n_neg})"),
        Line2D([], [], marker="x", color="0.10",
               markersize=5, linewidth=0, markeredgewidth=0.9,
               label=f"Flagged (n = {n_pos})"),
        Line2D([], [], linestyle="--", color="0.45", linewidth=0.9,
               label="Cohort median"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.5,
        handletextpad=0.4,
    )

    # ---- save ----
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = outdir / f"fig1.{ext}"
        fig.savefig(path)
        print(f"  Saved {path}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate IEEE-quality Figure 1 (hemodynamic coupling dot plots)."
    )
    parser.add_argument(
        "data", type=Path,
        help="Path to CSV with columns: " + ", ".join(REQUIRED),
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("results"),
        help="Output directory for fig1.pdf / fig1.png  [default: results/]",
    )
    args = parser.parse_args()

    # ---- load & validate ----
    if not args.data.exists():
        sys.exit(f"ERROR: file not found: {args.data}")

    df = pd.read_csv(args.data)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        sys.exit(
            f"ERROR: missing required columns: {missing}\n"
            f"Required schema:\n"
            + "\n".join(f"  {k:25s} – {v}" for k, v in REQUIRED.items())
        )

    valid_flags = {"yes", "no"}
    bad = set(df["flagged"].unique()) - valid_flags
    if bad:
        sys.exit(f"ERROR: unexpected 'flagged' values: {bad}  (expected {valid_flags})")

    print(f"Loaded {len(df)} participants from {args.data}")
    configure_mpl()
    make_figure(df, args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()
