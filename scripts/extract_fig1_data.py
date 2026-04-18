#!/usr/bin/env python3
"""
Extract Figure 1 data from per_subject_metrics.csv.

Produces a clean 4-column CSV suitable for fig1_ieee.py.

The ``flagged`` column is a rule-based visual marker derived from the
plotted metrics (A_{i,cog} > 50 % OR DeltaBias_{i,cog} > +2.0 mmHg).
It is used solely for visual stratification in the figure and does NOT
represent an independent patient grouping.
"""

from pathlib import Path

import pandas as pd

ANOMALY_THRESH = 50.0   # %
BIAS_THRESH    = 2.0    # mmHg

src = Path(__file__).resolve().parent.parent / "results" / "per_subject_metrics.csv"
dst = Path(__file__).resolve().parent.parent / "results" / "fig1_data.csv"

raw = pd.read_csv(src)

# Keep only participants with valid cognitive-task data
valid = raw[raw["DBP_Cognitive Task_N"] > 0].copy()

out = pd.DataFrame({
    "participant_id":    valid["participant_id"].values,
    "flagged":           valid.apply(
        lambda r: "yes"
        if (r["DBP_Cognitive Task_Anomaly"] > ANOMALY_THRESH
            or r["DBP_Cognitive Task_DeltaBias"] > BIAS_THRESH)
        else "no",
        axis=1
    ).values,
    "mae_inflation_pct": valid["DBP_Cognitive Task_Anomaly"].values,
    "signed_bias_mmhg":  valid["DBP_Cognitive Task_DeltaBias"].values,
})

out.to_csv(dst, index=False)
print(f"Wrote {len(out)} rows -> {dst}")
