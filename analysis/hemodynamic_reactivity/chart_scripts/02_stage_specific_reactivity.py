from __future__ import annotations

import pandas as pd

from plot_common import BLUE, DARK, GREY, RED, save, style_axes


agg = pd.read_csv("analysis/hemodynamic_reactivity/data/aggregated_data.csv")
records = []
for vital in ["SBP", "DBP", "HR"]:
    for modality, color in [("cog", GREY), ("phys", BLUE if vital != "HR" else RED)]:
        for stage in [1, 2]:
            column = f"{vital}_{modality}_{stage}_ratio"
            records.append(
                {
                    "vital": vital,
                    "label": f"{modality.upper()}-{stage}",
                    "value": agg[column].mean(),
                    "color": color,
                }
            )
plot = pd.DataFrame(records)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=False)
for ax, vital in zip(axes, ["SBP", "DBP", "HR"]):
    subset = plot[plot["vital"] == vital]
    ax.bar(subset["label"], subset["value"], color=subset["color"])
    for _, row in subset.iterrows():
        ax.text(row["label"], row["value"] + 0.01, f"{row['value']:.2f}", ha="center", color=DARK)
    ax.set_title(vital)
    ax.set_ylim(0.75, 1.35 if vital != "HR" else 1.45)
    style_axes(ax)
axes[0].set_ylabel("Mean stage ratio")
fig.suptitle("The modality gap sharpens in stage 2: physical SBP rises further while cognitive SBP softens")
sbp_subset = plot[plot["vital"] == "SBP"].reset_index(drop=True)
axes[0].annotate(
    "Stage-2 physical SBP climbs",
    xy=(3, sbp_subset.loc[3, "value"]),
    xytext=(1.2, 1.28),
    arrowprops={"arrowstyle": "->", "color": DARK},
    color=DARK,
)
save(fig, "02_stage_specific_reactivity.png")
