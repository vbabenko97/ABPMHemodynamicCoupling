from __future__ import annotations

import pandas as pd

from plot_common import BLUE, DARK, GREY, RED, save, style_axes


agg = pd.read_csv("analysis/hemodynamic_reactivity/data/aggregated_data.csv").rename(
    columns={"avg_reaction_time_battey_2": "avg_reaction_time_battery_2"}
)
agg["physical_reactivity_mean"] = agg[
    [
        "SBP_phys_1_ratio",
        "DBP_phys_1_ratio",
        "HR_phys_1_ratio",
        "SBP_phys_2_ratio",
        "DBP_phys_2_ratio",
        "HR_phys_2_ratio",
    ]
].mean(axis=1, skipna=True)
load = (
    agg.assign(group=pd.qcut(agg["bp_load_%"], 3, labels=["Low", "Mid", "High"]))
    .groupby("group", observed=False)["physical_reactivity_mean"]
    .mean()
    .reset_index()
)
dipping = (
    agg.assign(group=pd.qcut(agg["sbp_dip_%"], 3, labels=["Low dip", "Mid dip", "High dip"]))
    .groupby("group", observed=False)["physical_reactivity_mean"]
    .mean()
    .reset_index()
)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
axes[0].bar(load["group"], load["physical_reactivity_mean"], color=[GREY, GREY, RED])
axes[0].set_title("Higher BP load tiers carry the strongest physical reactivity")
axes[0].set_ylim(0, 1.3)
axes[0].set_ylabel("Physical reactivity mean")
axes[0].annotate("High-load tertile", xy=(2, load.iloc[2, 1]), xytext=(1.25, 1.22), arrowprops={"arrowstyle": "->", "color": DARK}, color=DARK)
style_axes(axes[0])

axes[1].bar(dipping["group"], dipping["physical_reactivity_mean"], color=[BLUE, GREY, GREY])
axes[1].set_title("Lower nocturnal SBP dipping concentrates stronger responders")
axes[1].set_ylim(0, 1.3)
axes[1].annotate("Low-dip tertile", xy=(0, dipping.iloc[0, 1]), xytext=(0.45, 1.22), arrowprops={"arrowstyle": "->", "color": DARK}, color=DARK)
style_axes(axes[1])

fig.suptitle("Load and nocturnal recovery stratification explain where the physical-task signal concentrates")
save(fig, "03_reactivity_vs_load_and_dipping.png")
