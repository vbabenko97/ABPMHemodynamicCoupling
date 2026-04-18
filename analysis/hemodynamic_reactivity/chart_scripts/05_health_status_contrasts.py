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
grouped = agg.groupby("monitoring_diagnosis_is_healthy").agg(
    bp_load=("bp_load_%", "mean"),
    stroop2=("stroop_effect_battery_2", "mean"),
    physical=("physical_reactivity_mean", "mean"),
)
grouped.index = ["Non-healthy", "Healthy"]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
metrics = [
    ("bp_load", "BP load separates health status sharply", RED),
    ("stroop2", "Battery-2 Stroop effect also separates the groups", BLUE),
    ("physical", "Mean physical reactivity barely separates the groups", GREY),
]
for ax, (metric, title, color) in zip(axes, metrics):
    ax.bar(grouped.index, grouped[metric], color=[GREY, color])
    for i, value in enumerate(grouped[metric]):
        ax.text(i, value + (2 if metric != "physical" else 0.015), f"{value:.2f}", ha="center", color=DARK)
    ax.set_title(title)
    if metric == "physical":
        ax.set_ylim(0, 1.3)
    style_axes(ax)
fig.suptitle("Diagnostic status tracks baseline load and repeated Stroop performance more than task reactivity")
save(fig, "05_health_status_contrasts.png")
