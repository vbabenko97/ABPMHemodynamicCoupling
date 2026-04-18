from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from plot_common import BLUE, DARK, GREY, save, style_axes


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
df = agg[["stroop_effect_battery_2", "physical_reactivity_mean"]].dropna()
rho, _ = spearmanr(df["stroop_effect_battery_2"], df["physical_reactivity_mean"])
trend = df.sort_values("stroop_effect_battery_2")
coef = pd.Series(index=["intercept", "slope"], dtype=float)
coef["slope"], coef["intercept"] = __import__("numpy").polyfit(
    trend["stroop_effect_battery_2"], trend["physical_reactivity_mean"], 1
)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.scatter(df["stroop_effect_battery_2"], df["physical_reactivity_mean"], color=GREY, edgecolor="white", s=55)
ax.plot(
    trend["stroop_effect_battery_2"],
    coef["intercept"] + coef["slope"] * trend["stroop_effect_battery_2"],
    color=BLUE,
    linewidth=2.5,
)
ax.set_xlabel("Battery-2 Stroop effect")
ax.set_ylabel("Physical reactivity mean")
ax.set_title("Lower battery-2 Stroop interference aligns with stronger physical-task reactivity")
ax.annotate(f"Spearman rho = {rho:.2f}", xy=(0.66, 0.88), xycoords="axes fraction", color=DARK)
style_axes(ax)
save(fig, "04_stroop_vs_physical_reactivity.png")
