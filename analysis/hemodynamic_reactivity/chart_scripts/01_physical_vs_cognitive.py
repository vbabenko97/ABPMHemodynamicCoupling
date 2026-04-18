from __future__ import annotations

import pandas as pd

from plot_common import BLUE, DARK, GREY, RED, save, style_axes


agg = pd.read_csv("analysis/hemodynamic_reactivity/data/aggregated_data.csv").rename(
    columns={"avg_reaction_time_battey_2": "avg_reaction_time_battery_2"}
)
rows = []
for vital in ["SBP", "DBP", "HR"]:
    rows.append(
        {
            "vital": vital,
            "cognitive": agg[[f"{vital}_cog_1_ratio", f"{vital}_cog_2_ratio"]]
            .mean(axis=1, skipna=True)
            .mean(),
            "physical": agg[[f"{vital}_phys_1_ratio", f"{vital}_phys_2_ratio"]]
            .mean(axis=1, skipna=True)
            .mean(),
        }
    )
plot = pd.DataFrame(rows)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.2, 4.3))
x = range(len(plot))
width = 0.35
ax.bar([i - width / 2 for i in x], plot["cognitive"], width=width, color=GREY)
ax.bar([i + width / 2 for i in x], plot["physical"], width=width, color=[BLUE, GREY, RED])
for idx, row in plot.iterrows():
    ax.text(idx - width / 2, row["cognitive"] + 0.01, f"{row['cognitive']:.2f}", ha="center", color="#475569")
    ax.text(idx + width / 2, row["physical"] + 0.01, f"{row['physical']:.2f}", ha="center", color=DARK)
ax.set_xticks(list(x), plot["vital"])
ax.set_ylim(0, 1.35)
ax.set_ylabel("Mean task ratio")
ax.set_title("Physical tasks lift SBP and HR more than cognitive tasks, while DBP stays flat")
ax.annotate("Largest modality gaps", xy=(2.35, plot.loc[2, "physical"]), xytext=(1.8, 1.28), arrowprops={"arrowstyle": "->", "color": DARK}, color=DARK)
style_axes(ax)
save(fig, "01_physical_vs_cognitive.png")
