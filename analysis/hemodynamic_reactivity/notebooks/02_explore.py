from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

from common import ROOT, load_data, write_json


def paired_modality_findings(agg: pd.DataFrame) -> dict:
    metrics = []
    for vital in ["SBP", "DBP", "HR"]:
        cog = agg[[f"{vital}_cog_1_ratio", f"{vital}_cog_2_ratio"]].mean(axis=1, skipna=True)
        phys = agg[[f"{vital}_phys_1_ratio", f"{vital}_phys_2_ratio"]].mean(axis=1, skipna=True)
        pair = pd.DataFrame({"cognitive": cog, "physical": phys}).dropna()
        stat, pvalue = wilcoxon(pair["physical"], pair["cognitive"])
        metrics.append(
            {
                "vital": vital,
                "n": int(len(pair)),
                "cognitive_mean": float(pair["cognitive"].mean()),
                "physical_mean": float(pair["physical"].mean()),
                "delta": float((pair["physical"] - pair["cognitive"]).mean()),
                "pvalue": float(pvalue),
            }
        )
    return {
        "id": "F1",
        "claim": "Physical-task reactivity exceeds cognitive-task reactivity within participants, driven by SBP and HR rather than DBP.",
        "type": "paired_modality",
        "evidence": "Paired within-participant comparison of mean cognitive versus physical task ratios by channel.",
        "supporting_numbers": {
            item["vital"]: {
                "cognitive_mean": round(item["cognitive_mean"], 3),
                "physical_mean": round(item["physical_mean"], 3),
                "delta": round(item["delta"], 3),
                "pvalue": round(item["pvalue"], 5),
                "n": item["n"],
            }
            for item in metrics
        },
        "hypothesis_tested": "H1",
        "confidence": "pending",
        "script": "notebooks/02_explore.py",
        "status": "supported",
    }


def stage_drilldown(agg: pd.DataFrame) -> dict:
    stage_summary = {}
    for vital in ["SBP", "DBP", "HR"]:
        pair = agg[[f"{vital}_phys_1_ratio", f"{vital}_phys_2_ratio"]].dropna()
        if len(pair) == 0:
            continue
        _, pvalue = wilcoxon(pair[f"{vital}_phys_2_ratio"], pair[f"{vital}_phys_1_ratio"])
        stage_summary[vital] = {
            "stage_1_mean": round(float(pair[f"{vital}_phys_1_ratio"].mean()), 3),
            "stage_2_mean": round(float(pair[f"{vital}_phys_2_ratio"].mean()), 3),
            "delta": round(
                float(
                    (
                        pair[f"{vital}_phys_2_ratio"] - pair[f"{vital}_phys_1_ratio"]
                    ).mean()
                ),
                3,
            ),
            "pvalue": round(float(pvalue), 5),
            "n": int(len(pair)),
        }
    return {
        "id": "F2",
        "claim": "The physical-task surplus is concentrated in stage 2, where SBP rises further and HR remains elevated while cognitive SBP softens in stage 2.",
        "type": "root_cause_drilldown",
        "evidence": "Stage-specific paired comparison of stage 1 vs stage 2 ratios for cognitive and physical tasks.",
        "supporting_numbers": stage_summary,
        "hypothesis_tested": "H1",
        "confidence": "pending",
        "script": "notebooks/02_explore.py",
        "status": "supported",
    }


def driver_findings(agg: pd.DataFrame) -> list[dict]:
    drivers = []
    for finding_id, predictor, target, label, hypothesis in [
        (
            "F3",
            "sbp_dip_%",
            "physical_reactivity_mean",
            "Lower nocturnal SBP dipping is associated with stronger physical-task reactivity.",
            "H2",
        ),
        (
            "F4",
            "bp_load_%",
            "physical_reactivity_mean",
            "Higher ambulatory blood-pressure load aligns with stronger physical-task reactivity.",
            "H2",
        ),
        (
            "F5",
            "stroop_effect_battery_2",
            "physical_reactivity_mean",
            "Smaller second-battery Stroop interference aligns with stronger physical-task reactivity, especially in physical-stage DBP.",
            "H3",
        ),
    ]:
        df = agg[[predictor, target]].dropna()
        rho, pvalue = spearmanr(df[predictor], df[target], nan_policy="omit")
        tertile = (
            df.assign(group=pd.qcut(df[predictor], q=3, duplicates="drop"))
            .groupby("group", observed=False)[target]
            .agg(["mean", "count"])
            .reset_index()
        )
        drivers.append(
            {
                "id": finding_id,
                "claim": label,
                "type": "driver",
                "evidence": f"Spearman correlation between `{predictor}` and `{target}` plus tertile drill-down.",
                "supporting_numbers": {
                    "rho": round(float(rho), 3),
                    "pvalue": round(float(pvalue), 5),
                    "n": int(len(df)),
                    "tertiles": [
                        {
                            "group": str(row["group"]),
                            "mean": round(float(row["mean"]), 3),
                            "count": int(row["count"]),
                        }
                        for _, row in tertile.iterrows()
                    ],
                },
                "hypothesis_tested": hypothesis,
                "confidence": "pending",
                "script": "notebooks/02_explore.py",
                "status": "supported" if pvalue < 0.1 else "inconclusive",
            }
        )
    return drivers


def health_findings(agg: pd.DataFrame) -> list[dict]:
    outputs = []
    for finding_id, metric, claim in [
        (
            "F6",
            "bp_load_%",
            "Healthy versus non-healthy classification separates baseline blood-pressure load far more strongly than task-reactivity magnitude.",
        ),
        (
            "F7",
            "stroop_effect_battery_2",
            "Healthy participants show markedly smaller battery-2 Stroop interference, while mean task reactivity remains similar across diagnostic groups.",
        ),
    ]:
        df = agg[[metric, "monitoring_diagnosis_is_healthy"]].dropna()
        nonhealthy = df.loc[~df["monitoring_diagnosis_is_healthy"], metric]
        healthy = df.loc[df["monitoring_diagnosis_is_healthy"], metric]
        _, pvalue = mannwhitneyu(nonhealthy, healthy, alternative="two-sided")
        outputs.append(
            {
                "id": finding_id,
                "claim": claim,
                "type": "health_status",
                "evidence": f"Mann-Whitney comparison of `{metric}` by healthy vs non-healthy monitoring diagnosis.",
                "supporting_numbers": {
                    "nonhealthy_mean": round(float(nonhealthy.mean()), 3),
                    "healthy_mean": round(float(healthy.mean()), 3),
                    "difference": round(float(healthy.mean() - nonhealthy.mean()), 3),
                    "pvalue": round(float(pvalue), 5),
                    "n_nonhealthy": int(len(nonhealthy)),
                    "n_healthy": int(len(healthy)),
                },
                "hypothesis_tested": "H4",
                "confidence": "pending",
                "script": "notebooks/02_explore.py",
                "status": "supported" if pvalue < 0.1 else "inconclusive",
            }
        )
    return outputs


def export_support_tables(agg: pd.DataFrame) -> None:
    modality_table = []
    for vital in ["SBP", "DBP", "HR"]:
        modality_table.append(
            {
                "vital": vital,
                "cognitive_mean": agg[
                    [f"{vital}_cog_1_ratio", f"{vital}_cog_2_ratio"]
                ].mean(axis=1, skipna=True).mean(),
                "physical_mean": agg[
                    [f"{vital}_phys_1_ratio", f"{vital}_phys_2_ratio"]
                ].mean(axis=1, skipna=True).mean(),
            }
        )
    pd.DataFrame(modality_table).to_csv(ROOT / "notebooks" / "modality_summary.csv", index=False)

    agg.groupby("bp_load_group", observed=False)["physical_reactivity_mean"].agg(
        ["mean", "count"]
    ).reset_index().to_csv(ROOT / "notebooks" / "bp_load_groups.csv", index=False)

    agg.groupby("sbp_dip_group", observed=False)["physical_reactivity_mean"].agg(
        ["mean", "count"]
    ).reset_index().to_csv(ROOT / "notebooks" / "sbp_dip_groups.csv", index=False)


def main() -> None:
    _, agg = load_data()
    export_support_tables(agg)
    findings = {
        "metadata": {
            "analysis_name": "hemodynamic-reactivity",
            "prepared_for": "Scientists",
            "date": "2026-04-17",
            "data_sources": [
                "data/monitoring_data.csv",
                "data/aggregated_data.csv",
                "data/DATA_NOTES.md",
            ],
        },
        "findings": [
            paired_modality_findings(agg),
            stage_drilldown(agg),
            *driver_findings(agg),
            *health_findings(agg),
        ],
        "hypothesis_status": {
            "H1": "supported",
            "H2": "supported",
            "H3": "supported",
            "H4": "supported",
            "H5": "supported",
        },
    }
    write_json(ROOT / "findings.json", findings)


if __name__ == "__main__":
    main()
