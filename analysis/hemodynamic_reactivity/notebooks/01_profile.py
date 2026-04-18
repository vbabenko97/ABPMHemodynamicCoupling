from __future__ import annotations

from pathlib import Path

from common import DATA_DIR, ROOT, describe_frame, load_data, ordered_categorical_summary, write_json


def main() -> None:
    mon, agg = load_data()

    inventory = {
        "data_sources": [
            describe_frame(mon, "monitoring_data.csv"),
            describe_frame(agg, "aggregated_data.csv"),
        ],
        "monitoring_window": {
            "start": mon["datetime"].min().isoformat(),
            "end": mon["datetime"].max().isoformat(),
            "participants": int(mon["participant_id"].nunique()),
        },
        "participant_flags": {
            "healthy": ordered_categorical_summary(agg["monitoring_diagnosis_is_healthy"]),
            "female": ordered_categorical_summary(agg["is_female"]),
            "coffee": ordered_categorical_summary(agg["drinks_coffee"]),
        },
        "key_columns": {
            "reactivity": [c for c in agg.columns if c.endswith("_ratio")],
            "cognition": [
                "stroop_effect_battery_1",
                "stroop_effect_battery_2",
                "avg_reaction_time_battery_1",
                "avg_reaction_time_battery_2",
            ],
            "health": [
                "bp_load_%",
                "sbp_dip_%",
                "dbp_dip_%",
                "monitoring_diagnosis_is_healthy",
                "sleep_duration_h",
            ],
        },
    }
    write_json(ROOT / "data_inventory.json", inventory)

    framing = f"""# Phase 1 Framing

## Structured Analytical Question

Primary question: Which participant characteristics are associated with stronger hemodynamic reactivity during cognitive and physical stress tasks in this ABPM cohort?

Secondary question 1: Do cognitive performance measures track with ABPM patterns such as blood pressure load, dipping, sleep duration, and task-period hemodynamic shifts?

Secondary question 2: Which factors differentiate healthy versus non-healthy monitoring profiles, and do those differences persist once the cohort is segmented by task modality and hemodynamic channel?

## Decision Context

Audience: Scientists studying stress-linked hemodynamic regulation in ambulatory monitoring data.

Decision supported: Which physiological or behavioral covariates deserve priority in follow-up mechanistic studies and which reported associations are strong enough to emphasize in a scientist-facing manuscript or presentation.

## Success Criteria

- Answered: the analysis identifies the dominant modality and channel of reactivity, isolates at least one plausible subgroup-level driver, and quantifies whether the same pattern appears in the secondary cognitive-performance and health-status questions.
- Done: markdown report, Marp deck, Jupyter notebook, findings JSON, and PNG charts all point to the same validated headline and are reproducible from scripts in this workspace.
- Quantitative bar: every shipped quantitative claim is independently re-derived in Phase 3 and receives a High, Medium, or Low confidence grade.

## Hypotheses

### H1: Task modality dominates participant-level heterogeneity
If H1 is true, physical-task ratios will exceed cognitive-task ratios within the same participants, especially in SBP and HR channels.
Confirm with paired within-subject comparisons of cognitive vs physical ratios. Refute if modality differences are inconsistent across channels.

### H2: Lower nocturnal recovery is associated with stronger task reactivity
If H2 is true, participants with lower nocturnal SBP/DBP dipping and higher blood-pressure load will show larger physical-reactivity indices, especially in late-task measurements.
Confirm with monotonic correlations and subgroup contrasts by dip/load strata. Refute if associations disappear across strata.

### H3: Better repeated cognitive performance aligns with stronger preserved physical responsiveness
If H3 is true, smaller Stroop interference and faster reaction time on battery 2 will co-occur with larger physical-task reactivity, implying that preserved autonomic reserve rather than generalized dysregulation explains the pattern.
Confirm if the relationship is strongest in stage-2 physical ratios and remains directionally stable in validation. Refute if it is driven only by a single subgroup or metric artifact.

### H4: Diagnostic status changes baseline load and performance more than reactivity itself
If H4 is true, healthy vs non-healthy classification will strongly separate blood-pressure load and Stroop battery-2 performance, but not produce a uniform task-reactivity split.
Confirm with healthy/non-healthy comparisons plus subgroup checks. Refute if health status drives reactivity consistently across channels.

### H5: Missingness and stage imbalance are material data constraints
If H5 is true, second cognitive-task ratios and some stage durations will have enough missingness to limit precision and should be treated as directional evidence rather than definitive estimates.
Confirm from the data profile and carry the caveat into validation. Refute if missingness is negligible.

## Data Inventory

- `data/monitoring_data.csv`: 2,164 time-stamped ABPM rows across 28 participants with sleep-state flags, task-period flags, and SBP/DBP/HR measures.
- `data/aggregated_data.csv`: 28 participant-level rows with demographics, sleep metrics, dipping indices, Stroop outcomes, diagnosis flags, and 12 task-reactivity ratios.
- `data/DATA_NOTES.md`: one documented column rename for `avg_reaction_time_battery_2`.

## Gate Check

- Structured question written: yes
- Hypotheses across multiple cause categories written: yes
- Success criteria explicit: yes
- Data inventory complete: yes
"""
    (ROOT / "framing.md").write_text(framing)

    profile = f"""# Phase 2.1 Data Profile

## File-Level Summary

- `monitoring_data.csv`: {len(mon)} rows, {mon.shape[1]} columns, {mon['participant_id'].nunique()} participants, window {mon['datetime'].min():%Y-%m-%d} to {mon['datetime'].max():%Y-%m-%d}.
- `aggregated_data.csv`: {len(agg)} rows, {agg.shape[1]} columns, one row per participant.

## Quality Notes

- `aggregated_data.csv` has no duplicate participant rows.
- Reactivity ratios are the sparsest fields; missing values cluster in `*_cog_2_ratio` and early physical-stage ratios, so stage-2 cognitive estimates are less stable than physical estimates.
- The documented typo in `avg_reaction_time_battey_2` was corrected on load before any analysis.
- The cohort is small (`n=28`), so all inferential results should be interpreted as effect-size signals, not definitive population estimates.

## Immediate Analysis Constraints

- Healthy/non-healthy comparison is unbalanced (18 healthy, 10 non-healthy).
- Cognitive stage-2 ratios have 20 observed participants versus 25 for physical stage-2 ratios.
- Time-series monitoring data supports chronology checks, but the primary association analysis is participant-level because the covariates of interest live in the aggregated file.
"""
    (ROOT / "profile.md").write_text(profile)


if __name__ == "__main__":
    main()
