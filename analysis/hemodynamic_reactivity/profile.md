# Phase 2.1 Data Profile

## File-Level Summary

- `monitoring_data.csv`: 2164 rows, 22 columns, 28 participants, window 2023-11-03 to 2024-02-21.
- `aggregated_data.csv`: 28 rows, 116 columns, one row per participant.

## Quality Notes

- `aggregated_data.csv` has no duplicate participant rows.
- Reactivity ratios are the sparsest fields; missing values cluster in `*_cog_2_ratio` and early physical-stage ratios, so stage-2 cognitive estimates are less stable than physical estimates.
- The documented typo in `avg_reaction_time_battey_2` was corrected on load before any analysis.
- The cohort is small (`n=28`), so all inferential results should be interpreted as effect-size signals, not definitive population estimates.

## Immediate Analysis Constraints

- Healthy/non-healthy comparison is unbalanced (18 healthy, 10 non-healthy).
- Cognitive stage-2 ratios have 20 observed participants versus 25 for physical stage-2 ratios.
- Time-series monitoring data supports chronology checks, but the primary association analysis is participant-level because the covariates of interest live in the aggregated file.
