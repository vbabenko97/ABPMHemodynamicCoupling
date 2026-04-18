# Phase 3 Validation

## F1 Physical-task reactivity exceeds cognitive-task reactivity within participants, driven by SBP and HR rather than DBP.

- Re-derivation: Matched using subgroup-level physical-minus-cognitive gaps by diagnostic status.
- Simpson's Paradox check: No reversal: SBP and HR gaps stay positive in healthy and non-healthy subgroups; DBP stays approximately flat.
- Confidence: High
- Bias checks:
  - selection: small single-cohort sample limits generalization
  - composition shift: checked by healthy-status subgroups, no sign reversal
  - metric drift: same ratio definitions across participants

## F2 The physical-task surplus is concentrated in stage 2, where SBP rises further and HR remains elevated while cognitive SBP softens in stage 2.

- Re-derivation: Confirmed from independent stage-specific contrasts and a separate battery-2 DBP correlation.
- Simpson's Paradox check: No reversal after splitting by health status; stage-2 physical SBP remains above stage-1 in both strata.
- Confidence: Medium
- Bias checks:
  - selection: stage-2 measurements missing for some participants
  - composition shift: similar direction in healthy and non-healthy strata
  - base rate: ratios normalize stage comparisons within participant

## F3 Lower nocturnal SBP dipping is associated with stronger physical-task reactivity.

- Re-derivation: Matched using an independent subgroup-by-diagnosis re-derivation rather than the tertile summary.
- Simpson's Paradox check: Direction remains stable by diagnostic stratum; no aggregate reversal detected.
- Confidence: Medium
- Bias checks:
  - selection: small n inflates uncertainty
  - composition shift: checked by diagnostic strata
  - lookahead: all predictors are contemporaneous participant-level features

## F4 Higher ambulatory blood-pressure load aligns with stronger physical-task reactivity.

- Re-derivation: Matched using an independent subgroup-by-diagnosis re-derivation rather than the tertile summary.
- Simpson's Paradox check: Direction remains stable by diagnostic stratum; no aggregate reversal detected.
- Confidence: Medium
- Bias checks:
  - selection: small n inflates uncertainty
  - composition shift: checked by diagnostic strata
  - lookahead: all predictors are contemporaneous participant-level features

## F5 Smaller second-battery Stroop interference aligns with stronger physical-task reactivity, especially in physical-stage DBP.

- Re-derivation: Matched using an independent subgroup-by-diagnosis re-derivation rather than the tertile summary.
- Simpson's Paradox check: Direction remains stable by diagnostic stratum; no aggregate reversal detected.
- Confidence: Medium
- Bias checks:
  - selection: small n inflates uncertainty
  - composition shift: checked by diagnostic strata
  - lookahead: all predictors are contemporaneous participant-level features

## F6 Healthy versus non-healthy classification separates baseline blood-pressure load far more strongly than task-reactivity magnitude.

- Re-derivation: Confirmed with an independent non-parametric group contrast and a same-strata check on physical reactivity.
- Simpson's Paradox check: No paradox because the primary comparison is already stratified by diagnosis.
- Confidence: High
- Bias checks:
  - selection: imbalance between healthy and non-healthy groups
  - base rate: effect expressed against group means and p-value
  - composition shift: task reactivity remains similar despite baseline-load separation

## F7 Healthy participants show markedly smaller battery-2 Stroop interference, while mean task reactivity remains similar across diagnostic groups.

- Re-derivation: Confirmed with an independent non-parametric group contrast and a same-strata check on physical reactivity.
- Simpson's Paradox check: No paradox because the primary comparison is already stratified by diagnosis.
- Confidence: High
- Bias checks:
  - selection: imbalance between healthy and non-healthy groups
  - base rate: effect expressed against group means and p-value
  - composition shift: task reactivity remains similar despite baseline-load separation

## Validation Summary

- Every shipped finding was re-derived with a different grouping or subgroup path than the exploration step.
- No aggregate finding flipped direction under the required Simpson's Paradox checks, but the sample is small enough that subgroup stability should be read as directional rather than definitive.
- Findings tied to stage-2 cognitive or stage-specific task ratios remain lower confidence because those fields are the sparsest in the dataset.
