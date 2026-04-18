from __future__ import annotations

import json

from common import ROOT


def main() -> None:
    findings = json.loads((ROOT / "findings.json").read_text())
    headline = (
        "Task reactivity in this ABPM cohort is primarily a physical-stage SBP and HR phenomenon, with the clearest participant-level signals coming from lower nocturnal dipping, smaller battery-2 Stroop interference, and a directional increase with BP load."
    )
    findings["metadata"]["headline"] = headline
    findings["recommendations"] = [
        {
            "id": "R1",
            "action": "Prioritize physical-task SBP and HR endpoints, especially stage 2, as the primary physiological readouts in the next mechanistic study.",
            "owner": "Study investigators",
            "success_metric": "Replicate the physical-vs-cognitive SBP and HR gap in an expanded cohort with pre-registered thresholds.",
            "confidence": "High",
            "follow_up_date": "2026-06-15",
            "supports_findings": ["F1", "F2"],
        },
        {
            "id": "R2",
            "action": "Stratify future analyses by nocturnal SBP dipping and ABPM load before aggregating task responses.",
            "owner": "Biostatistics team",
            "success_metric": "Prospective models retain the same direction and improve explained variance over unstratified models.",
            "confidence": "Medium",
            "follow_up_date": "2026-06-15",
            "supports_findings": ["F3", "F4"],
        },
        {
            "id": "R3",
            "action": "Test whether battery-2 Stroop interference is a surrogate of preserved autonomic reserve rather than a marker of dysregulation.",
            "owner": "Cognitive physiology workstream",
            "success_metric": "A replication dataset confirms the inverse Stroop-effect to physical-reactivity relationship after multivariable adjustment.",
            "confidence": "Medium",
            "follow_up_date": "2026-07-01",
            "supports_findings": ["F5", "F7"],
        },
    ]
    findings["narrative"] = {
        "arc": "CTR",
        "beats": [
            {
                "id": "B1",
                "section": "Context",
                "title": "Physical tasks produce the main hemodynamic excursion in this cohort",
                "purpose": "Establish the dominant modality before looking for drivers.",
                "findings": ["F1"],
                "chart": "charts/01_physical_vs_cognitive.png",
            },
            {
                "id": "B2",
                "section": "Tension",
                "title": "The modality gap is a stage-2 SBP and HR effect, not a uniform rise across all channels",
                "purpose": "Show that the aggregate modality gap is not homogeneous.",
                "findings": ["F2"],
                "chart": "charts/02_stage_specific_reactivity.png",
            },
            {
                "id": "B3",
                "section": "Tension",
                "title": "Higher BP load and lower nocturnal dipping concentrate the strongest physical responders",
                "purpose": "Drill from overall modality into physiologic recovery and load strata.",
                "findings": ["F3", "F4"],
                "chart": "charts/03_reactivity_vs_load_and_dipping.png",
            },
            {
                "id": "B4",
                "section": "Resolution",
                "title": "Participants with smaller battery-2 Stroop interference show stronger physical-task reactivity",
                "purpose": "Connect the hemodynamic pattern to cognitive performance rather than treating it as isolated physiology.",
                "findings": ["F5"],
                "chart": "charts/04_stroop_vs_physical_reactivity.png",
            },
            {
                "id": "B5",
                "section": "Resolution",
                "title": "Diagnostic status separates baseline load and battery-2 performance more than task-reactivity magnitude",
                "purpose": "Keep the health-status secondary question from being overstated as a reactivity driver.",
                "findings": ["F6", "F7"],
                "chart": "charts/05_health_status_contrasts.png",
            },
            {
                "id": "B6",
                "section": "Resolution",
                "title": "Recommendations",
                "purpose": "Translate the validated findings into next-study priorities.",
                "findings": [],
                "chart": None,
            },
        ],
    }
    write_path = ROOT / "findings.json"
    write_path.write_text(json.dumps(findings, indent=2) + "\n")

    storyboard = """# Phase 4 Storyboard

## Headline

Task reactivity in this ABPM cohort is primarily a physical-stage SBP and HR phenomenon, with the clearest participant-level signals coming from lower nocturnal dipping, smaller battery-2 Stroop interference, and a directional increase with BP load.

## Context

| Beat | Purpose | Findings | Proposed visualization |
|---|---|---|---|
| B1 | Establish that task modality matters before searching for covariates. | F1 | Channel-level cognitive vs physical comparison |

## Tension

| Beat | Purpose | Findings | Proposed visualization |
|---|---|---|---|
| B2 | Show that the modality gap is concentrated in stage 2 and not uniform across channels. | F2 | Stage-specific paired comparison |
| B3 | Identify the subgroup physiology that concentrates stronger responders. | F3, F4 | Load and dipping strata plot |

## Resolution

| Beat | Purpose | Findings | Proposed visualization |
|---|---|---|---|
| B4 | Link the physiologic pattern to cognitive-performance preservation. | F5 | Stroop effect vs physical reactivity scatter |
| B5 | Separate baseline diagnostic differences from task-reactivity differences. | F6, F7 | Healthy vs non-healthy contrast bars |
| B6 | State the next-study actions. | Recommendations | Recommendation table |

## So What

The aggregate story is not “all stress tasks raise BP.” The code and data show a narrower and more interesting result: late physical-task responses, especially SBP and HR, carry the clearest hemodynamic signal, and the strongest responders cluster in participants with worse ambulatory recovery profiles, a directional increase in ambulatory load, and better repeated Stroop performance.

## Recommendations With Owners

| # | Recommendation | Owner | Success metric | Confidence | Follow-up |
|---|---|---|---|---|---|
| 1 | Prioritize physical-task stage-2 SBP and HR endpoints in the next study. | Study investigators | Replicate the modality gap in a larger cohort. | High | 2026-06-15 |
| 2 | Stratify or adjust by BP load and nocturnal dipping before reporting aggregate task reactivity. | Biostatistics team | Improved model fit without sign reversals. | Medium | 2026-06-15 |
| 3 | Test battery-2 Stroop interference as a marker of preserved autonomic reserve. | Cognitive physiology workstream | Replication after multivariable adjustment. | Medium | 2026-07-01 |

## Narrative Coherence Check

- Each beat advances from modality -> stage mechanism -> physiologic drivers -> cognitive linkage -> diagnostic nuance -> actions.
- The story lands on specific follow-up study design changes rather than generic “more research” language.
- No beat depends on an unvalidated number.
"""
    (ROOT / "storyboard.md").write_text(storyboard)


if __name__ == "__main__":
    main()
