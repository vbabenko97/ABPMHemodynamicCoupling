from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

from common import ROOT


def load_findings() -> dict:
    return json.loads((ROOT / "findings.json").read_text())


def build_report(findings: dict) -> str:
    headline = findings["metadata"]["headline"]
    recommendations = findings["recommendations"]
    return f"""# {headline}

*Prepared for Scientists • 2026-04-17*

## Executive Summary

The primary question was which participant characteristics align with stronger hemodynamic reactivity during cognitive and physical stress tasks. The validated answer is narrower than a generic stress-response story: in this cohort, the dominant signal is a physical-task surplus in SBP and HR, concentrated in stage 2 rather than spread evenly across all channels.

The clearest participant-level signals around that physical-task pattern are lower nocturnal SBP dipping and smaller battery-2 Stroop interference, with BP load moving in the same direction but at lower certainty. The healthy versus non-healthy split is important for baseline load and repeated cognitive performance, but it does not create a comparably strong separation in mean task reactivity itself.

## Context

Physical-task ratios exceed cognitive-task ratios within participants, especially in SBP and HR. DBP remains close to flat across modalities, which matters because it means the aggregate modality effect is not a generic “everything rises” response.

![Physical tasks lift SBP and HR more than cognitive tasks, while DBP stays flat](charts/01_physical_vs_cognitive.png)

## What's Happening

The modality gap sharpens in stage 2. Physical SBP rises further from stage 1 to stage 2, while cognitive SBP softens, so the late-stage physical segment carries the clearest contrast in the dataset.

![The modality gap sharpens in stage 2: physical SBP rises further while cognitive SBP softens](charts/02_stage_specific_reactivity.png)

Stratifying the cohort shows where the strongest physical responders cluster. Higher BP load tiers have the largest mean physical reactivity, while stronger nocturnal SBP dipping corresponds to weaker physical reactivity. The code does not show a subgroup reversal under diagnostic-status stratification, so the aggregate direction is stable.

![Load and nocturnal recovery stratification explain where the physical-task signal concentrates](charts/03_reactivity_vs_load_and_dipping.png)

## Root Cause And Recommendations

The secondary cognitive-performance question points to an unexpected but coherent interpretation: participants with smaller battery-2 Stroop interference also show stronger physical-task reactivity. In this sample, that looks more like preserved autonomic reserve than generalized dysregulation, but the sample is small enough that the result stays medium confidence.

![Lower battery-2 Stroop interference aligns with stronger physical-task reactivity](charts/04_stroop_vs_physical_reactivity.png)

Health status sharpens the baseline phenotype instead of the acute reactivity phenotype. Non-healthy participants carry much higher BP load and worse battery-2 Stroop interference, yet the mean physical-task reactivity difference remains small.

![Diagnostic status tracks baseline load and repeated Stroop performance more than task reactivity](charts/05_health_status_contrasts.png)

### Recommendations

| # | Recommendation | Owner | Success metric | Confidence | Follow-up |
|---|---|---|---|---|---|
| 1 | {recommendations[0]['action']} | {recommendations[0]['owner']} | {recommendations[0]['success_metric']} | {recommendations[0]['confidence']} | {recommendations[0]['follow_up_date']} |
| 2 | {recommendations[1]['action']} | {recommendations[1]['owner']} | {recommendations[1]['success_metric']} | {recommendations[1]['confidence']} | {recommendations[1]['follow_up_date']} |
| 3 | {recommendations[2]['action']} | {recommendations[2]['owner']} | {recommendations[2]['success_metric']} | {recommendations[2]['confidence']} | {recommendations[2]['follow_up_date']} |

## Appendix

### Methodology

- Phase 1 framing: `framing.md`
- Exploration and reproducible tables: `notebooks/02_explore.py`
- Validation and confidence grading: `validation.md`
- Chart generation scripts: `chart_scripts/`

### Validation Summary

- Every headline claim was independently re-derived in `notebooks/03_validate.py`.
- Simpson's Paradox checks were run on all aggregate claims by diagnostic-status subgroup.
- Confidence grades vary by result: the modality headline is High, while the load, dipping, and Stroop associations remain Medium-confidence directional signals.

### Caveats

- The cohort is small (`n=28`), so the findings are best used to prioritize follow-up studies rather than to make definitive population claims.
- Cognitive stage-2 ratios are the sparsest measures in the dataset and should be read directionally.
- The analysis is observational and association-based; it does not identify causal mechanisms.
"""


def build_deck(findings: dict) -> str:
    headline = findings["metadata"]["headline"]
    beats = findings["narrative"]["beats"]
    slides = [
        "---",
        "marp: true",
        "theme: default",
        "paginate: true",
        "---",
        "",
        f"# {headline}",
        "Prepared for Scientists • 2026-04-17",
        "",
    ]
    notes = {
        "B1": ["Physical-task SBP and HR are the clearest cohort-level signal.", "DBP does not carry the same modality gap."],
        "B2": ["Stage 2 is where the physical-task divergence sharpens.", "This narrows the mechanistic target to late-task regulation."],
        "B3": ["Higher BP load and lower dipping cluster among stronger responders.", "Aggregate direction survives subgroup checks."],
        "B4": ["Lower battery-2 Stroop interference aligns with stronger physical reactivity.", "Interpret as a medium-confidence reserve signal, not proof of causality."],
        "B5": ["Diagnostic status separates baseline load and Stroop performance.", "It does not cleanly separate acute task-reactivity magnitude."],
    }
    for beat in beats:
        slides.extend(["---", "", f"## {beat['title']}"])
        if beat["chart"]:
            slides.append(f"![bg right:55%]({beat['chart']})")
        if beat["id"] in notes:
            slides.extend([f"- {line}" for line in notes[beat["id"]]])
        else:
            for rec in findings["recommendations"]:
                slides.append(f"- {rec['action']}")
        slides.append("")
    return "\n".join(slides) + "\n"


def build_notebook(findings: dict) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {findings['metadata']['headline']}"))
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "Prepared for Scientists on 2026-04-17.\n\nThis notebook mirrors the markdown report and keeps the code required to reproduce the key figures inline."
        )
    )
    nb.cells.append(
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "%matplotlib inline\n"
            "root = Path('analysis/hemodynamic_reactivity')\n"
            "agg = pd.read_csv(root / 'data' / 'aggregated_data.csv').rename(columns={'avg_reaction_time_battey_2':'avg_reaction_time_battery_2'})\n"
        )
    )
    beat_text = {
        "B1": "Physical tasks lift SBP and HR more than cognitive tasks, making modality the first-order framing signal.",
        "B2": "The modality gap is a stage-2 effect rather than a uniform elevation across all stages and channels.",
        "B3": "BP load and nocturnal dipping show where the strongest physical responders cluster.",
        "B4": "Battery-2 Stroop interference is inversely related to physical-task reactivity.",
        "B5": "Health status separates baseline phenotype more than task-reactivity magnitude.",
    }
    chart_paths = [
        "charts/01_physical_vs_cognitive.png",
        "charts/02_stage_specific_reactivity.png",
        "charts/03_reactivity_vs_load_and_dipping.png",
        "charts/04_stroop_vs_physical_reactivity.png",
        "charts/05_health_status_contrasts.png",
    ]
    for beat, chart in zip(findings["narrative"]["beats"][:5], chart_paths):
        nb.cells.append(nbf.v4.new_markdown_cell(f"## {beat['title']}\n\n{beat_text[beat['id']]}"))
        nb.cells.append(
            nbf.v4.new_code_cell(
                "from IPython.display import Image, display\n"
                f"display(Image(filename=str(root / '{chart}')))\n"
            )
        )
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## Recommendations\n\n"
            + "\n".join(
                [
                    f"- **{rec['id']}** {rec['action']} ({rec['confidence']} confidence; follow-up {rec['follow_up_date']})"
                    for rec in findings["recommendations"]
                ]
            )
        )
    )
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## How To Reproduce\n\nRun `python analysis/hemodynamic_reactivity/notebooks/01_profile.py` through `06_deliver.py` in order from the repository root."
        )
    )
    return nb


def main() -> None:
    findings = load_findings()
    (ROOT / "report.md").write_text(build_report(findings))
    (ROOT / "deck.md").write_text(build_deck(findings))
    nb = build_notebook(findings)
    nbf.write(nb, ROOT / "analysis.ipynb")
    (ROOT / "methodology.md").write_text(
        "# Methodology\n\nThe analysis follows the six-phase ai-analyst pipeline with reproducible scripts under `notebooks/` and chart generators under `chart_scripts/`.\n"
    )


if __name__ == "__main__":
    main()
