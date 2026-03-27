# Contributing

## Scope

This repository focuses on cleanup, reproducibility, and scientific traceability for the ABPM hemodynamic coupling pipeline. Keep pull requests small, reviewable, and anchored to a concrete problem statement.

## Development Setup

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python -m pytest
ruff check .
```

## Change Rules

- Do not change the 6D DBP feature space without a documented scientific rationale.
- Do not change `FDR_ALPHA`, `BOOTSTRAP_ITERATIONS`, `RANDOM_SEED`, `RESPONDER_ANOMALY_THRESHOLD`, or `RESPONDER_BIAS_THRESHOLD` in cleanup-only pull requests.
- Keep data-loading and preprocessing changes backward compatible with the documented CSV schema.
- Separate scientific-method changes from repository-hygiene changes.

## Pull Requests

- Describe the problem, the exact files changed, and the validation you ran.
- Include before/after screenshots for figure or presentation layout changes.
- Call out any result changes explicitly if a code change can affect outputs.

## Reporting Issues

Use the bug report template and include the command you ran, the input files involved, and the traceback or log output.
