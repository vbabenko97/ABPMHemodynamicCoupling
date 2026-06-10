# CLAUDE.md

Project memory for ABPM Hemodynamic Coupling — a scientific Python pipeline +
Streamlit app analyzing peripheral arterial tone from ABPM data (IEEE ELNANO
2026 + Ukrainian bachelor's thesis). Keep this file concise; it is prepended to
every prompt.

## Layout
- Package code lives in `src/abpm_hemodynamic_coupling/`; import as `from abpm_hemodynamic_coupling import ...`.
- `run_pipeline.py` is a top-level module (declared in `[tool.setuptools] py-modules`), not inside the package.
- `app.py` is the single-page Streamlit entrypoint. Optional pages are parked in `disabled_pages/`; move one to `pages/` to re-enable it.
- `analysis/` is local-only exploration — gitignored except `analysis/README.md`. Do not commit its contents; promote reusable code into `src/`.

## Environments & commands
- Use `.venv` (Python 3.13) for the app, pipeline, and tests: `.venv/bin/python`.
- `.thesis-venv` holds the heavier thesis-notebook extras (`pip install -e .[thesis]`).
- Tests: `.venv/bin/python -m pytest`  •  Lint: `.venv/bin/python -m ruff check .`
- Both must pass before commit; CI runs ruff + pytest on Python 3.11.
- Ruff: line-length 100, `E501` ignored, `select = E,F,I,B`; `analysis/` and `scripts/generate_presentation.py` are excluded.

## Dependencies
- `pyproject.toml` + `uv.lock` are canonical. Extras: `dev`, `thesis`, `modeling` (shap/xgboost).
- `requirements.txt` / `requirements-thesis.txt` are lightweight compatibility lists. Never hand-edit `uv.lock`.

## Data & privacy
- Patient data lives only in `data/` (gitignored; only `data/README.md` is tracked). Never commit data.
- Also gitignored: any nested `data/`, `artifacts/`, `docs/ieee_elnano/`, `docs/bshka/`, presentation exports, `tmp/`.

## Scientific rigor (important)
- Reproducibility constants are in `src/abpm_hemodynamic_coupling/config.py`: `RANDOM_SEED = 42`, `FDR_ALPHA = 0.1`, `BOOTSTRAP_ITERATIONS`.
- Changing statistical constants or modeling logic alters published/thesis results — treat as a scientific change, not hygiene, and call it out explicitly.
- Defence is near: do not refactor `modeling.py` / `stats_analysis.py` / `visualization.py` without characterization tests pinning current outputs first. Test coverage is currently thin (smoke + data-processing).

## Conventions
- Keep repo-hygiene commits separate from scientific-method commits (one concern per commit).
- The Streamlit UI and result labels are Ukrainian — preserve Ukrainian wording when editing UI/labels.
