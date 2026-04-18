# Repository Layout

This repository is a mixed research and application codebase. The structure is intentionally split so reusable analysis logic stays in Python modules while publication materials, notebooks, and generated artifacts remain discoverable.

## Top-level map

- `src/`: reusable analysis package used by the pipeline and tests
- `analysis/`: study-specific analysis code and notebooks that sit outside the main package surface
  - `analysis/thesis/`: thesis-track statistics helpers and EDA notebooks
  - `analysis/hemodynamic_reactivity/`: hemodynamic-reactivity sub-analysis (scripts, notebooks, reports)
- `app.py` and `pages/`: Streamlit application entrypoints
- `run_pipeline.py`: batch analysis entrypoint
- `scripts/`: one-off operational helpers for figure, presentation, and article preparation
- `tests/`: regression and smoke tests
- `data/`: local input datasets (gitignored; only `data/README.md` is tracked)
- `results/`: committed research outputs referenced by the published paper
- `artifacts/`: larger generated outputs such as thesis figures, tables, and intermediate exports (gitignored)
- `docs/`: papers, article drafts, conference material, and supporting documentation
  - `docs/ieee_elnano/`: IEEE ELNANO 2026 paper sources, presentation, and brand assets
  - `docs/thesis/`: Ukrainian thesis drafts (`drafts/`), figures, and tables
- `assets/`: static assets such as the project logo referenced by README and docs

## Working conventions

- Put reusable code in `src/`, not in notebooks.
- Keep repo-hygiene changes separate from scientific-method changes.
- Commit only research outputs that are part of the documented research record; route larger or reproducible artifacts through `artifacts/` so they stay out of version control.
- Use `scripts/` for repeatable helpers. Do not add ad hoc shell snippets to the repository root.
- Keep temporary exports, lock files, and local scratch artifacts out of version control.
- Patient data never leaves `data/`. Any nested `data/` directory inside `analysis/` or elsewhere is also gitignored by policy.
