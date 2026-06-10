# Repository Layout

This repository is a mixed research and application codebase. The structure is intentionally split so reusable analysis logic stays in Python modules while publication materials, notebooks, and generated artifacts remain discoverable.

## Top-level map

- `src/abpm_hemodynamic_coupling/`: reusable analysis package used by the pipeline and tests
- `analysis/`: local-only, study-specific exploration (notebooks, draft figures, self-contained sub-analyses); gitignored except `analysis/README.md`
- `app.py`: Streamlit application entrypoint (single page). Optional pages are parked in `disabled_pages/` and can be restored under `pages/` to re-enable them
- `run_pipeline.py`: batch analysis entrypoint
- `scripts/`: one-off operational helpers for figure, presentation, and article preparation
- `tests/`: regression and smoke tests
- `data/`: local input datasets (gitignored; only `data/README.md` is tracked)
- `results/`: committed research outputs referenced by the published paper
- `artifacts/`: larger generated outputs such as thesis figures, tables, and intermediate exports (gitignored)
- `docs/`: papers, article drafts, conference material, and supporting documentation
  - `docs/ieee_elnano/`: IEEE ELNANO 2026 paper sources, presentation, and brand assets (gitignored)
  - `docs/bshka/`: Ukrainian-language article drafts (`drafts/`), figures, and tables (gitignored)
- `assets/`: static assets such as the project logo referenced by README and docs

## Working conventions

- Put reusable code in `src/abpm_hemodynamic_coupling/`, not in notebooks.
- Keep repo-hygiene changes separate from scientific-method changes.
- Commit only research outputs that are part of the documented research record; route larger or reproducible artifacts through `artifacts/` so they stay out of version control.
- Use `scripts/` for repeatable helpers. Do not add ad hoc shell snippets to the repository root.
- Keep temporary exports, lock files, and local scratch artifacts out of version control.
- Patient data never leaves `data/`. Any nested `data/` directory inside `analysis/` or elsewhere is also gitignored by policy.
