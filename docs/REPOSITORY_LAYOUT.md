# Repository Layout

This repository is a mixed research and application codebase. The structure is intentionally split so reusable analysis logic stays in Python modules while publication materials, notebooks, and generated artifacts remain discoverable.

## Top-level map

- `src/`: reusable analysis package used by the pipeline and tests
- `analysis/`: thesis-specific or publication-specific analysis helpers that sit outside the main package surface
- `app.py` and `pages/`: Streamlit application entrypoints
- `run_pipeline.py`: batch analysis entrypoint
- `scripts/`: one-off operational helpers for figure and presentation preparation
- `tests/`: regression and smoke tests
- `data/`: local input datasets and data notes
- `results/`: generated outputs, including thesis artifacts intentionally kept in version control
- `docs/`: papers, article drafts, conference material, and supporting documentation
- `notebooks/`: exploratory notebooks that should not become the only source of truth for production logic

## Working conventions

- Put reusable code in `src/`, not in notebooks.
- Keep repo-hygiene changes separate from scientific-method changes.
- Treat `results/` as generated output. Only commit artifacts that are part of the documented research record.
- Use `scripts/` for repeatable helpers. Do not add ad hoc shell snippets to the repository root.
- Keep temporary exports, lock files, and local scratch artifacts out of version control.
