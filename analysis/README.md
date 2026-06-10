# Analysis (local-only)

This directory holds **exploratory, study-specific analysis** — notebooks,
one-off scripts, draft figures, and self-contained sub-projects used while
developing the methodology.

By policy it is **not version-controlled**: everything under `analysis/` is
gitignored except this README (see the `/analysis/*` rule in
[`.gitignore`](../.gitignore)). This keeps generated figures, caches, and
churn-heavy research scratch out of the repository history.

## Conventions

- Reusable, tested logic belongs in [`src/abpm_hemodynamic_coupling/`](../src/abpm_hemodynamic_coupling),
  not here. Promote code out of `analysis/` once it stabilizes.
- Each sub-analysis lives in its own subdirectory and is self-contained.
- Treat anything here as reproducible from `src/` plus `data/`; do not rely on
  `analysis/` outputs as a source of truth.

## Typical contents (local, not tracked)

- `coupling-master/` — master coupling/reactivity analysis package and manuscript drafts
- `hemodynamic-coupling/` — hemodynamic-coupling sub-analysis (notebooks, charts, reports)
- `initial/` — initial EDA and descriptive-statistics notebooks
