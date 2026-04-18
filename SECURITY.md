# Security Policy

## Supported Versions

This repository is maintained on the default branch. Security fixes are
applied there first.

## Patient data policy

Patient-level ambulatory blood pressure monitoring data is treated as
Protected Health Information (PHI) and must never be committed to git.
The `.gitignore` rules enforce this at two levels:

- `/data/*` is ignored except for the tracked `/data/README.md` that
  documents the expected schema.
- `**/data/` is ignored everywhere else in the tree so that nested
  `data/` directories inside `analysis/` or similar locations also stay
  out of version control.

The `docs/ieee_elnano/` and `docs/thesis/` areas are likewise ignored;
they contain manuscript drafts and per-subject tables that may carry
re-identification risk in a small cohort and are kept on local disk
only.

If patient data ever lands in a commit, the remediation path used in
this repository is `git filter-repo --invert-paths --path <file>`
followed by a coordinated force-push to rewrite the remote history. A
precedent of this procedure was executed on 2026-04-18 to purge
previously committed monitoring CSVs and XLSX files.

## Reporting a Vulnerability

Do not open a public GitHub issue for suspected vulnerabilities,
sensitive data exposure, or accidental disclosure of participant
information.

Report security concerns privately to the maintainers:

- open a private GitHub security advisory on this repository, or
- email the repository owner directly at `vbabenko2191@gmail.com`.

Include:

- a clear description of the issue
- steps to reproduce or verify it
- the affected files, commands, or workflows
- whether any real participant or health-related data may be exposed

You will receive an acknowledgement after the report is reviewed.
Public disclosure should wait until the issue is understood and a
mitigation path is available. If PHI was committed, expect a history
rewrite and coordinated force-push as part of the fix.
