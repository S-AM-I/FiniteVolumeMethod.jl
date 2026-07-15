# Workflow status

## Live (trigger automatically)

- `CI.yml` — pushes to `main`, tags, and PRs. Jobs: environment-integrity,
  unit-interop (Julia stable/LTS/pre), scientific-smoke, published-benchmarks
  (gated: all 5 benchmarks must execute their physics assertions — deferred
  benchmarks are recorded as broken and fail the job), and docs
  (build + GitHub Pages deploy — the single docs deploy path).
- `FormatCheck.yml` — Runic format check on pushes to `main`, tags, and PRs.
- `Nightly.yml` — weekly cron (Mondays 03:00 UTC) plus manual dispatch. Runs
  the full `docs/src/literate_verification` suite and the
  `FVM_RUN_VANDV`-gated collocated V&V cases (Ghia cavity, Poiseuille grid
  convergence).

## Manual only (`workflow_dispatch`)

- `benchmarks.yml` — allocation-budget regression lane.
- `Docs.yml` — docs build verification without deployment (the Cloudflare
  Pages host was retired 2026-06; deployment happens only via `CI.yml`).
- `docs-quality.yml` — lychee link check + cspell spell check.
- `jet.yml` — JET type-stability audit.

## Disabled (`*.yml.disabled`)

- `Release.yml.disabled` — release automation, disabled during the v2/v3
  overhaul (see `validation/release-checklist.md` for the manual release
  process).
- `TagBot.yml.disabled` — TagBot reacts to Julia General registry
  registrations, and this fork is not registered (and must be renamed before
  it can be), so the workflow could never fire. Re-enable only after
  registration under a new name.
