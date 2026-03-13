# Known Failures

This file documents known test failures and their status.

## Pre-existing

| Test | Status | Notes |
|------|--------|-------|
| `Aqua.test_unbound_args` | Broken (`broken = true`) | `Val{N}` pattern in AMR constructors is a known false positive. Tracked in `test/QUALITY_LEDGER.toml`. |
| `keller_segel_chemotaxis.jl` | Skipped | Excluded from tutorial test loop (marked `manual_review` in manifest). |

## Phase 3 Stretch Goals (May Require New Features)

| Test | Status | Notes |
|------|--------|-------|
| `heated_cavity.jl` | Needs review | May require Boussinesq source term not yet in solver. Uses simplified NS-only approximation. |
| `fishbone_moncrief_torus.jl` | Needs review | Uses simplified uniform-density torus rather than exact FM analytical profile. Memory-intensive with GRMHD + Kerr metric. |

## Validation Level Notes

- Scripts marked `run_in_ci = false` in `validation/manifest.toml` are excluded from CI due to memory or runtime constraints. They are exercised in the Nightly and Release workflows.
- All numerical acceptance criteria use fixed `@test` assertions. Image regression tests use `JULIA_REFERENCETESTS_UPDATE=true` and are not part of the scientific contract.
