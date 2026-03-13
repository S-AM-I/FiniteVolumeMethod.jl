# Known Failures

This file documents known test failures and their status.
The authoritative machine-readable source of truth for exclusions and demotions
is `validation/manifest.toml`; this document is a human-readable companion.

## Pre-existing

| Test | Status | Notes |
|------|--------|-------|
| `Aqua.test_unbound_args` | Broken (`broken = true`) | `Val{N}` pattern in AMR constructors is a known false positive. Tracked in `test/QUALITY_LEDGER.toml`. |
| `keller_segel_chemotaxis.jl` | Skipped | Excluded from tutorial test loop (marked `manual_review` in manifest). |

## Demoted From V&V Claims

| Test | Status | Notes |
|------|--------|-------|
| `heated_cavity.jl` | Demoted | Uses a simplified compressible surrogate, not a De Vahl Davis validation case. |
| `fishbone_moncrief_torus.jl` | Demoted | Uses an approximate torus initial condition, not a Fishbone-Moncrief equilibrium solution. |
| `lid_driven_cavity.jl` | Demoted | Does not impose the literature benchmark boundary treatment or compare against published profiles quantitatively. |
| `bondi_accretion_schwarzschild.jl` | Demoted | Current setup is not an actual Bondi solution and therefore cannot support a Bondi validation claim. |
| `amr_convergence.jl` | Demoted | Current assertions are regression/smoke checks, not a rigorous AMR convergence study. |
| `mhd_solver_comparison.jl` | Demoted | Relative solver comparison without external truth is not treated as scientific evidence. |
| `premixed_flame_1d.jl` | Demoted | Current checks are qualitative combustion regression checks, not a literature-backed validation case. |

## Validation Level Notes

- Scripts marked `run_in_ci = false` in `validation/manifest.toml` are excluded from CI due to memory or runtime constraints. They are exercised in the Nightly and Release workflows.
- All numerical acceptance criteria use fixed `@test` assertions. Image regression tests use `JULIA_REFERENCETESTS_UPDATE=true` and are not part of the scientific contract.
