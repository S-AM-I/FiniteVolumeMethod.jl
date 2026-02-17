# STATE — FiniteVolumeMethod.jl
<!-- This file is highly transient. Overwrite it at the end of every session. -->

## Last Updated
2026-02-17 — Repository audit and context refresh.

## Current Standing
- Branch: `main` (commit `0eefbdf`, in sync with `origin/main`)
- Version: `1.2.0` (Project.toml)
- Codebase: **109 source files** (~23k lines), **32 test files** (~15k lines)
- Unstaged changes: modified `STATE.md`, deleted stale CI log files under `logs/` and `logs_57507113439/`
- No TODO/FIXME/HACK annotations anywhere in `src/`

## Features Landed Since Last Major Session
- `AbstractCoordinateSystem` type hierarchy (`Cartesian`, `Cylindrical`, `Spherical`)
- `geometric_volume_weight` / `geometric_flux_weight` for coordinate-specific Jacobian weighting
- `FVMGeometry{T, S, C <: AbstractCoordinateSystem}` parameterized on coordinate system
- Sub-CV areas and face fluxes weighted by coordinate Jacobian (r for cylindrical, r²sinθ for spherical)
- `Base.:(==)` for `FVMGeometry` to fix Julia 1.12 `===` bug on large immutable structs
- Default `Cartesian()` preserves full backward compatibility

## Known Issues
1. **`test_coordinate_systems.jl` not in `runtests.jl`** — 11 tests exist in `test/test_coordinate_systems.jl` but are never invoked by the test runner. Should add a testset entry.
2. Aqua.jl `unbound_args` test: marked `broken = true` (Val{N} AMR false positive, pre-existing)
3. Reference image PSNR test: threshold flaky (~23.95 vs 25, pre-existing), CI sets `JULIA_REFERENCETESTS_UPDATE=true` to work around it
4. `keller_segel_chemotaxis.jl` tutorial: commented out in test runner (line 142) and the file no longer exists in `docs/src/literate_tutorials/`, but still referenced in docs `_PAGES` → potential doc build breakage
5. 5 specific_problem files (`diffusion_equation.jl`, `laplaces_equation.jl`, `linear_reaction_diffusion_equations.jl`, `mean_exit_time.jl`, `poissons_equation.jl`) live in `src/specific_problems/` but are **not included** by the main module — intentional (wyos reference implementations), but inconsistent with CHARTER listing them as shipped templates
6. Stale CI log directories (`logs/`, `logs_57507113439/`) tracked as deleted but not committed

## CI Status
- **CI.yml**: Tests on Julia `1.11`, `lts`, `pre` (nightly, allowed to fail). 8 GB swap. Docs build + GitHub Pages deploy.
- **FormatCheck.yml**: Runic v1 formatter check.
- **TagBot.yml**, **DocCleanup.yml**: Standard automation.

## Next Steps
- Add `test_coordinate_systems.jl` to `runtests.jl`
- Clean up deleted log files (commit or `.gitignore`)
- Resolve keller_segel reference in docs `_PAGES` (remove or recreate)
- Integrate cylindrical coordinate support into CRUD.jl for axisymmetric PWR fuel rod simulation
