# STATE — FiniteVolumeMethod.jl
<!-- This file is highly transient. Overwrite it at the end of every session. -->

## Last Updated
2026-02-17 — CI/CD assessment and improvement.

## Current Standing
- Branch: `main` (commit `fb1d6eb`, in sync with `origin/main`)
- Version: `1.2.0` (Project.toml)
- Codebase: **109 source files** (~23k lines), **32 test files** (~15k lines)
- Unstaged changes: `.github/workflows/CI.yml` (Julia version `'1.11'` → `'1'`)
- No TODO/FIXME/HACK annotations anywhere in `src/`

## Changes This Session
- **CI.yml**: Changed test matrix Julia version from `'1.11'` (hardcoded) to `'1'` (auto-latest stable). This follows SciML convention and auto-updates when new Julia releases ship (e.g., 1.12). Matrix is now `['1', 'pre', 'lts']`.
- Assessed all 4 workflows and confirmed alignment with SciML best practices:
  - `FormatCheck.yml` — Runic formatting (passing)
  - `CI.yml` — Test matrix + docs deploy (test and docs run in parallel, `pre` has `continue-on-error`)
  - `TagBot.yml` — Automated release tagging
  - `DocCleanup.yml` — PR preview cleanup

## Known Issues
1. **`test_coordinate_systems.jl` not in `runtests.jl`** — 11 tests exist but are never invoked by the test runner
2. Aqua.jl `unbound_args` test: marked `broken = true` (Val{N} AMR false positive, pre-existing)
3. Reference image PSNR test: threshold flaky (~23.95 vs 25), CI sets `JULIA_REFERENCETESTS_UPDATE=true`
4. `keller_segel_chemotaxis.jl` tutorial: commented out in test runner, file deleted, but still in docs `_PAGES`
5. 5 specific_problem files in `src/specific_problems/` not included by main module (intentional wyos refs)

## CI Status
- **CI.yml**: Tests on Julia `1` (latest stable), `lts`, `pre` (nightly, allowed to fail). 8 GB swap. Docs build + GitHub Pages deploy. Test and docs jobs run in parallel (no `needs:` dependency).
- **FormatCheck.yml**: Runic v1 formatter check.
- **TagBot.yml**, **DocCleanup.yml**: Standard automation.

## Next Steps
- Commit and push CI.yml change
- Add `test_coordinate_systems.jl` to `runtests.jl`
- Resolve keller_segel reference in docs `_PAGES` (remove or recreate)
- Integrate cylindrical coordinate support into CRUD.jl for axisymmetric PWR fuel rod simulation
