# ROADMAP — FiniteVolumeMethod.jl

## Current Milestone
**Test coverage and CI hygiene** — Ensure all tests run in CI, clean up stale artifacts, and stabilize documentation.

## Objectives

- [x] Migrate cylindrical/spherical coordinate assembly from Simu.jl's SimuFVM (landed in `a545f6b`)
- [ ] Add `test_coordinate_systems.jl` to `runtests.jl` (11 tests exist but are not invoked)
- [ ] Remove or recreate `keller_segel_chemotaxis` tutorial (file deleted but still in docs `_PAGES`)
- [ ] Clean up stale CI log directories (`logs/`, `logs_57507113439/`)
- [ ] Fix reference image PSNR test threshold (CI workaround: `JULIA_REFERENCETESTS_UPDATE=true`)
- [ ] Expand tutorials for advanced features (AMR, GRMHD, operator splitting)
- [ ] Verify documentation deployment pipeline (Documenter.jl + GitHub Pages)

## Future Milestones

### Next
- Validation against published benchmark problems (Sod shock tube, Orszag-Tang vortex, etc.)
- Performance profiling and optimization (especially 3D solvers and AMR)

### Backlog
- Higher-order time integration (SSP-RK4, ADER)
- Multigrid preconditioning for implicit solves
- GPU backend via KernelAbstractions or CUDA.jl
- Package registration on Julia General registry
