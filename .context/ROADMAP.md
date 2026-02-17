# ROADMAP — FiniteVolumeMethod.jl

## Current Milestone
**Documentation and CI stability** — Ensure documentation builds and deploys correctly, fix remaining known test failures, and stabilize CI pipeline.

## Objectives

- [ ] Migrate cylindrical/spherical coordinate assembly from Simu.jl's SimuFVM (required for CRUD.jl — curvilinear FVM on PWR fuel rod geometry)
- [ ] Fix Aqua.jl compat bounds (missing entries for LinearAlgebra, SparseArrays, Test in Project.toml)
- [ ] Fix reference image PSNR test threshold (slightly below 25 in one tutorial)
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
