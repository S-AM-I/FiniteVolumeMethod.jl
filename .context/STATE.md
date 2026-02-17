# STATE — FiniteVolumeMethod.jl
<!-- This file is highly transient. Overwrite it at the end of every session. -->

## Last Updated
2026-02-16 — Initial context setup (orchestration scheme bootstrap)

## Current Standing
- Version 1.2.0 — mature, production-ready package
- ~22,900 lines of source code across 60+ files
- **Triangular FVM:** All 7 template problems work, all BC types (including nonlinear, periodic, coupled), gradient reconstruction, MUSCL scheme, flux limiters, parallel solving, turbulence models (k-ε, κ-ω SST)
- **Hyperbolic solver:** 1D/2D/3D Euler, MHD (ideal/resistive/Hall), Navier-Stokes, shallow water, SR hydro, SRMHD, GRMHD all functional
- All 4 Riemann solvers, 6 reconstruction schemes, constrained transport, positivity limiter, AMR with flux correction
- IMEX time integration, operator splitting (Lie-Trotter, Strang), unstructured hyperbolic solver
- **1,288,388 tests passing** — only 4-5 known failures (Aqua compat bounds, reference image PSNR)
- Recent development (Feb 2026): Phases 9-14 implementation, documentation reorganization, Runic formatting
- CI: GitHub Actions being tested/refined

## Known Issues
1. Aqua.jl compat bounds: missing entries for LinearAlgebra, SparseArrays, Test
2. Reference image test: PSNR slightly below 25 threshold in one tutorial
3. Some advanced feature tutorials need expansion

## Next Step
Fix the Aqua.jl compat bounds in Project.toml (add compat entries for stdlib packages) to clear the known test failures.
