# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FiniteVolumeMethod.jl is a Julia package for solving PDEs with three solver families:
- **Parabolic/elliptic**: Cell-vertex solver on unstructured triangular meshes (via DelaunayTriangulation.jl)
- **Hyperbolic**: Cell-centered solver on structured 1D/2D/3D meshes (Euler, MHD, Navier-Stokes, GRMHD, etc.)
- **Collocated incompressible**: OpenFOAM-style cell-centered solver on unstructured polyhedral meshes with SIMPLE/PISO/PIMPLE pressure-velocity coupling, turbulence (RANS/LES/hybrid), heat transfer, radiation, combustion, multiphase VOF, Lagrangian DPM, and dynamic mesh

Requires Julia 1.10+. Supports current stable + LTS releases. Targets eventual inclusion in the SciML ecosystem. Currently in a v2 research-grade overhaul — only features marked `stable` in the capability matrix and validation manifest are publication-grade. The collocated solver stack is `experimental`.

The capability matrix (`docs/src/capability_matrix.md`) and validation manifest (`validation/manifest.toml`) are the authoritative contracts for feature maturity, V&V status, and CI inclusion.

## Common Commands

### Running Tests
```bash
# Full test suite (slow — runs tutorials, verification, and governance checks)
julia --project -e 'using Pkg; Pkg.test()'

# Single test file (test env must have FiniteVolumeMethod dev'd)
# Recommended fast-iteration loop — collocated test files ship their own
# `build_cartesian_unstructured_mesh` helper so they run standalone.
julia --project=test test/<filename>.jl

# Single test file via Docker
TEST_FILE=test/geometry.jl make ci-test-file

# Scientific evidence subset (used by CI scientific-smoke lane)
julia --project=test test/scientific_evidence.jl
```

### Formatting (Runic)
All `.jl` files must pass Runic formatting. A pre-commit hook at `.git/hooks/pre-commit` enforces this locally; CI runs `fredrikekre/runic-action@v1` on every push/PR.
```bash
# Check formatting (Runic lives in the global/default Julia env, not the project env)
julia -e 'using Runic; Runic.main(["--check", "."])'

# Auto-fix formatting
julia -e 'using Runic; Runic.main(["--inplace", "."])'

# Via Docker
make ci-format       # check only
make ci-format-fix   # auto-fix
```
Key Runic rules: spaces around `=` in kwargs (`atol = 0.01`), 4-space continuation indent (not aligned to `(`), spaces around `/` in arithmetic.

### Local CI (Docker-based)
GitHub Actions CI is active (`.github/workflows/CI.yml`, `Docs.yml`) with four jobs: environment-integrity, unit-interop, scientific-smoke, and docs. Other workflows (FormatCheck, Nightly, Release, TagBot) are disabled (`.yml.disabled`) during the v2 overhaul — see `validation/CI_REENABLE_PLAN.md` for staged re-enable criteria. For local iteration, use the Makefile Docker lanes:
```bash
make ci-fast              # Fast API/interop lane
make ci-smoke             # Scientific smoke tests
make ci-full-evidence     # Full scientific evidence
make ci-performance       # Performance baselines
make ci-release-audit     # Release audit lane
make ci-format            # Format check only
make ci-all               # All lanes
make ci-repl              # Interactive Julia REPL in container
```
First run requires `make ci-build` (downloads + precompiles all deps). Docker Desktop needs ≥12 GB memory.

### Documentation
```bash
# Build docs locally (executes examples — slow)
make ci-docs

# Build docs with CI subset only
make ci-docs-ci

# Live-server for docs development
julia --project=docs docs/liveserver.jl
```

## Architecture

### Layered Include System
The main module (`src/FiniteVolumeMethod.jl`) loads code through four layer files in `src/layers/`. The ordering is a strict dependency chain — never import backwards across layers.

1. **`domain_problem_definitions.jl`** — Foundational types, mesh definitions (parabolic and structured), coordinate systems, geometry, conditions, problem types. Also: Phase 0 collocated operators (`src/collocated/`), Phase 4 mesh I/O (`src/mesh/`), cyclic BC assembly
2. **`discretization_assembly_kernels.jl`** — FVM equation assembly, reconstruction schemes, all hyperbolic solvers, AMR, WENO/PPM/IMEX, coupling infrastructure. Also: Phase 1 incompressible solvers (`src/incompressible/`), Phase 5 linear solver config (`src/linear_solvers/`), Phase 2a/2b turbulence (`src/turbulence/`), Phase 3 thermal (`src/thermal/`), Phase 7 multiphase (`src/multiphase/`), Phase 8 combustion (`src/combustion/`), Phase 9 radiation (`src/radiation/`), Phase 10 dynamic mesh (`src/dynamic_mesh/`), Phase 11 Lagrangian DPM (`src/lagrangian/`), Phase 6 MPI stubs (`src/parallel/`)
3. **`sciml_adapters_and_accessors.jl`** — SciML integration: cache types, state mapping (fold/unfold), CFL callbacks, ODE/SplitODE construction, solution accessors, `remake`. Also: `IncompressibleSolution` wrapper, `CommonSolve.solve` dispatch for incompressible problems
4. **`extensions_tooling_output.jl`** — Dashboard types, I/O (VTK, HDF5, CSV), diagnostics, checkpointing, capability matrix. Also: Phase 12 post-processing (`src/postprocessing/`)

### Key Source Directories
- `src/parabolic/` — Cell-vertex solver: types, mesh variants, assembly, boundary conditions, gradients, limiters, turbulence models
- `src/hyperbolic/` — Cell-centered solver: conservation laws, Riemann solvers (HLL/HLLC/HLLD), reconstruction (MUSCL/PPM/WENO), plus advanced physics (Navier-Stokes, MHD variants, GRMHD, IMEX)
- `src/core/` — SciML bridge: semidiscrete caches, state mapping, CFL callback, ODE problem construction, symbolic indexing (`symbolic_indexing.jl`), SciMLStructures parameter partitioning (`sciml_structures.jl`), `remake` support (`remake.jl`). `sciml_problem(prob)` returns the underlying `ODEProblem` for hyperbolic solvers (used to access `p`/`u0` for e.g. `compute_initial_dt`).
- `src/collocated/` — Phase 0: Cell-centered operators on `UnstructuredFVMMesh` (types, interpolation, gradient, laplacian, divergence, ddt, cyclic BC assembly). Foundation for all OpenFOAM-style solvers
- `src/incompressible/` — Phase 1: SIMPLE/PISO/PIMPLE pressure-velocity coupling (types, BCs, momentum, pressure, correction, residuals, solver loops, SciML interface with `CommonSolve.solve` dispatch and `IncompressibleSolution`)
- `src/turbulence/` — Phase 2a/2b: RANS (k-ε, k-ω, k-ω SST, SA), LES (Smagorinsky, WALE, dynamic), hybrid (DDES). Interface, strain rate, wall distance, wall functions, solver wrappers
- `src/thermal/` — Phase 3: Energy equation, buoyancy (Boussinesq), solid conduction, conjugate heat transfer (Dirichlet-Neumann iteration)
- `src/mesh/` — Phase 4: OpenFOAM polyMesh reader/writer, `UnstructuredMesh3D` → `UnstructuredFVMMesh` converter, polyhedral volumes (hex/prism/pyramid), mesh quality metrics
- `src/linear_solvers/` — Phase 5: `FVMSolverConfig` per-field solver/preconditioner config, `_dispatch_solve` for field-aware solver selection, preconditioner dispatch with Val-based extension pattern
- `src/multiphase/` — Phase 7: VOF alpha transport with interface compression, boundedness limiter, mixture properties, CSF surface tension
- `src/combustion/` — Phase 8: Species transport, EDM and EDC reaction models, heat release coupling
- `src/radiation/` — Phase 9: P1 and fvDOM radiation models, Marshak wall BCs, thermal coupling
- `src/dynamic_mesh/` — Phase 10: ALE formulation, solid body motion, Laplacian motion solver, mesh update, face sweep flux
- `src/lagrangian/` — Phase 11: Drag models (Stokes, Schiller-Naumann), Ranz-Marshall heat transfer, spray breakup (TAB, KHRT), two-way coupling (PSI-cell)
- `src/postprocessing/` — Phase 12: Vorticity, Q-criterion, wall shear stress, y+, Nusselt, force coefficients, line sampling, field statistics
- `src/parallel/` — Phase 6: MPI stubs (overridden by FVMMPIExt)
- `src/amr/` — Block-structured AMR with prolongation, restriction, flux correction, subcycling
- `src/constrained_transport/` — Divergence-free magnetic field evolution (div-B preservation for 2D and 3D)
- `src/coupling/` — Multi-physics operator splitting (Lie-Trotter, Strang)

### Key Design Patterns

**Ghost-cell state mapping**: The hyperbolic solver maintains two representations of the solution. The ODE integrator sees a flat `Vector{SVector{N,FT}}` of interior cells only. Inside the RHS function, `unfold_to_padded!()` copies this into `cache.padded_U` (an array with ghost cells for boundary stencils), the RHS is computed on the padded array, and `fold_from_padded!()` copies the result back. This pattern enables allocation-free time-stepping and appears identically for 1D/2D/3D/AMR.

**Cache-as-parameter**: All hyperbolic solvers use pre-allocated `AbstractSemidiscreteCache` subtypes (`HyperbolicCache1D`, `HyperbolicCache2D`, `MHDCTCache2D`, `AMRCache`, etc.) that hold padded arrays, flux buffers, the problem object, and grid metadata. The cache is passed as the ODE parameter `p` so the RHS function never allocates.

**Conservation law interface**: The hyperbolic solver is built on `AbstractConservationLaw{Dim}`. To add new physics, subtype it and implement: `nvariables(law)`, `physical_flux(law, u, dir)`, `max_wave_speed(law, u, dir)`, `conserved_to_primitive(law, u)`, `primitive_to_conserved(law, w)`. Existing laws: `EulerEquations`, `IdealMHDEquations`, `NavierStokesEquations`, `GRMHDEquations`, `ShallowWaterEquations`, `ReactiveEulerEquations`, etc.

**Parabolic assembly**: The parabolic solver assembles `M du/dt + A u = b` matrices (dimension-specific files in `src/parabolic/assembly/`), then converts to `ODEProblem`/`LinearProblem` via Layer 3 helpers. Problem types are `FVMProblem` (single-field), `FVMSystem` (multi-field coupled), and `SteadyFVMProblem` (steady-state wrapper).

**Collocated incompressible assembly**: The collocated solver assembles `CollocatedEquation{T}` (sparse A + RHS b) using Phase 0 operators (`assemble_convection!`, `assemble_laplacian!`, `assemble_ddt_euler!`), then converts to `LinearProblem` via `to_linear_problem()`. Pressure-velocity coupling via SIMPLE (steady) / PISO / PIMPLE (transient). All sub-solves dispatch through `_dispatch_solve(lp, linear_solver, solver_config, field_name)` which routes to `FVMSolverConfig` when provided.

**SciML integration for incompressible**: `CommonSolve.solve(prob::IncompressibleProblem, alg)` returns `IncompressibleSolution` with symbolic indexing (`sol[:U]`, `sol[:p]`). `SciMLBase.remake(prob; nu=...)` modifies problem parameters. `SciMLStructures.Tunable` exposes `[nu, density, alpha_U, alpha_p, tolerance]` for parameter sensitivity workflows.

### Package Extensions
Defined in `Project.toml` under `[extensions]`:
- `FVMCUDAExt` (CUDA) — GPU backend (currently only 2D Euler; most solvers are CPU-only)
- `FVMVTKExt` (WriteVTK) — VTK output
- `FVMHdf5Ext` (HDF5) — HDF5 I/O
- `FVMCheckpointExt` (JLD2) — Checkpointing
- `FVMDashboardExt` / `FVMDashboardServerExt` (JSON3, HTTP) — Live dashboard
- `FVMRecipesExt` (RecipesBase) — Plot recipes for 1D/2D hyperbolic solutions
- `FVMAMGExt` (AlgebraicMultigrid) — AMG preconditioner for pressure Poisson
- `FVMILUExt` (IncompleteLU) — ILU preconditioner for velocity/scalar equations
- `FVMLinearSolveExt` (LinearSolve) — Krylov solver constructors (CG, BiCGSTAB, GMRES)
- `FVMMPIExt` (MPI + PartitionedArrays) — Distributed mesh, halo exchange, parallel SIMPLE

### Test Organization
`test/runtests.jl` orchestrates all tests via `safe_include()`, which runs each test file in its own anonymous module to prevent namespace pollution between tests. The test suite includes:

- **Unit tests** — `test/geometry.jl`, `test/conditions.jl`, `test/hyperbolic.jl`, `test/mhd.jl`, `test/advanced_bcs.jl` (parabolic boundary gradient / segment utilities), `test/advanced_numerics.jl` (Phase 13: PPM, positivity-preserving limiter), `test/extended_physics.jl` (extended conservation laws), etc.
- **Collocated solver tests** — `test/incompressible.jl` (94 tests), `test/incompressible_sciml.jl` (58 tests), `test/turbulence_rans.jl` (127), `test/turbulence_les.jl` (92), `test/thermal.jl` (132), `test/mesh_io.jl` (37), `test/linear_solvers.jl` (35), `test/multiphase_vof.jl` (57), `test/combustion.jl` (49), `test/radiation.jl` (71), `test/lagrangian_dpm.jl` (53), `test/dynamic_mesh.jl` (72), `test/postprocessing.jl` (100), `test/remaining_features.jl` (116)
- **Tutorials as tests** — Literate.jl scripts from `docs/src/literate_tutorials/` and `docs/src/literate_wyos/` are executed as testsets (docs are tested code)
- **Verification cases** — driven by `validation/manifest.toml` via the `RepoValidationManifest` module; scripts from `docs/src/literate_verification/`
- **Governance** — Aqua.jl quality, environment integrity, repository governance, reproducibility bundles, quality ledger
- **MPI tests** — `test/mpi_test.jl` (NOT in runtests.jl — requires `mpiexec -n 2 julia --project=test test/mpi_test.jl`)

Note: `keller_segel_chemotaxis.jl` is explicitly excluded from the tutorials testset. Each collocated solver test file includes its own `build_cartesian_unstructured_mesh` helper due to `safe_include` module isolation.

To add a new test file, create it in `test/` and add a `safe_include("filename.jl")` entry in `test/runtests.jl` under the appropriate testset. Orphaned test files (`test/parabolic_solver.jl`, `test/parabolic_mesh.jl`, `test/io.jl`, `test/engine.jl`) exist but are NOT included in `runtests.jl`.

### Validation Infrastructure
- `validation/manifest.toml` — Machine-readable source of truth for feature maturity, V&V status, and CI inclusion. Features are `stable`, `experimental`, or `deprecated`.
- `validation/manifest.jl` — Julia module (`RepoValidationManifest`) that parses the manifest; used by both tests and docs builds
- `test/KNOWN_FAILURES.md` — Documents known broken/skipped/demoted tests

### Collocated Solver Key Types
- `IncompressibleProblem{Dim, T}` — problem definition (mesh, BCs, algorithm, nu, density)
- `IncompressibleState{Dim, T}` — mutable flow state (U, p, phi, A_P, H_U)
- `IncompressibleSolution{Dim, T}` — SciML-compatible result with `sol[:U]`, `sol[:p]` access
- `SolveResult{Dim, T}` — raw result with converged, iterations, residuals
- `SIMPLE{T}`, `PISO{T}`, `PIMPLE{T}` — pressure-velocity coupling algorithms
- `CollocatedEquation{T}` — assembled linear system (A, b, source) → `to_linear_problem()` → `LinearProblem`
- `CollocatedScalarField{T}`, `CollocatedVectorField{Dim,T}`, `FaceFluxField{T}` — cell-centered fields
- `FVMSolverConfig` + `FieldSolverConfig` — per-field solver/preconditioner configuration
- 15 boundary condition types: `FixedVelocityBC`, `FixedPressureBC`, `NoSlipWallBC`, `SlipWallBC`, `InletOutletBC`, `ZeroGradientBC`, `TotalPressureBC`, `SymmetryBC`, `FlowRateInletBC`, `TimeDependentVelocityBC`, `WallFunctionBC`, `ConvectiveOutletBC`, `PressureInletVelocityBC`, `CyclicBC`, `CustomBC`

### Collocated Solver SciML Integration
- `CommonSolve.solve(prob, SIMPLE())` / `solve(prob, PISO(); tspan, dt)` — standard SciML dispatch
- `sol[:U]`, `sol[:p]`, `sol[:Ux]`, `sol[:Uy]`, `sol[:phi]` — symbolic field access
- `SciMLBase.remake(prob; nu=1e-4, density=2.0)` — immutable problem modification
- `SciMLStructures.Tunable` — parameter extraction `[nu, density, alpha_U, alpha_p, tolerance]`
- Optional physics via keyword args: `solve(prob, SIMPLE(); turb_model=StandardKEpsilon(), thermal_props=..., bcs_T=..., rad_model=P1Model(), ...)`

## Known Issues

The repo is in a v2→v3 overhaul; the authoritative issue list is `test/KNOWN_FAILURES.md`. High-level summary as of v3.108:

### Still-open correctness items
- WENO5 has a ghost-cell bug in the 1D solver (`nghost=3` unsupported at small grid sizes)
- Vertex-centered FVM on unstructured meshes converges at ~O(h^1.5) in L∞, not O(h^2)
- Collocated SIMPLE: normalized Uy residual can still plateau on very coarse meshes (small `‖b‖` denominator); OpenFOAM-style scale-invariant normalization in `src/incompressible/residuals.jl` reduced the 80×80 floor from ~2e-2 to ~3e-3 but does not eliminate it
- CyclicBC face matching converges slowly on coarse meshes (Stage 1a follow-up)
- IDDES uses `V_c^(1/Dim)` as a surrogate for `h_max`; full real-edge-length variant is a v3.2 follow-up
- v3.108 still has no end-to-end published-benchmark gate run in CI — the harness in `validation/published_benchmarks/` is gated by `FVM_RUN_BENCHMARKS=true` and runs at the user's terminal only

### v3 fast-path delivered features (production-ready, awaiting published-benchmark stable promotion)
The v3.102→v3.108 waves moved the following items from "simplification" to production. None are yet promoted to `stable` because that requires ≥3 published-benchmark gates green in CI, but each is a real, tested implementation rather than a stub:

- **v3.102 (Wave 1)** — pressure-based compressible SIMPLE/PIMPLE (`src/pressure_based/compressible_*`); Durbin realizability ON BY DEFAULT in k-ε; full-tensor production via `_sym_self_magnitude_sq`; EquilibriumWMLES + SADDES (full implementations); per-face Patankar CHT (replaces scalar interface T); enthalpy energy equation selectable via `use_enthalpy=true`; MULES wired into alpha_transport (was primitive-only since v3.91); isoAdvector (`src/multiphase/iso_advector.jl`); static + Cox-Voinov contact angles; over-relaxed non-orthogonal correction is now the default; LSQ gradient
- **v3.103 (Wave 2)** — Cantera weak-dep extension + multi-step combustion + variable Lewis + FGM (`src/combustion/{multi_step,variable_lewis,fgm}.jl`); fvDOM scattering, S6/S8/S12 quadratures, WSGGM (`src/radiation/wsggm.jl`); HardSphere/SoftSphere DEM + agglomeration (`src/lagrangian/{collisions,agglomeration}.jl`); primary breakup KH-ACT + LISA + cone/hollow/flat-fan/solid injectors (`src/lagrangian/{primary_breakup,injection}.jl`); 6-DOF + topoChanger + overset + AMI (`src/dynamic_mesh/{six_dof,topo_changer,overset,ami}.jl`); Kunz/Schnerr-Sauer/Merkle cavitation (`src/cavitation/`); Darcy-Forchheimer porous media (`src/porous/`)
- **v3.104 (Wave 3)** — multi-zone MRF; linear elasticity + updated-Lagrangian finite strain (`src/solid_mechanics/`); Aitken FSI partitioned coupling (`src/fsi/`); FW-H aeroacoustics (Curle + Lighthill stub) + PML sponge zones (`src/aeroacoustics/`); QMoM + DQMoM + Class Method PBM (`src/population_balance/`)
- **v3.105 (Wave 4)** — Gmsh weak-dep extension + snappy stub (`src/mesh_generation/`); collocated AMR + ZZ + residual indicators (`src/amr_collocated/`); steady-SIMPLE adjoint + transient adjoint stub (`src/adjoint/`); KernelAbstractions weak-dep extension; Enzyme weak-dep stub; CoolProp + PETSc weak-dep extensions; ExpressionBC (renamed StringExpressionBC in v3.108) + probe + Unitful hook
- **v3.106 (Wave 5)** — `LocalFVMMesh` + Metis partitioner (`src/parallel/{local_mesh,rcb_partitioner,metis_stub}.jl`); distributed assembly via PartitionedArrays `PSparseMatrix`; Eulerian two-fluid types (experimental at this point)
- **v3.107 (v3.1 wave)** — production Eulerian two-fluid solver via `BlockCollocatedEquation{T,2}` with off-diagonal drag linearization (`src/multiphase/two_fluid_solver.jl`); transient PIMPLE adjoint with uniform checkpointing (`solve_transient_adjoint_linear`, no longer a stub); snappyHexMesh native castellated + surface snap (layer addition deferred to v3.2); IDDES full Shur-2008 shielding (no longer a stub); primary-breakup FSI handshake (`couple_primary_breakup_fsi!`); published-benchmark harness scaffold gated by `FVM_RUN_BENCHMARKS=true`
- **v3.108 (full-suite triage)** — `ExpressionBC` → `StringExpressionBC` rename (resolves collision with parabolic `ExpressionBC`); KA backend dispatch fixed (no method overwrite during precompile); stale stub-warn tests updated to verify production behaviour

### Structural items already addressed
The following structural items previously listed here have been resolved during the v2→v3 overhaul. See `test/KNOWN_FAILURES.md` for fix-stage details and assertions:
- `CollocatedEquation` random-pattern CSC insertion (Stage 1a — `SparsityPattern` pre-computes `nzval` indices, O(1) writes)
- Operator hot-loop allocations (Stage 1b — `Dict{Int,Int}` replaced with `Vector{Int}`; audit confirmed gradients/interpolation are zero-alloc on the hot path)
- `BlockCollocatedEquation` for vector-valued / two-fluid systems (Stage 1c, in production use by v3.107 two-fluid solver)
- `SciMLStructures.Tunable` named registry (Stage 1e — `register_tunable!` + `tunable_schema`, no longer hardcoded length-5)
- MPI full-mesh-per-rank (Stage 2 + Wave 5 — `LocalFVMMesh`, RCB + Metis partitioners, distributed `PSparseMatrix` assembly via PartitionedArrays.jl)
- OpenFOAM binary polyMesh reader landed alongside Gmsh v4 reader

### Validation status (v3.108)
- All collocated solver features in `src/{incompressible,turbulence,thermal,multiphase,combustion,radiation,lagrangian,dynamic_mesh,solid_mechanics,fsi,aeroacoustics,population_balance,cavitation,porous,mrf,adjoint,mesh_generation,amr_collocated}/` are marked `provisional` (or `experimental` for `mpi_parallel`) in `validation/manifest.toml`. The 7-Evidence-entry harness per feature establishes algebraic + invariant coverage, but published-benchmark execution (≥3 per feature) is the explicit gate for `stable` promotion and is not yet wired into CI.
- Outstanding deferrals to v3.2 / v3.3: layer addition in snappyHexMesh, Enzyme full-solver AD, IDDES `h_max` from real edge lengths, Sandia Flame D combustion benchmark, all `stable`-tier promotions.
- v3 roadmap remains in `plans/i-m-not-sure-of-ticklish-squid.md`; `test/KNOWN_FAILURES.md` is the authoritative per-item status list.
