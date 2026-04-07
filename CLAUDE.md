# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FiniteVolumeMethod.jl is a Julia package for solving PDEs with three solver families:
- **Parabolic/elliptic**: Cell-vertex solver on unstructured triangular meshes (via DelaunayTriangulation.jl)
- **Hyperbolic**: Cell-centered solver on structured 1D/2D/3D meshes (Euler, MHD, Navier-Stokes, GRMHD, etc.)
- **Collocated incompressible**: OpenFOAM-style cell-centered solver on unstructured polyhedral meshes with SIMPLE/PISO/PIMPLE pressure-velocity coupling, turbulence (RANS/LES/hybrid), heat transfer, radiation, combustion, multiphase VOF, Lagrangian DPM, and dynamic mesh

Requires Julia 1.10+. Supports current stable + LTS releases. Targets eventual inclusion in the SciML ecosystem.

The capability matrix (`docs/src/capability_matrix.md`) and validation manifest (`validation/manifest.toml`) are the authoritative contracts for feature maturity. Only `stable` features are publication-grade. The collocated solver stack is `experimental`.

## Common Commands

### Running Tests
```bash
# Full test suite (slow — runs tutorials, verification, and governance checks)
julia --project -e 'using Pkg; Pkg.test()'

# Single test file
julia --project=test test/<filename>.jl

# Single test file via Docker
TEST_FILE=test/geometry.jl make ci-test-file
```

### Formatting (Runic)
All `.jl` files must pass Runic formatting. A pre-commit hook enforces this.
```bash
# Check formatting
julia --project -e 'using Runic; Runic.main(["--check", "."])'

# Auto-fix formatting
julia --project -e 'using Runic; Runic.main(["--inplace", "."])'
```
Runic is in the global Julia environment, not the project env. Key rules: spaces around `=` in kwargs (`atol = 0.01`), 4-space continuation indent (not aligned to `(`), spaces around `/` in arithmetic.

### Local CI (Docker-based)
GitHub Actions CI is enabled (`.github/workflows/CI.yml`) with four lanes: environment-integrity, unit-interop, scientific-smoke, and docs. For local iteration, use the Makefile Docker lanes:
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
- `src/core/` — SciML bridge: semidiscrete caches, state mapping, CFL callback, ODE problem construction, symbolic indexing (`symbolic_indexing.jl`), SciMLStructures parameter partitioning (`sciml_structures.jl`), `remake` support (`remake.jl`)
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

- **Unit tests** — `test/geometry.jl`, `test/conditions.jl`, `test/hyperbolic.jl`, `test/mhd.jl`, etc.
- **Collocated solver tests** — `test/incompressible.jl` (94 tests), `test/incompressible_sciml.jl` (58 tests), `test/turbulence_rans.jl` (127), `test/turbulence_les.jl` (92), `test/thermal.jl` (132), `test/mesh_io.jl` (37), `test/linear_solvers.jl` (35), `test/multiphase_vof.jl` (57), `test/combustion.jl` (49), `test/radiation.jl` (71), `test/lagrangian_dpm.jl` (53), `test/dynamic_mesh.jl` (72), `test/postprocessing.jl` (100), `test/remaining_features.jl` (116)
- **Tutorials as tests** — Literate.jl scripts from `docs/src/literate_tutorials/` and `docs/src/literate_wyos/` are executed as testsets (docs are tested code)
- **Verification cases** — driven by `validation/manifest.toml` via the `RepoValidationManifest` module; scripts from `docs/src/literate_verification/`
- **Governance** — Aqua.jl quality, environment integrity, repository governance, reproducibility bundles, quality ledger
- **MPI tests** — `test/mpi_test.jl` (NOT in runtests.jl — requires `mpiexec -n 2 julia --project=test test/mpi_test.jl`)

Note: `keller_segel_chemotaxis.jl` is explicitly excluded from the tutorials testset. Each collocated solver test file includes its own `build_cartesian_unstructured_mesh` helper due to `safe_include` module isolation.

To add a new test file, create it in `test/` and add a `safe_include("filename.jl")` entry in `test/runtests.jl` under the appropriate testset. Orphaned test files (`test/engine.jl`, `test/parabolic_solver.jl`, `test/parabolic_mesh.jl`, `test/io.jl`) exist but are NOT included in `runtests.jl`.

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
- WENO5 has a ghost cell bug in the 1D solver (`nghost=3` not supported at small grid sizes)
- Vertex-centered FVM on unstructured meshes converges at ~O(h^1.5) in L-inf norm, not O(h^2)
- Collocated SIMPLE convergence: normalized Uy residual can plateau on coarse meshes (small `||b||` denominator)
- Conjugate heat transfer uses scalar (face-averaged) interface temperature, not per-face
- Dynamic Smagorinsky uses simplified scalar Germano identity, not full tensor form
- CyclicBC assembly is approximate — true periodicity requires mesh-level periodicity support
- MPI extension uses full mesh per rank (not memory-efficient) — production MPI needs true submesh decomposition
- All collocated solver features are `experimental` maturity — not yet validated against published benchmarks
