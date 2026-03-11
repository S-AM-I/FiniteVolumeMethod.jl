# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

FiniteVolumeMethod.jl is a Julia package for solving partial differential equations using the finite volume method. It provides three solver families:

1. **Triangular (vertex-centered) FVM** for parabolic/elliptic PDEs on unstructured meshes via DelaunayTriangulation.jl
2. **Cell-centered FVM** for hyperbolic conservation laws on structured/unstructured grids
3. **Parabolic solver** (migrated from Simu.jl) for diffusion/advection-diffusion on structured, curvilinear, and unstructured meshes with a full simulation engine and I/O stack

Version: `1.2.0`.

## Commands

```bash
# Run all tests (memory-intensive; CI adds 8 GB swap)
julia --project -e 'using Pkg; Pkg.test()'

# Run a specific test file
julia --project test/geometry.jl

# Run a Literate tutorial test
julia --project docs/src/literate_tutorials/diffusion_equation_on_a_square_plate.jl

# Format code in-place (uses Runic — must be installed globally)
julia --project -e 'using Runic; Runic.main(["--inplace", "."])'

# Check formatting without modifying files
julia --project -e 'using Runic; Runic.main(["--check", "."])'

# Build documentation locally
julia --project=docs docs/make.jl

# Run quality checks
julia --project -e 'using Aqua; Aqua.test_all(FiniteVolumeMethod)'
```

## Formatting: Runic

- **CI blocks PRs** that fail the Runic check (`fredrikekre/runic-action@v1` in `FormatCheck.yml`).
- A local pre-commit hook at `.git/hooks/pre-commit` runs `Runic.main(["--check", ...])` on staged `.jl` files.
- Runic is **not** in `Project.toml` — it must be installed in your global Julia environment (`]add Runic` from the default env).
- Always run `julia --project -e 'using Runic; Runic.main(["--inplace", "."])'` before committing.
- Key style rules enforced: spaces around `=` in keyword args (`atol = 0.01`), 4-space continuation indent (not aligned to `(`), spaces around `/` in arithmetic.

## CI

- **Julia versions**: stable (`'1'`), LTS (`'lts'`), nightly (`'pre'`, allowed to fail) — all on `ubuntu-latest` x64.
- Tests require significant memory; CI adds an 8 GB swapfile before running.
- Reference tests run with `JULIA_REFERENCETESTS_UPDATE=true` in CI (images regenerated, not checked against fixed baselines).
- `dependabot.yml` monitors Julia package deps (daily) and GitHub Actions (weekly).

## Architecture

### Core Type Hierarchy

**Problem Types** (in `src/problem.jl`):
- `FVMProblem` - Single PDE (most general form)
- `FVMSystem` - System of coupled PDEs
- `SteadyFVMProblem` - Wrapper for steady-state problems
- `AbstractFVMTemplate` - Optimized templates for specific problem classes

**Geometry** (in `src/geometry.jl`):
- `FVMGeometry{T, S, C <: AbstractCoordinateSystem}` - Main mesh structure wrapping DelaunayTriangulation.jl, parameterized on coordinate system
- `TriangleProperties` - Pre-computed geometric properties per triangle

**Coordinate Systems** (in `src/coordinate_systems.jl`):
- `AbstractCoordinateSystem` → `Cartesian`, `Cylindrical`, `Spherical`
- `geometric_volume_weight(cs, x, y)` / `geometric_flux_weight(cs, x, y)` for Jacobian weighting
- Cylindrical: axisymmetric (r,z), weight = r. Spherical: (r,θ), weight = r²sinθ
- Default `Cartesian()` preserves backward compatibility

**Conditions** (in `src/conditions.jl` and `src/conditions/`):
- `BoundaryConditions` - Boundary condition specifications
- `InternalConditions` - Internal node constraints
- `ConditionType` enum: `Neumann`, `Dirichlet`, `Dudt`, `Constrained`, `Robin`
- Advanced: `NonlinearDirichlet`, `NonlinearNeumann`, `NonlinearRobin`, `PeriodicBC`, `CoupledBC`

### Module Structure

```
src/
├── FiniteVolumeMethod.jl   # Main module, exports (~500 symbols), precompile workload
├── coordinate_systems.jl   # AbstractCoordinateSystem hierarchy
├── geometry.jl             # FVMGeometry mesh wrapper
├── conditions.jl           # Core BC types
├── conditions/             # Advanced BCs (nonlinear, periodic, coupled)
├── problem.jl              # Problem definitions
├── solve.jl                # fvm_eqs!, jacobian_sparsity, threading
├── utils.jl                # Utilities
├── remake.jl               # SciMLBase.remake for all problem types
├── equations/              # Core FVM discretization (shape functions, triangle/boundary contributions)
├── schemes/                # MUSCL, gradient reconstruction, limiters
├── specific_problems/      # Templates (abstract_templates, advection_diffusion, anisotropic_diffusion)
│                           # NOTE: diffusion_equation, laplaces_equation, mean_exit_time,
│                           # poissons_equation, linear_reaction_diffusion are reference
│                           # implementations for wyos tutorials — NOT included by main module
├── physics/turbulence/     # k-epsilon turbulence model
├── mesh/                   # AbstractMesh, StructuredMesh{1D,2D,3D}, UnstructuredHyperbolicMesh
├── eos/                    # EOS interface, IdealGasEOS, StiffenedGasEOS
├── hyperbolic/             # Cell-centered FVM for conservation laws (see below)
├── constrained_transport/  # CT for div(B)=0 (2D and 3D)
├── metric/                 # Spacetime metrics (Minkowski, Schwarzschild, Kerr)
├── amr/                    # Block-structured AMR with Berger-Colella flux correction
├── coupling/               # Multi-physics operator splitting
├── dashboard/              # Export + callbacks (served via FVMDashboardExt extension)
├── parabolic/              # Simu.jl migration — parabolic PDE solver (see below)
│   ├── types.jl            # Core abstract types (tags, simulation, problem, solution)
│   ├── models.jl           # Equation models (Diffusion, Advection, AdvectionDiffusion per dim)
│   ├── mesh/               # Structured, curvilinear, unstructured mesh types + I/O + partitioning
│   ├── assembly/           # FVM matrix assembly per geometry (1D, 2D, 3D, cylindrical, spherical, ...)
│   ├── boundary_conditions.jl, gradients.jl, limiters.jl, schemes.jl
│   ├── compressible_fluxes.jl, turbulence.jl, particles.jl, fsi.jl, kernels.jl
│   └── utils.jl
├── engine/                 # Simu.jl migration — simulation engine
│   ├── orchestration.jl    # Simulation container, TimeGrid, TimeController, events
│   ├── steppers.jl         # ForwardEuler, RK2, ImplicitEuler, Rosenbrock23, CrankNicolson
│   ├── newton.jl           # Newton-Raphson, Newton-Krylov, Anderson acceleration
│   ├── solvers.jl          # solve_steady_state, solve_transient, solve_adaptive
│   ├── coloring.jl         # Graph coloring for Jacobian computation
│   ├── adjoint.jl          # Adjoint sensitivity analysis
│   └── estimation.jl       # InverseProblem / calibrate_model
└── io/                     # Simu.jl migration — I/O and diagnostics
    ├── manager.jl          # OutputManager, scheduling
    ├── diagnostics.jl      # volume_integral, conservation_summary, boundary_fluxes
    ├── vtk.jl              # VTK output
    ├── hdf5.jl             # HDF5 stubs
    ├── checkpointing.jl    # Save/load checkpoints
    ├── insitu.jl           # Probes, integral monitors
    ├── registry.jl         # Model package save/load
    └── utils.jl            # CSV, TOML, formatting helpers
ext/
└── FVMDashboardExt.jl      # Package extension (weak deps: HTTP + JSON3) — serve_dashboard, export/import_session
```

### Simu.jl Migration (Parabolic / Engine / I/O)

Three subsystems migrated from Simu.jl live alongside the original FVM code. They share the same Julia module but use **prefixed names** to avoid collisions with the original types:

| Original FVM type | Simu.jl migration type |
|---|---|
| `Dirichlet`, `Neumann`, `Robin` | `ParabolicDirichlet`, `ParabolicNeumann`, `ParabolicRobin` |
| `AbstractMesh` | `AbstractParabolicMesh` |
| `AbstractOperator` (coupling) | `AbstractPhysicsOperator` |

**Parabolic solver** (`src/parabolic/`): Cell-centered FVM for diffusion/advection-diffusion on structured, curvilinear, and unstructured meshes (1D/2D/3D, cylindrical, spherical). Includes MUSCL/QUICK/WENO5 reconstruction, Green-Gauss and least-squares gradient methods, parabolic k-epsilon turbulence, Lagrangian particle tracking, and FSI.

**Engine** (`src/engine/`): Simulation orchestration with `Simulation` container, time steppers (ForwardEuler through CrankNicolson), Newton/Krylov nonlinear solvers, graph-colored Jacobian computation, adjoint sensitivity, and parameter estimation.

**I/O** (`src/io/`): `OutputManager` with scheduled writes, VTK/HDF5 output, diagnostics (conservation, boundary fluxes), in-situ probes, and checkpointing.

### Key Function Signatures

**Flux function**: `q(x, y, t, α, β, γ, p) → (qx, qy)` where `(α, β, γ)` are shape function coefficients

**Diffusion shortcut**: `D(x, y, t, u, p)` - auto-converted to flux

**Boundary condition**: `(x, y, t, u, p) → value` or `(x, y, t, u, p) → (a=, b=, c=)` for Robin

### Pipeline

1. Create mesh: `FVMGeometry(triangulate(...))` or `FVMGeometry(tri; coordinate_system=Cylindrical())`
2. Define BCs: `BoundaryConditions(mesh, bc_fn, Dirichlet)`
3. Create problem: `FVMProblem(mesh, BCs; diffusion_function=..., initial_condition=..., final_time=...)`
4. Solve: `solve(prob, Tsit5())` using DifferentialEquations.jl

### Threading

Multi-threading via `Threads.nthreads()` with ChunkSplitters. Thread-safe temporaries use PreallocationTools.DiffCache.

### Hyperbolic Solver Framework (Cell-Centered FVM)

A separate cell-centered finite volume solver for hyperbolic conservation laws on structured Cartesian meshes. Uses explicit time integration (forward Euler, SSP-RK3) with Godunov-type Riemann solvers.

**Mesh** (in `src/mesh/`):
- `AbstractMesh{Dim}` with `StructuredMesh1D`, `StructuredMesh2D`, `StructuredMesh3D`
- `UnstructuredHyperbolicMesh` for unstructured grids

**EOS** (in `src/eos/`):
- `AbstractEOS` → `IdealGasEOS`, `StiffenedGasEOS`

**Conservation Laws** (in `src/hyperbolic/`):
- `AbstractConservationLaw{Dim}` interface: `nvariables`, `physical_flux`, `max_wave_speed`, `wave_speeds`, `conserved_to_primitive`, `primitive_to_conserved`
- `EulerEquations{Dim,EOS}` — 3/4/5 variables (1D/2D/3D)
- `IdealMHDEquations{Dim,EOS}` — 8 variables [ρ,ρv,E,B]
- `NavierStokesEquations{Dim,EOS}` — wraps Euler + viscosity
- `ResistiveMHDEquations` — magnetic diffusivity
- `HallMHDEquations` — whistler waves, ion-scale dynamics
- `ShallowWaterEquations` — with bottom topography
- `SRHydroEquations` — relativistic hydro without B
- `TwoFluidEquations` — separate ion/electron fluids
- `SRMHDEquations{Dim,EOS}` — 8 variables, relativistic con2prim
- `GRMHDEquations{Dim,EOS,Metric}` — Valencia formulation + geometric source terms

**Riemann Solvers**: `LaxFriedrichsSolver`, `HLLSolver`, `HLLCSolver`, `HLLDSolver`

**Reconstruction**: `NoReconstruction`, `CellCenteredMUSCL`, `PPMReconstruction`, `WENO3`, `WENO5`, `CharacteristicWENO`

**Time Integration**: Forward Euler, SSP-RK3, IMEX-RK (`IMEX_SSP3_433`, `IMEX_ARS222`, `IMEX_Midpoint`)

**Constrained Transport** (in `src/constrained_transport/`):
- `CTData2D`/`CTData3D` — face-centered B, edge-centered EMF
- Guarantees div(B) = 0 to machine precision for MHD

**Spacetime Metrics** (in `src/metric/`):
- `AbstractMetric{Dim}` → `MinkowskiMetric`, `SchwarzschildMetric`, `KerrMetric`

**AMR** (in `src/amr/`):
- Block-structured AMR with Berger-Colella flux correction
- `AMRBlock`, `AMRGrid`, `GradientRefinement`, `CurrentSheetRefinement`
- Multi-rate subcycling via `SubcyclingScheme`

**Multi-physics Coupling** (in `src/coupling/`):
- `LieTrotterSplitting`, `StrangSplitting`
- `HyperbolicOperator`, `SourceOperator`, `CoupledProblem`

**Ghost-cell padding**: 2 cells per side (or `nghost(recon)` for WENO5 = 3). Interior cell `(ix,iy)` maps to `U[ix+2, iy+2]` in the padded array.

## Test Organization

Tests run via `test/runtests.jl` using the `safe_include` pattern: each test file runs in a fresh anonymous module (`gensym()`) to prevent name collisions and allow independent failure.

**Test categories**:
- Unit tests (in `test/`): geometry, conditions, robin, problem, equations, schemes, advanced_bcs, physics, coordinate_systems, dashboard, remake, reactive_euler, and all hyperbolic solver tests (1D/2D/3D Euler, MHD, Navier-Stokes, SRMHD, GRMHD, AMR, WENO, IMEX, unstructured, coupling, performance, advanced_numerics, extended_physics)
- Literate tutorials (14 scripts from `docs/src/literate_tutorials/`)
- Custom templates (5 scripts from `docs/src/literate_wyos/`)
- Verification suite (19 scripts from `docs/src/literate_verification/`)
- Quality: Aqua (with `ambiguities=false, project_extras=false, unbound_args=false`) and Explicit Imports

Known pre-existing failures are documented in `test/KNOWN_FAILURES.md`.

## Package Extension

`FVMDashboardExt` (in `ext/FVMDashboardExt.jl`) is a Julia package extension triggered by weak deps `HTTP` and `JSON3`. It implements `serve_dashboard`, `export_session`, and `import_session`. The dashboard data types and callbacks live in `src/dashboard/`; the extension provides the HTTP server and JSON serialization.

## Key Integration Points

- **DelaunayTriangulation.jl** - Mesh representation
- **SciMLBase.jl/DifferentialEquations.jl** - Problem types and solvers (`SciMLBase.remake` supported)
- **NonlinearSolve.jl** - Steady-state solvers
- **LinearSolve.jl** - Linear system solvers for templates
- **StaticArrays.jl** - SVector for conserved variable tuples in hyperbolic solver
- **PreallocationTools.jl** - DiffCache for AD compatibility
- **ChunkSplitters.jl** - Thread-parallel loop decomposition
- **HTTP.jl + JSON3.jl** - Weak deps for `FVMDashboardExt` package extension (`serve_dashboard`)
