# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

FiniteVolumeMethod.jl is a Julia package for solving partial differential equations using the finite volume method. It provides two solver families:

1. **Triangular (vertex-centered) FVM** for parabolic/elliptic PDEs on unstructured meshes via DelaunayTriangulation.jl
2. **Cell-centered FVM** for hyperbolic conservation laws on structured/unstructured grids

Version: `1.2.0`. Codebase: ~109 source files (~23k lines), 32 test files (~15k lines).

## Commands

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run a specific test file
julia --project test/geometry.jl

# Format code (uses Runic)
julia --project -e 'using Runic; Runic.format(".")'

# Build documentation locally
julia --project=docs docs/make.jl

# Run quality checks
julia --project -e 'using Aqua; Aqua.test_all(FiniteVolumeMethod)'
```

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
├── FiniteVolumeMethod.jl   # Main module, exports
├── coordinate_systems.jl   # AbstractCoordinateSystem hierarchy
├── geometry.jl             # FVMGeometry mesh wrapper
├── conditions.jl           # Core BC types
├── conditions/             # Advanced BCs (nonlinear, periodic, coupled)
├── problem.jl              # Problem definitions
├── solve.jl                # fvm_eqs!, jacobian_sparsity, threading
├── utils.jl                # Utilities
├── equations/              # Core FVM discretization
│   ├── main_equations.jl
│   ├── triangle_contributions.jl
│   ├── boundary_edge_contributions.jl
│   ├── individual_flux_contributions.jl
│   ├── control_volumes.jl
│   ├── shape_functions.jl
│   ├── source_contributions.jl
│   └── dirichlet.jl
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
└── coupling/               # Multi-physics operator splitting
```

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

Tests run via `test/runtests.jl` with 30+ testsets covering: Geometry, Conditions, Robin BCs, Problem, Equations, Schemes, Advanced BCs, Physics Models, Hyperbolic (1D/2D/3D), MHD (1D/2D/3D + CT), Navier-Stokes, SRMHD (1D/2D), GRMHD (1D/2D), AMR, WENO, IMEX, Unstructured Hyperbolic, Multi-Physics Coupling, Performance & Threading, Advanced Numerics, Extended Physics, README, Tutorials (14 Literate scripts), Custom Templates (5 wyos scripts), Aqua, Explicit Imports.

**Note**: `test/test_coordinate_systems.jl` exists (11 tests) but is not yet included in `runtests.jl`.

## Key Integration Points

- **DelaunayTriangulation.jl** - Mesh representation
- **SciMLBase.jl/DifferentialEquations.jl** - Problem types and solvers
- **NonlinearSolve.jl** - Steady-state solvers
- **LinearSolve.jl** - Linear system solvers for templates
- **StaticArrays.jl** - SVector for conserved variable tuples in hyperbolic solver
- **PreallocationTools.jl** - DiffCache for AD compatibility
- **ChunkSplitters.jl** - Thread-parallel loop decomposition
