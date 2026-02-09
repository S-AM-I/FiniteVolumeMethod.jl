# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

FiniteVolumeMethod.jl is a Julia package for solving partial differential equations using the finite volume method on triangular meshes. It solves the general PDE: ∂u/∂t + ∇·q(u) = S(u) with various boundary conditions.

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
- `FVMGeometry` - Main mesh structure wrapping DelaunayTriangulation.jl
- `TriangleProperties` - Pre-computed geometric properties per triangle

**Conditions** (in `src/conditions.jl` and `src/conditions/`):
- `BoundaryConditions` - Boundary condition specifications
- `InternalConditions` - Internal node constraints
- `ConditionType` enum: `Neumann`, `Dirichlet`, `Dudt`, `Constrained`, `Robin`
- Advanced: `NonlinearDirichlet`, `NonlinearNeumann`, `NonlinearRobin`, `PeriodicBC`, `CoupledBC`

### Module Structure

```
src/
├── FiniteVolumeMethod.jl   # Main module, exports
├── geometry.jl             # FVMGeometry mesh wrapper
├── conditions.jl           # Core BC types
├── conditions/             # Advanced BCs (nonlinear, periodic, coupled)
├── problem.jl              # Problem definitions
├── solve.jl                # fvm_eqs!, jacobian_sparsity, threading
├── equations/              # Core FVM discretization
│   ├── main_equations.jl   # Main equation assembly
│   ├── triangle_contributions.jl
│   ├── boundary_edge_contributions.jl
│   └── dirichlet.jl
├── specific_problems/      # Optimized templates (DiffusionEquation, etc.)
├── schemes/                # MUSCL, gradient reconstruction, limiters
└── physics/                # Domain-specific models (turbulence)
```

### Key Function Signatures

**Flux function**: `q(x, y, t, α, β, γ, p) → (qx, qy)` where `(α, β, γ)` are shape function coefficients

**Diffusion shortcut**: `D(x, y, t, u, p)` - auto-converted to flux

**Boundary condition**: `(x, y, t, u, p) → value` or `(x, y, t, u, p) → (a=, b=, c=)` for Robin

### Pipeline

1. Create mesh: `FVMGeometry(triangulate(...))`
2. Define BCs: `BoundaryConditions(mesh, bc_fn, Dirichlet)`
3. Create problem: `FVMProblem(mesh, BCs; diffusion_function=..., initial_condition=..., final_time=...)`
4. Solve: `solve(prob, Tsit5())` using DifferentialEquations.jl

### Threading

Multi-threading via `Threads.nthreads()` with ChunkSplitters. Thread-safe temporaries use PreallocationTools.DiffCache.

### Hyperbolic Solver Framework (Cell-Centered FVM)

A separate cell-centered finite volume solver for hyperbolic conservation laws on structured Cartesian meshes. Uses explicit time integration (forward Euler, SSP-RK3) with Godunov-type Riemann solvers.

**Mesh** (in `src/mesh/`):
- `AbstractMesh{Dim}` with `StructuredMesh1D`, `StructuredMesh2D`, `StructuredMesh3D`

**EOS** (in `src/eos/`):
- `AbstractEOS` → `IdealGasEOS`, `StiffenedGasEOS`

**Conservation Laws** (in `src/hyperbolic/`):
- `AbstractConservationLaw{Dim}` interface: `nvariables`, `physical_flux`, `max_wave_speed`, `wave_speeds`, `conserved_to_primitive`, `primitive_to_conserved`
- `EulerEquations{Dim,EOS}` — 3/4/5 variables (1D/2D/3D)
- `IdealMHDEquations{Dim,EOS}` — 8 variables [ρ,ρv,E,B]
- `NavierStokesEquations{Dim,EOS}` — wraps Euler + viscosity
- `SRMHDEquations{Dim,EOS}` — 8 variables, relativistic con2prim
- `GRMHDEquations{Dim,EOS,Metric}` — Valencia formulation + geometric source terms

**Riemann Solvers**: `LaxFriedrichsSolver`, `HLLSolver`, `HLLCSolver`, `HLLDSolver`

**Reconstruction**: `NoReconstruction`, `CellCenteredMUSCL`, `WENO3`, `WENO5`, `CharacteristicWENO`

**Time Integration**: Forward Euler, SSP-RK3, IMEX-RK (`IMEX_SSP3_433`, `IMEX_ARS222`, `IMEX_Midpoint`)

**Constrained Transport** (in `src/constrained_transport/`):
- `CTData2D`/`CTData3D` — face-centered B, edge-centered EMF
- Guarantees div(B) = 0 to machine precision for MHD

**Spacetime Metrics** (in `src/metric/`):
- `AbstractMetric{Dim}` → `MinkowskiMetric`, `SchwarzschildMetric`, `KerrMetric`

**AMR** (in `src/amr/`):
- Block-structured AMR with Berger-Colella flux correction
- `AMRBlock`, `AMRGrid`, `GradientRefinement`, `CurrentSheetRefinement`

**Ghost-cell padding**: 2 cells per side (or `nghost(recon)` for WENO5 = 3). Interior cell `(ix,iy)` maps to `U[ix+2, iy+2]` in the padded array.

## Key Integration Points

- **DelaunayTriangulation.jl** - Mesh representation
- **SciMLBase.jl/DifferentialEquations.jl** - Problem types and solvers
- **NonlinearSolve.jl** - Steady-state solvers
- **LinearSolve.jl** - Linear system solvers for templates

