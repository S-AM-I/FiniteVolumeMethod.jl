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

## Key Integration Points

- **DelaunayTriangulation.jl** - Mesh representation
- **SciMLBase.jl/DifferentialEquations.jl** - Problem types and solvers
- **NonlinearSolve.jl** - Steady-state solvers
- **LinearSolve.jl** - Linear system solvers for templates
