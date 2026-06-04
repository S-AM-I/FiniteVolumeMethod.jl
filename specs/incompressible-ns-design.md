---
date: 2026-04-06
---

# Phase 1: Incompressible Navier-Stokes — SIMPLE/PISO/PIMPLE

**Status**: Design
**Depends on**: Phase 0 (Collocated Cell-Centered Operators) — complete

## Goal

Implement pressure-velocity coupling algorithms for incompressible flow on `UnstructuredFVMMesh`, producing `LinearProblem` sub-solves compatible with LinearSolve.jl. This is OpenFOAM's core — every industrial CFD feature (turbulence, multiphase, combustion) builds on top of it.

## Architecture

### File Layout

All files in `src/incompressible/`:

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `types.jl` | Problem, state, algorithm config types | ~150 |
| `momentum.jl` | Momentum predictor assembly | ~120 |
| `pressure.jl` | Pressure Poisson equation + Rhie-Chow flux | ~130 |
| `correction.jl` | Velocity and flux correction | ~80 |
| `simple.jl` | SIMPLE steady-state loop | ~120 |
| `piso.jl` | PISO transient corrector loop | ~100 |
| `pimple.jl` | PIMPLE hybrid loop | ~100 |
| `boundary_conditions.jl` | Incompressible-specific BCs | ~100 |

Wired into Layer 2 (`src/layers/discretization_assembly_kernels.jl`) since it depends on Phase 0 operators from Layer 1.

### Type Design

```julia
# ── Algorithm configuration ──────────────────────────────────

abstract type AbstractPVCoupling end

struct SIMPLE{T} <: AbstractPVCoupling
    alpha_U::T          # velocity under-relaxation (default 0.7)
    alpha_p::T          # pressure under-relaxation (default 0.3)
    max_iterations::Int # outer loop limit (default 1000)
    tolerance::T        # residual convergence threshold (default 1e-6)
end

struct PISO{T} <: AbstractPVCoupling
    n_correctors::Int   # pressure correction steps (default 2)
end

struct PIMPLE{T} <: AbstractPVCoupling
    n_outer::Int        # outer corrections (SIMPLE-like, default 2)
    n_correctors::Int   # inner corrections (PISO-like, default 1)
    alpha_U::T          # velocity under-relaxation for outer loop
    alpha_p::T          # pressure under-relaxation for outer loop
    tolerance::T        # outer loop convergence threshold
end

# ── Problem definition ───────────────────────────────────────

struct IncompressibleProblem{Dim, T, Mesh, BC, Algo <: AbstractPVCoupling}
    mesh::Mesh
    bcs::BC                 # Dict{Symbol, AbstractBoundaryCondition} per field
    algorithm::Algo
    nu::T                   # kinematic viscosity (scalar; variable viscosity via Phase 2)
    density::T              # reference density (default 1.0 for incompressible)
end

# ── Flow state ───────────────────────────────────────────────

mutable struct IncompressibleState{Dim, T}
    U::CollocatedVectorField{Dim, T}    # cell-centered velocity
    p::CollocatedScalarField{T}         # cell-centered pressure
    phi::FaceFluxField{T}               # face volumetric flux (U_f . S_f)
    A_P::Vector{T}                      # momentum diagonal coefficients (for Rhie-Chow)
    H_U::Vector{SVector{Dim, T}}        # H(U) operator (off-diagonal + source)
end

# ── Convergence result ───────────────────────────────────────

struct SolveResult{T}
    converged::Bool
    iterations::Int
    residuals::Dict{Symbol, Vector{T}}  # per-field residual history
    state::IncompressibleState
end
```

### Key Design Decisions

**1. State is mutable, equations are rebuilt each iteration.**
`IncompressibleState` holds the current fields. Each SIMPLE/PISO iteration creates fresh `CollocatedEquation`s via `reset!()` and reassembles. This matches OpenFOAM's pattern and avoids stale matrix entries.

**2. `A_P` and `H_U` stored in state for Rhie-Chow.**
The momentum equation diagonal `A_P[c]` and off-diagonal operator `H(U)` are needed by both the pressure equation and velocity correction. Computing them once in the momentum predictor and storing them in state avoids redundant assembly.

**3. BCs are `Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}`.**
Outer key is field name (`:U`, `:p`), inner key is patch name. This allows different BC types per field per patch, matching OpenFOAM's `0/U`, `0/p` pattern. Reuses existing `ParabolicDirichlet`, `ParabolicNeumann`, `ParabolicRobin` from `src/parabolic/types.jl`.

**4. Each sub-solve produces `LinearProblem`.**
Momentum predictor assembles one `CollocatedEquation` per velocity component → `to_linear_problem()` → `solve(prob, algorithm)`. Pressure equation does the same. Users can pass any LinearSolve.jl algorithm.

**5. No new dependencies.**
Uses existing SparseArrays, LinearAlgebra, StaticArrays, SciMLBase. LinearSolve.jl is already accessible. No new packages needed.

## Algorithm Details

### SIMPLE (Steady-State)

```
Initialize U, p, phi
for iter = 1:max_iterations
    1. Assemble momentum: A_U * U* = H(U) - grad(p)
       - Convection: assemble_convection!(eq, phi, mesh, bcs; scheme=CONV_UPWIND)
       - Diffusion:  assemble_laplacian!(eq, nu, mesh, bcs)  [negative sign]
       - Pressure:   explicit grad(p) → RHS
       - Store A_P = diag(A_U) BEFORE under-relaxation (needed by pressure eq)
       - Under-relax: A_U[c,c] /= alpha_U; b[c] += (1-alpha_U)/alpha_U * A_U[c,c] * U_old[c]
    2. Solve momentum → U* (per component)
    3. Compute H(U) using stored A_P: H_c = b_c - sum_{N!=P} A_{P,N} * U_N
    4. Assemble pressure: div(1/A_P * grad(p)) = div(H(U)/A_P)
       - LHS: Laplacian with gamma = V_P/A_P at faces
       - RHS: face flux of H(U)/A_P via Rhie-Chow
    5. Solve pressure → p'
    6. Under-relax pressure: p = p_old + alpha_p * p'
    7. Correct velocity: U = H(U)/A_P - (1/A_P) * grad(p)
    8. Update face fluxes: phi via Rhie-Chow with new p
    9. Check convergence (L2 norm of momentum + continuity residuals)
end
```

### PISO (Transient)

```
for each time step:
    1. Assemble + solve momentum predictor (no under-relaxation)
    2. for k = 1:n_correctors
        a. Assemble pressure equation from current H(U)
        b. Solve pressure → p
        c. Correct velocity + fluxes
        d. Recompute H(U) if k < n_correctors
    end
    3. Advance time
```

### PIMPLE (Hybrid)

```
for each time step:
    for outer = 1:n_outer
        1. Assemble + solve momentum (with under-relaxation if outer > 1)
        2. for k = 1:n_correctors
            a. Pressure solve + correction (PISO inner loop)
        end
        3. Check outer convergence; break if converged
    end
    Advance time
```

## Momentum Predictor

The momentum equation for incompressible flow:

```
dU/dt + div(phi * U) = div(nu * grad(U)) - grad(p) + sources
```

Discretized per velocity component `U_i`:

```julia
function assemble_momentum!(
    eq::CollocatedEquation{T},           # one per component
    state::IncompressibleState{Dim, T},
    prob::IncompressibleProblem{Dim, T},
    component::Int;                       # 1, 2, or 3
    dt::Union{Nothing, T} = nothing,      # nothing for steady SIMPLE
    scheme::ConvectionScheme = CONV_UPWIND,
)
    mesh = prob.mesh
    bcs_U = prob.bcs[:U]

    # Convection: div(phi * U_component)
    assemble_convection!(eq, state.phi, mesh, bcs_U; scheme)

    # Diffusion: -div(nu * grad(U_component))  [negative → subtract]
    assemble_laplacian!(eq, prob.nu, mesh, bcs_U)
    # Laplacian adds +div(gamma*grad) to diagonal, which is correct sign
    # for diffusion (both sides of equation)

    # Temporal term (transient only)
    if dt !== nothing
        phi_old = _extract_component(state.U, component)
        assemble_ddt_euler!(eq, prob.density, phi_old, mesh, dt)
    end

    # Pressure gradient source: -grad(p) → explicit RHS
    grad_p = gradient(state.p, mesh)
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        eq.b[c] -= grad_p[c][component] * mesh.cell_volumes[c]
    end
end
```

After solving, extract `A_P` and `H(U)`:

```julia
function extract_momentum_operators!(state, eq_components, mesh)
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        state.A_P[c] = eq_components[1].A[c, c]  # same diagonal for all components
    end
    # H(U) = sum of off-diagonal contributions + RHS
    # H_i = b_i - sum_{N != P} A_{P,N} * U_N  (per component)
    # Simplified: H = (A * U - diag(A) * U) + b  →  H = b - offdiag(A) * U
    for c in 1:nc
        h = zero(SVector{Dim, T})
        for (d, eq) in enumerate(eq_components)
            u_d = _extract_component(state.U, d)
            h_d = eq.b[c]
            for idx in nzrange(eq.A, c)
                row = eq.A.rowval[idx]
                if row != c
                    h_d -= eq.A.nzval[idx] * u_d[row]
                end
            end
            h = setindex(h, h_d, d)
        end
        state.H_U[c] = h
    end
end
```

## Pressure Equation

The pressure Poisson equation enforces continuity:

```
div(1/A_P * grad(p)) = div(H(U)/A_P)
```

LHS: Laplacian with face diffusivity `D_f = V_f / A_{P,f}` (interpolated).
RHS: Rhie-Chow face flux of `H(U)/A_P`.

```julia
function assemble_pressure!(
    eq::CollocatedEquation{T},
    state::IncompressibleState{Dim, T},
    prob::IncompressibleProblem{Dim, T},
)
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    # Diffusivity for pressure Laplacian: D_P = V_P / A_P
    D = Vector{T}(undef, nc)
    for c in 1:nc
        D[c] = mesh.cell_volumes[c] / state.A_P[c]
    end

    # LHS: div(D * grad(p))
    assemble_laplacian!(eq, D, mesh, prob.bcs[:p])

    # RHS: div(phi_HbyA) where phi_HbyA is the Rhie-Chow flux of H(U)/A_P
    phi_HbyA = compute_HbyA_flux(state, mesh)
    for f in 1:size(mesh.face_cells, 2)
        P = owner(mesh, f)
        eq.b[P] -= phi_HbyA[f]  # negative because div goes to LHS
        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            eq.b[N] += phi_HbyA[f]
        end
    end
end
```

## Velocity Correction

After solving for pressure:

```julia
function correct_velocity!(
    state::IncompressibleState{Dim, T},
    mesh::UnstructuredFVMMesh{Dim, T},
)
    nc = length(mesh.cell_volumes)
    grad_p = gradient(state.p, mesh)
    for c in 1:nc
        state.U.internal[c] = state.H_U[c] / state.A_P[c] -
            mesh.cell_volumes[c] / state.A_P[c] * grad_p[c]
    end
    # Update face fluxes with Rhie-Chow
    rhie_chow_correction!(state.phi, state.U, state.p, state.A_P, mesh)
end
```

## Boundary Conditions

Reuses existing BC types. Incompressible-specific convenience constructors:

| BC Name | Velocity | Pressure | Use Case |
|---------|----------|----------|----------|
| `FixedVelocity` | `ParabolicDirichlet(value)` | `ParabolicNeumann(0)` | Inlet, moving wall |
| `FixedPressure` | `ParabolicNeumann(0)` | `ParabolicDirichlet(value)` | Outlet |
| `NoSlipWall` | `ParabolicDirichlet(0)` | `ParabolicNeumann(0)` | Stationary wall |
| `SlipWall` | Tangential projection | `ParabolicNeumann(0)` | Symmetry plane |
| `InletOutlet` | Switches Dirichlet/Neumann on flow direction | `ParabolicNeumann(0)` | Open boundary |

```julia
struct FixedVelocityBC{Dim, T} <: AbstractBoundaryCondition
    value::SVector{Dim, T}
end

struct FixedPressureBC{T} <: AbstractBoundaryCondition
    value::T
end

struct NoSlipWallBC <: AbstractBoundaryCondition end

struct SlipWallBC <: AbstractBoundaryCondition end

struct InletOutletBC{Dim, T} <: AbstractBoundaryCondition
    inlet_value::SVector{Dim, T}
end
```

These are translated to `ParabolicDirichlet`/`ParabolicNeumann` pairs at problem construction time via `expand_bcs(prob)` which returns the per-field BC dicts expected by Phase 0 operators.

## SciML Integration

### Steady SIMPLE

```julia
function solve_simple(prob::IncompressibleProblem{Dim, T, Mesh, BC, SIMPLE{T}};
    linear_solver = nothing,  # default: LU for small, GMRES for large
) -> SolveResult{T}
```

Returns `SolveResult` with converged state. No ODE wrapper — it's a fixed-point iteration, analogous to `SteadyFVMProblem`.

### Transient PISO/PIMPLE

```julia
function solve_incompressible(
    prob::IncompressibleProblem,
    tspan::Tuple{T, T},
    dt::T;
    save_every::Int = 1,
    linear_solver = nothing,
) -> Vector{IncompressibleState}
```

Explicit time loop with `LinearProblem` sub-solves at each step. Returns saved states. For advanced time integration (adaptive dt, CFL control), a future version can wrap this as a discrete `ODEProblem` with callbacks.

### Sub-solve dispatch

Every internal linear solve goes through:
```julia
sol = solve(to_linear_problem(eq), linear_solver)
```
where `linear_solver` defaults to `nothing` (LinearSolve.jl picks automatically) or the user passes e.g. `KrylovJL_GMRES()`.

## Residual Monitoring

Each iteration computes:
- **Momentum residual**: `||A_U * U - b||_2 / ||b||_2` (per component, take max)
- **Continuity residual**: `||div(phi)||_1` (sum of absolute face flux imbalance per cell)

Both are stored in `SolveResult.residuals` for convergence plotting.

## Validation Target

**Lid-driven cavity, Re = 100** (Ghia et al. 1982):
- Square domain [0,1]^2
- Top wall: U = (1, 0), other walls: no-slip
- Pressure: zero-gradient everywhere, fixed reference at one cell
- Compare u-velocity along vertical centerline and v-velocity along horizontal centerline
- Pass criterion: max deviation < 2% from Ghia tabulated data on 32x32 mesh

## Export List

```julia
# Types
export IncompressibleProblem, IncompressibleState, SolveResult
export SIMPLE, PISO, PIMPLE
export AbstractPVCoupling

# BCs
export FixedVelocityBC, FixedPressureBC, NoSlipWallBC, SlipWallBC, InletOutletBC

# Solvers
export solve_simple, solve_incompressible

# Assembly (for advanced users)
export assemble_momentum!, assemble_pressure!, correct_velocity!
export extract_momentum_operators!
```
