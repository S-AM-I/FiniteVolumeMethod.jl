# Incompressible Navier-Stokes (SIMPLE/PISO/PIMPLE) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement pressure-velocity coupling algorithms (SIMPLE/PISO/PIMPLE) for incompressible flow on `UnstructuredFVMMesh`, producing `LinearProblem` sub-solves compatible with LinearSolve.jl.

**Architecture:** Eight files in `src/incompressible/` wired into Layer 2. Types and BCs first, then momentum/pressure/correction building blocks, then solver loops (SIMPLE → PISO → PIMPLE). Each sub-solve assembles a `CollocatedEquation` and dispatches to `solve(to_linear_problem(eq), solver)`. Tests in `test/incompressible.jl`.

**Tech Stack:** Julia, SparseArrays, LinearAlgebra, StaticArrays, SciMLBase (LinearProblem), Phase 0 collocated operators (CollocatedEquation, assemble_laplacian!, assemble_convection!, gradient, rhie_chow_correction!)

---

## File Map

| File | Purpose | Creates/Modifies |
|------|---------|-----------------|
| `src/incompressible/types.jl` | Problem, state, algorithm config, result types | Create |
| `src/incompressible/boundary_conditions.jl` | Incompressible-specific BC types + expansion | Create |
| `src/incompressible/momentum.jl` | Momentum predictor assembly + A_P/H(U) extraction | Create |
| `src/incompressible/pressure.jl` | Pressure Poisson equation + HbyA flux | Create |
| `src/incompressible/correction.jl` | Velocity/flux correction | Create |
| `src/incompressible/residuals.jl` | Convergence monitoring (momentum + continuity) | Create |
| `src/incompressible/simple.jl` | SIMPLE steady-state loop | Create |
| `src/incompressible/piso.jl` | PISO transient loop | Create |
| `src/incompressible/pimple.jl` | PIMPLE hybrid loop | Create |
| `src/layers/discretization_assembly_kernels.jl` | Wire incompressible includes | Modify (append) |
| `src/FiniteVolumeMethod.jl` | Add exports | Modify (add export block) |
| `test/incompressible.jl` | All tests | Create |
| `test/runtests.jl` | Register test | Modify (add safe_include) |
| `validation/manifest.toml` | Register features | Modify (append) |

---

### Task 1: Create types.jl — Algorithm configs, problem, state, result

**Files:**
- Create: `src/incompressible/types.jl`

- [ ] **Step 1: Create the directory and types file**

```bash
mkdir -p src/incompressible
```

Write `src/incompressible/types.jl`:

```julia
# incompressible/types.jl — Core types for incompressible Navier-Stokes solvers
#
# Defines algorithm configuration (SIMPLE/PISO/PIMPLE), problem definition,
# mutable flow state, and convergence result types.

using SparseArrays: nzrange

# ── Algorithm configuration ──────────────────────────────────

"""
    AbstractPVCoupling

Abstract supertype for pressure-velocity coupling algorithms.
"""
abstract type AbstractPVCoupling end

"""
    SIMPLE{T} <: AbstractPVCoupling

Semi-Implicit Method for Pressure-Linked Equations (steady-state).

# Fields
- `alpha_U` — velocity under-relaxation factor (default 0.7)
- `alpha_p` — pressure under-relaxation factor (default 0.3)
- `max_iterations` — outer loop iteration limit (default 1000)
- `tolerance` — residual convergence threshold (default 1e-6)
"""
struct SIMPLE{T} <: AbstractPVCoupling
    alpha_U::T
    alpha_p::T
    max_iterations::Int
    tolerance::T
end

function SIMPLE(; alpha_U = 0.7, alpha_p = 0.3, max_iterations = 1000, tolerance = 1e-6)
    T = promote_type(typeof(alpha_U), typeof(alpha_p), typeof(tolerance))
    return SIMPLE{T}(T(alpha_U), T(alpha_p), max_iterations, T(tolerance))
end

"""
    PISO{T} <: AbstractPVCoupling

Pressure-Implicit with Splitting of Operators (transient).

# Fields
- `n_correctors` — number of pressure correction steps per time step (default 2)
"""
struct PISO{T} <: AbstractPVCoupling
    n_correctors::Int
end

PISO(; n_correctors = 2) = PISO{Float64}(n_correctors)

"""
    PIMPLE{T} <: AbstractPVCoupling

Merged PISO-SIMPLE algorithm (transient with outer corrections).

# Fields
- `n_outer` — outer correction loops (SIMPLE-like, default 2)
- `n_correctors` — inner pressure corrections (PISO-like, default 1)
- `alpha_U` — velocity under-relaxation for non-final outer iterations
- `alpha_p` — pressure under-relaxation for non-final outer iterations
- `tolerance` — outer loop convergence threshold
"""
struct PIMPLE{T} <: AbstractPVCoupling
    n_outer::Int
    n_correctors::Int
    alpha_U::T
    alpha_p::T
    tolerance::T
end

function PIMPLE(;
        n_outer = 2, n_correctors = 1,
        alpha_U = 0.7, alpha_p = 0.3, tolerance = 1e-6,
    )
    T = promote_type(typeof(alpha_U), typeof(alpha_p), typeof(tolerance))
    return PIMPLE{T}(n_outer, n_correctors, T(alpha_U), T(alpha_p), T(tolerance))
end

# ── Problem definition ───────────────────────────────────────

"""
    IncompressibleProblem{Dim, T, Mesh, BC, Algo}

Defines an incompressible flow problem on an unstructured mesh.

# Fields
- `mesh` — `UnstructuredFVMMesh{Dim, T}`
- `bcs` — boundary conditions: `Dict{Symbol, <:AbstractBoundaryCondition}` per patch
- `algorithm` — pressure-velocity coupling algorithm
- `nu` — kinematic viscosity (scalar)
- `density` — reference density (default 1.0)
"""
struct IncompressibleProblem{Dim, T, Mesh, BC, Algo <: AbstractPVCoupling}
    mesh::Mesh
    bcs::BC
    algorithm::Algo
    nu::T
    density::T
end

function IncompressibleProblem(
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs,
        algorithm::AbstractPVCoupling;
        nu::T = T(1e-3),
        density::T = one(T),
    ) where {Dim, T}
    return IncompressibleProblem{Dim, T, typeof(mesh), typeof(bcs), typeof(algorithm)}(
        mesh, bcs, algorithm, nu, density,
    )
end

# ── Flow state ───────────────────────────────────────────────

"""
    IncompressibleState{Dim, T}

Mutable flow state for incompressible solvers.

# Fields
- `U` — cell-centered velocity (`CollocatedVectorField`)
- `p` — cell-centered pressure (`CollocatedScalarField`)
- `phi` — face volumetric flux (`FaceFluxField`)
- `A_P` — momentum equation diagonal coefficients (length `ncells`)
- `H_U` — H(U) operator: off-diagonal + RHS per cell (length `ncells`)
"""
mutable struct IncompressibleState{Dim, T}
    U::CollocatedVectorField{Dim, T}
    p::CollocatedScalarField{T}
    phi::FaceFluxField{T}
    A_P::Vector{T}
    H_U::Vector{SVector{Dim, T}}
end

"""
    IncompressibleState(mesh::UnstructuredFVMMesh{Dim, T})

Construct a zero-initialized flow state on `mesh`.
"""
function IncompressibleState(mesh::UnstructuredFVMMesh{Dim, T}) where {Dim, T}
    nc = length(mesh.cell_volumes)
    U = CollocatedVectorField(:U, mesh)
    p = CollocatedScalarField(:p, mesh)
    phi = FaceFluxField(:phi, mesh)
    A_P = ones(T, nc)  # initialized to 1 to avoid division by zero before first solve
    H_U = fill(zero(SVector{Dim, T}), nc)
    return IncompressibleState{Dim, T}(U, p, phi, A_P, H_U)
end

# ── Result ───────────────────────────────────────────────────

"""
    SolveResult{Dim, T}

Result from an incompressible flow solve.

# Fields
- `converged` — whether the solver reached the convergence criterion
- `iterations` — number of outer iterations completed
- `residuals` — per-field residual history (`Dict{Symbol, Vector{T}}`)
- `state` — final `IncompressibleState`
"""
struct SolveResult{Dim, T}
    converged::Bool
    iterations::Int
    residuals::Dict{Symbol, Vector{T}}
    state::IncompressibleState{Dim, T}
end

# ── Helpers ──────────────────────────────────────────────────

"""
    _extract_component(U::CollocatedVectorField{Dim, T}, d::Int) -> Vector{T}

Extract component `d` of the vector field as a plain `Vector{T}`.
"""
function _extract_component(U::CollocatedVectorField{Dim, T}, d::Int) where {Dim, T}
    return T[U.internal[c][d] for c in eachindex(U.internal)]
end

"""
    _set_component!(U::CollocatedVectorField{Dim, T}, d::Int, vals::Vector{T})

Set component `d` of the vector field from a plain `Vector{T}`.
"""
function _set_component!(
        U::CollocatedVectorField{Dim, T}, d::Int, vals::Vector{T},
    ) where {Dim, T}
    for c in eachindex(U.internal)
        v = U.internal[c]
        U.internal[c] = setindex(v, vals[c], d)
    end
    return nothing
end
```

- [ ] **Step 2: Verify the file parses**

Run:
```bash
julia --project -e 'include("src/FiniteVolumeMethod.jl")'
```
This will fail because we haven't wired it in yet — that's expected at this stage. Just verify the file has no syntax errors:
```bash
julia --project -e 'using FiniteVolumeMethod; include("src/incompressible/types.jl")'
```
Expected: No errors (types defined in the context of the parent module).

---

### Task 2: Create boundary_conditions.jl — Incompressible BC types

**Files:**
- Create: `src/incompressible/boundary_conditions.jl`

- [ ] **Step 1: Write the boundary conditions file**

```julia
# incompressible/boundary_conditions.jl — Boundary condition types for incompressible flow
#
# Convenience BC types that expand into per-field ParabolicDirichlet/ParabolicNeumann
# pairs for the Phase 0 collocated operators.

# ── Incompressible-specific BC types ─────────────────────────

"""
    FixedVelocityBC{Dim, T} <: AbstractBoundaryCondition

Fixed velocity at a boundary patch (inlet, moving wall).
Expands to: velocity = Dirichlet(value), pressure = Neumann(0).
"""
struct FixedVelocityBC{Dim, T} <: AbstractBoundaryCondition
    value::SVector{Dim, T}
end

function FixedVelocityBC(vals::NTuple{N, T}) where {N, T}
    return FixedVelocityBC{N, T}(SVector{N, T}(vals))
end

"""
    FixedPressureBC{T} <: AbstractBoundaryCondition

Fixed pressure at a boundary patch (outlet).
Expands to: velocity = Neumann(0), pressure = Dirichlet(value).
"""
struct FixedPressureBC{T} <: AbstractBoundaryCondition
    value::T
end

"""
    NoSlipWallBC <: AbstractBoundaryCondition

No-slip stationary wall.
Expands to: velocity = Dirichlet(0), pressure = Neumann(0).
"""
struct NoSlipWallBC <: AbstractBoundaryCondition end

"""
    SlipWallBC <: AbstractBoundaryCondition

Slip wall / symmetry plane.
Expands to: velocity = Neumann(0) with tangential projection, pressure = Neumann(0).
For now, treated as zero-gradient for both fields (symmetry approximation).
"""
struct SlipWallBC <: AbstractBoundaryCondition end

"""
    InletOutletBC{Dim, T} <: AbstractBoundaryCondition

Switches between Dirichlet (inflow) and Neumann (outflow) based on flux direction.
Expands to: velocity = Dirichlet(inlet_value) for inflow faces, Neumann(0) for outflow.
"""
struct InletOutletBC{Dim, T} <: AbstractBoundaryCondition
    inlet_value::SVector{Dim, T}
end

# ── BC expansion ─────────────────────────────────────────────

"""
    expand_velocity_bc(bc::AbstractBoundaryCondition, component::Int) -> AbstractBoundaryCondition

Convert an incompressible BC into a scalar BC for the velocity component `component`.
"""
expand_velocity_bc(bc::FixedVelocityBC, component::Int) = ParabolicDirichlet(bc.value[component])
expand_velocity_bc(::FixedPressureBC, ::Int) = ParabolicNeumann(0.0)
expand_velocity_bc(::NoSlipWallBC, ::Int) = ParabolicDirichlet(0.0)
expand_velocity_bc(::SlipWallBC, ::Int) = ParabolicNeumann(0.0)
expand_velocity_bc(bc::InletOutletBC, component::Int) = ParabolicDirichlet(bc.inlet_value[component])

"""
    expand_pressure_bc(bc::AbstractBoundaryCondition) -> AbstractBoundaryCondition

Convert an incompressible BC into a scalar BC for the pressure field.
"""
expand_pressure_bc(::FixedVelocityBC) = ParabolicNeumann(0.0)
expand_pressure_bc(bc::FixedPressureBC) = ParabolicDirichlet(bc.value)
expand_pressure_bc(::NoSlipWallBC) = ParabolicNeumann(0.0)
expand_pressure_bc(::SlipWallBC) = ParabolicNeumann(0.0)
expand_pressure_bc(::InletOutletBC) = ParabolicNeumann(0.0)

"""
    expand_bcs(bcs::Dict{Symbol, <:AbstractBoundaryCondition}, component::Int)

Expand incompressible BCs into scalar BCs for velocity component `component`.
Returns `Dict{Symbol, AbstractBoundaryCondition}`.
"""
function expand_bcs_velocity(
        bcs::Dict{Symbol, <:AbstractBoundaryCondition}, component::Int,
    )
    return Dict{Symbol, AbstractBoundaryCondition}(
        tag => expand_velocity_bc(bc, component) for (tag, bc) in bcs
    )
end

"""
    expand_bcs_pressure(bcs::Dict{Symbol, <:AbstractBoundaryCondition})

Expand incompressible BCs into scalar BCs for pressure.
Returns `Dict{Symbol, AbstractBoundaryCondition}`.
"""
function expand_bcs_pressure(bcs::Dict{Symbol, <:AbstractBoundaryCondition})
    return Dict{Symbol, AbstractBoundaryCondition}(
        tag => expand_pressure_bc(bc) for (tag, bc) in bcs
    )
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/incompressible/boundary_conditions.jl")'
```
Expected: No errors.

---

### Task 3: Create momentum.jl — Momentum predictor assembly

**Files:**
- Create: `src/incompressible/momentum.jl`

- [ ] **Step 1: Write the momentum predictor**

```julia
# incompressible/momentum.jl — Momentum predictor for incompressible flow
#
# Assembles the discretized momentum equation per velocity component:
#   A_U * U* = H(U) - grad(p)
# using Phase 0 collocated operators (convection, Laplacian, ddt).

"""
    assemble_momentum!(
        eq::CollocatedEquation{T},
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        component::Int;
        dt::Union{Nothing, T} = nothing,
        scheme::ConvectionScheme = CONV_UPWIND,
    )

Assemble the momentum equation for velocity component `component` into `eq`.

Adds convection `div(phi * U_d)`, diffusion `-div(nu * grad(U_d))`, and
the explicit pressure gradient to the RHS. For transient solves, also
adds the temporal discretization term.

# Arguments
- `eq` — equation (modified in-place)
- `state` — current flow state (provides phi, U, p)
- `prob` — problem definition (provides mesh, BCs, nu)
- `component` — velocity component index (1-based)
- `dt` — time step size (nothing for steady SIMPLE)
- `scheme` — convection interpolation scheme
"""
function assemble_momentum!(
        eq::CollocatedEquation{T},
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        component::Int;
        dt::Union{Nothing, T} = nothing,
        scheme::ConvectionScheme = CONV_UPWIND,
    ) where {Dim, T}
    mesh = prob.mesh
    bcs_U = expand_bcs_velocity(prob.bcs, component)

    # Convection: div(phi * U_component)
    assemble_convection!(eq, state.phi, mesh, bcs_U; scheme = scheme)

    # Diffusion: div(nu * grad(U_component)) — Laplacian with positive sign
    # The Laplacian operator already contributes with the correct sign:
    # +flux_coeff on diagonal, -flux_coeff on off-diagonal
    assemble_laplacian!(eq, prob.nu, mesh, bcs_U)

    # Temporal term (transient only)
    if dt !== nothing
        u_old = _extract_component(state.U, component)
        assemble_ddt_euler!(eq, prob.density, u_old, mesh, dt)
    end

    # Pressure gradient source: -grad(p) → explicit contribution to RHS
    grad_p = gradient(state.p, mesh)
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        eq.b[c] -= grad_p[c][component] * mesh.cell_volumes[c]
    end

    return nothing
end

"""
    extract_momentum_operators!(
        state::IncompressibleState{Dim, T},
        eqs::Vector{CollocatedEquation{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
    )

Extract diagonal coefficients `A_P` and the H(U) operator from the
assembled momentum equations. Must be called after `assemble_momentum!`
and before under-relaxation is applied.

`A_P[c]` is the diagonal coefficient of the momentum matrix at cell `c`.
`H_U[c]` is the vector formed from the RHS minus off-diagonal contributions:
  `H_d[c] = b[c] - sum_{N != c} A[c,N] * U_d[N]`
"""
function extract_momentum_operators!(
        state::IncompressibleState{Dim, T},
        eqs::Vector{CollocatedEquation{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)

    # Extract A_P from diagonal (same for all components in standard assembly)
    for c in 1:nc
        state.A_P[c] = eqs[1].A[c, c]
    end

    # Compute H(U) per cell
    for c in 1:nc
        h_components = zeros(SVector{Dim, T})
        for d in 1:Dim
            eq = eqs[d]
            u_d = _extract_component(state.U, d)

            # H_d = b_c - sum_{N!=c} A_{c,N} * U_d[N]
            h_d = eq.b[c]
            for idx in nzrange(eq.A, c)
                row = eq.A.rowval[idx]
                if row != c
                    h_d -= eq.A.nzval[idx] * u_d[row]
                end
            end

            h_components = setindex(h_components, h_d, d)
        end
        state.H_U[c] = h_components
    end

    return nothing
end

"""
    under_relax_momentum!(eq::CollocatedEquation{T}, U_old_d::Vector{T}, alpha_U::T)

Apply under-relaxation to the momentum equation:
  A[c,c] /= alpha_U
  b[c] += (1 - alpha_U) / alpha_U * A[c,c] * U_old_d[c]

Must be called AFTER `extract_momentum_operators!` has saved the
un-relaxed diagonal into `state.A_P`.
"""
function under_relax_momentum!(
        eq::CollocatedEquation{T}, U_old_d::Vector{T}, alpha_U::T,
    ) where {T}
    nc = length(eq.b)
    for c in 1:nc
        a_P = eq.A[c, c]
        eq.A[c, c] = a_P / alpha_U
        eq.b[c] += (one(T) - alpha_U) / alpha_U * a_P * U_old_d[c]
    end
    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/incompressible/types.jl"); include("src/incompressible/boundary_conditions.jl"); include("src/incompressible/momentum.jl")'
```
Expected: No errors.

---

### Task 4: Create pressure.jl — Pressure Poisson equation

**Files:**
- Create: `src/incompressible/pressure.jl`

- [ ] **Step 1: Write the pressure equation assembly**

```julia
# incompressible/pressure.jl — Pressure Poisson equation for incompressible flow
#
# Assembles: div(1/A_P * grad(p)) = div(H(U)/A_P)
# LHS: Laplacian with per-cell diffusivity D_P = V_P / A_P
# RHS: Divergence of the H(U)/A_P flux (Rhie-Chow consistent)

"""
    compute_HbyA_flux(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) -> Vector{T}

Compute the face flux of `H(U)/A_P` using Rhie-Chow-consistent interpolation.
Returns a vector of length `nfaces`.

For internal faces:
  phi_HbyA[f] = (H(U)/A_P)_f · S_f
where the face value is linearly interpolated from cell values.

For boundary faces: uses the boundary velocity directly.
"""
function compute_HbyA_flux(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    phi_HbyA = Vector{T}(undef, nf)

    ubmap = Dict(f => i for (i, f) in enumerate(state.U.boundary_face_indices))

    for f in 1:nf
        S_f = face_normal_area(mesh, f)

        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)

            # H(U)/A_P at cell centers
            HbyA_P = state.H_U[P] / state.A_P[P]
            HbyA_N = state.H_U[N] / state.A_P[N]

            # Linear interpolation to face
            HbyA_f = w * HbyA_P + (one(T) - w) * HbyA_N

            phi_HbyA[f] = dot(HbyA_f, S_f)
        else
            # Boundary: use boundary velocity
            bi = get(ubmap, f, nothing)
            if bi !== nothing
                U_f = state.U.boundary[bi]
                phi_HbyA[f] = dot(U_f, S_f)
            else
                phi_HbyA[f] = zero(T)
            end
        end
    end

    return phi_HbyA
end

"""
    assemble_pressure!(
        eq::CollocatedEquation{T},
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
    )

Assemble the pressure Poisson equation into `eq`:

    div(D * grad(p)) = div(phi_HbyA)

where `D[c] = V_c / A_P[c]` is the per-cell diffusivity derived from the
momentum equation diagonal.

# Arguments
- `eq` — equation (modified in-place)
- `state` — current flow state (provides A_P, H_U)
- `prob` — problem definition (provides mesh, BCs)
"""
function assemble_pressure!(
        eq::CollocatedEquation{T},
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
    ) where {Dim, T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)
    bcs_p = expand_bcs_pressure(prob.bcs)

    # Per-cell diffusivity: D_P = V_P / A_P
    D = Vector{T}(undef, nc)
    for c in 1:nc
        D[c] = mesh.cell_volumes[c] / state.A_P[c]
    end

    # LHS: Laplacian operator div(D * grad(p))
    assemble_laplacian!(eq, D, mesh, bcs_p)

    # RHS: divergence of H(U)/A_P flux
    phi_HbyA = compute_HbyA_flux(state, mesh)
    nf = size(mesh.face_cells, 2)
    for f in 1:nf
        P = owner(mesh, f)
        eq.b[P] += phi_HbyA[f]

        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            eq.b[N] -= phi_HbyA[f]
        end
    end

    return nothing
end

"""
    fix_pressure_reference!(eq::CollocatedEquation{T}, ref_cell::Int, ref_value::T)

Fix the pressure at a single cell to remove the null space of the
pressure Poisson equation (which is singular up to a constant when
all BCs are Neumann).

Modifies the equation so that `p[ref_cell] = ref_value`.
"""
function fix_pressure_reference!(
        eq::CollocatedEquation{T}, ref_cell::Int, ref_value::T,
    ) where {T}
    nc = length(eq.b)

    # Zero out the row
    for idx in nzrange(eq.A, ref_cell)
        eq.A.nzval[idx] = zero(T)
    end
    # Set diagonal to 1
    eq.A[ref_cell, ref_cell] = one(T)
    # Set RHS
    eq.b[ref_cell] = ref_value

    return nothing
end

"""
    _needs_pressure_reference(bcs::Dict{Symbol, <:AbstractBoundaryCondition}) -> Bool

Return `true` if all boundary conditions expand to Neumann for pressure
(i.e., no Dirichlet pressure BC exists), meaning the pressure equation
has a null space that must be fixed.
"""
function _needs_pressure_reference(bcs::Dict{Symbol, <:AbstractBoundaryCondition})
    for bc in values(bcs)
        if bc isa FixedPressureBC
            return false
        end
    end
    return true
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/incompressible/types.jl"); include("src/incompressible/boundary_conditions.jl"); include("src/incompressible/momentum.jl"); include("src/incompressible/pressure.jl")'
```
Expected: No errors.

---

### Task 5: Create correction.jl — Velocity and flux correction

**Files:**
- Create: `src/incompressible/correction.jl`

- [ ] **Step 1: Write the correction module**

```julia
# incompressible/correction.jl — Velocity and flux correction
#
# After solving the pressure equation, corrects the velocity field
# and face fluxes to satisfy continuity.

"""
    correct_velocity!(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    )

Correct the cell-centered velocity using the pressure field:

    U_c = H(U)_c / A_P[c] - (V_c / A_P[c]) * (grad p)_c

Then update face fluxes using Rhie-Chow interpolation.
"""
function correct_velocity!(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    grad_p = gradient(state.p, mesh)

    for c in 1:nc
        D_c = mesh.cell_volumes[c] / state.A_P[c]
        state.U.internal[c] = state.H_U[c] / state.A_P[c] - D_c * grad_p[c]
    end

    return nothing
end

"""
    correct_fluxes!(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    )

Update face fluxes from the corrected velocity and pressure fields
using Rhie-Chow momentum interpolation.
"""
function correct_fluxes!(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    rhie_chow_correction!(state.phi, state.U, state.p, state.A_P, mesh)
    return nothing
end

"""
    update_boundary_velocity!(
        state::IncompressibleState{Dim, T},
        bcs::Dict{Symbol, <:AbstractBoundaryCondition},
        mesh::UnstructuredFVMMesh{Dim, T},
    )

Update boundary face values of the velocity field from BCs.
Dirichlet BCs set the boundary value directly; Neumann BCs
copy the owner cell value (zero-gradient).
"""
function update_boundary_velocity!(
        state::IncompressibleState{Dim, T},
        bcs::Dict{Symbol, <:AbstractBoundaryCondition},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)

    for (i, f) in enumerate(state.U.boundary_face_indices)
        P = owner(mesh, f)
        tag = _face_tag(mesh, f)
        bc = get(bcs, tag, nothing)

        if bc isa FixedVelocityBC
            state.U.boundary[i] = bc.value
        elseif bc isa NoSlipWallBC
            state.U.boundary[i] = zero(SVector{Dim, T})
        else
            # Zero-gradient: copy owner cell value
            state.U.boundary[i] = state.U.internal[P]
        end
    end

    return nothing
end

"""
    update_boundary_pressure!(
        state::IncompressibleState,
        bcs::Dict{Symbol, <:AbstractBoundaryCondition},
        mesh::UnstructuredFVMMesh,
    )

Update boundary face values of the pressure field from BCs.
"""
function update_boundary_pressure!(
        state::IncompressibleState{Dim, T},
        bcs::Dict{Symbol, <:AbstractBoundaryCondition},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    for (i, f) in enumerate(state.p.boundary_face_indices)
        P = owner(mesh, f)
        tag = _face_tag(mesh, f)
        bc = get(bcs, tag, nothing)

        if bc isa FixedPressureBC
            state.p.boundary[i] = bc.value
        else
            # Zero-gradient: copy owner cell value
            state.p.boundary[i] = state.p.internal[P]
        end
    end

    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/incompressible/types.jl"); include("src/incompressible/boundary_conditions.jl"); include("src/incompressible/momentum.jl"); include("src/incompressible/pressure.jl"); include("src/incompressible/correction.jl")'
```
Expected: No errors.

---

### Task 6: Create residuals.jl — Convergence monitoring

**Files:**
- Create: `src/incompressible/residuals.jl`

- [ ] **Step 1: Write the residual computation**

```julia
# incompressible/residuals.jl — Convergence residual computation
#
# Computes momentum and continuity residuals for convergence monitoring
# in SIMPLE/PISO/PIMPLE loops.

"""
    momentum_residual(
        eq::CollocatedEquation{T}, u_d::Vector{T},
    ) -> T

Compute the normalized L2 momentum residual for one velocity component:
  `||A * u - b||_2 / ||b||_2`

Returns 0 if `||b||_2 == 0` to avoid division by zero.
"""
function momentum_residual(eq::CollocatedEquation{T}, u_d::Vector{T}) where {T}
    r = eq.A * u_d - eq.b
    b_norm = norm(eq.b)
    return b_norm > eps(T) ? norm(r) / b_norm : zero(T)
end

"""
    continuity_residual(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) -> T

Compute the L1 continuity residual: sum of absolute cell flux imbalance.

    R_cont = sum_c |sum_f phi_f|

where the face contributions respect the owner/neighbour sign convention.
"""
function continuity_residual(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    cell_imbalance = zeros(T, nc)

    for f in 1:nf
        P = owner(mesh, f)
        cell_imbalance[P] += state.phi.values[f]

        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            cell_imbalance[N] -= state.phi.values[f]
        end
    end

    return sum(abs, cell_imbalance)
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/incompressible/types.jl"); include("src/incompressible/residuals.jl")'
```
Expected: No errors.

---

### Task 7: Create simple.jl — SIMPLE steady-state solver

**Files:**
- Create: `src/incompressible/simple.jl`

- [ ] **Step 1: Write the SIMPLE loop**

```julia
# incompressible/simple.jl — SIMPLE pressure-velocity coupling algorithm
#
# Semi-Implicit Method for Pressure-Linked Equations.
# Steady-state iterative solver: momentum predict → pressure solve →
# correct velocity → check convergence.

"""
    solve_simple(
        prob::IncompressibleProblem{Dim, T};
        linear_solver = nothing,
        verbose::Bool = false,
    ) -> SolveResult{Dim, T}

Solve a steady-state incompressible flow problem using the SIMPLE algorithm.

Each iteration:
1. Assemble and solve momentum equations (per component, with under-relaxation)
2. Extract A_P and H(U) operators
3. Assemble and solve pressure Poisson equation
4. Under-relax pressure
5. Correct velocity and face fluxes
6. Check convergence

# Arguments
- `prob` — `IncompressibleProblem` with `SIMPLE` algorithm
- `linear_solver` — LinearSolve.jl algorithm (default: auto-select)
- `verbose` — print residuals each iteration

# Returns
`SolveResult` with convergence status, iteration count, residual history, and final state.
"""
function solve_simple(
        prob::IncompressibleProblem{Dim, T};
        linear_solver = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    alg = prob.algorithm
    mesh = prob.mesh

    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    residual_history = Dict{Symbol, Vector{T}}(
        :Ux => T[], :Uy => T[], :continuity => T[],
    )
    if Dim == 3
        residual_history[:Uz] = T[]
    end

    need_pref = _needs_pressure_reference(prob.bcs)

    for iter in 1:alg.max_iterations
        # ── 1. Momentum predictor (per component) ──
        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(eq, state, prob, d; scheme = CONV_UPWIND)
            push!(eqs, eq)
        end

        # ── 2. Extract A_P and H(U) BEFORE under-relaxation ──
        extract_momentum_operators!(state, eqs, mesh)

        # ── 3. Under-relax and solve momentum ──
        for d in 1:Dim
            u_old_d = _extract_component(state.U, d)
            under_relax_momentum!(eqs[d], u_old_d, alg.alpha_U)

            sol = solve(to_linear_problem(eqs[d]), linear_solver)
            _set_component!(state.U, d, sol.u)
        end

        update_boundary_velocity!(state, prob.bcs, mesh)

        # ── 4. Pressure equation ──
        p_eq = CollocatedEquation(mesh)
        assemble_pressure!(p_eq, state, prob)

        if need_pref
            fix_pressure_reference!(p_eq, 1, zero(T))
        end

        p_sol = solve(to_linear_problem(p_eq), linear_solver)

        # ── 5. Under-relax pressure ──
        p_new = p_sol.u
        for c in eachindex(state.p.internal)
            state.p.internal[c] += alg.alpha_p * (p_new[c] - state.p.internal[c])
        end

        update_boundary_pressure!(state, prob.bcs, mesh)

        # ── 6. Correct velocity and fluxes ──
        correct_velocity!(state, mesh)
        update_boundary_velocity!(state, prob.bcs, mesh)
        correct_fluxes!(state, mesh)

        # ── 7. Check convergence ──
        max_res = zero(T)
        for d in 1:Dim
            u_d = _extract_component(state.U, d)
            res = momentum_residual(eqs[d], u_d)
            key = d == 1 ? :Ux : (d == 2 ? :Uy : :Uz)
            push!(residual_history[key], res)
            max_res = max(max_res, res)
        end

        cont_res = continuity_residual(state, mesh)
        push!(residual_history[:continuity], cont_res)
        max_res = max(max_res, cont_res)

        if verbose
            @info "SIMPLE iter $iter: max_residual = $max_res, continuity = $cont_res"
        end

        if max_res < alg.tolerance
            return SolveResult{Dim, T}(true, iter, residual_history, state)
        end
    end

    return SolveResult{Dim, T}(false, alg.max_iterations, residual_history, state)
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e '
using FiniteVolumeMethod
for f in ["types", "boundary_conditions", "momentum", "pressure", "correction", "residuals", "simple"]
    include("src/incompressible/$f.jl")
end
'
```
Expected: No errors.

---

### Task 8: Create piso.jl — PISO transient solver

**Files:**
- Create: `src/incompressible/piso.jl`

- [ ] **Step 1: Write the PISO time loop**

```julia
# incompressible/piso.jl — PISO pressure-velocity coupling algorithm
#
# Pressure-Implicit with Splitting of Operators.
# Transient solver: momentum predict → N pressure corrections → advance time.

"""
    _piso_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T,
        n_correctors::Int;
        linear_solver = nothing,
    )

Perform one PISO time step: momentum predictor + N pressure corrections.
Modifies `state` in-place.
"""
function _piso_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T,
        n_correctors::Int;
        linear_solver = nothing,
    ) where {Dim, T}
    mesh = prob.mesh
    need_pref = _needs_pressure_reference(prob.bcs)

    # ── Momentum predictor (no under-relaxation) ──
    eqs = CollocatedEquation{T}[]
    for d in 1:Dim
        eq = CollocatedEquation(mesh)
        assemble_momentum!(eq, state, prob, d; dt = dt, scheme = CONV_UPWIND)
        push!(eqs, eq)
    end

    extract_momentum_operators!(state, eqs, mesh)

    for d in 1:Dim
        sol = solve(to_linear_problem(eqs[d]), linear_solver)
        _set_component!(state.U, d, sol.u)
    end

    update_boundary_velocity!(state, prob.bcs, mesh)

    # ── Pressure correction loop ──
    for k in 1:n_correctors
        p_eq = CollocatedEquation(mesh)
        assemble_pressure!(p_eq, state, prob)

        if need_pref
            fix_pressure_reference!(p_eq, 1, zero(T))
        end

        p_sol = solve(to_linear_problem(p_eq), linear_solver)
        state.p.internal .= p_sol.u

        update_boundary_pressure!(state, prob.bcs, mesh)
        correct_velocity!(state, mesh)
        update_boundary_velocity!(state, prob.bcs, mesh)
        correct_fluxes!(state, mesh)

        # Recompute H(U) for subsequent corrections
        if k < n_correctors
            for d in 1:Dim
                eq = CollocatedEquation(mesh)
                assemble_momentum!(eq, state, prob, d; dt = dt, scheme = CONV_UPWIND)
                eqs[d] = eq
            end
            extract_momentum_operators!(state, eqs, mesh)
        end
    end

    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e '
using FiniteVolumeMethod
for f in ["types", "boundary_conditions", "momentum", "pressure", "correction", "residuals", "simple", "piso"]
    include("src/incompressible/$f.jl")
end
'
```
Expected: No errors.

---

### Task 9: Create pimple.jl — PIMPLE hybrid solver + unified entry point

**Files:**
- Create: `src/incompressible/pimple.jl`

- [ ] **Step 1: Write the PIMPLE loop and unified solver**

```julia
# incompressible/pimple.jl — PIMPLE algorithm + unified transient solver
#
# Merged PISO-SIMPLE: outer corrections (SIMPLE-like) + inner corrections
# (PISO-like) per time step.

"""
    _pimple_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T;
        linear_solver = nothing,
    )

Perform one PIMPLE time step with outer and inner correction loops.
"""
function _pimple_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T;
        linear_solver = nothing,
    ) where {Dim, T}
    alg = prob.algorithm
    mesh = prob.mesh
    need_pref = _needs_pressure_reference(prob.bcs)

    for outer in 1:alg.n_outer
        # Momentum predictor (under-relax for non-final outer iterations)
        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(eq, state, prob, d; dt = dt, scheme = CONV_UPWIND)
            push!(eqs, eq)
        end

        extract_momentum_operators!(state, eqs, mesh)

        if outer < alg.n_outer
            for d in 1:Dim
                u_old_d = _extract_component(state.U, d)
                under_relax_momentum!(eqs[d], u_old_d, alg.alpha_U)
            end
        end

        for d in 1:Dim
            sol = solve(to_linear_problem(eqs[d]), linear_solver)
            _set_component!(state.U, d, sol.u)
        end

        update_boundary_velocity!(state, prob.bcs, mesh)

        # Inner PISO-like pressure corrections
        for k in 1:alg.n_correctors
            p_eq = CollocatedEquation(mesh)
            assemble_pressure!(p_eq, state, prob)

            if need_pref
                fix_pressure_reference!(p_eq, 1, zero(T))
            end

            p_sol = solve(to_linear_problem(p_eq), linear_solver)

            if outer < alg.n_outer
                for c in eachindex(state.p.internal)
                    state.p.internal[c] += alg.alpha_p *
                        (p_sol.u[c] - state.p.internal[c])
                end
            else
                state.p.internal .= p_sol.u
            end

            update_boundary_pressure!(state, prob.bcs, mesh)
            correct_velocity!(state, mesh)
            update_boundary_velocity!(state, prob.bcs, mesh)
            correct_fluxes!(state, mesh)
        end
    end

    return nothing
end

# ── Unified transient solver ─────────────────────────────────

"""
    solve_incompressible(
        prob::IncompressibleProblem{Dim, T},
        tspan::Tuple{T, T},
        dt::T;
        save_every::Int = 1,
        linear_solver = nothing,
        verbose::Bool = false,
    ) -> Vector{IncompressibleState{Dim, T}}

Solve a transient incompressible flow problem using the PISO or PIMPLE algorithm.

# Arguments
- `prob` — `IncompressibleProblem` with `PISO` or `PIMPLE` algorithm
- `tspan` — `(t_start, t_end)` time interval
- `dt` — fixed time step size
- `save_every` — save state every N time steps (1 = every step)
- `linear_solver` — LinearSolve.jl algorithm (default: auto-select)
- `verbose` — print progress

# Returns
Vector of saved `IncompressibleState` snapshots.
"""
function solve_incompressible(
        prob::IncompressibleProblem{Dim, T},
        tspan::Tuple{T, T},
        dt::T;
        save_every::Int = 1,
        linear_solver = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    mesh = prob.mesh
    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    t_start, t_end = tspan
    t = t_start
    n_steps = round(Int, (t_end - t_start) / dt)
    saved_states = IncompressibleState{Dim, T}[]

    for step in 1:n_steps
        t += dt

        if prob.algorithm isa PISO
            _piso_step!(state, prob, dt, prob.algorithm.n_correctors;
                linear_solver = linear_solver)
        elseif prob.algorithm isa PIMPLE
            _pimple_step!(state, prob, dt; linear_solver = linear_solver)
        else
            error("solve_incompressible requires PISO or PIMPLE algorithm, got $(typeof(prob.algorithm))")
        end

        if step % save_every == 0 || step == n_steps
            push!(saved_states, _copy_state(state, mesh))
        end

        if verbose && step % max(1, n_steps ÷ 20) == 0
            cont = continuity_residual(state, mesh)
            @info "Step $step/$n_steps, t = $t, continuity = $cont"
        end
    end

    return saved_states
end

"""Deep copy an IncompressibleState."""
function _copy_state(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    U_copy = CollocatedVectorField{Dim, T}(
        state.U.name, copy(state.U.internal),
        copy(state.U.boundary), copy(state.U.boundary_face_indices),
    )
    p_copy = CollocatedScalarField{T}(
        state.p.name, copy(state.p.internal),
        copy(state.p.boundary), copy(state.p.boundary_face_indices),
    )
    phi_copy = FaceFluxField{T}(state.phi.name, copy(state.phi.values))
    return IncompressibleState{Dim, T}(
        U_copy, p_copy, phi_copy, copy(state.A_P), copy(state.H_U),
    )
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e '
using FiniteVolumeMethod
for f in ["types", "boundary_conditions", "momentum", "pressure", "correction", "residuals", "simple", "piso", "pimple"]
    include("src/incompressible/$f.jl")
end
'
```
Expected: No errors.

---

### Task 10: Wire into module — Layer 2 includes + exports

**Files:**
- Modify: `src/layers/discretization_assembly_kernels.jl` (append at end)
- Modify: `src/FiniteVolumeMethod.jl` (add export block)

- [ ] **Step 1: Add includes to Layer 2**

Append to `src/layers/discretization_assembly_kernels.jl` after the last `include` (line 134, after `coupling/coupled_solve.jl`):

```julia
# Incompressible Navier-Stokes — SIMPLE/PISO/PIMPLE (Phase 1)
# Depends on Phase 0 collocated operators from Layer 1.
include("../incompressible/types.jl")
include("../incompressible/boundary_conditions.jl")
include("../incompressible/momentum.jl")
include("../incompressible/pressure.jl")
include("../incompressible/correction.jl")
include("../incompressible/residuals.jl")
include("../incompressible/simple.jl")
include("../incompressible/piso.jl")
include("../incompressible/pimple.jl")
```

- [ ] **Step 2: Add exports to FiniteVolumeMethod.jl**

Add a new export block after the Phase 0 collocated operators block (after `reset!` on line 295) and before the `export FVMGeometry` block:

```julia
# --- Incompressible Navier-Stokes (Phase 1) ---
export
    # Algorithm types
    AbstractPVCoupling,
    SIMPLE,
    PISO,
    PIMPLE,
    # Problem and state
    IncompressibleProblem,
    IncompressibleState,
    SolveResult,
    # Boundary conditions
    FixedVelocityBC,
    FixedPressureBC,
    NoSlipWallBC,
    SlipWallBC,
    InletOutletBC,
    # Solvers
    solve_simple,
    solve_incompressible,
    # Assembly (advanced)
    assemble_momentum!,
    assemble_pressure!,
    extract_momentum_operators!,
    correct_velocity!,
    correct_fluxes!,
    momentum_residual,
    continuity_residual
```

- [ ] **Step 3: Verify module loads**

```bash
julia --project -e 'using FiniteVolumeMethod; println("Phase 1 loaded: ", IncompressibleProblem)'
```
Expected: `Phase 1 loaded: IncompressibleProblem` (type prints its name).

- [ ] **Step 4: Commit**

```bash
git add src/incompressible/ src/layers/discretization_assembly_kernels.jl src/FiniteVolumeMethod.jl
git commit -m "feat: add incompressible NS types, operators, and SIMPLE/PISO/PIMPLE solvers (Phase 1)"
```

---

### Task 11: Write tests + helper mesh builder

**Files:**
- Create: `test/incompressible.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the test file**

```julia
using FiniteVolumeMethod
using Test
using LinearAlgebra
using StaticArrays
using SparseArrays

# ── Test mesh builder: 2D Cartesian grid → UnstructuredFVMMesh ──

"""
    build_cartesian_unstructured_mesh(nx, ny, Lx, Ly) -> UnstructuredFVMMesh{2, Float64}

Build a uniform Cartesian grid as an UnstructuredFVMMesh for testing.
Cells are numbered row-major: cell (i,j) = (j-1)*nx + i.
Faces: internal vertical, internal horizontal, then boundary faces.
Tags: :left, :right, :bottom, :top.
"""
function build_cartesian_unstructured_mesh(
        nx::Int, ny::Int, Lx::Float64 = 1.0, Ly::Float64 = 1.0,
    )
    dx = Lx / nx
    dy = Ly / ny
    nc = nx * ny
    T = Float64
    Dim = 2

    # Cell centers
    cell_centers = zeros(Dim, nc)
    cell_volumes = zeros(nc)
    for j in 1:ny, i in 1:nx
        c = (j - 1) * nx + i
        cell_centers[1, c] = (i - 0.5) * dx
        cell_centers[2, c] = (j - 0.5) * dy
        cell_volumes[c] = dx * dy
    end

    # Count faces
    n_internal_v = (nx - 1) * ny  # vertical internal faces
    n_internal_h = nx * (ny - 1)  # horizontal internal faces
    n_boundary = 2 * nx + 2 * ny  # boundary faces
    nf = n_internal_v + n_internal_h + n_boundary

    face_cells = zeros(Int, 2, nf)
    face_normals = zeros(Dim, nf)
    face_areas = zeros(nf)
    face_centers = zeros(Dim, nf)
    face_tags = Vector{Symbol}(undef, nf)

    f_idx = 0
    cell_faces_dict = Dict{Int, Vector{Int}}()
    for c in 1:nc
        cell_faces_dict[c] = Int[]
    end

    function add_face!(P, N, nx_f, ny_f, area, cx, cy, tag)
        f_idx += 1
        face_cells[1, f_idx] = P
        face_cells[2, f_idx] = N
        face_normals[1, f_idx] = nx_f
        face_normals[2, f_idx] = ny_f
        face_areas[f_idx] = area
        face_centers[1, f_idx] = cx
        face_centers[2, f_idx] = cy
        face_tags[f_idx] = tag
        push!(cell_faces_dict[P], f_idx)
        if N != 0
            push!(cell_faces_dict[N], f_idx)
        end
    end

    # Internal vertical faces (between cell i and i+1)
    for j in 1:ny, i in 1:(nx - 1)
        P = (j - 1) * nx + i
        N = (j - 1) * nx + i + 1
        add_face!(P, N, 1.0, 0.0, dy, i * dx, (j - 0.5) * dy, :internal)
    end

    # Internal horizontal faces (between cell j and j+1)
    for j in 1:(ny - 1), i in 1:nx
        P = (j - 1) * nx + i
        N = j * nx + i
        add_face!(P, N, 0.0, 1.0, dx, (i - 0.5) * dx, j * dy, :internal)
    end

    # Boundary: left (i=1, normal = -x)
    for j in 1:ny
        P = (j - 1) * nx + 1
        add_face!(P, 0, -1.0, 0.0, dy, 0.0, (j - 0.5) * dy, :left)
    end

    # Boundary: right (i=nx, normal = +x)
    for j in 1:ny
        P = (j - 1) * nx + nx
        add_face!(P, 0, 1.0, 0.0, dy, Lx, (j - 0.5) * dy, :right)
    end

    # Boundary: bottom (j=1, normal = -y)
    for i in 1:nx
        P = i
        add_face!(P, 0, 0.0, -1.0, dx, (i - 0.5) * dx, 0.0, :bottom)
    end

    # Boundary: top (j=ny, normal = +y)
    for i in 1:nx
        P = (ny - 1) * nx + i
        add_face!(P, 0, 0.0, 1.0, dx, (i - 0.5) * dx, Ly, :top)
    end

    @assert f_idx == nf "Expected $nf faces, got $f_idx"

    cell_faces = [cell_faces_dict[c] for c in 1:nc]

    return UnstructuredFVMMesh{Dim, T}(
        cell_centers, cell_volumes, cell_faces,
        nothing,  # cell_types
        face_cells, face_normals, face_areas, face_centers,
        face_tags,
    )
end

@testset "Incompressible NS — Phase 1" begin

    @testset "Type construction" begin
        # Algorithm defaults
        alg_s = SIMPLE()
        @test alg_s.alpha_U ≈ 0.7
        @test alg_s.alpha_p ≈ 0.3
        @test alg_s.max_iterations == 1000

        alg_pi = PISO()
        @test alg_pi.n_correctors == 2

        alg_pm = PIMPLE()
        @test alg_pm.n_outer == 2
        @test alg_pm.n_correctors == 1

        # Mesh and state
        mesh = build_cartesian_unstructured_mesh(4, 4)
        state = IncompressibleState(mesh)
        @test length(state.U.internal) == 16
        @test length(state.p.internal) == 16
        @test length(state.A_P) == 16
    end

    @testset "BC expansion" begin
        bc_wall = NoSlipWallBC()
        @test FiniteVolumeMethod.expand_velocity_bc(bc_wall, 1) isa ParabolicDirichlet
        @test FiniteVolumeMethod.expand_pressure_bc(bc_wall) isa ParabolicNeumann

        bc_inlet = FixedVelocityBC((1.0, 0.0))
        @test FiniteVolumeMethod.expand_velocity_bc(bc_inlet, 1).value ≈ 1.0
        @test FiniteVolumeMethod.expand_velocity_bc(bc_inlet, 2).value ≈ 0.0

        bc_outlet = FixedPressureBC(0.0)
        @test FiniteVolumeMethod.expand_pressure_bc(bc_outlet).value ≈ 0.0
    end

    @testset "Momentum assembly smoke test" begin
        mesh = build_cartesian_unstructured_mesh(4, 4)
        state = IncompressibleState(mesh)

        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((1.0, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )

        prob = IncompressibleProblem(mesh, bcs, SIMPLE(); nu = 0.01)

        eq = CollocatedEquation(mesh)
        assemble_momentum!(eq, state, prob, 1)

        # Matrix should be non-zero after assembly
        @test nnz(eq.A) > 0
        @test any(!iszero, eq.b)
    end

    @testset "Pressure equation smoke test" begin
        mesh = build_cartesian_unstructured_mesh(4, 4)
        state = IncompressibleState(mesh)

        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((1.0, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )

        prob = IncompressibleProblem(mesh, bcs, SIMPLE(); nu = 0.01)

        p_eq = CollocatedEquation(mesh)
        assemble_pressure!(p_eq, state, prob)

        @test nnz(p_eq.A) > 0
    end

    @testset "Pressure reference fix" begin
        mesh = build_cartesian_unstructured_mesh(4, 4)
        eq = CollocatedEquation(mesh)
        # Make diagonal nonzero
        for c in 1:16
            eq.A[c, c] = 2.0
        end
        eq.b .= 1.0

        FiniteVolumeMethod.fix_pressure_reference!(eq, 1, 0.0)
        @test eq.A[1, 1] ≈ 1.0
        @test eq.b[1] ≈ 0.0
    end

    @testset "SIMPLE convergence — Poiseuille-like flow" begin
        # Simple channel flow: left inlet, right outlet, top/bottom walls
        # At Re << 1 with coarse mesh, SIMPLE should converge
        nx, ny = 8, 4
        mesh = build_cartesian_unstructured_mesh(nx, ny)

        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )

        prob = IncompressibleProblem(
            mesh, bcs, SIMPLE(; max_iterations = 500, tolerance = 1e-4);
            nu = 0.1,
        )

        result = solve_simple(prob)

        # Should converge (low Re flow on coarse mesh)
        @test result.converged
        @test result.iterations < 500

        # Velocity should be positive in x-direction at interior cells
        u_x = [result.state.U.internal[c][1] for c in 1:nx * ny]
        @test all(u_x .>= -0.01)  # allow tiny numerical noise

        # Continuity residual should be small
        @test result.residuals[:continuity][end] < 1e-3
    end

    @testset "PISO transient smoke test" begin
        nx, ny = 4, 4
        mesh = build_cartesian_unstructured_mesh(nx, ny)

        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )

        prob = IncompressibleProblem(mesh, bcs, PISO(); nu = 0.1)
        states = solve_incompressible(prob, (0.0, 0.01), 0.005; save_every = 1)

        @test length(states) == 2  # 2 steps
        @test length(states[end].U.internal) == 16
    end

    @testset "PIMPLE transient smoke test" begin
        nx, ny = 4, 4
        mesh = build_cartesian_unstructured_mesh(nx, ny)

        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )

        prob = IncompressibleProblem(mesh, bcs, PIMPLE(); nu = 0.1)
        states = solve_incompressible(prob, (0.0, 0.01), 0.005; save_every = 1)

        @test length(states) == 2
    end

    @testset "Residual computation" begin
        mesh = build_cartesian_unstructured_mesh(4, 4)
        state = IncompressibleState(mesh)

        # Zero flux → zero continuity residual
        res = continuity_residual(state, mesh)
        @test res ≈ 0.0 atol = 1e-15
    end

end
```

- [ ] **Step 2: Register test in runtests.jl**

Add after the last `safe_include` in the appropriate testset in `test/runtests.jl`. Find the section near the end of the unit tests (after `coupling.jl` or similar) and add:

```julia
        safe_include("incompressible.jl")
```

- [ ] **Step 3: Run the test**

```bash
julia --project=test test/incompressible.jl
```
Expected: All tests pass.

- [ ] **Step 4: Run Runic formatting**

```bash
julia --project -e 'using Runic; Runic.main(["--inplace", "src/incompressible/"])'
julia --project -e 'using Runic; Runic.main(["--inplace", "test/incompressible.jl"])'
```

- [ ] **Step 5: Commit**

```bash
git add test/incompressible.jl test/runtests.jl
git commit -m "test: add incompressible NS test suite (SIMPLE/PISO/PIMPLE)"
```

---

### Task 12: Register features in validation manifest + Phase 0

**Files:**
- Modify: `validation/manifest.toml`

- [ ] **Step 1: Add collocated_operators and incompressible_ns features**

Append to `validation/manifest.toml`:

```toml
[[features]]
feature = "collocated_operators"
maturity = "experimental"
validation = "smoke_tested"
role = "research_support_tooling"
solver_family = "collocated"
precision_policy = "float64_cpu_reference"
random_seed_policy = "deterministic"
backend_policy = "cpu_reference"
required_ladder_stages = ["verification"]
summary = "Collocated cell-centered FVM operators on unstructured polyhedral meshes: Laplacian, divergence, gradient, temporal, interpolation."
limitations = [
  "Experimental — operators tested via unit tests only; no MMS convergence verification yet.",
  "Requires UnstructuredFVMMesh with cell_faces and face_tags populated.",
]

[[features]]
feature = "incompressible_ns"
maturity = "experimental"
validation = "smoke_tested"
role = "research_tooling"
solver_family = "collocated"
precision_policy = "float64_cpu_reference"
random_seed_policy = "deterministic"
backend_policy = "cpu_reference"
required_ladder_stages = ["verification", "benchmark"]
summary = "Incompressible Navier-Stokes with SIMPLE, PISO, and PIMPLE pressure-velocity coupling on unstructured meshes."
limitations = [
  "Experimental — validated via smoke tests only; lid-driven cavity benchmark pending.",
  "Scalar viscosity only; turbulence models (Phase 2) not yet integrated.",
  "Fixed time step only; adaptive CFL control deferred.",
]
```

- [ ] **Step 2: Commit**

```bash
git add validation/manifest.toml
git commit -m "feat: register collocated_operators and incompressible_ns in validation manifest"
```

---

### Task 13: End-to-end verification — module load + full test suite

**Files:** None (verification only)

- [ ] **Step 1: Verify clean module load**

```bash
julia --project -e '
using FiniteVolumeMethod
# Verify Phase 1 exports exist
@assert isdefined(FiniteVolumeMethod, :IncompressibleProblem)
@assert isdefined(FiniteVolumeMethod, :SIMPLE)
@assert isdefined(FiniteVolumeMethod, :PISO)
@assert isdefined(FiniteVolumeMethod, :PIMPLE)
@assert isdefined(FiniteVolumeMethod, :solve_simple)
@assert isdefined(FiniteVolumeMethod, :solve_incompressible)
@assert isdefined(FiniteVolumeMethod, :FixedVelocityBC)
@assert isdefined(FiniteVolumeMethod, :NoSlipWallBC)
println("All Phase 1 exports verified")
'
```
Expected: `All Phase 1 exports verified`

- [ ] **Step 2: Run incompressible test suite**

```bash
julia --project=test test/incompressible.jl
```
Expected: All tests pass.

- [ ] **Step 3: Run Runic check on all new files**

```bash
julia --project -e 'using Runic; Runic.main(["--check", "src/incompressible/"])'
julia --project -e 'using Runic; Runic.main(["--check", "test/incompressible.jl"])'
```
Expected: No formatting issues.

- [ ] **Step 4: Spot-check regression — existing tests still pass**

```bash
julia --project=test -e '
using FiniteVolumeMethod, Test
# Quick smoke: existing parabolic types still work
@test ParabolicDirichlet(1.0) isa AbstractBoundaryCondition
@test ParabolicNeumann(0.0) isa AbstractBoundaryCondition
println("Regression check passed")
'
```
Expected: `Regression check passed`
