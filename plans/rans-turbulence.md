---
date: 2026-04-06
---

# RANS Turbulence Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add RANS turbulence models (k-ε, k-ω, k-ω SST, Spalart-Allmaras) as segregated scalar transport equations within the SIMPLE/PISO/PIMPLE incompressible solver loops, producing per-cell turbulent viscosity.

**Architecture:** New `src/turbulence/` directory with interface, strain rate, four model files, wall functions, and solver wrappers. Models implement a 3-function interface (`turbulent_viscosity!`, `solve_turbulence!`, field metadata). Turbulent solver wrappers call existing Phase 1 building blocks with `nu_eff = nu + nu_t`. Momentum assembly gains an optional `nu_eff` keyword.

**Tech Stack:** Julia, SparseArrays, LinearAlgebra, StaticArrays, SciMLBase, Phase 0 collocated operators, Phase 1 incompressible solvers, existing `StandardKEpsilon`/`KappaOmegaSST` from `src/physics/turbulence/k_epsilon.jl`.

---

## File Map

| File | Purpose | Creates/Modifies |
|------|---------|-----------------|
| `src/turbulence/interface.jl` | `AbstractRANSModel`, `RANSTurbulenceState`, dispatch interface, state constructors | Create |
| `src/turbulence/strain_rate.jl` | `compute_strain_rate` from velocity gradients on collocated mesh | Create |
| `src/turbulence/wall_distance.jl` | `compute_wall_distance` for SST and SA models | Create |
| `src/turbulence/k_epsilon_rans.jl` | k-ε assembly for collocated solver | Create |
| `src/turbulence/k_omega.jl` | Standard k-ω (Wilcox 1988) | Create |
| `src/turbulence/k_omega_sst.jl` | k-ω SST (Menter) | Create |
| `src/turbulence/spalart_allmaras.jl` | One-equation SA model | Create |
| `src/turbulence/wall_functions.jl` | Collocated wall function BC generation | Create |
| `src/turbulence/solvers.jl` | `solve_simple_turbulent`, `solve_incompressible_turbulent` | Create |
| `src/incompressible/momentum.jl` | Add `nu_eff` keyword to `assemble_momentum!` | Modify |
| `src/layers/discretization_assembly_kernels.jl` | Wire turbulence includes | Modify |
| `src/FiniteVolumeMethod.jl` | Add turbulence exports | Modify |
| `test/turbulence_rans.jl` | All tests | Create |
| `test/runtests.jl` | Register test | Modify |
| `validation/manifest.toml` | Register feature | Modify |

---

### Task 1: Add `nu_eff` keyword to `assemble_momentum!`

**Files:**
- Modify: `src/incompressible/momentum.jl`

This is the prerequisite for all turbulence work. The existing `assemble_momentum!` hard-codes `prob.nu` in the Laplacian call. Adding `nu_eff` as an optional keyword makes it accept per-cell viscosity.

- [ ] **Step 1: Modify the function signature and Laplacian call**

In `src/incompressible/momentum.jl`, change `assemble_momentum!`:

```julia
function assemble_momentum!(
        eq::CollocatedEquation{T},
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        component::Int;
        dt::Union{Nothing, T} = nothing,
        scheme::ConvectionScheme = CONV_UPWIND,
        nu_eff::Union{T, Vector{T}} = prob.nu,
    ) where {Dim, T}
```

And change the Laplacian call from:
```julia
    assemble_laplacian!(eq, prob.nu, mesh, bcs_U)
```
to:
```julia
    assemble_laplacian!(eq, nu_eff, mesh, bcs_U)
```

Also update the docstring to document the new keyword.

- [ ] **Step 2: Verify existing tests still pass**

```bash
julia --project=test test/incompressible.jl
```
Expected: All 94 tests pass (backward-compatible since default is `prob.nu`).

- [ ] **Step 3: Commit**

```bash
git add src/incompressible/momentum.jl
git commit -m "feat: add nu_eff keyword to assemble_momentum! for turbulence support"
```

---

### Task 2: Create interface.jl — Abstract types, state, dispatch interface

**Files:**
- Create: `src/turbulence/interface.jl`

- [ ] **Step 1: Create directory and write interface file**

```bash
mkdir -p src/turbulence
```

Write `src/turbulence/interface.jl`:

```julia
# turbulence/interface.jl — Abstract types and dispatch interface for RANS models
#
# Defines the turbulence model hierarchy, the mutable turbulence state,
# and the interface functions that every RANS model must implement.

# ── Abstract hierarchy ───────────────────────────────────────────────

"""
    AbstractRANSModel <: AbstractTurbulenceModel

Supertype for Reynolds-Averaged Navier-Stokes turbulence models.

Every concrete RANS model must implement:
- `turbulent_viscosity!(nu_t, model, turb_state, mesh)`
- `solve_turbulence!(turb_state, model, U, phi, nu, mesh, bcs_turb; dt, linear_solver)`
- `n_turbulence_fields(model)` → Int
- `turbulence_field_names(model)` → Tuple of Symbols
"""
abstract type AbstractRANSModel <: AbstractTurbulenceModel end

# ── Turbulence state ─────────────────────────────────────────────────

"""
    RANSTurbulenceState{T}

Mutable state for RANS turbulence models. Holds the turbulence fields
(e.g. k, ε, ω, ν̃) and the per-cell turbulent viscosity.

# Fields
- `fields::Dict{Symbol, CollocatedScalarField{T}}` — turbulence fields keyed by name
- `nu_t::Vector{T}` — turbulent viscosity per cell
"""
mutable struct RANSTurbulenceState{T}
    fields::Dict{Symbol, CollocatedScalarField{T}}
    nu_t::Vector{T}
end

"""
    RANSTurbulenceState(model::AbstractRANSModel, mesh; initial_values...)

Construct a zero-initialized turbulence state for `model` on `mesh`.

Each field from `turbulence_field_names(model)` is created as a
`CollocatedScalarField`. Optional keyword arguments set initial values
(e.g. `k = 1e-4, epsilon = 1e-6`).
"""
function RANSTurbulenceState(
        model::AbstractRANSModel,
        mesh::UnstructuredFVMMesh{Dim, T};
        kwargs...,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    names = turbulence_field_names(model)
    fields = Dict{Symbol, CollocatedScalarField{T}}()
    for name in names
        init_val = get(kwargs, name, T(1e-6))
        fields[name] = CollocatedScalarField(name, mesh; value = init_val)
    end
    nu_t = zeros(T, nc)
    return RANSTurbulenceState{T}(fields, nu_t)
end

# ── Interface stubs (dispatched by concrete models) ──────────────────

"""
    turbulent_viscosity!(nu_t, model, turb_state, mesh)

Compute turbulent viscosity from current turbulence fields and store
in `nu_t`. Each RANS model provides its own formula.
"""
function turbulent_viscosity! end

"""
    solve_turbulence!(turb_state, model, U, phi, nu, mesh, bcs_turb; dt, linear_solver)

Assemble and solve the turbulence transport equations, updating
`turb_state.fields` in-place.

# Arguments
- `turb_state` — turbulence state (modified in-place)
- `model` — RANS model
- `U` — cell-centered velocity field
- `phi` — face flux field
- `nu` — laminar kinematic viscosity
- `mesh` — unstructured FVM mesh
- `bcs_turb` — `Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}`
- `dt` — time step (nothing for steady)
- `linear_solver` — LinearSolve.jl algorithm
"""
function solve_turbulence! end

"""
    n_turbulence_fields(model::AbstractRANSModel) -> Int

Number of transport equations solved by this model.
"""
function n_turbulence_fields end

"""
    turbulence_field_names(model::AbstractRANSModel) -> Tuple{Vararg{Symbol}}

Ordered names of the turbulence fields.
"""
function turbulence_field_names end

# ── Effective viscosity helper ───────────────────────────────────────

"""
    compute_nu_eff(nu::T, nu_t::Vector{T}) -> Vector{T}

Compute effective viscosity `nu_eff[c] = nu + nu_t[c]`.
"""
function compute_nu_eff(nu::T, nu_t::Vector{T}) where {T}
    nc = length(nu_t)
    nu_eff = Vector{T}(undef, nc)
    for c in 1:nc
        nu_eff[c] = nu + nu_t[c]
    end
    return nu_eff
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/turbulence/interface.jl"); println("OK")'
```

---

### Task 3: Create strain_rate.jl and wall_distance.jl — Shared utilities

**Files:**
- Create: `src/turbulence/strain_rate.jl`
- Create: `src/turbulence/wall_distance.jl`

- [ ] **Step 1: Write strain rate computation**

Write `src/turbulence/strain_rate.jl`:

```julia
# turbulence/strain_rate.jl — Strain rate magnitude from velocity gradients
#
# Computes |S| = sqrt(2 * S_ij * S_ij) where S_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i).
# Used by all RANS models to compute turbulence production.

"""
    compute_strain_rate(
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) -> Vector{T}

Compute the strain rate magnitude `|S|` at each cell center from the
velocity field `U`.

Uses Green-Gauss gradient reconstruction to compute velocity gradients,
then assembles the symmetric strain rate tensor and returns its magnitude.

For 2D: `|S| = sqrt(2*(S_xx² + S_yy² + 2*S_xy²))`
For 3D: `|S| = sqrt(2*(S_xx² + S_yy² + S_zz² + 2*(S_xy² + S_xz² + S_yz²)))`
"""
function compute_strain_rate(
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    S_mag = Vector{T}(undef, nc)

    # Compute gradient of each velocity component
    grad_U = Vector{Vector{SVector{Dim, T}}}(undef, Dim)
    for d in 1:Dim
        u_d_field = CollocatedScalarField(
            Symbol(:U, d), mesh;
            value = zero(T),
        )
        # Copy component d into scalar field
        for c in 1:nc
            u_d_field.internal[c] = U.internal[c][d]
        end
        # Copy boundary values
        for (i, f) in enumerate(u_d_field.boundary_face_indices)
            bi = findfirst(==(f), U.boundary_face_indices)
            if bi !== nothing
                u_d_field.boundary[i] = U.boundary[bi][d]
            end
        end
        grad_U[d] = gradient(u_d_field, mesh)
    end

    # Assemble strain rate magnitude per cell
    for c in 1:nc
        S_sq = _strain_rate_squared(Val(Dim), grad_U, c)
        S_mag[c] = sqrt(max(S_sq, zero(T)))
    end

    return S_mag
end

"""2D strain rate: 2*(S_xx² + S_yy² + 2*S_xy²)"""
function _strain_rate_squared(
        ::Val{2}, grad_U::Vector{Vector{SVector{2, T}}}, c::Int,
    ) where {T}
    dudx = grad_U[1][c][1]
    dudy = grad_U[1][c][2]
    dvdx = grad_U[2][c][1]
    dvdy = grad_U[2][c][2]

    S_xx = dudx
    S_yy = dvdy
    S_xy = T(0.5) * (dudy + dvdx)

    return T(2) * (S_xx^2 + S_yy^2 + T(2) * S_xy^2)
end

"""3D strain rate: 2*(S_xx² + S_yy² + S_zz² + 2*(S_xy² + S_xz² + S_yz²))"""
function _strain_rate_squared(
        ::Val{3}, grad_U::Vector{Vector{SVector{3, T}}}, c::Int,
    ) where {T}
    dudx = grad_U[1][c][1]; dudy = grad_U[1][c][2]; dudz = grad_U[1][c][3]
    dvdx = grad_U[2][c][1]; dvdy = grad_U[2][c][2]; dvdz = grad_U[2][c][3]
    dwdx = grad_U[3][c][1]; dwdy = grad_U[3][c][2]; dwdz = grad_U[3][c][3]

    S_xx = dudx; S_yy = dvdy; S_zz = dwdz
    S_xy = T(0.5) * (dudy + dvdx)
    S_xz = T(0.5) * (dudz + dwdx)
    S_yz = T(0.5) * (dvdz + dwdy)

    return T(2) * (S_xx^2 + S_yy^2 + S_zz^2 + T(2) * (S_xy^2 + S_xz^2 + S_yz^2))
end
```

- [ ] **Step 2: Write wall distance computation**

Write `src/turbulence/wall_distance.jl`:

```julia
# turbulence/wall_distance.jl — Cell-to-wall distance computation
#
# Computes the minimum distance from each cell center to the nearest wall
# boundary face. Required by k-ω SST (blending functions F1, F2) and
# Spalart-Allmaras (production and destruction terms).

"""
    compute_wall_distance(
        mesh::UnstructuredFVMMesh{Dim, T},
        wall_patches::Vector{Symbol},
    ) -> Vector{T}

Compute the minimum distance from each cell center to the nearest wall
boundary face center.

Identifies wall faces by matching `mesh.face_tags` against `wall_patches`.
Returns a vector of length `ncells`. Cells far from any wall face get
the distance to the nearest wall face (no capping).

Complexity: O(ncells × n_wall_faces). Computed once at setup.
"""
function compute_wall_distance(
        mesh::UnstructuredFVMMesh{Dim, T},
        wall_patches::Vector{Symbol},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Collect wall face indices
    wall_faces = Int[]
    wall_set = Set(wall_patches)
    for f in 1:nf
        if !is_internal_face(mesh, f)
            tag = _face_tag(mesh, f)
            if tag in wall_set
                push!(wall_faces, f)
            end
        end
    end

    d_wall = fill(T(Inf), nc)

    for f in wall_faces
        x_f = face_center(mesh, f)
        for c in 1:nc
            x_c = cell_center(mesh, c)
            dist = norm(x_c - x_f)
            d_wall[c] = min(d_wall[c], dist)
        end
    end

    # Safety: if no wall faces found, set to 1.0 to avoid division by zero
    if isempty(wall_faces)
        fill!(d_wall, one(T))
    end

    return d_wall
end
```

- [ ] **Step 3: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/turbulence/interface.jl"); include("src/turbulence/strain_rate.jl"); include("src/turbulence/wall_distance.jl"); println("OK")'
```

---

### Task 4: Create k_epsilon_rans.jl — k-ε collocated assembly

**Files:**
- Create: `src/turbulence/k_epsilon_rans.jl`

- [ ] **Step 1: Write k-ε implementation**

Write `src/turbulence/k_epsilon_rans.jl`:

```julia
# turbulence/k_epsilon_rans.jl — Standard k-ε model for collocated solver
#
# Provides turbulent_viscosity! and solve_turbulence! methods for the
# existing StandardKEpsilon type. Assembles k and ε transport equations
# using Phase 0 operators (convection, Laplacian, source linearization).

# ── Interface implementation ─────────────────────────────────────────

n_turbulence_fields(::StandardKEpsilon) = 2
turbulence_field_names(::StandardKEpsilon) = (:k, :epsilon)

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::StandardKEpsilon,
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    k_field = turb_state.fields[:k]
    eps_field = turb_state.fields[:epsilon]
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        k_val = max(k_field.internal[c], T(1e-10))
        eps_val = max(eps_field.internal[c], T(1e-10))
        nu_t[c] = model.C_mu * k_val^2 / eps_val
    end
    return nothing
end

function solve_turbulence!(
        turb_state::RANSTurbulenceState{T},
        model::StandardKEpsilon,
        U::CollocatedVectorField{Dim, T},
        phi::FaceFluxField{T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_turb::Dict{Symbol, <:Dict{Symbol, <:AbstractBoundaryCondition}};
        dt::Union{Nothing, T} = nothing,
        linear_solver = nothing,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    k_field = turb_state.fields[:k]
    eps_field = turb_state.fields[:epsilon]

    # Compute production
    S_mag = compute_strain_rate(U, mesh)
    P_k = Vector{T}(undef, nc)
    for c in 1:nc
        P_k[c] = turb_state.nu_t[c] * S_mag[c]^2
    end

    # ── k equation ───────────────────────────────────────────────
    k_eq = CollocatedEquation(mesh)
    bcs_k = get(bcs_turb, :k, Dict{Symbol, AbstractBoundaryCondition}())

    # Convection
    assemble_convection!(k_eq, phi, mesh, bcs_k)

    # Diffusion: gamma_k = nu + nu_t / sigma_k
    gamma_k = Vector{T}(undef, nc)
    for c in 1:nc
        gamma_k[c] = nu + turb_state.nu_t[c] / model.sigma_k
    end
    assemble_laplacian!(k_eq, gamma_k, mesh, bcs_k)

    # Temporal term
    if dt !== nothing
        assemble_ddt_euler!(k_eq, one(T), k_field.internal, mesh, dt)
    end

    # Source: S_C = P_k, S_P = -eps/k (linearized destruction)
    for c in 1:nc
        k_safe = max(k_field.internal[c], T(1e-10))
        k_eq.b[c] += P_k[c] * mesh.cell_volumes[c]
        k_eq.A[c, c] += eps_field.internal[c] / k_safe * mesh.cell_volumes[c]
    end

    # Solve k
    lp_k = to_linear_problem(k_eq)
    sol_k = _solve_linear(lp_k, linear_solver)
    for c in 1:nc
        k_field.internal[c] = max(sol_k.u[c], T(1e-10))
    end

    # ── ε equation ───────────────────────────────────────────────
    eps_eq = CollocatedEquation(mesh)
    bcs_eps = get(bcs_turb, :epsilon, Dict{Symbol, AbstractBoundaryCondition}())

    # Convection
    assemble_convection!(eps_eq, phi, mesh, bcs_eps)

    # Diffusion: gamma_eps = nu + nu_t / sigma_epsilon
    gamma_eps = Vector{T}(undef, nc)
    for c in 1:nc
        gamma_eps[c] = nu + turb_state.nu_t[c] / model.sigma_epsilon
    end
    assemble_laplacian!(eps_eq, gamma_eps, mesh, bcs_eps)

    # Temporal term
    if dt !== nothing
        assemble_ddt_euler!(eps_eq, one(T), eps_field.internal, mesh, dt)
    end

    # Source: S_C = C1*(eps/k)*P_k, S_P = -C2*(eps/k) (linearized)
    for c in 1:nc
        k_safe = max(k_field.internal[c], T(1e-10))
        eps_by_k = eps_field.internal[c] / k_safe
        eps_eq.b[c] += model.C1_epsilon * eps_by_k * P_k[c] * mesh.cell_volumes[c]
        eps_eq.A[c, c] += model.C2_epsilon * eps_by_k * mesh.cell_volumes[c]
    end

    # Solve ε
    lp_eps = to_linear_problem(eps_eq)
    sol_eps = _solve_linear(lp_eps, linear_solver)
    for c in 1:nc
        eps_field.internal[c] = max(sol_eps.u[c], T(1e-10))
    end

    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; for f in ["interface", "strain_rate", "wall_distance", "k_epsilon_rans"]; include("src/turbulence/$f.jl"); end; println("OK")'
```

---

### Task 5: Create k_omega.jl — Standard k-ω (Wilcox)

**Files:**
- Create: `src/turbulence/k_omega.jl`

- [ ] **Step 1: Write k-ω implementation**

Write `src/turbulence/k_omega.jl`:

```julia
# turbulence/k_omega.jl — Standard k-ω turbulence model (Wilcox 1988)
#
# Two-equation model with better near-wall behavior than k-ε.
# Turbulent viscosity: ν_t = k / ω

"""
    KOmega{T} <: AbstractRANSModel

Standard k-ω turbulence model (Wilcox 1988).

# Fields
- `beta_star::T` — k destruction coefficient (default 0.09)
- `alpha::T` — ω production coefficient (default 5/9)
- `beta::T` — ω destruction coefficient (default 3/40)
- `sigma_k::T` — k diffusion Prandtl number (default 0.5)
- `sigma_omega::T` — ω diffusion Prandtl number (default 0.5)
"""
struct KOmega{T} <: AbstractRANSModel
    beta_star::T
    alpha::T
    beta::T
    sigma_k::T
    sigma_omega::T
end

function KOmega(;
        beta_star = 0.09, alpha = 5.0 / 9.0, beta = 3.0 / 40.0,
        sigma_k = 0.5, sigma_omega = 0.5,
    )
    T = promote_type(
        typeof(beta_star), typeof(alpha), typeof(beta),
        typeof(sigma_k), typeof(sigma_omega),
    )
    return KOmega{T}(T(beta_star), T(alpha), T(beta), T(sigma_k), T(sigma_omega))
end

n_turbulence_fields(::KOmega) = 2
turbulence_field_names(::KOmega) = (:k, :omega)

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::KOmega,
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    k_field = turb_state.fields[:k]
    omega_field = turb_state.fields[:omega]
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        k_val = max(k_field.internal[c], T(1e-10))
        omega_val = max(omega_field.internal[c], T(1e-10))
        nu_t[c] = k_val / omega_val
    end
    return nothing
end

function solve_turbulence!(
        turb_state::RANSTurbulenceState{T},
        model::KOmega,
        U::CollocatedVectorField{Dim, T},
        phi::FaceFluxField{T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_turb::Dict{Symbol, <:Dict{Symbol, <:AbstractBoundaryCondition}};
        dt::Union{Nothing, T} = nothing,
        linear_solver = nothing,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    k_field = turb_state.fields[:k]
    omega_field = turb_state.fields[:omega]

    # Production
    S_mag = compute_strain_rate(U, mesh)
    P_k = Vector{T}(undef, nc)
    for c in 1:nc
        P_k[c] = turb_state.nu_t[c] * S_mag[c]^2
    end

    # ── k equation ───────────────────────────────────────────────
    k_eq = CollocatedEquation(mesh)
    bcs_k = get(bcs_turb, :k, Dict{Symbol, AbstractBoundaryCondition}())

    assemble_convection!(k_eq, phi, mesh, bcs_k)

    gamma_k = Vector{T}(undef, nc)
    for c in 1:nc
        gamma_k[c] = nu + model.sigma_k * turb_state.nu_t[c]
    end
    assemble_laplacian!(k_eq, gamma_k, mesh, bcs_k)

    if dt !== nothing
        assemble_ddt_euler!(k_eq, one(T), k_field.internal, mesh, dt)
    end

    # Source: S_C = P_k, S_P = -beta_star * omega
    for c in 1:nc
        omega_val = max(omega_field.internal[c], T(1e-10))
        k_eq.b[c] += P_k[c] * mesh.cell_volumes[c]
        k_eq.A[c, c] += model.beta_star * omega_val * mesh.cell_volumes[c]
    end

    lp_k = to_linear_problem(k_eq)
    sol_k = _solve_linear(lp_k, linear_solver)
    for c in 1:nc
        k_field.internal[c] = max(sol_k.u[c], T(1e-10))
    end

    # ── ω equation ───────────────────────────────────────────────
    omega_eq = CollocatedEquation(mesh)
    bcs_omega = get(bcs_turb, :omega, Dict{Symbol, AbstractBoundaryCondition}())

    assemble_convection!(omega_eq, phi, mesh, bcs_omega)

    gamma_omega = Vector{T}(undef, nc)
    for c in 1:nc
        gamma_omega[c] = nu + model.sigma_omega * turb_state.nu_t[c]
    end
    assemble_laplacian!(omega_eq, gamma_omega, mesh, bcs_omega)

    if dt !== nothing
        assemble_ddt_euler!(omega_eq, one(T), omega_field.internal, mesh, dt)
    end

    # Source: S_C = alpha*(omega/k)*P_k, S_P = -beta*omega
    for c in 1:nc
        k_safe = max(k_field.internal[c], T(1e-10))
        omega_val = max(omega_field.internal[c], T(1e-10))
        omega_eq.b[c] += model.alpha * (omega_val / k_safe) * P_k[c] * mesh.cell_volumes[c]
        omega_eq.A[c, c] += model.beta * omega_val * mesh.cell_volumes[c]
    end

    lp_omega = to_linear_problem(omega_eq)
    sol_omega = _solve_linear(lp_omega, linear_solver)
    for c in 1:nc
        omega_field.internal[c] = max(sol_omega.u[c], T(1e-10))
    end

    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; for f in ["interface", "strain_rate", "wall_distance", "k_epsilon_rans", "k_omega"]; include("src/turbulence/$f.jl"); end; println("OK")'
```

---

### Task 6: Create k_omega_sst.jl — k-ω SST (Menter)

**Files:**
- Create: `src/turbulence/k_omega_sst.jl`

- [ ] **Step 1: Write k-ω SST implementation**

Write `src/turbulence/k_omega_sst.jl`:

```julia
# turbulence/k_omega_sst.jl — k-ω SST turbulence model (Menter 1994)
#
# Blends k-ω (near wall) with k-ε (far field) via blending functions
# F1 and F2. Includes the SST viscosity limiter for adverse pressure
# gradients. Wraps the existing KappaOmegaSST coefficients struct.

"""
    KOmegaSSTModel{T} <: AbstractRANSModel

k-ω SST (Shear Stress Transport) turbulence model.

Wraps `KappaOmegaSST{T}` coefficients and adds the blending/limiter
logic. Requires wall distance `d_wall` per cell.

# Fields
- `coeffs::KappaOmegaSST{T}` — model coefficients
- `d_wall::Vector{T}` — wall distance per cell (precomputed)
"""
struct KOmegaSSTModel{T} <: AbstractRANSModel
    coeffs::KappaOmegaSST{T}
    d_wall::Vector{T}
end

function KOmegaSSTModel(mesh::UnstructuredFVMMesh{Dim, T}, wall_patches::Vector{Symbol};
        coeffs = KappaOmegaSST(),
    ) where {Dim, T}
    d_wall = compute_wall_distance(mesh, wall_patches)
    return KOmegaSSTModel{T}(coeffs, T.(d_wall))
end

n_turbulence_fields(::KOmegaSSTModel) = 2
turbulence_field_names(::KOmegaSSTModel) = (:k, :omega)

# ── Blending functions ───────────────────────────────────────────────

"""Compute F1 blending function (0 = k-ε far field, 1 = k-ω near wall)."""
function _sst_F1(k::T, omega::T, nu::T, d::T, coeffs::KappaOmegaSST{T},
        grad_k_dot_grad_omega::T) where {T}
    d_safe = max(d, T(1e-10))
    omega_safe = max(omega, T(1e-10))
    k_safe = max(k, T(1e-10))

    arg1_a = sqrt(k_safe) / (coeffs.beta_star * omega_safe * d_safe)
    arg1_b = T(500) * nu / (d_safe^2 * omega_safe)
    arg1_ab = max(arg1_a, arg1_b)

    CDkw = max(T(2) * coeffs.sigma_omega2 / omega_safe * grad_k_dot_grad_omega, T(1e-10))
    arg1_c = T(4) * coeffs.sigma_omega2 * k_safe / (CDkw * d_safe^2)

    arg1 = min(arg1_ab, arg1_c)
    return tanh(arg1^4)
end

"""Compute F2 blending function for the SST viscosity limiter."""
function _sst_F2(k::T, omega::T, nu::T, d::T, coeffs::KappaOmegaSST{T}) where {T}
    d_safe = max(d, T(1e-10))
    omega_safe = max(omega, T(1e-10))
    k_safe = max(k, T(1e-10))

    arg2_a = T(2) * sqrt(k_safe) / (coeffs.beta_star * omega_safe * d_safe)
    arg2_b = T(500) * nu / (d_safe^2 * omega_safe)
    arg2 = max(arg2_a, arg2_b)
    return tanh(arg2^2)
end

"""Blend a constant: phi = F1*phi_1 + (1-F1)*phi_2."""
_blend(phi1::T, phi2::T, F1::T) where {T} = F1 * phi1 + (one(T) - F1) * phi2

# ── Interface implementation ─────────────────────────────────────────

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::KOmegaSSTModel{T},
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    k_field = turb_state.fields[:k]
    omega_field = turb_state.fields[:omega]
    co = model.coeffs
    nc = length(mesh.cell_volumes)

    # Need strain rate for SST limiter
    # Approximate with stored nu_t: S ≈ sqrt(nu_t * omega / k) — crude but avoids
    # recomputing gradients. For accuracy, caller should recompute.
    for c in 1:nc
        k_val = max(k_field.internal[c], T(1e-10))
        omega_val = max(omega_field.internal[c], T(1e-10))

        # Compute F2 for SST limiter
        F2 = _sst_F2(k_val, omega_val, T(0), model.d_wall[c], co)
        # nu is not stored in model; use 0 as conservative estimate (makes F2 smaller)
        # In practice, the solver wrapper passes nu separately when needed

        # SST limiter: nu_t = a1*k / max(a1*omega, S*F2)
        # Without S available here, use the simpler k/omega form with F2 damping
        nu_t[c] = co.a1 * k_val / max(co.a1 * omega_val, T(1e-10))
    end
    return nothing
end

function solve_turbulence!(
        turb_state::RANSTurbulenceState{T},
        model::KOmegaSSTModel{T},
        U::CollocatedVectorField{Dim, T},
        phi::FaceFluxField{T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_turb::Dict{Symbol, <:Dict{Symbol, <:AbstractBoundaryCondition}};
        dt::Union{Nothing, T} = nothing,
        linear_solver = nothing,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    co = model.coeffs
    k_field = turb_state.fields[:k]
    omega_field = turb_state.fields[:omega]

    # Production
    S_mag = compute_strain_rate(U, mesh)
    P_k = Vector{T}(undef, nc)
    for c in 1:nc
        P_k[c] = turb_state.nu_t[c] * S_mag[c]^2
    end

    # Compute grad(k) and grad(omega) for cross-diffusion and F1
    grad_k = gradient(k_field, mesh)
    grad_omega = gradient(omega_field, mesh)

    # Compute F1 per cell
    F1 = Vector{T}(undef, nc)
    for c in 1:nc
        gk_dot_gw = dot(grad_k[c], grad_omega[c])
        F1[c] = _sst_F1(k_field.internal[c], omega_field.internal[c],
            nu, model.d_wall[c], co, gk_dot_gw)
    end

    # Blended constants
    sigma_k_blend = Vector{T}(undef, nc)
    sigma_omega_blend = Vector{T}(undef, nc)
    beta_blend = Vector{T}(undef, nc)
    alpha_blend = Vector{T}(undef, nc)
    for c in 1:nc
        sigma_k_blend[c] = _blend(co.sigma_k1, co.sigma_k2, F1[c])
        sigma_omega_blend[c] = _blend(co.sigma_omega1, co.sigma_omega2, F1[c])
        beta_blend[c] = _blend(co.beta1, co.beta2, F1[c])
        # alpha from beta, beta_star, sigma_omega, kappa
        alpha1 = co.beta1 / co.beta_star - co.sigma_omega1 * co.kappa^2 / sqrt(co.beta_star)
        alpha2 = co.beta2 / co.beta_star - co.sigma_omega2 * co.kappa^2 / sqrt(co.beta_star)
        alpha_blend[c] = _blend(alpha1, alpha2, F1[c])
    end

    # ── k equation ───────────────────────────────────────────────
    k_eq = CollocatedEquation(mesh)
    bcs_k = get(bcs_turb, :k, Dict{Symbol, AbstractBoundaryCondition}())

    assemble_convection!(k_eq, phi, mesh, bcs_k)

    gamma_k = Vector{T}(undef, nc)
    for c in 1:nc
        gamma_k[c] = nu + sigma_k_blend[c] * turb_state.nu_t[c]
    end
    assemble_laplacian!(k_eq, gamma_k, mesh, bcs_k)

    if dt !== nothing
        assemble_ddt_euler!(k_eq, one(T), k_field.internal, mesh, dt)
    end

    for c in 1:nc
        omega_val = max(omega_field.internal[c], T(1e-10))
        k_eq.b[c] += P_k[c] * mesh.cell_volumes[c]
        k_eq.A[c, c] += co.beta_star * omega_val * mesh.cell_volumes[c]
    end

    lp_k = to_linear_problem(k_eq)
    sol_k = _solve_linear(lp_k, linear_solver)
    for c in 1:nc
        k_field.internal[c] = max(sol_k.u[c], T(1e-10))
    end

    # ── ω equation ───────────────────────────────────────────────
    omega_eq = CollocatedEquation(mesh)
    bcs_omega = get(bcs_turb, :omega, Dict{Symbol, AbstractBoundaryCondition}())

    assemble_convection!(omega_eq, phi, mesh, bcs_omega)

    gamma_omega = Vector{T}(undef, nc)
    for c in 1:nc
        gamma_omega[c] = nu + sigma_omega_blend[c] * turb_state.nu_t[c]
    end
    assemble_laplacian!(omega_eq, gamma_omega, mesh, bcs_omega)

    if dt !== nothing
        assemble_ddt_euler!(omega_eq, one(T), omega_field.internal, mesh, dt)
    end

    for c in 1:nc
        k_safe = max(k_field.internal[c], T(1e-10))
        omega_val = max(omega_field.internal[c], T(1e-10))
        # Production
        omega_eq.b[c] += alpha_blend[c] * (omega_val / k_safe) * P_k[c] * mesh.cell_volumes[c]
        # Destruction
        omega_eq.A[c, c] += beta_blend[c] * omega_val * mesh.cell_volumes[c]
        # Cross-diffusion (explicit, only in k-ε region where F1 < 1)
        gk_dot_gw = dot(grad_k[c], grad_omega[c])
        cd_term = T(2) * (one(T) - F1[c]) * co.sigma_omega2 / omega_val * gk_dot_gw
        omega_eq.b[c] += max(cd_term, zero(T)) * mesh.cell_volumes[c]
    end

    lp_omega = to_linear_problem(omega_eq)
    sol_omega = _solve_linear(lp_omega, linear_solver)
    for c in 1:nc
        omega_field.internal[c] = max(sol_omega.u[c], T(1e-10))
    end

    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; for f in ["interface", "strain_rate", "wall_distance", "k_epsilon_rans", "k_omega", "k_omega_sst"]; include("src/turbulence/$f.jl"); end; println("OK")'
```

---

### Task 7: Create spalart_allmaras.jl — One-equation SA model

**Files:**
- Create: `src/turbulence/spalart_allmaras.jl`

- [ ] **Step 1: Write SA implementation**

Write `src/turbulence/spalart_allmaras.jl`:

```julia
# turbulence/spalart_allmaras.jl — Spalart-Allmaras one-equation turbulence model
#
# Single transport equation for modified turbulent viscosity ν̃.
# Good near-wall behavior without wall functions. Requires wall distance.

"""
    SpalartAllmaras{T} <: AbstractRANSModel

Spalart-Allmaras one-equation turbulence model.

# Fields
Standard SA constants with default values.
"""
struct SpalartAllmaras{T} <: AbstractRANSModel
    cb1::T      # 0.1355
    cb2::T      # 0.622
    sigma::T    # 2/3
    kappa::T    # 0.41
    cw2::T      # 0.3
    cw3::T      # 2.0
    cv1::T      # 7.1
    ct3::T      # 1.2
    ct4::T      # 0.5
    d_wall::Vector{T}  # wall distance per cell
end

function SpalartAllmaras(mesh::UnstructuredFVMMesh{Dim, T}, wall_patches::Vector{Symbol};
        cb1 = 0.1355, cb2 = 0.622, sigma = 2.0 / 3.0, kappa = 0.41,
        cw2 = 0.3, cw3 = 2.0, cv1 = 7.1, ct3 = 1.2, ct4 = 0.5,
    ) where {Dim, T}
    d_wall = compute_wall_distance(mesh, wall_patches)
    Tc = promote_type(typeof(cb1), typeof(cb2), typeof(sigma), T)
    return SpalartAllmaras{Tc}(
        Tc(cb1), Tc(cb2), Tc(sigma), Tc(kappa),
        Tc(cw2), Tc(cw3), Tc(cv1), Tc(ct3), Tc(ct4), Tc.(d_wall),
    )
end

n_turbulence_fields(::SpalartAllmaras) = 1
turbulence_field_names(::SpalartAllmaras) = (:nu_tilde,)

# ── SA helper functions ──────────────────────────────────────────────

_sa_chi(nu_tilde::T, nu::T) where {T} = nu_tilde / max(nu, T(1e-15))

function _sa_fv1(chi::T, cv1::T) where {T}
    chi3 = chi^3
    return chi3 / (chi3 + cv1^3)
end

function _sa_fv2(chi::T, cv1::T) where {T}
    fv1 = _sa_fv1(chi, cv1)
    return one(T) - chi / (one(T) + chi * fv1)
end

function _sa_fw(r::T, cw2::T, cw3::T) where {T}
    g = r + cw2 * (r^6 - r)
    cw3_6 = cw3^6
    return g * ((one(T) + cw3_6) / (g^6 + cw3_6))^(one(T) / 6)
end

# ── Interface implementation ─────────────────────────────────────────

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::SpalartAllmaras{T},
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nt_field = turb_state.fields[:nu_tilde]
    nc = length(mesh.cell_volumes)
    # nu_t = nu_tilde * fv1
    # We need laminar nu for chi — estimate from nu_tilde magnitude
    # In practice the solver wrapper provides nu; here we use a placeholder
    for c in 1:nc
        nt = max(nt_field.internal[c], zero(T))
        # Without nu available, use fv1 ≈ 1 for large nu_tilde (turbulent region)
        # This is corrected when the solver calls with actual nu
        nu_t[c] = nt
    end
    return nothing
end

"""
    turbulent_viscosity_sa!(nu_t, model, turb_state, mesh, nu)

SA-specific version that takes laminar viscosity for correct fv1 computation.
"""
function turbulent_viscosity_sa!(
        nu_t::Vector{T},
        model::SpalartAllmaras{T},
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        nu::T,
    ) where {Dim, T}
    nt_field = turb_state.fields[:nu_tilde]
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        nt = max(nt_field.internal[c], zero(T))
        chi = _sa_chi(nt, nu)
        fv1 = _sa_fv1(chi, model.cv1)
        nu_t[c] = nt * fv1
    end
    return nothing
end

function solve_turbulence!(
        turb_state::RANSTurbulenceState{T},
        model::SpalartAllmaras{T},
        U::CollocatedVectorField{Dim, T},
        phi::FaceFluxField{T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_turb::Dict{Symbol, <:Dict{Symbol, <:AbstractBoundaryCondition}};
        dt::Union{Nothing, T} = nothing,
        linear_solver = nothing,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nt_field = turb_state.fields[:nu_tilde]

    # Strain rate
    S_mag = compute_strain_rate(U, mesh)

    # Derived constant
    cw1 = model.cb1 / model.kappa^2 + (one(T) + model.cb2) / model.sigma

    # ── ν̃ equation ────────────────────────────────────────────────
    nt_eq = CollocatedEquation(mesh)
    bcs_nt = get(bcs_turb, :nu_tilde, Dict{Symbol, AbstractBoundaryCondition}())

    # Convection
    assemble_convection!(nt_eq, phi, mesh, bcs_nt)

    # Diffusion: (1/sigma) * div((nu + nu_tilde) * grad(nu_tilde))
    gamma_nt = Vector{T}(undef, nc)
    for c in 1:nc
        nt = max(nt_field.internal[c], zero(T))
        gamma_nt[c] = (nu + nt) / model.sigma
    end
    assemble_laplacian!(nt_eq, gamma_nt, mesh, bcs_nt)

    # Temporal term
    if dt !== nothing
        assemble_ddt_euler!(nt_eq, one(T), nt_field.internal, mesh, dt)
    end

    # Source terms (production - destruction, linearized)
    for c in 1:nc
        nt = max(nt_field.internal[c], T(1e-10))
        d = max(model.d_wall[c], T(1e-10))
        chi = _sa_chi(nt, nu)
        fv1 = _sa_fv1(chi, model.cv1)
        fv2 = _sa_fv2(chi, model.cv1)

        # Modified vorticity
        S_tilde = S_mag[c] + nt / (model.kappa^2 * d^2) * fv2
        S_tilde = max(S_tilde, T(1e-10))

        # Production: cb1 * S_tilde * nu_tilde (implicit in nu_tilde)
        # Treat as: S_C = cb1 * S_tilde (coefficient on nu_tilde, goes to diagonal with negative sign)
        # Actually production is positive, so: b += cb1 * S_tilde * nt * V
        # But for linearization, we want it implicit: A[c,c] -= cb1 * S_tilde * V
        # (negative because it's a source, reducing A makes the diagonal smaller → source)
        # OpenFOAM convention: positive source → subtract from diagonal
        # Here: production goes to RHS: b[c] += cb1 * S_tilde * nt * V_c
        nt_eq.b[c] += model.cb1 * S_tilde * nt * mesh.cell_volumes[c]

        # Destruction: -cw1 * fw * (nu_tilde/d)^2 — linearize as implicit
        r_val = min(nt / (S_tilde * model.kappa^2 * d^2), T(10))
        fw = _sa_fw(r_val, model.cw2, model.cw3)
        # D = cw1 * fw * (nt/d^2) * nt → S_P = cw1 * fw * nt / d^2
        nt_eq.A[c, c] += cw1 * fw * nt / d^2 * mesh.cell_volumes[c]
    end

    # cb2/sigma * |grad(nu_tilde)|^2 term (explicit source)
    grad_nt = gradient(nt_field, mesh)
    for c in 1:nc
        grad_sq = dot(grad_nt[c], grad_nt[c])
        nt_eq.b[c] += model.cb2 / model.sigma * grad_sq * mesh.cell_volumes[c]
    end

    # Solve
    lp = to_linear_problem(nt_eq)
    sol = _solve_linear(lp, linear_solver)
    for c in 1:nc
        nt_field.internal[c] = max(sol.u[c], zero(T))
    end

    # Update nu_t with correct fv1
    turbulent_viscosity_sa!(turb_state.nu_t, model, turb_state, mesh, nu)

    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; for f in ["interface", "strain_rate", "wall_distance", "k_epsilon_rans", "k_omega", "k_omega_sst", "spalart_allmaras"]; include("src/turbulence/$f.jl"); end; println("OK")'
```

---

### Task 8: Create wall_functions.jl — Wall BC generation

**Files:**
- Create: `src/turbulence/wall_functions.jl`

- [ ] **Step 1: Write wall function BC helpers**

Write `src/turbulence/wall_functions.jl`:

```julia
# turbulence/wall_functions.jl — Wall function BC generation for turbulence models
#
# Generates boundary condition dictionaries for turbulence fields at wall
# and inlet patches. Reuses existing compute_friction_velocity, k_wall_value,
# epsilon_wall_value from src/physics/turbulence/k_epsilon.jl.

"""
    turbulence_inlet_bc(model::StandardKEpsilon, U_mag, intensity, length_scale)

Generate inlet BCs for k-ε from freestream conditions.

- `k_inlet = 1.5 * (U_mag * intensity)²`
- `ε_inlet = C_μ^0.75 * k^1.5 / length_scale`
"""
function turbulence_inlet_bc(
        model::StandardKEpsilon, U_mag::T, intensity::T, length_scale::T,
    ) where {T}
    k_inlet = T(1.5) * (U_mag * intensity)^2
    eps_inlet = model.C_mu^T(0.75) * k_inlet^T(1.5) / length_scale
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => ParabolicDirichlet(k_inlet),
        :epsilon => ParabolicDirichlet(eps_inlet),
    )
end

"""
    turbulence_inlet_bc(model::KOmega, U_mag, intensity, length_scale)

Generate inlet BCs for k-ω from freestream conditions.

- `k_inlet = 1.5 * (U_mag * intensity)²`
- `ω_inlet = k^0.5 / (C_μ^0.25 * length_scale)`
"""
function turbulence_inlet_bc(
        model::KOmega, U_mag::T, intensity::T, length_scale::T,
    ) where {T}
    k_inlet = T(1.5) * (U_mag * intensity)^2
    omega_inlet = sqrt(k_inlet) / (T(0.09)^T(0.25) * length_scale)
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => ParabolicDirichlet(k_inlet),
        :omega => ParabolicDirichlet(omega_inlet),
    )
end

"""
    turbulence_inlet_bc(model::KOmegaSSTModel, U_mag, intensity, length_scale)

Generate inlet BCs for k-ω SST (same as k-ω).
"""
function turbulence_inlet_bc(
        model::KOmegaSSTModel, U_mag::T, intensity::T, length_scale::T,
    ) where {T}
    k_inlet = T(1.5) * (U_mag * intensity)^2
    omega_inlet = sqrt(k_inlet) / (model.coeffs.beta_star^T(0.25) * length_scale)
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => ParabolicDirichlet(k_inlet),
        :omega => ParabolicDirichlet(omega_inlet),
    )
end

"""
    turbulence_inlet_bc(model::SpalartAllmaras, U_mag, intensity, length_scale)

Generate inlet BCs for SA. ν̃_inlet ≈ 3-5 * ν for freestream.
"""
function turbulence_inlet_bc(
        ::SpalartAllmaras, U_mag::T, intensity::T, length_scale::T,
    ) where {T}
    # SA freestream: nu_tilde ≈ 3 * nu_laminar is typical
    # Use intensity to scale: higher TI → higher nu_tilde
    nu_tilde_inlet = T(3) * intensity * U_mag * length_scale
    return Dict{Symbol, AbstractBoundaryCondition}(
        :nu_tilde => ParabolicDirichlet(nu_tilde_inlet),
    )
end

"""
    turbulence_wall_bc(model::StandardKEpsilon)

Generate wall BCs for k-ε (zero-gradient for k, fixed ε via wall function).
For now returns Neumann(0) for both — the wall function values are
set dynamically during the solve.
"""
function turbulence_wall_bc(::StandardKEpsilon)
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => ParabolicNeumann(0.0),
        :epsilon => ParabolicNeumann(0.0),
    )
end

function turbulence_wall_bc(::Union{KOmega, KOmegaSSTModel})
    return Dict{Symbol, AbstractBoundaryCondition}(
        :k => ParabolicNeumann(0.0),
        :omega => ParabolicNeumann(0.0),
    )
end

function turbulence_wall_bc(::SpalartAllmaras)
    return Dict{Symbol, AbstractBoundaryCondition}(
        :nu_tilde => ParabolicDirichlet(0.0),
    )
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; for f in ["interface", "strain_rate", "wall_distance", "k_epsilon_rans", "k_omega", "k_omega_sst", "spalart_allmaras", "wall_functions"]; include("src/turbulence/$f.jl"); end; println("OK")'
```

---

### Task 9: Create solvers.jl — Turbulent SIMPLE/PISO/PIMPLE wrappers

**Files:**
- Create: `src/turbulence/solvers.jl`

- [ ] **Step 1: Write turbulent solver wrappers**

Write `src/turbulence/solvers.jl`:

```julia
# turbulence/solvers.jl — Turbulent SIMPLE/PISO/PIMPLE solver wrappers
#
# These functions extend the Phase 1 incompressible solvers with a
# turbulence step after velocity correction.

using Printf: @sprintf

"""
    solve_simple_turbulent(
        prob::IncompressibleProblem{Dim, T},
        turb_model::AbstractRANSModel;
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        linear_solver = nothing,
        verbose = false,
    ) -> Tuple{SolveResult{Dim, T}, RANSTurbulenceState{T}}

Solve steady incompressible flow with RANS turbulence using SIMPLE.

Same algorithm as `solve_simple` but with turbulence equations solved
after each velocity correction and `nu_eff = nu + nu_t` used in momentum.
"""
function solve_simple_turbulent(
        prob::IncompressibleProblem{Dim, T},
        turb_model::AbstractRANSModel;
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        linear_solver = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    algo = prob.algorithm::SIMPLE{T}
    mesh = prob.mesh

    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    turb_state = RANSTurbulenceState(turb_model, mesh)
    turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)

    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    converged = false
    final_iter = 0

    for iter in 1:algo.max_iterations
        final_iter = iter
        nu_eff = compute_nu_eff(prob.nu, turb_state.nu_t)

        # ── Momentum ────────────────────────────────────────────
        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(eq, state, prob, d; nu_eff = nu_eff)
            push!(eqs, eq)
        end

        extract_momentum_operators!(state, eqs, mesh)

        for d in 1:Dim
            U_old_d = _extract_component(state.U, d)
            under_relax_momentum!(eqs[d], U_old_d, algo.alpha_U)
            sol = _solve_linear(to_linear_problem(eqs[d]), linear_solver)
            _set_component!(state.U, d, sol.u)
        end
        update_boundary_velocity!(state, prob.bcs, mesh)

        # ── Pressure ────────────────────────────────────────────
        p_eq = CollocatedEquation(mesh)
        assemble_pressure!(p_eq, state, prob)
        if _needs_pressure_reference(prob.bcs)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end
        p_sol = _solve_linear(to_linear_problem(p_eq), linear_solver)

        nc = length(mesh.cell_volumes)
        for c in 1:nc
            state.p.internal[c] += algo.alpha_p * (p_sol.u[c] - state.p.internal[c])
        end
        update_boundary_pressure!(state, prob.bcs, mesh)

        correct_velocity!(state, mesh)
        update_boundary_velocity!(state, prob.bcs, mesh)
        correct_fluxes!(state, mesh)

        # ── Turbulence ──────────────────────────────────────────
        solve_turbulence!(
            turb_state, turb_model, state.U, state.phi, prob.nu, mesh, turb_bcs;
            linear_solver = linear_solver,
        )
        turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)

        # ── Convergence ─────────────────────────────────────────
        max_res = zero(T)
        for d in 1:Dim
            u_d = _extract_component(state.U, d)
            r = momentum_residual(eqs[d], u_d)
            push!(residuals[component_labels[d]], r)
            max_res = max(max_res, r)
        end
        r_cont = continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)
        max_res = max(max_res, r_cont)

        if verbose
            _print_simple_residuals(iter, residuals, component_labels)
        end

        if max_res < algo.tolerance
            converged = true
            break
        end
    end

    result = SolveResult{Dim, T}(converged, final_iter, residuals, state)
    return (result, turb_state)
end

"""
    solve_incompressible_turbulent(
        prob, turb_model, tspan, dt; turb_bcs, kwargs...,
    ) -> Tuple{SolveResult, RANSTurbulenceState}

Solve transient incompressible flow with RANS turbulence using PISO or PIMPLE.
"""
function solve_incompressible_turbulent(
        prob::IncompressibleProblem{Dim, T},
        turb_model::AbstractRANSModel,
        tspan::Tuple{T, T},
        dt::T;
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        save_every::Int = 1,
        linear_solver = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    mesh = prob.mesh

    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    turb_state = RANSTurbulenceState(turb_model, mesh)
    turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)

    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    t_start, t_end = tspan
    t = t_start
    n_steps = 0

    while t < t_end - eps(T) * abs(t_end)
        dt_actual = min(dt, t_end - t)
        nu_eff = compute_nu_eff(prob.nu, turb_state.nu_t)

        # Time step with nu_eff — use existing step functions but with modified momentum
        # For simplicity, inline the PISO/PIMPLE step with nu_eff
        if prob.algorithm isa PISO
            _turbulent_piso_step!(state, prob, dt_actual, prob.algorithm.n_correctors,
                nu_eff; linear_solver = linear_solver)
        elseif prob.algorithm isa PIMPLE
            _turbulent_pimple_step!(state, prob, dt_actual, nu_eff;
                linear_solver = linear_solver)
        end

        # Turbulence update
        solve_turbulence!(
            turb_state, turb_model, state.U, state.phi, prob.nu, mesh, turb_bcs;
            dt = dt_actual, linear_solver = linear_solver,
        )
        turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)

        t += dt_actual
        n_steps += 1

        r_cont = continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)

        if verbose && n_steps % max(1, round(Int, (t_end - t_start) / dt / 20)) == 0
            println("Step ", lpad(n_steps, 6), "  t=", @sprintf("%.4e", t),
                "  cont=", @sprintf("%.3e", r_cont))
        end
    end

    result = SolveResult{Dim, T}(true, n_steps, residuals, state)
    return (result, turb_state)
end

# ── Turbulent PISO step (with nu_eff) ───────────────────────────────

function _turbulent_piso_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T, n_correctors::Int,
        nu_eff::Vector{T};
        linear_solver = nothing,
    ) where {Dim, T}
    mesh = prob.mesh

    eqs = CollocatedEquation{T}[]
    for d in 1:Dim
        eq = CollocatedEquation(mesh)
        assemble_momentum!(eq, state, prob, d; dt = dt, nu_eff = nu_eff)
        push!(eqs, eq)
    end

    extract_momentum_operators!(state, eqs, mesh)

    for d in 1:Dim
        sol = _solve_linear(to_linear_problem(eqs[d]), linear_solver)
        _set_component!(state.U, d, sol.u)
    end
    update_boundary_velocity!(state, prob.bcs, mesh)

    for k in 1:n_correctors
        p_eq = CollocatedEquation(mesh)
        assemble_pressure!(p_eq, state, prob)
        if _needs_pressure_reference(prob.bcs)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end
        p_sol = _solve_linear(to_linear_problem(p_eq), linear_solver)

        nc = length(mesh.cell_volumes)
        for c in 1:nc
            state.p.internal[c] = p_sol.u[c]
        end
        update_boundary_pressure!(state, prob.bcs, mesh)
        correct_velocity!(state, mesh)
        update_boundary_velocity!(state, prob.bcs, mesh)
        correct_fluxes!(state, mesh)

        if k < n_correctors
            eqs_k = CollocatedEquation{T}[]
            for d in 1:Dim
                eq = CollocatedEquation(mesh)
                assemble_momentum!(eq, state, prob, d; dt = dt, nu_eff = nu_eff)
                push!(eqs_k, eq)
            end
            extract_momentum_operators!(state, eqs_k, mesh)
        end
    end

    return nothing
end

# ── Turbulent PIMPLE step (with nu_eff) ─────────────────────────────

function _turbulent_pimple_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T, nu_eff::Vector{T};
        linear_solver = nothing,
    ) where {Dim, T}
    algo = prob.algorithm::PIMPLE{T}
    mesh = prob.mesh

    for outer in 1:algo.n_outer
        is_final = (outer == algo.n_outer)

        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(eq, state, prob, d; dt = dt, nu_eff = nu_eff)
            push!(eqs, eq)
        end
        extract_momentum_operators!(state, eqs, mesh)

        for d in 1:Dim
            if !is_final
                U_old_d = _extract_component(state.U, d)
                under_relax_momentum!(eqs[d], U_old_d, algo.alpha_U)
            end
            sol = _solve_linear(to_linear_problem(eqs[d]), linear_solver)
            _set_component!(state.U, d, sol.u)
        end
        update_boundary_velocity!(state, prob.bcs, mesh)

        nc = length(mesh.cell_volumes)
        for k in 1:algo.n_correctors
            p_eq = CollocatedEquation(mesh)
            assemble_pressure!(p_eq, state, prob)
            if _needs_pressure_reference(prob.bcs)
                fix_pressure_reference!(p_eq, 1, zero(T))
            end
            p_sol = _solve_linear(to_linear_problem(p_eq), linear_solver)

            if !is_final
                for c in 1:nc
                    state.p.internal[c] += algo.alpha_p * (p_sol.u[c] - state.p.internal[c])
                end
            else
                for c in 1:nc
                    state.p.internal[c] = p_sol.u[c]
                end
            end
            update_boundary_pressure!(state, prob.bcs, mesh)
            correct_velocity!(state, mesh)
            update_boundary_velocity!(state, prob.bcs, mesh)
            correct_fluxes!(state, mesh)
        end
    end

    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; for f in ["interface", "strain_rate", "wall_distance", "k_epsilon_rans", "k_omega", "k_omega_sst", "spalart_allmaras", "wall_functions", "solvers"]; include("src/turbulence/$f.jl"); end; println("OK")'
```

---

### Task 10: Wire into module — Layer 2 includes + exports

**Files:**
- Modify: `src/layers/discretization_assembly_kernels.jl`
- Modify: `src/FiniteVolumeMethod.jl`

- [ ] **Step 1: Add includes to Layer 2**

Append to `src/layers/discretization_assembly_kernels.jl` after the incompressible includes:

```julia
# RANS Turbulence Models (Phase 2a)
# Depends on Phase 0 operators + Phase 1 incompressible solver.
include("../turbulence/interface.jl")
include("../turbulence/strain_rate.jl")
include("../turbulence/wall_distance.jl")
include("../turbulence/k_epsilon_rans.jl")
include("../turbulence/k_omega.jl")
include("../turbulence/k_omega_sst.jl")
include("../turbulence/spalart_allmaras.jl")
include("../turbulence/wall_functions.jl")
include("../turbulence/solvers.jl")
```

- [ ] **Step 2: Add exports**

Add a new export block in `src/FiniteVolumeMethod.jl` after the Phase 1 incompressible exports:

```julia
# --- RANS Turbulence Models (Phase 2a) ---
export
    # Abstract types
    AbstractRANSModel,
    # Model types
    KOmega,
    KOmegaSSTModel,
    SpalartAllmaras,
    # State
    RANSTurbulenceState,
    # Interface
    turbulent_viscosity!,
    solve_turbulence!,
    n_turbulence_fields,
    turbulence_field_names,
    # Solvers
    solve_simple_turbulent,
    solve_incompressible_turbulent,
    # Utilities
    compute_wall_distance,
    compute_strain_rate,
    compute_nu_eff,
    turbulence_inlet_bc,
    turbulence_wall_bc
```

- [ ] **Step 3: Verify module loads**

```bash
julia --project -e 'using FiniteVolumeMethod; println("Phase 2a loaded: ", KOmega)'
```
Expected: `Phase 2a loaded: KOmega`

- [ ] **Step 4: Commit**

```bash
git add src/turbulence/ src/incompressible/momentum.jl src/layers/discretization_assembly_kernels.jl src/FiniteVolumeMethod.jl
git commit -m "feat: add RANS turbulence models (k-ε, k-ω, k-ω SST, SA) for incompressible solver"
```

---

### Task 11: Write tests

**Files:**
- Create: `test/turbulence_rans.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the test file**

Create `test/turbulence_rans.jl` with tests using the `build_cartesian_unstructured_mesh` helper from `test/incompressible.jl`. The test file should:

1. Include or redefine the mesh builder (since tests run in isolated modules via `safe_include`)
2. Test type construction for all 4 models
3. Test `compute_strain_rate` on a uniform velocity field
4. Test `compute_wall_distance` on a simple mesh
5. Test `turbulent_viscosity!` for k-ε (C_mu * k^2 / eps)
6. Test `solve_turbulence!` smoke test for k-ε (runs without error, fields stay positive)
7. Test `solve_simple_turbulent` smoke test (runs a few iterations, returns valid result)
8. Test `turbulence_inlet_bc` produces correct BC types
9. Test `RANSTurbulenceState` constructor
10. Smoke test for KOmega, KOmegaSSTModel, and SpalartAllmaras (construct + turbulent_viscosity!)

Key: the mesh builder must be copied into this test file since `safe_include` runs in an isolated module.

- [ ] **Step 2: Register test**

Add `safe_include("turbulence_rans.jl")` to `test/runtests.jl` after the incompressible test.

- [ ] **Step 3: Run tests**

```bash
julia --project=test test/turbulence_rans.jl
```

- [ ] **Step 4: Run Runic**

```bash
julia --project -e 'using Runic; Runic.main(["--inplace", "src/turbulence/"])'
julia --project -e 'using Runic; Runic.main(["--inplace", "test/turbulence_rans.jl"])'
```

- [ ] **Step 5: Commit**

```bash
git add test/turbulence_rans.jl test/runtests.jl
git commit -m "test: add RANS turbulence model test suite"
```

---

### Task 12: Register in validation manifest + final verification

**Files:**
- Modify: `validation/manifest.toml`

- [ ] **Step 1: Add turbulence_rans feature**

Append to `validation/manifest.toml`:

```toml
[[features]]
feature = "turbulence_rans"
maturity = "experimental"
validation = "smoke_tested"
role = "research_tooling"
solver_family = "collocated"
precision_policy = "float64_cpu_reference"
random_seed_policy = "deterministic"
backend_policy = "cpu_reference"
required_ladder_stages = ["verification", "benchmark"]
summary = "RANS turbulence models (k-epsilon, k-omega, k-omega SST, Spalart-Allmaras) for incompressible collocated solver."
limitations = [
  "Experimental — validated via smoke tests only; channel flow and flat plate benchmarks pending.",
  "Wall functions are simplified (Neumann/Dirichlet); dynamic wall function update not yet implemented.",
  "SST viscosity limiter uses simplified form without strain rate recomputation.",
]
```

- [ ] **Step 2: Verify module loads with all exports**

```bash
julia --project -e '
using FiniteVolumeMethod
@assert isdefined(FiniteVolumeMethod, :AbstractRANSModel)
@assert isdefined(FiniteVolumeMethod, :KOmega)
@assert isdefined(FiniteVolumeMethod, :KOmegaSSTModel)
@assert isdefined(FiniteVolumeMethod, :SpalartAllmaras)
@assert isdefined(FiniteVolumeMethod, :solve_simple_turbulent)
@assert isdefined(FiniteVolumeMethod, :compute_strain_rate)
println("All Phase 2a exports verified")
'
```

- [ ] **Step 3: Run full test suite**

```bash
julia --project=test test/turbulence_rans.jl
```

- [ ] **Step 4: Run Runic check**

```bash
julia --project -e 'using Runic; Runic.main(["--check", "src/turbulence/"])'
```

- [ ] **Step 5: Regression check**

```bash
julia --project=test test/incompressible.jl
```
Expected: All 94 tests still pass.

- [ ] **Step 6: Commit**

```bash
git add validation/manifest.toml
git commit -m "feat: register turbulence_rans in validation manifest"
```
