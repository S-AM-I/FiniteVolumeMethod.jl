# LES & Hybrid Turbulence Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add algebraic LES models (Smagorinsky, WALE, dynamic Smagorinsky) and a DDES hybrid model to the incompressible solver, completing the turbulence modeling capability.

**Architecture:** Five new files in `src/turbulence/` wired into Layer 2 after existing RANS turbulence. LES models compute ν_sgs algebraically from velocity gradients — no transport equations. An `_update_turbulence!` dispatcher routes LES (direct ν_t computation) vs RANS (solve + update) for the existing solver wrappers. DDES wraps Spalart-Allmaras with a modified length scale.

**Tech Stack:** Julia, LinearAlgebra (dot, norm), StaticArrays (SVector), Phase 0 collocated operators (gradient), Phase 2a turbulence infrastructure (AbstractTurbulenceModel, compute_strain_rate, compute_wall_distance, SpalartAllmaras).

---

## File Map

| File | Purpose | Creates/Modifies |
|------|---------|-----------------|
| `src/turbulence/les_types.jl` | AbstractLESModel, AbstractHybridModel, LESTurbulenceState, compute_filter_width, _update_turbulence!, LES no-ops | Create |
| `src/turbulence/smagorinsky.jl` | Smagorinsky SGS model | Create |
| `src/turbulence/wale.jl` | WALE SGS model | Create |
| `src/turbulence/dynamic_smagorinsky.jl` | Dynamic Smagorinsky with test filtering | Create |
| `src/turbulence/ddes.jl` | DDES hybrid wrapping Spalart-Allmaras | Create |
| `src/turbulence/solvers.jl` | Replace 2-step turbulence update with `_update_turbulence!` | Modify |
| `src/layers/discretization_assembly_kernels.jl` | Wire LES includes | Modify |
| `src/FiniteVolumeMethod.jl` | Add exports | Modify |
| `test/turbulence_les.jl` | All tests | Create |
| `test/runtests.jl` | Register test | Modify |
| `validation/manifest.toml` | Register feature | Modify |

---

### Task 1: Create les_types.jl — LES abstract types, state, filter width, dispatcher

**Files:**
- Create: `src/turbulence/les_types.jl`

- [ ] **Step 1: Write the LES types and utilities**

```julia
# turbulence/les_types.jl — Abstract types and utilities for LES models
#
# Defines the LES and hybrid model type hierarchy, the lightweight
# LES turbulence state (nu_t only, no transport fields), the grid
# filter width computation, and the _update_turbulence! dispatcher.

# ── Abstract types ───────────────────────────────────────────────────

"""
    AbstractLESModel <: AbstractTurbulenceModel

Supertype for Large Eddy Simulation subgrid-scale models.

LES models compute turbulent viscosity algebraically from the resolved
velocity field — no transport equations to solve.

Every concrete LES model must implement:
- `turbulent_viscosity!(nu_t, model, U, mesh)` — compute ν_sgs from velocity
"""
abstract type AbstractLESModel <: AbstractTurbulenceModel end

"""
    AbstractHybridModel <: AbstractTurbulenceModel

Supertype for hybrid RANS/LES turbulence models (DES, DDES, IDDES).
"""
abstract type AbstractHybridModel <: AbstractTurbulenceModel end

# ── LES no-ops ───────────────────────────────────────────────────────

n_turbulence_fields(::AbstractLESModel) = 0
turbulence_field_names(::AbstractLESModel) = ()

function solve_turbulence!(
        turb_state, model::AbstractLESModel,
        U, phi, nu, mesh, bcs_turb;
        dt = nothing, linear_solver = nothing,
    )
    return nothing
end

# ── LES state ────────────────────────────────────────────────────────

"""
    LESTurbulenceState{T}

Lightweight turbulence state for LES models. Only stores the per-cell
turbulent viscosity — no transport equation fields.

Compatible with solver wrappers via the `nu_t` field (duck typing).
"""
mutable struct LESTurbulenceState{T}
    nu_t::Vector{T}
end

"""
    LESTurbulenceState(mesh::UnstructuredFVMMesh{Dim, T})

Construct a zero-initialized LES state.
"""
function LESTurbulenceState(mesh::UnstructuredFVMMesh{Dim, T}) where {Dim, T}
    nc = length(mesh.cell_volumes)
    return LESTurbulenceState{T}(zeros(T, nc))
end

# ── Filter width ─────────────────────────────────────────────────────

"""
    compute_filter_width(mesh::UnstructuredFVMMesh{Dim, T}) -> Vector{T}

Compute the grid filter width per cell:
`Δ[c] = V_c^(1/Dim)`

For 3D: cube root of cell volume. For 2D: square root of cell area.
"""
function compute_filter_width(mesh::UnstructuredFVMMesh{Dim, T}) where {Dim, T}
    nc = length(mesh.cell_volumes)
    delta = Vector{T}(undef, nc)
    inv_dim = one(T) / T(Dim)
    for c in 1:nc
        delta[c] = mesh.cell_volumes[c]^inv_dim
    end
    return delta
end

# ── Turbulence update dispatcher ─────────────────────────────────────

"""
    _update_turbulence!(turb_state, turb_model::AbstractLESModel, state, prob, mesh, turb_bcs; kwargs...)

Update turbulent viscosity for LES models (no transport equations —
directly computes ν_sgs from velocity).
"""
function _update_turbulence!(
        turb_state, turb_model::AbstractLESModel,
        state, prob, mesh, turb_bcs;
        dt = nothing, linear_solver = nothing,
    )
    turbulent_viscosity!(turb_state.nu_t, turb_model, state.U, mesh)
    return nothing
end

"""
    _update_turbulence!(turb_state, turb_model, state, prob, mesh, turb_bcs; kwargs...)

Update turbulent viscosity for RANS and hybrid models (solve transport
equations, then compute ν_t from the fields).
"""
function _update_turbulence!(
        turb_state, turb_model,
        state, prob, mesh, turb_bcs;
        dt = nothing, linear_solver = nothing,
    )
    solve_turbulence!(
        turb_state, turb_model, state.U, state.phi, prob.nu, mesh, turb_bcs;
        dt = dt, linear_solver = linear_solver,
    )
    turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/turbulence/les_types.jl"); println("OK")'
```

---

### Task 2: Create smagorinsky.jl — Smagorinsky SGS model

**Files:**
- Create: `src/turbulence/smagorinsky.jl`

- [ ] **Step 1: Write the Smagorinsky model**

```julia
# turbulence/smagorinsky.jl — Smagorinsky subgrid-scale model
#
# Simplest LES model: ν_sgs = (Cs · Δ)² · |S|
# where Cs is the Smagorinsky constant, Δ is the filter width,
# and |S| is the strain rate magnitude.

"""
    Smagorinsky{T} <: AbstractLESModel

Smagorinsky subgrid-scale model.

# Fields
- `Cs::T` — Smagorinsky constant (default 0.1, range 0.065–0.2)
- `delta::Vector{T}` — grid filter width per cell
"""
struct Smagorinsky{T} <: AbstractLESModel
    Cs::T
    delta::Vector{T}
end

"""
    Smagorinsky(mesh; Cs = 0.1)

Construct a Smagorinsky model, computing filter width from `mesh`.
"""
function Smagorinsky(mesh::UnstructuredFVMMesh{Dim, T}; Cs::Real = 0.1) where {Dim, T}
    delta = compute_filter_width(mesh)
    return Smagorinsky{T}(T(Cs), delta)
end

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::Smagorinsky{T},
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    S_mag = compute_strain_rate(U, mesh)
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        nu_t[c] = (model.Cs * model.delta[c])^2 * S_mag[c]
    end
    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/turbulence/les_types.jl"); include("src/turbulence/smagorinsky.jl"); println("OK")'
```

---

### Task 3: Create wale.jl — WALE SGS model

**Files:**
- Create: `src/turbulence/wale.jl`

- [ ] **Step 1: Write the WALE model**

```julia
# turbulence/wale.jl — WALE (Wall-Adapting Local Eddy-viscosity) SGS model
#
# Better near-wall behavior than Smagorinsky — ν_sgs vanishes at walls
# without explicit damping functions.
#
# ν_sgs = (Cw·Δ)² · (S_d:S_d)^(3/2) / ((S:S)^(5/2) + (S_d:S_d)^(5/4))
# where S_d is the traceless symmetric part of the squared velocity gradient.

"""
    WALE{T} <: AbstractLESModel

WALE (Wall-Adapting Local Eddy-viscosity) SGS model.

# Fields
- `Cw::T` — WALE constant (default 0.325)
- `delta::Vector{T}` — grid filter width per cell
"""
struct WALE{T} <: AbstractLESModel
    Cw::T
    delta::Vector{T}
end

"""
    WALE(mesh; Cw = 0.325)

Construct a WALE model, computing filter width from `mesh`.
"""
function WALE(mesh::UnstructuredFVMMesh{Dim, T}; Cw::Real = 0.325) where {Dim, T}
    delta = compute_filter_width(mesh)
    return WALE{T}(T(Cw), delta)
end

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::WALE{T},
        U::CollocatedVectorField{2, T},
        mesh::UnstructuredFVMMesh{2, T},
    ) where {T}
    nc = length(mesh.cell_volumes)
    grad_U = FiniteVolumeMethod._compute_velocity_gradients(U, mesh)

    for c in 1:nc
        g11 = grad_U[1][c][1]; g12 = grad_U[1][c][2]
        g21 = grad_U[2][c][1]; g22 = grad_U[2][c][2]

        # Squared velocity gradient g²_ij = g_ik * g_kj
        g2_11 = g11 * g11 + g12 * g21
        g2_12 = g11 * g12 + g12 * g22
        g2_21 = g21 * g11 + g22 * g21
        g2_22 = g21 * g12 + g22 * g22

        # Traceless symmetric part: S_d_ij = 0.5*(g²_ij + g²_ji) - (1/3)*δ_ij*g²_kk
        # For 2D, use (1/2)*δ_ij*trace instead of (1/3) since we're in 2D
        trace_g2 = g2_11 + g2_22
        sd_11 = T(0.5) * (g2_11 + g2_11) - T(0.5) * trace_g2
        sd_22 = T(0.5) * (g2_22 + g2_22) - T(0.5) * trace_g2
        sd_12 = T(0.5) * (g2_12 + g2_21)

        # S_d:S_d = sd_ij * sd_ij
        sd_sq = sd_11^2 + sd_22^2 + T(2) * sd_12^2

        # S:S (strain rate)
        S_11 = g11; S_22 = g22
        S_12 = T(0.5) * (g12 + g21)
        s_sq = S_11^2 + S_22^2 + T(2) * S_12^2

        # WALE viscosity
        denom = s_sq^T(2.5) + sd_sq^T(1.25)
        if denom > eps(T)
            nu_t[c] = (model.Cw * model.delta[c])^2 * sd_sq^T(1.5) / denom
        else
            nu_t[c] = zero(T)
        end
    end

    return nothing
end

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::WALE{T},
        U::CollocatedVectorField{3, T},
        mesh::UnstructuredFVMMesh{3, T},
    ) where {T}
    nc = length(mesh.cell_volumes)
    grad_U = FiniteVolumeMethod._compute_velocity_gradients(U, mesh)

    for c in 1:nc
        g = ntuple(i -> ntuple(j -> grad_U[i][c][j], Val(3)), Val(3))

        # g²_ij = g_ik * g_kj
        g2 = ntuple(Val(3)) do i
            ntuple(Val(3)) do j
                g[i][1] * g[1][j] + g[i][2] * g[2][j] + g[i][3] * g[3][j]
            end
        end

        trace_g2 = g2[1][1] + g2[2][2] + g2[3][3]

        # S_d_ij = 0.5*(g²_ij + g²_ji) - (1/3)*δ_ij*trace
        sd = ntuple(Val(3)) do i
            ntuple(Val(3)) do j
                sym = T(0.5) * (g2[i][j] + g2[j][i])
                diag_part = (i == j) ? trace_g2 / T(3) : zero(T)
                sym - diag_part
            end
        end

        sd_sq = zero(T)
        s_sq = zero(T)
        for i in 1:3, j in 1:3
            sd_sq += sd[i][j]^2
            S_ij = T(0.5) * (g[i][j] + g[j][i])
            s_sq += S_ij^2
        end

        denom = s_sq^T(2.5) + sd_sq^T(1.25)
        if denom > eps(T)
            nu_t[c] = (model.Cw * model.delta[c])^2 * sd_sq^T(1.5) / denom
        else
            nu_t[c] = zero(T)
        end
    end

    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/turbulence/les_types.jl"); include("src/turbulence/smagorinsky.jl"); include("src/turbulence/wale.jl"); println("OK")'
```

---

### Task 4: Create dynamic_smagorinsky.jl — Dynamic Smagorinsky

**Files:**
- Create: `src/turbulence/dynamic_smagorinsky.jl`

- [ ] **Step 1: Write the dynamic Smagorinsky model**

```julia
# turbulence/dynamic_smagorinsky.jl — Dynamic Smagorinsky SGS model
#
# Computes the Smagorinsky constant Cs dynamically from the Germano
# identity using a test filter (volume-weighted neighbor average).

"""
    DynamicSmagorinsky{T} <: AbstractLESModel

Dynamic Smagorinsky SGS model with Germano identity.

The Smagorinsky constant Cs is computed dynamically each time step
using a test filter (volume-weighted average over face-connected
neighbors). This makes the model self-calibrating.

# Fields
- `delta::Vector{T}` — grid filter width per cell
- `test_filter_ratio::T` — test filter / grid filter ratio (default 2.0)
"""
struct DynamicSmagorinsky{T} <: AbstractLESModel
    delta::Vector{T}
    test_filter_ratio::T
end

"""
    DynamicSmagorinsky(mesh; test_filter_ratio = 2.0)
"""
function DynamicSmagorinsky(
        mesh::UnstructuredFVMMesh{Dim, T};
        test_filter_ratio::Real = 2.0,
    ) where {Dim, T}
    delta = compute_filter_width(mesh)
    return DynamicSmagorinsky{T}(delta, T(test_filter_ratio))
end

"""
    _test_filter(values, mesh) -> Vector

Volume-weighted average of `values` over each cell and its face-connected neighbors.
"""
function _test_filter(
        values::Vector{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    filtered = zeros(T, nc)
    weights = zeros(T, nc)

    # Self-contribution
    for c in 1:nc
        filtered[c] += values[c] * mesh.cell_volumes[c]
        weights[c] += mesh.cell_volumes[c]
    end

    # Neighbor contributions via faces
    for f in 1:nf
        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            filtered[P] += values[N] * mesh.cell_volumes[N]
            weights[P] += mesh.cell_volumes[N]
            filtered[N] += values[P] * mesh.cell_volumes[P]
            weights[N] += mesh.cell_volumes[P]
        end
    end

    for c in 1:nc
        filtered[c] /= max(weights[c], eps(T))
    end

    return filtered
end

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::DynamicSmagorinsky{T},
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    alpha = model.test_filter_ratio

    # Strain rate at grid level
    S_mag = compute_strain_rate(U, mesh)

    # |S| * S_ij approximation: use |S|² as the contraction magnitude
    S_sq = [S_mag[c]^2 for c in 1:nc]

    # Test-filtered quantities
    S_mag_filtered = _test_filter(S_mag, mesh)
    S_sq_filtered = _test_filter(S_sq, mesh)

    # Compute dynamic Cs² per cell via simplified Germano-Lilly
    # M = 2Δ²(α² |S̃|² - |S|²) (simplified scalar version)
    # L = test_filter(|S|²) - test_filter(|S|)²  (scalar Leonard stress proxy)
    for c in 1:nc
        delta_c = model.delta[c]
        M = T(2) * delta_c^2 * (alpha^2 * S_mag_filtered[c]^2 - S_sq[c])

        L = S_sq_filtered[c] - S_mag_filtered[c]^2

        if abs(M) > eps(T)
            Cs_sq = max(L / M, zero(T))  # clip negative for stability
        else
            Cs_sq = T(0.01)  # fallback
        end

        # Cap Cs to prevent excessive values
        Cs_sq = min(Cs_sq, T(0.04))  # Cs < 0.2

        nu_t[c] = Cs_sq * delta_c^2 * S_mag[c]
    end

    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; include("src/turbulence/les_types.jl"); include("src/turbulence/smagorinsky.jl"); include("src/turbulence/wale.jl"); include("src/turbulence/dynamic_smagorinsky.jl"); println("OK")'
```

---

### Task 5: Create ddes.jl — DDES hybrid model

**Files:**
- Create: `src/turbulence/ddes.jl`

- [ ] **Step 1: Write the DDES model**

```julia
# turbulence/ddes.jl — Delayed Detached Eddy Simulation
#
# Hybrid RANS/LES that wraps a base RANS model (Spalart-Allmaras)
# and modifies the length scale to switch from RANS in the boundary
# layer to LES in separated regions.

"""
    DDES{B, T} <: AbstractHybridModel

Delayed Detached Eddy Simulation.

Wraps a base RANS model and modifies its turbulent length scale:
`l_DDES = l_RANS - f_d · max(0, l_RANS - l_LES)`

The shielding function `f_d` protects the boundary layer from
premature LES switching.

# Fields
- `base_model::B` — base RANS model (e.g., SpalartAllmaras)
- `C_DES::T` — DES constant (default 0.65)
- `delta::Vector{T}` — grid filter width per cell
- `d_wall::Vector{T}` — wall distance per cell
"""
struct DDES{B, T} <: AbstractHybridModel
    base_model::B
    C_DES::T
    delta::Vector{T}
    d_wall::Vector{T}
end

"""
    DDES(base_model, mesh, wall_patches; C_DES = 0.65)

Construct a DDES model from a base RANS model.
"""
function DDES(
        base_model,
        mesh::UnstructuredFVMMesh{Dim, T},
        wall_patches::Vector{Symbol};
        C_DES::Real = 0.65,
    ) where {Dim, T}
    delta = compute_filter_width(mesh)
    d_wall = compute_wall_distance(mesh, wall_patches)
    return DDES{typeof(base_model), T}(base_model, T(C_DES), delta, T.(d_wall))
end

# ── DDES interface ───────────────────────────────────────────────────

n_turbulence_fields(model::DDES) = n_turbulence_fields(model.base_model)
turbulence_field_names(model::DDES) = turbulence_field_names(model.base_model)

"""
    _ddes_shielding(nu, nu_t, d, S, kappa) -> f_d

Compute the DDES shielding function. Returns ~0 in boundary layer
(RANS mode) and ~1 in separated regions (LES mode).
"""
function _ddes_shielding(nu::T, nu_t::T, d::T, S::T, kappa::T) where {T}
    d_safe = max(d, T(1e-10))
    S_safe = max(S, T(1e-10))
    r_d = (nu + nu_t) / (kappa^2 * d_safe^2 * S_safe)
    return one(T) - tanh((T(8) * r_d)^3)
end

"""
    _ddes_length_scale(l_RANS, l_LES, f_d) -> l_DDES

Compute the DDES modified length scale.
"""
function _ddes_length_scale(l_RANS::T, l_LES::T, f_d::T) where {T}
    return l_RANS - f_d * max(zero(T), l_RANS - l_LES)
end

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::DDES{B, T},
        turb_state::RANSTurbulenceState{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {B, Dim, T}
    # Delegate to base model for viscosity computation
    turbulent_viscosity!(nu_t, model.base_model, turb_state, mesh)
    return nothing
end

function solve_turbulence!(
        turb_state::RANSTurbulenceState{T},
        model::DDES{B, T},
        U::CollocatedVectorField{Dim, T},
        phi::FaceFluxField{T},
        nu::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_turb::Dict{Symbol, <:Dict{Symbol, <:AbstractBoundaryCondition}};
        dt::Union{Nothing, T} = nothing,
        linear_solver = nothing,
    ) where {B, Dim, T}
    nc = length(mesh.cell_volumes)
    kappa = T(0.41)

    # Compute strain rate for shielding function
    S_mag = compute_strain_rate(U, mesh)

    # Compute DDES length scale per cell
    l_ddes = Vector{T}(undef, nc)
    for c in 1:nc
        l_RANS = model.d_wall[c]  # For SA-based DDES, l_RANS = d
        l_LES = model.C_DES * model.delta[c]
        f_d = _ddes_shielding(nu, turb_state.nu_t[c], model.d_wall[c], S_mag[c], kappa)
        l_ddes[c] = _ddes_length_scale(l_RANS, l_LES, f_d)
    end

    # Solve base RANS model with modified wall distance
    # For SA: temporarily modify the d_wall in the base model
    # Since SpalartAllmaras stores d_wall, we create a temporary copy
    if model.base_model isa SpalartAllmaras
        # Save original d_wall
        d_orig = copy(model.base_model.d_wall)
        # Replace with DDES length scale
        model.base_model.d_wall .= l_ddes
        # Solve
        solve_turbulence!(turb_state, model.base_model, U, phi, nu, mesh, bcs_turb;
            dt = dt, linear_solver = linear_solver)
        # Restore original
        model.base_model.d_wall .= d_orig
    else
        # Generic fallback: just solve the base model
        solve_turbulence!(turb_state, model.base_model, U, phi, nu, mesh, bcs_turb;
            dt = dt, linear_solver = linear_solver)
    end

    return nothing
end
```

- [ ] **Step 2: Verify syntax**

```bash
julia --project -e 'using FiniteVolumeMethod; for f in ["les_types", "smagorinsky", "wale", "dynamic_smagorinsky", "ddes"]; include("src/turbulence/$f.jl"); end; println("OK")'
```

---

### Task 6: Update solvers.jl + wire into module + exports

**Files:**
- Modify: `src/turbulence/solvers.jl` — replace 2-step turbulence calls with `_update_turbulence!`
- Modify: `src/layers/discretization_assembly_kernels.jl` — add LES includes
- Modify: `src/FiniteVolumeMethod.jl` — add exports

- [ ] **Step 1: Update solvers.jl**

In `src/turbulence/solvers.jl`, find all occurrences of the 2-step pattern:
```julia
        solve_turbulence!(
            turb_state, turb_model, state.U, state.phi, prob.nu, mesh, turb_bcs;
            linear_solver = linear_solver,
        )
        turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
```

Replace each occurrence with:
```julia
        _update_turbulence!(
            turb_state, turb_model, state, prob, mesh, turb_bcs;
            linear_solver = linear_solver,
        )
```

There should be multiple occurrences (in `solve_simple_turbulent`, `solve_incompressible_turbulent`, and the transient step functions). Also add `dt = dt_actual` to the transient calls.

- [ ] **Step 2: Add includes to Layer 2**

In `src/layers/discretization_assembly_kernels.jl`, add AFTER `include("../turbulence/solvers.jl")`:

```julia
# LES & Hybrid Turbulence Models (Phase 2b)
include("../turbulence/les_types.jl")
include("../turbulence/smagorinsky.jl")
include("../turbulence/wale.jl")
include("../turbulence/dynamic_smagorinsky.jl")
include("../turbulence/ddes.jl")
```

- [ ] **Step 3: Add exports**

In `src/FiniteVolumeMethod.jl`, add after the Phase 2a RANS exports:

```julia
# --- LES & Hybrid Turbulence Models (Phase 2b) ---
export
    AbstractLESModel,
    AbstractHybridModel,
    LESTurbulenceState,
    Smagorinsky,
    WALE,
    DynamicSmagorinsky,
    DDES,
    compute_filter_width
```

- [ ] **Step 4: Verify module loads**

```bash
julia --project -e 'using FiniteVolumeMethod; println("Phase 2b: ", Smagorinsky)'
```

- [ ] **Step 5: Run RANS regression**

```bash
julia --project=test test/turbulence_rans.jl
```
Expected: 127 tests pass (backward-compatible).

- [ ] **Step 6: Commit**

```bash
git add src/turbulence/ src/layers/discretization_assembly_kernels.jl src/FiniteVolumeMethod.jl
git commit -m "feat: add LES (Smagorinsky, WALE, dynamic) and DDES hybrid turbulence models (Phase 2b)"
```

---

### Task 7: Write tests

**Files:**
- Create: `test/turbulence_les.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the test file**

Create `test/turbulence_les.jl`. Copy `build_cartesian_unstructured_mesh` from `test/incompressible.jl`. Tests:

1. **compute_filter_width** — 4x4 mesh (dx=dy=0.25), filter width should be sqrt(0.0625) ≈ 0.25 for 2D
2. **LESTurbulenceState construction** — verify nu_t is zero-initialized, correct length
3. **Smagorinsky construction** — verify Cs, delta length matches ncells
4. **Smagorinsky viscosity on shear flow** — U = (y, 0) linear shear: ν_sgs should be > 0 and proportional to (Cs·Δ)²
5. **Smagorinsky viscosity on uniform flow** — U = (1, 0): ν_sgs should be ≈ 0 (no strain)
6. **WALE construction** — verify Cw, delta
7. **WALE viscosity on shear flow** — should produce ν_sgs ≥ 0
8. **WALE viscosity near wall** — WALE should produce small ν_sgs for pure shear (S_d → 0)
9. **DynamicSmagorinsky construction** — verify delta, test_filter_ratio
10. **DynamicSmagorinsky viscosity** — should produce finite ν_sgs ≥ 0
11. **_test_filter** — constant field should return same constant
12. **DDES construction** — wraps SpalartAllmaras, verify C_DES, delta, d_wall
13. **n_turbulence_fields** — 0 for LES models, 1 for DDES (from SA base)
14. **solve_turbulence! no-op for LES** — call doesn't error, returns nothing
15. **_update_turbulence! dispatcher** — LES path calls turbulent_viscosity!, verify nu_t is updated

- [ ] **Step 2: Register test**

Add `safe_include("turbulence_les.jl")` to `test/runtests.jl` after the linear_solvers test.

- [ ] **Step 3: Run tests**

```bash
julia --project=test test/turbulence_les.jl
```

- [ ] **Step 4: Run Runic**

```bash
julia --project -e 'using Runic; Runic.main(["--inplace", "src/turbulence/les_types.jl", "src/turbulence/smagorinsky.jl", "src/turbulence/wale.jl", "src/turbulence/dynamic_smagorinsky.jl", "src/turbulence/ddes.jl"])'
julia --project -e 'using Runic; Runic.main(["--inplace", "test/turbulence_les.jl"])'
```

- [ ] **Step 5: Commit**

```bash
git add test/turbulence_les.jl test/runtests.jl
git commit -m "test: add LES and hybrid turbulence model tests"
```

---

### Task 8: Register in validation manifest + final verification

**Files:**
- Modify: `validation/manifest.toml`

- [ ] **Step 1: Add turbulence_les feature**

Append to `validation/manifest.toml`:

```toml
# ── Phase 2b: LES & Hybrid Turbulence Models ──────────────────────

[[features]]
feature = "turbulence_les"
maturity = "experimental"
validation = "smoke_tested"
role = "research_tooling"
solver_family = "collocated"
precision_policy = "float64_cpu_reference"
random_seed_policy = "deterministic"
backend_policy = "cpu_reference"
required_ladder_stages = ["verification", "benchmark"]
summary = "LES (Smagorinsky, WALE, dynamic Smagorinsky) and hybrid (DDES) turbulence models for incompressible collocated solver."
limitations = [
  "Experimental — validated via smoke tests only; decaying turbulence benchmark pending.",
  "Dynamic Smagorinsky uses simplified scalar Germano identity, not full tensor form.",
  "DDES only wraps Spalart-Allmaras; SST-based DDES deferred.",
]
```

- [ ] **Step 2: Verify all exports**

```bash
julia --project -e '
using FiniteVolumeMethod
for sym in [:AbstractLESModel, :AbstractHybridModel, :LESTurbulenceState,
            :Smagorinsky, :WALE, :DynamicSmagorinsky, :DDES, :compute_filter_width]
    @assert isdefined(FiniteVolumeMethod, sym) "Missing: $sym"
end
println("All Phase 2b exports verified")
'
```

- [ ] **Step 3: Run all turbulence tests**

```bash
julia --project=test test/turbulence_rans.jl && julia --project=test test/turbulence_les.jl
```

- [ ] **Step 4: Runic check**

```bash
julia --project -e 'using Runic; Runic.main(["--check", "src/turbulence/les_types.jl", "src/turbulence/smagorinsky.jl", "src/turbulence/wale.jl", "src/turbulence/dynamic_smagorinsky.jl", "src/turbulence/ddes.jl"])'
```

- [ ] **Step 5: Commit**

```bash
git add validation/manifest.toml
git commit -m "feat: register turbulence_les in validation manifest"
```
