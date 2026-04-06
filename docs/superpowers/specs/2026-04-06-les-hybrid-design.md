# Phase 2b: LES & Hybrid Turbulence Models

**Date**: 2026-04-06
**Status**: Design
**Depends on**: Phase 2a (RANS Turbulence — provides AbstractTurbulenceModel, turbulent_viscosity!, solver wrappers)

## Goal

Add subgrid-scale (SGS) LES models (Smagorinsky, WALE, dynamic Smagorinsky) and a hybrid RANS/LES model (DDES) to the incompressible solver. These are algebraic models that compute turbulent viscosity directly from the resolved velocity field without solving additional transport equations.

## Architecture

### Key Difference from RANS

LES models are algebraic: `ν_sgs = f(∇U, Δ)` where `Δ` is the grid filter width. No transport equations to solve, no turbulence fields to store. The `solve_turbulence!` call is a no-op for LES — `turbulent_viscosity!` does all the work.

DDES is a hybrid that wraps a base RANS model and modifies its length scale based on grid resolution, switching from RANS in the boundary layer to LES in separated regions.

### File Layout

New files in `src/turbulence/`:

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `les_types.jl` | AbstractLESModel, AbstractHybridModel, LESTurbulenceState, compute_filter_width | ~70 |
| `smagorinsky.jl` | Smagorinsky SGS: ν_sgs = (Cs·Δ)²·|S| | ~60 |
| `wale.jl` | WALE SGS: better near-wall behavior | ~100 |
| `dynamic_smagorinsky.jl` | Dynamic Smagorinsky with test filtering (Germano identity) | ~120 |
| `ddes.jl` | DDES hybrid wrapping a base RANS model | ~100 |

Wired into Layer 2 after existing turbulence files.

## Type Hierarchy

```julia
abstract type AbstractLESModel <: AbstractTurbulenceModel end
abstract type AbstractHybridModel <: AbstractTurbulenceModel end
```

Both sit alongside `AbstractRANSModel` under `AbstractTurbulenceModel`.

## LES State

```julia
mutable struct LESTurbulenceState{T}
    nu_t::Vector{T}
end

LESTurbulenceState(mesh::UnstructuredFVMMesh{Dim, T}) where {Dim, T} =
    LESTurbulenceState{T}(zeros(T, length(mesh.cell_volumes)))
```

The existing turbulent solver wrappers access `turb_state.nu_t` — `LESTurbulenceState` has this field so it's compatible via duck typing.

## Filter Width

All LES models need the grid filter width per cell:

```julia
function compute_filter_width(mesh::UnstructuredFVMMesh{Dim, T}) -> Vector{T}
```

Formula: `Δ[c] = V_c^(1/Dim)` — cube root of volume in 3D, square root of area in 2D. Computed once at model construction time and stored in the model struct.

## Interface for LES Models

LES models implement a simplified interface:

```julia
# Required: compute nu_t algebraically from velocity
turbulent_viscosity!(nu_t, model::AbstractLESModel, U, mesh)

# No-ops for LES (no transport equations):
n_turbulence_fields(::AbstractLESModel) = 0
turbulence_field_names(::AbstractLESModel) = ()
solve_turbulence!(turb_state, model::AbstractLESModel, U, phi, nu, mesh, bcs; kwargs...) = nothing
```

Note: the `turbulent_viscosity!` signature for LES takes `U::CollocatedVectorField` directly (not `turb_state`) since there are no turbulence fields to read. To maintain compatibility with the existing `turbulent_viscosity!(nu_t, model, turb_state, mesh)` signature from RANS, we provide both:

```julia
# LES-specific (takes velocity field)
turbulent_viscosity_les!(nu_t, model::AbstractLESModel, U, mesh)

# Generic interface (calls LES-specific internally, U passed via state)
# The solver wrapper passes U separately when calling LES
```

Actually, the cleanest approach: the solver wrappers already have access to `state.U`. We add a method:

```julia
function turbulent_viscosity!(nu_t, model::AbstractLESModel, turb_state, mesh, U)
    # LES models use U directly
end
```

But changing the signature is messy. **Simplest**: store a reference to the velocity field in `LESTurbulenceState`:

```julia
mutable struct LESTurbulenceState{Dim, T}
    nu_t::Vector{T}
    U_ref::CollocatedVectorField{Dim, T}  # reference to current velocity
end
```

Then `turbulent_viscosity!(nu_t, model, turb_state, mesh)` reads `turb_state.U_ref` for the velocity. The solver wrapper updates `turb_state.U_ref = state.U` before calling `turbulent_viscosity!`.

**Even simpler**: just have `turbulent_viscosity!` for LES models accept the state and ignore the fields dict, using the velocity gradients computed from a velocity field passed through the state. But `LESTurbulenceState` doesn't have `fields`...

**Final decision**: Add an overloaded `turbulent_viscosity!` that takes extra `U` argument:

```julia
# For LES: requires velocity field
function turbulent_viscosity!(nu_t, model::AbstractLESModel, U, mesh)
    # model-specific computation
end

# Solver wrapper calls this for LES models
```

The solver wrappers already distinguish between RANS and LES by type — they can call the right signature. This is clean and doesn't require storing velocity references.

## Models

### Smagorinsky

```julia
struct Smagorinsky{T} <: AbstractLESModel
    Cs::T              # Smagorinsky constant (default 0.1)
    delta::Vector{T}   # filter width per cell
end

Smagorinsky(mesh; Cs = 0.1)
```

Formula: `ν_sgs[c] = (Cs · Δ[c])² · |S[c]|`

where `|S|` is strain rate magnitude from `compute_strain_rate`.

### WALE

```julia
struct WALE{T} <: AbstractLESModel
    Cw::T              # WALE constant (default 0.325)
    delta::Vector{T}   # filter width per cell
end

WALE(mesh; Cw = 0.325)
```

Formula:
```
S_d_ij = 0.5*(g²_ij + g²_ji) - (1/3)*δ_ij*g²_kk
where g_ij = ∂u_i/∂x_j, g²_ij = g_ik * g_kj

ν_sgs = (Cw·Δ)² · (S_d:S_d)^(3/2) / ((S:S)^(5/2) + (S_d:S_d)^(5/4))
```

The WALE model naturally produces ν_sgs → 0 at walls without explicit damping, making it superior to Smagorinsky near walls.

### Dynamic Smagorinsky

```julia
struct DynamicSmagorinsky{T} <: AbstractLESModel
    delta::Vector{T}           # filter width per cell
    test_filter_ratio::T       # test filter / grid filter ratio (default 2.0)
end

DynamicSmagorinsky(mesh; test_filter_ratio = 2.0)
```

Computes Cs dynamically using the Germano identity:
1. Compute strain rate `|S|` at grid level
2. Apply test filter (volume-weighted average over neighbor cells) to get `|S̄|`
3. Compute Leonard stress `L_ij = test_filter(u_i·u_j) - test_filter(u_i)·test_filter(u_j)`
4. `Cs² = <L_ij M_ij> / <M_ij M_ij>` where `M_ij = 2Δ²(|S̄|S̄_ij - α²|S|S_ij)`
5. Clip Cs² ≥ 0 for stability

The test filter uses cell-face connectivity: for each cell, average over the cell and its face-connected neighbors weighted by volume.

### DDES (Delayed Detached Eddy Simulation)

```julia
struct DDES{B, T} <: AbstractHybridModel
    base_model::B         # base RANS model (e.g., SpalartAllmaras or KOmegaSSTModel)
    C_DES::T              # DES constant (default 0.65)
    delta::Vector{T}      # filter width per cell
    d_wall::Vector{T}     # wall distance per cell
end

DDES(base_model, mesh, wall_patches; C_DES = 0.65)
```

DDES modifies the RANS model's length scale:
```
l_RANS = model-specific (e.g., d for SA, sqrt(k)/ω for SST)
l_LES = C_DES · Δ
l_DDES = l_RANS - f_d · max(0, l_RANS - l_LES)
```

The shielding function `f_d` protects the boundary layer from premature LES switching:
```
f_d = 1 - tanh((8·r_d)³)
r_d = (ν + ν_t) / (κ²·d²·max(S, 1e-10))
```

For the first implementation, DDES wraps Spalart-Allmaras and modifies its destruction term length scale from `d` to `l_DDES`. The `solve_turbulence!` delegates to the base model with the modified length scale.

## Solver Integration

The existing `solve_simple_turbulent` and `solve_incompressible_turbulent` need minor adaptation for LES:

```julia
# In the solver loop, after velocity correction:
if turb_model isa AbstractLESModel
    turbulent_viscosity!(turb_state.nu_t, turb_model, state.U, mesh)
elseif turb_model isa AbstractHybridModel
    solve_turbulence!(turb_state, turb_model, state.U, state.phi, prob.nu, mesh, turb_bcs; ...)
    turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
else  # RANS
    solve_turbulence!(turb_state, turb_model, state.U, state.phi, prob.nu, mesh, turb_bcs; ...)
    turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
end
```

The simplest integration: add an `_update_turbulence!` dispatcher function:

```julia
function _update_turbulence!(turb_state, turb_model::AbstractLESModel, state, prob, mesh, turb_bcs; kwargs...)
    turbulent_viscosity!(turb_state.nu_t, turb_model, state.U, mesh)
end

function _update_turbulence!(turb_state, turb_model, state, prob, mesh, turb_bcs; kwargs...)
    # RANS / hybrid path
    solve_turbulence!(turb_state, turb_model, state.U, state.phi, prob.nu, mesh, turb_bcs; kwargs...)
    turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
end
```

Then the solver wrappers call `_update_turbulence!` instead of the two-step sequence. This is backward-compatible.

## Export List

```julia
# Types
export AbstractLESModel, AbstractHybridModel, LESTurbulenceState

# Models
export Smagorinsky, WALE, DynamicSmagorinsky, DDES

# Utilities
export compute_filter_width
```

## Validation

- **Smagorinsky on uniform shear**: ν_sgs should be proportional to |S| and (Cs·Δ)²
- **WALE near wall**: ν_sgs should vanish at a no-slip wall (S_d → 0 in pure shear)
- **Dynamic Smagorinsky**: Cs should be ~0.1 for isotropic turbulence
- **DDES shielding**: f_d should be ~0 in boundary layer (RANS mode) and ~1 in separated regions (LES mode)
