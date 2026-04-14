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

    # Apply equilibrium wall functions to wall-adjacent cells
    wall_patches = _detect_wall_patches(prob.bcs)
    if !isempty(wall_patches)
        apply_wall_functions!(turb_state, turb_model, state.U, mesh, prob.nu, wall_patches)
    end

    return nothing
end

"""
    _detect_wall_patches(bcs) -> Vector{Symbol}

Identify wall boundary patches from the incompressible BCs dict.
Returns patch names whose BC is `NoSlipWallBC` or `WallFunctionBC`.
"""
function _detect_wall_patches(bcs)
    patches = Symbol[]
    for (name, bc) in bcs
        if bc isa NoSlipWallBC || bc isa WallFunctionBC
            push!(patches, name)
        end
    end
    return patches
end

# ── State initialization override ────────────────────────────────────

"""LES models use LESTurbulenceState (no transport fields)."""
_init_turb_state(model::AbstractLESModel, mesh) = LESTurbulenceState(mesh)
