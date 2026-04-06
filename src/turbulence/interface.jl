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
        model,
        mesh::UnstructuredFVMMesh{Dim, T};
        kwargs...,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    names = turbulence_field_names(model)
    fields = Dict{Symbol, CollocatedScalarField{T}}()
    for name in names
        init_val = get(kwargs, name, T(1.0e-6))
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
