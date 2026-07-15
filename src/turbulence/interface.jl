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
- `eq_cache::Dict{Symbol, CollocatedEquation{T}}` — reusable assembled
  equations per transport field (lazily created; `reset!` + reassemble
  each solve instead of rebuilding the sparsity pattern every iteration)
"""
mutable struct RANSTurbulenceState{T}
    fields::Dict{Symbol, CollocatedScalarField{T}}
    nu_t::Vector{T}
    eq_cache::Dict{Symbol, CollocatedEquation{T}}
end

"""Backward-compatible 2-argument constructor (empty equation cache)."""
function RANSTurbulenceState{T}(
        fields::Dict{Symbol, CollocatedScalarField{T}},
        nu_t::Vector{T},
    ) where {T}
    return RANSTurbulenceState{T}(
        fields, nu_t, Dict{Symbol, CollocatedEquation{T}}(),
    )
end

"""
    _cached_equation!(turb_state, name, mesh) -> CollocatedEquation

Fetch (or lazily create) the reusable equation for turbulence field
`name`, `reset!` it, and return it ready for assembly.
"""
function _cached_equation!(
        turb_state::RANSTurbulenceState{T}, name::Symbol,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    eq = get!(() -> CollocatedEquation(mesh), turb_state.eq_cache, name)
    reset!(eq)
    return eq
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

# ── Turbulence BC validation ─────────────────────────────────────────

"""
    _validate_turbulence_bcs(bcs_turb, mesh, model)

Validate up front that boundary conditions exist for every turbulence
transport field of `model` on every boundary patch of `mesh`.

Without this check, a missing entry defaulted to an empty `Dict` and the
solver errored mid-solve at the first boundary face with an unhelpful
"No boundary condition for patch" message.  This raises a clear error
listing the missing fields/patches before any assembly starts.
"""
function _validate_turbulence_bcs(
        bcs_turb, mesh::UnstructuredFVMMesh, model,
    )
    required = turbulence_field_names(model)
    isempty(required) && return nothing

    # Collect boundary patch tags present in the mesh
    tags = Set{Symbol}()
    nf = size(mesh.face_cells, 2)
    for f in 1:nf
        is_internal_face(mesh, f) && continue
        push!(tags, _face_tag(mesh, f))
    end

    problems = String[]
    for fname in required
        if !haskey(bcs_turb, fname)
            push!(
                problems,
                "no boundary conditions given for turbulence field :$fname " *
                    "(required patches: $(join(sort!(collect(tags)), ", ")))",
            )
            continue
        end
        bcs_f = bcs_turb[fname]
        missing_patches = sort!([tag for tag in tags if !haskey(bcs_f, tag)])
        if !isempty(missing_patches)
            push!(
                problems,
                "turbulence field :$fname is missing boundary conditions " *
                    "for patches: $(join(missing_patches, ", "))",
            )
        end
    end

    if !isempty(problems)
        error(
            "Incomplete turbulence boundary conditions for " *
                "$(typeof(model)):\n  - " * join(problems, "\n  - "),
        )
    end
    return nothing
end

# ── Realizability hook ───────────────────────────────────────────────

"""
    _apply_realizability!(turb_state, model, U, mesh)

Re-apply any realizability constraint on `turb_state.nu_t` after the
eddy viscosity has been (re)computed from the transport fields.  Default
is a no-op; models with a realizability cap (e.g. the Durbin cap in
`StandardKEpsilon`) specialize this so the capped `nu_t` is what the
momentum equation actually sees.
"""
_apply_realizability!(turb_state, model, U, mesh) = nothing
