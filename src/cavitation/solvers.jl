# cavitation/solvers.jl — Dispatch on cavitation model type to produce
# the per-cell vapour mass source array [kg/(m³·s)] that the v3.0 VOF
# α-transport equation consumes as an explicit source term.
#
# All three models share a single entry point:
#
#   compute_vapor_source(model, p, alpha_v, mesh, props) -> Vector{T}
#
# Sign convention of the returned vector: positive → vapour produced,
# negative → vapour destroyed. Callers wire it into the α-transport RHS
# (typically divided by ρ_v for a volumetric α source).

"""
    compute_vapor_source(model, p, alpha_v, mesh, props) -> Vector{T}

Return the per-cell vapour mass source [kg/(m³·s)] for the active
cavitation `model` given pressure `p`, vapour-fraction `alpha_v`,
mesh `mesh`, and two-phase `props::CavitationProperties`. The returned
vector has length `ncells(mesh)` and obeys the sign convention
    positive ⇒ vapour produced (evaporation)
    negative ⇒ vapour destroyed (condensation).

# Arguments
- `model` — `KunzModel`, `SchnerrSauerModel`, or `MerkleModel`.
- `p::AbstractVector` — cell-centred pressure field.
- `alpha_v::AbstractVector` — cell-centred vapour volume fraction.
- `mesh::UnstructuredFVMMesh` — only used for `length(cell_volumes)`.
- `props::CavitationProperties` — `(rho_l, rho_v, p_sat)`.

# Notes
- Inputs `p` and `alpha_v` must have length equal to the cell count.
- The kernel is allocation-light: a single `Vector{T}` is produced.
"""
function compute_vapor_source(
        model::AbstractCavitationVaporModel{T},
        p::AbstractVector,
        alpha_v::AbstractVector,
        mesh,
        props::CavitationProperties{T},
    ) where {T}
    nc = length(mesh.cell_volumes)
    length(p) == nc || throw(
        DimensionMismatch(
            "pressure vector length $(length(p)) != mesh cell count $nc",
        )
    )
    length(alpha_v) == nc || throw(
        DimensionMismatch(
            "alpha_v vector length $(length(alpha_v)) != mesh cell count $nc",
        )
    )
    source = Vector{T}(undef, nc)
    @inbounds for c in 1:nc
        source[c] = _vapor_source_cell(model, T(p[c]), T(alpha_v[c]), props)
    end
    return source
end

# --- Per-cell kernel dispatch --------------------------------------------
function _vapor_source_cell(
        m::KunzModel{T}, p::T, alpha_v::T, props::CavitationProperties{T},
    ) where {T}
    return kunz_rate(m, p, alpha_v, props.rho_l, props.rho_v, props.p_sat)
end

function _vapor_source_cell(
        m::SchnerrSauerModel{T}, p::T, alpha_v::T, props::CavitationProperties{T},
    ) where {T}
    return schnerr_sauer_rate(m, p, alpha_v, props.rho_l, props.rho_v, props.p_sat)
end

function _vapor_source_cell(
        m::MerkleModel{T}, p::T, alpha_v::T, props::CavitationProperties{T},
    ) where {T}
    return merkle_rate(m, p, alpha_v, props.rho_l, props.rho_v, props.p_sat)
end
