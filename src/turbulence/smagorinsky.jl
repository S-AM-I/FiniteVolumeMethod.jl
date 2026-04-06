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
