# porous/darcy_forchheimer.jl — Full-tensor Darcy-Forchheimer porous
# momentum source.
#
# Momentum sink (OpenFOAM convention):
#
#     S_U = − ( μ · K⁻¹  +  0.5 · ρ · F · |U| ) · U
#
# where `K` is the permeability tensor [m²] and `F` is the Forchheimer
# inertial resistance tensor [1/m]. Both are stored as 3×3
# `SMatrix{3,3,T}` to keep the solver dimension-agnostic; 2D velocities
# are promoted to 3D internally.
#
# Extensions to anisotropic / rotated porous media come for free: the
# full tensor `K⁻¹` is pre-computed once per zone and re-used per cell.

using LinearAlgebra: inv

"""
    PorousZone{T}

Darcy-Forchheimer porous zone with full 3×3 permeability and
Forchheimer tensors.

# Fields
- `cell_indices::Vector{Int}` — global cell indices in this zone.
- `K::SMatrix{3,3,T,9}` — permeability tensor [m²].
- `F::SMatrix{3,3,T,9}` — Forchheimer coefficient tensor [1/m].
- `K_inv::SMatrix{3,3,T,9}` — pre-computed `inv(K)` cached for speed.

Construct via `PorousZone(cell_indices; K, F)`:
- `K` and `F` accept scalar, `SVector{3}` (diagonal), or `SMatrix{3,3}`
  forms and are promoted internally.
"""
struct PorousZone{T}
    cell_indices::Vector{Int}
    K::SMatrix{3, 3, T, 9}
    F::SMatrix{3, 3, T, 9}
    K_inv::SMatrix{3, 3, T, 9}
end

"""
    _to_3x3(x, T) -> SMatrix{3,3,T,9}

Promote a scalar, length-3 vector, or 3×3 matrix to an `SMatrix{3,3,T,9}`.
- Scalar ⇒ scalar · I.
- 3-vector ⇒ diagonal tensor.
- 3×3 matrix ⇒ element-wise copy.
"""
function _to_3x3(x::Real, ::Type{T}) where {T}
    v = T(x)
    return SMatrix{3, 3, T, 9}(v, 0, 0, 0, v, 0, 0, 0, v)
end
function _to_3x3(x::AbstractVector, ::Type{T}) where {T}
    length(x) == 3 || throw(DimensionMismatch("expected length-3 vector"))
    return SMatrix{3, 3, T, 9}(
        T(x[1]), 0, 0,
        0, T(x[2]), 0,
        0, 0, T(x[3]),
    )
end
function _to_3x3(x::AbstractMatrix, ::Type{T}) where {T}
    size(x) == (3, 3) || throw(DimensionMismatch("expected 3×3 matrix"))
    return SMatrix{3, 3, T, 9}(
        T(x[1, 1]), T(x[2, 1]), T(x[3, 1]),
        T(x[1, 2]), T(x[2, 2]), T(x[3, 2]),
        T(x[1, 3]), T(x[2, 3]), T(x[3, 3]),
    )
end

"""
    PorousZone(cell_indices; K, F)

Construct a `PorousZone{T}` with promoted-type scalars `T`, where
`T` is derived from `K` and `F`.
"""
function PorousZone(
        cell_indices::AbstractVector{<:Integer};
        K = 1.0, F = 0.0,
    )
    T_K = _promoted_eltype(K)
    T_F = _promoted_eltype(F)
    T = promote_type(typeof(float(one(T_K))), typeof(float(one(T_F))))
    K3 = _to_3x3(K, T)
    F3 = _to_3x3(F, T)
    K_inv = inv(K3)
    return PorousZone{T}(collect(Int, cell_indices), K3, F3, K_inv)
end

_promoted_eltype(x::Real) = typeof(x)
_promoted_eltype(x::AbstractArray) = eltype(x)

"""
    is_in_zone(zone, c)

Return `true` if cell index `c` is a member of the porous `zone`.
"""
is_in_zone(zone::PorousZone, c::Integer) = c in zone.cell_indices

"""
    darcy_forchheimer_source(zone, U_cell, rho, mu) -> SVector

Compute the per-cell momentum source vector [N/m³] for a `PorousZone`:

    S_U = − ( μ · K⁻¹ + 0.5 · ρ · F · |U| ) · U

`U_cell` may be an `SVector{2,T}` or `SVector{3,T}`; 2D inputs are
lifted to 3D, the source is computed in 3D, then the relevant leading
components are returned.
"""
function darcy_forchheimer_source(
        zone::PorousZone{T}, U_cell::SVector{3, T}, rho::T, mu::T,
    ) where {T}
    u_mag = sqrt(U_cell[1]^2 + U_cell[2]^2 + U_cell[3]^2)
    resistance = mu * zone.K_inv + T(0.5) * rho * zone.F * u_mag
    return -(resistance * U_cell)
end

function darcy_forchheimer_source(
        zone::PorousZone{T}, U_cell::SVector{2, T}, rho::T, mu::T,
    ) where {T}
    U3 = SVector{3, T}(U_cell[1], U_cell[2], zero(T))
    S3 = darcy_forchheimer_source(zone, U3, rho, mu)
    return SVector{2, T}(S3[1], S3[2])
end

"""
    add_darcy_forchheimer_source!(source_U, U, porous_zone, rho, mu)

Accumulate the Darcy-Forchheimer momentum source into `source_U[c]`
for every `c` in `porous_zone.cell_indices`. Cells outside the zone
are untouched.

# Arguments
- `source_U::AbstractVector{SVector{Dim,T}}` — per-cell momentum source
  vector that is mutated in place.
- `U::AbstractVector{SVector{Dim,T}}` — per-cell velocity.
- `porous_zone::PorousZone{T}` — the zone.
- `rho::T`, `mu::T` — fluid density and dynamic viscosity.

A typical caller is the collocated momentum assembler, which adds this
sink to the explicit RHS before the velocity predictor is solved.
"""
function add_darcy_forchheimer_source!(
        source_U::AbstractVector{SVector{Dim, T}},
        U::AbstractVector{SVector{Dim, T}},
        porous_zone::PorousZone{T},
        rho::T, mu::T,
    ) where {Dim, T}
    length(source_U) == length(U) || throw(
        DimensionMismatch("source_U and U must have matching length")
    )
    @inbounds for c in porous_zone.cell_indices
        (1 <= c <= length(U)) || throw(
            BoundsError("porous zone cell index $c out of 1..$(length(U))")
        )
        source_U[c] = source_U[c] +
            darcy_forchheimer_source(porous_zone, U[c], rho, mu)
    end
    return source_U
end
