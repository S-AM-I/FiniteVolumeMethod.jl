# aeroacoustics/pml.jl — Perfectly Matched Layer sponge zone (Wave 3 Agent D)
#
# PML absorbs outgoing acoustic / convective waves at open boundaries by
# adding a stretched-coordinate damping source to the governing
# equations. In the sponge-zone formulation we keep track of a
# coordinate-dependent damping coefficient σ(x) that grows polynomially
# from 0 at the inner edge of the layer to σ_max at the outer edge, and
# pull the conservative state φ toward a prescribed far-field reference
# φ_∞ via the source term
#
#     S_PML(x) = −σ(x) · ( φ(x) − φ_∞ ).
#
# The coefficient profile is standard (Berenger 1994; Hu 1996): choose
# a polynomial order m ∈ {2 (quadratic), 4 (quartic)}; then for a
# point at local distance `s ∈ [0, L]` inside the layer (s=0 at the
# inner CFD-facing boundary, s=L at the outer boundary),
#
#     σ(s) = σ_max · (s / L)^m.
#
# This gives σ(0) = 0 (no reflection at the CFD/PML interface) and
# σ(L) = σ_max (maximum absorption at the far boundary).
#
# `PMLZone` carries the two boundary positions that define the sponge
# layer slab in a chosen axis; `pml_sigma(pml, x)` returns the damping
# coefficient at an arbitrary point (0 outside the zone); and
# `add_pml_source!` accumulates the source term onto a work array.

using StaticArrays: SVector
using LinearAlgebra: norm

"""
    PMLProfile

Polynomial order for the PML σ profile. Currently `Quadratic` (m = 2)
and `Quartic` (m = 4) are supported — these are the two profiles
documented by Berenger / Hu and used as defaults in production
aero-acoustic codes.
"""
@enum PMLProfile Quadratic = 2 Quartic = 4

"""
    PMLZone{Dim, T}

Sponge-zone description. The layer is an axis-aligned slab between
`inner_boundary` (the CFD side, σ = 0) and `outer_boundary`
(the far side, σ = σ_max). The absorbing direction is inferred from
the component of `outer_boundary − inner_boundary` with the largest
absolute value.

Fields:
- `inner_boundary` — SVector marking the CFD-facing face of the layer
- `outer_boundary` — SVector marking the outer face
- `sigma_max`     — peak absorption coefficient (1/s)
- `profile`       — `Quadratic` (default) or `Quartic`
"""
struct PMLZone{Dim, T}
    inner_boundary::SVector{Dim, T}
    outer_boundary::SVector{Dim, T}
    sigma_max::T
    profile::PMLProfile
    axis::Int              # 1 = x, 2 = y, 3 = z
    sign::T                # +1 if outer > inner along axis, else -1

    function PMLZone{Dim, T}(
            inner::SVector{Dim, T},
            outer::SVector{Dim, T},
            sigma_max::T,
            profile::PMLProfile,
        ) where {Dim, T}
        sigma_max >= zero(T) ||
            error("PMLZone: sigma_max must be non-negative, got $sigma_max")
        delta = outer - inner
        axis = 1
        best = abs(delta[1])
        @inbounds for k in 2:Dim
            if abs(delta[k]) > best
                best = abs(delta[k])
                axis = k
            end
        end
        best > T(1.0e-12) ||
            error("PMLZone: inner and outer boundaries coincide")
        sign = delta[axis] > zero(T) ? T(1) : T(-1)
        return new{Dim, T}(inner, outer, sigma_max, profile, axis, sign)
    end
end
PMLZone(
    inner::SVector{Dim, T},
    outer::SVector{Dim, T},
    sigma_max::Real;
    profile::PMLProfile = Quadratic,
) where {Dim, T} = PMLZone{Dim, T}(inner, outer, T(sigma_max), profile)

"""
    pml_layer_thickness(pml) -> T

Absorbing-slab thickness along the active axis.
"""
pml_layer_thickness(pml::PMLZone{Dim, T}) where {Dim, T} =
    abs(pml.outer_boundary[pml.axis] - pml.inner_boundary[pml.axis])

"""
    pml_sigma(pml, x) -> T

Damping coefficient σ(x) at spatial point `x`. Returns 0 outside the
PML slab, σ_max · (s/L)^m inside, where s is the normalised distance
from the inner boundary along the absorbing axis.
"""
function pml_sigma(pml::PMLZone{Dim, T}, x::SVector{Dim, T}) where {Dim, T}
    L = pml_layer_thickness(pml)
    L > T(1.0e-12) || return zero(T)
    s = pml.sign * (x[pml.axis] - pml.inner_boundary[pml.axis])
    # Outside: s<0 means still in CFD domain; s>L means past the far wall.
    (s < zero(T) || s > L) && return zero(T)
    frac = s / L
    m = Int(pml.profile)
    return pml.sigma_max * frac^m
end

"""
    add_pml_source!(source, phi, phi_far, pml, points) -> source

Accumulate the PML damping source term

    S[i] += −σ(x_i) · ( φ[i] − φ_∞ )

onto `source` for every sample point `points[i]`. `phi` may be a
vector of scalars or a vector of SVectors; `phi_far` must be the same
element type as `phi[1]`. Returns `source` for chaining.

This is the generic kernel; solver-facing wrappers supply the
cell-centre array (from `cell_center(mesh, c)` for each owned cell) as
`points`.
"""
function add_pml_source!(
        source::AbstractVector,
        phi::AbstractVector,
        phi_far,
        pml::PMLZone{Dim, T},
        points::AbstractVector{SVector{Dim, T}},
    ) where {Dim, T}
    n = length(points)
    length(source) == n ||
        error("add_pml_source!: source length $(length(source)) ≠ points length $n")
    length(phi) == n ||
        error("add_pml_source!: phi length $(length(phi)) ≠ points length $n")

    @inbounds for i in 1:n
        sigma = pml_sigma(pml, points[i])
        sigma == zero(T) && continue
        source[i] = source[i] - sigma * (phi[i] - phi_far)
    end
    return source
end
