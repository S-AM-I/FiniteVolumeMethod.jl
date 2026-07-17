# mrf/momentum_source.jl — Coriolis + centrifugal momentum source (Wave 3)
#
# In a rotating reference frame the momentum equation picks up two
# pseudo-force densities:
#
#   S_Coriolis    = -2 ρ (ω × u)
#   S_centrifugal = -ρ (ω × (ω × r))
#
# where ω is the (constant) angular velocity of the frame, u is the
# relative-frame velocity, and r = x - x_axis is the position relative
# to a point on the rotation axis.
#
# The force primitives `coriolis_force` and `centrifugal_force` take 3D
# SVectors so they work uniformly for planar MRF (ω_z only) and truly
# 3D MRF alike.

using StaticArrays: SVector
using LinearAlgebra: cross

"""
    coriolis_force(omega, U, rho) -> SVector{3, T}

Coriolis body-force density `-2 ρ (ω × U)` per unit volume. Returns a
3-vector; planar problems embed the 2D velocity as `U = (Ux, Uy, 0)`.
"""
@inline function coriolis_force(
        omega::SVector{3, T}, U::SVector{3, T}, rho::T,
    ) where {T}
    return -T(2) * rho * cross(omega, U)
end

"""
    centrifugal_force(omega, r, rho) -> SVector{3, T}

Centrifugal body-force density `-ρ (ω × (ω × r))` per unit volume. `r`
is the position relative to a point on the rotation axis.
"""
@inline function centrifugal_force(
        omega::SVector{3, T}, r::SVector{3, T}, rho::T,
    ) where {T}
    return -rho * cross(omega, cross(omega, r))
end

"""
    mrf_cell_source(zone::MRFZone{T}, x_cell, u_cell, rho) -> SVector{3, T}

Per-cell total MRF momentum source
`coriolis_force(ω, u, ρ) + centrifugal_force(ω, r, ρ)`.

Caller is responsible for ensuring `cell_index ∈ zone.cells`; this
function does not check membership (use `add_mrf_source!` for the
mesh-level loop that does).
"""
@inline function mrf_cell_source(
        zone::MRFZone{T}, x_cell::SVector{3, T},
        u_cell::SVector{3, T}, rho::T,
    ) where {T}
    r_vec = x_cell - zone.origin
    return coriolis_force(zone.omega, u_cell, rho) +
        centrifugal_force(zone.omega, r_vec, rho)
end

"""
    add_mrf_source!(source_U, U, mesh, zone::MRFZone{T}, rho) -> source_U

Accumulate (`+=`) the MRF momentum source density into `source_U` for
every cell in `zone.cells`. Intended signature for the collocated
momentum assembly hook.

# Arguments
- `source_U::AbstractVector{<:SVector{3, T}}` — per-cell accumulator, one
  entry per mesh cell.
- `U::AbstractVector{<:SVector{3, T}}` — current relative-frame velocity
  field (length = number of cells).
- `mesh` — any object supporting `cell_center(mesh, c)` returning an
  `SVector{Dim, T}` (2D is lifted to 3D by appending a zero).
- `zone::MRFZone{T}` — zone descriptor.
- `rho::T` — constant density (kg/m³) in the zone.

Cells outside the zone are left untouched; out-of-range cell indices in
`zone.cells` raise `BoundsError` via normal indexing.
"""
function add_mrf_source!(
        source_U::AbstractVector{SVector{3, T}},
        U::AbstractVector{SVector{3, T}},
        mesh,
        zone::MRFZone{T},
        rho::T,
    ) where {T}
    for c in zone.cells
        x3 = _lift_to_3d(cell_center(mesh, c), T)
        u3 = U[c]
        source_U[c] = source_U[c] + mrf_cell_source(zone, x3, u3, rho)
    end
    return source_U
end

# Lift a 2D SVector to 3D with z = 0; pass through 3D unchanged. Used so
# the same `add_mrf_source!` works against 2D Cartesian meshes where
# `cell_center` returns an SVector{2, T}.
@inline _lift_to_3d(x::SVector{3, T}, ::Type{T}) where {T} = x
@inline _lift_to_3d(x::SVector{2, T}, ::Type{T}) where {T} =
    SVector{3, T}(x[1], x[2], zero(T))

# ── Frame velocity / face-flux frame conversion (makeRelative) ───────

"""
    mrf_frame_velocity(zone::MRFZone{T}, x) -> SVector{3, T}

Frame (rigid-rotation) velocity `Ω × (x - origin)` at position `x`
(2D positions are lifted to 3D with `z = 0`).  For a planar zone with
`Ω = (0, 0, ω_z)` this is the exact in-plane `ω_z (-(y-y₀), x-x₀)` — the
proper 2D behaviour that replaces the `_cross` 2D stub.
"""
@inline function mrf_frame_velocity(zone::MRFZone{T}, x::SVector{3, T}) where {T}
    return cross(zone.omega, x - zone.origin)
end
@inline mrf_frame_velocity(zone::MRFZone{T}, x::SVector{2, T}) where {T} =
    mrf_frame_velocity(zone, _lift_to_3d(x, T))

"""
    mrf_frame_flux(zone::MRFZone{T}, mesh, f::Int) -> T

Frame flux `(Ω × (x_f - origin)) · S_f` through face `f` — the volumetric
flux carried by the rigid frame rotation.  Subtracting it from an
absolute flux gives the relative flux (OpenFOAM `makeRelative`).
"""
@inline function mrf_frame_flux(zone::MRFZone{T}, mesh, f::Int) where {T}
    x_f = _lift_to_3d(face_center(mesh, f), T)
    S_f = _lift_to_3d(face_normal_area(mesh, f), T)
    return dot(mrf_frame_velocity(zone, x_f), S_f)
end

"""
    _mrf_zone_id_by_cell(mesh, zones) -> Vector{Int}

Per-cell zone index (`0` = not in any zone).  Zones are assumed disjoint
(enforce via [`build_multi_mrf_from_zones`](@ref)); with overlapping
zones the last zone in the list wins.
"""
function _mrf_zone_id_by_cell(mesh, zones::Vector{MRFZone{T}}) where {T}
    nc = length(mesh.cell_volumes)
    zone_id = zeros(Int, nc)
    for (k, zone) in enumerate(zones)
        for c in zone.cells
            zone_id[c] = k
        end
    end
    return zone_id
end

"""
    mrf_make_relative!(phi_values, mesh, zones)

Convert an absolute face flux to the MRF relative flux in place
(OpenFOAM `MRFZone::makeRelative` + `correctBoundaryFlux` equivalent):
for every face with its owner — or, for internal faces, its neighbour —
inside a zone, subtract the frame flux `(Ω × r_f) · S_f`.

Interface faces (one cell in the zone, one outside) use the zone of
whichever cell is inside.  As in OpenFOAM, MRF interfaces should be
(approximate) surfaces of revolution about the rotation axis so the frame
flux through them vanishes; the rigid-rotation velocity field is exactly
divergence-free (linear in `x`), so the conversion never changes the
discrete divergence of the flux field — mass conservation across the zone
interface is preserved identically.

`phi_values` is any face-indexed `AbstractVector` (e.g.
`state.phi.values` or a raw `phi_HbyA` vector).
"""
function mrf_make_relative!(
        phi_values::AbstractVector{T}, mesh, zones::Vector{MRFZone{T}},
    ) where {T}
    return _mrf_shift_flux!(phi_values, mesh, zones, -one(T))
end

"""
    mrf_make_absolute!(phi_values, mesh, zones)

Inverse of [`mrf_make_relative!`](@ref): add the frame flux back so
`mrf_make_absolute!(mrf_make_relative!(phi, ...), ...)` returns the
original flux exactly.
"""
function mrf_make_absolute!(
        phi_values::AbstractVector{T}, mesh, zones::Vector{MRFZone{T}},
    ) where {T}
    return _mrf_shift_flux!(phi_values, mesh, zones, one(T))
end

function _mrf_shift_flux!(
        phi_values::AbstractVector{T}, mesh, zones::Vector{MRFZone{T}},
        sgn::T,
    ) where {T}
    zone_id = _mrf_zone_id_by_cell(mesh, zones)
    nf = size(mesh.face_cells, 2)
    @inbounds for f in 1:nf
        P = owner(mesh, f)
        k = zone_id[P]
        if k == 0 && is_internal_face(mesh, f)
            k = zone_id[neighbour(mesh, f)]
        end
        k == 0 && continue
        phi_values[f] += sgn * mrf_frame_flux(zones[k], mesh, f)
    end
    return phi_values
end
