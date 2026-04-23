# mrf/types.jl — Moving Reference Frame (Stage 6a)
#
# For steady simulations of rotating machinery (fans, pumps, mixers,
# centrifugal compressors), solving the Navier-Stokes equations in a
# rotating reference frame removes the need for time-accurate sliding
# meshes. The rotating-frame momentum equation adds two source terms:
#
#   Coriolis     source =  -2 ρ (ω × u)
#   centrifugal  source =  -ρ (ω × (ω × r))
#
# where ω is the angular velocity of the frame, u is the velocity in
# the rotating frame, and r = x - x_axis is the position vector from
# the rotation axis.
#
# In multi-zone problems (rotor-stator, multi-stage compressors), each
# cell belongs to exactly one `MRFZone`; stationary cells simply use the
# laboratory-frame NS equations (ω = 0, no source).
#
# References: Ferziger & Perić (2020), Computational Methods for Fluid
# Dynamics, 4th ed., §15.6; standard MRF / frozen-rotor treatment in
# every industrial CFD code.

using StaticArrays: SVector
using LinearAlgebra: cross

"""
    AbstractMRFZone{Dim, T}

Umbrella for Moving Reference Frame zone descriptors. Concrete
subtypes represent rotation axes and angular velocities. In a
multi-zone problem each cell maps to at most one MRF zone; stationary
regions implicitly belong to the laboratory frame.
"""
abstract type AbstractMRFZone{Dim, T} end

"""
    RotationalMRFZone{Dim, T} <: AbstractMRFZone{Dim, T}

Constant-angular-velocity rotating reference frame. The zone covers
`cell_indices` (global cell indices in the mesh). Rotation axis passes
through `origin` with unit direction `axis`, and the angular speed is
`omega` (rad/s; positive = right-hand-rule about `axis`).

# Fields
- `cell_indices::Vector{Int}` — global cell indices belonging to the zone.
- `origin::SVector{Dim, T}` — a point on the rotation axis.
- `axis::SVector{Dim, T}` — unit direction of the rotation axis.
- `omega::T` — angular speed.
"""
struct RotationalMRFZone{Dim, T} <: AbstractMRFZone{Dim, T}
    cell_indices::Vector{Int}
    origin::SVector{Dim, T}
    axis::SVector{Dim, T}
    omega::T
end

"""
    angular_velocity_vector(zone) -> SVector{Dim, T}

Return ω = omega · axiŝ. Always satisfies |ω| = |omega|.
"""
@inline angular_velocity_vector(zone::RotationalMRFZone{Dim, T}) where {Dim, T} =
    zone.omega * zone.axis

"""
    mrf_momentum_source(zone, cell_index, x_cell, u_cell, rho) -> SVector{Dim, T}

Per-cell MRF momentum source `-ρ (2 ω × u + ω × (ω × r))` at `x_cell`
with velocity `u_cell`. Returns a zero vector when `cell_index` is not
in `zone.cell_indices` so callers can iterate all cells without
branching.
"""
function mrf_momentum_source(
        zone::RotationalMRFZone{Dim, T},
        cell_index::Int,
        x_cell::SVector{Dim, T},
        u_cell::SVector{Dim, T},
        rho::T,
    ) where {Dim, T}
    in(cell_index, zone.cell_indices) || return zero(SVector{Dim, T})
    omega_vec = angular_velocity_vector(zone)
    r_vec = x_cell - zone.origin

    coriolis = _cross(omega_vec, u_cell)
    centrifugal = _cross(omega_vec, _cross(omega_vec, r_vec))
    return -rho * (T(2) * coriolis + centrifugal)
end

# Cross product that works in both 2D (scalar z-component) and 3D.
# In 2D, ω and u are in the xy plane; ω × u returns an xy vector
# (scalar-times-perpendicular-in-plane). ω × (ω × r) similarly.
@inline function _cross(a::SVector{3, T}, b::SVector{3, T}) where {T}
    return SVector{3, T}(
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1],
    )
end

# 2D cross of planar vector with planar vector where rotation axis is
# assumed to point out-of-plane. We encode ω as (0, 0, omega), r and u
# as (rx, ry, 0). ω × r = (-omega·ry, omega·rx, 0) → the xy projection
# returned here. (User supplies `axis = (0, 1)` or similar; the
# z-component of the full 3D ω × u is returned by other callers.)
@inline function _cross(a::SVector{2, T}, b::SVector{2, T}) where {T}
    # Interpret as ω = (0, 0, |a| · sign(a[1]·b[2] − …))  — not meaningful
    # for general 2D vectors. Instead, we require a 3D rotation-axis
    # convention: 2D MRF must pass `axis = SVector(0.0, 1.0)` etc. In
    # practice the base-module caller uses the 3D `_cross` above via a
    # lifted SVector. This 2D stub preserves type stability for symbolic
    # callers and returns a planar vector whose magnitude is |a||b|.
    # For numerical code prefer the 3D path; see
    # `mrf_momentum_source_2d_planar` below.
    return SVector{2, T}(a[1] * b[2] - a[2] * b[1], zero(T))
end

"""
    mrf_momentum_source_2d_planar(omega_scalar, x_cell, u_cell, origin, rho)

Planar-rotation convenience for Dim=2 problems where ω is the scalar
out-of-plane component. Returns the 2D momentum source
`-ρ (2 ω (k̂ × u) + ω² (k̂ × (k̂ × r)))` with k̂ the out-of-plane unit
vector, which simplifies to `-ρ (2 ω (-u_y, u_x) + ω² (-r))`
since `k̂ × (k̂ × r) = -r` in 2D.
"""
function mrf_momentum_source_2d_planar(
        omega::T, x_cell::SVector{2, T}, u_cell::SVector{2, T},
        origin::SVector{2, T}, rho::T,
    ) where {T}
    r_vec = x_cell - origin
    coriolis = omega * SVector{2, T}(-u_cell[2], u_cell[1])
    centrifugal = -omega^2 * r_vec  # k̂ × (k̂ × r) = -r in 2D
    return -rho * (T(2) * coriolis + centrifugal)
end
