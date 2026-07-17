# lagrangian/drag_models.jl — Particle drag force models
#
# Provides Stokes and Schiller-Naumann drag correlations for
# Lagrangian particle tracking on collocated FVM meshes.

"""Abstract supertype for particle drag models."""
abstract type AbstractDragModel end

"""
    StokesDrag <: AbstractDragModel

Stokes drag law (`Cd = 24 / Re`), valid for `Re_p << 1`.

Drag force: `F = (m_p / tau_p) * (U_f - U_p)` where
`tau_p = rho_p * d^2 / (18 * mu_f)`.
"""
struct StokesDrag <: AbstractDragModel end

"""
    SchillerNaumann <: AbstractDragModel

Schiller-Naumann drag correlation with correction factor
`f(Re) = 1 + 0.15 * Re^0.687`, capped at `Re = 1000`.
"""
struct SchillerNaumann <: AbstractDragModel end

"""
    _particle_reynolds(U_fluid, U_particle, diameter, rho_f, mu_f) -> T

Compute the particle Reynolds number: `Re_p = rho_f * |U_f - U_p| * d / mu_f`.
"""
function _particle_reynolds(
        U_fluid::SVector{Dim, T},
        U_particle::SVector{Dim, T},
        diameter::T, rho_f::T, mu_f::T,
    ) where {Dim, T}
    slip = norm(U_fluid - U_particle)
    return rho_f * slip * diameter / mu_f
end

"""
    compute_drag_force(model, U_fluid, U_particle, diameter, density_p, rho_f, mu_f)

Compute the drag force vector on a particle.

Returns an `SVector{Dim, T}` force in the direction of the slip velocity.
"""
function compute_drag_force(
        ::StokesDrag,
        U_fluid::SVector{Dim, T},
        U_particle::SVector{Dim, T},
        diameter::T, density_p::T,
        rho_f::T, mu_f::T,
    ) where {Dim, T}
    mass = T(pi) / 6 * diameter^3 * density_p
    tau_p = density_p * diameter^2 / (18 * mu_f)
    return (mass / tau_p) * (U_fluid - U_particle)
end

function compute_drag_force(
        ::SchillerNaumann,
        U_fluid::SVector{Dim, T},
        U_particle::SVector{Dim, T},
        diameter::T, density_p::T,
        rho_f::T, mu_f::T,
    ) where {Dim, T}
    mass = T(pi) / 6 * diameter^3 * density_p
    tau_p = density_p * diameter^2 / (18 * mu_f)
    Re_p = _particle_reynolds(U_fluid, U_particle, diameter, rho_f, mu_f)
    # Schiller-Naumann correction, capped at Re = 1000
    Re_capped = min(Re_p, T(1000))
    f_corr = one(T) + T(0.15) * Re_capped^T(0.687)
    return (mass / tau_p) * f_corr * (U_fluid - U_particle)
end
