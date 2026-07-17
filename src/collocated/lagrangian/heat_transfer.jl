# lagrangian/heat_transfer.jl — Particle heat transfer models
#
# Provides Ranz-Marshall correlation for convective heat transfer
# between a spherical particle and the surrounding fluid.

"""Abstract supertype for particle heat transfer models."""
abstract type AbstractParticleHeatTransfer end

"""
    RanzMarshall <: AbstractParticleHeatTransfer

Ranz-Marshall correlation: `Nu = 2 + 0.6 * Re^0.5 * Pr^0.33`.

Heat transfer rate: `q = pi * d * k_f * Nu * (T_f - T_p)` [W].
"""
struct RanzMarshall <: AbstractParticleHeatTransfer end

"""
    compute_particle_heat_transfer(model, T_fluid, T_particle,
        U_fluid, U_particle, diameter, rho_f, mu_f, k_f, Pr)

Compute the convective heat transfer rate `q` [W] to the particle.

Positive `q` means heat flows into the particle (T_fluid > T_particle).
"""
function compute_particle_heat_transfer(
        ::RanzMarshall,
        T_fluid::T, T_particle::T,
        U_fluid::SVector{Dim, T},
        U_particle::SVector{Dim, T},
        diameter::T, rho_f::T, mu_f::T,
        k_f::T, Pr::T,
    ) where {Dim, T}
    Re_p = _particle_reynolds(U_fluid, U_particle, diameter, rho_f, mu_f)
    Nu = T(2) + T(0.6) * Re_p^T(0.5) * Pr^T(0.33)
    return T(pi) * diameter * k_f * Nu * (T_fluid - T_particle)
end
