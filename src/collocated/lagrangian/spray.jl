# lagrangian/spray.jl — Spray breakup models for Lagrangian particle tracking
#
# Implements two secondary breakup models:
# - TAB (Taylor Analogy Breakup) — oscillation analogy, suitable for low We
# - KHRT (Kelvin-Helmholtz Rayleigh-Taylor) — wave instability, suitable for
#   high-speed sprays

"""Abstract supertype for spray breakup models."""
abstract type AbstractBreakupModel end

"""
    TABBreakup{T} <: AbstractBreakupModel

Taylor Analogy Breakup model.  Models droplet distortion as a
damped spring-mass system; breakup occurs when the Weber number
exceeds `We_crit`.

# Fields
- `We_crit::T` --- critical Weber number (default 12.0)
- `C_b::T` --- breakup constant controlling child size (default 0.5)
"""
struct TABBreakup{T} <: AbstractBreakupModel
    We_crit::T
    C_b::T
end

"""
    TABBreakup(; We_crit = 12.0, C_b = 0.5)

Construct a [`TABBreakup`](@ref) model with default constants.
"""
TABBreakup(; We_crit::Real = 12.0, C_b::Real = 0.5) = TABBreakup(Float64(We_crit), Float64(C_b))

"""
    KHRTBreakup{T} <: AbstractBreakupModel

Kelvin-Helmholtz / Rayleigh-Taylor hybrid breakup model.  Combines
KH surface wave stripping (dominant at high speeds) with RT
instability (dominant when deceleration is large).

# Fields
- `B0::T` --- KH wavelength constant (default 0.61)
- `B1::T` --- KH breakup time constant (default 10.0)
"""
struct KHRTBreakup{T} <: AbstractBreakupModel
    B0::T
    B1::T
end

"""
    KHRTBreakup(; B0 = 0.61, B1 = 10.0)

Construct a [`KHRTBreakup`](@ref) model with default constants.
"""
KHRTBreakup(; B0::Real = 0.61, B1::Real = 10.0) = KHRTBreakup(Float64(B0), Float64(B1))

"""
    weber_number(U_rel, d, rho_f, sigma) -> T

Compute the aerodynamic Weber number:

    We = rho_f * |U_rel|^2 * d / sigma

# Arguments
- `U_rel` --- relative velocity magnitude or `SVector`
- `d` --- droplet diameter [m]
- `rho_f` --- gas/fluid density [kg/m^3]
- `sigma` --- surface tension coefficient [N/m]
"""
function weber_number(U_rel::SVector{Dim, T}, d::T, rho_f::T, sigma::T) where {Dim, T}
    return rho_f * dot(U_rel, U_rel) * d / sigma
end

function weber_number(U_rel_mag::T, d::T, rho_f::T, sigma::T) where {T <: Real}
    return rho_f * U_rel_mag^2 * d / sigma
end

"""
    should_breakup(model::TABBreakup, We) -> Bool

Return `true` if the Weber number exceeds the critical threshold
for the TAB model.
"""
function should_breakup(model::TABBreakup, We::Real)
    return We > model.We_crit
end

"""
    breakup_diameter(model::TABBreakup, d_parent, We) -> T

Compute the child droplet diameter after TAB breakup:

    d_child = d_parent * (We_crit / We)^(1/3)

Only valid when `We > We_crit`.
"""
function breakup_diameter(model::TABBreakup{T}, d_parent::T, We::T) where {T}
    return d_parent * (model.We_crit / We)^(one(T) / T(3))
end

"""
    apply_breakup!(tracker, breakup_model, rho_f, sigma)

Check each active particle in `tracker` for breakup conditions and
split particles that exceed the critical Weber number.

For each particle that breaks up:
1. The parent diameter is reduced to `d_child`
2. A new child particle is created with the same diameter and velocity
3. Both particles retain the parent's velocity (no velocity perturbation)

# Arguments
- `tracker::ParticleTracker` --- particle tracker (modified in-place)
- `breakup_model::TABBreakup` --- breakup model with critical We
- `rho_f::T` --- surrounding fluid density [kg/m^3]
- `sigma::T` --- surface tension coefficient [N/m]
"""
function apply_breakup!(
        tracker::ParticleTracker{Dim, T},
        breakup_model::TABBreakup{BT},
        rho_f::T,
        sigma::T,
    ) where {Dim, T, BT}
    new_particles = LagrangianParticle{Dim, T}[]

    for p in tracker.particles
        p.active || continue

        d_p = T(p.properties[:diameter])
        rho_p = T(p.properties[:density])

        # Weber number from slip velocity (use velocity magnitude as proxy
        # for relative velocity — full coupling would use interpolated U_f)
        U_mag = norm(p.velocity)
        We = rho_f * U_mag^2 * d_p / sigma

        if should_breakup(breakup_model, We)
            d_child = breakup_diameter(breakup_model, d_p, T(We))

            # Update parent to child size
            p.properties[:diameter] = d_child
            m_child = T(pi) / 6 * d_child^3 * rho_p
            p.properties[:mass] = m_child

            # Create a new child particle with same properties
            new_id = tracker.next_id[]
            tracker.next_id[] = new_id + 1
            child = LagrangianParticle{Dim, T}(
                p.position,
                p.velocity,
                p.cell_index,
                new_id,
                true,
                Dict{Symbol, Any}(
                    :diameter => d_child,
                    :density => rho_p,
                    :mass => m_child,
                    :temperature => p.properties[:temperature],
                    :Cp => p.properties[:Cp],
                ),
            )
            push!(new_particles, child)
        end
    end

    append!(tracker.particles, new_particles)
    return nothing
end
