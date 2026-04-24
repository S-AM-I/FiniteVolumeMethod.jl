# cavitation/schnerr_sauer.jl — Schnerr-Sauer (2001) cavitation model.
#
# Physics: Rayleigh-Plesset-based bubble-dynamics mass transfer with a
# fixed bubble number density `n_0` [1/m³].
#
#   R_B   = ( 3 · α_v / (4π · n_0 · (1 − α_v)) )^(1/3)           [m]
#   m_dot = (3 ρ_v · α_v · (1 − α_v) / R_B) · sign(p_sat − p) ·
#           sqrt( |p_sat − p| · 2 / (3 ρ_l) )                     [kg/(m³·s)]
#
# Vanishes at α_v = 0 and α_v = 1 (by construction via the α_v·(1−α_v)
# factor). A small floor is applied to `(1 − α_v)` inside R_B to
# prevent division-by-zero without changing the algebraic limits.
#
# Reference: Schnerr & Sauer, ICMF-2001, "Physical and numerical
# modeling of unsteady cavitation dynamics".

"""
    SchnerrSauerModel{T}

Schnerr-Sauer (2001) Rayleigh-Plesset cavitation model.

# Fields
- `n_0::T` — nuclei number density [1/m³]. Default `1.0e13` per OpenFOAM.
"""
struct SchnerrSauerModel{T} <: AbstractCavitationVaporModel{T}
    n_0::T
end

function SchnerrSauerModel(; n_0::Real = 1.0e13)
    T = typeof(float(n_0))
    return SchnerrSauerModel{T}(T(n_0))
end

"""
    schnerr_sauer_bubble_radius(model, alpha_v) -> T

Closed-form bubble radius `R_B` [m] for volume fraction `α_v`:

    R_B = ( 3 α_v / (4π n_0 (1 − α_v)) )^(1/3)

Returns `zero(T)` when `α_v == 0` exactly.
"""
function schnerr_sauer_bubble_radius(m::SchnerrSauerModel{T}, alpha_v::T) where {T}
    iszero(alpha_v) && return zero(T)
    denom = T(4) * T(pi) * m.n_0 * max(one(T) - alpha_v, eps(T))
    return cbrt(T(3) * alpha_v / denom)
end

"""
    schnerr_sauer_rate(model, p, alpha_v, rho_l, rho_v, p_sat) -> T

Net Schnerr-Sauer vapour mass source [kg/(m³·s)] with the sign
convention
    positive ⇒ vapour produced (p < p_sat)
    negative ⇒ vapour destroyed (p > p_sat).

`sign(p_sat − p)` preserves the above convention: a drop below saturation
(p < p_sat) gives a positive rate.
"""
function schnerr_sauer_rate(
        m::SchnerrSauerModel{T}, p::T, alpha_v::T, rho_l::T, rho_v::T, p_sat::T,
    ) where {T}
    # Vanish at the two trivial limits.
    (iszero(alpha_v) || isone(alpha_v)) && return zero(T)
    R_B = schnerr_sauer_bubble_radius(m, alpha_v)
    R_B <= zero(T) && return zero(T)
    dp = p_sat - p
    iszero(dp) && return zero(T)
    speed = sqrt(abs(dp) * T(2) / (T(3) * max(rho_l, eps(T))))
    prefactor = T(3) * rho_v * alpha_v * (one(T) - alpha_v) / R_B
    return prefactor * sign(dp) * speed
end
