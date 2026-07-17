# cavitation/merkle.jl — Merkle (1998) pressure-based cavitation model.
#
# Merkle, Feng & Buelow (1998), 3rd Int. Symp. Cavitation.
#
# Two branches driven by the pressure deficit/excess relative to
# saturation:
#
#   m_dot_vap  = C_dest · ρ_l · α_l · min(0, p − p_sat) /
#                (0.5 · ρ_l · U_inf² · τ_inf)
#   m_dot_cond = C_prod · ρ_v · α_v · max(0, p − p_sat) /
#                (0.5 · ρ_l · U_inf² · τ_inf)
#
# `m_dot_vap` is ≤ 0 (destruction of liquid, i.e. production of
# vapour). `m_dot_cond` is ≥ 0 (production of liquid, i.e. destruction
# of vapour).  The net **vapour source** returned by `merkle_rate` uses
# the sign convention
#
#   positive ⇒ vapour produced
#   negative ⇒ vapour destroyed.

"""
    MerkleModel{T}

Merkle (1998) pressure-based cavitation mass-transfer model.

# Fields
- `C_dest::T` — vaporisation (liquid-destruction) coefficient.
- `C_prod::T` — condensation  (liquid-production) coefficient.
- `U_inf::T`  — reference free-stream velocity [m/s].
- `t_inf::T`  — characteristic time scale [s] (τ_inf). Typically
  `L_inf / U_inf`.

Use `MerkleModel(; C_dest, C_prod, U_inf, t_inf)`.
"""
struct MerkleModel{T} <: AbstractCavitationVaporModel{T}
    C_dest::T
    C_prod::T
    U_inf::T
    t_inf::T
end

function MerkleModel(;
        C_dest::Real = 1.0, C_prod::Real = 80.0,
        U_inf::Real = 1.0, t_inf::Real = 1.0,
    )
    T = promote_type(
        typeof(float(C_dest)), typeof(float(C_prod)),
        typeof(float(U_inf)), typeof(float(t_inf))
    )
    return MerkleModel{T}(T(C_dest), T(C_prod), T(U_inf), T(t_inf))
end

"""
    merkle_vap_rate(model, p, alpha_v, rho_l, p_sat) -> T

Raw vaporisation branch (≤ 0 for all inputs, zero when p ≥ p_sat):

    C_dest · ρ_l · α_l · min(0, p − p_sat) / (0.5 · ρ_l · U_inf² · τ_inf)
"""
function merkle_vap_rate(
        m::MerkleModel{T}, p::T, alpha_v::T, rho_l::T, p_sat::T,
    ) where {T}
    alpha_l = one(T) - alpha_v
    ref = T(0.5) * rho_l * m.U_inf^2 * m.t_inf
    ref = max(ref, eps(T))
    return m.C_dest * rho_l * alpha_l * min(zero(T), p - p_sat) / ref
end

"""
    merkle_cond_rate(model, p, alpha_v, rho_v, rho_l, p_sat) -> T

Raw condensation branch (≥ 0, zero when p ≤ p_sat):

    C_prod · ρ_v · α_v · max(0, p − p_sat) / (0.5 · ρ_l · U_inf² · τ_inf)

Note: the reference-pressure denominator is built with `ρ_l` per the
Merkle (1998) non-dimensionalisation.
"""
function merkle_cond_rate(
        m::MerkleModel{T}, p::T, alpha_v::T, rho_v::T, rho_l::T, p_sat::T,
    ) where {T}
    ref = T(0.5) * rho_l * m.U_inf^2 * m.t_inf
    ref = max(ref, eps(T))
    return m.C_prod * rho_v * alpha_v * max(zero(T), p - p_sat) / ref
end

"""
    merkle_rate(model, p, alpha_v, rho_l, rho_v, p_sat) -> T

Net vapour mass source [kg/(m³·s)], sign convention
    positive ⇒ vapour produced (p < p_sat)
    negative ⇒ vapour destroyed (p > p_sat).

Composed as `−m_dot_vap − m_dot_cond`:
 - `m_dot_vap ≤ 0` ⇒ `−m_dot_vap ≥ 0` supplies vapour production.
 - `m_dot_cond ≥ 0` ⇒ `−m_dot_cond ≤ 0` subtracts vapour (condensation).
"""
function merkle_rate(
        m::MerkleModel{T}, p::T, alpha_v::T, rho_l::T, rho_v::T, p_sat::T,
    ) where {T}
    m_vap = merkle_vap_rate(m, p, alpha_v, rho_l, p_sat)
    m_cond = merkle_cond_rate(m, p, alpha_v, rho_v, rho_l, p_sat)
    return -m_vap - m_cond
end
