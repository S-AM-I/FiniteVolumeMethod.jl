# cavitation/kunz.jl — Kunz (2000) cavitation mass-transfer model.
#
# Kunz et al., Computers & Fluids 29 (2000), pp. 849-875.
#
# The v3.0 fast-path API exposes three branch functions for direct
# algebraic verification and one combined `kunz_rate` used by
# `compute_vapor_source`:
#
#   m_dot_vap  = C_v · ρ_l · α_l · min(0, p − p_sat) /
#                (0.5 · ρ_l · U_inf²) / τ_inf
#   m_dot_cond = C_c · ρ_v · α_v² · (1 − α_v) / τ_inf     (p > p_sat only)
#
# Sign convention for the returned vapour source:
#
#   positive ⇒ vapour produced (evaporation, p < p_sat)
#   negative ⇒ vapour destroyed (condensation, p > p_sat)

"""
    KunzModel{T}

Kunz (2000) cavitation mass-transfer model, driven by the vapour volume
fraction `α_v`.

# Fields
- `C_v::T` — vaporisation rate coefficient (dimensionless).
- `C_c::T` — condensation rate coefficient (dimensionless).
- `U_inf::T` — reference free-stream velocity [m/s].
- `L_inf::T` — reference length scale [m]; sets τ_inf = L_inf / U_inf.

Use `KunzModel(; C_v, C_c, U_inf, L_inf)`.
"""
struct KunzModel{T} <: AbstractCavitationVaporModel{T}
    C_v::T
    C_c::T
    U_inf::T
    L_inf::T
end

function KunzModel(;
        C_v::Real = 100.0, C_c::Real = 100.0,
        U_inf::Real = 1.0, L_inf::Real = 1.0,
    )
    T = promote_type(
        typeof(float(C_v)), typeof(float(C_c)),
        typeof(float(U_inf)), typeof(float(L_inf))
    )
    return KunzModel{T}(T(C_v), T(C_c), T(U_inf), T(L_inf))
end

"""
    tau_inf(m::KunzModel)

Characteristic time τ_inf = L_inf / U_inf [s].
"""
tau_inf(m::KunzModel{T}) where {T} = m.L_inf / max(m.U_inf, eps(T))

"""
    kunz_vap_rate(model, p, alpha_v, rho_l, p_sat) -> T

Raw vaporisation branch:

    C_v · ρ_l · α_l · min(0, p − p_sat) / (0.5 · ρ_l · U_inf²) / τ_inf

Negative (or zero) for all inputs. Zero when p ≥ p_sat or α_l = 0.
"""
function kunz_vap_rate(
        m::KunzModel{T}, p::T, alpha_v::T, rho_l::T, p_sat::T,
    ) where {T}
    alpha_l = one(T) - alpha_v
    tau = tau_inf(m)
    ref = T(0.5) * rho_l * m.U_inf^2
    ref = max(ref, eps(T))
    return m.C_v * rho_l * alpha_l * min(zero(T), p - p_sat) / ref / tau
end

"""
    kunz_cond_rate(model, p, alpha_v, rho_v, p_sat) -> T

Raw condensation branch (gated on p > p_sat):

    C_c · ρ_v · α_v² · (1 − α_v) / τ_inf   if p > p_sat
    0                                       otherwise

Non-negative.
"""
function kunz_cond_rate(
        m::KunzModel{T}, p::T, alpha_v::T, rho_v::T, p_sat::T,
    ) where {T}
    (p > p_sat) || return zero(T)
    tau = tau_inf(m)
    return m.C_c * rho_v * alpha_v^2 * (one(T) - alpha_v) / tau
end

"""
    kunz_rate(model, p, alpha_v, rho_l, rho_v, p_sat) -> T

Net vapour mass source [kg/(m³·s)] with the sign convention
    positive ⇒ vapour produced
    negative ⇒ vapour destroyed.

Composed from `kunz_vap_rate` and `kunz_cond_rate`:

    source = −m_dot_vap − m_dot_cond

because `m_dot_vap` is a negative number representing vaporisation,
so producing vapour means flipping its sign.
"""
function kunz_rate(
        m::KunzModel{T}, p::T, alpha_v::T, rho_l::T, rho_v::T, p_sat::T,
    ) where {T}
    m_vap = kunz_vap_rate(m, p, alpha_v, rho_l, p_sat)
    m_cond = kunz_cond_rate(m, p, alpha_v, rho_v, p_sat)
    return -m_vap - m_cond
end
