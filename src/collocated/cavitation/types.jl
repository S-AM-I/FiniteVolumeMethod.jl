# cavitation/types.jl — Cavitation mass-transfer models (Stage 6d)
#
# Each model returns the vapour-generation / condensation mass rate per
# unit volume (kg/(m³·s)) that augments the VOF α-transport equation as
# a source term. When the local pressure drops below the saturation
# pressure, the vapour fraction grows; above p_sat, vapour condenses.
#
# All three implemented models take the same inputs:
#   p        — local pressure
#   alpha_l  — liquid volume fraction (so vapour fraction = 1 - alpha_l)
#   rho_l    — liquid density
#   rho_v    — vapour density
#   p_sat    — saturation pressure
#
# and return a pair (m_plus, m_minus) where m_plus is the evaporation
# rate (→ vapour) and m_minus is the condensation rate (→ liquid),
# each non-negative.
#
# References:
# - Kunz et al. (2000), Computers & Fluids 29, 849-875.
# - Schnerr & Sauer (2001), ICMF-2001.
# - Merkle et al. (1998), 3rd Int. Symp. Cavitation.

"""
    AbstractCavitationModel{T}

Umbrella for cavitation mass-transfer models. Each model exposes
`cavitation_source(model, p, alpha_l, rho_l, rho_v, p_sat)` returning
`(m_plus, m_minus)` in kg/(m³·s).
"""
abstract type AbstractCavitationModel{T} end

"""
    KunzCavitation{T}(; C_prod = 1.0e2, C_dest = 1.0e2, t_inf = 1.0e-3, U_inf = 1.0)

Kunz et al. (2000) — most common industrial model for propeller and
hydrofoil cavitation. Vapour mass production / destruction:

    m_+ = C_prod · ρ_v · α_l · max(0, p − p_sat) / (0.5 ρ_l U_inf² · t_inf)
    m_- = C_dest · ρ_v · (1 − α_l) · α_l² / t_inf

where `t_inf` and `U_inf` are characteristic time and velocity scales.
"""
struct KunzCavitation{T} <: AbstractCavitationModel{T}
    C_prod::T
    C_dest::T
    t_inf::T
    U_inf::T
end
function KunzCavitation(;
        C_prod::Real = 1.0e2, C_dest::Real = 1.0e2,
        t_inf::Real = 1.0e-3, U_inf::Real = 1.0,
    )
    T = promote_type(
        typeof(float(C_prod)), typeof(float(C_dest)),
        typeof(float(t_inf)), typeof(float(U_inf))
    )
    return KunzCavitation{T}(T(C_prod), T(C_dest), T(t_inf), T(U_inf))
end

function cavitation_source(
        m::KunzCavitation{T}, p::T, alpha_l::T, rho_l::T, rho_v::T, p_sat::T,
    ) where {T}
    ref_pressure = T(0.5) * rho_l * m.U_inf^2 * m.t_inf
    ref_pressure = max(ref_pressure, eps(T))
    m_plus = m.C_prod * rho_v * alpha_l * max(zero(T), p - p_sat) / ref_pressure
    m_minus = m.C_dest * rho_v * (one(T) - alpha_l) * alpha_l^2 / m.t_inf *
        (p < p_sat ? one(T) : zero(T))
    return m_plus, m_minus
end

"""
    SchnerrSauerCavitation{T}(; n_b = 1.0e13, R_b = 1.0e-6)

Schnerr-Sauer (2001) model — physics-based, tied to a single bubble
population of number density `n_b` (1/m³) and equilibrium radius
`R_b` (m). Vapour volume fraction α_v = n_b · (4π/3) · R_b³ is used
to compute mass transfer via a Rayleigh-Plesset-like Ṙ.

For numerical stability we pre-compute the formal Rayleigh expansion
`R_b_eff = R_b` and drive m_± with `sign(p_sat − p) · |Ṙ|`.
"""
struct SchnerrSauerCavitation{T} <: AbstractCavitationModel{T}
    n_b::T
    R_b::T
end
function SchnerrSauerCavitation(; n_b::Real = 1.0e13, R_b::Real = 1.0e-6)
    T = promote_type(typeof(float(n_b)), typeof(float(R_b)))
    return SchnerrSauerCavitation{T}(T(n_b), T(R_b))
end

function cavitation_source(
        m::SchnerrSauerCavitation{T}, p::T, alpha_l::T, rho_l::T, rho_v::T, p_sat::T,
    ) where {T}
    # alpha_v from bubble density
    alpha_v = one(T) - alpha_l
    # Rayleigh expansion: Ṙ² ≈ 2·|p_sat − p| / (3 ρ_l)
    dp = p_sat - p
    R_dot_sq = T(2) * abs(dp) / (T(3) * max(rho_l, eps(T)))
    R_dot = sqrt(max(R_dot_sq, zero(T))) * sign(dp)
    # Surface area of bubble distribution per unit volume:
    surf_area_per_vol = T(3) * alpha_v / max(m.R_b, eps(T))
    # Mass-transfer magnitude
    rate = rho_v * surf_area_per_vol * R_dot
    if dp > 0
        return abs(rate), zero(T)  # evaporation
    else
        return zero(T), abs(rate)  # condensation
    end
end

"""
    MerkleCavitation{T}(; C_prod = 1.0e2, C_dest = 1.0e2, t_inf = 1.0e-3, U_inf = 1.0)

Merkle et al. (1998) — linearised version of Kunz's source. Differs in
that the destruction term uses `α_l · α_v` rather than `α_v · α_l²`,
giving a more symmetric response near α_l = 0.5.
"""
struct MerkleCavitation{T} <: AbstractCavitationModel{T}
    C_prod::T
    C_dest::T
    t_inf::T
    U_inf::T
end
function MerkleCavitation(;
        C_prod::Real = 1.0e2, C_dest::Real = 1.0e2,
        t_inf::Real = 1.0e-3, U_inf::Real = 1.0,
    )
    T = promote_type(
        typeof(float(C_prod)), typeof(float(C_dest)),
        typeof(float(t_inf)), typeof(float(U_inf))
    )
    return MerkleCavitation{T}(T(C_prod), T(C_dest), T(t_inf), T(U_inf))
end

function cavitation_source(
        m::MerkleCavitation{T}, p::T, alpha_l::T, rho_l::T, rho_v::T, p_sat::T,
    ) where {T}
    ref_pressure = T(0.5) * rho_l * m.U_inf^2 * m.t_inf
    ref_pressure = max(ref_pressure, eps(T))
    alpha_v = one(T) - alpha_l
    m_plus = m.C_prod * rho_v * alpha_l * max(zero(T), p - p_sat) / ref_pressure
    m_minus = m.C_dest * rho_v * alpha_l * alpha_v / m.t_inf *
        (p < p_sat ? one(T) : zero(T))
    return m_plus, m_minus
end

# ---------------------------------------------------------------------------
# v3.0 fast-path API: vapour-fraction (α_v) driven models with direct
# per-cell source arrays. These models are the modern OpenFOAM-style
# counterparts to the legacy (α_l, m_plus, m_minus) API above and are
# what the v3.0 VOF solver wires into the α-transport equation.
# ---------------------------------------------------------------------------

"""
    AbstractCavitationVaporModel{T}

Vapour-fraction-driven cavitation mass-transfer model. Concrete types
implement

    compute_vapor_source(model, p, alpha_v, mesh, props) -> Vector{T}

returning the per-cell vapour mass source [kg/(m³·s)]. Positive values
indicate vapour production (p < p_sat); negative values indicate
condensation (p > p_sat).
"""
abstract type AbstractCavitationVaporModel{T} end

"""
    CavitationProperties{T}

Bundle of two-phase fluid properties required by all vapour-fraction
cavitation models.

# Fields
- `rho_l::T` — liquid density [kg/m³].
- `rho_v::T` — vapour density [kg/m³].
- `p_sat::T` — saturation pressure [Pa].
"""
struct CavitationProperties{T}
    rho_l::T
    rho_v::T
    p_sat::T
end
function CavitationProperties(;
        rho_l::Real = 1000.0,
        rho_v::Real = 0.02308,
        p_sat::Real = 2300.0,
    )
    T = promote_type(typeof(float(rho_l)), typeof(float(rho_v)), typeof(float(p_sat)))
    return CavitationProperties{T}(T(rho_l), T(rho_v), T(p_sat))
end
