# lagrangian/primary_breakup.jl — Primary atomisation models
#
# Two primary break-up models are provided:
#
# 1. **KH-ACT** (Kelvin-Helmholtz with Aerodynamically-Cascaded Atomisation
#    and Turbulence) — Reitz's curve-fit correlations for the KH wavelength
#    `Λ_KH` and growth rate `Ω_KH` of the most unstable surface wave on a
#    cylindrical liquid jet.
# 2. **LISA** (Linearised Instability Sheet Atomisation) — sheet break-up
#    correlation used for pressure-swirl atomisers and hollow-cone sprays.

using LinearAlgebra: norm

"""
    AbstractPrimaryBreakupModel

Supertype for primary atomisation models. Primary break-up acts on the
liquid core / sheet that emerges directly from the injector orifice,
whereas secondary break-up (TAB, KHRT) acts on already-formed droplets.
"""
abstract type AbstractPrimaryBreakupModel end

# ═════════════════════════════════════════════════════════════════════════
# KH-ACT primary break-up
# ═════════════════════════════════════════════════════════════════════════

"""
    KHACTBreakup{T} <: AbstractPrimaryBreakupModel

Kelvin-Helmholtz with Aerodynamically-Cascaded Atomisation and
Turbulence. Reitz (1987) curve-fit correlations for the fastest-
growing wave on a cylindrical liquid jet of radius `a`:

```
Λ_KH / a = 9.02 · (1 + 0.45·Z^0.5) · (1 + 0.4·Ta^0.7)
                 / (1 + 0.87·We_g^1.67)^0.6

Ω_KH · √(ρ_l·a³/σ) = (0.34 + 0.38·We_g^1.5)
                     / ((1 + Z)·(1 + 1.4·Ta^0.6))
```

with `Z = √We_l / Re_l`, `We_g = ρ_g·|U_rel|²·a/σ`,
`We_l = ρ_l·|U_rel|²·a/σ`, `Re_l = ρ_l·|U_rel|·a/μ_l`,
`Ta = Z·√We_g`.

Child radius and break-up time use Reitz's `B0 = 0.61`, `B1 = 1.73`
defaults:

```
r_child = B0·Λ_KH
τ_b     = 3.726·B1·a / (Λ_KH·Ω_KH)
```

# Fields
- `B0::T` — child-radius constant (default `0.61`)
- `B1::T` — break-up-time constant (default `1.73`)
"""
struct KHACTBreakup{T} <: AbstractPrimaryBreakupModel
    B0::T
    B1::T
end

KHACTBreakup(; B0::Real = 0.61, B1::Real = 1.73) =
    KHACTBreakup{Float64}(Float64(B0), Float64(B1))

"""
    kh_act_wavelength_growth(d_parent, U_rel_mag, ρ_g, ρ_l, μ_l, σ) ->
        (Λ_KH, Ω_KH, We_g, Z, Ta)

Return the KH-ACT wavelength `Λ_KH [m]`, growth rate `Ω_KH [1/s]` and
the dimensionless groups used in Reitz's correlations.
"""
function kh_act_wavelength_growth(
        d_parent::T, U_rel_mag::T,
        ρ_g::T, ρ_l::T, μ_l::T, σ::T,
    ) where {T <: Real}
    a = d_parent / 2
    if U_rel_mag <= eps(T) || σ <= zero(T)
        return T(Inf), zero(T), zero(T), zero(T), zero(T)
    end
    We_g = ρ_g * U_rel_mag^2 * a / σ
    We_l = ρ_l * U_rel_mag^2 * a / σ
    Re_l = ρ_l * U_rel_mag * a / max(μ_l, eps(T))
    Z = sqrt(We_l) / Re_l
    Ta = Z * sqrt(We_g)

    Λ_KH = a * T(9.02) *
        (one(T) + T(0.45) * sqrt(Z)) *
        (one(T) + T(0.4) * Ta^T(0.7)) /
        (one(T) + T(0.87) * We_g^T(1.67))^T(0.6)

    Ω_dim = (T(0.34) + T(0.38) * We_g^T(1.5)) /
        ((one(T) + Z) * (one(T) + T(1.4) * Ta^T(0.6)))
    Ω_KH = Ω_dim / sqrt(ρ_l * a^3 / σ)

    return Λ_KH, Ω_KH, We_g, Z, Ta
end

"""
    kh_act_breakup(d_parent, U_rel, ρ_g, ρ_l, μ_l, σ;
                   model = KHACTBreakup()) -> (d_child, breakup_time)

Convenience wrapper that returns the child-droplet diameter and
break-up time for a parent droplet of diameter `d_parent [m]` at
relative velocity `U_rel` ([`SVector`] or scalar magnitude).

Zero relative velocity ⇒ `(d_parent, Inf)` (no break-up).
"""
function kh_act_breakup(
        d_parent::T, U_rel_mag::T,
        ρ_g::T, ρ_l::T, μ_l::T, σ::T;
        model::KHACTBreakup{T} = KHACTBreakup{T}(T(0.61), T(1.73)),
    ) where {T <: Real}
    if U_rel_mag <= eps(T)
        return d_parent, T(Inf)
    end
    Λ_KH, Ω_KH, _, _, _ = kh_act_wavelength_growth(d_parent, U_rel_mag, ρ_g, ρ_l, μ_l, σ)
    if !(isfinite(Λ_KH) && Ω_KH > zero(T))
        return d_parent, T(Inf)
    end
    a = d_parent / 2
    r_child = model.B0 * Λ_KH
    # Clamp child radius to the parent radius (correlation diverges at
    # low We_g).
    r_child = min(r_child, a)
    τ_b = T(3.726) * model.B1 * a / (Λ_KH * Ω_KH)
    return 2 * r_child, τ_b
end

function kh_act_breakup(
        d_parent::T, U_rel::SVector{Dim, T},
        ρ_g::T, ρ_l::T, μ_l::T, σ::T;
        model::KHACTBreakup{T} = KHACTBreakup{T}(T(0.61), T(1.73)),
    ) where {Dim, T <: Real}
    return kh_act_breakup(d_parent, norm(U_rel), ρ_g, ρ_l, μ_l, σ; model = model)
end

# ═════════════════════════════════════════════════════════════════════════
# LISA primary break-up
# ═════════════════════════════════════════════════════════════════════════

"""
    LISABreakup{T} <: AbstractPrimaryBreakupModel

Linearised Instability Sheet Atomisation. Used for the primary
break-up of hollow-cone sprays from pressure-swirl atomisers, where
the liquid leaves the nozzle as a thin conical sheet. For a thin
inviscid sheet the dominant wavelength of the Squire (1953)
instability is

```
Λ_LISA = 2π · σ · (1 + We_g) / (ρ_g · U²)
```

with `We_g = ρ_g·U²·h/σ` and `h` the local sheet thickness. (See
Senecal *et al.* 1999 for the full derivation.)

# Fields
- `C_λ::T` — wavelength scaling factor (default `1.0`); multiplies `Λ_LISA`
"""
struct LISABreakup{T} <: AbstractPrimaryBreakupModel
    C_λ::T
end

LISABreakup(; C_λ::Real = 1.0) = LISABreakup{Float64}(Float64(C_λ))

"""
    lisa_wavelength(h, U, ρ_g, σ; model = LISABreakup()) -> Λ [m]

Return the LISA-predicted sheet-break-up wavelength.

# Arguments
- `h::T` — local sheet thickness [m]
- `U::T` — sheet velocity [m/s]
- `ρ_g::T` — gas density [kg/m³]
- `σ::T` — surface tension [N/m]
"""
function lisa_wavelength(
        h::T, U::T, ρ_g::T, σ::T;
        model::LISABreakup{T} = LISABreakup{T}(one(T)),
    ) where {T <: Real}
    if U <= eps(T) || ρ_g <= zero(T)
        return T(Inf)
    end
    We_g = ρ_g * U^2 * h / σ
    Λ = T(2) * T(pi) * σ * (one(T) + We_g) / (ρ_g * U^2)
    return model.C_λ * Λ
end

"""
    lisa_breakup(h, U, ρ_g, ρ_l, σ; model = LISABreakup()) -> (d_child, breakup_time)

Return the ligament-diameter and break-up time for a LISA sheet.
Following Senecal *et al.* (1999) the ligament diameter is taken as
`d_L = √(8 h / K)` with wavenumber `K = 2π/Λ`, and ligament break-up
obeys Weber's classical `τ = (ρ_l · d_L³ / σ)^(1/2)`.
"""
function lisa_breakup(
        h::T, U::T, ρ_g::T, ρ_l::T, σ::T;
        model::LISABreakup{T} = LISABreakup{T}(one(T)),
    ) where {T <: Real}
    Λ = lisa_wavelength(h, U, ρ_g, σ; model = model)
    if !isfinite(Λ)
        return T(Inf), T(Inf)
    end
    K = T(2) * T(pi) / Λ
    d_L = sqrt(T(8) * h / K)
    τ = sqrt(ρ_l * d_L^3 / σ)
    return d_L, τ
end
