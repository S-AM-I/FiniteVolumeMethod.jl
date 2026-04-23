# pressure_based/rheology.jl — Stage 3b non-Newtonian rheology models
#
# Stage 3b addition. Concrete rheologies compute an effective viscosity
# from the local strain-rate magnitude `gamma_dot = sqrt(2·S:S)`:
#
#   mu_eff(rheo, gamma_dot, T)
#
# The pressure-based momentum assembly already computes per-face viscosity
# via a `_face_viscosity` helper (in compressible mode) or passes a scalar
# (in the default incompressible path). This module lets callers plug any
# rheology into the existing Laplacian assembly: `viscosity_at(rheo, γ̇, T)`
# replaces the scalar `mu` wherever the assembly previously used it.

"""
    AbstractRheology

Stage 3b umbrella for rheology models. Concrete subtypes must implement
`viscosity_at(rheo, strain_rate, T)` returning the effective dynamic
viscosity at the given strain-rate magnitude and temperature.

All models currently return a strain-rate-only correction; temperature
variation is pass-through (the caller can apply a `AbstractThermoModel`'s
`viscosity_at(model, T)` separately and combine).
"""
abstract type AbstractRheology end

"""
    NewtonianRheology(mu) <: AbstractRheology

Constant Newtonian viscosity. Pass-through: `μ_eff = μ`.
"""
struct NewtonianRheology{T} <: AbstractRheology
    mu::T
end
NewtonianRheology(; mu = 1.0e-3) = NewtonianRheology(mu)

viscosity_at(r::NewtonianRheology, gamma_dot, T) = r.mu

"""
    PowerLawRheology(; K = 1.0e-3, n = 1.0, gamma_min = 1.0e-8, gamma_max = 1.0e8)

Power-law model: `μ = K · γ̇^(n-1)`. `n < 1` is shear-thinning,
`n > 1` shear-thickening. Strain-rate is clamped to
`[gamma_min, gamma_max]` to avoid singular behaviour near zero shear.
"""
struct PowerLawRheology{T} <: AbstractRheology
    K::T
    n::T
    gamma_min::T
    gamma_max::T
end
function PowerLawRheology(; K = 1.0e-3, n = 1.0, gamma_min = 1.0e-8, gamma_max = 1.0e8)
    return PowerLawRheology(promote(K, n, gamma_min, gamma_max)...)
end

function viscosity_at(r::PowerLawRheology, gamma_dot, T)
    g = clamp(gamma_dot, r.gamma_min, r.gamma_max)
    return r.K * g^(r.n - 1)
end

"""
    BirdCarreauRheology(; mu_0 = 1.0, mu_inf = 1.0e-3, lambda = 1.0, n = 0.5)

Bird-Carreau model: `μ = μ∞ + (μ₀ - μ∞) · (1 + (λ γ̇)²)^((n-1)/2)`.
Smooth blend between zero-shear viscosity `μ₀` and infinite-shear
viscosity `μ∞` with relaxation time `λ` and power index `n`.
"""
struct BirdCarreauRheology{T} <: AbstractRheology
    mu_0::T
    mu_inf::T
    lambda::T
    n::T
end
function BirdCarreauRheology(; mu_0 = 1.0, mu_inf = 1.0e-3, lambda = 1.0, n = 0.5)
    return BirdCarreauRheology(promote(mu_0, mu_inf, lambda, n)...)
end

function viscosity_at(r::BirdCarreauRheology, gamma_dot, T)
    return r.mu_inf + (r.mu_0 - r.mu_inf) *
        (1 + (r.lambda * gamma_dot)^2)^((r.n - 1) / 2)
end

"""
    HerschelBulkleyRheology(; tau_y = 0.0, K = 1.0e-3, n = 1.0, gamma_c = 1.0e-6)

Herschel-Bulkley model: yield-stress + power-law behaviour. Below the
critical strain rate `gamma_c`, the model is regularised with a
bi-viscous linearisation to avoid infinite apparent viscosity:

    μ = K · γ̇^(n-1) + τ_y / max(γ̇, γ_c)

For `τ_y = 0` this reduces to the power-law model; for `n = 1` to a
Bingham plastic.
"""
struct HerschelBulkleyRheology{T} <: AbstractRheology
    tau_y::T
    K::T
    n::T
    gamma_c::T
end
function HerschelBulkleyRheology(; tau_y = 0.0, K = 1.0e-3, n = 1.0, gamma_c = 1.0e-6)
    return HerschelBulkleyRheology(promote(tau_y, K, n, gamma_c)...)
end

function viscosity_at(r::HerschelBulkleyRheology, gamma_dot, T)
    g_eff = max(gamma_dot, r.gamma_c)
    return r.K * g_eff^(r.n - 1) + r.tau_y / g_eff
end

"""
    CassonRheology(; tau_y = 0.0, mu_inf = 1.0e-3, gamma_c = 1.0e-6)

Casson model, common for blood rheology:
`√μ = √μ∞ + √(τ_y / γ̇)`, regularised by `gamma_c` as above.
"""
struct CassonRheology{T} <: AbstractRheology
    tau_y::T
    mu_inf::T
    gamma_c::T
end
CassonRheology(; tau_y = 0.0, mu_inf = 1.0e-3, gamma_c = 1.0e-6) =
    CassonRheology(promote(tau_y, mu_inf, gamma_c)...)

function viscosity_at(r::CassonRheology, gamma_dot, T)
    g_eff = max(gamma_dot, r.gamma_c)
    sqrt_mu = sqrt(r.mu_inf) + sqrt(r.tau_y / g_eff)
    return sqrt_mu^2
end
