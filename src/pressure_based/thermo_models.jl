# pressure_based/thermo_models.jl — Stage 3a thermo / equation-of-state models
#
# Unified hierarchy used by both incompressible (ρ = const) and compressible
# (ρ = ρ(p, T)) pressure-based solvers. Each concrete model exposes:
#
#   density_at(model, p, T)   → ρ  (cell-local evaluation)
#   viscosity_at(model, T)    → μ  (molecular viscosity; rheology applies a
#                                   strain-rate-dependent correction on top)
#   cp_at(model, T)           → specific heat at constant pressure
#   beta_at(model, T)         → thermal expansion coefficient (Boussinesq)
#
# Incompressible flows use `IncompressibleThermo(; rho, mu, cp, beta)` so the
# existing `IncompressibleProblem.nu` / `.density` fields can be preserved as
# a backward-compatible shim while the pressure-based stack is generalized.

using Printf

"""
    AbstractThermoModel

Stage 3a umbrella for thermo / equation-of-state models used by the
pressure-based solver family. Every concrete subtype must implement
`density_at`, `viscosity_at`, `cp_at`, and `beta_at` (see module docs).
"""
abstract type AbstractThermoModel end

"""
    IncompressibleThermo(; rho = 1.0, mu = 1.0e-3, cp = 1004.0, beta = 0.0) <: AbstractThermoModel

Constant-property incompressible thermo model. `rho` and `mu` are
independent of `p` and `T`; `beta` enables Boussinesq-style buoyancy
coupling when non-zero.
"""
struct IncompressibleThermo{T} <: AbstractThermoModel
    rho::T
    mu::T
    cp::T
    beta::T
end
IncompressibleThermo(; rho = 1.0, mu = 1.0e-3, cp = 1004.0, beta = 0.0) =
    IncompressibleThermo(promote(rho, mu, cp, beta)...)

density_at(m::IncompressibleThermo, p, T) = m.rho
viscosity_at(m::IncompressibleThermo, T) = m.mu
cp_at(m::IncompressibleThermo, T) = m.cp
beta_at(m::IncompressibleThermo, T) = m.beta

"""
    IdealGas(; gamma = 1.4, R = 287.05, mu = 1.8e-5, cp = 1004.0, beta = 0.0) <: AbstractThermoModel

Ideal-gas equation of state: `ρ = p / (R·T)`. Used by the compressible
pressure-based variant (Stage 3 follow-up for rhoSimpleFoam / rhoPimpleFoam
parity). `mu` is molecular viscosity at reference temperature; for variable
viscosity use `Sutherland` below.
"""
struct IdealGas{T} <: AbstractThermoModel
    gamma::T
    R::T
    mu::T
    cp::T
    beta::T
end
IdealGas(; gamma = 1.4, R = 287.05, mu = 1.8e-5, cp = 1004.0, beta = 0.0) =
    IdealGas(promote(gamma, R, mu, cp, beta)...)

density_at(m::IdealGas, p, T) = p / (m.R * max(T, eps(typeof(T))))
viscosity_at(m::IdealGas, T) = m.mu
cp_at(m::IdealGas, T) = m.cp
beta_at(m::IdealGas, T) = m.beta

"""
    BoussinesqThermo(; rho0 = 1.0, T0 = 300.0, mu = 1.0e-3, cp = 1004.0, beta = 3.33e-3) <: AbstractThermoModel

Boussinesq approximation: `ρ = ρ₀ · (1 - β·(T - T₀))`. Lightweight
buoyancy-coupled incompressible model — density varies *only* through the
momentum source, not through continuity.
"""
struct BoussinesqThermo{T} <: AbstractThermoModel
    rho0::T
    T0::T
    mu::T
    cp::T
    beta::T
end
BoussinesqThermo(; rho0 = 1.0, T0 = 300.0, mu = 1.0e-3, cp = 1004.0, beta = 3.33e-3) =
    BoussinesqThermo(promote(rho0, T0, mu, cp, beta)...)

density_at(m::BoussinesqThermo, p, T) = m.rho0 * (1 - m.beta * (T - m.T0))
viscosity_at(m::BoussinesqThermo, T) = m.mu
cp_at(m::BoussinesqThermo, T) = m.cp
beta_at(m::BoussinesqThermo, T) = m.beta

"""
    SutherlandViscosity(mu_ref, T_ref, S) -> Function

Return a viscosity closure implementing Sutherland's law:
`μ(T) = μ_ref · (T/T_ref)^(3/2) · (T_ref + S) / (T + S)`. Useful for
wrapping around an `IdealGas` when temperature-dependent viscosity is
needed.
"""
function SutherlandViscosity(mu_ref, T_ref, S)
    T = promote_type(typeof(mu_ref), typeof(T_ref), typeof(S))
    mu_ref_T = T(mu_ref); T_ref_T = T(T_ref); S_T = T(S)
    return Tval -> mu_ref_T * (Tval / T_ref_T)^(T(3) / T(2)) *
        (T_ref_T + S_T) / (max(Tval, eps(T)) + S_T)
end

"""
    SutherlandGas{F}(; gamma = 1.4, R = 287.05, mu_ref = 1.716e-5, T_ref = 273.15, S = 110.4, cp = 1004.0, beta = 0.0)

Ideal gas with Sutherland-law temperature-dependent viscosity; a concrete
`AbstractThermoModel` variant of `IdealGas` with a viscosity closure `F`.
"""
struct SutherlandGas{T, F} <: AbstractThermoModel
    gamma::T
    R::T
    mu_fun::F
    cp::T
    beta::T
end
function SutherlandGas(; gamma = 1.4, R = 287.05, mu_ref = 1.716e-5, T_ref = 273.15, S = 110.4, cp = 1004.0, beta = 0.0)
    T = promote_type(typeof(gamma), typeof(R), typeof(mu_ref), typeof(T_ref), typeof(S), typeof(cp), typeof(beta))
    mu_fun = SutherlandViscosity(mu_ref, T_ref, S)
    return SutherlandGas{T, typeof(mu_fun)}(T(gamma), T(R), mu_fun, T(cp), T(beta))
end

density_at(m::SutherlandGas, p, T) = p / (m.R * max(T, eps(typeof(T))))
viscosity_at(m::SutherlandGas, T) = m.mu_fun(T)
cp_at(m::SutherlandGas, T) = m.cp
beta_at(m::SutherlandGas, T) = m.beta

# ── is_compressible trait ────────────────────────────────────────────

"""
    is_compressible(model::AbstractThermoModel) -> Bool

Returns `true` if density depends on pressure (and temperature) in a way
that requires the continuity equation to be updated with a `∂ρ/∂t` term.
Incompressible and Boussinesq models return `false`; ideal-gas variants
return `true`.
"""
is_compressible(::IncompressibleThermo) = false
is_compressible(::BoussinesqThermo) = false
is_compressible(::IdealGas) = true
is_compressible(::SutherlandGas) = true
