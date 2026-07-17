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

# ── Specific heat trait (alias) ──────────────────────────────────────

"""
    specific_heat(model::AbstractThermoModel, T) -> T

Alias for [`cp_at`](@ref). Returns specific heat at constant pressure.
Provided so `density`, `viscosity`, `specific_heat` can be used as a
uniform OpenFOAM-style trio.
"""
specific_heat(m::AbstractThermoModel, T) = cp_at(m, T)

"""
    density(model::AbstractThermoModel, p, T) -> T

Alias for [`density_at`](@ref). Provided for OpenFOAM-style lookup.
"""
density(m::AbstractThermoModel, p, T) = density_at(m, p, T)

"""
    viscosity(model::AbstractThermoModel, T) -> T

Alias for [`viscosity_at`](@ref). Provided for OpenFOAM-style lookup.
"""
viscosity(m::AbstractThermoModel, T) = viscosity_at(m, T)

# ── Sutherland (standalone thermo wrapper) ──────────────────────────

"""
    Sutherland(; gamma = 1.4, R = 287.05, mu_ref = 1.716e-5, T_ref = 273.15,
                 S = 110.4, cp = 1004.0, beta = 0.0) <: AbstractThermoModel

Alias constructor for [`SutherlandGas`](@ref): ideal-gas equation of
state with Sutherland-law viscosity. Added in Stage 3 compressible
extension so the OpenFOAM-style `Sutherland` name is available
alongside `SutherlandGas`.
"""
Sutherland(; kwargs...) = SutherlandGas(; kwargs...)

# ── Peng-Robinson EOS ───────────────────────────────────────────────

"""
    PengRobinson{T}(; Tc, pc, omega, R = 8.3144621, M = 0.02897, mu = 1.8e-5,
                     cp = 1004.0, beta = 0.0) <: AbstractThermoModel

Peng-Robinson cubic equation of state: departs from ideal gas by
accounting for molecular attraction and finite molecular volume. Used
for real-gas compressibility corrections (natural-gas pipelines,
supercritical CO₂, etc.).

Solves the cubic form
```
    p = R_s·T / (v - b) - a·α(T) / (v² + 2·b·v - b²)
```
where
```
    R_s     = R / M             (specific gas constant)
    a       = 0.45724 · R_s² · Tc² / pc
    b       = 0.07780 · R_s · Tc / pc
    κ       = 0.37464 + 1.54226·ω - 0.26992·ω²
    α(T)    = (1 + κ·(1 - sqrt(T/Tc)))²
```

Density is computed by solving the cubic for molar volume `v` (gas
root) at each `(p, T)`. In the low-density (or `a = 0`) limit this
reduces to the ideal gas law `ρ = p / (R_s·T)`.

# Fields
- `Tc::T`    — critical temperature [K]
- `pc::T`    — critical pressure [Pa]
- `omega::T` — acentric factor [-]
- `R::T`     — universal gas constant [J/(mol·K)] (default 8.3144621)
- `M::T`     — molar mass [kg/mol] (default air = 0.02897)
- `mu::T`    — reference dynamic viscosity [Pa·s]
- `cp::T`    — specific heat at constant pressure [J/(kg·K)]
- `beta::T`  — thermal expansion coefficient [1/K]
"""
struct PengRobinson{T} <: AbstractThermoModel
    Tc::T
    pc::T
    omega::T
    R::T
    M::T
    mu::T
    cp::T
    beta::T
end

function PengRobinson(;
        Tc = 304.13, pc = 7.3773e6, omega = 0.22394,
        R = 8.3144621, M = 0.02897,
        mu = 1.8e-5, cp = 1004.0, beta = 0.0,
    )
    return PengRobinson(promote(Tc, pc, omega, R, M, mu, cp, beta)...)
end

"""
    _pr_params(m::PengRobinson, T)

Compute `(a_alpha, b, R_s)` — attraction, co-volume, and specific gas
constant — at temperature `T`.
"""
@inline function _pr_params(m::PengRobinson{T}, T_val) where {T}
    R_s = m.R / m.M
    a = T(0.45724) * R_s^2 * m.Tc^2 / m.pc
    b = T(0.0778) * R_s * m.Tc / m.pc
    kappa = T(0.37464) + T(1.54226) * m.omega - T(0.26992) * m.omega^2
    alpha = (one(T) + kappa * (one(T) - sqrt(max(T_val, eps(T)) / m.Tc)))^2
    return a * alpha, b, R_s
end

"""
    density_at(m::PengRobinson, p, T) -> T

Solve the Peng-Robinson cubic for molar volume (gas root) and return
mass density `ρ = M / v`. Uses Newton iteration starting from the
ideal-gas estimate `v₀ = R_s·T / p`; converges in ≤10 iterations for
typical supercritical states.
"""
function density_at(m::PengRobinson{T}, p, T_val) where {T}
    a_alpha, b, R_s = _pr_params(m, T_val)
    # Ideal-gas initial guess (specific volume, m³/kg).
    v = T(R_s * T_val / max(p, eps(T)))
    # Newton iteration on f(v) = p - R_s·T/(v - b_s) + a_α·/(v² + 2·b_s·v - b_s²)
    # where b_s, a_α_s are mass-specific.
    b_s = b / m.M
    a_s = a_alpha / (m.M^2)
    for _ in 1:50
        denom1 = v - b_s
        denom2 = v * v + T(2) * b_s * v - b_s * b_s
        f = p - R_s * T_val / denom1 + a_s / denom2
        df = R_s * T_val / (denom1 * denom1) - a_s * (T(2) * v + T(2) * b_s) / (denom2 * denom2)
        abs(df) < eps(T) && break
        dv = f / df
        v_new = v - dv
        v_new = max(v_new, b_s * T(1.0001))   # stay above co-volume
        abs(dv) < T(1.0e-12) * abs(v_new) && (v = v_new; break)
        v = v_new
    end
    return one(T) / v
end

viscosity_at(m::PengRobinson, T) = m.mu
cp_at(m::PengRobinson, T) = m.cp
beta_at(m::PengRobinson, T) = m.beta

# ── Redlich-Kwong EOS ───────────────────────────────────────────────

"""
    RedlichKwong{T}(; Tc, pc, R = 8.3144621, M = 0.02897, mu = 1.8e-5,
                    cp = 1004.0, beta = 0.0) <: AbstractThermoModel

Redlich-Kwong cubic equation of state. Two-parameter real-gas model:
```
    p = R_s·T / (v - b) - a / (sqrt(T) · v · (v + b))
```
with
```
    a = 0.42748 · R_s² · Tc^(5/2) / pc
    b = 0.08664 · R_s · Tc / pc
```

Simpler than Peng-Robinson (no acentric factor), used for warm-gas
and modest-pressure applications. Reduces to ideal gas as `a → 0`.

# Fields
- `Tc::T`   — critical temperature [K]
- `pc::T`   — critical pressure [Pa]
- `R::T`    — universal gas constant [J/(mol·K)] (default 8.3144621)
- `M::T`    — molar mass [kg/mol]
- `mu::T`   — dynamic viscosity [Pa·s]
- `cp::T`   — specific heat at constant pressure
- `beta::T` — thermal expansion coefficient
"""
struct RedlichKwong{T} <: AbstractThermoModel
    Tc::T
    pc::T
    R::T
    M::T
    mu::T
    cp::T
    beta::T
end

function RedlichKwong(;
        Tc = 304.13, pc = 7.3773e6,
        R = 8.3144621, M = 0.02897,
        mu = 1.8e-5, cp = 1004.0, beta = 0.0,
    )
    return RedlichKwong(promote(Tc, pc, R, M, mu, cp, beta)...)
end

"""
    density_at(m::RedlichKwong, p, T) -> T

Newton iteration on the Redlich-Kwong cubic; see [`PengRobinson`](@ref)
for the general scheme. Returns mass density `ρ = M / v`.
"""
function density_at(m::RedlichKwong{T}, p, T_val) where {T}
    R_s = m.R / m.M
    a = T(0.42748) * R_s^2 * m.Tc^(T(5) / T(2)) / m.pc
    b = T(0.08664) * R_s * m.Tc / m.pc
    b_s = b / m.M
    a_s = a / (m.M^2)
    v = T(R_s * T_val / max(p, eps(T)))
    sqrt_T = sqrt(max(T_val, eps(T)))
    for _ in 1:50
        denom1 = v - b_s
        denom2 = v * (v + b_s)
        f = p - R_s * T_val / denom1 + a_s / (sqrt_T * denom2)
        df = R_s * T_val / (denom1 * denom1) -
            a_s / sqrt_T * (T(2) * v + b_s) / (denom2 * denom2)
        abs(df) < eps(T) && break
        dv = f / df
        v_new = v - dv
        v_new = max(v_new, b_s * T(1.0001))
        abs(dv) < T(1.0e-12) * abs(v_new) && (v = v_new; break)
        v = v_new
    end
    return one(T) / v
end

viscosity_at(m::RedlichKwong, T) = m.mu
cp_at(m::RedlichKwong, T) = m.cp
beta_at(m::RedlichKwong, T) = m.beta

# ── Tabulated properties ────────────────────────────────────────────

"""
    TabulatedProperties{T}(T_table, rho_table, mu_table, cp_table;
                            pref = 1.01325e5, R = 287.05,
                            beta = 0.0) <: AbstractThermoModel

Look-up-table thermo model. `T_table` must be sorted ascending. Values
are linearly interpolated; out-of-range queries are clamped to the
table endpoints.

Density is treated as `rho(T) · p / pref` — a pressure-linear extension
of the tabulated isobaric values, exposing compressibility through
pressure at constant temperature. For a pure incompressible lookup
keep `pref = 0` to disable the pressure scaling (caller must ensure
input `p` matches `pref`).

# Fields
- `T_table::Vector{T}`  — reference temperatures [K]
- `rho_table::Vector{T}` — density at `pref` [kg/m³]
- `mu_table::Vector{T}` — dynamic viscosity [Pa·s]
- `cp_table::Vector{T}` — specific heat [J/(kg·K)]
- `pref::T`  — reference pressure [Pa]
- `R::T`     — gas constant (for low-density extrapolation) [J/(kg·K)]
- `beta::T`  — thermal expansion coefficient
"""
struct TabulatedProperties{T} <: AbstractThermoModel
    T_table::Vector{T}
    rho_table::Vector{T}
    mu_table::Vector{T}
    cp_table::Vector{T}
    pref::T
    R::T
    beta::T
end

function TabulatedProperties(
        T_table::Vector, rho_table::Vector,
        mu_table::Vector, cp_table::Vector;
        pref = 1.01325e5, R = 287.05, beta = 0.0,
    )
    length(T_table) == length(rho_table) == length(mu_table) == length(cp_table) ||
        error("TabulatedProperties: all tables must have identical length")
    issorted(T_table) || error("TabulatedProperties: T_table must be sorted ascending")
    V = promote_type(
        eltype(T_table), eltype(rho_table),
        eltype(mu_table), eltype(cp_table),
        typeof(pref), typeof(R), typeof(beta),
    )
    return TabulatedProperties{V}(
        Vector{V}(T_table), Vector{V}(rho_table),
        Vector{V}(mu_table), Vector{V}(cp_table),
        V(pref), V(R), V(beta),
    )
end

"""
    _interp_clamped(xs, ys, x)

Linear interpolation on sorted `xs`; out-of-range queries are clamped.
"""
@inline function _interp_clamped(xs::Vector{T}, ys::Vector{T}, x) where {T}
    n = length(xs)
    n == 1 && return ys[1]
    x <= xs[1] && return ys[1]
    x >= xs[n] && return ys[n]
    # binary search for the interval
    lo = 1
    hi = n
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        if xs[mid] > x
            hi = mid
        else
            lo = mid
        end
    end
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])
end

function density_at(m::TabulatedProperties{T}, p, T_val) where {T}
    rho_ref = _interp_clamped(m.T_table, m.rho_table, T(T_val))
    m.pref > zero(T) || return rho_ref
    return rho_ref * (T(p) / m.pref)
end
viscosity_at(m::TabulatedProperties, T_val) = _interp_clamped(m.T_table, m.mu_table, T_val)
cp_at(m::TabulatedProperties, T_val) = _interp_clamped(m.T_table, m.cp_table, T_val)
beta_at(m::TabulatedProperties, T_val) = m.beta

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
is_compressible(::PengRobinson) = true
is_compressible(::RedlichKwong) = true
is_compressible(m::TabulatedProperties) = m.pref > zero(typeof(m.pref))
