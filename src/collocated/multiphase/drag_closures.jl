# multiphase/drag_closures.jl — Interphase drag closures for Eulerian two-fluid
#
# Closed-form algebraic drag laws used by the Eulerian two-fluid model.
# Only primitive (algebraic) operations — no mesh, no field, no solver
# state. Keeps these laws unit-testable against textbook correlations
# without requiring the full coupled solver.
#
# Implemented correlations:
#   * `IshiiZuberDrag` — Ishii-Zuber bubbly-flow drag (Re_b ∈ [1, 1000]).
#   * `GibilaroDrag`  — Gibilaro-like cluster-corrected drag used as
#     an alternate closure for denser dispersed phases.
#
# Experimental — deferred to v3.1 production-hardening.

"""
    AbstractDragClosure

Root type for interphase drag correlations used by the Eulerian
two-fluid model. Concrete subtypes implement `drag_coefficient` and
`drag_force_density`.
"""
abstract type AbstractDragClosure end

"""
    IshiiZuberDrag

Ishii-Zuber drag closure for bubbly dispersed flow.

The drag coefficient follows the Ishii-Zuber correlation for the
`Re_b ∈ [1, 1000]` regime:

```
C_D = (24 / Re_b) · (1 + 0.1 · Re_b^0.75) · (1 − α_g)^(−1.5)
```

Intended for gas-in-liquid bubbly flows. The `(1 − α_g)^(−1.5)`
cluster correction enhances drag as the gas holdup rises.
"""
struct IshiiZuberDrag <: AbstractDragClosure end

"""
    GibilaroDrag

Gibilaro-style cluster-corrected drag closure for denser dispersed
phases. Uses the same single-bubble drag but a stronger cluster
correction exponent:

```
C_D = (24 / Re_b) · (1 + 0.1 · Re_b^0.75) · (1 − α_g)^(−2.65)
```

The `(1 − α_g)^(−2.65)` exponent mirrors the Richardson-Zaki /
Gibilaro fluidised-bed correlation for denser dispersed clusters.
"""
struct GibilaroDrag <: AbstractDragClosure end

"""
    bubble_reynolds(rho_l, d_b, U_rel, mu_l)

Return the bubble Reynolds number
`Re_b = ρ_l · d_b · |U_rel| / μ_l`. `U_rel` may be a scalar or a
`StaticArrays.SVector`.
"""
function bubble_reynolds(rho_l::T, d_b::T, U_rel, mu_l::T) where {T}
    d_b > zero(T) || throw(ArgumentError("bubble diameter must be positive"))
    mu_l > zero(T) || throw(ArgumentError("liquid viscosity must be positive"))
    slip = _norm(U_rel)
    return rho_l * d_b * slip / mu_l
end

# Local scalar norm that works for Real or SVector without dragging
# LinearAlgebra into this file.
_norm(x::Real) = abs(x)
_norm(x::AbstractVector) = sqrt(sum(xi -> xi * xi, x))

"""
    drag_coefficient(closure, Re_b, alpha_g)

Return the dimensionless drag coefficient `C_D` for the given closure,
bubble Reynolds number, and gas volume fraction `α_g ∈ [0, 1)`.

A small `Re_b` floor (`eps`) is used to keep `24 / Re_b` finite in
the `Re_b → 0` limit; the Stokes-like behaviour is preserved.
"""
function drag_coefficient(::IshiiZuberDrag, Re_b::T, alpha_g::T) where {T}
    zero(T) <= alpha_g < one(T) || throw(ArgumentError("alpha_g must be in [0, 1)"))
    Re = max(Re_b, eps(T))
    stokes_part = T(24) / Re
    inertial_part = one(T) + T(0.1) * Re^T(0.75)
    cluster_correction = (one(T) - alpha_g)^T(-1.5)
    return stokes_part * inertial_part * cluster_correction
end

function drag_coefficient(::GibilaroDrag, Re_b::T, alpha_g::T) where {T}
    zero(T) <= alpha_g < one(T) || throw(ArgumentError("alpha_g must be in [0, 1)"))
    Re = max(Re_b, eps(T))
    stokes_part = T(24) / Re
    inertial_part = one(T) + T(0.1) * Re^T(0.75)
    cluster_correction = (one(T) - alpha_g)^T(-2.65)
    return stokes_part * inertial_part * cluster_correction
end

"""
    drag_force_density(closure, rho_l, U_rel, alpha_g, d_b, mu_l)

Compute the interphase drag force per unit mixture volume,

```
F_D = (3 / 4) · C_D · ρ_l · α_g · |U_rel| · U_rel / d_b
```

where `U_rel = U_g − U_l`. Returns zero (element-wise) when either
`α_g = 0` (no dispersed phase) or `|U_rel| = 0` (no slip), bypassing
the `24 / Re_b` singular limit cleanly.

`U_rel` may be a scalar, a `Tuple`, or an `SVector`. The return type
matches `U_rel`.
"""
function drag_force_density(
        closure::AbstractDragClosure,
        rho_l::T, U_rel, alpha_g::T, d_b::T, mu_l::T,
    ) where {T}
    d_b > zero(T) || throw(ArgumentError("bubble diameter must be positive"))
    zero(T) <= alpha_g < one(T) || throw(ArgumentError("alpha_g must be in [0, 1)"))

    if alpha_g == zero(T) || _is_zero_slip(U_rel)
        return _zero_like(U_rel, T)
    end

    slip = _norm(U_rel)
    Re_b = rho_l * d_b * slip / mu_l
    C_D = drag_coefficient(closure, Re_b, alpha_g)
    prefactor = T(0.75) * C_D * rho_l * alpha_g * slip / d_b
    return prefactor * U_rel
end

_is_zero_slip(x::Real) = iszero(x)
_is_zero_slip(x::AbstractVector) = all(iszero, x)

_zero_like(x::Real, ::Type{T}) where {T} = zero(T)
_zero_like(x::AbstractVector, ::Type{T}) where {T} = zero(x)

"""
    stokes_limit_drag(rho_l, U_rel, alpha_g, d_b, mu_l)

Analytical Stokes-only drag (no inertial correction, no cluster
correction). Returned form: `F = 18 μ_l α_g U_rel / d_b²`. Used
purely as a V&V reference for the `Re_b → 0` limit of
[`drag_force_density`](@ref).
"""
function stokes_limit_drag(rho_l::T, U_rel, alpha_g::T, d_b::T, mu_l::T) where {T}
    d_b > zero(T) || throw(ArgumentError("bubble diameter must be positive"))
    return T(18) * mu_l * alpha_g * U_rel / (d_b * d_b)
end
