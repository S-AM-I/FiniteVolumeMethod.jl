# ============================================================
# IMEX Runge-Kutta Tableau Types
# ============================================================
#
# Implicit-Explicit (IMEX) Runge-Kutta methods for systems of the form:
#   dU/dt = F_explicit(U) + F_implicit(U)
#
# The explicit part (hyperbolic fluxes) is integrated with an explicit
# RK tableau, and the implicit part (stiff sources) with a diagonally
# implicit RK (DIRK) tableau.
#
# Each scheme is defined by its Butcher tableaux:
#   Explicit: (A_ex, b_ex, c_ex)
#   Implicit: (A_im, b_im, c_im)

"""
    AbstractIMEXScheme

Abstract supertype for IMEX Runge-Kutta time integration schemes.

All subtypes should have a corresponding `imex_tableau` method returning
the Butcher tableaux.
"""
abstract type AbstractIMEXScheme end

"""
    imex_tableau(scheme::AbstractIMEXScheme) -> NamedTuple

Return the Butcher tableaux for the IMEX scheme as a named tuple:
  `(A_ex, b_ex, c_ex, A_im, b_im, c_im, s)`

where `s` is the number of stages, and:
- `A_ex`: Explicit RK matrix (s x s, strictly lower triangular).
- `b_ex`: Explicit weights (length s).
- `c_ex`: Explicit abscissae (length s).
- `A_im`: Implicit RK matrix (s x s, lower triangular with non-zero diagonal).
- `b_im`: Implicit weights (length s).
- `c_im`: Implicit abscissae (length s).
"""
function imex_tableau end

"""
    imex_nstages(scheme::AbstractIMEXScheme) -> Int

Return the number of stages for the IMEX scheme.
"""
function imex_nstages end

# ============================================================
# IMEX SSP3(4,3,3) — Pareschi & Russo (2005)
# ============================================================

"""
    IMEX_SSP3_433 <: AbstractIMEXScheme

Third-order IMEX scheme with 4 explicit stages and 3 implicit stages,
based on the SSP3(4,3,3) method of Pareschi & Russo (2005).

This is a good default for moderately stiff problems. The explicit
part is a 4-stage, 3rd-order SSP scheme. The implicit part is a
3rd-order L-stable DIRK scheme.
"""
struct IMEX_SSP3_433 <: AbstractIMEXScheme end

imex_nstages(::IMEX_SSP3_433) = 4

function imex_tableau(::IMEX_SSP3_433)
    # Explicit tableau (4 stages, 3rd order SSP)
    # The SSP3(4,3,3) explicit part:
    #   Stage 1: U^(1) = U^n
    #   Stage 2: U^(2) = U^n + 0.5*dt*F(U^(1))
    #   Stage 3: U^(3) = U^n + 0.5*dt*F(U^(1)) + 0.5*dt*F(U^(2))
    #   Stage 4: U^(4) = U^n + dt/6*(F1 + F2 + F3) combined
    #   Final:   U^{n+1} = combination

    α = 0.24169426078821
    β = 0.06042356519705
    η = 0.1291528696059

    A_ex = (
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0, 0.0),
        (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.0),
    )
    b_ex = (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.5)
    c_ex = (0.0, 0.5, 1.0, 0.5)

    # Implicit tableau (4 stages, L-stable DIRK)
    # Diagonal elements are all equal to α for L-stability
    A_im = (
        (α, 0.0, 0.0, 0.0),
        (-α, α, 0.0, 0.0),
        (0.0, 1.0 - α, α, 0.0),
        (β, η, 1.0 - α - β - η, α),
    )
    b_im = (β, η, 1.0 - α - β - η, α)
    c_im = (α, 0.0, 1.0, 1.0)

    return (
        A_ex = A_ex, b_ex = b_ex, c_ex = c_ex,
        A_im = A_im, b_im = b_im, c_im = c_im,
        s = 4,
    )
end

# ============================================================
# IMEX ARS(2,2,2) — Ascher, Ruuth, Spiteri (1997)
# ============================================================

"""
    IMEX_ARS222 <: AbstractIMEXScheme

Second-order IMEX scheme with 2 implicit stages and 2 explicit stages,
from Ascher, Ruuth & Spiteri (1997).

A simple, robust choice for mildly stiff problems.
"""
struct IMEX_ARS222 <: AbstractIMEXScheme end

imex_nstages(::IMEX_ARS222) = 3

function imex_tableau(::IMEX_ARS222)
    # ARS(2,2,2) method (also called ARS-222 or IMEX-SSP2(2,2,2))
    #
    # The ARS method has s=3 stages (the first is trivial).

    γ = 1.0 - 1.0 / sqrt(2.0)   # ≈ 0.29289...
    δ = 1.0 - 1.0 / (2.0 * γ)   # ≈ -0.70711...

    # Explicit tableau (first stage is trivial)
    A_ex = (
        (0.0, 0.0, 0.0),
        (γ, 0.0, 0.0),
        (δ, 1.0 - δ, 0.0),
    )
    b_ex = (0.0, 1.0 - γ, γ)
    c_ex = (0.0, γ, 1.0)

    # Implicit tableau
    A_im = (
        (0.0, 0.0, 0.0),
        (0.0, γ, 0.0),
        (0.0, 1.0 - γ, γ),
    )
    b_im = (0.0, 1.0 - γ, γ)
    c_im = (0.0, γ, 1.0)

    return (
        A_ex = A_ex, b_ex = b_ex, c_ex = c_ex,
        A_im = A_im, b_im = b_im, c_im = c_im,
        s = 3,
    )
end

# ============================================================
# IMEX Midpoint — first-order, for testing
# ============================================================

"""
    IMEX_Midpoint <: AbstractIMEXScheme

First-order IMEX method based on the implicit-explicit midpoint rule.
Primarily useful for testing the IMEX infrastructure.

2 stages:
- Stage 1: explicit evaluation at t^n
- Stage 2: implicit solve at t^{n+1}
"""
struct IMEX_Midpoint <: AbstractIMEXScheme end

imex_nstages(::IMEX_Midpoint) = 2

function imex_tableau(::IMEX_Midpoint)
    A_ex = (
        (0.0, 0.0),
        (1.0, 0.0),
    )
    b_ex = (0.5, 0.5)
    c_ex = (0.0, 1.0)

    A_im = (
        (0.0, 0.0),
        (0.0, 1.0),
    )
    b_im = (0.0, 1.0)
    c_im = (0.0, 1.0)

    return (
        A_ex = A_ex, b_ex = b_ex, c_ex = c_ex,
        A_im = A_im, b_im = b_im, c_im = c_im,
        s = 2,
    )
end
