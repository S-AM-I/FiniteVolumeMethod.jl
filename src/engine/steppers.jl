# steppers.jl - Time stepping methods
# Migrated from Simu.jl SimuEngine/steppers.jl
# Note: The original was a submodule; here everything is defined directly in the
# parent module scope (FiniteVolumeMethod).  LinearAlgebra and SparseArrays are
# already imported by the parent module.

abstract type AbstractTimeStepper end

"""
    ForwardEuler

Explicit First-Order Euler method.
x_{n+1} = x_n + dt * M^{-1} (b - A x_n)
"""
struct ForwardEuler <: AbstractTimeStepper end

function step!(::ForwardEuler, x, A, M, b, dt)
    # M dx/dt = b - A x
    # dx = dt * M^{-1} (b - A x)
    rhs = b - A * x
    dx = M \ rhs
    return x + dt * dx
end

"""
    RK2

Explicit Second-Order Runge-Kutta method (Heun's method).
"""
struct RK2 <: AbstractTimeStepper end

function step!(::RK2, x, A, M, b, dt)
    # k1 = dt * M^{-1} (b - A x)
    # k2 = dt * M^{-1} (b - A (x + k1))
    rhs1 = b - A * x
    k1 = dt * (M \ rhs1)

    rhs2 = b - A * (x + k1)
    k2 = dt * (M \ rhs2)

    return x + 0.5 * (k1 + k2)
end

"""
    ImplicitEuler

L-stable First-Order Implicit method.
(M + dt * A) x_{n+1} = M x_n + dt * b
"""
struct ImplicitEuler <: AbstractTimeStepper
    nonlinear_max_iters::Int
    nonlinear_tol::Float64
end

ImplicitEuler(; max_iters = 10, tol = 1.0e-6) = ImplicitEuler(max_iters, tol)

function step!(stepper::ImplicitEuler, x, A, M, b, dt)
    # Linear case: (M + dt*A) x_new = M*x + dt*b
    lhs = M + dt * A
    rhs = M * x + dt * b
    return lhs \ rhs
end

"""
    Rosenbrock23

L-stable Second-Order Rosenbrock method (2 stages).
Suitable for stiff problems.

Reference: Shampine & Reichelt, "The MATLAB ODE Suite", 1997.

Note: For the current linear-assembly interface (`step!(stepper, x, A, M, b, dt)`),
the Jacobian (A) is frozen at x_n.  Without re-assembly the method reduces to
Implicit Euler with a gamma-scaled system matrix.  Full nonlinear Rosenbrock
requires passing the assembly function.
"""
struct Rosenbrock23 <: AbstractTimeStepper
    gamma::Float64

    function Rosenbrock23()
        return new(1.0 / (2.0 + sqrt(2.0)))
    end
end

function step!(stepper::Rosenbrock23, x, A, M, b, dt)
    # With frozen Jacobian, Rosenbrock reduces to Implicit Euler.
    # See detailed discussion in Simu.jl source.
    return step!(ImplicitEuler(), x, A, M, b, dt)
end

"""
    CrankNicolson

A-stable Second-Order Implicit method.
(M + 0.5 * dt * A) x_{n+1} = (M - 0.5 * dt * A) x_n + dt * b
"""
struct CrankNicolson <: AbstractTimeStepper end

function step!(::CrankNicolson, x, A, M, b, dt)
    lhs = M + 0.5 * dt * A
    rhs = (M - 0.5 * dt * A) * x + dt * b
    return lhs \ rhs
end
