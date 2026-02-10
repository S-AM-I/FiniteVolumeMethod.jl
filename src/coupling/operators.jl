# ============================================================
# Concrete Operators for Operator Splitting
# ============================================================
#
# HyperbolicOperator: wraps a HyperbolicProblem (1D or 2D) and
#   advances the state via one SSP-RK3 time step.
#
# SourceOperator: wraps an AbstractStiffSource and advances the
#   state via cell-by-cell implicit Newton solves.

# ============================================================
# HyperbolicOperator
# ============================================================

"""
    HyperbolicOperator{P} <: AbstractOperator

Wraps a `HyperbolicProblem` or `HyperbolicProblem2D` as an operator
for use in operator splitting. The `advance!` method performs one
SSP-RK3 time step of the hyperbolic solver.

# Fields
- `problem::P`: The underlying hyperbolic problem (provides law,
  mesh, Riemann solver, reconstruction, BCs, and CFL number).
"""
struct HyperbolicOperator{P} <: AbstractOperator
    problem::P
end

# ---------- 1D advance! ----------

function advance!(
        U::AbstractVector{<:SVector}, op::HyperbolicOperator{<:HyperbolicProblem},
        dt, t, workspace
    )
    prob = op.problem
    nc = ncells(prob.mesh)
    dU = workspace.dU_vec
    U1 = workspace.U1_vec
    U2 = workspace.U2_vec

    # SSP-RK3 stage 1: U1 = U + dt * L(U)
    hyperbolic_rhs!(dU, U, prob, t)
    for i in 3:(nc + 2)
        U1[i] = U[i] + dt * dU[i]
    end

    # SSP-RK3 stage 2: U2 = 3/4 U + 1/4 (U1 + dt * L(U1))
    hyperbolic_rhs!(dU, U1, prob, t + dt)
    for i in 3:(nc + 2)
        U2[i] = 0.75 * U[i] + 0.25 * (U1[i] + dt * dU[i])
    end

    # SSP-RK3 stage 3: U = 1/3 U + 2/3 (U2 + dt * L(U2))
    hyperbolic_rhs!(dU, U2, prob, t + 0.5 * dt)
    for i in 3:(nc + 2)
        U[i] = (1.0 / 3.0) * U[i] + (2.0 / 3.0) * (U2[i] + dt * dU[i])
    end

    return nothing
end

# ---------- 2D advance! ----------

function advance!(
        U::AbstractMatrix{<:SVector}, op::HyperbolicOperator{<:HyperbolicProblem2D},
        dt, t, workspace
    )
    prob = op.problem
    nx, ny = prob.mesh.nx, prob.mesh.ny
    dU = workspace.dU_mat
    U1 = workspace.U1_mat
    U2 = workspace.U2_mat

    # SSP-RK3 stage 1
    hyperbolic_rhs_2d!(dU, U, prob, t)
    for iy in 1:ny, ix in 1:nx
        ii, jj = ix + 2, iy + 2
        U1[ii, jj] = U[ii, jj] + dt * dU[ii, jj]
    end

    # SSP-RK3 stage 2
    hyperbolic_rhs_2d!(dU, U1, prob, t + dt)
    for iy in 1:ny, ix in 1:nx
        ii, jj = ix + 2, iy + 2
        U2[ii, jj] = 0.75 * U[ii, jj] + 0.25 * (U1[ii, jj] + dt * dU[ii, jj])
    end

    # SSP-RK3 stage 3
    hyperbolic_rhs_2d!(dU, U2, prob, t + 0.5 * dt)
    for iy in 1:ny, ix in 1:nx
        ii, jj = ix + 2, iy + 2
        U[ii, jj] = (1.0 / 3.0) * U[ii, jj] + (2.0 / 3.0) * (U2[ii, jj] + dt * dU[ii, jj])
    end

    return nothing
end

# ---------- CFL time step ----------

function compute_operator_dt(op::HyperbolicOperator{<:HyperbolicProblem}, U::AbstractVector, t)
    prob = op.problem
    law = prob.law
    mesh = prob.mesh
    nc = ncells(mesh)

    λ_max = zero(mesh.dx)
    for i in 1:nc
        w = conserved_to_primitive(law, U[i + 2])
        λ = max_wave_speed(law, w, 1)
        λ_max = max(λ_max, λ)
    end

    return prob.cfl * cell_volume(mesh, 1) / λ_max
end

function compute_operator_dt(op::HyperbolicOperator{<:HyperbolicProblem2D}, U::AbstractMatrix, t)
    prob = op.problem
    law = prob.law
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    max_speed = zero(dx)
    for iy in 1:ny, ix in 1:nx
        w = conserved_to_primitive(law, U[ix + 2, iy + 2])
        λx = max_wave_speed(law, w, 1)
        λy = max_wave_speed(law, w, 2)
        speed = λx / dx + λy / dy
        max_speed = max(max_speed, speed)
    end

    return prob.cfl / max_speed
end

# ============================================================
# SourceOperator
# ============================================================

"""
    SourceOperator{Law, Src, FT} <: AbstractOperator

Cell-local stiff source operator solved implicitly via Newton iteration.

The implicit solve finds `U*` satisfying `U* = U + dt * S(U*)`.
Uses the existing `_implicit_solve_1d!` and `_implicit_solve_2d!`
infrastructure from the IMEX framework.

# Fields
- `law::Law`: Conservation law for state conversion.
- `source::Src`: The `AbstractStiffSource` implementation.
- `newton_tol::FT`: Newton iteration tolerance.
- `newton_maxiter::Int`: Maximum Newton iterations.
"""
struct SourceOperator{Law, Src, FT} <: AbstractOperator
    law::Law
    source::Src
    newton_tol::FT
    newton_maxiter::Int
end

function SourceOperator(
        law, source::AbstractStiffSource;
        newton_tol = 1.0e-10, newton_maxiter = 10
    )
    return SourceOperator(law, source, Float64(newton_tol), newton_maxiter)
end

# ---------- 1D advance! ----------

function advance!(U::AbstractVector{<:SVector}, op::SourceOperator, dt, t, workspace)
    nc = workspace.nc
    N = nvariables(op.law)
    _implicit_solve_1d!(U, op.law, op.source, dt, nc, N, op.newton_tol, op.newton_maxiter)
    return nothing
end

# ---------- 2D advance! ----------

function advance!(U::AbstractMatrix{<:SVector}, op::SourceOperator, dt, t, workspace)
    nx, ny = workspace.nx, workspace.ny
    N = nvariables(op.law)
    _implicit_solve_2d!(U, op.law, op.source, dt, nx, ny, N, op.newton_tol, op.newton_maxiter)
    return nothing
end

# ---------- dt (no CFL constraint for implicit operator) ----------

function compute_operator_dt(::SourceOperator, U, t)
    return typemax(Float64)
end
