# adjoint/steady.jl — Discrete adjoint for the steady linear system.
#
# For a linear state equation `R(u, p) = A(p)·u − b(p) = 0` and a cost
# functional `J(u, p)`, the discrete adjoint solves the transposed system
#
#     A(p)^T · λ = (∂J/∂u)^T
#
# and evaluates the total design-parameter derivative
#
#     dJ/dp = ∂J/∂p − λ^T · ∂R/∂p.
#
# This is the textbook reverse-mode derivative through a linear solve;
# its cost is a single transposed solve irrespective of |p|.

"""
    solve_steady_adjoint(A, b, u, dJ_du, dR_dp;
                         partial_dJ_dp = nothing,
                         linear_solver = nothing) -> (lambda, dJ_dp)

Discrete steady adjoint.

# Arguments
- `A` — square system matrix (e.g. assembled momentum Jacobian).
- `b` — right-hand side (unused here but kept in the signature for
  symmetry with a forthcoming nonlinear-adjoint version).
- `u` — converged forward solution `A·u = b`.
- `dJ_du` — the row gradient `∂J/∂u` as an `AbstractVector`.
- `dR_dp` — `∂R/∂p` as a matrix with `length(u)` rows and `n_p` columns.
- `partial_dJ_dp` — optional explicit partial `∂J/∂p`. Defaults to zero.
- `linear_solver` — unused for the default backslash path; reserved so
  that callers can plumb a `LinearSolve` algorithm through once
  `_dispatch_solve` learns a transposed variant.

Returns `(lambda, dJ_dp)`.
"""
function solve_steady_adjoint(
        A, b, u, dJ_du, dR_dp;
        partial_dJ_dp = nothing,
        linear_solver = nothing,
    )
    # Lagrangian derivation with A^T·λ = (∂J/∂u)^T gives
    # dJ/dp = ∂J/∂p − λ^T · ∂R/∂p (sign from ∂u/∂p = −A^{−1}·∂R/∂p).
    lambda = transpose(A) \ collect(dJ_du)
    grad = -(transpose(lambda) * dR_dp)
    result = collect(vec(grad))
    if partial_dJ_dp !== nothing
        result .+= collect(partial_dJ_dp)
    end
    return (lambda, result)
end

"""
    verify_adjoint_gradient(J, solve_forward, p; epsilon = 1e-6) -> (dJ_fd,)

Symmetric-difference finite-difference gradient of `J` with respect to
`p`. Used by the V&V harness to compare against
[`solve_steady_adjoint`](@ref).

`solve_forward(p) -> u` must return the converged forward solution for
the supplied parameter vector.
"""
function verify_adjoint_gradient(
        J::Function, solve_forward::Function, p::AbstractVector{T};
        epsilon::Real = 1.0e-6,
    ) where {T}
    n = length(p)
    dJ_fd = zeros(T, n)
    for i in 1:n
        p_plus = copy(p); p_plus[i] += T(epsilon)
        p_minus = copy(p); p_minus[i] -= T(epsilon)
        u_plus = solve_forward(p_plus)
        u_minus = solve_forward(p_minus)
        dJ_fd[i] = (J(u_plus, p_plus) - J(u_minus, p_minus)) / (2 * T(epsilon))
    end
    return dJ_fd
end
