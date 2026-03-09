# adjoint.jl - Adjoint-based sensitivity analysis
# Migrated from Simu.jl SimuEngine/adjoint.jl
# LinearAlgebra and SparseArrays are already imported by the parent module.

"""
    compute_adjoint(Jacobian_T, dJdu; linear_solver=nothing)

Compute adjoint vector lambda satisfying: J^T * lambda = dJ/du^T.

# Arguments
- `Jacobian_T`: Transpose of the Jacobian matrix (dR/du)^T.
- `dJdu`: Gradient of objective function w.r.t state variables.
- `linear_solver`: Optional custom linear solver `(A, b) -> x`. Falls back to backslash.

# Returns
- `lambda`: Adjoint vector.
"""
function compute_adjoint(Jacobian_T, dJdu; linear_solver = nothing)
    # Solve J^T * lambda = dJdu
    rhs = dJdu

    if linear_solver !== nothing
        return linear_solver(Jacobian_T, rhs)
    else
        return Jacobian_T \ rhs
    end
end

"""
    compute_sensitivity(lambda, dRdp, dJdp)

Compute total sensitivity dJ/dp = dJ/dp_partial - lambda^T * dR/dp.

# Arguments
- `lambda`: Adjoint vector (n_states).
- `dRdp`: Partial derivative of residual w.r.t parameters (n_states x n_params).
- `dJdp`: Partial derivative of objective w.r.t parameters (n_params).

# Returns
- Sensitivity vector (n_params).
"""
function compute_sensitivity(lambda, dRdp, dJdp)
    # term2 = lambda^T * dRdp = (dRdp^T * lambda)
    term2 = dRdp' * lambda
    return dJdp - term2
end
