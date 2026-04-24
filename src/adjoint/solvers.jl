# adjoint/solvers.jl — Dispatch entry point for adjoint solves.

"""
    solve_adjoint(algorithm::AbstractAdjointAlgorithm, A, b, u, dJ_du, dR_dp; kwargs...)

Dispatch wrapper that chooses between [`SteadyAdjoint`](@ref) and
[`TransientAdjoint`](@ref). Returns `(lambda, dJ_dp)` for steady
problems; throws for transient.
"""
function solve_adjoint(
        ::SteadyAdjoint, A, b, u, dJ_du, dR_dp;
        partial_dJ_dp = nothing, linear_solver = nothing,
    )
    return solve_steady_adjoint(
        A, b, u, dJ_du, dR_dp;
        partial_dJ_dp = partial_dJ_dp, linear_solver = linear_solver,
    )
end

function solve_adjoint(::TransientAdjoint, args...; kwargs...)
    return solve_transient_adjoint(args...; kwargs...)
end
