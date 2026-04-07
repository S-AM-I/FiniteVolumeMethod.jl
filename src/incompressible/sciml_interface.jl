# incompressible/sciml_interface.jl — SciML CommonSolve dispatch
#
# Enables standard `solve(prob, alg; kwargs...)` pattern for
# IncompressibleProblem, returning IncompressibleSolution.

"""
    solve(prob::IncompressibleProblem, alg::SIMPLE; kwargs...)

Solve a steady-state incompressible problem using SIMPLE.
Returns an [`IncompressibleSolution`](@ref) with symbolic field access.
"""
function CommonSolve.solve(
        prob::IncompressibleProblem{Dim, T},
        alg::SIMPLE;
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    actual_prob = alg === prob.algorithm ? prob : remake(prob; algorithm = alg)
    result = solve_simple(
        actual_prob;
        linear_solver = linear_solver,
        solver_config = solver_config, verbose = verbose
    )
    return IncompressibleSolution(result, actual_prob)
end

"""
    solve(prob::IncompressibleProblem, alg::Union{PISO, PIMPLE}; tspan, dt, kwargs...)

Solve a transient incompressible problem using PISO or PIMPLE.
"""
function CommonSolve.solve(
        prob::IncompressibleProblem{Dim, T},
        alg::Union{PISO, PIMPLE};
        tspan::Tuple{T, T},
        dt::T,
        save_every::Int = 1,
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    actual_prob = alg === prob.algorithm ? prob : remake(prob; algorithm = alg)
    result = solve_incompressible(
        actual_prob, tspan, dt;
        save_every = save_every, linear_solver = linear_solver,
        solver_config = solver_config, verbose = verbose
    )
    return IncompressibleSolution(result, actual_prob)
end

"""
    solve(prob::IncompressibleProblem; kwargs...)

Solve using the algorithm stored in `prob.algorithm`.
"""
function CommonSolve.solve(
        prob::IncompressibleProblem{Dim, T};
        kwargs...,
    ) where {Dim, T}
    return CommonSolve.solve(prob, prob.algorithm; kwargs...)
end
