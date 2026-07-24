# ============================================================
# Semi-discrete Problem Constructors
# ============================================================
#
# Maps each FVM problem type to its SciML problem form:
#   Parabolic/elliptic  → ODEProblem  or  SteadyStateProblem
#   Hyperbolic (MOL)    → ODEProblem
#   Split (IMEX)        → SplitODEProblem
#
# `sciml_problem` is the family-agnostic entry point; the
# type-specific `ODEProblem` / `SteadyStateProblem` constructors
# remain available as convenience wrappers.

# The semidiscrete SciML entry points dispatch on the family root. This was a
# hand-maintained Union of the five concrete problem types until Stage 5c; an
# abstract root means a new problem type joins by subtyping rather than by
# being added here.
#
# The split (IMEX) entry points stay a Union: only the 1D and 2D problems have
# a SplitODEProblem formulation, so this is a genuine subset rather than the
# whole family.
const _SplitSemidiscreteSciMLProblem = Union{
    HyperbolicProblem,
    HyperbolicProblem2D,
}

"""
    sciml_problem(prob; kwargs...)

Construct the canonical SciML problem corresponding to `prob`.

This is the preferred entry point when writing solver-family-generic
code, tests, or reproducibility scripts.
"""
sciml_problem(prob::Union{FVMProblem, FVMSystem}; kwargs...) = ODEProblem(prob; kwargs...)
sciml_problem(prob::SteadyFVMProblem; kwargs...) = SteadyStateProblem(prob; kwargs...)
sciml_problem(prob::AbstractHyperbolicProblem; kwargs...) = ODEProblem(prob; kwargs...)

function sciml_problem(prob::AbstractFVMTemplate; kwargs...)
    if isempty(kwargs)
        return prob.problem
    end
    return SciMLBase.remake(prob.problem; kwargs...)
end

"""
    sciml_problem(prob, stiff_source; kwargs...)

Construct the canonical split SciML problem for semidiscrete
hyperbolic problems with a stiff source.
"""
sciml_problem(
    prob::_SplitSemidiscreteSciMLProblem,
    stiff_source::AbstractStiffSource;
    kwargs...,
) = SplitODEProblem(prob, stiff_source; kwargs...)

function CommonSolve.init(prob::AbstractHyperbolicProblem, args...; callback = nothing, kwargs...)
    ode_prob = sciml_problem(prob; kwargs...)
    merged_callback = _merge_problem_callbacks(_problem_callback(ode_prob), callback)
    if merged_callback === nothing
        return CommonSolve.init(ode_prob, args...; kwargs...)
    end
    return CommonSolve.init(ode_prob, args...; callback = merged_callback, kwargs...)
end

function CommonSolve.solve(prob::AbstractHyperbolicProblem, args...; callback = nothing, kwargs...)
    ode_prob = sciml_problem(prob; kwargs...)
    merged_callback = _merge_problem_callbacks(_problem_callback(ode_prob), callback)
    if merged_callback === nothing
        return CommonSolve.solve(ode_prob, args...; kwargs...)
    end
    return CommonSolve.solve(ode_prob, args...; callback = merged_callback, kwargs...)
end

function CommonSolve.init(
        prob::_SplitSemidiscreteSciMLProblem,
        stiff_source::AbstractStiffSource,
        args...;
        callback = nothing,
        kwargs...,
    )
    split_prob = sciml_problem(prob, stiff_source; kwargs...)
    merged_callback = _merge_problem_callbacks(_problem_callback(split_prob), callback)
    if merged_callback === nothing
        return CommonSolve.init(split_prob, args...; kwargs...)
    end
    return CommonSolve.init(split_prob, args...; callback = merged_callback, kwargs...)
end

function CommonSolve.solve(
        prob::_SplitSemidiscreteSciMLProblem,
        stiff_source::AbstractStiffSource,
        args...;
        callback = nothing,
        kwargs...,
    )
    split_prob = sciml_problem(prob, stiff_source; kwargs...)
    merged_callback = _merge_problem_callbacks(_problem_callback(split_prob), callback)
    if merged_callback === nothing
        return CommonSolve.solve(split_prob, args...; kwargs...)
    end
    return CommonSolve.solve(split_prob, args...; callback = merged_callback, kwargs...)
end

function CommonSolve.init(prob::AbstractFVMTemplate, args...; kwargs...)
    return CommonSolve.init(sciml_problem(prob; kwargs...), args...; kwargs...)
end

"""
    solve(prob::AbstractFVMTemplate, args...; kwargs...)

Solve a template problem through its canonical SciML problem.

This is equivalent to
`solve(sciml_problem(prob; kwargs...), args...; kwargs...)` and keeps the
template families aligned with the same SciML-oriented execution contract used
by the rest of the package.
"""
function CommonSolve.solve(prob::AbstractFVMTemplate, args...; kwargs...)
    return CommonSolve.solve(sciml_problem(prob; kwargs...), args...; kwargs...)
end
