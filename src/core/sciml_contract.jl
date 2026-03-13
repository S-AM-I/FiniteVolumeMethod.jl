# ============================================================
# Canonical SciML Execution Contract
# ============================================================
#
# `sciml_problem` is the single family-agnostic constructor for the
# SciML problem behind a FiniteVolumeMethod problem definition.
# Existing `ODEProblem`, `SteadyStateProblem`, and `SplitODEProblem`
# constructors remain available as convenience wrappers over this
# canonical path.

const _SemidiscreteSciMLProblem = Union{
    HyperbolicProblem,
    HyperbolicProblem2D,
    HyperbolicProblem3D,
    UnstructuredHyperbolicProblem,
    AMRProblem,
}

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
sciml_problem(prob::_SemidiscreteSciMLProblem; kwargs...) = ODEProblem(prob; kwargs...)

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

function CommonSolve.init(prob::_SemidiscreteSciMLProblem, args...; kwargs...)
    return CommonSolve.init(sciml_problem(prob; kwargs...), args...; kwargs...)
end

function CommonSolve.solve(prob::_SemidiscreteSciMLProblem, args...; kwargs...)
    return CommonSolve.solve(sciml_problem(prob; kwargs...), args...; kwargs...)
end

function CommonSolve.init(
        prob::_SplitSemidiscreteSciMLProblem,
        stiff_source::AbstractStiffSource,
        args...;
        kwargs...,
    )
    return CommonSolve.init(sciml_problem(prob, stiff_source; kwargs...), args...; kwargs...)
end

function CommonSolve.solve(
        prob::_SplitSemidiscreteSciMLProblem,
        stiff_source::AbstractStiffSource,
        args...;
        kwargs...,
    )
    return CommonSolve.solve(sciml_problem(prob, stiff_source; kwargs...), args...; kwargs...)
end

function CommonSolve.init(prob::AbstractFVMTemplate, args...; kwargs...)
    return CommonSolve.init(sciml_problem(prob; kwargs...), args...; kwargs...)
end

function CommonSolve.solve(prob::AbstractFVMTemplate, args...; kwargs...)
    return CommonSolve.solve(sciml_problem(prob; kwargs...), args...; kwargs...)
end
