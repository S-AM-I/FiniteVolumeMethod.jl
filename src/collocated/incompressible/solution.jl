# incompressible/solution.jl — SciML-compatible solution wrapper
#
# Wraps SolveResult with symbolic field access and SciML traits.

"""
    AbstractFVMSolution

Stage 1f umbrella supertype for solution wrappers owned by this repo.
Currently just `IncompressibleSolution`; parabolic and hyperbolic
problems return standard `SciMLBase.ODESolution` which is already
`sol[:field]`-enabled via `SymbolicIndexingInterface`. The
`is_fvm_solution` trait below lets downstream code recognize both
families without forcing `SciMLBase.ODESolution` under this supertype
(we don't own that type).
"""
abstract type AbstractFVMSolution end

"""
    IncompressibleSolution{Dim, T}

SciML-compatible solution for incompressible flow problems.
Supports symbolic field access: `sol[:U]`, `sol[:p]`, `sol[:Ux]`, etc.
"""
struct IncompressibleSolution{Dim, T} <: AbstractFVMSolution
    result::SolveResult{Dim, T}
    prob::IncompressibleProblem
    retcode::Symbol
end

function IncompressibleSolution(result::SolveResult{Dim, T}, prob) where {Dim, T}
    retcode = result.converged ? :Success : :MaxIters
    return IncompressibleSolution{Dim, T}(result, prob, retcode)
end

# Field access
function Base.propertynames(::IncompressibleSolution)
    return (:converged, :iterations, :residuals, :state, :retcode, :prob)
end

function Base.getproperty(sol::IncompressibleSolution, sym::Symbol)
    sym === :retcode && return getfield(sol, :retcode)
    sym === :prob && return getfield(sol, :prob)
    sym === :result && return getfield(sol, :result)
    sym === :converged && return getfield(sol, :result).converged
    sym === :iterations && return getfield(sol, :result).iterations
    sym === :residuals && return getfield(sol, :result).residuals
    sym === :state && return getfield(sol, :result).state
    return getfield(sol, sym)
end

# Symbolic indexing: sol[:U], sol[:p], sol[:Ux], sol[:Uy], sol[:Uz], sol[:phi]
function Base.getindex(sol::IncompressibleSolution{Dim, T}, sym::Symbol) where {Dim, T}
    state = getfield(sol, :result).state
    sym === :U && return state.U.internal
    sym === :p && return state.p.internal
    sym === :phi && return state.phi.values
    sym === :Ux && return _extract_component(state.U, 1)
    sym === :Uy && return Dim >= 2 ? _extract_component(state.U, 2) :
        error("No Uy in $(Dim)D")
    sym === :Uz && return Dim >= 3 ? _extract_component(state.U, 3) :
        error("No Uz in $(Dim)D")
    return error(
        "Unknown field :$sym. Available: :U, :p, :phi, :Ux, :Uy" *
            (Dim >= 3 ? ", :Uz" : ""),
    )
end

function Base.keys(::IncompressibleSolution{2})
    return (:U, :p, :phi, :Ux, :Uy)
end

function Base.keys(::IncompressibleSolution{3})
    return (:U, :p, :phi, :Ux, :Uy, :Uz)
end

function Base.show(
        io::IO, ::MIME"text/plain", sol::IncompressibleSolution{Dim, T},
    ) where {Dim, T}
    status = sol.retcode === :Success ? "converged" : "not converged"
    print(io, "IncompressibleSolution{$Dim, $T} ($status in $(sol.iterations) iterations)")
    return nothing
end

# ── Stage 1f generic-solution trait ──────────────────────────────────

"""
    is_fvm_solution(sol) -> Bool

Trait-style predicate returning `true` if `sol` was produced by any of
this repo's solvers (either a custom wrapper like `IncompressibleSolution`
or a `SciMLBase.AbstractODESolution` coming out of `solve(::FVMProblem)`,
`solve(::HyperbolicProblem)`, etc.). Downstream generic utilities can use
this to decide whether to invoke FVM-specific post-processing, symbolic
indexing, or plotting recipes.

Default: `false`.
"""
is_fvm_solution(::Any) = false
is_fvm_solution(::AbstractFVMSolution) = true
is_fvm_solution(::SciMLBase.AbstractODESolution) = true
is_fvm_solution(::SciMLBase.AbstractNonlinearSolution) = true
is_fvm_solution(::SciMLBase.AbstractLinearSolution) = true
