# incompressible/solution.jl — SciML-compatible solution wrapper
#
# Wraps SolveResult with symbolic field access and SciML traits.

"""
    IncompressibleSolution{Dim, T}

SciML-compatible solution for incompressible flow problems.
Supports symbolic field access: `sol[:U]`, `sol[:p]`, `sol[:Ux]`, etc.
"""
struct IncompressibleSolution{Dim, T}
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
