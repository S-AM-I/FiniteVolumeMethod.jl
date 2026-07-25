# incompressible/symbolic.jl — SymbolicIndexingInterface for the collocated
# incompressible family.
#
# `sol[:U]`, `sol[:p]`, `sol[:Ux]`, … are resolved through SII rather than a
# hand-written lookup: the flat solution vector `u` (Stage 5f-1) has a known
# block layout, so each field name maps to an extractor over `u`.
#
# Deliberately NOT subtyping `SciMLBase.AbstractODESolution`: that contract
# implies a real time axis and an interpolating `sol(t)`, which this solver
# cannot honour — the steady SIMPLE solve has no time axis at all, and the
# transient solvers store snapshots only every `save_every` steps rather than a
# dense trajectory. The SII traits give the symbolic-access interop without
# claiming an accuracy the solver does not deliver.

import SymbolicIndexingInterface as SII

@doc """
    CollocatedSymbolicIndex

Describes the block layout of the flat incompressible solution vector
`u = [U-block (ncells·dim, component-interleaved) ; p-block (ncells)]` so that
[`SymbolicIndexingInterface`](https://github.com/SciML/SymbolicIndexingInterface.jl)
can extract named fields from it.

# Fields
- `dim::Int` — spatial dimension
- `ncells::Int` — number of interior cells
"""
struct CollocatedSymbolicIndex
    dim::Int
    ncells::Int
end

# Field names carried by the flat state vector. `:phi` is deliberately absent:
# the face flux is derived state, not part of `u`, so it cannot be an SII
# observable over the solution vector (it stays reachable via `sol[:phi]`).
function _collocated_symbols(sys::CollocatedSymbolicIndex)
    sys.dim == 3 && return (:U, :p, :Ux, :Uy, :Uz)
    return (:U, :p, :Ux, :Uy)
end

# ── SII traits ──────────────────────────────────────────────────────

SII.constant_structure(::CollocatedSymbolicIndex) = true
# The steady solve has no time axis and the transient one exposes its state at
# the current step only, so the container is not a timeseries.
SII.is_time_dependent(::CollocatedSymbolicIndex) = false
SII.is_variable(::CollocatedSymbolicIndex, _) = false
SII.variable_symbols(::CollocatedSymbolicIndex) = Symbol[]
SII.is_parameter(::CollocatedSymbolicIndex, _) = false
SII.parameter_symbols(::CollocatedSymbolicIndex) = Symbol[]
SII.is_independent_variable(::CollocatedSymbolicIndex, _) = false
SII.independent_variable_symbols(::CollocatedSymbolicIndex) = Symbol[]

# The named fields are blocks of / strided views into `u`, so they are exposed
# as observables rather than as scalar state variables.
SII.is_observed(sys::CollocatedSymbolicIndex, sym::Symbol) = sym in _collocated_symbols(sys)
SII.is_observed(::CollocatedSymbolicIndex, _) = false
SII.all_variable_symbols(sys::CollocatedSymbolicIndex) = collect(_collocated_symbols(sys))
SII.all_symbols(sys::CollocatedSymbolicIndex) = SII.all_variable_symbols(sys)

@doc """
    SII.observed(sys::CollocatedSymbolicIndex, sym::Symbol)

Return `(u, p) -> field`, a view-producing extractor for `sym` over the flat
solution vector. Time-independent, so the returned function takes no `t`.
"""
function SII.observed(sys::CollocatedSymbolicIndex, sym::Symbol)
    dim = sys.dim
    nu = sys.ncells * dim
    sym === :U && return (u, _) -> reinterpret(SVector{dim, eltype(u)}, view(u, 1:nu))
    sym === :p && return (u, _) -> view(u, (nu + 1):(nu + sys.ncells))
    sym === :Ux && return (u, _) -> view(u, 1:dim:nu)
    sym === :Uy && return dim >= 2 ? (u, _) -> view(u, 2:dim:nu) :
        error("No Uy in $(dim)D")
    sym === :Uz && return dim >= 3 ? (u, _) -> view(u, 3:dim:nu) :
        error("No Uz in $(dim)D")
    return error(
        "Unknown field :$sym. Available: " *
            join(string.(':', _collocated_symbols(sys)), ", ")
    )
end

# ── Solution / integrator as SII value providers ────────────────────

function SII.symbolic_container(sol::IncompressibleSolution{Dim}) where {Dim}
    state = getfield(sol, :result).state
    return CollocatedSymbolicIndex(Dim, length(state.p.internal))
end

SII.state_values(sol::IncompressibleSolution) = getfield(sol, :result).state.u
SII.parameter_values(sol::IncompressibleSolution) = getfield(sol, :prob)

# A problem is an SII value provider too. These types root at SciMLBase problem
# supertypes, so generic ecosystem code reaches for the problem's state; they do
# not store a `u0` field, because the initial state is built from the mesh at
# solve time, so report that initial state instead of throwing a `FieldError`.
function SII.symbolic_container(prob::AnyIncompressibleProblem{Dim}) where {Dim}
    return CollocatedSymbolicIndex(Dim, length(prob.mesh.cell_volumes))
end

SII.state_values(prob::AnyIncompressibleProblem) = IncompressibleState(prob.mesh).u
SII.parameter_values(prob::AnyIncompressibleProblem) = prob

function SII.symbolic_container(integrator::IncompressibleIntegrator)
    state = getfield(integrator, :state)
    return CollocatedSymbolicIndex(
        _problem_dim(getfield(integrator, :prob)), length(state.p.internal),
    )
end

SII.state_values(integrator::IncompressibleIntegrator) = getfield(integrator, :state).u
SII.parameter_values(integrator::IncompressibleIntegrator) = getfield(integrator, :prob)
