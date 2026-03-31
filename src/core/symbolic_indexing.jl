# ============================================================
# SymbolicIndexingInterface integration
# ============================================================
#
# Enables `sol[:rho]`, `sol[:E]`, etc. for accessing field variables
# from ODE solutions produced by FiniteVolumeMethod.jl.
#
# SciMLBase's ODEFunction.observed dispatches through getproperty on
# the sys object: `observed(fn, getproperty(fn.sys, sym))`.  We use
# a lightweight FVMVar wrapper so the returned value is not a plain
# Symbol (which would re-enter the same dispatch) and has the correct
# SII ScalarSymbolic trait.

import SciMLBase.SymbolicIndexingInterface as SII

"""
    FVMVar

Symbolic wrapper for a field variable name.  Returned by
`getproperty(::FVMSymbolicIndex, sym)` so that SciMLBase's
`observed(::ODEFunction, sym)` can resolve it via SII.
"""
struct FVMVar
    name::Symbol
end
SII.symbolic_type(::Type{FVMVar}) = SII.ScalarSymbolic()

"""
    FVMSymbolicIndex

Lightweight symbolic container attached to `ODEFunction` via the `sys`
keyword.  Maps field names to stride-based extractors on the flat ODE
state vector.

# Fields
- `names::Vector{Symbol}` — ordered field names.
- `n_vars::Int` — number of variables per cell/node.
"""
struct FVMSymbolicIndex
    names::Vector{Symbol}
    n_vars::Int
end

function Base.getproperty(sys::FVMSymbolicIndex, sym::Symbol)
    sym === :names && return getfield(sys, :names)
    sym === :n_vars && return getfield(sys, :n_vars)
    sym in getfield(sys, :names) && return FVMVar(sym)
    return error("Unknown FVM variable: $sym")
end

# ---- SII trait implementation ----

SII.constant_structure(::FVMSymbolicIndex) = true
SII.is_time_dependent(::FVMSymbolicIndex) = true
SII.is_markovian(::FVMSymbolicIndex) = true
SII.is_timeseries(::FVMSymbolicIndex) = false
SII.is_parameter_timeseries(::FVMSymbolicIndex, _) = false
SII.get_all_timeseries_indexes(::FVMSymbolicIndex, _) = Set([SII.ContinuousTimeseries()])
SciMLBase.create_parameter_timeseries_collection(::FVMSymbolicIndex, _, _) = nothing

SII.variable_symbols(sys::FVMSymbolicIndex) = [FVMVar(n) for n in sys.names]
SII.all_variable_symbols(sys::FVMSymbolicIndex) = SII.variable_symbols(sys)
SII.all_symbols(sys::FVMSymbolicIndex) = SII.variable_symbols(sys)
SII.is_variable(::FVMSymbolicIndex, _) = false
SII.is_parameter(::FVMSymbolicIndex, _) = false
SII.parameter_symbols(::FVMSymbolicIndex) = FVMVar[]
SII.is_independent_variable(::FVMSymbolicIndex, _) = false
SII.independent_variable_symbols(::FVMSymbolicIndex) = Symbol[]

# Handle both Symbol (from sol[:rho]) and FVMVar (from getproperty chain).
SII.is_observed(sys::FVMSymbolicIndex, sym::Symbol) = sym in sys.names
SII.is_observed(sys::FVMSymbolicIndex, sym::FVMVar) = sym.name in sys.names
SII.is_observed(::FVMSymbolicIndex, _) = false

function _field_index(sys::FVMSymbolicIndex, name::Symbol)
    for (i, n) in enumerate(sys.names)
        n === name && return i
    end
    return error("Unknown FVM variable: $name")
end

function SII.observed(sys::FVMSymbolicIndex, sym::Symbol)
    idx = _field_index(sys, sym)
    N = sys.n_vars
    return (u, p, t) -> u[idx:N:end]
end

function SII.observed(sys::FVMSymbolicIndex, sym::FVMVar)
    return SII.observed(sys, sym.name)
end

# ---- Constructors for each problem family ----

function fvm_symbolic_index(prob::FVMProblem)
    return FVMSymbolicIndex([:u], 1)
end

function fvm_symbolic_index(prob::FVMSystem{N}) where {N}
    names = [Symbol("u_", i) for i in 1:N]
    return FVMSymbolicIndex(names, N)
end

function fvm_symbolic_index(prob::HyperbolicProblem)
    names = Symbol.(variable_names(prob.law))
    N = nvariables(prob.law)
    return FVMSymbolicIndex(names, N)
end

function fvm_symbolic_index(prob::HyperbolicProblem2D)
    names = Symbol.(variable_names(prob.law))
    N = nvariables(prob.law)
    return FVMSymbolicIndex(names, N)
end

function fvm_symbolic_index(prob::HyperbolicProblem3D)
    names = Symbol.(variable_names(prob.law))
    N = nvariables(prob.law)
    return FVMSymbolicIndex(names, N)
end

function fvm_symbolic_index(prob::UnstructuredHyperbolicProblem)
    names = Symbol.(variable_names(prob.law))
    N = nvariables(prob.law)
    return FVMSymbolicIndex(names, N)
end
