# ============================================================
# SciMLStructures.jl integration
# ============================================================
#
# Enables parameter partitioning for optimization and sensitivity
# workflows.  Tunable parameters are extracted from the EOS and CFL
# stored in `cache.prob`; the padded arrays and mesh data are caches.

import SciMLBase.SciMLStructures as SS

# ---- Hyperbolic caches ----

# Helper: extract tunable floats from an EOS
_eos_tunables(eos::IdealGasEOS) = [eos.gamma]
_eos_tunables(eos::StiffenedGasEOS) = [eos.gamma, eos.P_inf]
_eos_tunables(::Any) = Float64[]   # fallback for unknown EOS types

_rebuild_eos(::IdealGasEOS, vals) = IdealGasEOS(vals[1])
_rebuild_eos(::StiffenedGasEOS, vals) = StiffenedGasEOS(vals[1], vals[2])
_rebuild_eos(eos, _) = eos

_eos_from_law(law) = law.eos
_eos_from_law(law::NavierStokesEquations) = law.euler.eos
_n_eos_tunables(eos::IdealGasEOS) = 1
_n_eos_tunables(eos::StiffenedGasEOS) = 2
_n_eos_tunables(::Any) = 0

# The tunable vector layout for any hyperbolic cache:
#   [eos_param_1, ..., eos_param_k, cfl]
function _hyp_tunable_values(cache)
    prob = cache.prob
    eos_vals = _eos_tunables(_eos_from_law(prob.law))
    return vcat(eos_vals, [prob.cfl])
end

function _hyp_n_tunables(cache)
    return _n_eos_tunables(_eos_from_law(cache.prob.law)) + 1
end

function _hyp_repack(cache, new_vals)
    prob = cache.prob
    eos = _eos_from_law(prob.law)
    n_eos = _n_eos_tunables(eos)
    new_eos = _rebuild_eos(eos, @view new_vals[1:n_eos])
    new_cfl = new_vals[n_eos + 1]
    new_law = _rebuild_law(prob.law, new_eos)
    new_prob = _replace_law_cfl(prob, new_law, new_cfl)
    return _replace_prob(cache, new_prob)
end

# Law reconstruction: create a new law with a new EOS
_rebuild_law(law::EulerEquations{D}, eos) where {D} = EulerEquations{D}(eos)
_rebuild_law(law::IdealMHDEquations, eos) = IdealMHDEquations(eos)
_rebuild_law(law::NavierStokesEquations{D}, eos) where {D} =
    NavierStokesEquations{D}(eos; mu = law.mu, Pr = law.Pr)
_rebuild_law(law, _) = law  # fallback — no EOS to replace

# Problem field replacement helpers using remake
function _replace_law_cfl(prob::HyperbolicProblem, new_law, new_cfl)
    return remake(prob; law = new_law, cfl = new_cfl)
end
function _replace_law_cfl(prob::HyperbolicProblem2D, new_law, new_cfl)
    return remake(prob; law = new_law, cfl = new_cfl)
end
function _replace_law_cfl(prob::HyperbolicProblem3D, new_law, new_cfl)
    return remake(prob; law = new_law, cfl = new_cfl)
end
function _replace_law_cfl(prob, new_law, new_cfl)
    return prob  # fallback
end

# Cache reconstruction: create new cache with replaced prob
function _replace_prob(cache::HyperbolicCache1D{N, FT}, new_prob) where {N, FT}
    return HyperbolicCache1D{N, FT, typeof(new_prob)}(
        new_prob, cache.padded_U, cache.padded_dU, cache.nc, cache.ng,
    )
end
function _replace_prob(cache::HyperbolicCache2D{N, FT}, new_prob) where {N, FT}
    return HyperbolicCache2D{N, FT, typeof(new_prob)}(
        new_prob, cache.padded_U, cache.padded_dU, cache.nx, cache.ny, cache.ng,
    )
end
function _replace_prob(cache::HyperbolicCache3D{N, FT}, new_prob) where {N, FT}
    return HyperbolicCache3D{N, FT, typeof(new_prob)}(
        new_prob, cache.padded_U, cache.padded_dU, cache.nx, cache.ny, cache.nz, cache.ng,
    )
end

# ---- SciMLStructures interface for hyperbolic caches ----

const _HyperbolicCaches = Union{
    HyperbolicCache1D,
    HyperbolicCache2D,
    HyperbolicCache3D,
}

SS.isscimlstructure(::_HyperbolicCaches) = true
SS.ismutablescimlstructure(::_HyperbolicCaches) = false

SS.hasportion(::SS.Tunable, ::_HyperbolicCaches) = true
SS.hasportion(::SS.Constants, ::_HyperbolicCaches) = false
SS.hasportion(::SS.Caches, ::_HyperbolicCaches) = false
SS.hasportion(::SS.Discrete, ::_HyperbolicCaches) = false

function SS.canonicalize(::SS.Tunable, cache::_HyperbolicCaches)
    vals = _hyp_tunable_values(cache)
    repack = new_vals -> _hyp_repack(cache, new_vals)
    return vals, repack, false  # no aliasing
end

function SS.replace(::SS.Tunable, cache::_HyperbolicCaches, new_vals)
    return _hyp_repack(cache, new_vals)
end

# ---- Incompressible problem ----
#
# Stage 1e: extensible named-entry Tunable schema.
#
# The canonical `SciMLStructures.canonicalize(Tunable, prob)` still returns
# a flat `Vector{T}` — that's what downstream SciML consumers expect. What
# we add is a *named schema* so new tunables (turbulence closure constants,
# thermal models, rheology parameters, etc.) can register their own
# getters/setters without editing any positional index logic in this file.
#
# Schema registry: problem type → ordered list of (name, getter, setter).
# - `getter(prob) -> T` pulls the current value.
# - `setter(prob, v) -> new_prob` builds a new problem with that field updated.
#
# Adding a tunable is one `register_tunable!` call. No positional fragility.
# `tunable_names(prob)` and `tunable_namedtuple(prob)` give introspection.

"""
    TunableEntry{P, T}

Describes one named tunable scalar for a problem type `P`. `getter(prob) -> T`
returns the current value; `setter(prob, v) -> prob'` returns a new problem
with the scalar updated. Used by Stage 1e's named-partition schema for
`SciMLStructures.Tunable`.
"""
struct TunableEntry{F, G}
    name::Symbol
    getter::F
    setter::G
end

# Registry: keyed on concrete problem type. Declared as a `Dict{DataType, …}`
# so any concrete `IncompressibleProblem{Dim, T, …}` looks up the generic
# `IncompressibleProblem` schema. Test helpers may register new entries.
const _TUNABLE_REGISTRY = Dict{Type, Vector{TunableEntry}}()

"""
    register_tunable!(PType, name, getter, setter)

Register a new tunable scalar on problem type `PType`. Appended to the
named Tunable schema; order determines the position in the flat
canonical vector. Stage 1e extension hook — downstream solver extensions
(turbulence closures, thermal models, rheology) should use this to
advertise their own tunables without editing this file.
"""
function register_tunable!(
        PType::Type, name::Symbol, getter::Function, setter::Function,
    )
    entries = get!(Vector{TunableEntry}, _TUNABLE_REGISTRY, PType)
    push!(entries, TunableEntry(name, getter, setter))
    return nothing
end

"""
    tunable_schema(prob) -> Vector{TunableEntry}

Return the ordered list of tunable entries for this problem type. Empty
if the problem is not registered. Matches the concrete problem type
against every registered type (concrete DataType or parametric UnionAll)
via `prob isa key`.
"""
function tunable_schema(prob)
    entries = TunableEntry[]
    for (key, list) in _TUNABLE_REGISTRY
        if prob isa key
            append!(entries, list)
        end
    end
    return entries
end

"""
    tunable_names(prob) -> Vector{Symbol}

Return the names of every tunable for this problem, in canonical order.
"""
tunable_names(prob) = [entry.name for entry in tunable_schema(prob)]

"""
    tunable_namedtuple(prob) -> NamedTuple

Return the current tunable values as a `NamedTuple` keyed by their
registered names. Useful for human-readable introspection.
"""
function tunable_namedtuple(prob)
    schema = tunable_schema(prob)
    names = Tuple(entry.name for entry in schema)
    vals = Tuple(entry.getter(prob) for entry in schema)
    return NamedTuple{names}(vals)
end

function _schema_values(prob::IncompressibleProblem{Dim, T}) where {Dim, T}
    schema = tunable_schema(prob)
    return T[T(entry.getter(prob)) for entry in schema]
end

function _schema_repack(prob::IncompressibleProblem{Dim, T}, new_vals) where {Dim, T}
    schema = tunable_schema(prob)
    length(schema) == length(new_vals) || error(
        "Tunable schema has $(length(schema)) entries but new_vals has $(length(new_vals)); ",
        "schema: ", [e.name for e in schema],
    )
    out = prob
    @inbounds for (i, entry) in pairs(schema)
        out = entry.setter(out, T(new_vals[i]))
    end
    return out
end

# Register the built-in IncompressibleProblem tunables. Order here is the
# canonical order in the flat SciMLStructures vector.
_register_builtin_incomp_tunables() = let
    register_tunable!(
        IncompressibleProblem, :nu,
        prob -> prob.nu,
        (prob, v) -> remake(prob; nu = v),
    )
    register_tunable!(
        IncompressibleProblem, :density,
        prob -> prob.density,
        (prob, v) -> remake(prob; density = v),
    )
    register_tunable!(
        IncompressibleProblem, :alpha_U,
        prob -> _algo_field(prob.algorithm, :alpha_U, prob.nu),
        (prob, v) -> remake(prob; algorithm = _with_algo_field(prob.algorithm, :alpha_U, v)),
    )
    register_tunable!(
        IncompressibleProblem, :alpha_p,
        prob -> _algo_field(prob.algorithm, :alpha_p, prob.nu),
        (prob, v) -> remake(prob; algorithm = _with_algo_field(prob.algorithm, :alpha_p, v)),
    )
    register_tunable!(
        IncompressibleProblem, :tolerance,
        prob -> _algo_field(prob.algorithm, :tolerance, prob.nu),
        (prob, v) -> remake(prob; algorithm = _with_algo_field(prob.algorithm, :tolerance, v)),
    )
    return nothing
end

# Algorithm-field accessors tolerant of absent fields (PISO has no alpha_U, etc.)
@inline function _algo_field(algo, name::Symbol, fallback)
    return hasfield(typeof(algo), name) ? getfield(algo, name) : fallback
end

# Return a new algorithm of the same type with `name` set to `v`; no-op for
# algorithm types that don't have that field (e.g. PISO has no alpha_U).
function _with_algo_field(algo::SIMPLE{T}, name::Symbol, v) where {T}
    name === :alpha_U && return SIMPLE{T}(T(v), algo.alpha_p, algo.max_iterations, algo.tolerance)
    name === :alpha_p && return SIMPLE{T}(algo.alpha_U, T(v), algo.max_iterations, algo.tolerance)
    name === :tolerance && return SIMPLE{T}(algo.alpha_U, algo.alpha_p, algo.max_iterations, T(v))
    return algo
end
function _with_algo_field(algo::PIMPLE{T}, name::Symbol, v) where {T}
    name === :alpha_U && return PIMPLE{T}(algo.n_outer, algo.n_correctors, T(v), algo.alpha_p, algo.tolerance)
    name === :alpha_p && return PIMPLE{T}(algo.n_outer, algo.n_correctors, algo.alpha_U, T(v), algo.tolerance)
    name === :tolerance && return PIMPLE{T}(algo.n_outer, algo.n_correctors, algo.alpha_U, algo.alpha_p, T(v))
    return algo
end
_with_algo_field(algo, ::Symbol, _) = algo

# Call registration once when this module is loaded.
_register_builtin_incomp_tunables()

SS.isscimlstructure(::IncompressibleProblem) = true
SS.ismutablescimlstructure(::IncompressibleProblem) = false

SS.hasportion(::SS.Tunable, ::IncompressibleProblem) = true
SS.hasportion(::SS.Constants, ::IncompressibleProblem) = false
SS.hasportion(::SS.Caches, ::IncompressibleProblem) = false
SS.hasportion(::SS.Discrete, ::IncompressibleProblem) = false

function SS.canonicalize(::SS.Tunable, prob::IncompressibleProblem)
    vals = _schema_values(prob)
    repack = new_vals -> _schema_repack(prob, new_vals)
    return vals, repack, false
end

function SS.replace(::SS.Tunable, prob::IncompressibleProblem, new_vals)
    return _schema_repack(prob, new_vals)
end
