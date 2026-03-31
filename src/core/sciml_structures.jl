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
