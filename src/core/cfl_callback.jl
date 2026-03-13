# ============================================================
# CFL Timestep Callback
# ============================================================
#
# A DiscreteCallback that recomputes the CFL-constrained time step
# after each integration step and proposes it for the next step.

"""
    cfl_stepsize_callback(cache::AbstractSemidiscreteCache)

Create a `DiscreteCallback` that enforces CFL timestep control.

After each step, the callback unfolds the current state into the
ghost-padded array, computes the CFL-limited dt, and proposes it
for the next step via `set_proposed_dt!`.
"""
function cfl_stepsize_callback(cache::AbstractSemidiscreteCache)
    condition = (u, t, integrator) -> true
    function affect!(integrator)
        dt_cfl = _compute_cfl_dt(integrator.p, integrator.u, integrator.t)
        # Don't overshoot final time
        t_remaining = last(integrator.sol.prob.tspan) - integrator.t
        if dt_cfl > t_remaining && t_remaining > zero(t_remaining)
            dt_cfl = t_remaining
        end
        set_proposed_dt!(integrator, dt_cfl)
    end
    return DiscreteCallback(condition, affect!; save_positions = (false, false))
end

"""
    _compute_cfl_dt(cache, u, t) -> dt

Compute the CFL-limited timestep from the current state.
Dispatches on cache type.
"""
function _compute_cfl_dt(cache::HyperbolicCache1D{N, FT}, u::AbstractVector, t) where {N, FT}
    unfold_to_padded!(cache, u)
    return compute_dt(cache.prob, cache.padded_U, t)
end

function _compute_cfl_dt(cache::HyperbolicCache2D{N, FT}, u::AbstractVector, t) where {N, FT}
    unfold_to_padded!(cache, u)
    return compute_dt_2d(cache.prob, cache.padded_U, t)
end

function _compute_cfl_dt(cache::HyperbolicCache3D{N, FT}, u::AbstractVector, t) where {N, FT}
    unfold_to_padded!(cache, u)
    return compute_dt_3d(cache.prob, cache.padded_U, t)
end

function _compute_cfl_dt(cache::UnstructuredCache{N, FT}, u::AbstractVector, t) where {N, FT}
    unfold_to_padded!(cache, u)
    return compute_dt_unstructured(cache.prob, cache.U, t)
end

function _compute_cfl_dt(cache::MHDCTCache2D{N, FT}, u::AbstractVector, t) where {N, FT}
    unfold_mhd_augmented!(cache, u)
    return compute_dt_2d(cache.prob, cache.padded_U, t)
end

function _compute_cfl_dt(cache::GRMHDCTCache2D{N, FT}, u::AbstractVector, t) where {N, FT}
    unfold_mhd_augmented!(cache, u)
    return compute_dt_2d(cache.prob, cache.padded_U, t, cache.metric_data)
end

function _compute_cfl_dt(cache::AMRCache{N, FT}, u::AbstractVector, t) where {N, FT}
    # Use minimum dt across all active blocks
    dt_min = typemax(FT)
    for bid in cache.block_ids
        block = cache.grid.blocks[bid]
        dt_block = _compute_dt_block(block, cache.law_ref, cache.cfl)
        dt_min = min(dt_min, dt_block)
    end
    return dt_min
end

"""
    compute_initial_dt(cache, u0) -> dt

Compute the CFL-limited timestep for the initial state.
Used to set the initial dt for `solve`.
"""
function compute_initial_dt(cache::AbstractSemidiscreteCache, u0::AbstractVector)
    prob = _get_prob(cache)
    return _compute_cfl_dt(cache, u0, prob.initial_time)
end

function _get_prob(cache::HyperbolicCache1D)
    return cache.prob
end
function _get_prob(cache::HyperbolicCache2D)
    return cache.prob
end
function _get_prob(cache::HyperbolicCache3D)
    return cache.prob
end
function _get_prob(cache::UnstructuredCache)
    return cache.prob
end
function _get_prob(cache::MHDCTCache2D)
    return cache.prob
end
function _get_prob(cache::GRMHDCTCache2D)
    return cache.prob
end
