# adjoint/checkpointing.jl — Uniform-interval checkpointing for transient adjoints.
#
# Classical Griewank–Walther (2000) uniform checkpointing: for a forward
# trajectory of `N_t` steps, store state every `interval` steps. Memory is
# O(N_t / interval); reconstructing an arbitrary step costs at most
# `interval` forward recomputations. This trades compute for memory and is
# the simplest scheme that beats the naive "store every step" strategy for
# long transients.
#
# The design keeps `interval == 1` as a pass-through (full storage) so the
# V&V harness can diff a checkpointed vs fully-stored reverse sweep.

"""
    UniformCheckpoint{T}

Uniform-interval checkpoint schedule holding states of element type `T`.

Each stored state is a `Vector{T}` snapshot of the flat solution at a
particular step index. The forward sweep calls [`add_checkpoint!`](@ref)
at every multiple of `interval` (plus the endpoints). The reverse sweep
calls [`nearest_checkpoint`](@ref) / [`restore_between`](@ref) to look up
the closest earlier snapshot and re-run forward to reach the requested
step.

# Fields
- `interval::Int` — number of forward steps between checkpoints
  (`interval == 1` ⇒ full storage).
- `checkpoints::Dict{Int, Vector{T}}` — step-index → state snapshot.
"""
struct UniformCheckpoint{T}
    interval::Int
    checkpoints::Dict{Int, Vector{T}}
end

"""
    UniformCheckpoint{T}(interval::Integer) where {T}

Construct an empty uniform checkpoint schedule with the given interval.
"""
function UniformCheckpoint{T}(interval::Integer) where {T}
    interval >= 1 || throw(ArgumentError("checkpoint interval must be >= 1"))
    return UniformCheckpoint{T}(Int(interval), Dict{Int, Vector{T}}())
end

"""
    should_checkpoint(schedule, step) -> Bool

Return `true` when step `step` lies on the uniform checkpoint grid. The
endpoints (`step == 0` and the terminal step) are always checkpointed by
the caller regardless of the interval — this helper only covers the
interior grid.
"""
function should_checkpoint(schedule::UniformCheckpoint, step::Integer)
    return step % schedule.interval == 0
end

"""
    add_checkpoint!(schedule, step, state) -> schedule

Store a copy of `state` at step index `step`. Overwrites any existing
snapshot at the same index.
"""
function add_checkpoint!(schedule::UniformCheckpoint{T}, step::Integer, state::AbstractVector) where {T}
    schedule.checkpoints[Int(step)] = collect(T.(state))
    return schedule
end

"""
    nearest_checkpoint(schedule, step) -> (step_idx, state)

Return the nearest stored checkpoint at or before `step`. Throws if no
earlier checkpoint exists (step 0 is always checkpointed by the caller).
"""
function nearest_checkpoint(schedule::UniformCheckpoint{T}, step::Integer) where {T}
    stored = sort!(collect(keys(schedule.checkpoints)))
    candidates = filter(s -> s <= step, stored)
    isempty(candidates) && throw(
        ArgumentError(
            "no checkpoint stored at or before step $step; ensure step 0 is seeded",
        )
    )
    step_idx = last(candidates)
    return (step_idx, copy(schedule.checkpoints[step_idx]))
end

"""
    restore_between(schedule, target_step, forward_step_fn) -> state

Reconstruct the forward state at `target_step` by locating the nearest
earlier checkpoint and re-running `forward_step_fn(state, k)` for each
intermediate step `k = anchor+1, anchor+2, …, target_step`. The callback
receives the *about-to-be-produced* step index `k` so that time-varying
forcing can be indexed consistently with the forward sweep.

No recomputation occurs when the target is itself a checkpoint; the
stored snapshot is returned directly.
"""
function restore_between(schedule::UniformCheckpoint, target_step::Integer, forward_step_fn::Function)
    anchor, state = nearest_checkpoint(schedule, target_step)
    for k in (anchor + 1):target_step
        state = forward_step_fn(state, k)
    end
    return state
end
