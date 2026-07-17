# adjoint/reverse_sweep.jl — Checkpointed reverse-time adjoint propagation.
#
# The reverse sweep is the generic scaffold that any transient discrete
# adjoint plugs into:
#
#   1. Run the forward integration, seeding an `UniformCheckpoint`
#      schedule at step 0, at every `interval` steps, and at the terminal
#      step. Record the final state (and, optionally, intermediate
#      snapshots) on the way.
#   2. Seed the terminal adjoint `λ^{N_t}` from the terminal partial
#      `∂J/∂u^{N_t}`.
#   3. Walk `n = N_t, N_t − 1, …, 1` in reverse. At each step, look up
#      `u^{n-1}` (nearest checkpoint + forward replay) and `u^n` (either
#      cached by the caller or regenerated) and invoke the caller's
#      `adjoint_step(λ_next, u_n, u_prev, n)` to produce `λ^{n-1}`.
#
# The scaffold is deliberately math-agnostic: it does not assume linear
# vs. nonlinear dynamics, single-field vs. coupled systems, or
# backward-Euler vs. Crank–Nicolson. The caller supplies the one-step
# forward and the one-step adjoint kernels.

"""
    reverse_sweep(forward_step, adjoint_step, u_init, p, n_steps, interval;
                  terminal_lambda = nothing,
                  accumulate = nothing) -> (lambdas, trajectory)

Generic checkpointed reverse sweep.

# Arguments
- `forward_step(u, k) -> u_next` — advance the state from step `k-1` to
  step `k`. Called during the forward sweep and during reverse-phase
  trajectory reconstruction.
- `adjoint_step(lambda_next, u_k, u_km1, k) -> lambda_prev` — propagate
  the adjoint from step `k` back to step `k-1`.
- `u_init` — initial state (step 0).
- `p` — design parameters, forwarded to `accumulate` (if supplied).
  Unused by the sweep itself.
- `n_steps` — number of forward steps `N_t`.
- `interval` — uniform checkpoint interval (≥ 1).
- `terminal_lambda` — optional seed for `λ^{N_t}`. Defaults to zero.
- `accumulate(lambda_k, u_k, k, p)` — optional callback invoked for each
  reverse step with the freshly-produced `λ^{k}` (useful for running
  `dJ/dp += λ^T · ∂b/∂p|_k` without materialising all adjoints).

Returns `(lambdas, trajectory)`:
- `lambdas::Vector{Vector{T}}` — `lambdas[k+1] == λ^k` for
  `k = 0, …, N_t`.
- `trajectory::Vector{Vector{T}}` — forward states,
  `trajectory[k+1] == u^k`. Always full-length (the interval only
  controls how many of these are materialised *during* the reverse
  sweep; the forward pass records every state for the caller's
  convenience).
"""
function reverse_sweep(
        forward_step::Function,
        adjoint_step::Function,
        u_init::AbstractVector{T},
        p,
        n_steps::Integer,
        interval::Integer;
        terminal_lambda::Union{Nothing, AbstractVector} = nothing,
        accumulate::Union{Nothing, Function} = nothing,
    ) where {T}

    n_steps >= 0 || throw(ArgumentError("n_steps must be >= 0"))
    interval >= 1 || throw(ArgumentError("interval must be >= 1"))

    schedule = UniformCheckpoint{T}(interval)
    add_checkpoint!(schedule, 0, u_init)

    # Forward pass — record every state so the reverse walk can grab
    # u^{n-1} / u^n without replay when it is cheap to keep them.
    # Checkpoints still form the authoritative fallback when the caller
    # later requests `restore_between`.
    trajectory = Vector{Vector{T}}(undef, n_steps + 1)
    trajectory[1] = collect(T.(u_init))
    u = trajectory[1]
    for k in 1:n_steps
        u = forward_step(u, k)
        trajectory[k + 1] = collect(T.(u))
        if should_checkpoint(schedule, k) || k == n_steps
            add_checkpoint!(schedule, k, u)
        end
    end

    # Reverse pass.
    n_state = length(u_init)
    lambdas = Vector{Vector{T}}(undef, n_steps + 1)
    lambda = terminal_lambda === nothing ? zeros(T, n_state) : collect(T.(terminal_lambda))
    lambdas[n_steps + 1] = copy(lambda)

    if accumulate !== nothing
        accumulate(lambda, trajectory[n_steps + 1], n_steps, p)
    end

    for k in n_steps:-1:1
        u_k = trajectory[k + 1]
        # Use the recorded state when available; fall back to checkpoint
        # replay. Both branches produce the same value for a
        # deterministic `forward_step`; the replay path exercises the
        # checkpointing code for long trajectories where the caller may
        # eventually prefer to drop `trajectory` to save memory.
        u_km1 = restore_between(schedule, k - 1, forward_step)
        lambda = adjoint_step(lambda, u_k, u_km1, k)
        lambdas[k] = copy(lambda)
        if accumulate !== nothing
            accumulate(lambda, u_km1, k - 1, p)
        end
    end

    return (lambdas, trajectory)
end
