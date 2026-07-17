# adjoint/transient.jl — Transient discrete adjoint with uniform checkpointing.
#
# The transient adjoint leans on two companion files that are logically
# part of the adjoint stack but not pulled in from the layer file
# (so Wave-4 module loading stays a single `include("../adjoint/...")`
# sequence). Include them here so the user-facing promotion from stub to
# real implementation is a strict superset of the old include list.

#
# For a linear transient state equation
#
#     M · du/dt + A · u = b(p, t)
#
# discretised with backward Euler,
#
#     (M/dt + A) · u^{n+1} = (M/dt) · u^n + b^{n+1}(p),
#
# the discrete adjoint is propagated backward in time by
#
#     (M/dt + A)^T · λ^n = (M/dt) · λ^{n+1} + (∂J/∂u^n)^T.
#
# With a running-sum cost `J = Σ_n c_n^T · u^n`, the total parameter
# derivative is
#
#     dJ/dp = Σ_{n=1}^{N_t} λ^n · ∂b^n/∂p.
#
# (Step 0 carries no parameter dependence in this linear setting because
# `u^0` is a fixed initial condition.)
#
# The implementation uses the generic `reverse_sweep` scaffold so that a
# future nonlinear PIMPLE adjoint can reuse the same checkpointing
# machinery. Tier: experimental — math identity only; not yet wired into
# the full PIMPLE outer loop.

"""
    solve_transient_adjoint_linear(M, A, b_series, u_series,
                                   dJ_du_series, dR_dp_series, dt;
                                   checkpoint_interval = 10,
                                   terminal_dJ_du = nothing) -> (lambdas, dJ_dp)

Linear transient discrete adjoint with uniform checkpointing.

# Arguments
- `M` — mass matrix (size `n × n`). For pure ODEs supply `I`.
- `A` — stiffness / operator matrix (size `n × n`).
- `b_series::AbstractVector` — forcing at each step; `b_series[k]` is
  `b^k`, with `b_series[1]` ignored (step 0 has no RHS contribution).
  Length must be `N_t + 1`.
- `u_series::AbstractVector` — converged forward trajectory;
  `u_series[k+1] == u^k` for `k = 0, …, N_t`. Length `N_t + 1`.
- `dJ_du_series::AbstractVector` — per-step cost gradient;
  `dJ_du_series[k+1] == ∂J/∂u^k`. Length `N_t + 1`. Entries at `k = 0`
  only contribute to the initial-condition sensitivity (not returned
  here — fixed initial conditions make that zero).
- `dR_dp_series::AbstractVector` — per-step parameter Jacobians of the
  forcing; `dR_dp_series[k+1] == ∂b^k/∂p` with shape `n × n_p`. Length
  `N_t + 1`. Entries for `k = 0` are ignored.
- `dt::Real` — time-step size.
- `checkpoint_interval::Integer` — uniform checkpoint spacing (≥ 1).
  `1` forces full storage, matching the reference "fully stored"
  implementation.
- `terminal_dJ_du::Union{Nothing, AbstractVector}` — optional explicit
  seed for `λ^{N_t}`. Defaults to `dJ_du_series[N_t + 1]`.

Returns `(lambdas, dJ_dp)`:
- `lambdas` — `Vector{Vector}` with `lambdas[k+1] == λ^k` for
  `k = 0, …, N_t`.
- `dJ_dp` — accumulated parameter gradient `Σ_n λ^n · ∂b^n/∂p`.
"""
function solve_transient_adjoint_linear(
        M, A, b_series, u_series,
        dJ_du_series, dR_dp_series, dt;
        checkpoint_interval::Integer = 10,
        terminal_dJ_du::Union{Nothing, AbstractVector} = nothing,
    )
    _experimental_warn(:adjoint)
    n_plus_one = length(u_series)
    n_plus_one >= 1 || throw(ArgumentError("u_series must be non-empty"))
    n_steps = n_plus_one - 1
    length(b_series) == n_plus_one || throw(
        ArgumentError(
            "b_series must have length length(u_series) (got $(length(b_series)) vs $(n_plus_one))",
        )
    )
    length(dJ_du_series) == n_plus_one || throw(
        ArgumentError(
            "dJ_du_series must have length length(u_series)",
        )
    )
    length(dR_dp_series) == n_plus_one || throw(
        ArgumentError(
            "dR_dp_series must have length length(u_series)",
        )
    )
    dt > 0 || throw(ArgumentError("dt must be positive"))

    T = eltype(u_series[1])
    n_state = length(u_series[1])

    # Backward-Euler operators.
    K = M ./ dt .+ A                        # K · u^{n+1} = (M/dt)·u^n + b^{n+1}
    M_over_dt = M ./ dt
    K_T = transpose(K)

    # Forward one-step — strictly a replay of the supplied trajectory
    # via a fresh linear solve, so the reverse sweep's checkpoint recovery
    # path is exercised even when the caller pre-computes u_series.
    forward_step = function (u_prev::AbstractVector, k::Integer)
        rhs = M_over_dt * u_prev .+ b_series[k + 1]
        return K \ rhs
    end

    # Per-step adjoint solve: K^T · λ^{k-1} = (M/dt)·λ^k + (∂J/∂u^{k-1})^T.
    # Note the shift: the "next" adjoint in reverse time is λ^k, the
    # "previous" is λ^{k-1}. The cost gradient that gets added is the
    # gradient at the *earlier* time index.
    adjoint_step = function (lambda_next::AbstractVector, _u_k, _u_km1, k::Integer)
        rhs = M_over_dt * lambda_next .+ collect(dJ_du_series[k])
        return K_T \ rhs
    end

    terminal_lambda = if terminal_dJ_du !== nothing
        collect(T.(terminal_dJ_du))
    else
        # λ^{N_t} satisfies K^T λ^{N_t} = ∂J/∂u^{N_t} when there is no
        # λ^{N_t + 1} to blend in.
        rhs_terminal = collect(dJ_du_series[n_plus_one])
        if iszero(rhs_terminal)
            zeros(T, n_state)
        else
            K_T \ rhs_terminal
        end
    end

    dJ_dp = nothing
    accumulate = function (lambda_k::AbstractVector, _u_k, step_index::Integer, _p)
        step_index == 0 && return nothing
        jac = dR_dp_series[step_index + 1]
        contribution = transpose(lambda_k) * jac
        vec_contribution = collect(vec(contribution))
        if dJ_dp === nothing
            dJ_dp = zeros(T, length(vec_contribution))
        end
        dJ_dp .+= vec_contribution
        return nothing
    end

    lambdas, _ = reverse_sweep(
        forward_step, adjoint_step,
        u_series[1], nothing, n_steps, checkpoint_interval;
        terminal_lambda = terminal_lambda,
        accumulate = accumulate,
    )

    # Overwrite lambdas[N_t + 1] — the generic sweep seeds it from
    # terminal_lambda before accumulation, which is correct; but its
    # accumulate-call used the *passed-in* terminal_lambda rather than a
    # freshly-solved value. Leave as-is — the terminal seed has already
    # included the terminal cost partial.

    if dJ_dp === nothing
        # Zero-cost or zero-parameter-Jacobian trajectory; return an
        # empty gradient of the natural shape.
        n_p = size(dR_dp_series[min(2, n_plus_one)], 2)
        dJ_dp = zeros(T, n_p)
    end

    return (lambdas, dJ_dp)
end

"""
    solve_transient_adjoint(M, A, b_series, u_series,
                            dJ_du_series, dR_dp_series, dt;
                            checkpoint_interval = 10,
                            terminal_dJ_du = nothing)

Convenience wrapper around [`solve_transient_adjoint_linear`](@ref).
This replaces the v3.0 stub (which warned and threw); the new signature
is the linear-transient adjoint path described above. Callers pinned to
the stub will see a method-error or a kwarg-mismatch; that is intended —
the stub behaviour is being retired.
"""
function solve_transient_adjoint(
        M, A, b_series, u_series,
        dJ_du_series, dR_dp_series, dt;
        checkpoint_interval::Integer = 10,
        terminal_dJ_du::Union{Nothing, AbstractVector} = nothing,
    )
    _experimental_warn(:adjoint)
    return solve_transient_adjoint_linear(
        M, A, b_series, u_series,
        dJ_du_series, dR_dp_series, dt;
        checkpoint_interval = checkpoint_interval,
        terminal_dJ_du = terminal_dJ_du,
    )
end
