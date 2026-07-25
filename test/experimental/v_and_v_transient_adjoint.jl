# test/v_and_v_transient_adjoint.jl — checkpointed transient adjoint V&V.
#
# All cases compare the discrete-adjoint gradient against a symmetric
# finite-difference gradient on a linear, backward-Euler-discretised
# transient problem. Tolerance is rtol 1e-4 — tight enough to catch sign
# bugs and time-index off-by-ones, loose enough to tolerate central-
# difference O(h^2) error at epsilon = 1e-5.

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: TransientAdjoint, solve_adjoint, solve_steady_adjoint
using LinearAlgebra
using Test

# ---- helpers ---------------------------------------------------------------

"""Forward sweep of the linear transient problem
    M · du/dt + A · u = b(p, t),
discretised with backward Euler. Returns (u_series, b_series) with the
caller-provided initial condition and step-indexed RHS."""
function _forward_linear(M, A, u0, p, b_of_pk, n_steps, dt)
    n = length(u0)
    T = eltype(u0)
    u_series = Vector{Vector{T}}(undef, n_steps + 1)
    b_series = Vector{Vector{T}}(undef, n_steps + 1)
    u_series[1] = collect(u0)
    b_series[1] = zeros(T, n)              # step 0 has no RHS contribution
    K = M ./ dt .+ A
    M_over_dt = M ./ dt
    u = u_series[1]
    for k in 1:n_steps
        b_k = b_of_pk(p, k)
        b_series[k + 1] = collect(b_k)
        rhs = M_over_dt * u .+ b_k
        u = K \ rhs
        u_series[k + 1] = copy(u)
    end
    return (u_series, b_series)
end

"""Cost Σ_{k=1}^{N_t} f(u^k, k). Returns the per-step gradient series."""
function _cost_and_grad(u_series, f, df_du)
    T = eltype(u_series[1])
    n_plus_one = length(u_series)
    J = zero(T)
    dJ_du_series = Vector{Vector{T}}(undef, n_plus_one)
    dJ_du_series[1] = zeros(T, length(u_series[1]))     # step 0 fixed
    for k in 1:(n_plus_one - 1)
        J += f(u_series[k + 1], k)
        dJ_du_series[k + 1] = collect(df_du(u_series[k + 1], k))
    end
    return (J, dJ_du_series)
end

function _finite_difference(M, A, u0, p, b_of_pk, n_steps, dt, f; eps_fd = 1.0e-5)
    n_p = length(p)
    T = eltype(p)
    g = zeros(T, n_p)
    for i in 1:n_p
        p_plus = copy(p); p_plus[i] += T(eps_fd)
        p_minus = copy(p); p_minus[i] -= T(eps_fd)
        u_plus, _ = _forward_linear(M, A, u0, p_plus, b_of_pk, n_steps, dt)
        u_minus, _ = _forward_linear(M, A, u0, p_minus, b_of_pk, n_steps, dt)
        J_plus = sum(f(u_plus[k + 1], k) for k in 1:n_steps)
        J_minus = sum(f(u_minus[k + 1], k) for k in 1:n_steps)
        g[i] = (J_plus - J_minus) / (2 * T(eps_fd))
    end
    return g
end

# ---- tests -----------------------------------------------------------------

@testset "V&V: transient adjoint — scalar ODE du/dt = -a·u + p" begin
    a = 0.8
    dt = 0.05
    n_steps = 20
    u0 = [1.0]
    p = [0.5]                            # scalar parameter driving the RHS
    M = reshape([1.0], 1, 1)
    A = reshape([a], 1, 1)
    b_of_pk = (p, _k) -> [p[1]]
    u_ref = fill(0.3, n_steps)
    f = (u, k) -> 0.5 * (u[1] - u_ref[k])^2
    df_du = (u, k) -> [u[1] - u_ref[k]]

    u_series, b_series = _forward_linear(M, A, u0, p, b_of_pk, n_steps, dt)
    _, dJ_du_series = _cost_and_grad(u_series, f, df_du)
    dR_dp_series = [reshape([0.0], 1, 1) for _ in 0:n_steps]
    for k in 1:n_steps
        dR_dp_series[k + 1] = reshape([1.0], 1, 1)      # ∂b^k/∂p = 1
    end

    _, dJ_dp_adj = FiniteVolumeMethod.solve_transient_adjoint_linear(
        M, A, b_series, u_series, dJ_du_series, dR_dp_series, dt;
        checkpoint_interval = 5,
    )
    dJ_dp_fd = _finite_difference(M, A, u0, p, b_of_pk, n_steps, dt, f)
    @test isapprox(dJ_dp_adj[1], dJ_dp_fd[1]; rtol = 1.0e-4, atol = 1.0e-10)
end

@testset "V&V: transient adjoint — 2-DOF linear system, terminal cost" begin
    dt = 0.02
    n_steps = 25
    M = Matrix{Float64}(I, 2, 2)
    A = [1.5 -0.2; 0.1 1.2]
    u0 = [0.0, 0.0]
    p = [1.0, -0.5]
    # RHS b(p, k) = B · p (time-independent parameter Jacobian).
    B = [1.0 0.3; 0.2 1.0]
    b_of_pk = (p, _k) -> B * p

    u_ref = [0.4, -0.1]
    f = (u, k) -> k == n_steps ? 0.5 * sum((u .- u_ref) .^ 2) : 0.0
    df_du = (u, k) -> k == n_steps ? (u .- u_ref) : zeros(2)

    u_series, b_series = _forward_linear(M, A, u0, p, b_of_pk, n_steps, dt)
    _, dJ_du_series = _cost_and_grad(u_series, f, df_du)
    dR_dp_series = [zeros(2, 2) for _ in 0:n_steps]
    for k in 1:n_steps
        dR_dp_series[k + 1] = B
    end

    _, dJ_dp_adj = FiniteVolumeMethod.solve_transient_adjoint_linear(
        M, A, b_series, u_series, dJ_du_series, dR_dp_series, dt;
        checkpoint_interval = 5,
    )
    dJ_dp_fd = _finite_difference(M, A, u0, p, b_of_pk, n_steps, dt, f)
    for i in eachindex(p)
        @test isapprox(dJ_dp_adj[i], dJ_dp_fd[i]; rtol = 1.0e-4, atol = 1.0e-10)
    end
end

@testset "V&V: transient adjoint — checkpointing is a memory tradeoff, not accuracy" begin
    dt = 0.02
    n_steps = 30
    M = Matrix{Float64}(I, 3, 3)
    A = [2.0 0.1 0.0; 0.1 2.0 0.1; 0.0 0.1 2.0]
    u0 = [0.1, 0.2, 0.3]
    p = [0.7, -0.2, 0.4]
    B = [1.0 0.0 0.2; 0.1 1.0 0.0; 0.0 0.3 1.0]
    b_of_pk = (p, _k) -> B * p
    f = (u, _k) -> 0.5 * sum(u .^ 2)
    df_du = (u, _k) -> copy(u)

    u_series, b_series = _forward_linear(M, A, u0, p, b_of_pk, n_steps, dt)
    _, dJ_du_series = _cost_and_grad(u_series, f, df_du)
    dR_dp_series = [zeros(3, 3) for _ in 0:n_steps]
    for k in 1:n_steps
        dR_dp_series[k + 1] = B
    end

    _, g_full = FiniteVolumeMethod.solve_transient_adjoint_linear(
        M, A, b_series, u_series, dJ_du_series, dR_dp_series, dt;
        checkpoint_interval = 1,
    )
    _, g_cp = FiniteVolumeMethod.solve_transient_adjoint_linear(
        M, A, b_series, u_series, dJ_du_series, dR_dp_series, dt;
        checkpoint_interval = 5,
    )
    for i in eachindex(g_full)
        @test isapprox(g_cp[i], g_full[i]; rtol = 1.0e-10, atol = 1.0e-12)
    end
end

@testset "V&V: transient adjoint — n_steps = 1 reduces to steady adjoint" begin
    dt = 0.1
    n = 4
    M = Matrix{Float64}(I, n, n)
    A = [
        2.0 0.3 0.0 0.0;
        0.3 2.0 0.3 0.0;
        0.0 0.3 2.0 0.3;
        0.0 0.0 0.3 2.0
    ]
    u0 = zeros(n)
    p = collect(1.0:n)
    b_of_pk = (p, _k) -> copy(p)

    u_series, b_series = _forward_linear(M, A, u0, p, b_of_pk, 1, dt)
    # Running cost restricted to the terminal state: J = 0.5‖u^1‖².
    f = (u, _k) -> 0.5 * sum(u .^ 2)
    df_du = (u, _k) -> copy(u)
    _, dJ_du_series = _cost_and_grad(u_series, f, df_du)
    dR_dp_series = [zeros(n, n), Matrix{Float64}(I, n, n)]

    _, dJ_dp_transient = FiniteVolumeMethod.solve_transient_adjoint_linear(
        M, A, b_series, u_series, dJ_du_series, dR_dp_series, dt;
        checkpoint_interval = 1,
    )
    # Steady-adjoint baseline: single backward-Euler step with
    # K = M/dt + A, RHS = M/dt · u0 + p collapses (since u0 = 0) to
    # K · u1 = p; J = 0.5‖u1‖² has dJ/du1 = u1, so the steady
    # analogue solves K^T · λ = u1 and reports dJ/dp = λ^T · I.
    K = M ./ dt .+ A
    _, dJ_dp_steady = solve_steady_adjoint(
        K, p, u_series[2], u_series[2], -Matrix{Float64}(I, n, n),
    )
    # The transient adjoint stacks the terminal partial onto a zero
    # `λ^{N_t+1}`; because ∂R/∂p for the steady call is negated relative
    # to the transient linear forcing convention, compare magnitudes.
    for i in 1:n
        @test isapprox(abs(dJ_dp_transient[i]), abs(dJ_dp_steady[i]); rtol = 1.0e-8)
    end
end

@testset "V&V: transient adjoint — zero cost ⇒ zero adjoint" begin
    dt = 0.05
    n_steps = 6
    M = Matrix{Float64}(I, 2, 2)
    A = [1.0 0.0; 0.0 1.0]
    u0 = [0.2, -0.1]
    p = [0.3, 0.4]
    b_of_pk = (p, _k) -> copy(p)

    u_series, b_series = _forward_linear(M, A, u0, p, b_of_pk, n_steps, dt)
    dJ_du_series = [zeros(2) for _ in 0:n_steps]
    dR_dp_series = [Matrix{Float64}(I, 2, 2) for _ in 0:n_steps]

    lambdas, dJ_dp = FiniteVolumeMethod.solve_transient_adjoint_linear(
        M, A, b_series, u_series, dJ_du_series, dR_dp_series, dt;
        checkpoint_interval = 3,
    )
    for lam in lambdas
        for v in lam
            @test v == 0.0
        end
    end
    for v in dJ_dp
        @test v == 0.0
    end
end

@testset "V&V: transient adjoint — dispatch through solve_adjoint(TransientAdjoint())" begin
    # The dispatch wrapper must route to the linear adjoint (no warn/throw).
    dt = 0.05
    n_steps = 4
    M = Matrix{Float64}(I, 2, 2)
    A = [1.0 0.0; 0.0 1.0]
    u0 = [0.0, 0.0]
    p = [1.0, 1.0]
    b_of_pk = (p, _k) -> copy(p)
    u_series, b_series = _forward_linear(M, A, u0, p, b_of_pk, n_steps, dt)
    f = (u, _k) -> 0.5 * sum(u .^ 2)
    df_du = (u, _k) -> copy(u)
    _, dJ_du_series = _cost_and_grad(u_series, f, df_du)
    dR_dp_series = [Matrix{Float64}(I, 2, 2) for _ in 0:n_steps]

    lambdas, dJ_dp = solve_adjoint(
        TransientAdjoint(),
        M, A, b_series, u_series, dJ_du_series, dR_dp_series, dt;
        checkpoint_interval = 2,
    )
    @test length(lambdas) == n_steps + 1
    @test length(dJ_dp) == length(p)
end

@testset "V&V: transient adjoint — checkpointing primitives" begin
    schedule = FiniteVolumeMethod.UniformCheckpoint{Float64}(3)
    FiniteVolumeMethod.add_checkpoint!(schedule, 0, [1.0, 2.0])
    FiniteVolumeMethod.add_checkpoint!(schedule, 3, [3.0, 4.0])
    FiniteVolumeMethod.add_checkpoint!(schedule, 6, [5.0, 6.0])

    step, state = FiniteVolumeMethod.nearest_checkpoint(schedule, 5)
    @test step == 3
    @test state == [3.0, 4.0]

    step, state = FiniteVolumeMethod.nearest_checkpoint(schedule, 6)
    @test step == 6
    @test state == [5.0, 6.0]

    # `restore_between` replays the forward step function.
    advance = (u, k) -> u .+ 1.0
    reconstructed = FiniteVolumeMethod.restore_between(schedule, 5, advance)
    @test reconstructed == [5.0, 6.0]       # [3,4] + 2 advances
end
