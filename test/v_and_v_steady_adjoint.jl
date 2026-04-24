# test/v_and_v_steady_adjoint.jl — steady-adjoint identity V&V.

using FiniteVolumeMethod
using LinearAlgebra
using Test

@testset "V&V: steady adjoint — A = I, J = c·u recovers λ = c" begin
    n = 8
    A = Matrix{Float64}(I, n, n)
    b = collect(1.0:n)
    u = A \ b
    c = collect(0.1:0.1:(0.1 * n))
    dJ_du = c
    # If A ≡ I and b ≡ p, then R = u − p and ∂R/∂p = −I. Skip total-gradient
    # check here; focus on λ == c.
    dR_dp = -Matrix{Float64}(I, n, n)
    lambda, _ = solve_steady_adjoint(A, b, u, dJ_du, dR_dp)
    for i in 1:n
        @test isapprox(lambda[i], c[i]; rtol = 1.0e-12)
    end
end

@testset "V&V: steady adjoint — symmetric A ⇒ λ matches primal with swapped RHS" begin
    n = 6
    A = [
        2.0 1.0 0.0 0.0 0.0 0.0;
        1.0 2.0 1.0 0.0 0.0 0.0;
        0.0 1.0 2.0 1.0 0.0 0.0;
        0.0 0.0 1.0 2.0 1.0 0.0;
        0.0 0.0 0.0 1.0 2.0 1.0;
        0.0 0.0 0.0 0.0 1.0 2.0
    ]
    b = collect(1.0:n)
    u = A \ b
    c = collect(0.1:0.1:0.6)
    u_primal_c = A \ c
    dR_dp = -Matrix{Float64}(I, n, n)
    lambda, _ = solve_steady_adjoint(A, b, u, c, dR_dp)
    for i in 1:n
        @test isapprox(lambda[i], u_primal_c[i]; rtol = 1.0e-10)
    end
end

@testset "V&V: steady adjoint — zero cost ⇒ zero λ" begin
    n = 4
    A = [
        3.0 0.5 0.0 0.0;
        0.5 3.0 0.5 0.0;
        0.0 0.5 3.0 0.5;
        0.0 0.0 0.5 3.0
    ]
    b = [1.0, 2.0, 3.0, 4.0]
    u = A \ b
    dJ_du = zeros(n)
    dR_dp = Matrix{Float64}(I, n, n)
    lambda, dJ_dp = solve_steady_adjoint(A, b, u, dJ_du, dR_dp)
    for v in lambda
        @test v == 0.0
    end
    for v in dJ_dp
        @test v == 0.0
    end
end

@testset "V&V: steady adjoint — gradient matches finite-difference" begin
    # Parameterised RHS b(p) = p; state A·u = p; J(u) = 0.5·‖u − u_ref‖².
    n = 4
    A = [
        2.0 0.3 0.0 0.0;
        0.3 2.0 0.3 0.0;
        0.0 0.3 2.0 0.3;
        0.0 0.0 0.3 2.0
    ]
    u_ref = [0.5, 0.5, 0.5, 0.5]
    solve_forward = p -> A \ p
    J = (u, p) -> 0.5 * sum((u .- u_ref) .^ 2)
    p = [1.0, 2.0, 3.0, 4.0]
    u = solve_forward(p)
    dJ_du = u .- u_ref
    dR_dp = -Matrix{Float64}(I, n, n)    # ∂(A·u − p)/∂p = −I
    lambda, dJ_dp_adj = solve_steady_adjoint(A, p, u, dJ_du, dR_dp)
    dJ_dp_fd = verify_adjoint_gradient(J, solve_forward, p; epsilon = 1.0e-6)
    for i in 1:n
        @test isapprox(dJ_dp_adj[i], dJ_dp_fd[i]; rtol = 1.0e-4, atol = 1.0e-8)
    end
end

@testset "V&V: steady adjoint — linearity in dJ_du" begin
    n = 3
    A = [2.0 0.5 0.0; 0.5 2.0 0.5; 0.0 0.5 2.0]
    b = [1.0, 2.0, 3.0]
    u = A \ b
    c1 = [1.0, 0.0, 0.0]
    c2 = [0.0, 1.0, 0.0]
    dR_dp = Matrix{Float64}(I, n, n)
    lam1, _ = solve_steady_adjoint(A, b, u, c1, dR_dp)
    lam2, _ = solve_steady_adjoint(A, b, u, c2, dR_dp)
    lam_sum, _ = solve_steady_adjoint(A, b, u, 2.0 .* c1 .+ 3.0 .* c2, dR_dp)
    for i in 1:n
        @test isapprox(lam_sum[i], 2.0 * lam1[i] + 3.0 * lam2[i]; rtol = 1.0e-12)
    end
end
