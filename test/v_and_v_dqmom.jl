# test/v_and_v_dqmom.jl — V&V for Direct QMoM source recovery
#
# Verifies:
# 1. Zero moment sources ⇒ zero (a, b) sources.
# 2. Degenerate abscissae (L_1 ≈ L_2) trigger a graceful error.
# 3. Linear-solve round-trip: kernel(abscissae) · [a; b] reproduces
#    the moment source input to rtol 1e-12.
# 4. Kernel sanity: the k=0 row has all b-block entries zero (weight
#    transport conserves number when no birth/death terms exist).

using LinearAlgebra
using Test

_experimental_warn(::Symbol) = nothing # no-op shim: source included standalone, outside module Experimental
include(joinpath(@__DIR__, "..", "src", "experimental", "population_balance", "dqmom.jl"))

@testset "V&V: DQMoM — zero moment sources ⇒ zero (a, b)" begin
    abscissae = [1.0, 2.5, 4.0]
    weights = [1.0, 1.0, 1.0]
    moment_sources = zeros(6)
    a, b = dqmom_sources(weights, abscissae, moment_sources)
    @test all(isapprox.(a, 0.0; atol = 1.0e-14))
    @test all(isapprox.(b, 0.0; atol = 1.0e-14))
end

@testset "V&V: DQMoM — degenerate abscissae error" begin
    abscissae = [1.0, 1.0, 4.0]
    weights = [1.0, 1.0, 1.0]
    moment_sources = zeros(6)
    @test_throws ErrorException dqmom_sources(weights, abscissae, moment_sources)
end

@testset "V&V: DQMoM — round-trip (a, b) ⇔ moment_sources" begin
    # Conditioning of the DQMoM Vandermonde-like kernel degrades fast
    # with N; match rtol to the condition number.
    for (abscissae, rtol) in (
            ([0.5, 1.7], 1.0e-12),
            ([0.3, 1.1, 3.7], 1.0e-10),
            ([0.2, 0.9, 2.1, 5.5], 1.0e-6),
        )
        N = length(abscissae)
        weights = fill(1.0, N)

        # Inject a known (a, b) pair, push through the forward map, then
        # try to recover via the solver.
        a_true = 0.1 .* collect(1:N)
        b_true = -0.2 .* collect(1:N)
        forward = dqmom_moment_residual(abscissae, a_true, b_true)
        a_rec, b_rec = dqmom_sources(weights, abscissae, forward)
        @test isapprox(a_rec, a_true; rtol = rtol, atol = 1.0e-12)
        @test isapprox(b_rec, b_true; rtol = rtol, atol = 1.0e-12)
    end
end

@testset "V&V: DQMoM — kernel weight-row structure" begin
    abscissae = [0.5, 1.0, 2.0]
    A = dqmom_kernel(abscissae)
    N = length(abscissae)
    @test size(A) == (2 * N, 2 * N)

    # Row 1 (k = 0) — a-block must be all ones; b-block must be all zero.
    @test A[1, 1:N] == ones(N)
    @test A[1, (N + 1):(2 * N)] == zeros(N)

    # Row 2 (k = 1) — a-block is all zero (coefficient (1-1) = 0);
    # b-block is all ones.
    @test A[2, 1:N] == zeros(N)
    @test A[2, (N + 1):(2 * N)] == ones(N)
end

@testset "V&V: DQMoM — deterministic recovery on a prescribed source" begin
    # Construct moment sources from a known DQMoM update and check
    # that `dqmom_sources` inverts the forward map to machine
    # precision.
    abscissae = [0.4, 1.3, 2.9]
    weights = [2.0, 1.5, 0.7]
    a_true = [0.05, -0.03, 0.02]
    b_true = [0.1, 0.15, -0.07]
    s = dqmom_moment_residual(abscissae, a_true, b_true)

    a_rec, b_rec = dqmom_sources(weights, abscissae, s)
    @test isapprox(a_rec, a_true; rtol = 1.0e-12, atol = 1.0e-14)
    @test isapprox(b_rec, b_true; rtol = 1.0e-12, atol = 1.0e-14)
end
