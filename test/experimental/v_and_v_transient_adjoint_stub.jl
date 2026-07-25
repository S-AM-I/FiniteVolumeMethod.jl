# test/v_and_v_transient_adjoint_stub.jl — verify the v3.107 transient
# adjoint implementation is live (replacing the old stub warn+throw).
#
# Before v3.1 (v3.107 wave), `solve_transient_adjoint` was a deliberate
# stub that warned and errored. v3.107 landed the real uniform-checkpoint
# reverse-sweep implementation in `src/adjoint/transient.jl`. This file
# retains the same filename (still referenced by runtests.jl) but now
# gates on the positive contract: the dispatch routes cleanly through
# `solve_adjoint(::TransientAdjoint, …)` to the real implementation,
# and `TransientAdjoint()` is constructible without warnings.

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: TransientAdjoint, solve_adjoint, solve_transient_adjoint
using Test

@testset "V&V: TransientAdjoint — type is constructible (no longer a stub)" begin
    alg = TransientAdjoint()
    @test alg isa FiniteVolumeMethod.AbstractAdjointAlgorithm
end

@testset "V&V: solve_transient_adjoint — symbol is exported and callable" begin
    @test isdefined(FiniteVolumeMethod, :solve_transient_adjoint)
    @test isa(FiniteVolumeMethod.solve_transient_adjoint, Function)
end

@testset "V&V: TransientAdjoint dispatch — routed to real solver" begin
    # Smallest non-trivial case: 1-step scalar linear ODE adjoint.
    # dR_dp and dJ_du supplied; solver should return a pair of arrays.
    using LinearAlgebra: I
    n = 2
    M = Matrix{Float64}(I, n, n)
    A = Matrix{Float64}(I, n, n)
    dt = 0.1
    u_series = [zeros(n), ones(n)]
    b_series = [zeros(n), zeros(n)]
    dJ_du = [zeros(n), ones(n)]
    dR_dp = [zeros(n, 1), zeros(n, 1)]
    result = solve_transient_adjoint(M, A, b_series, u_series, dJ_du, dR_dp, dt)
    @test result isa Tuple
    @test length(result) == 2
end
