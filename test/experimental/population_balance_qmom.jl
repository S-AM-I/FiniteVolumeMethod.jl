# test/experimental/population_balance_qmom.jl — quadrature method of moments.

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: qmom_moment_source_growth, qmom_recover_abscissae_weights
using Test

@testset "QMoM recovers exact abscissae/weights from bi-disperse moments" begin
    # A bi-disperse distribution: n_1 at L_1, n_2 at L_2 gives
    # m_k = n_1 · L_1^k + n_2 · L_2^k.
    L_true = [1.0, 3.0]
    w_true = [0.4, 0.6]
    moments = [sum(w_true .* (L_true .^ k)) for k in 0:3]  # 4 moments → N=2.
    abscissae, weights = qmom_recover_abscissae_weights(moments, 2)

    # QMoM recovers the exact underlying distribution for a 2N-moment-
    # realizable bi-disperse input.
    @test length(abscissae) == 2
    @test length(weights) == 2
    @test sort(abscissae) ≈ sort(L_true) atol = 1.0e-8
    # Weights permuted by the sort — compare as sets.
    @test Set(round.(weights; digits = 8)) == Set(round.(w_true; digits = 8))
    # m_0 conservation.
    @test sum(weights) ≈ moments[1] atol = 1.0e-10
end

@testset "QMoM growth moment source matches analytical" begin
    # Constant growth rate G(L) = g. Analytical: dm_k/dt = k·g·m_{k-1}.
    # With N=2, pick symmetric bi-disperse distribution.
    weights = [0.3, 0.7]
    abscissae = [1.0, 3.0]
    g = 2.0

    m1_analytical = sum(weights .* abscissae)  # = 0.3·1 + 0.7·3 = 2.4
    m1_growth = qmom_moment_source_growth(weights, abscissae, L -> g, 2)
    # dm_2/dt = 2·g·m_1 = 2·2·2.4 = 9.6
    @test m1_growth ≈ 9.6 atol = 1.0e-10

    # dm_0/dt from growth = 0 · g · m_{-1} = 0 (growth alone doesn't change total count)
    m0_growth = qmom_moment_source_growth(weights, abscissae, L -> g, 1)
    # Note: k=1 → dm_1/dt = 1·g·m_0 = g·(0.3+0.7) = 2.0
    @test m0_growth ≈ 2.0 atol = 1.0e-10
end
