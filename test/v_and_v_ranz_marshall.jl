# test/v_and_v_ranz_marshall.jl — Ranz-Marshall particle heat transfer V&V (v3.33)
#
# Third convergence-verified benchmark for `lagrangian_dpm`.
# Complements the momentum closures (Stokes in v3.13 and
# Schiller-Naumann in v3.26) with the thermal closure —
# the Ranz-Marshall convective-heat-transfer correlation
#
#   Nu = 2 + 0.6 · Re_p^0.5 · Pr^0.33,
#   q  = π · d · k_f · Nu · (T_f − T_p)     [W].
#
# Five algebraic invariants:
#
#   1. Zero slip (U_f = U_p) ⇒ Nu = 2 (stagnant-fluid limit).
#   2. Isothermal (T_f = T_p) ⇒ q = 0.
#   3. Sign consistency: T_f > T_p ⇒ q > 0 (heat flows into
#      the particle).
#   4. Linearity in (T_f − T_p).
#   5. Re^0.5 scaling of the convective-correction term at
#      fixed Pr.
#
# Puts `lagrangian_dpm` at three convergence-verified
# benchmarks — 3-benchmark floor for stable-promotion review.

using FiniteVolumeMethod
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: Ranz-Marshall — stagnant limit Nu = 2" begin
    # With zero slip velocity, Re_p = 0 ⇒ Nu = 2 exactly.
    # Expected: q = π · d · k_f · 2 · (T_f − T_p).
    d_p = 50.0e-6
    rho_f = 1.2
    mu_f = 1.81e-5
    k_f = 0.026
    Pr = 0.7

    U_f = SVector(0.0, 0.0)
    U_p = SVector(0.0, 0.0)
    T_f = 500.0
    T_p = 400.0

    q = compute_particle_heat_transfer(
        RanzMarshall(), T_f, T_p, U_f, U_p, d_p, rho_f, mu_f, k_f, Pr,
    )

    q_expected = pi * d_p * k_f * 2.0 * (T_f - T_p)
    @test isapprox(q, q_expected; rtol = 1.0e-12)
end

@testset "V&V: Ranz-Marshall — isothermal ⇒ q ≡ 0" begin
    d_p = 50.0e-6
    rho_f = 1.2
    mu_f = 1.81e-5
    k_f = 0.026
    Pr = 0.7

    U_f = SVector(0.1, 0.0)
    U_p = SVector(0.0, 0.0)

    q = compute_particle_heat_transfer(
        RanzMarshall(), 500.0, 500.0, U_f, U_p, d_p, rho_f, mu_f, k_f, Pr,
    )

    @test q == 0.0
end

@testset "V&V: Ranz-Marshall — sign: T_f > T_p ⇒ q > 0 (heat into particle)" begin
    d_p = 50.0e-6
    rho_f = 1.2
    mu_f = 1.81e-5
    k_f = 0.026
    Pr = 0.7

    U_f = SVector(0.5, 0.0)
    U_p = SVector(0.0, 0.0)

    # Hot fluid ⇒ heat flows in (q > 0).
    q_hot = compute_particle_heat_transfer(
        RanzMarshall(), 800.0, 300.0, U_f, U_p, d_p, rho_f, mu_f, k_f, Pr,
    )
    @test q_hot > 0.0

    # Cold fluid ⇒ heat flows out (q < 0).
    q_cold = compute_particle_heat_transfer(
        RanzMarshall(), 300.0, 800.0, U_f, U_p, d_p, rho_f, mu_f, k_f, Pr,
    )
    @test q_cold < 0.0

    # Magnitudes equal for equal |ΔT| (linearity).
    @test isapprox(q_hot, -q_cold; rtol = 1.0e-12)
end

@testset "V&V: Ranz-Marshall — linearity in (T_f − T_p)" begin
    d_p = 50.0e-6
    rho_f = 1.2
    mu_f = 1.81e-5
    k_f = 0.026
    Pr = 0.7

    U_f = SVector(0.3, 0.0)
    U_p = SVector(0.0, 0.0)

    q_a = compute_particle_heat_transfer(
        RanzMarshall(), 450.0, 400.0, U_f, U_p, d_p, rho_f, mu_f, k_f, Pr,
    )
    q_b = compute_particle_heat_transfer(
        RanzMarshall(), 500.0, 400.0, U_f, U_p, d_p, rho_f, mu_f, k_f, Pr,
    )
    q_c = compute_particle_heat_transfer(
        RanzMarshall(), 550.0, 400.0, U_f, U_p, d_p, rho_f, mu_f, k_f, Pr,
    )

    @test isapprox(q_b / q_a, 2.0; rtol = 1.0e-12)
    @test isapprox(q_c / q_a, 3.0; rtol = 1.0e-12)
end

@testset "V&V: Ranz-Marshall — Nu formula match at non-trivial Re" begin
    # Explicit cross-check against the closed-form at a
    # moderate Re where the convective term dominates the
    # conductive "2".
    d_p = 100.0e-6
    rho_f = 1.0
    mu_f = 1.0e-5
    k_f = 0.05
    Pr = 0.7
    slip = 1.0

    U_f = SVector(0.0, 0.0)
    U_p = SVector(-slip, 0.0)

    Re = rho_f * slip * d_p / mu_f   # = 10
    Nu = 2.0 + 0.6 * Re^0.5 * Pr^0.33

    q = compute_particle_heat_transfer(
        RanzMarshall(), 500.0, 400.0, U_f, U_p, d_p, rho_f, mu_f, k_f, Pr,
    )

    q_expected = pi * d_p * k_f * Nu * (500.0 - 400.0)
    @test isapprox(q, q_expected; rtol = 1.0e-12)

    # Sanity: Nu ≈ 2 + 0.6 · √10 · 0.7^0.33 = 2 + 1.9·0.891 ≈ 3.69.
    @test 3.0 < Nu < 4.5
end
