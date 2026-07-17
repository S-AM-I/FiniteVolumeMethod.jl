# test/v_and_v_schnerr_sauer.jl — Schnerr-Sauer (2001) cavitation V&V.
#
# Algebraic invariants:
#   1. p = p_sat    ⇒ m_dot = 0.
#   2. Monotone in (p_sat − p) at fixed α_v:
#        |m_dot| is strictly increasing as pressure moves farther from p_sat.
#   3. R_B closed-form: doubling n_0 ⇒ R_B · 2^(−1/3).
#   4. R_B increases with α_v (sanity of the bubble-volume scaling).
#   5. α_v = 0 or α_v = 1 ⇒ source vanishes.

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_vapor_source
using Test

const SSM = FiniteVolumeMethod.SchnerrSauerModel
const ss_rate = FiniteVolumeMethod.schnerr_sauer_rate
const ss_R = FiniteVolumeMethod.schnerr_sauer_bubble_radius

include("TestHelpers.jl")

@testset "V&V: Schnerr-Sauer — p = p_sat zeroes source" begin
    m = SSM(; n_0 = 1.0e13)
    rho_l, rho_v, p_sat = 1000.0, 1.0, 2300.0
    for alpha_v in (0.0, 0.1, 0.5, 0.9, 1.0)
        @test ss_rate(m, p_sat, alpha_v, rho_l, rho_v, p_sat) == 0.0
    end
end

@testset "V&V: Schnerr-Sauer — α_v = 0 or 1 zeroes source" begin
    m = SSM(; n_0 = 1.0e13)
    rho_l, rho_v, p_sat = 1000.0, 1.0, 2300.0
    for p in (1500.0, 3000.0)
        @test ss_rate(m, p, 0.0, rho_l, rho_v, p_sat) == 0.0
        @test ss_rate(m, p, 1.0, rho_l, rho_v, p_sat) == 0.0
    end
end

@testset "V&V: Schnerr-Sauer — sign convention (p < p_sat ⇒ vapour produced)" begin
    m = SSM(; n_0 = 1.0e13)
    rho_l, rho_v, p_sat = 1000.0, 1.0, 2300.0
    @test ss_rate(m, 1500.0, 0.3, rho_l, rho_v, p_sat) > 0
    @test ss_rate(m, 3000.0, 0.3, rho_l, rho_v, p_sat) < 0
end

@testset "V&V: Schnerr-Sauer — monotonicity in |p_sat − p|" begin
    m = SSM(; n_0 = 1.0e13)
    rho_l, rho_v, p_sat = 1000.0, 1.0, 2300.0
    alpha_v = 0.3
    # Evaporation branch: p < p_sat, |m_dot| grows as p decreases.
    s1 = abs(ss_rate(m, 2200.0, alpha_v, rho_l, rho_v, p_sat))
    s2 = abs(ss_rate(m, 2000.0, alpha_v, rho_l, rho_v, p_sat))
    s3 = abs(ss_rate(m, 1500.0, alpha_v, rho_l, rho_v, p_sat))
    @test s1 < s2 < s3
    # Condensation branch: p > p_sat, |m_dot| grows as p increases.
    c1 = abs(ss_rate(m, 2400.0, alpha_v, rho_l, rho_v, p_sat))
    c2 = abs(ss_rate(m, 2600.0, alpha_v, rho_l, rho_v, p_sat))
    c3 = abs(ss_rate(m, 3100.0, alpha_v, rho_l, rho_v, p_sat))
    @test c1 < c2 < c3
end

@testset "V&V: Schnerr-Sauer — R_B closed form and α_v monotonicity" begin
    m = SSM(; n_0 = 1.0e13)
    # Sample closed-form radius values.
    for alpha_v in (0.1, 0.3, 0.5, 0.7)
        expected = cbrt(3 * alpha_v / (4pi * m.n_0 * (1 - alpha_v)))
        @test ss_R(m, alpha_v) ≈ expected rtol = 1.0e-12
    end
    # Monotone in α_v at fixed n_0.
    @test ss_R(m, 0.1) < ss_R(m, 0.3) < ss_R(m, 0.5) < ss_R(m, 0.7)
end

@testset "V&V: Schnerr-Sauer — n_0 scaling (doubling ⇒ R_B · 2^(−1/3))" begin
    m1 = SSM(; n_0 = 1.0e13)
    m2 = SSM(; n_0 = 2.0e13)
    alpha_v = 0.4
    ratio = ss_R(m2, alpha_v) / ss_R(m1, alpha_v)
    @test ratio ≈ 2.0^(-1 / 3) rtol = 1.0e-12
end

@testset "V&V: Schnerr-Sauer — closed-form rate at a sample" begin
    m = SSM(; n_0 = 1.0e13)
    rho_l, rho_v, p_sat = 1000.0, 1.0, 2300.0
    p = 1500.0
    alpha_v = 0.4
    R_B = cbrt(3 * alpha_v / (4pi * m.n_0 * (1 - alpha_v)))
    dp = p_sat - p
    expected = (3 * rho_v * alpha_v * (1 - alpha_v) / R_B) *
        sign(dp) * sqrt(abs(dp) * 2 / (3 * rho_l))
    @test ss_rate(m, p, alpha_v, rho_l, rho_v, p_sat) ≈ expected rtol = 1.0e-12
end

@testset "V&V: Schnerr-Sauer — compute_vapor_source produces per-cell array" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    m = SSM(; n_0 = 1.0e13)
    props = FiniteVolumeMethod.CavitationProperties(;
        rho_l = 1000.0, rho_v = 1.0, p_sat = 2300.0,
    )
    p = fill(1500.0, nc)
    alpha_v = fill(0.3, nc)
    src = FiniteVolumeMethod.compute_vapor_source(m, p, alpha_v, mesh, props)
    @test length(src) == nc
    @test all(src .> 0)
end
