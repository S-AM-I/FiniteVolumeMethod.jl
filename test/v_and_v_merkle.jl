# test/v_and_v_merkle.jl — Merkle (1998) cavitation rate algebra V&V.
#
# Algebraic invariants:
#   1. p = p_sat          ⇒ both branches are zero.
#   2. p < p_sat          ⇒ destruction (negative m_dot_vap).
#   3. p > p_sat          ⇒ production  (positive m_dot_cond).
#   4. α_v linear scaling on each branch.
#   5. C_dest, C_prod linear scaling.

using FiniteVolumeMethod
using Test

const MM = FiniteVolumeMethod.MerkleModel
const merkle_vap = FiniteVolumeMethod.merkle_vap_rate
const merkle_cond = FiniteVolumeMethod.merkle_cond_rate
const merkle_net = FiniteVolumeMethod.merkle_rate

include("TestHelpers.jl")

@testset "V&V: Merkle — p = p_sat zeroes both branches" begin
    m = MM(; C_dest = 1.0, C_prod = 80.0, U_inf = 2.0, t_inf = 0.3)
    rho_l, rho_v, p_sat = 1000.0, 1.0, 2300.0
    for alpha_v in (0.0, 0.2, 0.5, 0.8, 1.0)
        @test merkle_vap(m, p_sat, alpha_v, rho_l, p_sat) == 0.0
        @test merkle_cond(m, p_sat, alpha_v, rho_v, rho_l, p_sat) == 0.0
        @test merkle_net(m, p_sat, alpha_v, rho_l, rho_v, p_sat) == 0.0
    end
end

@testset "V&V: Merkle — p < p_sat ⇒ destruction (m_vap < 0)" begin
    m = MM(; C_dest = 1.0, C_prod = 80.0, U_inf = 1.0, t_inf = 1.0)
    rho_l, rho_v, p_sat = 1000.0, 1.0, 2300.0
    p = 1800.0
    for alpha_v in (0.0, 0.2, 0.5, 0.9)
        m_vap = merkle_vap(m, p, alpha_v, rho_l, p_sat)
        m_cond = merkle_cond(m, p, alpha_v, rho_v, rho_l, p_sat)
        @test m_cond == 0.0
        if alpha_v < 1.0
            @test m_vap < 0.0
        else
            @test m_vap == 0.0
        end
    end
end

@testset "V&V: Merkle — p > p_sat ⇒ production (m_cond > 0)" begin
    m = MM(; C_dest = 1.0, C_prod = 80.0, U_inf = 1.0, t_inf = 1.0)
    rho_l, rho_v, p_sat = 1000.0, 1.0, 2300.0
    p = 3000.0
    for alpha_v in (0.1, 0.3, 0.5, 0.7, 0.9)
        m_vap = merkle_vap(m, p, alpha_v, rho_l, p_sat)
        m_cond = merkle_cond(m, p, alpha_v, rho_v, rho_l, p_sat)
        @test m_vap == 0.0
        @test m_cond > 0.0
    end
end

@testset "V&V: Merkle — linear scaling in α_v on each branch" begin
    m = MM(; C_dest = 1.0, C_prod = 80.0, U_inf = 2.0, t_inf = 0.5)
    rho_l, rho_v, p_sat = 1000.0, 1.0, 2300.0

    # Destruction: m_vap ∝ (1 − α_v).
    p = 1500.0
    v_03 = merkle_vap(m, p, 0.3, rho_l, p_sat)
    v_06 = merkle_vap(m, p, 0.6, rho_l, p_sat)
    @test v_06 ≈ v_03 * ((1 - 0.6) / (1 - 0.3)) rtol = 1.0e-12

    # Production: m_cond ∝ α_v.
    p = 3000.0
    c_03 = merkle_cond(m, p, 0.3, rho_v, rho_l, p_sat)
    c_06 = merkle_cond(m, p, 0.6, rho_v, rho_l, p_sat)
    @test c_06 ≈ c_03 * (0.6 / 0.3) rtol = 1.0e-12
end

@testset "V&V: Merkle — linear scaling in C_dest and C_prod" begin
    base = MM(; C_dest = 1.0, C_prod = 80.0, U_inf = 1.5, t_inf = 0.2)
    dbl_d = MM(; C_dest = 2.0, C_prod = 80.0, U_inf = 1.5, t_inf = 0.2)
    dbl_p = MM(; C_dest = 1.0, C_prod = 160.0, U_inf = 1.5, t_inf = 0.2)
    rho_l, rho_v, p_sat = 1000.0, 1.0, 2300.0

    p = 1800.0
    alpha_v = 0.3
    @test merkle_vap(dbl_d, p, alpha_v, rho_l, p_sat) ≈
        2 * merkle_vap(base, p, alpha_v, rho_l, p_sat) rtol = 1.0e-12

    p = 3000.0
    @test merkle_cond(dbl_p, p, alpha_v, rho_v, rho_l, p_sat) ≈
        2 * merkle_cond(base, p, alpha_v, rho_v, rho_l, p_sat) rtol = 1.0e-12
end

@testset "V&V: Merkle — closed-form match at a sample" begin
    m = MM(; C_dest = 1.2, C_prod = 90.0, U_inf = 2.5, t_inf = 0.4)
    rho_l, rho_v, p_sat = 998.0, 0.023, 2300.0
    ref = 0.5 * rho_l * m.U_inf^2 * m.t_inf

    # Destruction branch.
    p, alpha_v = 1700.0, 0.25
    expected_vap = m.C_dest * rho_l * (1 - alpha_v) * min(0.0, p - p_sat) / ref
    @test merkle_vap(m, p, alpha_v, rho_l, p_sat) ≈ expected_vap rtol = 1.0e-12

    # Production branch.
    p, alpha_v = 3200.0, 0.45
    expected_cond = m.C_prod * rho_v * alpha_v * max(0.0, p - p_sat) / ref
    @test merkle_cond(m, p, alpha_v, rho_v, rho_l, p_sat) ≈ expected_cond rtol = 1.0e-12
end

@testset "V&V: Merkle — compute_vapor_source per-cell array" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    m = MM(; C_dest = 1.0, C_prod = 80.0, U_inf = 1.0, t_inf = 1.0)
    props = FiniteVolumeMethod.CavitationProperties(;
        rho_l = 1000.0, rho_v = 0.02, p_sat = 2300.0,
    )
    p = fill(3000.0, nc)
    p[1] = 1500.0
    alpha_v = fill(0.3, nc)
    src = FiniteVolumeMethod.compute_vapor_source(m, p, alpha_v, mesh, props)
    @test length(src) == nc
    @test src[1] > 0
    @test src[2] < 0
end
