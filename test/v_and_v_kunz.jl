# test/v_and_v_kunz.jl — Kunz (2000) cavitation rate algebra V&V.
#
# The Kunz model has two branches; the raw vaporisation and
# condensation rates have closed-form expressions that we test here.
# The net `kunz_rate` returned by `compute_vapor_source` is the
# combination with the sign convention described in `src/cavitation/kunz.jl`.
#
# Algebraic invariants:
#   1. p = p_sat  ⇒ both branches are zero.
#   2. p < p_sat  ⇒ only vaporisation contributes (m_vap < 0, m_cond = 0).
#   3. p > p_sat  ⇒ only condensation contributes (m_cond > 0, m_vap = 0).
#   4. α_v = 0    ⇒ m_cond = 0.
#   5. α_v = 1    ⇒ m_cond = 0 (α_v²·(1−α_v) vanishes).
#   6. C_v and C_c scale each branch linearly.
#   7. Closed-form match at three (p, α_v) sample points.

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_vapor_source
using Test

const K = FiniteVolumeMethod.KunzModel
const kunz_vap = FiniteVolumeMethod.kunz_vap_rate
const kunz_cond = FiniteVolumeMethod.kunz_cond_rate
const kunz_net = FiniteVolumeMethod.kunz_rate

include("TestHelpers.jl")

@testset "V&V: Kunz — p = p_sat zeroes both branches" begin
    m = K(; C_v = 100.0, C_c = 100.0, U_inf = 2.0, L_inf = 0.5)
    rho_l, rho_v = 1000.0, 1.0
    p_sat = 2000.0
    p = p_sat
    for alpha_v in (0.0, 0.1, 0.5, 0.9, 1.0)
        @test kunz_vap(m, p, alpha_v, rho_l, p_sat) == 0.0
        @test kunz_cond(m, p, alpha_v, rho_v, p_sat) == 0.0
        @test kunz_net(m, p, alpha_v, rho_l, rho_v, p_sat) == 0.0
    end
end

@testset "V&V: Kunz — p < p_sat ⇒ only vaporisation branch" begin
    m = K(; C_v = 50.0, C_c = 50.0, U_inf = 1.0, L_inf = 1.0)
    rho_l, rho_v = 1000.0, 1.0
    p_sat = 2000.0
    p = 1500.0
    for alpha_v in (0.0, 0.2, 0.5, 0.8)
        m_vap = kunz_vap(m, p, alpha_v, rho_l, p_sat)
        m_cond = kunz_cond(m, p, alpha_v, rho_v, p_sat)
        @test m_cond == 0.0
        if alpha_v < 1.0
            @test m_vap < 0.0      # liquid fraction present ⇒ negative rate
        else
            @test m_vap == 0.0
        end
    end
end

@testset "V&V: Kunz — p > p_sat ⇒ only condensation branch" begin
    m = K(; C_v = 50.0, C_c = 50.0, U_inf = 1.0, L_inf = 1.0)
    rho_l, rho_v = 1000.0, 1.0
    p_sat = 2000.0
    p = 2500.0
    for alpha_v in (0.1, 0.25, 0.5, 0.75, 0.9)
        m_vap = kunz_vap(m, p, alpha_v, rho_l, p_sat)
        m_cond = kunz_cond(m, p, alpha_v, rho_v, p_sat)
        @test m_vap == 0.0
        @test m_cond > 0.0
    end
end

@testset "V&V: Kunz — α_v = 0 and α_v = 1 zero the condensation branch" begin
    m = K(; C_v = 50.0, C_c = 50.0, U_inf = 1.0, L_inf = 1.0)
    p_sat = 2000.0
    p = 2500.0
    @test kunz_cond(m, p, 0.0, 1.0, p_sat) == 0.0
    @test kunz_cond(m, p, 1.0, 1.0, p_sat) == 0.0
end

@testset "V&V: Kunz — linear scaling in C_v and C_c" begin
    base = K(; C_v = 100.0, C_c = 100.0, U_inf = 1.5, L_inf = 0.25)
    dbl_v = K(; C_v = 200.0, C_c = 100.0, U_inf = 1.5, L_inf = 0.25)
    dbl_c = K(; C_v = 100.0, C_c = 200.0, U_inf = 1.5, L_inf = 0.25)
    rho_l, rho_v, p_sat = 1000.0, 1.0, 2000.0

    # C_v linear scaling on the vaporisation branch
    p = 1500.0
    alpha_v = 0.3
    @test kunz_vap(dbl_v, p, alpha_v, rho_l, p_sat) ≈
        2 * kunz_vap(base, p, alpha_v, rho_l, p_sat) rtol = 1.0e-12

    # C_c linear scaling on the condensation branch
    p = 2500.0
    @test kunz_cond(dbl_c, p, alpha_v, rho_v, p_sat) ≈
        2 * kunz_cond(base, p, alpha_v, rho_v, p_sat) rtol = 1.0e-12
end

@testset "V&V: Kunz — closed-form match at three samples" begin
    m = K(; C_v = 120.0, C_c = 80.0, U_inf = 3.0, L_inf = 0.6)
    rho_l, rho_v, p_sat = 1000.0, 0.05, 2300.0

    tau = m.L_inf / m.U_inf
    ref_vap = 0.5 * rho_l * m.U_inf^2

    # Sample 1: p < p_sat, α_v = 0.2 ⇒ vaporisation only.
    p1, a1 = 1800.0, 0.2
    expected_vap1 = m.C_v * rho_l * (1 - a1) * min(0.0, p1 - p_sat) / ref_vap / tau
    @test kunz_vap(m, p1, a1, rho_l, p_sat) ≈ expected_vap1 rtol = 1.0e-12
    @test kunz_cond(m, p1, a1, rho_v, p_sat) == 0.0

    # Sample 2: p > p_sat, α_v = 0.3 ⇒ condensation only.
    p2, a2 = 3000.0, 0.3
    expected_cond2 = m.C_c * rho_v * a2^2 * (1 - a2) / tau
    @test kunz_cond(m, p2, a2, rho_v, p_sat) ≈ expected_cond2 rtol = 1.0e-12
    @test kunz_vap(m, p2, a2, rho_l, p_sat) == 0.0

    # Sample 3: p = p_sat ⇒ both zero.
    p3, a3 = 2300.0, 0.4
    @test kunz_vap(m, p3, a3, rho_l, p_sat) == 0.0
    @test kunz_cond(m, p3, a3, rho_v, p_sat) == 0.0
end

@testset "V&V: Kunz — compute_vapor_source produces per-cell array" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    m = K(; C_v = 100.0, C_c = 100.0, U_inf = 1.0, L_inf = 1.0)
    props = FiniteVolumeMethod.CavitationProperties(;
        rho_l = 1000.0, rho_v = 1.0, p_sat = 2000.0,
    )
    p = fill(2500.0, nc)          # uniform ⇒ condensation
    p[1] = 1500.0                 # one cell vaporising
    alpha_v = fill(0.3, nc)
    src = FiniteVolumeMethod.compute_vapor_source(m, p, alpha_v, mesh, props)
    @test length(src) == nc
    @test src[1] > 0              # vapour produced at cell 1
    @test src[2] < 0              # vapour destroyed elsewhere
end
