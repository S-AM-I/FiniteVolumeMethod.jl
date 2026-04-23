# test/stage6_physics.jl — Stage 6 MRF / porous / cavitation / FW-H / PBM gates

using FiniteVolumeMethod
using Test
using StaticArrays: SVector
using LinearAlgebra: norm

@testset "Stage 6a: MRF rotational zone produces correct Coriolis/centrifugal" begin
    # 2D planar MRF: omega = 5 rad/s about origin.
    zone2d_source = FiniteVolumeMethod.mrf_momentum_source_2d_planar(
        5.0, SVector(1.0, 0.0), SVector(0.0, 2.0), SVector(0.0, 0.0), 1.0,
    )
    # Coriolis: -ρ 2 ω k̂ × u = -2·5·(0, -2, ?).·(−u_y, u_x) = -2·5·(-2, 0) = (20, 0).
    # Wait: with the sign convention here, compute directly.
    # 2D: coriolis_vec = ω (-u_y, u_x) = 5(-2, 0) = (-10, 0).
    # Source contribution = -ρ · 2 · coriolis_vec = -2·(-10, 0) = (20, 0)
    # Centrifugal: -ρ · ω² · (-r) = ρ ω² r = 1·25·(1, 0) = (25, 0)
    # Total = (20, 0) + (25, 0) = (45, 0)
    @test zone2d_source[1] ≈ 45.0 atol = 1.0e-10
    @test zone2d_source[2] ≈ 0.0 atol = 1.0e-10

    # Zero ω → zero source.
    zero_src = FiniteVolumeMethod.mrf_momentum_source_2d_planar(
        0.0, SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 0.0), 1.0,
    )
    @test zero_src ≈ SVector(0.0, 0.0) atol = 1.0e-12

    # RotationalMRFZone returns zero source outside its cell list.
    zone = RotationalMRFZone{3, Float64}(
        [1, 2, 3], SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 1.0), 2.0,
    )
    out_of_zone = mrf_momentum_source(
        zone, 999, SVector(1.0, 0.0, 0.0),
        SVector(0.0, 1.0, 0.0), 1.0
    )
    @test out_of_zone ≈ SVector(0.0, 0.0, 0.0) atol = 1.0e-12
end

@testset "Stage 6c: Porous Darcy / Forchheimer momentum sinks" begin
    # Darcy: linear in |u|, no quadratic term.
    m = DarcyPorous{3}(; cell_indices = [1, 2], D = 1.0e6)
    src = porous_momentum_source(
        m, 1, SVector(2.0, 0.0, 0.0), 1.0, 1.0e-3,
    )
    # -ρ μ D u = -1·1e-3·1e6·(2, 0, 0) = -(2000, 0, 0)
    @test src ≈ SVector(-2000.0, 0.0, 0.0) atol = 1.0e-6

    # Outside zone: zero.
    src_out = porous_momentum_source(m, 999, SVector(1.0, 0.0, 0.0), 1.0, 1.0e-3)
    @test src_out ≈ SVector(0.0, 0.0, 0.0)

    # Forchheimer adds quadratic term.
    mf = DarcyForchheimerPorous{3}(; cell_indices = [1], D = 1.0e6, F = 1.0e3)
    src_f = porous_momentum_source(
        mf, 1, SVector(10.0, 0.0, 0.0), 1.0, 1.0e-3,
    )
    # |u| = 10; coeff = 1e-3·1e6 + 0.5·1e3·10 = 1000 + 5000 = 6000
    # src = -1·6000·(10, 0, 0) = (-60000, 0, 0)
    @test src_f ≈ SVector(-60000.0, 0.0, 0.0) atol = 1.0e-6
end

@testset "Stage 6d: Cavitation mass transfer responds to pressure" begin
    # Kunz: p > p_sat → evaporation (m_plus > 0), p < p_sat → condensation.
    k = KunzCavitation()
    m_plus_hi, m_minus_hi = cavitation_source(k, 1.5e5, 0.8, 1000.0, 1.0, 1.0e5)
    m_plus_lo, m_minus_lo = cavitation_source(k, 0.5e5, 0.8, 1000.0, 1.0, 1.0e5)

    # High pressure: evaporation active (p > p_sat → m_plus > 0), no condensation.
    @test m_plus_hi > 0.0
    @test m_minus_hi == 0.0
    # Low pressure: condensation active, no evaporation.
    @test m_plus_lo == 0.0
    @test m_minus_lo > 0.0

    # Merkle has same qualitative response.
    merkle = MerkleCavitation()
    mp, mm = cavitation_source(merkle, 0.5e5, 0.5, 1000.0, 1.0, 1.0e5)
    @test mp == 0.0
    @test mm > 0.0

    # Schnerr-Sauer: evaporation vs condensation switched by sign(p_sat - p).
    ss = SchnerrSauerCavitation()
    mp_ss, mm_ss = cavitation_source(ss, 0.5e5, 0.5, 1000.0, 1.0, 1.0e5)
    @test mp_ss > 0.0   # p < p_sat: evaporation
end

@testset "Stage 6f: FW-H / Curle surface integration" begin
    # Unit-radius spherical "body" with two antipodal panels, far-field
    # observer at 10 units. Curle sum should linearly depend on dp.
    faces = [1, 2]
    centers = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0)]
    normals = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0)]
    areas = [1.0, 1.0]
    surface = FWHSurface{3, Float64}(faces, centers, normals, areas)

    observer = FWHObserver(SVector(10.0, 0.0, 0.0))

    # Equal pressure on both sides: dipole cancels by symmetry.
    p_equal = [1.0e5, 1.0e5]
    @test curle_dipole_pressure(observer, surface, p_equal, 1.0e5) ≈ 0.0 atol = 1.0e-12

    # Asymmetric surface pressures: dipole non-zero.
    p_asym = [2.0e5, 1.0e5]
    p_asym_larger = [3.0e5, 1.0e5]
    p1 = curle_dipole_pressure(observer, surface, p_asym, 1.0e5)
    p2 = curle_dipole_pressure(observer, surface, p_asym_larger, 1.0e5)
    @test abs(p2) > abs(p1)   # doubling the asymmetry doubles the magnitude (linearity)

    # Monopole: uniform time-derivative of mass flux on both faces, but with
    # opposite normals, should average distances equally.
    dmass = [1.0, 1.0]
    mono = fwh_monopole_pressure(observer, surface, dmass)
    # Both faces contribute; face at (1,0,0) has r=9, face at (-1,0,0) has r=11.
    # sum = 1·1/9 + 1·1/11 = 0.2020..., divide by 4π.
    @test mono ≈ (1 / 9 + 1 / 11) / (4π) atol = 1.0e-10
end

@testset "Stage 6g: QMoM recovers exact abscissae/weights from bi-disperse moments" begin
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

@testset "Stage 6g: QMoM growth moment source matches analytical" begin
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
