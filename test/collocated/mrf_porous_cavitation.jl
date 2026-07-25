# test/collocated/mrf_porous_cavitation.jl — rotating reference frames, porous
# media, and cavitation mass transfer.

using FiniteVolumeMethod
using FiniteVolumeMethod: DarcyForchheimerPorous, DarcyPorous, KunzCavitation, MerkleCavitation, RotationalMRFZone, SchnerrSauerCavitation, cavitation_source, mrf_momentum_source, porous_momentum_source
using Test
using StaticArrays: SVector

@testset "MRF rotational zone produces correct Coriolis/centrifugal" begin
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

@testset "Porous Darcy / Forchheimer momentum sinks" begin
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

@testset "Cavitation mass transfer responds to pressure" begin
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
