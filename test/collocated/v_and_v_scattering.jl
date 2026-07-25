# test/v_and_v_scattering.jl — fvDOM scattering source algebra V&V
#
# Verifies the in-scattering source term added to `solve_fvdom_radiation`
# in Wave 2. The scattering contribution per direction Ω̂_m at cell c is
#
#   S_sc(m, c) = (σ_s / 4π) · Σ_{m'} w_{m'} · Φ(Ω̂_m, Ω̂_{m'}) · I_{m'}(c)
#
# with phase function Φ selected by `FvDOMModel.scattering_phase`:
#   :isotropic            → Φ = 1
#   :linear_anisotropic   → Φ = 1 + g · (Ω̂ · Ω̂')
#
# Invariants checked:
#   1. σ_s = 0 ⇒ S_sc = 0 everywhere (backwards compatibility)
#   2. Isotropic phase: S_sc(m, c) = σ_s · G(c) / (4π) for any m, with
#      G(c) = Σ_{m'} w_{m'} · I_{m'}(c)
#   3. Linear-anisotropic phase with g = 0 matches isotropic
#   4. Σ_{m'} w_{m'} = 4π for the fvDOM quadrature used (3D)
#   5. Per-cell scalar σ_s handled identically to scalar σ_s
#   6. FvDOMModel defaults: sigma_s = 0, scattering_phase = :isotropic
#   7. Unknown phase function errors

using FiniteVolumeMethod
using FiniteVolumeMethod: solve_fvdom_radiation
using LinearAlgebra: dot, norm
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

const _scat_contrib = FiniteVolumeMethod.scattering_source_contribution
const _phase_val = FiniteVolumeMethod.scattering_phase_value

# --------------------------------------------------------------------
# Default-state invariants
# --------------------------------------------------------------------

@testset "V&V: scattering — default FvDOMModel disables scattering" begin
    m = FvDOMModel()
    @test m.sigma_s == 0.0
    @test m.scattering_phase === :isotropic
    @test m.scattering_g == 0.0
end

@testset "V&V: scattering — σ_s = 0 zeroes the in-scattering source" begin
    m = FvDOMModel(; sigma_s = 0.0, Dim = 3, order = :S2)
    n_dirs = length(m.directions)
    nc = 4
    # Non-trivial intensity field to distinguish source = 0 from I = 0.
    I_prev = rand(n_dirs, nc)
    for c in 1:nc, k in 1:n_dirs
        @test _scat_contrib(m, I_prev, k, c) == 0.0
    end
end

# --------------------------------------------------------------------
# Isotropic-phase algebra
# --------------------------------------------------------------------

@testset "V&V: scattering — isotropic Σ w = 4π (3D)" begin
    # All SN sets we add in Wave 2 normalize to Σ w = 4π in 3D. The
    # pre-existing 2D S4 quadrature uses a different convention
    # (Σ w = 8π/3) that's exercised in `v_and_v_fvdom_quadrature.jl`;
    # here we focus on the 3D invariant used by the in-scattering
    # source term.
    for order in (:S2, :S4, :S6, :S8)
        m3 = FvDOMModel(; sigma_s = 1.0, Dim = 3, order = order)
        @test isapprox(sum(m3.weights), 4 * pi; rtol = 1.0e-12)
    end
    # New SN sets (S6/S8/S12) are already 2π-normalized in 2D.
    for order in (:S6, :S8, :S12)
        m2 = FvDOMModel(; sigma_s = 1.0, Dim = 2, order = order)
        @test isapprox(sum(m2.weights), 2 * pi; rtol = 1.0e-12)
    end
end

@testset "V&V: scattering — isotropic S_sc = σ_s · G / (4π)" begin
    m = FvDOMModel(; sigma_s = 0.7, Dim = 3, order = :S4)
    n_dirs = length(m.directions)
    nc = 3
    # Build an arbitrary positive intensity field.
    I_prev = reshape(collect(0.1:0.1:((n_dirs * nc) * 0.1)), n_dirs, nc)

    # Analytic expectation: S_sc(m, c) = σ_s · G(c) / (4π), independent of m.
    for c in 1:nc
        G_c = sum(m.weights[mp] * I_prev[mp, c] for mp in 1:n_dirs)
        expected = 0.7 * G_c / (4 * pi)
        # Should be identical for every direction m under isotropic Φ.
        for m_idx in 1:n_dirs
            got = _scat_contrib(m, I_prev, m_idx, c)
            @test isapprox(got, expected; rtol = 1.0e-12)
        end
    end
end

# --------------------------------------------------------------------
# Linear-anisotropic phase algebra
# --------------------------------------------------------------------

@testset "V&V: scattering — linear-anisotropic with g = 0 reduces to isotropic" begin
    m_iso = FvDOMModel(;
        sigma_s = 0.5, Dim = 3, order = :S4,
        scattering_phase = :isotropic,
    )
    m_la_g0 = FvDOMModel(;
        sigma_s = 0.5, Dim = 3, order = :S4,
        scattering_phase = :linear_anisotropic, scattering_g = 0.0,
    )
    n_dirs = length(m_iso.directions)
    nc = 2
    I_prev = rand(n_dirs, nc)

    for c in 1:nc, k in 1:n_dirs
        s_iso = _scat_contrib(m_iso, I_prev, k, c)
        s_la = _scat_contrib(m_la_g0, I_prev, k, c)
        @test isapprox(s_iso, s_la; rtol = 1.0e-12)
    end
end

@testset "V&V: scattering — phase value is 1 (isotropic) and 1+g·cosθ (linear)" begin
    m_iso = FvDOMModel(; sigma_s = 0.1, Dim = 3, order = :S2)
    for k in 1:length(m_iso.directions), kp in 1:length(m_iso.directions)
        @test _phase_val(m_iso, k, kp) == 1.0
    end
    g = 0.4
    m_la = FvDOMModel(;
        sigma_s = 0.1, Dim = 3, order = :S2,
        scattering_phase = :linear_anisotropic, scattering_g = g,
    )
    for k in 1:length(m_la.directions), kp in 1:length(m_la.directions)
        cos_theta = dot(m_la.directions[k], m_la.directions[kp])
        expected = max(1.0 + g * cos_theta, 0.0)
        @test isapprox(_phase_val(m_la, k, kp), expected; rtol = 1.0e-12)
    end
end

@testset "V&V: scattering — linear-anisotropic phase clamped non-negative" begin
    # g > 1 would make Φ(Ω, -Ω) = 1 - g < 0; code clamps to zero.
    m = FvDOMModel(;
        sigma_s = 0.1, Dim = 3, order = :S2,
        scattering_phase = :linear_anisotropic, scattering_g = 1.5,
    )
    # Find Ω̂, -Ω̂ pair by octant symmetry.
    dirs = m.directions
    n = length(dirs)
    found = false
    for k in 1:n, kp in 1:n
        if isapprox(dirs[k], -dirs[kp]; rtol = 1.0e-12)
            @test _phase_val(m, k, kp) == 0.0
            found = true
            break
        end
    end
    @test found
end

# --------------------------------------------------------------------
# Per-cell scattering coefficient
# --------------------------------------------------------------------

@testset "V&V: scattering — per-cell σ_s matches scalar per-cell" begin
    sigma_s_vec = [0.3, 0.0, 0.7]
    m_vec = FvDOMModel(;
        sigma_s = sigma_s_vec, Dim = 3, order = :S2,
    )
    n_dirs = length(m_vec.directions)
    I_prev = ones(n_dirs, 3)

    # Cell 2 has σ_s = 0, so S_sc must be zero there.
    @test _scat_contrib(m_vec, I_prev, 1, 2) == 0.0

    # Cell 1 and 3 should match a scalar-σ_s model at the same σ_s.
    m_scalar_1 = FvDOMModel(; sigma_s = 0.3, Dim = 3, order = :S2)
    m_scalar_3 = FvDOMModel(; sigma_s = 0.7, Dim = 3, order = :S2)
    for k in 1:n_dirs
        @test isapprox(
            _scat_contrib(m_vec, I_prev, k, 1),
            _scat_contrib(m_scalar_1, I_prev, k, 1);
            rtol = 1.0e-12,
        )
        @test isapprox(
            _scat_contrib(m_vec, I_prev, k, 3),
            _scat_contrib(m_scalar_3, I_prev, k, 3);
            rtol = 1.0e-12,
        )
    end
end

# --------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------

@testset "V&V: scattering — unknown phase function errors" begin
    m = FvDOMModel(;
        sigma_s = 0.1, scattering_phase = :henyey_greenstein,
    )
    @test_throws ErrorException _phase_val(m, 1, 1)
end

@testset "V&V: scattering — I_prev = 0 ⇒ S_sc = 0 for any σ_s" begin
    m = FvDOMModel(; sigma_s = 1.0, Dim = 3, order = :S4)
    n_dirs = length(m.directions)
    I_prev = zeros(n_dirs, 2)
    for c in 1:2, k in 1:n_dirs
        @test _scat_contrib(m, I_prev, k, c) == 0.0
    end
end

@testset "V&V: scattering — model constructor backwards-compatible" begin
    # The legacy call pattern (no sigma_s keyword) must still work and
    # produce a model with sigma_s = 0 / isotropic phase.
    m_legacy = FvDOMModel(; a = 0.2, Dim = 2, order = :S2)
    @test m_legacy.sigma_s == 0.0
    @test m_legacy.scattering_phase === :isotropic
    # Per-direction in-scattering source must collapse to zero.
    I_prev = zeros(length(m_legacy.directions), 2)
    for k in 1:length(m_legacy.directions), c in 1:2
        @test _scat_contrib(m_legacy, I_prev, k, c) == 0.0
    end
end
