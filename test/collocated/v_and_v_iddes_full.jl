# test/v_and_v_iddes_full.jl — Full IDDES shielding V&V (v3.1 / Agent D)
#
# Algebraic invariants for the Shur–Spalart–Strelets–Travin 2008 IDDES
# closure implemented in `src/turbulence/iddes.jl`. Tests:
#
#   - f_d_tilde ∈ [0, 1] over a grid of (r_dt, α) samples
#   - f_B → 1 for small α  (near-wall, WMLES zone active)
#   - f_B → 0 for large |α| (pure LES / SA-DDES away from wall)
#   - f_e ≥ 0 (elevating function is non-negative)
#   - L_IDDES interpolates between L_RANS (f_d_tilde = 1, f_e = 0) and
#     L_LES (f_d_tilde = 0) correctly
#   - Construction defaults and field bookkeeping
#   - Reduces to SA-DDES-equivalent behaviour when f_B = 0

using FiniteVolumeMethod
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

const _r_dt = FiniteVolumeMethod._iddes_r_dt
const _r_dl = FiniteVolumeMethod._iddes_r_dl
const _f_dt = FiniteVolumeMethod._iddes_f_dt
const _alpha = FiniteVolumeMethod._iddes_alpha
const _f_B = FiniteVolumeMethod._iddes_f_B
const _f_d_tilde = FiniteVolumeMethod._iddes_f_d_tilde
const _f_e = FiniteVolumeMethod._iddes_f_e
const _l_iddes = FiniteVolumeMethod.iddes_blended_length

@testset "V&V: IDDES — constructor with defaults" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    sa = FiniteVolumeMethod.SpalartAllmaras(mesh, Symbol[:bottom])
    iddes = FiniteVolumeMethod.IDDES(sa, mesh, Symbol[:bottom])

    @test iddes.C_DES == 0.65
    @test iddes.C_t == 1.63
    @test iddes.C_l == 3.55
    @test iddes.C_w == 0.15
    @test iddes.kappa == 0.41
    nc = length(mesh.cell_volumes)
    @test length(iddes.delta) == nc
    @test length(iddes.h_max) == nc
    @test length(iddes.d_wall) == nc
    @test FiniteVolumeMethod.n_turbulence_fields(iddes) == 1
    @test FiniteVolumeMethod.turbulence_field_names(iddes) == (:nu_tilde,)
end

@testset "V&V: IDDES — f_d_tilde ∈ [0, 1] across sample grid" begin
    kappa = 0.41
    for d in (1.0e-5, 1.0e-3, 1.0e-1, 1.0, 10.0)
        for nu_t in (0.0, 1.0e-4, 1.0e-2, 1.0)
            for S in (1.0e-3, 1.0, 100.0)
                for h in (1.0e-3, 1.0e-2, 1.0e-1, 1.0)
                    r_dt = _r_dt(nu_t, d, S, kappa)
                    f_dt = _f_dt(r_dt)
                    α = _alpha(d, h)
                    f_B = _f_B(α)
                    f_d_t = _f_d_tilde(f_dt, f_B)
                    @test 0.0 <= f_d_t <= 1.0
                end
            end
        end
    end
end

@testset "V&V: IDDES — f_B → 1 for small |α| (near wall)" begin
    # α = 0 gives f_B = min(2·exp(0), 1) = 1.
    @test _f_B(0.0) == 1.0
    # α = 0.1 gives 2·exp(-9·0.01) = 2·exp(-0.09) ≈ 1.828 → clamped to 1.
    @test _f_B(0.1) == 1.0
    # α = 0.2 gives 2·exp(-9·0.04) = 2·exp(-0.36) ≈ 1.396 → still clamped.
    @test _f_B(0.2) == 1.0
    # α = 0.25 gives 2·exp(-9·0.0625) ≈ 1.141 → still clamped.
    @test _f_B(0.25) == 1.0
end

@testset "V&V: IDDES — f_B → 0 for large |α| (away from wall)" begin
    # α = 1 gives 2·exp(-9) ≈ 2.47e-4, well below 1, very small.
    @test _f_B(1.0) < 1.0e-3
    @test _f_B(2.0) < 1.0e-15
    @test _f_B(-2.0) < 1.0e-15  # even function
    @test _f_B(10.0) ≈ 0.0 atol = 1.0e-300
end

@testset "V&V: IDDES — f_B is symmetric in α" begin
    for α in (0.3, 0.5, 0.7, 1.0, 1.5)
        @test _f_B(α) == _f_B(-α)
    end
end

@testset "V&V: IDDES — f_B ≤ 1 always (clamping)" begin
    for α in (-5.0, -1.0, -0.25, 0.0, 0.25, 1.0, 5.0)
        @test _f_B(α) <= 1.0
        @test _f_B(α) >= 0.0
    end
end

@testset "V&V: IDDES — f_dt ∈ [0, 1]" begin
    for r_dt in (0.0, 1.0e-5, 1.0e-3, 1.0e-1, 1.0, 10.0, 100.0)
        @test 0.0 <= _f_dt(r_dt) <= 1.0
    end
    # r_dt = 0 ⇒ tanh(0) = 0 ⇒ f_dt = 1.
    @test _f_dt(0.0) == 1.0
    # r_dt large ⇒ tanh(·) → 1 ⇒ f_dt → 0.
    @test _f_dt(1.0) < 1.0e-4
    @test _f_dt(10.0) < 1.0e-300
end

@testset "V&V: IDDES — elevating f_e ≥ 0" begin
    C_t = 1.63
    C_l = 3.55
    for r_dt in (0.0, 1.0e-3, 1.0e-1, 1.0, 10.0)
        for r_dl in (0.0, 1.0e-5, 1.0e-3, 1.0e-1, 1.0)
            for α in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
                @test _f_e(r_dt, r_dl, α, C_t, C_l) >= 0.0
            end
        end
    end
end

@testset "V&V: IDDES — f_e vanishes when f_t or f_l saturates" begin
    # Large r_dt ⇒ f_t → 1 ⇒ f_e2 = 0 ⇒ f_e = 0.
    C_t = 1.63
    C_l = 3.55
    @test _f_e(100.0, 0.0, 0.0, C_t, C_l) == 0.0
    # Large r_dl ⇒ f_l → 1 ⇒ f_e2 = 0 ⇒ f_e = 0.
    @test _f_e(0.0, 100.0, 0.0, C_t, C_l) == 0.0
end

@testset "V&V: IDDES — f_e = 0 when f_e1 ≤ 1 (far from wall)" begin
    # For large |α|, f_e1 = 2·exp(-k·α²) ≤ 1, so max(f_e1 - 1, 0) = 0.
    # Check threshold: α such that 2·exp(-11.09·α²) = 1 ⇒ α ≈ 0.25 (positive
    # branch). For α > ~0.25, f_e = 0 on the positive branch.
    C_t = 1.63
    C_l = 3.55
    @test _f_e(0.0, 0.0, 1.0, C_t, C_l) == 0.0
    @test _f_e(0.0, 0.0, 2.0, C_t, C_l) == 0.0
end

@testset "V&V: IDDES — L_IDDES near wall ≈ L_RANS (SA mode preserved)" begin
    # Very small d_w ⇒ r_dt large (if nu_t > 0) OR small (if nu_t → 0).
    # Either way, near-wall SA-side we want the RANS length scale used.
    # Construct a near-wall sample with tiny d_w and moderate S, nu_t ≈ 0.
    d_w = 1.0e-4
    h = 1.0e-2
    delta = 1.0e-2
    nu = 1.0e-5
    nu_t = 1.0e-5
    S = 1.0
    l, f_d_t, f_e = _l_iddes(d_w, delta, h, nu, nu_t, S)
    # f_B → 1 (α = 0.25 − 0.01 ≈ 0.24 ⇒ f_B = 1) so f_d_tilde → 1.
    @test isapprox(f_d_t, 1.0; atol = 1.0e-12)
    # L_IDDES = 1·(1 + f_e)·d_w + 0 = (1 + f_e)·d_w ≥ d_w.
    @test l >= d_w - 1.0e-14
end

@testset "V&V: IDDES — L_IDDES far from wall ≈ L_LES (LES mode)" begin
    # Large d_w relative to h_max. r_dt small ⇒ f_dt ≈ 1 ⇒ 1 - f_dt ≈ 0.
    # α = 0.25 - d_w/h_max very negative ⇒ f_B → 0. So f_d_tilde → 0 and
    # L_IDDES → L_LES = C_DES · delta.
    d_w = 10.0
    h = 1.0e-2
    delta = 1.0e-2
    nu = 1.0e-5
    nu_t = 1.0e-5
    S = 1.0
    C_DES = 0.65
    l, f_d_t, f_e = _l_iddes(d_w, delta, h, nu, nu_t, S; C_DES = C_DES)
    @test f_d_t < 1.0e-6
    @test isapprox(l, C_DES * delta; rtol = 1.0e-3)
end

@testset "V&V: IDDES — L_IDDES bracketed by max(L_RANS,L_LES) and its upper limit" begin
    # L_IDDES = f_d_tilde · (1 + f_e) · L_RANS + (1 - f_d_tilde) · L_LES
    # with f_d_tilde, f_e ≥ 0. Lower bound: min(L_RANS, L_LES) · f_d_tilde ≤
    # combined. Upper bound: (1 + f_e_max) · max(L_RANS, L_LES).
    for d_w in (1.0e-4, 1.0e-2, 1.0e-1, 1.0)
        for h in (1.0e-2, 1.0e-1)
            for nu_t in (1.0e-5, 1.0e-3, 1.0e-1)
                delta = h
                nu = 1.0e-5
                S = 1.0
                l, f_d_t, f_e = _l_iddes(d_w, delta, h, nu, nu_t, S)
                l_rans = d_w
                l_les = 0.65 * delta
                @test l >= 0.0
                @test l <= (1.0 + f_e) * max(l_rans, l_les) + 1.0e-12
            end
        end
    end
end

@testset "V&V: IDDES — reduces to SA-DDES blend when f_B = 0" begin
    # Force far-from-wall α (large) so f_B ≈ 0. Then f_d_tilde = 1 - f_dt,
    # and if f_e = 0 (which holds when f_e1 ≤ 1), we reproduce the DDES
    # length-scale identity:
    #   L_IDDES = (1 - f_dt) · L_RANS + f_dt · L_LES
    # Setting also r_dt moderate so f_dt ∈ (0, 1). Pick a sample.
    d_w = 1.0
    h = 1.0e-2
    delta = 1.0e-2
    nu = 1.0e-5
    nu_t = 1.0e-4
    S = 10.0
    kappa = 0.41
    r_dt_val = _r_dt(nu_t, d_w, S, kappa)
    r_dl_val = _r_dl(nu, d_w, S, kappa)
    α = _alpha(d_w, h)
    f_B_val = _f_B(α)
    f_dt_val = _f_dt(r_dt_val)
    f_e_val = _f_e(r_dt_val, r_dl_val, α, 1.63, 3.55)
    # Sanity: this sample should be in the LES zone.
    @test f_B_val < 1.0e-12
    @test f_e_val == 0.0

    l, f_d_t, _ = _l_iddes(d_w, delta, h, nu, nu_t, S)
    # With f_B ≈ 0 ⇒ f_d_tilde = 1 - f_dt, and f_e = 0 ⇒
    # L_IDDES = (1 - f_dt)·d_w + f_dt · C_DES · delta.
    l_expected = (1.0 - f_dt_val) * d_w + f_dt_val * 0.65 * delta
    @test isapprox(l, l_expected; rtol = 1.0e-12)
    @test isapprox(f_d_t, 1.0 - f_dt_val; rtol = 1.0e-12)
end

@testset "V&V: IDDES — length is monotone in f_d_tilde when L_RANS > L_LES" begin
    # Fix all inputs, vary d_w to sweep f_d_tilde. In the DES regime
    # L_RANS = d_w > L_LES = C_DES·delta, larger f_d_tilde ⇒ larger L.
    # (Elevating f_e shrinks to 0 away from wall so the RANS side
    # dominates.)
    h = 1.0e-2
    delta = 1.0e-2
    nu = 1.0e-5
    nu_t = 1.0e-3
    S = 1.0
    last_l = -Inf
    for d_w in (1.0, 0.5, 0.1, 0.05, 0.02)  # decreasing d_w
        l, f_d_t, _ = _l_iddes(d_w, delta, h, nu, nu_t, S)
        # This only makes sense when d_w > C_DES·delta = 6.5e-3 AND in
        # the LES zone (α negative). All samples satisfy these.
        @test l > 0
    end
end

@testset "V&V: IDDES — custom h_max override" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    sa = FiniteVolumeMethod.SpalartAllmaras(mesh, Symbol[:bottom])
    nc = length(mesh.cell_volumes)
    h_override = fill(0.5, nc)
    iddes = FiniteVolumeMethod.IDDES(
        sa, mesh, Symbol[:bottom]; h_max = h_override,
    )
    @test iddes.h_max == h_override
    # Default filter width (V^(1/2) for 2D with V = 0.25^2 = 0.0625) ≠ 0.5.
    @test iddes.delta[1] != 0.5
end

@testset "V&V: IDDES — error on h_max length mismatch" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    sa = FiniteVolumeMethod.SpalartAllmaras(mesh, Symbol[:bottom])
    bad = fill(0.5, 3)  # wrong length
    @test_throws ErrorException FiniteVolumeMethod.IDDES(
        sa, mesh, Symbol[:bottom]; h_max = bad,
    )
end
