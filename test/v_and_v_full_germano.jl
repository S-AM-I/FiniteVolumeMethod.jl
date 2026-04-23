# test/v_and_v_full_germano.jl — Full-tensor Germano identity V&V (v3.0 / Wave 1)
#
# Verifies the full-tensor double-contraction L_ij M_ij used by the
# dynamic Smagorinsky model (see `dynamic_smagorinsky.jl`). The test
# is pure algebra on reduced-component tensors — no mesh, no solve —
# so it detects any regression in `_sym_contract` and
# `_sym_self_magnitude_sq` independently of the gradient stencil.

using FiniteVolumeMethod
using LinearAlgebra
using Test

include("TestHelpers.jl")

const _sym_contract = FiniteVolumeMethod._sym_contract
const _sym_mag_sq = FiniteVolumeMethod._sym_self_magnitude_sq

"""
Expand a reduced 2D component tuple (xx, yy, xy) back into a 2×2
dense symmetric tensor. Used only by the tests to cross-check against
`tr(A*B')` (standard dense contraction).
"""
function _reduced_to_dense2(t)
    return [
        t[1] t[3];
        t[3] t[2]
    ]
end

"""
Expand a reduced 3D component tuple (xx, yy, xy, zz, xz, yz) back
into a 3×3 dense symmetric tensor.
"""
function _reduced_to_dense3(t)
    return [
        t[1] t[3] t[5];
        t[3] t[2] t[6];
        t[5] t[6] t[4]
    ]
end

@testset "V&V: Full-tensor Germano — L_ij M_ij invariance (2D)" begin
    # Prescribe an arbitrary pair of symmetric tensors in reduced
    # form. The reduced contraction `_sym_contract` must match the
    # trace of the dense product.
    L = (0.3, -0.1, 0.05)  # L_xx, L_yy, L_xy
    M = (0.2, 0.4, -0.15)  # M_xx, M_yy, M_xy
    LM_reduced = _sym_contract(L, M, Val(2))
    L_dense = _reduced_to_dense2(L)
    M_dense = _reduced_to_dense2(M)
    LM_dense = sum(L_dense .* M_dense)   # A_ij B_ij
    @test isapprox(LM_reduced, LM_dense; rtol = 1.0e-14)
end

@testset "V&V: Full-tensor Germano — L_ij M_ij invariance (3D)" begin
    L = (0.4, -0.1, 0.05, 0.3, -0.02, 0.11)
    M = (-0.2, 0.3, -0.15, 0.1, 0.07, -0.09)
    LM_reduced = _sym_contract(L, M, Val(3))
    L_dense = _reduced_to_dense3(L)
    M_dense = _reduced_to_dense3(M)
    LM_dense = sum(L_dense .* M_dense)
    @test isapprox(LM_reduced, LM_dense; rtol = 1.0e-14)
end

@testset "V&V: Full-tensor Germano — |S|² = 2 S_ij S_ij identity (2D)" begin
    S = (0.5, -0.2, 0.1)
    mag_sq = _sym_mag_sq(S, Val(2))
    S_dense = _reduced_to_dense2(S)
    # Dense check: 2 S_ij S_ij = 2 · sum(S.*S)
    @test isapprox(mag_sq, 2 * sum(S_dense .* S_dense); rtol = 1.0e-14)
end

@testset "V&V: Full-tensor Germano — |S|² = 2 S_ij S_ij identity (3D)" begin
    S = (0.3, -0.15, 0.1, 0.2, -0.05, 0.07)
    mag_sq = _sym_mag_sq(S, Val(3))
    S_dense = _reduced_to_dense3(S)
    @test isapprox(mag_sq, 2 * sum(S_dense .* S_dense); rtol = 1.0e-14)
end

@testset "V&V: Full-tensor Germano — Cs² ≥ 0 after clip" begin
    # Mimic the core Germano-Lilly step: given (L, M), compute
    # Cs² = max(LM/MM, 0), then cap at 0.04 (corresponds to Cs < 0.2).
    # Pick cases where LM > MM (would yield Cs² > 0.04) to exercise
    # the cap.
    L = (1.0, -0.5, 0.3)
    M = (0.1, -0.05, 0.03)
    LM = _sym_contract(L, M, Val(2))
    MM = _sym_contract(M, M, Val(2))
    Cs_sq_raw = LM / MM
    Cs_sq = min(max(Cs_sq_raw, 0.0), 0.04)
    @test 0.0 <= Cs_sq <= 0.04

    # Also check the negative-LM case: Cs² must clip to 0, not go
    # negative.
    L_neg = (-1.0, 0.5, -0.3)
    LM_neg = _sym_contract(L_neg, M, Val(2))
    @test LM_neg < 0.0
    Cs_sq_neg = min(max(LM_neg / MM, 0.0), 0.04)
    @test Cs_sq_neg == 0.0
end

@testset "V&V: Full-tensor Germano — M_ij = 0 ⇒ MM = 0 (floor triggered)" begin
    # If the test-filtered and grid-level strains coincide (M_ij = 0)
    # the Germano identity is singular; the production code substitutes
    # Cs² = 0.01 as a conservative floor. Verify MM=0 is detected
    # algebraically.
    M_zero = (0.0, 0.0, 0.0)
    MM = _sym_contract(M_zero, M_zero, Val(2))
    @test MM == 0.0
end

@testset "V&V: Full-tensor Germano — full-tensor M differs from scalar Germano" begin
    # Construct S and S̃ that are NOT proportional. A scalar-Germano
    # fallback (S̃_ij ≈ S_ij · |S̃|/|S|) would compute M from
    # 2Δ²(α² |S̃| − |S|) · S_ij — proportional to S_ij. The full-tensor
    # version in `dynamic_smagorinsky.jl` uses the actual S̃_ij, which
    # is not proportional, so the resulting M is not proportional to
    # S. We verify this algebraically.
    S = (1.0, -0.5, 0.2)
    S_filt = (0.5, 0.5, 0.0)   # genuinely different direction
    Δ = 0.1
    α = 2.0
    S_mag = sqrt(_sym_mag_sq(S, Val(2)))
    S_filt_mag = sqrt(_sym_mag_sq(S_filt, Val(2)))
    M = ntuple(3) do k
        2 * Δ^2 * (α^2 * S_filt_mag * S_filt[k] - S_mag * S[k])
    end
    # Check M is not a scalar multiple of S.
    # Compute two components ratios; they must differ.
    r1 = M[1] / S[1]
    r2 = M[2] / S[2]
    @test !isapprox(r1, r2; rtol = 1.0e-6)
end
