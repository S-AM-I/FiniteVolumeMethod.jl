# test/v_and_v_test_filter.jl — Dynamic Smagorinsky primitives V&V (v3.90)

using FiniteVolumeMethod
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

const _test_filter = FiniteVolumeMethod._test_filter
const _sym_contract = FiniteVolumeMethod._sym_contract
const _sym_self_magnitude_sq = FiniteVolumeMethod._sym_self_magnitude_sq

@testset "V&V: _test_filter — constant preservation" begin
    # Volume-weighted average of a constant field equals the constant.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    for cval in (0.0, 1.0, -3.5, 7.25)
        values = fill(cval, nc)
        filtered = _test_filter(values, mesh)
        for c in 1:nc
            @test filtered[c] ≈ cval rtol = 1.0e-14
        end
    end
end

@testset "V&V: _test_filter — shape and type" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    values = rand(nc)
    filtered = _test_filter(values, mesh)
    @test length(filtered) == nc
    @test eltype(filtered) == Float64
    # Smoothing cannot exceed the original range.
    vmin, vmax = extrema(values)
    for c in 1:nc
        @test vmin - 1.0e-12 <= filtered[c] <= vmax + 1.0e-12
    end
end

@testset "V&V: _test_filter — linearity" begin
    # filter(α·u + β·v) = α·filter(u) + β·filter(v).
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    u = rand(nc)
    v = rand(nc)
    alpha = 2.3
    beta = -1.7
    fu = _test_filter(u, mesh)
    fv = _test_filter(v, mesh)
    fcomb = _test_filter(alpha .* u .+ beta .* v, mesh)
    for c in 1:nc
        @test fcomb[c] ≈ alpha * fu[c] + beta * fv[c] rtol = 1.0e-12
    end
end

@testset "V&V: _sym_contract 2D — A⊙A closed form" begin
    # Double contraction with symmetric tensor in reduced-component form.
    # For 2D: A = (A_xx, A_yy, A_xy), A⊙A = A_xx² + A_yy² + 2·A_xy².
    for (Axx, Ayy, Axy) in (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 2.0, 3.0),
            (-2.0, 0.5, 0.25),
        )
        A = (Axx, Ayy, Axy)
        expected = Axx^2 + Ayy^2 + 2.0 * Axy^2
        @test _sym_contract(A, A, Val(2)) ≈ expected rtol = 1.0e-14
    end
end

@testset "V&V: _sym_contract 2D — bilinearity + commutativity" begin
    A = (1.2, -0.7, 0.4)
    B = (0.3, 2.1, -0.5)
    C = (2.0, 1.0, 0.0)
    # A⊙B == B⊙A.
    @test _sym_contract(A, B, Val(2)) ≈ _sym_contract(B, A, Val(2)) rtol = 1.0e-14
    # Linearity in second arg: A⊙(B + C) == A⊙B + A⊙C.
    lhs = _sym_contract(A, (B[1] + C[1], B[2] + C[2], B[3] + C[3]), Val(2))
    rhs = _sym_contract(A, B, Val(2)) + _sym_contract(A, C, Val(2))
    @test lhs ≈ rhs rtol = 1.0e-14
    # Scaling: (α·A)⊙B = α·(A⊙B).
    alpha = 3.7
    lhs2 = _sym_contract((alpha * A[1], alpha * A[2], alpha * A[3]), B, Val(2))
    @test lhs2 ≈ alpha * _sym_contract(A, B, Val(2)) rtol = 1.0e-14
end

@testset "V&V: _sym_contract 3D — A⊙A closed form" begin
    # 3D: A = (A_xx, A_yy, A_xy, A_zz, A_xz, A_yz).
    # A⊙A = A_xx² + A_yy² + A_zz² + 2·(A_xy² + A_xz² + A_yz²).
    for (xx, yy, xy, zz, xz, yz) in (
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            (-0.5, 0.3, 0.1, -1.2, 0.7, -0.4),
        )
        A = (xx, yy, xy, zz, xz, yz)
        expected = xx^2 + yy^2 + zz^2 + 2.0 * (xy^2 + xz^2 + yz^2)
        @test _sym_contract(A, A, Val(3)) ≈ expected rtol = 1.0e-14
    end
end

@testset "V&V: _sym_self_magnitude_sq — |S|² = 2·S_ij·S_ij identity" begin
    # By definition |S|² = 2·(S⊙S) — factor of 2 comes from the
    # strain-rate magnitude convention |S| = √(2 S_ij S_ij).
    for (Sxx, Syy, Sxy) in (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 2.0, 3.0),
            (-0.5, 0.25, 0.75),
        )
        S = (Sxx, Syy, Sxy)
        expected = 2.0 * (Sxx^2 + Syy^2 + 2.0 * Sxy^2)
        @test _sym_self_magnitude_sq(S, Val(2)) ≈ expected rtol = 1.0e-14
        # Check 2D: equals 2 · (A⊙A).
        @test _sym_self_magnitude_sq(S, Val(2)) ≈ 2.0 * _sym_contract(S, S, Val(2)) rtol = 1.0e-14
    end
    # 3D spot checks.
    S3 = (1.0, 2.0, 0.5, 3.0, 0.25, 0.75)
    expected3 = 2.0 * (1.0 + 4.0 + 9.0 + 2.0 * (0.25 + 0.0625 + 0.5625))
    @test _sym_self_magnitude_sq(S3, Val(3)) ≈ expected3 rtol = 1.0e-14
    @test _sym_self_magnitude_sq(S3, Val(3)) ≈ 2.0 * _sym_contract(S3, S3, Val(3)) rtol = 1.0e-14
end

@testset "V&V: _sym_contract zeros ⇒ zero" begin
    Z2 = (0.0, 0.0, 0.0)
    @test _sym_contract(Z2, Z2, Val(2)) == 0.0
    Z3 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    @test _sym_contract(Z3, Z3, Val(3)) == 0.0
    # Any tensor contracted with zero ⇒ zero.
    A = (1.0, 2.0, 3.0)
    @test _sym_contract(A, Z2, Val(2)) == 0.0
    @test _sym_contract(Z2, A, Val(2)) == 0.0
end
