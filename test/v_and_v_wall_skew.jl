# test/v_and_v_wall_skew.jl — Skew-penalty wall-function algebra V&V (v3.0 / Wave 1)
#
# Exercises the wall-projection primitive `_wall_projection` in
# `wall_functions.jl` that extracts (y, U_par) from (cell centre,
# face centre, face normal, cell velocity). The pre-v3 code used the
# raw Euclidean distance and |U_cell|, which systematically
# over-counted on skewed cells. The new projection computes
#
#     y     = |(x_c - x_f) · n̂|
#     U_par = |U_cell - (U_cell · n̂) n̂|
#
# which is the correct wall-normal distance and wall-tangential speed
# regardless of cell skew and regardless of any spurious wall-normal
# component in `U_cell` left over by an iterative solve.
#
# This test file builds tiny synthetic meshes (1-cell, single boundary
# face) so the projection is tested as a pure geometric operation.

using FiniteVolumeMethod
using LinearAlgebra
using StaticArrays
using Test

include("TestHelpers.jl")

const _proj = FiniteVolumeMethod._wall_projection

"""
    _one_cell_mesh(xc, xf, n̂, A)

Build the smallest possible `UnstructuredFVMMesh{2, Float64}` that
supports `_wall_projection`: a single cell with centre `xc`, and a
single boundary face with centre `xf`, outward normal `n̂`, area `A`.
All other mesh data (volumes, internal faces) is populated with
dummy values that are not read by the projection helper.
"""
function _one_cell_mesh(xc, xf, n_hat, A)
    ncells = 1; nfaces = 1
    cell_centers = reshape([xc[1], xc[2]], (2, 1))
    cell_volumes = [1.0]
    face_cells = reshape([1, 0], (2, 1))
    face_centers = reshape([xf[1], xf[2]], (2, 1))
    face_areas = [A]
    face_normals = reshape([n_hat[1], n_hat[2]], (2, 1))
    face_tags = [:wall]
    cell_faces = [[1]]
    return FiniteVolumeMethod.UnstructuredFVMMesh{2, Float64}(
        cell_centers, cell_volumes, face_cells, face_centers,
        face_areas, face_normals, face_tags, nothing, cell_faces,
    )
end

@testset "V&V: Wall skew — axis-aligned Cartesian wall reduces to raw values" begin
    # Cell centre at (0.5, 0.1); wall face at (0.5, 0.0) with outward
    # normal (0, -1). Cell velocity is purely tangential. The
    # projection must give y = 0.1 exactly and U_par = |U_cell|.
    mesh = _one_cell_mesh((0.5, 0.1), (0.5, 0.0), (0.0, -1.0), 1.0)
    U_cell = SVector{2, Float64}(1.5, 0.0)
    y, U_par = _proj(mesh, 1, 1, U_cell)
    @test isapprox(y, 0.1; rtol = 1.0e-14)
    @test isapprox(U_par, 1.5; rtol = 1.0e-14)
end

@testset "V&V: Wall skew — wall-normal velocity removed by projection" begin
    # Same mesh; now the cell has a spurious wall-normal component
    # (e.g. from an iterative solve before the BC is enforced). The
    # projected U_par must strip that component cleanly.
    mesh = _one_cell_mesh((0.5, 0.1), (0.5, 0.0), (0.0, -1.0), 1.0)
    U_cell = SVector{2, Float64}(2.0, 0.5)   # 0.5 is wall-normal
    y, U_par = _proj(mesh, 1, 1, U_cell)
    # Wall-normal speed must vanish from U_par; only the x-component
    # (tangential) should remain.
    @test isapprox(U_par, 2.0; rtol = 1.0e-14)
    @test isapprox(y, 0.1; rtol = 1.0e-14)
end

@testset "V&V: Wall skew — 45° skewed face: |U_par| preserved" begin
    # Wall face with normal n̂ = (1, 1)/√2. Cell centre displaced by
    # (0.1, 0.1) from the face centre (directly along n̂, so y = √0.02
    # = 0.1·√2). Cell velocity tangential to the wall, i.e. parallel
    # to (-1, 1)/√2 with magnitude 1.
    n_hat = (1 / sqrt(2), 1 / sqrt(2))
    xf = (0.0, 0.0)
    xc = (0.1, 0.1)
    mesh = _one_cell_mesh(xc, xf, n_hat, 1.0)

    # Tangent direction: t = (-1, 1)/√2
    t_hat = (-1 / sqrt(2), 1 / sqrt(2))
    U_mag = 1.0
    U_cell = SVector{2, Float64}(U_mag * t_hat[1], U_mag * t_hat[2])

    y, U_par = _proj(mesh, 1, 1, U_cell)
    # y = |d · n̂| = |(0.1, 0.1) · (1,1)/√2| = 0.2/√2 = 0.1·√2
    @test isapprox(y, 0.1 * sqrt(2); rtol = 1.0e-14)
    # U_par should equal |U_cell| exactly since U_cell is perpendicular
    # to n̂.
    @test isapprox(U_par, U_mag; rtol = 1.0e-14)
end

@testset "V&V: Wall skew — 45° skewed face: wall-normal component removed" begin
    # Same skewed mesh; now inject a spurious wall-normal component
    # into U_cell. The projected U_par must drop it.
    n_hat = (1 / sqrt(2), 1 / sqrt(2))
    xf = (0.0, 0.0)
    xc = (0.1, 0.1)
    mesh = _one_cell_mesh(xc, xf, n_hat, 1.0)

    t_hat = (-1 / sqrt(2), 1 / sqrt(2))
    U_tang_mag = 2.0
    U_norm_mag = 0.7   # spurious normal component
    U_cell = SVector{2, Float64}(
        U_tang_mag * t_hat[1] + U_norm_mag * n_hat[1],
        U_tang_mag * t_hat[2] + U_norm_mag * n_hat[2],
    )
    y, U_par = _proj(mesh, 1, 1, U_cell)
    # The tangential magnitude must be recovered exactly.
    @test isapprox(U_par, U_tang_mag; rtol = 1.0e-12)
end

@testset "V&V: Wall skew — y ≥ 0 regardless of normal sign convention" begin
    # Flipping the face normal (inward vs outward) must not flip the
    # sign of y. `_wall_projection` uses abs() on the dot product for
    # exactly this robustness.
    mesh_out = _one_cell_mesh((0.5, 0.1), (0.5, 0.0), (0.0, -1.0), 1.0)
    mesh_in = _one_cell_mesh((0.5, 0.1), (0.5, 0.0), (0.0, 1.0), 1.0)
    U = SVector{2, Float64}(1.0, 0.0)
    y_out, _ = _proj(mesh_out, 1, 1, U)
    y_in, _ = _proj(mesh_in, 1, 1, U)
    @test isapprox(y_out, y_in; rtol = 1.0e-14)
    @test y_out >= 0.0
    @test y_in >= 0.0
end

@testset "V&V: Wall skew — pure wall-normal velocity projects to zero U_par" begin
    # If U_cell is purely wall-normal, U_par must be zero. This is
    # the degenerate "no slip achieved up to solver tolerance" case.
    mesh = _one_cell_mesh((0.5, 0.1), (0.5, 0.0), (0.0, -1.0), 1.0)
    U_cell = SVector{2, Float64}(0.0, 0.3)
    y, U_par = _proj(mesh, 1, 1, U_cell)
    @test isapprox(U_par, 0.0; atol = 1.0e-14)
    @test isapprox(y, 0.1; rtol = 1.0e-14)
end
