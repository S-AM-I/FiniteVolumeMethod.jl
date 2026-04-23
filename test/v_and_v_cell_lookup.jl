# test/v_and_v_cell_lookup.jl — find_nearest_cell V&V (v3.83)

using FiniteVolumeMethod
using LinearAlgebra: norm
using StaticArrays
using Test

include("TestHelpers.jl")

const _nearest = FiniteVolumeMethod.find_nearest_cell

@testset "V&V: find_nearest_cell — exact cell-center lookup returns that cell" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    for c in 1:length(mesh.cell_volumes)
        x_c = SVector(mesh.cell_centers[1, c], mesh.cell_centers[2, c])
        @test _nearest(mesh, x_c) == c
    end
end

@testset "V&V: find_nearest_cell — nearby point returns same cell" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    # Offset each cell center by much less than half-cell width.
    for c in 1:length(mesh.cell_volumes)
        x_c = SVector(mesh.cell_centers[1, c], mesh.cell_centers[2, c])
        x_offset = x_c + SVector(0.001, 0.001)
        @test _nearest(mesh, x_offset) == c
    end
end

@testset "V&V: find_nearest_cell — empty-like mesh returns sensible" begin
    # With at least one cell, find_nearest_cell returns a valid index.
    mesh = build_cartesian_unstructured_mesh(2, 2, 1.0, 1.0)
    result = _nearest(mesh, SVector(0.5, 0.5))
    @test 1 <= result <= 4
end

@testset "V&V: find_nearest_cell — monotone distance property" begin
    # The nearest cell's center must be ≤ distance of all other cells.
    mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)
    for query in (SVector(0.1, 0.2), SVector(0.7, 0.3), SVector(0.5, 0.5))
        c_nearest = _nearest(mesh, query)
        x_near = SVector(
            mesh.cell_centers[1, c_nearest], mesh.cell_centers[2, c_nearest],
        )
        d_near = norm(query - x_near)
        for c in 1:length(mesh.cell_volumes)
            x_other = SVector(mesh.cell_centers[1, c], mesh.cell_centers[2, c])
            @test d_near <= norm(query - x_other) + 1.0e-14
        end
    end
end
