# test/v_and_v_sampling.jl — sample_field_at_point + sample_line V&V (v3.96)

using FiniteVolumeMethod
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: sample_field_at_point — constant scalar recovered" begin
    # Constant field should sample to the same value at any point and
    # under either interpolation scheme.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    field = CollocatedScalarField(:T, mesh; value = 7.25)
    for pt in (
            SVector(0.1, 0.1), SVector(0.5, 0.5), SVector(0.9, 0.9),
            SVector(0.25, 0.75), SVector(0.001, 0.999),
        )
        @test sample_field_at_point(field, mesh, pt) == 7.25
        @test isapprox(
            sample_field_at_point(field, mesh, pt; interpolation = :idw),
            7.25; rtol = 1.0e-14,
        )
    end
end

@testset "V&V: sample_field_at_point — IDW exact hit at cell center" begin
    # When the sampled point coincides with a cell center, IDW falls
    # back to the exact field value at that cell (guarded by the
    # eps(T)*100 proximity check).
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    field = CollocatedScalarField(:T, mesh; value = 0.0)
    # Inject a known value at a specific cell.
    field.internal[13] = 42.0
    pt = SVector(mesh.cell_centers[1, 13], mesh.cell_centers[2, 13])
    @test sample_field_at_point(field, mesh, pt; interpolation = :idw) == 42.0
end

@testset "V&V: sample_field_at_point — nearest matches find_nearest_cell" begin
    # :nearest interpolation just reads the value at the nearest cell.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    field = CollocatedScalarField(:T, mesh; value = 0.0)
    for c in 1:nc
        field.internal[c] = Float64(c)
    end
    for c in 1:nc
        pt = SVector(mesh.cell_centers[1, c], mesh.cell_centers[2, c])
        v = sample_field_at_point(field, mesh, pt)
        @test v == Float64(c)
    end
end

@testset "V&V: sample_line — endpoints, distances, positions correct" begin
    # sample_line returns positions along the line, distances from p1,
    # and field values. Check geometric correctness of the first three.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    field = CollocatedScalarField(:T, mesh; value = 5.0)
    p1 = SVector(0.1, 0.5)
    p2 = SVector(0.9, 0.5)
    n = 11
    result = sample_line(field, mesh, p1, p2, n)
    @test length(result.positions) == n
    @test length(result.distances) == n
    @test length(result.values) == n
    # First position is p1, last is p2.
    @test result.positions[1] == p1
    @test result.positions[end] == p2
    # Distances span 0 to |p2-p1|.
    @test result.distances[1] == 0.0
    @test result.distances[end] ≈ 0.8 rtol = 1.0e-14
    # Constant field ⇒ all values equal.
    for v in result.values
        @test v == 5.0
    end
    # Monotone distances.
    for i in 2:n
        @test result.distances[i] > result.distances[i - 1]
    end
end

@testset "V&V: sample_line — distance scales with line length" begin
    # Doubling p2 - p1 doubles the total length and every distance.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    field = CollocatedScalarField(:T, mesh; value = 1.0)
    p1 = SVector(0.1, 0.5)
    r1 = sample_line(field, mesh, p1, SVector(0.3, 0.5), 5)
    r2 = sample_line(field, mesh, p1, SVector(0.5, 0.5), 5)
    # r2 line is twice r1 line; each distance[i] should double.
    for i in 1:5
        @test r2.distances[i] ≈ 2.0 * r1.distances[i] rtol = 1.0e-14
    end
end

@testset "V&V: sample_line — vector field round-trip" begin
    # Vector-field sampling returns SVector{Dim, T} values.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    U = CollocatedVectorField(:U, mesh; value = SVector(2.0, -1.0))
    r = sample_line(U, mesh, SVector(0.1, 0.1), SVector(0.9, 0.9), 5)
    @test length(r.values) == 5
    for v in r.values
        @test v == SVector(2.0, -1.0)
    end
end

@testset "V&V: sample_line — linear ramp field recovers gradient" begin
    # Field T(x, y) = x evaluated along a horizontal line at mid-y.
    # Sampled values should increase roughly linearly with distance.
    mesh = build_cartesian_unstructured_mesh(32, 32, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    field = CollocatedScalarField(:T, mesh; value = 0.0)
    for c in 1:nc
        field.internal[c] = mesh.cell_centers[1, c]
    end
    r = sample_line(field, mesh, SVector(0.1, 0.5), SVector(0.9, 0.5), 9)
    # Each step should monotonically increase by roughly Δx ≈ 0.1.
    for i in 2:9
        @test r.values[i] >= r.values[i - 1] - 1.0e-10
    end
    # First sample close to x = 0.1, last close to x = 0.9.
    @test abs(r.values[1] - 0.1) < 0.1
    @test abs(r.values[end] - 0.9) < 0.1
end

@testset "V&V: sample_line — n_points = 1 degenerate case" begin
    # With n = 1 the sample collapses to p1 only.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    field = CollocatedScalarField(:T, mesh; value = 9.0)
    r = sample_line(field, mesh, SVector(0.3, 0.7), SVector(0.7, 0.3), 1)
    @test length(r.positions) == 1
    @test r.positions[1] == SVector(0.3, 0.7)
    @test r.distances[1] == 0.0
    @test r.values[1] == 9.0
end
