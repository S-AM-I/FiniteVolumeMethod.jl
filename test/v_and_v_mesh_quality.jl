# test/v_and_v_mesh_quality.jl — check_mesh_quality V&V (v3.98)

using FiniteVolumeMethod
using Test

include("TestHelpers.jl")

@testset "V&V: MeshQualityReport — Cartesian mesh ⇒ zero non-orthogonality" begin
    # An axis-aligned Cartesian mesh has face normals that point along
    # owner-to-neighbour vectors exactly, so the angle between them is
    # zero on every internal face.
    for N in (4, 8, 16)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        report = check_mesh_quality(mesh)
        @test report.max_non_orthogonality < 1.0e-10
        @test report.avg_non_orthogonality < 1.0e-10
        for v in report.non_orthogonality
            @test v < 1.0e-10
        end
    end
end

@testset "V&V: MeshQualityReport — Cartesian mesh ⇒ zero skewness" begin
    # For a uniform Cartesian mesh, the face center is exactly on the
    # owner-neighbour line, so skewness ≡ 0.
    for N in (4, 8, 16)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        report = check_mesh_quality(mesh)
        @test report.max_skewness < 1.0e-12
        @test report.avg_skewness < 1.0e-12
        for v in report.skewness
            @test v < 1.0e-12
        end
    end
end

@testset "V&V: MeshQualityReport — output shape matches mesh" begin
    # non_orthogonality and skewness arrays have length = # internal faces.
    # aspect_ratio has length = # cells.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    n_internal = 0
    for f in 1:size(mesh.face_cells, 2)
        if FiniteVolumeMethod.is_internal_face(mesh, f)
            n_internal += 1
        end
    end
    report = check_mesh_quality(mesh)
    @test length(report.non_orthogonality) == n_internal
    @test length(report.skewness) == n_internal
    @test length(report.aspect_ratio) == nc
end

@testset "V&V: MeshQualityReport — aspect ratio positive + bounded" begin
    # aspect_ratio = max_face_area / V^(2/3) is strictly positive and
    # finite on a regular Cartesian mesh.
    for N in (4, 8, 16, 32)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        report = check_mesh_quality(mesh)
        @test report.max_aspect_ratio > 0.0
        @test isfinite(report.max_aspect_ratio)
        for v in report.aspect_ratio
            @test v > 0.0
            @test isfinite(v)
        end
    end
end

@testset "V&V: MeshQualityReport — summary matches array maxima" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    report = check_mesh_quality(mesh)
    # max_* and avg_* fields must match the corresponding reductions.
    @test report.max_non_orthogonality == maximum(report.non_orthogonality)
    @test report.avg_non_orthogonality ≈
        sum(report.non_orthogonality) / length(report.non_orthogonality) rtol = 1.0e-14
    @test report.max_skewness == maximum(report.skewness)
    @test report.avg_skewness ≈
        sum(report.skewness) / length(report.skewness) rtol = 1.0e-14
    @test report.max_aspect_ratio == maximum(report.aspect_ratio)
end

@testset "V&V: MeshQualityReport — anisotropic Cartesian aspect scaling" begin
    # For a Cartesian mesh with cells of different aspect, the aspect-ratio
    # value grows as the domain anisotropy grows.
    mesh_iso = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    mesh_aniso = build_cartesian_unstructured_mesh(8, 8, 10.0, 1.0)
    ar_iso = check_mesh_quality(mesh_iso).max_aspect_ratio
    ar_aniso = check_mesh_quality(mesh_aniso).max_aspect_ratio
    @test ar_aniso > ar_iso
end

@testset "V&V: MeshQualityReport — refinement preserves zero non-ortho" begin
    # Refining a Cartesian mesh cannot introduce non-orthogonality.
    # All levels must remain at round-off zero.
    for N in (4, 8, 16, 32, 64)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        @test check_mesh_quality(mesh).max_non_orthogonality < 1.0e-10
    end
end
