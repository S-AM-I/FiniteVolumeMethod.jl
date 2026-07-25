# test/v_and_v_mesh_accessors.jl — Mesh accessor primitives V&V (v3.74)

using FiniteVolumeMethod
using FiniteVolumeMethod: cell_center, face_area, face_normal_area, face_weight, is_internal_face
using LinearAlgebra: norm
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: mesh accessors — face_weight ∈ [0, 1]" begin
    mesh = build_cartesian_unstructured_mesh(10, 10, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)
    for f in 1:nf
        if is_internal_face(mesh, f)
            w = face_weight(mesh, f)
            @test 0.0 <= w <= 1.0
        end
    end
end

@testset "V&V: mesh accessors — face_weight = 0.5 on uniform Cartesian" begin
    # On a uniform Cartesian mesh, owner and neighbour are
    # equidistant from the face ⇒ linear-interp weight = 0.5.
    mesh = build_cartesian_unstructured_mesh(10, 10, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)
    for f in 1:nf
        if is_internal_face(mesh, f)
            w = face_weight(mesh, f)
            @test isapprox(w, 0.5; rtol = 1.0e-12)
        end
    end
end

@testset "V&V: mesh accessors — cell_center inside domain" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    for c in 1:length(mesh.cell_volumes)
        x_c = FiniteVolumeMethod.cell_center(mesh, c)
        @test 0.0 < x_c[1] < 1.0
        @test 0.0 < x_c[2] < 1.0
    end
end

@testset "V&V: mesh accessors — face_center on boundary or internal" begin
    mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)
    for f in 1:nf
        x_f = FiniteVolumeMethod.face_center(mesh, f)
        # face center must lie in [0, 1]² (closed).
        @test 0.0 <= x_f[1] <= 1.0
        @test 0.0 <= x_f[2] <= 1.0
    end
end

@testset "V&V: mesh accessors — face_normal_area magnitude matches face_area" begin
    mesh = build_cartesian_unstructured_mesh(10, 10, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)
    for f in 1:nf
        S_f = face_normal_area(mesh, f)
        A_f = mesh.face_areas[f]
        @test isapprox(norm(S_f), A_f; rtol = 1.0e-12)
    end
end

@testset "V&V: mesh accessors — is_internal_face partitions" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)
    n_internal = 0
    n_boundary = 0
    for f in 1:nf
        if is_internal_face(mesh, f)
            n_internal += 1
        else
            n_boundary += 1
        end
    end
    # Partition is disjoint + complete.
    @test n_internal + n_boundary == nf
    # 8×8 Cartesian mesh has 2·(8-1)·8 = 112 internal faces,
    # 4·8 = 32 boundary faces, total 144.
    @test n_internal == 112
    @test n_boundary == 32
end
