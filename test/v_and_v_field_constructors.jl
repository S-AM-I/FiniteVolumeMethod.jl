# test/v_and_v_field_constructors.jl — Field-container constructor V&V (v3.85)

using FiniteVolumeMethod
using FiniteVolumeMethod: nfaces
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: CollocatedScalarField — default zero init" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    f = CollocatedScalarField(:phi, mesh)
    @test f.name == :phi
    @test length(f.internal) == length(mesh.cell_volumes)
    for v in f.internal
        @test v == 0.0
    end
end

@testset "V&V: CollocatedScalarField — custom value" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    f = CollocatedScalarField(:T, mesh; value = 42.5)
    for v in f.internal
        @test v == 42.5
    end
end

@testset "V&V: CollocatedVectorField — default zero init" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    U = CollocatedVectorField(:U, mesh)
    @test U.name == :U
    @test length(U.internal) == length(mesh.cell_volumes)
    for u in U.internal
        @test u == SVector(0.0, 0.0)
    end
end

@testset "V&V: CollocatedVectorField — custom value" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    U = CollocatedVectorField(:U, mesh; value = SVector(1.5, -0.5))
    for u in U.internal
        @test u == SVector(1.5, -0.5)
    end
end

@testset "V&V: FaceFluxField — default zero init" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    phi = FaceFluxField(:phi, mesh)
    @test phi.name == :phi
    @test length(phi.values) == size(mesh.face_cells, 2)
    for v in phi.values
        @test v == 0.0
    end
end

@testset "V&V: FaceFluxField — custom value" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    phi = FaceFluxField(:phi, mesh; value = 3.14)
    for v in phi.values
        @test v == 3.14
    end
end

@testset "V&V: nfaces matches face_cells size" begin
    for N in (4, 8, 16)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        phi = FaceFluxField(:phi, mesh)
        @test FiniteVolumeMethod.nfaces(phi) == size(mesh.face_cells, 2)
    end
end
