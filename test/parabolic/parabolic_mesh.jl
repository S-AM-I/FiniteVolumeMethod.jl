using Test
using FiniteVolumeMethod
using FiniteVolumeMethod: CellField, STATEVAR, build_axisymmetric_rz_mesh, generate_mesh_1d, generate_mesh_2d
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC, RobinBC

@testset "Parabolic Mesh" begin
    @testset "1D Mesh Generation" begin
        mesh = generate_mesh_1d(10, 1.0)
        @test length(mesh.cells) == 10
        @test length(mesh.nodes) == 11
        @test mesh.cells[1].volume ≈ 0.1
    end

    @testset "2D Mesh Generation" begin
        mesh = generate_mesh_2d(5, 5, 1.0, 1.0)
        @test length(mesh.cells) == 25
    end

    @testset "Axisymmetric R-Z Mesh" begin
        r_edges = collect(range(0.001, 0.01, length = 6))
        z_edges = collect(range(0.0, 0.1, length = 11))
        mesh = build_axisymmetric_rz_mesh(r_edges, z_edges)
        @test size(mesh.cell_volumes) == (5, 10)
    end

    @testset "Parabolic BC Types" begin
        d = DirichletBC(1.0)
        @test d.value == 1.0
        n = NeumannBC(0.0)
        @test n.value == 0.0
        r = RobinBC(1.0, 2.0, 3.0)
        @test r.a == 1.0
    end

    @testset "CellField" begin
        v = Variable(:T, STATEVAR, :K, "temperature")
        f = CellField(v, [300.0, 310.0, 320.0])
        @test length(f.values) == 3
    end
end
