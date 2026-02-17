using Test
using FiniteVolumeMethod
using DelaunayTriangulation

@testset "Coordinate Systems" begin
    @testset "Weight Functions" begin
        @test geometric_volume_weight(Cartesian(), 1.0, 2.0) == 1.0
        @test geometric_volume_weight(Cylindrical(), 3.0, 2.0) == 3.0
        @test geometric_volume_weight(Spherical(), 2.0, π / 4) ≈ 4.0 * sin(π / 4)

        @test geometric_flux_weight(Cartesian(), 1.0, 2.0) == 1.0
        @test geometric_flux_weight(Cylindrical(), 3.0, 2.0) == 3.0
    end

    @testset "FVMGeometry with Cartesian (default)" begin
        tri = triangulate_rectangle(0, 1, 0, 1, 5, 5, single_boundary = true)
        mesh = FVMGeometry(tri)
        @test mesh.coordinate_system isa Cartesian
        # Total volume should be approximately 1.0
        @test sum(mesh.cv_volumes) ≈ 1.0 atol = 0.01
    end

    @testset "FVMGeometry with Cylindrical" begin
        # Create a triangulation in (r, z) space: r ∈ [1, 2], z ∈ [0, 1]
        tri = triangulate_rectangle(1, 2, 0, 1, 10, 10, single_boundary = true)
        mesh_cart = FVMGeometry(tri)
        mesh_cyl = FVMGeometry(tri; coordinate_system = Cylindrical())

        @test mesh_cyl.coordinate_system isa Cylindrical

        # Cylindrical volumes should be larger because r > 1
        # The r-weighted area ∫∫ r dr dz for r ∈ [1,2], z ∈ [0,1] = ∫₁² r dr = 3/2
        @test sum(mesh_cyl.cv_volumes) > sum(mesh_cart.cv_volumes)
        @test sum(mesh_cyl.cv_volumes) ≈ 1.5 atol = 0.15  # ∫₁² r dr = 1.5
    end

    @testset "Cartesian backward compatibility" begin
        # Ensure existing Cartesian problems still work correctly
        tri = triangulate_rectangle(0, 1, 0, 1, 5, 5, single_boundary = true)
        mesh = FVMGeometry(tri)
        bc = (x, y, t, u, p) -> zero(u)
        BCs = BoundaryConditions(mesh, bc, Dirichlet)
        D = (x, y, t, u, p) -> one(u)
        initial_condition = zeros(DelaunayTriangulation.num_points(tri))
        prob = FVMProblem(mesh, BCs; diffusion_function = D, initial_condition, final_time = 0.1)
        @test get_coordinate_system(prob) isa Cartesian
    end
end
