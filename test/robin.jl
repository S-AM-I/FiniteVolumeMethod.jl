using FiniteVolumeMethod
using FiniteVolumeMethod: Conditions, is_robin_edge, get_robin_fidx, has_robin_edges, get_robin_edges, ConditionType
using DelaunayTriangulation
using Test

@testset "Robin Boundary Conditions" begin
    @testset "Robin enum type" begin
        @test Robin isa ConditionType
        @test Robin !== Neumann
        @test Robin !== Dirichlet
        @test Robin !== Constrained
        @test Robin !== Dudt
    end

    @testset "BoundaryConditions with Robin" begin
        tri = triangulate_rectangle(0, 1, 0, 1, 5, 5, single_boundary=true)
        mesh = FVMGeometry(tri)

        # Robin BC function returns (a, b, c) tuple
        robin_fn = (x, y, t, u, p) -> (p.a, p.b, p.c)
        params = (a=1.0, b=2.0, c=3.0)

        BCs = BoundaryConditions(mesh, robin_fn, Robin; parameters=params)

        @test BCs.condition_types == (Robin,)
        @test length(BCs.functions) == 1
    end

    @testset "Conditions with Robin edges" begin
        tri = triangulate_rectangle(0, 1, 0, 1, 5, 5, single_boundary=true)
        mesh = FVMGeometry(tri)

        robin_fn = (x, y, t, u, p) -> (1.0, 2.0, 3.0)
        BCs = BoundaryConditions(mesh, robin_fn, Robin)

        conditions = Conditions(mesh, BCs)

        @test has_robin_edges(conditions)
        @test length(get_robin_edges(conditions)) > 0

        # All boundary edges should be Robin
        for (edge, fidx) in get_robin_edges(conditions)
            i, j = edge
            @test is_robin_edge(conditions, i, j)
        end
    end

    @testset "Mixed boundary conditions with Robin" begin
        # Create a mesh with multiple boundary segments
        tri = triangulate_rectangle(0, 1, 0, 1, 5, 5, single_boundary=false)
        mesh = FVMGeometry(tri)

        # Different BC types on different segments
        # Bottom (1): Dirichlet
        # Right (2): Neumann
        # Top (3): Robin
        # Left (4): Dirichlet
        dirichlet_fn = (x, y, t, u, p) -> 0.0
        neumann_fn = (x, y, t, u, p) -> 0.0
        robin_fn = (x, y, t, u, p) -> (1.0, 1.0, 0.0)  # (a, b, c)

        BCs = BoundaryConditions(mesh,
            (dirichlet_fn, neumann_fn, robin_fn, dirichlet_fn),
            (Dirichlet, Neumann, Robin, Dirichlet))

        conditions = Conditions(mesh, BCs)

        @test has_robin_edges(conditions)
        @test length(get_robin_edges(conditions)) > 0

        # Check that we have all three types
        @test length(conditions.dirichlet_nodes) > 0
        @test length(conditions.neumann_edges) > 0
        @test length(conditions.robin_edges) > 0
    end

    @testset "Robin BC evaluation" begin
        tri = triangulate_rectangle(0, 1, 0, 1, 5, 5, single_boundary=true)
        mesh = FVMGeometry(tri)

        # Robin BC that returns position-dependent coefficients
        robin_fn = (x, y, t, u, p) -> (x + 1, y + 1, x * y)

        BCs = BoundaryConditions(mesh, robin_fn, Robin)
        conditions = Conditions(mesh, BCs)

        # Get a Robin edge and evaluate the condition
        robin_edges = get_robin_edges(conditions)
        @test !isempty(robin_edges)

        for ((i, j), fidx) in robin_edges
            x, y = get_point(mesh, i)
            a, b, c = FiniteVolumeMethod.eval_condition_fnc(conditions, fidx, x, y, 0.0, 0.0)
            @test a ≈ x + 1
            @test b ≈ y + 1
            @test c ≈ x * y
            break  # Just test one edge
        end
    end
end
