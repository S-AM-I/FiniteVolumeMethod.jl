using FiniteVolumeMethod
using FiniteVolumeMethod: compute_boundary_gradient, get_segment_nodes, _safe_get_triangle_props
using DelaunayTriangulation
using Test

@testset "Advanced Boundary Conditions" begin

    @testset "Nonlinear BCs" begin
        @testset "Type definitions" begin
            @test NonlinearDirichlet isa DataType
            @test NonlinearNeumann isa DataType
            @test NonlinearRobin isa DataType
        end

        @testset "linearize_bc for NonlinearDirichlet" begin
            # Test case: u = √(u + 1), fixed point at u = (1 + √5)/2 ≈ 1.618 (golden ratio)
            # This follows from φ² = φ + 1, so φ = √(φ + 1)
            f = (x, y, t, u, grad_u, p) -> sqrt(u + 1)

            # Starting from u = 1.5
            result = linearize_bc(NonlinearDirichlet, f, 0.0, 0.0, 0.0, 1.5, (0.0, 0.0), nothing)
            # Should move toward fixed point φ ≈ 1.618
            @test result > 1.5

            # Test near fixed point (golden ratio)
            φ = (1 + sqrt(5)) / 2
            result_near = linearize_bc(NonlinearDirichlet, f, 0.0, 0.0, 0.0, φ, (0.0, 0.0), nothing)
            @test isapprox(result_near, φ, atol = 1.0e-4)
        end

        @testset "linearize_bc for NonlinearNeumann" begin
            # Simple nonlinear flux: q·n = u^3
            f = (x, y, t, u, grad_u, p) -> u^3

            result = linearize_bc(NonlinearNeumann, f, 0.0, 0.0, 0.0, 2.0, (0.0, 0.0), nothing)
            @test result ≈ 8.0  # 2^3
        end

        @testset "linearize_bc for NonlinearRobin" begin
            # Nonlinear Robin with temperature-dependent coefficient
            f = (x, y, t, u, grad_u, p) -> (u, 1.0, u^2)  # a=u, b=1, c=u^2

            a, b, c = linearize_bc(NonlinearRobin, f, 0.0, 0.0, 0.0, 3.0, (0.0, 0.0), nothing)
            @test a ≈ 3.0
            @test b ≈ 1.0
            @test c ≈ 9.0
        end

        @testset "compute_boundary_gradient" begin
            # Create a simple mesh
            tri = triangulate_rectangle(0, 1, 0, 1, 3, 3, single_boundary = true)
            mesh = FVMGeometry(tri)

            # Linear solution u = x + y
            n_nodes = DelaunayTriangulation.num_solid_vertices(tri)
            u = zeros(n_nodes)
            for i in 1:n_nodes
                p = get_point(mesh, i)
                x, y = getxy(p)
                u[i] = x + y
            end

            # Find a boundary edge
            boundary_edges = collect(keys(get_boundary_edge_map(tri)))
            @test !isempty(boundary_edges)

            i, j = first(boundary_edges)
            grad = compute_boundary_gradient(mesh, u, i, j)

            # Gradient of (x + y) should be (1, 1)
            @test isapprox(grad[1], 1.0, atol = 0.1)
            @test isapprox(grad[2], 1.0, atol = 0.1)
        end
    end

    @testset "Periodic BCs" begin
        @testset "PeriodicBC construction" begin
            bc1 = PeriodicBC(1, 3)
            @test bc1.segment_pair == (1, 3)
            @test bc1.shift ≈ 0.0
            @test bc1.direction == :x

            bc2 = PeriodicBC(1, 3; shift = 1.0, direction = :y)
            @test bc2.shift ≈ 1.0
            @test bc2.direction == :y

            bc3 = PeriodicBC((2, 4); shift = 0.5)
            @test bc3.segment_pair == (2, 4)
        end

        @testset "get_segment_nodes" begin
            # Create mesh with distinct boundary segments
            tri = triangulate_rectangle(0, 1, 0, 1, 5, 5, single_boundary = false)
            mesh = FVMGeometry(tri)

            # Get nodes on segment 1 (bottom)
            nodes1 = get_segment_nodes(mesh, 1)
            @test !isempty(nodes1)

            # All nodes should be on y = 0
            for n in nodes1
                p = get_point(mesh, n)
                _, y = getxy(p)
                @test isapprox(y, 0.0, atol = 1.0e-10)
            end
        end

        @testset "compute_periodic_mapping" begin
            # Create mesh with left-right periodicity
            tri = triangulate_rectangle(0, 1, 0, 1, 5, 5, single_boundary = false)
            mesh = FVMGeometry(tri)

            # Segment 4 is left (x=0), segment 2 is right (x=1)
            bc = PeriodicBC(4, 2; direction = :x)
            mapping = compute_periodic_mapping(mesh, bc)

            # Should have matched pairs
            @test !isempty(mapping.node_pairs)

            # Verify pairs have matching y-coordinates
            for (n1, n2) in mapping.node_pairs
                p1 = get_point(mesh, n1)
                p2 = get_point(mesh, n2)
                _, y1 = getxy(p1)
                _, y2 = getxy(p2)
                @test isapprox(y1, y2, atol = 1.0e-8)
            end
        end

        @testset "apply_periodic_constraints!" begin
            # Create simple test
            mapping = PeriodicNodeMapping([(1, 3), (2, 4)], (1, 2), 0.0)
            u = [1.0, 2.0, 5.0, 8.0, 10.0]

            apply_periodic_constraints!(u, mapping)

            # After constraint: u[1] ≈ u[3] and u[2] ≈ u[4]
            @test isapprox(u[1], u[3], atol = 1.0e-10)
            @test isapprox(u[2], u[4], atol = 1.0e-10)
        end

        @testset "apply_periodic_constraints! with shift" begin
            mapping = PeriodicNodeMapping([(1, 2)], (1, 2), 2.0)
            u = [5.0, 3.0, 10.0]

            apply_periodic_constraints!(u, mapping)

            # After constraint: u[1] - u[2] = shift = 2.0
            @test isapprox(u[1] - u[2], 2.0, atol = 1.0e-10)
        end

        @testset "PeriodicConditions" begin
            tri = triangulate_rectangle(0, 1, 0, 1, 5, 5, single_boundary = false)
            mesh = FVMGeometry(tri)

            bcs = [PeriodicBC(4, 2; direction = :x)]
            pc = PeriodicConditions(mesh, bcs)

            @test has_periodic_conditions(pc)
            @test length(pc.mappings) == 1
        end
    end

    @testset "Coupled Multi-Field BCs" begin
        @testset "CoupledBC construction" begin
            bc = CoupledBC([1, 2], [2.0, 3.0], 5.0)
            @test bc.field_indices == [1, 2]
            @test bc.coefficients ≈ [2.0, 3.0]
            @test bc.rhs_value ≈ 5.0

            # Test with integer coefficients (should promote)
            bc2 = CoupledBC([1, 2], [2, 3], 5.0)
            @test bc2.coefficients ≈ [2.0, 3.0]
        end

        @testset "CoupledDirichlet" begin
            # u₁ = exp(-u₂)
            cd = CoupledDirichlet(1, (x, y, t, u, p) -> exp(-u[2]))
            @test get_target_field(cd) == 1

            # Evaluate
            u_vals = (1.0, 2.0)  # u₁=1, u₂=2
            result = evaluate_coupled_bc(cd, 0.0, 0.0, 0.0, u_vals)
            @test result ≈ exp(-2.0)
        end

        @testset "CoupledNeumann" begin
            # flux depends on gradient of another field
            cn = CoupledNeumann(1, (x, y, t, u, grad, p) -> -p.D * grad[2][1]; parameters = (D = 0.5,))
            @test get_target_field(cn) == 1

            u_vals = (1.0, 2.0)
            grad_vals = ((0.1, 0.2), (0.3, 0.4))  # grad[field][component]
            result = evaluate_coupled_bc(cn, 0.0, 0.0, 0.0, u_vals, grad_vals)
            @test result ≈ -0.5 * 0.3  # -D * ∂u₂/∂x
        end

        @testset "CoupledRobin" begin
            # Robin coefficients depend on another field
            cr = CoupledRobin(1, (x, y, t, u, grad, p) -> (u[2], 1.0, u[2]^2))
            @test get_target_field(cr) == 1

            u_vals = (1.0, 3.0)
            grad_vals = ((0.0, 0.0), (0.0, 0.0))
            a, b, c = evaluate_coupled_bc(cr, 0.0, 0.0, 0.0, u_vals, grad_vals)
            @test a ≈ 3.0   # u₂
            @test b ≈ 1.0
            @test c ≈ 9.0   # u₂²
        end

        @testset "evaluate_coupled_bc for CoupledBC" begin
            bc = CoupledBC([1, 2], [2.0, 3.0], 10.0)
            # Constraint: 2*u₁ + 3*u₂ = 10

            u_vals = (1.0, 2.0)  # 2*1 + 3*2 = 8, residual = 8 - 10 = -2
            residual = evaluate_coupled_bc(bc, u_vals)
            @test residual ≈ -2.0

            u_vals2 = (2.0, 2.0)  # 2*2 + 3*2 = 10, residual = 0
            residual2 = evaluate_coupled_bc(bc, u_vals2)
            @test residual2 ≈ 0.0
        end

        @testset "CoupledBoundaryConditions container" begin
            cbc = CoupledBoundaryConditions()

            cd = CoupledDirichlet(1, (x, y, t, u, p) -> u[2])
            add_coupled_bc!(cbc, 1, cd)

            @test has_coupled_bc(cbc, 1, 1)
            @test !has_coupled_bc(cbc, 1, 2)
            @test !has_coupled_bc(cbc, 2, 1)

            retrieved = get_coupled_bc(cbc, 1, 1)
            @test retrieved === cd
        end
    end
end
