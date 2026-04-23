# test/v_and_v_laplacian_motion.jl — Laplacian mesh motion V&V (v3.58)
#
# Fifth convergence-verified benchmark for `dynamic_mesh`, joining
# three-pattern GCL (v3.14), rotational GCL (v3.29), mesh sweep-
# flux (v3.34), and ALE-corrected flux (v3.49). Covers the
# diffusion-based mesh displacement solver `LaplacianMotion`
# which computes per-cell displacement by solving
#
#   div(γ · grad(d_i)) = 0   (one Laplace equation per dimension)
#
# with Dirichlet BCs on fixed and moving boundaries. Four
# invariants verified.

using FiniteVolumeMethod
using LinearAlgebra: norm
using LinearSolve
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: LaplacianMotion — zero-BC ⇒ d ≡ 0" begin
    # Boundary displacement zero everywhere ⇒ Laplace equation
    # has trivial solution d ≡ 0 in the interior.
    mesh = build_cartesian_unstructured_mesh(12, 12, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    solver = LaplacianMotion(; gamma = 1.0)

    bcs_zero = Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(0.0),
        :right => ParabolicDirichlet(0.0),
        :bottom => ParabolicDirichlet(0.0),
        :top => ParabolicDirichlet(0.0),
    )

    compute_displacement!(
        ms, solver, mesh, bcs_zero, 0.0;
        linear_solver = LUFactorization()
    )

    for c in 1:length(mesh.cell_volumes)
        @test norm(ms.displacement[c]) < 1.0e-10
    end
end

@testset "V&V: LaplacianMotion — uniform-BC ⇒ uniform displacement interior" begin
    # If *all* boundaries have the same Dirichlet value d₀, the
    # Laplace equation has the constant d ≡ d₀ as its unique
    # solution.
    mesh = build_cartesian_unstructured_mesh(12, 12, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    solver = LaplacianMotion(; gamma = 1.0)

    d0 = 0.1
    bcs_uniform = Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(d0),
        :right => ParabolicDirichlet(d0),
        :bottom => ParabolicDirichlet(d0),
        :top => ParabolicDirichlet(d0),
    )

    compute_displacement!(
        ms, solver, mesh, bcs_uniform, 0.0;
        linear_solver = LUFactorization()
    )

    # Each component should be d₀ at every interior cell.
    for c in 1:length(mesh.cell_volumes)
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.2 < x < 0.8 && 0.2 < y < 0.8
            @test isapprox(ms.displacement[c][1], d0; rtol = 1.0e-6)
            @test isapprox(ms.displacement[c][2], d0; rtol = 1.0e-6)
        end
    end
end

@testset "V&V: LaplacianMotion — γ-invariance (constant γ factors out)" begin
    # For Laplace equation with constant γ, the solution is
    # independent of γ. So changing γ from 1.0 to 10.0 should
    # give the same displacement field.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    ms_1 = MeshMotionState(mesh)
    ms_10 = MeshMotionState(mesh)

    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(0.0),
        :right => ParabolicDirichlet(0.0),
        :bottom => ParabolicDirichlet(0.0),
        :top => ParabolicDirichlet(0.05),  # moving top
    )

    compute_displacement!(
        ms_1, LaplacianMotion(; gamma = 1.0),
        mesh, bcs, 0.0; linear_solver = LUFactorization()
    )
    compute_displacement!(
        ms_10, LaplacianMotion(; gamma = 10.0),
        mesh, bcs, 0.0; linear_solver = LUFactorization()
    )

    # Displacement should match (γ-invariance of pure Laplace).
    for c in 1:length(mesh.cell_volumes)
        @test isapprox(
            ms_1.displacement[c][1], ms_10.displacement[c][1];
            rtol = 1.0e-6, atol = 1.0e-12
        )
        @test isapprox(
            ms_1.displacement[c][2], ms_10.displacement[c][2];
            rtol = 1.0e-6, atol = 1.0e-12
        )
    end
end

@testset "V&V: LaplacianMotion — interior max-principle (bounded by BCs)" begin
    # The solution of a Laplace equation satisfies the max/min
    # principle: interior values are bounded by the boundary values.
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    solver = LaplacianMotion(; gamma = 1.0)

    d_top = 0.05
    d_bot = 0.0
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(d_bot),
        :right => ParabolicDirichlet(d_bot),
        :bottom => ParabolicDirichlet(d_bot),
        :top => ParabolicDirichlet(d_top),
    )

    compute_displacement!(
        ms, solver, mesh, bcs, 0.0;
        linear_solver = LUFactorization()
    )

    # Interior values must lie in [d_bot, d_top].
    for c in 1:length(mesh.cell_volumes)
        d_mag_x = ms.displacement[c][1]
        d_mag_y = ms.displacement[c][2]
        @test d_bot - 1.0e-6 <= d_mag_x <= d_top + 1.0e-6
        @test d_bot - 1.0e-6 <= d_mag_y <= d_top + 1.0e-6
    end
end
