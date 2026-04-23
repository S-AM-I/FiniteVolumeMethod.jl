using FiniteVolumeMethod
using Test
using LinearAlgebra
using LinearSolve
using StaticArrays
using SparseArrays

# ── Mesh builder (shared helper) ─────────────────────────────────────
include("TestHelpers.jl")

# ── Tests ──────────────────────────────────────────────────────────────

@testset "Incompressible Navier-Stokes" begin

    # ── 1. Type construction ───────────────────────────────────────────
    @testset "Type construction" begin
        s = SIMPLE()
        @test s isa SIMPLE{Float64}
        @test s.alpha_U == 0.7
        @test s.alpha_p == 0.3
        @test s.max_iterations == 1000
        @test s.tolerance == 1.0e-6

        p = PISO()
        @test p isa PISO{Float64}
        @test p.n_correctors == 2

        pm = PIMPLE()
        @test pm isa PIMPLE{Float64}
        @test pm.n_outer == 2
        @test pm.n_correctors == 1
        @test pm.alpha_U == 0.7
        @test pm.alpha_p == 0.3

        # IncompressibleState construction
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        state = IncompressibleState(mesh)
        @test length(state.U.internal) == 9
        @test length(state.p.internal) == 9
        @test all(==(1.0), state.A_P)
        @test all(==(zero(SVector{2, Float64})), state.H_U)
    end

    # ── 2. BC expansion ───────────────────────────────────────────────
    @testset "BC expansion" begin
        # FixedVelocityBC -> Dirichlet on velocity, Neumann on pressure
        fv = FixedVelocityBC((1.0, 2.0))
        @test fv.value == SVector(1.0, 2.0)
        vel_bc_x = FiniteVolumeMethod.expand_velocity_bc(fv, 1)
        vel_bc_y = FiniteVolumeMethod.expand_velocity_bc(fv, 2)
        @test vel_bc_x isa ParabolicDirichlet
        @test vel_bc_x.value == 1.0
        @test vel_bc_y.value == 2.0
        p_bc = FiniteVolumeMethod.expand_pressure_bc(fv)
        @test p_bc isa ParabolicNeumann
        @test p_bc.value == 0.0

        # FixedPressureBC -> Neumann on velocity, Dirichlet on pressure
        fp = FixedPressureBC(5.0)
        vel_bc = FiniteVolumeMethod.expand_velocity_bc(fp, 1)
        @test vel_bc isa ParabolicNeumann
        p_bc2 = FiniteVolumeMethod.expand_pressure_bc(fp)
        @test p_bc2 isa ParabolicDirichlet
        @test p_bc2.value == 5.0

        # NoSlipWallBC -> Dirichlet(0) on velocity, Neumann on pressure
        ns = NoSlipWallBC()
        @test FiniteVolumeMethod.expand_velocity_bc(ns, 1) isa ParabolicDirichlet
        @test FiniteVolumeMethod.expand_velocity_bc(ns, 1).value == 0.0
        @test FiniteVolumeMethod.expand_pressure_bc(ns) isa ParabolicNeumann

        # SlipWallBC -> Neumann on velocity, Neumann on pressure
        sw = SlipWallBC()
        @test FiniteVolumeMethod.expand_velocity_bc(sw, 1) isa ParabolicNeumann
        @test FiniteVolumeMethod.expand_pressure_bc(sw) isa ParabolicNeumann
    end

    # ── 3. Momentum assembly smoke test ────────────────────────────────
    @testset "Momentum assembly smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = SIMPLE()
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.01)
        state = IncompressibleState(mesh)
        FiniteVolumeMethod.update_boundary_velocity!(state, bcs, mesh)

        eq = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_momentum!(eq, state, prob, 1)

        # A should be nonzero (has diffusion contributions at minimum)
        @test nnz(eq.A) > 0
        # Diagonal should be positive (diffusion adds positive diagonal)
        nc = length(mesh.cell_volumes)
        for c in 1:nc
            @test eq.A[c, c] > 0
        end
    end

    # ── 4. Pressure equation smoke test ────────────────────────────────
    @testset "Pressure assembly smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = SIMPLE()
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.01)
        state = IncompressibleState(mesh)
        FiniteVolumeMethod.update_boundary_velocity!(state, bcs, mesh)
        FiniteVolumeMethod.update_boundary_pressure!(state, bcs, mesh)

        # Need momentum operators for pressure assembly
        eqs = FiniteVolumeMethod.CollocatedEquation{Float64}[]
        for d in 1:2
            eq = FiniteVolumeMethod.CollocatedEquation(mesh)
            FiniteVolumeMethod.assemble_momentum!(eq, state, prob, d)
            push!(eqs, eq)
        end
        FiniteVolumeMethod.extract_momentum_operators!(state, eqs, mesh)

        p_eq = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_pressure!(p_eq, state, prob)

        @test nnz(p_eq.A) > 0
        nc = length(mesh.cell_volumes)
        # Pressure Laplacian should have positive diagonal
        for c in 1:nc
            @test p_eq.A[c, c] >= 0
        end
    end

    # ── 5. Pressure reference fix ──────────────────────────────────────
    @testset "Pressure reference fix" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        eq = FiniteVolumeMethod.CollocatedEquation(mesh)
        nc = length(mesh.cell_volumes)

        # Put some values in the matrix first
        for c in 1:nc
            eq.A[c, c] = 10.0
        end
        eq.b[1] = 5.0

        FiniteVolumeMethod.fix_pressure_reference!(eq, 1, 0.0)

        # Row 1 should be zeroed except diagonal = 1
        @test eq.A[1, 1] == 1.0
        for j in 2:nc
            @test eq.A[1, j] == 0.0
        end
        @test eq.b[1] == 0.0
        # Other rows should be untouched
        @test eq.A[2, 2] == 10.0
    end

    # ── 6. SIMPLE convergence ─────────────────────────────────────────
    @testset "SIMPLE convergence" begin
        mesh = build_cartesian_unstructured_mesh(8, 4, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = SIMPLE(; max_iterations = 100, tolerance = 1.0e-6)
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.1)

        result = solve_simple(prob)

        @test result isa SolveResult{2, Float64}
        @test result.iterations > 0

        # Residual histories should be populated and finite
        @test haskey(result.residuals, :Ux)
        @test haskey(result.residuals, :Uy)
        @test haskey(result.residuals, :continuity)
        @test all(isfinite, result.residuals[:Ux])
        @test all(isfinite, result.residuals[:continuity])

        # Ux residual should decrease from first to last (solver is stable)
        @test result.residuals[:Ux][end] < result.residuals[:Ux][1]

        # Velocity should be positive-x at interior
        u_x = [result.state.U.internal[c][1] for c in eachindex(result.state.U.internal)]
        @test all(u_x .>= -0.01)
    end

    # ── 7. PISO transient smoke test ───────────────────────────────────
    @testset "PISO transient smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = PISO(; n_correctors = 2)
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.1)

        result = solve_incompressible(prob, (0.0, 0.02), 0.01)

        @test result isa SolveResult{2, Float64}
        @test result.converged  # transient always returns true
        @test result.iterations == 2
        @test length(result.residuals[:continuity]) == 2
    end

    # ── 8. PIMPLE transient smoke test ─────────────────────────────────
    @testset "PIMPLE transient smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = PIMPLE(; n_outer = 2, n_correctors = 1)
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.1)

        result = solve_incompressible(prob, (0.0, 0.02), 0.01)

        @test result isa SolveResult{2, Float64}
        @test result.converged
        @test result.iterations == 2
        @test length(result.residuals[:continuity]) == 2
    end

    # ── 9. Residual computation ────────────────────────────────────────
    @testset "Residual computation" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        state = IncompressibleState(mesh)
        # Zero fluxes should give zero continuity residual
        r = FiniteVolumeMethod.continuity_residual(state, mesh)
        @test r == 0.0

        # Nonzero flux should give nonzero residual
        nf = size(mesh.face_cells, 2)
        for f in 1:nf
            state.phi.values[f] = 0.01
        end
        r2 = FiniteVolumeMethod.continuity_residual(state, mesh)
        @test r2 > 0.0
    end
end
