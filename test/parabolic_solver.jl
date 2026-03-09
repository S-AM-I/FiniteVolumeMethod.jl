using Test
using FiniteVolumeMethod
using SparseArrays
using LinearAlgebra

@testset "Parabolic Solver" begin
    @testset "1D Diffusion Assembly" begin
        # Simple 1D diffusion: d^2T/dx^2 = 0, T(0)=0, T(1)=1
        # Exact solution: T(x) = x
        mesh = generate_mesh_1d(0.0, 1.0, 10)
        model = Diffusion1D(1.0)
        bc_left = ParabolicDirichlet(0.0)
        bc_right = ParabolicDirichlet(1.0)

        A, b = assemble_system(model, mesh, bc_left, bc_right)

        @test size(A) == (10, 10)
        @test length(b) == 10

        # Solve
        phi = A \ b

        # Check solution: should be approximately linear
        for i in 1:10
            x_center = mesh.cells[i].center
            @test phi[i] ≈ x_center atol = 0.05
        end
    end

    @testset "1D Mass Matrix Assembly" begin
        mesh = generate_mesh_1d(0.0, 1.0, 5)
        model = Diffusion1D(1.0)

        # Use the generic mass matrix assembler from utils
        M = assemble_mass_matrix_1d(mesh)
        @test size(M) == (5, 5)

        # Check diagonal: each cell volume should be dx = 0.2
        dx = 1.0 / 5
        for i in 1:5
            @test M[i, i] ≈ dx atol = 1.0e-12
        end
    end

    @testset "1D Advection Assembly" begin
        mesh = generate_mesh_1d(0.0, 1.0, 10)
        model = Advection1D(1.0)  # rightward velocity
        bc_left = ParabolicDirichlet(1.0)
        bc_right = OutflowBC()

        A, b = assemble_system(model, mesh, bc_left, bc_right)

        @test size(A) == (10, 10)
        @test length(b) == 10
    end

    @testset "1D Advection-Diffusion Assembly" begin
        mesh = generate_mesh_1d(0.0, 1.0, 10)
        model = AdvectionDiffusion1D(Advection1D(1.0), Diffusion1D(0.01))
        bc_left = ParabolicDirichlet(1.0)
        bc_right = ParabolicDirichlet(0.0)

        A, b = assemble_system(model, mesh, bc_left, bc_right)

        @test size(A) == (10, 10)
        @test length(b) == 10

        # Solve and verify bounded solution
        phi = A \ b
        @test all(x -> -0.1 <= x <= 1.1, phi)
    end

    @testset "Model Types" begin
        # Test constructors
        @test Diffusion1D(1.0).gamma == 1.0
        @test Diffusion2D(2.0).gamma == 2.0
        @test Diffusion3D(3.0).gamma == 3.0
        @test Advection1D(1.5).v == 1.5
        @test Advection2D(1.0, 2.0).vx == 1.0
        @test Advection2D(1.0, 2.0).vy == 2.0
        @test CylindricalDiffusion1D(0.5).gamma == 0.5
        @test SphericalDiffusion1D(0.3).gamma == 0.3

        # Test turbulence model
        ke = ParabolicKEpsilon()
        @test ke.C_mu ≈ 0.09
        @test ke.sigma_k ≈ 1.0
        @test ke.C1_epsilon ≈ 1.44
    end

    @testset "Source Terms" begin
        mesh = generate_mesh_1d(0.0, 1.0, 5)

        # Constant source
        cs = ConstantSource(2.0)
        @test evaluate_source(cs, mesh, 1) == 2.0

        # Spatial source
        ss = SpatialSource([1.0, 2.0, 3.0, 4.0, 5.0])
        @test evaluate_source(ss, mesh, 3) == 3.0

        # Function source
        fs = FunctionSource(x -> x^2)
        val = evaluate_source(fs, mesh, 1)
        @test val ≈ mesh.cells[1].center^2
    end

    @testset "Boundary Condition Types" begin
        d = ParabolicDirichlet(1.0)
        @test d.value == 1.0

        n = ParabolicNeumann(0.5)
        @test n.value == 0.5

        r = ParabolicRobin(1.0, 2.0, 3.0)
        @test r.a == 1.0
        @test r.b == 2.0
        @test r.c == 3.0

        o = OutflowBC()
        @test o.type == :zero_gradient

        p = ParabolicPeriodicBC(:left, :right)
        @test p.pair == (:left, :right)
    end

    @testset "Cylindrical Diffusion 1D" begin
        # Radial diffusion in cylindrical coordinates
        mesh = generate_mesh_1d(0.1, 1.0, 10)  # r from 0.1 to 1.0
        model = CylindricalDiffusion1D(1.0)
        bc_left = ParabolicNeumann(0.0)
        bc_right = ParabolicDirichlet(1.0)

        A, b = assemble_system(model, mesh, bc_left, bc_right)

        @test size(A) == (10, 10)

        # Mass matrix
        M = assemble_mass_matrix(mesh, model)
        @test size(M) == (10, 10)
        # All diagonal entries should be positive
        for i in 1:10
            @test M[i, i] > 0
        end
    end

    @testset "Spherical Diffusion 1D" begin
        mesh = generate_mesh_1d(0.1, 1.0, 10)  # r from 0.1 to 1.0
        model = SphericalDiffusion1D(1.0)
        bc_left = ParabolicNeumann(0.0)
        bc_right = ParabolicDirichlet(1.0)

        A, b = assemble_system(model, mesh, bc_left, bc_right)

        @test size(A) == (10, 10)

        # Mass matrix
        M = assemble_mass_matrix(mesh, model)
        @test size(M) == (10, 10)
        for i in 1:10
            @test M[i, i] > 0
        end
    end

    @testset "Compressible Fluxes" begin
        # Test ideal gas pressure
        rho = 1.0
        rho_v = 0.0
        rho_E = 2.5
        gamma_gas = 1.4
        p = ideal_gas_pressure(rho, rho_v, rho_E, gamma_gas)
        @test p ≈ (gamma_gas - 1) * rho_E atol = 1.0e-12

        # Test sound speed
        c = parabolic_sound_speed(p, rho, gamma_gas)
        @test c > 0
        @test c ≈ sqrt(gamma_gas * p / rho) atol = 1.0e-12
    end

    @testset "ParabolicLimiters submodule" begin
        # Test that the submodule is accessible
        @test ParabolicLimiters.minmod(1.0, 2.0) ≈ 1.0
        @test ParabolicLimiters.minmod(-1.0, 2.0) ≈ 0.0
        @test ParabolicLimiters.superbee(1.0, 2.0) > 0
        @test ParabolicLimiters.van_leer(1.0) > 0
    end

    @testset "Particle Types" begin
        p = LagrangianParticle(1, [0.5, 0.5], [1.0, 0.0], 1.0, 1.0e-5)
        @test p.id == 1
        @test p.position == [0.5, 0.5]
        @test p.velocity == [1.0, 0.0]
        @test p.density == 1.0
        @test p.diameter == 1.0e-5
    end

    @testset "FSI Types" begin
        sys = SpringMassSystem
        @test sys <: AbstractStructuralModel
    end

    @testset "Coupled System Assembly Types" begin
        @test LinearCoupling <: AbstractCoupling
        lc = LinearCoupling(1, 2, 0.5)
        @test lc.target_idx == 1
        @test lc.source_idx == 2
        @test lc.coeff == 0.5
    end
end
