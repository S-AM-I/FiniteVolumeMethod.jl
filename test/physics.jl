using FiniteVolumeMethod
using DelaunayTriangulation
using Test

@testset "Physics Models" begin

    @testset "Anisotropic Diffusion" begin
        @testset "AnisotropicDiffusionEquation construction" begin
            # Create mesh
            tri = triangulate_rectangle(0, 1, 0, 1, 10, 10, single_boundary=true)
            mesh = FVMGeometry(tri)

            # Dirichlet boundary conditions
            bc = (x, y, t, u, p) -> 0.0
            BCs = BoundaryConditions(mesh, bc, Dirichlet)

            # Isotropic diffusion tensor (should behave like regular diffusion)
            diffusion_tensor = (x, y, p) -> (1.0, 0.0, 1.0)  # D = I

            # Initial condition
            u0 = [exp(-((x-0.5)^2 + (y-0.5)^2)/0.01) for (x, y) in DelaunayTriangulation.each_point(tri)]

            prob = AnisotropicDiffusionEquation(
                mesh, BCs;
                diffusion_tensor = diffusion_tensor,
                initial_condition = u0,
                final_time = 0.1
            )

            @test prob isa AnisotropicDiffusionEquation
            @test prob.mesh === mesh
            @test prob.final_time ≈ 0.1
        end

        @testset "make_rotation_tensor" begin
            # No rotation - should give diagonal tensor
            D_func = make_rotation_tensor(0.0, 2.0, 0.5)
            Dxx, Dxy, Dyy = D_func(0.0, 0.0, nothing)
            @test Dxx ≈ 2.0
            @test Dxy ≈ 0.0 atol=1e-10
            @test Dyy ≈ 0.5

            # 90 degree rotation - should swap coefficients
            D_func_90 = make_rotation_tensor(π/2, 2.0, 0.5)
            Dxx, Dxy, Dyy = D_func_90(0.0, 0.0, nothing)
            @test Dxx ≈ 0.5 atol=1e-10
            @test Dxy ≈ 0.0 atol=1e-10
            @test Dyy ≈ 2.0 atol=1e-10

            # 45 degree rotation - should give mixed terms
            D_func_45 = make_rotation_tensor(π/4, 2.0, 0.5)
            Dxx, Dxy, Dyy = D_func_45(0.0, 0.0, nothing)
            # Average of 2 and 0.5 for diagonal
            @test Dxx ≈ 1.25 atol=1e-10
            @test Dyy ≈ 1.25 atol=1e-10
            # Off-diagonal should be non-zero
            @test abs(Dxy) > 0.5
        end

        @testset "make_spatially_varying_tensor" begin
            Dxx_func = (x, y) -> 1.0 + x
            Dxy_func = (x, y) -> 0.1 * x * y
            Dyy_func = (x, y) -> 1.0 + y

            D_func = make_spatially_varying_tensor(Dxx_func, Dxy_func, Dyy_func)

            Dxx, Dxy, Dyy = D_func(2.0, 3.0, nothing)
            @test Dxx ≈ 3.0  # 1 + 2
            @test Dxy ≈ 0.6  # 0.1 * 2 * 3
            @test Dyy ≈ 4.0  # 1 + 3
        end
    end

    @testset "k-epsilon Turbulence" begin
        @testset "StandardKEpsilon construction" begin
            # Default model
            model = StandardKEpsilon()
            @test model.C_mu ≈ 0.09
            @test model.sigma_k ≈ 1.0
            @test model.sigma_epsilon ≈ 1.3
            @test model.C1_epsilon ≈ 1.44
            @test model.C2_epsilon ≈ 1.92
            @test model.kappa ≈ 0.41

            # Custom model
            model_custom = StandardKEpsilon(C_mu=0.1, C1_epsilon=1.5)
            @test model_custom.C_mu ≈ 0.1
            @test model_custom.C1_epsilon ≈ 1.5
        end

        @testset "compute_turbulent_viscosity" begin
            model = StandardKEpsilon()

            # Scalar inputs
            k = 1.0
            epsilon = 0.1
            nu_t = compute_turbulent_viscosity(model, k, epsilon)
            @test nu_t ≈ 0.09 * 1.0^2 / 0.1  # C_mu * k^2 / epsilon = 0.9

            # With density
            rho = 1000.0
            mu_t = compute_turbulent_viscosity(model, k, epsilon; rho=rho)
            @test mu_t ≈ rho * 0.09 * 1.0^2 / 0.1

            # Vector inputs
            k_vec = [1.0, 2.0, 0.5]
            eps_vec = [0.1, 0.2, 0.05]
            nu_t_vec = compute_turbulent_viscosity(model, k_vec, eps_vec)
            @test length(nu_t_vec) == 3
            @test nu_t_vec[1] ≈ 0.09 * 1.0^2 / 0.1
            @test nu_t_vec[2] ≈ 0.09 * 4.0 / 0.2
            @test nu_t_vec[3] ≈ 0.09 * 0.25 / 0.05

            # Safeguard against zero epsilon
            nu_t_safe = compute_turbulent_viscosity(model, 1.0, 0.0)
            @test isfinite(nu_t_safe)
        end

        @testset "compute_strain_rate_magnitude" begin
            # Pure shear: du/dy = 1, all else zero
            grad_u = (0.0, 1.0)
            grad_v = (0.0, 0.0)
            S_mag = compute_strain_rate_magnitude(grad_u, grad_v)
            # S_xy = 0.5, |S|^2 = 2 * 2 * 0.25 = 1, |S| = 1
            @test S_mag ≈ 1.0

            # Pure extension: du/dx = 1, dv/dy = -1 (incompressible)
            grad_u = (1.0, 0.0)
            grad_v = (0.0, -1.0)
            S_mag = compute_strain_rate_magnitude(grad_u, grad_v)
            # S_xx = 1, S_yy = -1, S_xy = 0
            # |S|^2 = 2*(1 + 1) = 4, |S| = 2
            @test S_mag ≈ 2.0

            # Zero velocity gradient
            grad_u = (0.0, 0.0)
            grad_v = (0.0, 0.0)
            S_mag = compute_strain_rate_magnitude(grad_u, grad_v)
            @test S_mag ≈ 0.0
        end

        @testset "compute_production" begin
            nu_t = 0.1
            S_mag = 2.0
            Pk = compute_production(nu_t, S_mag)
            @test Pk ≈ 0.1 * 4.0  # nu_t * S^2
        end

        @testset "compute_friction_velocity" begin
            # Viscous sublayer (low u_tan, small y)
            u_tau = compute_friction_velocity(0.1, 0.001, 1e-6)
            @test isfinite(u_tau)
            @test u_tau > 0

            # Log layer (higher Reynolds number)
            u_tau_log = compute_friction_velocity(10.0, 0.01, 1e-6)
            @test isfinite(u_tau_log)
            @test u_tau_log > 0
            @test u_tau_log < 10.0  # Must be less than bulk velocity
        end

        @testset "k_wall_value and epsilon_wall_value" begin
            u_tau = 0.5

            k_w = k_wall_value(u_tau)
            @test k_w ≈ u_tau^2 / sqrt(0.09)

            y = 0.01
            eps_w = epsilon_wall_value(u_tau, y)
            @test isfinite(eps_w)
            @test eps_w > 0

            # Epsilon should increase as y decreases
            eps_w_closer = epsilon_wall_value(u_tau, y/2)
            @test eps_w_closer > eps_w
        end

        @testset "TurbulentWallBC" begin
            bc = TurbulentWallBC()
            @test bc.roughness ≈ 0.0
            @test bc.C_mu ≈ 0.09
            @test bc.kappa ≈ 0.41

            bc_rough = TurbulentWallBC(roughness=0.001)
            @test bc_rough.roughness ≈ 0.001
        end

        @testset "KappaOmegaSST" begin
            model = KappaOmegaSST()
            @test model.a1 ≈ 0.31
            @test model.beta_star ≈ 0.09
            @test model.sigma_k1 ≈ 0.85
            @test model.sigma_omega1 ≈ 0.5
        end
    end
end
