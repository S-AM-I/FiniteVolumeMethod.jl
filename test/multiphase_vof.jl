using FiniteVolumeMethod
using Test
using LinearAlgebra
using LinearSolve
using StaticArrays
using SparseArrays

# ── Mesh builder (shared helper) ─────────────────────────────────────
include("TestHelpers.jl")

# ── Tests ─────────────────────────────────────────────────────────────

@testset "Multiphase VOF" begin

    # ── 1. TwoPhaseProperties defaults ────────────────────────────────
    @testset "TwoPhaseProperties defaults" begin
        props = TwoPhaseProperties()
        @test props isa TwoPhaseProperties{Float64}
        @test props.rho1 == 1000.0
        @test props.rho2 == 1.225
        @test props.mu1 == 1.0e-3
        @test props.mu2 == 1.8e-5
        @test props.sigma == 0.072
        @test has_surface_tension(props)
    end

    # ── 2. TwoPhaseProperties custom ──────────────────────────────────
    @testset "TwoPhaseProperties custom" begin
        props = TwoPhaseProperties(;
            rho1 = 800.0, rho2 = 2.0, mu1 = 5.0e-4, mu2 = 3.0e-5, sigma = 0.0,
        )
        @test props.rho1 == 800.0
        @test props.rho2 == 2.0
        @test props.mu1 == 5.0e-4
        @test props.mu2 == 3.0e-5
        @test props.sigma == 0.0
        @test !has_surface_tension(props)
    end

    # ── 3. VOFState uniform construction ──────────────────────────────
    @testset "VOFState uniform construction" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        vof = VOFState(mesh; alpha_init = 0.5)
        @test vof isa VOFState{Float64}
        @test length(vof.alpha.internal) == nc
        @test all(==(0.5), vof.alpha.internal)
    end

    # ── 4. VOFState function init ─────────────────────────────────────
    @testset "VOFState function init" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        props = TwoPhaseProperties()
        alpha_func = x -> x[1] < 0.5 ? 1.0 : 0.0
        vof = VOFState(mesh, alpha_func, props)

        @test length(vof.alpha.internal) == nc
        for c in 1:nc
            x_c = FiniteVolumeMethod.cell_center(mesh, c)
            if x_c[1] < 0.5
                @test vof.alpha.internal[c] == 1.0
            else
                @test vof.alpha.internal[c] == 0.0
            end
        end
    end

    # ── 5. update_mixture_properties! ─────────────────────────────────
    @testset "update_mixture_properties!" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        props = TwoPhaseProperties()

        # alpha = 1 everywhere -> rho = rho1, mu = mu1
        vof1 = VOFState(mesh; alpha_init = 1.0)
        update_mixture_properties!(vof1, props)
        @test all(==(props.rho1), vof1.rho)
        @test all(==(props.mu1), vof1.mu)

        # alpha = 0 everywhere -> rho = rho2, mu = mu2
        vof0 = VOFState(mesh; alpha_init = 0.0)
        update_mixture_properties!(vof0, props)
        @test all(==(props.rho2), vof0.rho)
        @test all(==(props.mu2), vof0.mu)

        # alpha = 0.5 -> arithmetic average
        vof_half = VOFState(mesh; alpha_init = 0.5)
        update_mixture_properties!(vof_half, props)
        expected_rho = 0.5 * props.rho1 + 0.5 * props.rho2
        expected_mu = 0.5 * props.mu1 + 0.5 * props.mu2
        @test all(x -> x ≈ expected_rho, vof_half.rho)
        @test all(x -> x ≈ expected_mu, vof_half.mu)
    end

    # ── 6. clip_alpha! within bounds ──────────────────────────────────
    @testset "clip_alpha! within bounds" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        vof = VOFState(mesh; alpha_init = 0.3)
        original = copy(vof.alpha.internal)
        clip_alpha!(vof.alpha, mesh)
        @test vof.alpha.internal == original
    end

    # ── 7. clip_alpha! clips and conserves ────────────────────────────
    @testset "clip_alpha! clips and conserves" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        vof = VOFState(mesh; alpha_init = 0.5)

        # Perturb a few cells outside [0,1]
        vof.alpha.internal[1] = 1.5
        vof.alpha.internal[2] = -0.3

        total_before = sum(vof.alpha.internal[c] * mesh.cell_volumes[c] for c in 1:nc)

        clip_alpha!(vof.alpha, mesh)

        # All values must be in [0,1]
        @test all(x -> 0.0 <= x <= 1.0, vof.alpha.internal)

        # Conservation: total alpha*V should be preserved
        total_after = sum(vof.alpha.internal[c] * mesh.cell_volumes[c] for c in 1:nc)
        @test total_after ≈ total_before atol = 1.0e-10
    end

    # ── 8. assemble_alpha! smoke ──────────────────────────────────────
    @testset "assemble_alpha! smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        vof = VOFState(mesh; alpha_init = 0.5)
        state = IncompressibleState(mesh)
        bcs_alpha = Dict{Symbol, AbstractBoundaryCondition}(
            :left => ParabolicNeumann(0.0),
            :right => ParabolicNeumann(0.0),
            :bottom => ParabolicNeumann(0.0),
            :top => ParabolicNeumann(0.0),
        )

        eq = FiniteVolumeMethod.CollocatedEquation(mesh)
        assemble_alpha!(
            eq, vof.alpha, state.phi, mesh, bcs_alpha;
            dt = 0.01, C_alpha = 1.0,
        )

        @test nnz(eq.A) > 0
    end

    # ── 9. compute_compression_flux uniform ───────────────────────────
    @testset "compute_compression_flux uniform" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        vof = VOFState(mesh; alpha_init = 0.5)
        state = IncompressibleState(mesh)

        phi_c = compute_compression_flux(vof.alpha, state.phi, mesh; C_alpha = 1.0)

        # Uniform alpha has zero gradient -> compression flux should be zero
        @test all(x -> abs(x) < 1.0e-12, phi_c)
    end

    # ── 10. compute_curvature uniform ─────────────────────────────────
    @testset "compute_curvature uniform" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        vof = VOFState(mesh; alpha_init = 0.5)

        kappa = compute_curvature(vof.alpha, mesh)

        @test length(kappa) == length(mesh.cell_volumes)
        @test all(x -> abs(x) < 1.0e-10, kappa)
    end

    # ── 11. compute_surface_tension_force sigma=0 ─────────────────────
    @testset "compute_surface_tension_force sigma=0" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        props = TwoPhaseProperties(; sigma = 0.0)
        vof = VOFState(mesh; alpha_init = 0.5)

        force = compute_surface_tension_force(vof.alpha, props, mesh)
        @test force === nothing
    end

    # ── 12. compute_surface_tension_force sigma>0 ─────────────────────
    @testset "compute_surface_tension_force sigma>0" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        props = TwoPhaseProperties(; sigma = 0.072)
        # Create an interface: left half alpha=1, right half alpha=0
        alpha_func = x -> x[1] < 0.5 ? 1.0 : 0.0
        vof = VOFState(mesh, alpha_func, props)

        force = compute_surface_tension_force(vof.alpha, props, mesh)

        @test force !== nothing
        @test force isa Vector{SVector{2, Float64}}
        @test length(force) == length(mesh.cell_volumes)
        # At least some interface cells should have nonzero force
        @test any(f -> norm(f) > 0, force)
    end

    # ── 13. solve_vof smoke ───────────────────────────────────────────
    @testset "solve_vof smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        props = TwoPhaseProperties()

        bcs_U = Dict{Symbol, AbstractBoundaryCondition}(
            :left => NoSlipWallBC(),
            :right => NoSlipWallBC(),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        bcs_p = Dict{Symbol, AbstractBoundaryCondition}(
            :left => ParabolicNeumann(0.0),
            :right => FixedPressureBC(0.0),
            :bottom => ParabolicNeumann(0.0),
            :top => ParabolicNeumann(0.0),
        )
        bcs_alpha = Dict{Symbol, AbstractBoundaryCondition}(
            :left => ParabolicNeumann(0.0),
            :right => ParabolicNeumann(0.0),
            :bottom => ParabolicNeumann(0.0),
            :top => ParabolicNeumann(0.0),
        )

        alpha_init_func = x -> x[1] < 0.5 ? 1.0 : 0.0
        g = SVector(0.0, -1.0)

        result, vof_state = solve_vof(
            mesh, props, bcs_U, bcs_p, bcs_alpha,
            (0.0, 0.02), 0.01;
            alpha_init = alpha_init_func,
            g = g,
            algorithm = PISO(; n_correctors = 1),
        )

        @test result isa SolveResult{2, Float64}
        @test vof_state isa VOFState{Float64}
        @test result.iterations == 2

        # Alpha should remain bounded
        @test all(x -> 0.0 <= x <= 1.0, vof_state.alpha.internal)

        # Mixture properties should be positive
        @test all(x -> x > 0, vof_state.rho)
        @test all(x -> x > 0, vof_state.mu)
    end
end
