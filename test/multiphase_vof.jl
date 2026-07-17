using FiniteVolumeMethod
using FiniteVolumeMethod: CavitationProperties, KunzModel, SolveResult, assemble_alpha!, clip_alpha!, compute_compression_flux, compute_curvature, compute_surface_tension_force, compute_vapor_source, has_surface_tension, update_mixture_properties!
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC
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
            :left => NeumannBC(0.0),
            :right => NeumannBC(0.0),
            :bottom => NeumannBC(0.0),
            :top => NeumannBC(0.0),
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
            :left => NeumannBC(0.0),
            :right => FixedPressureBC(0.0),
            :bottom => NeumannBC(0.0),
            :top => NeumannBC(0.0),
        )
        bcs_alpha = Dict{Symbol, AbstractBoundaryCondition}(
            :left => NeumannBC(0.0),
            :right => NeumannBC(0.0),
            :bottom => NeumannBC(0.0),
            :top => NeumannBC(0.0),
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

    # ── 14. MULES honors alpha BCs at inflow faces ─────────────────────
    @testset "MULES inflow alpha BC (water injection)" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        nf = size(mesh.face_cells, 2)
        dt = 0.1
        u_in = 0.25

        # Uniform +x flow: left boundary faces are INFLOW (F_f < 0
        # against the outward normal), right boundary faces are outflow.
        state = IncompressibleState(mesh)
        for f in 1:nf
            S_f = FiniteVolumeMethod.face_normal_area(mesh, f)
            state.phi.values[f] = u_in * S_f[1]
        end

        alpha = FiniteVolumeMethod.CollocatedScalarField(:alpha, mesh; value = 0.0)
        bcs_alpha = Dict{Symbol, AbstractBoundaryCondition}(
            :left => DirichletBC(1.0),   # water injected at inlet
            :right => NeumannBC(0.0),
            :bottom => NeumannBC(0.0),
            :top => NeumannBC(0.0),
        )

        eq = FiniteVolumeMethod.CollocatedEquation(mesh)
        assemble_alpha!(
            eq, alpha, state.phi, mesh, bcs_alpha;
            dt = dt, C_alpha = 1.0, use_mules = true,
        )
        sol = solve(FiniteVolumeMethod.to_linear_problem(eq))

        # Inflowing alpha equals the BC value: the first column of cells
        # receives water; total liquid volume grows at exactly the
        # injection rate (u_in * inlet_area * alpha_in * dt).
        V_new = sum(mesh.cell_volumes[c] * sol.u[c] for c in 1:nc)
        @test V_new ≈ u_in * 1.0 * 1.0 * dt rtol = 1.0e-12

        # Only the inlet-adjacent column has picked up alpha in one step
        for c in 1:nc
            x = mesh.cell_centers[1, c]
            if x < 0.25
                @test sol.u[c] > 0.0
            else
                @test abs(sol.u[c]) < 1.0e-14
            end
        end

        # Regression guard: with a zero-Dirichlet inlet BC nothing enters
        bcs_zero = Dict{Symbol, AbstractBoundaryCondition}(
            :left => DirichletBC(0.0),
            :right => NeumannBC(0.0),
            :bottom => NeumannBC(0.0),
            :top => NeumannBC(0.0),
        )
        eq0 = FiniteVolumeMethod.CollocatedEquation(mesh)
        assemble_alpha!(
            eq0, alpha, state.phi, mesh, bcs_zero;
            dt = dt, C_alpha = 1.0, use_mules = true,
        )
        sol0 = solve(FiniteVolumeMethod.to_linear_problem(eq0))
        @test all(x -> abs(x) < 1.0e-14, sol0.u)
    end
end

# ── Cavitation coupling in the VOF solver ─────────────────────────────
@testset "VOF cavitation coupling" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    cav_props = CavitationProperties(; rho_l = 1000.0, rho_v = 0.02, p_sat = 2300.0)
    kunz = KunzModel(; C_v = 100.0, C_c = 100.0, U_inf = 1.0, L_inf = 1.0)
    bcs_zero = Dict{Symbol, AbstractBoundaryCondition}()
    phi0 = FiniteVolumeMethod.FaceFluxField(:phi, mesh; value = 0.0)
    dt = 1.0e-4

    @testset "Frozen-p vapor growth matches implicit-analytic Kunz rate" begin
        alpha0 = 0.9
        alpha = FiniteVolumeMethod.CollocatedScalarField(:alpha, mesh; value = alpha0)
        p_field = fill(1000.0, nc)              # < p_sat -> evaporation
        alpha_v = fill(1.0 - alpha0, nc)
        mdot = compute_vapor_source(kunz, p_field, alpha_v, mesh, cav_props)
        @test all(>(0.0), mdot)                  # vapor produced

        eq = FiniteVolumeMethod.CollocatedEquation(mesh)
        assemble_alpha!(
            eq, alpha, phi0, mesh, bcs_zero;
            dt = dt, use_mules = true, mdot_v = mdot, rho_l = cav_props.rho_l,
        )
        sol = solve(FiniteVolumeMethod.to_linear_problem(eq), LUFactorization())
        # Patankar-implicit destruction: a_new = a0 / (1 + dt*mdot/(rho_l*a0))
        alpha_impl = alpha0 / (1.0 + dt * mdot[1] / (cav_props.rho_l * alpha0))
        @test isapprox(sol.u[1], alpha_impl; rtol = 1.0e-12)
        @test 0.0 <= sol.u[1] <= alpha0

        # Liquid-mass budget: rho_l dα/dt == -mdot_eff (Patankar effective)
        mdot_eff = mdot[1] * sol.u[1] / alpha0
        @test isapprox(
            cav_props.rho_l * (sol.u[1] - alpha0) / dt, -mdot_eff;
            rtol = 1.0e-10,
        )
    end

    @testset "Condensation branch is explicit and exact" begin
        alpha = FiniteVolumeMethod.CollocatedScalarField(:alpha, mesh; value = 0.7)
        p_field = fill(5000.0, nc)              # > p_sat -> condensation
        alpha_v = fill(0.3, nc)
        mdot = compute_vapor_source(kunz, p_field, alpha_v, mesh, cav_props)
        @test all(<(0.0), mdot)                  # vapor destroyed

        eq = FiniteVolumeMethod.CollocatedEquation(mesh)
        assemble_alpha!(
            eq, alpha, phi0, mesh, bcs_zero;
            dt = dt, use_mules = true, mdot_v = mdot, rho_l = cav_props.rho_l,
        )
        sol = solve(FiniteVolumeMethod.to_linear_problem(eq), LUFactorization())
        expected = 0.7 + dt * (-mdot[1]) / cav_props.rho_l
        @test isapprox(sol.u[1], expected; rtol = 1.0e-12)
    end

    @testset "Cavitating solve_vof: bounded, finite, vapor grows" begin
        props_int = CavitationProperties(; rho_l = 1000.0, rho_v = 1.0, p_sat = 100.0)
        model_int = KunzModel(; C_v = 0.01, C_c = 0.01, U_inf = 1.0, L_inf = 1.0)
        props2p = TwoPhaseProperties(;
            rho1 = 1000.0, rho2 = 1.0, mu1 = 1.0e-3, mu2 = 1.0e-5, sigma = 0.0,
        )
        bcs_U = Dict{Symbol, AbstractBoundaryCondition}(
            :left => NoSlipWallBC(), :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(), :top => NoSlipWallBC(),
        )
        result, vof = solve_vof(
            mesh, props2p, bcs_U, bcs_U, bcs_zero, (0.0, 50 * dt), dt;
            alpha_init = 0.95, algorithm = PISO(), linear_solver = LUFactorization(),
            cavitation_model = model_int, cavitation_props = props_int,
        )
        @test all(isfinite, vof.alpha.internal)
        @test all(u -> all(isfinite, u), result.state.U.internal)
        @test all(a -> 0.0 <= a <= 1.0, vof.alpha.internal)
        # p < p_sat everywhere at start -> net evaporation: liquid decreases
        @test sum(vof.alpha.internal) / nc < 0.95
    end

    @testset "Zero-rate cavitation model == cavitation-free (regression)" begin
        model_off = KunzModel(; C_v = 0.0, C_c = 0.0, U_inf = 1.0, L_inf = 1.0)
        props_off = CavitationProperties(; rho_l = 1000.0, rho_v = 1.0, p_sat = 100.0)
        props2p = TwoPhaseProperties(;
            rho1 = 1000.0, rho2 = 1.0, mu1 = 1.0e-3, mu2 = 1.0e-5, sigma = 0.0,
        )
        bcs_U = Dict{Symbol, AbstractBoundaryCondition}(
            :left => NoSlipWallBC(), :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(), :top => NoSlipWallBC(),
        )
        r1, v1 = solve_vof(
            mesh, props2p, bcs_U, bcs_U, bcs_zero, (0.0, 10 * dt), dt;
            alpha_init = 0.95, algorithm = PISO(), linear_solver = LUFactorization(),
        )
        r2, v2 = solve_vof(
            mesh, props2p, bcs_U, bcs_U, bcs_zero, (0.0, 10 * dt), dt;
            alpha_init = 0.95, algorithm = PISO(), linear_solver = LUFactorization(),
            cavitation_model = model_off, cavitation_props = props_off,
        )
        @test v1.alpha.internal == v2.alpha.internal
        for c in 1:nc
            @test r1.state.U.internal[c] == r2.state.U.internal[c]
        end
    end
end
