using FiniteVolumeMethod
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC, RobinBC
using Test
using LinearAlgebra
using LinearSolve
using StaticArrays
using SparseArrays

# ── Mesh builder (shared helper) ─────────────────────────────────────
include("TestHelpers.jl")

# ── Tests ──────────────────────────────────────────────────────────────

@testset "Conjugate Heat Transfer & Buoyancy" begin

    # ── 1. Type construction ───────────────────────────────────────────
    @testset "Type construction" begin
        # FluidThermalProperties defaults
        ftp = FluidThermalProperties{2}()
        @test ftp isa FluidThermalProperties{2, Float64}
        @test ftp.Cp == 1005.0
        @test ftp.k == 0.026
        @test ftp.Pr_t == 0.85
        @test ftp.beta == 0.0
        @test ftp.T_ref == 300.0
        @test ftp.g == SVector(0.0, -9.81)

        # FluidThermalProperties with custom values
        ftp2 = FluidThermalProperties{2}(;
            Cp = 2000.0, k = 0.6, beta = 3.4e-3, T_ref = 293.0,
            g = (0.0, -9.81),
        )
        @test ftp2.Cp == 2000.0
        @test ftp2.k == 0.6
        @test ftp2.beta == 3.4e-3

        # SolidThermalProperties defaults
        stp = SolidThermalProperties()
        @test stp isa SolidThermalProperties{Float64}
        @test stp.rho == 7800.0
        @test stp.Cp == 500.0
        @test stp.k == 50.0
        @test stp.Q_gen == 0.0

        # SolidThermalProperties with custom values
        stp2 = SolidThermalProperties(; rho = 2700.0, Cp = 900.0, k = 200.0, Q_gen = 1.0e5)
        @test stp2.rho == 2700.0
        @test stp2.Q_gen == 1.0e5

        # ThermalState construction
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        ts = ThermalState(mesh; T_init = 350.0, k_init = 0.05)
        @test length(ts.T_field.internal) == 9
        @test all(==(350.0), ts.T_field.internal)
        @test length(ts.k_eff) == 9
        @test all(==(0.05), ts.k_eff)

        # has_buoyancy
        @test has_buoyancy(ftp) == false
        @test has_buoyancy(ftp2) == true
    end

    # ── 2. BC convenience constructors ─────────────────────────────────
    @testset "BC convenience constructors" begin
        bc_inlet = thermal_inlet_bc(400.0)
        @test bc_inlet isa DirichletBC
        @test bc_inlet.value == 400.0

        bc_insulated = thermal_insulated_bc()
        @test bc_insulated isa NeumannBC
        @test bc_insulated.value == 0.0

        bc_heated = thermal_heated_wall_bc(5000.0)
        @test bc_heated isa NeumannBC
        @test bc_heated.value == 5000.0

        bc_conv = thermal_convective_bc(10.0, 300.0)
        @test bc_conv isa RobinBC
        @test bc_conv.a == 10.0
        @test bc_conv.b == 1.0
        @test bc_conv.c == 10.0 * 300.0
    end

    # ── 3. compute_alpha_eff ───────────────────────────────────────────
    @testset "compute_alpha_eff" begin
        k_eff = [0.5, 1.0, 2.0]
        rho = 1000.0
        Cp = 4000.0
        alpha = compute_alpha_eff(k_eff, rho, Cp)

        @test length(alpha) == 3
        @test alpha[1] ≈ 0.5 / (1000.0 * 4000.0)
        @test alpha[2] ≈ 1.0 / (1000.0 * 4000.0)
        @test alpha[3] ≈ 2.0 / (1000.0 * 4000.0)
    end

    # ── 4. update_k_eff! ──────────────────────────────────────────────
    @testset "update_k_eff!" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        props = FluidThermalProperties{2}(; k = 0.026, Cp = 1005.0, Pr_t = 0.85)
        ts = ThermalState(mesh; T_init = 300.0, k_init = 999.0)

        # Laminar only (nu_t = nothing)
        update_k_eff!(ts, props, nothing, 1.0)
        @test all(==(0.026), ts.k_eff)

        # With turbulent viscosity
        nu_t = fill(0.001, nc)
        update_k_eff!(ts, props, nu_t, 1.2)
        for c in 1:nc
            expected = 0.026 + 1.2 * 1005.0 * 0.001 / 0.85
            @test ts.k_eff[c] ≈ expected
        end
    end

    # ── 5. Buoyancy source ─────────────────────────────────────────────
    @testset "Buoyancy source" begin
        mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        density = 1.0

        # With buoyancy: T > T_ref should push upward (opposite to g)
        props = FluidThermalProperties{2}(;
            beta = 3.4e-3, T_ref = 300.0, g = (0.0, -9.81),
        )
        T_field = FiniteVolumeMethod.CollocatedScalarField(:T, mesh; value = 310.0)
        force = compute_buoyancy_source(T_field, props, density)
        @test force !== nothing
        @test length(force) == nc

        # F_b = -rho * beta * (T - T_ref) * g
        # = -1.0 * 3.4e-3 * 10.0 * (0, -9.81)
        # = (0.0, 0.33354)
        for c in 1:nc
            @test force[c][1] ≈ 0.0 atol = 1.0e-15
            @test force[c][2] > 0.0  # upward
            @test force[c][2] ≈ density * props.beta * 10.0 * 9.81
        end

        # Without buoyancy (beta=0)
        props_no = FluidThermalProperties{2}(; beta = 0.0)
        force_no = compute_buoyancy_source(T_field, props_no, density)
        @test force_no === nothing
    end

    # ── 6. Energy assembly smoke ───────────────────────────────────────
    @testset "Energy assembly smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        T_field = FiniteVolumeMethod.CollocatedScalarField(:T, mesh; value = 300.0)
        phi = FiniteVolumeMethod.FaceFluxField(:phi, mesh)
        alpha_eff = fill(1.0e-5, nc)

        bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
            :left => thermal_inlet_bc(350.0),
            :right => thermal_insulated_bc(),
            :bottom => thermal_insulated_bc(),
            :top => thermal_insulated_bc(),
        )

        eq = FiniteVolumeMethod.CollocatedEquation(mesh)
        assemble_energy!(eq, T_field, phi, alpha_eff, mesh, bcs_T)

        @test nnz(eq.A) > 0
        # Diagonal should be nonzero from diffusion
        for c in 1:nc
            @test eq.A[c, c] != 0.0
        end
    end

    # ── 7. Solid conduction solve ──────────────────────────────────────
    @testset "Solid conduction solve" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)

        solid = SolidThermalProperties(; rho = 7800.0, Cp = 500.0, k = 50.0)
        bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
            :left => DirichletBC(400.0),
            :right => DirichletBC(300.0),
            :bottom => thermal_insulated_bc(),
            :top => thermal_insulated_bc(),
        )

        T_field = solve_solid_conduction(mesh, solid, bcs_T)

        @test T_field isa FiniteVolumeMethod.CollocatedScalarField{Float64}
        @test length(T_field.internal) == 16
        @test all(isfinite, T_field.internal)

        # Temperature should decrease from left to right
        # Cell (1,1) should be hotter than cell (4,1)
        # cell_idx(i, j) = (j-1)*4 + i
        for j in 1:4
            T_left = T_field.internal[(j - 1) * 4 + 1]
            T_right = T_field.internal[(j - 1) * 4 + 4]
            @test T_left > T_right
        end

        # All values should be between 300 and 400
        for c in 1:16
            @test 300.0 <= T_field.internal[c] <= 400.0
        end
    end

    # ── 8. solve_simple_thermal smoke ──────────────────────────────────
    @testset "solve_simple_thermal smoke" begin
        mesh = build_cartesian_unstructured_mesh(8, 4, 2.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = SIMPLE(; max_iterations = 5, tolerance = 1.0e-12)
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.1)

        bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
            :left => thermal_inlet_bc(350.0),
            :right => thermal_insulated_bc(),
            :bottom => thermal_inlet_bc(300.0),
            :top => thermal_inlet_bc(300.0),
        )

        result, thermal_state = solve_simple_thermal(
            prob, FluidThermalProperties{2}(; k = 0.6, Cp = 4000.0);
            bcs_T = bcs_T,
        )

        @test result isa SolveResult{2, Float64}
        @test thermal_state isa ThermalState{Float64}
        @test result.iterations == 5
        @test all(isfinite, thermal_state.T_field.internal)
    end

    # ── 9. PISO thermal smoke ──────────────────────────────────────────
    @testset "PISO thermal smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = PISO(; n_correctors = 2)
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.1)

        bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
            :left => thermal_inlet_bc(350.0),
            :right => thermal_insulated_bc(),
            :bottom => thermal_insulated_bc(),
            :top => thermal_insulated_bc(),
        )

        result, thermal_state = solve_incompressible_thermal(
            prob, FluidThermalProperties{2}(; k = 0.6, Cp = 4000.0),
            (0.0, 0.02), 0.01;
            bcs_T = bcs_T,
        )

        @test result isa SolveResult{2, Float64}
        @test thermal_state isa ThermalState{Float64}
        @test result.converged
        @test result.iterations == 2
        @test all(isfinite, thermal_state.T_field.internal)
    end

    # ── 10. Interface heat flux ────────────────────────────────────────
    @testset "Interface heat flux" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        # Set up a temperature field with a linear gradient left-to-right
        T_field = FiniteVolumeMethod.CollocatedScalarField(:T, mesh; value = 300.0)
        for c in 1:nc
            x = mesh.cell_centers[1, c]
            T_field.internal[c] = 400.0 - 100.0 * x  # 400 at x=0, 300 at x=1
        end
        # Set boundary values on the right patch
        bmap = FiniteVolumeMethod.build_boundary_map(T_field)
        nf = size(mesh.face_cells, 2)
        for f in 1:nf
            if !FiniteVolumeMethod.is_internal_face(mesh, f)
                tag = FiniteVolumeMethod._face_tag(mesh, f)
                if tag == :right
                    T_field.boundary[bmap[f]] = 300.0
                elseif tag == :left
                    T_field.boundary[bmap[f]] = 400.0
                else
                    # Top/bottom: extrapolate from cell
                    P = mesh.face_cells[1, f]
                    T_field.boundary[bmap[f]] = T_field.internal[P]
                end
            end
        end

        k = 50.0
        flux = compute_interface_heat_flux(T_field, k, mesh, :right)

        @test !isempty(flux)
        @test length(flux) == 4  # 4 right boundary faces
        for (f, q) in flux
            @test isfinite(q)
        end
    end

    # ── Buoyancy is kinematic: velocity independent of density ────────
    @testset "Boussinesq buoyancy in kinematic form" begin
        mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        props = FluidThermalProperties{2}(; k = 0.6, Cp = 4000.0, beta = 2.0e-4, T_ref = 300.0)

        # Unit test: the buoyancy source is per unit mass (no rho factor)
        T_field = FiniteVolumeMethod.CollocatedScalarField(:T, mesh; value = 310.0)
        f1 = compute_buoyancy_source(T_field, props, 1.0)
        f1000 = compute_buoyancy_source(T_field, props, 1000.0)
        @test f1 !== nothing && f1000 !== nothing
        for c in 1:nc
            @test f1[c] ≈ f1000[c] atol = 1.0e-15
            # -beta * dT * g with dT = 10, g = (0, -9.81)
            @test f1[c] ≈ SVector(0.0, 2.0e-4 * 10.0 * 9.81) atol = 1.0e-12
        end

        # End-to-end: heated-cavity velocity must not depend on density
        # when nu AND the thermal diffusivity alpha = k/(rho*Cp) are held
        # fixed (k scales with rho).  Under Boussinesq, rho then cancels
        # everywhere — the previous rho-scaled buoyancy force broke this.
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => NoSlipWallBC(), :right => NoSlipWallBC(),
            :bottom => NoSlipWallBC(), :top => NoSlipWallBC(),
        )
        bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
            :left => DirichletBC(310.0),
            :right => DirichletBC(290.0),
            :bottom => NeumannBC(0.0),
            :top => NeumannBC(0.0),
        )
        # tolerance = 0 forces all iterations: the initial state (U = 0,
        # uniform T) has identically zero residuals, so any positive
        # tolerance would exit before the temperature field develops.
        algo = SIMPLE(; max_iterations = 10, tolerance = 0.0)
        props_a = FluidThermalProperties{2}(;
            k = 0.6, Cp = 1.0, beta = 2.0e-4, T_ref = 300.0,
        )
        props_b = FluidThermalProperties{2}(;
            k = 600.0, Cp = 1.0, beta = 2.0e-4, T_ref = 300.0,
        )
        prob_a = IncompressibleProblem(mesh, bcs, algo; nu = 0.01, density = 1.0)
        prob_b = IncompressibleProblem(mesh, bcs, algo; nu = 0.01, density = 1000.0)
        ra, tsa = solve_simple_thermal(prob_a, props_a; bcs_T = bcs_T)
        rb, tsb = solve_simple_thermal(prob_b, props_b; bcs_T = bcs_T)
        # Buoyancy must actually drive a flow...
        @test maximum(norm.(ra.state.U.internal)) > 0
        # ...and the velocity + temperature fields must be density-invariant
        for c in 1:nc
            @test ra.state.U.internal[c] ≈ rb.state.U.internal[c] rtol = 1.0e-10
            @test tsa.T_field.internal[c] ≈ tsb.T_field.internal[c] rtol = 1.0e-10
        end
    end
end
