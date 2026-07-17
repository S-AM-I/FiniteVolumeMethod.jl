using FiniteVolumeMethod
using FiniteVolumeMethod: KOmegaSSTModel, RANSTurbulenceState, SolveResult, compute_nu_eff, compute_strain_rate, compute_wall_distance, n_turbulence_fields, solve_turbulence!, turbulence_field_names, turbulence_inlet_bc, turbulence_wall_bc, turbulent_viscosity!
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC
using Test
using LinearAlgebra
using LinearSolve
using StaticArrays
using SparseArrays

# ── Mesh builder (shared helper) ─────────────────────────────────────
include("TestHelpers.jl")

# ── Tests ──────────────────────────────────────────────────────────────

@testset "RANS Turbulence Models" begin

    # ── 1. Type construction ──────────────────────────────────────────
    @testset "Type construction" begin
        # StandardKEpsilon (existing type)
        ke = StandardKEpsilon()
        @test ke.C_mu == 0.09
        @test ke.sigma_k == 1.0
        @test ke.sigma_epsilon == 1.3
        @test n_turbulence_fields(ke) == 2
        @test turbulence_field_names(ke) == (:k, :epsilon)

        # KOmega
        kw = KOmega()
        @test kw isa KOmega{Float64}
        @test kw.beta_star == 0.09
        @test kw.alpha == 5.0 / 9.0
        @test kw.beta == 3.0 / 40.0
        @test kw.sigma_k == 0.5
        @test kw.sigma_omega == 0.5
        @test n_turbulence_fields(kw) == 2
        @test turbulence_field_names(kw) == (:k, :omega)

        # KOmegaSSTModel
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        sst = KOmegaSSTModel(mesh, [:bottom, :top])
        @test sst isa KOmegaSSTModel{Float64}
        @test length(sst.d_wall) == 16
        @test all(isfinite, sst.d_wall)
        @test all(>(0), sst.d_wall)
        @test n_turbulence_fields(sst) == 2
        @test turbulence_field_names(sst) == (:k, :omega)

        # SpalartAllmaras
        sa = SpalartAllmaras(mesh, [:bottom, :top])
        @test sa isa SpalartAllmaras{Float64}
        @test sa.cb1 == 0.1355
        @test sa.cv1 == 7.1
        @test length(sa.d_wall) == 16
        @test n_turbulence_fields(sa) == 1
        @test turbulence_field_names(sa) == (:nu_tilde,)
    end

    # ── 2. RANSTurbulenceState construction ───────────────────────────
    @testset "RANSTurbulenceState construction" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        # k-ε state
        ke = StandardKEpsilon()
        ts_ke = RANSTurbulenceState(ke, mesh)
        @test haskey(ts_ke.fields, :k)
        @test haskey(ts_ke.fields, :epsilon)
        @test length(ts_ke.fields[:k].internal) == nc
        @test length(ts_ke.nu_t) == nc
        @test all(==(1.0e-6), ts_ke.fields[:k].internal)

        # k-ω state with custom initial values
        kw = KOmega()
        ts_kw = RANSTurbulenceState(kw, mesh; k = 0.01, omega = 100.0)
        @test haskey(ts_kw.fields, :k)
        @test haskey(ts_kw.fields, :omega)
        @test all(==(0.01), ts_kw.fields[:k].internal)
        @test all(==(100.0), ts_kw.fields[:omega].internal)

        # SA state
        sa = SpalartAllmaras(mesh, [:bottom])
        ts_sa = RANSTurbulenceState(sa, mesh; nu_tilde = 1.0e-4)
        @test haskey(ts_sa.fields, :nu_tilde)
        @test length(ts_sa.fields) == 1
        @test all(==(1.0e-4), ts_sa.fields[:nu_tilde].internal)
    end

    # ── 3. compute_nu_eff ─────────────────────────────────────────────
    @testset "compute_nu_eff" begin
        nu = 1.0e-3
        nu_t = [0.01, 0.02, 0.05, 0.1]
        nu_eff = compute_nu_eff(nu, nu_t)
        @test length(nu_eff) == 4
        for c in 1:4
            @test nu_eff[c] ≈ nu + nu_t[c]
        end
    end

    # ── 4. compute_strain_rate ────────────────────────────────────────
    @testset "compute_strain_rate" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        # Uniform velocity field U = (1, 0) → zero strain rate
        U_uniform = CollocatedVectorField(:U, mesh; value = SVector(1.0, 0.0))
        S_uniform = compute_strain_rate(U_uniform, mesh)
        @test length(S_uniform) == nc
        @test all(isfinite, S_uniform)
        @test all(s -> s < 0.1, S_uniform)

        # Linear velocity field: Ux = y → dudy = 1, nonzero strain rate
        U_linear = CollocatedVectorField(:U, mesh)
        for c in 1:nc
            y_c = mesh.cell_centers[2, c]
            U_linear.internal[c] = SVector(y_c, 0.0)
        end
        # Set boundary values too (needed for gradient computation)
        for (i, f) in enumerate(U_linear.boundary_face_indices)
            y_f = mesh.face_centers[2, f]
            U_linear.boundary[i] = SVector(y_f, 0.0)
        end
        S_linear = compute_strain_rate(U_linear, mesh)
        @test all(isfinite, S_linear)
        @test all(>(0), S_linear)
    end

    # ── 5. compute_wall_distance ──────────────────────────────────────
    @testset "compute_wall_distance" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        d_wall = compute_wall_distance(mesh, [:bottom])

        # Bottom row cells (j=1): centers at y = 0.125
        # Top row cells (j=4): centers at y = 0.875
        # Distance to bottom face (y=0) should be smaller for bottom row
        nx = 4
        for i in 1:nx
            bottom_cell = i           # j=1
            top_cell = (4 - 1) * nx + i  # j=4
            @test d_wall[bottom_cell] < d_wall[top_cell]
        end
        @test all(isfinite, d_wall)
        @test all(>(0), d_wall)
    end

    # ── 6. k-ε turbulent_viscosity! ──────────────────────────────────
    @testset "k-ε turbulent_viscosity!" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        ke = StandardKEpsilon()

        ts = RANSTurbulenceState(ke, mesh; k = 0.1, epsilon = 0.01)
        nu_t = zeros(Float64, nc)
        turbulent_viscosity!(nu_t, ke, ts, mesh)

        expected = ke.C_mu * 0.1^2 / 0.01
        for c in 1:nc
            @test nu_t[c] ≈ expected
        end
    end

    # ── 7. k-ε solve_turbulence! smoke ────────────────────────────────
    @testset "k-ε solve_turbulence! smoke" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        ke = StandardKEpsilon()

        ts = RANSTurbulenceState(ke, mesh; k = 1.0e-4, epsilon = 1.0e-5)
        turbulent_viscosity!(ts.nu_t, ke, ts, mesh)

        U = CollocatedVectorField(:U, mesh; value = SVector(0.1, 0.0))
        # Set boundary values for gradient computation
        for (i, _) in enumerate(U.boundary_face_indices)
            U.boundary[i] = SVector(0.1, 0.0)
        end
        phi = FaceFluxField(:phi, mesh; value = 0.0)

        bcs_turb = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(
            :k => Dict{Symbol, AbstractBoundaryCondition}(
                :left => DirichletBC(1.0e-4),
                :right => NeumannBC(0.0),
                :bottom => NeumannBC(0.0),
                :top => NeumannBC(0.0),
            ),
            :epsilon => Dict{Symbol, AbstractBoundaryCondition}(
                :left => DirichletBC(1.0e-5),
                :right => NeumannBC(0.0),
                :bottom => NeumannBC(0.0),
                :top => NeumannBC(0.0),
            ),
        )

        solve_turbulence!(ts, ke, U, phi, 1.0e-3, mesh, bcs_turb)

        @test all(isfinite, ts.fields[:k].internal)
        @test all(isfinite, ts.fields[:epsilon].internal)
        @test all(>(0), ts.fields[:k].internal)
        @test all(>(0), ts.fields[:epsilon].internal)
    end

    # ── 8. solve_simple_turbulent smoke ───────────────────────────────
    @testset "solve_simple_turbulent smoke" begin
        mesh = build_cartesian_unstructured_mesh(8, 4, 2.0, 1.0)
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.1, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        algo = SIMPLE(; max_iterations = 5, tolerance = 1.0e-12)
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.01)

        ke = StandardKEpsilon()
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(
            :k => Dict{Symbol, AbstractBoundaryCondition}(
                :left => DirichletBC(1.0e-4),
                :right => NeumannBC(0.0),
                :bottom => NeumannBC(0.0),
                :top => NeumannBC(0.0),
            ),
            :epsilon => Dict{Symbol, AbstractBoundaryCondition}(
                :left => DirichletBC(1.0e-5),
                :right => NeumannBC(0.0),
                :bottom => NeumannBC(0.0),
                :top => NeumannBC(0.0),
            ),
        )
        result, turb_state = solve_simple_turbulent(
            prob, ke; turb_bcs = turb_bcs,
        )

        @test result isa SolveResult{2, Float64}
        @test turb_state isa RANSTurbulenceState{Float64}
        @test result.iterations == 5
        @test all(isfinite, turb_state.fields[:k].internal)
        @test all(isfinite, turb_state.fields[:epsilon].internal)
        @test all(isfinite, turb_state.nu_t)
    end

    # ── 9. KOmega turbulent_viscosity! ────────────────────────────────
    @testset "KOmega turbulent_viscosity!" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        kw = KOmega()

        ts = RANSTurbulenceState(kw, mesh; k = 0.05, omega = 50.0)
        nu_t = zeros(Float64, nc)
        turbulent_viscosity!(nu_t, kw, ts, mesh)

        expected = 0.05 / 50.0
        for c in 1:nc
            @test nu_t[c] ≈ expected
        end
    end

    # ── 10. turbulence_inlet_bc ───────────────────────────────────────
    @testset "turbulence_inlet_bc" begin
        U_mag = 1.0
        intensity = 0.05
        length_scale = 0.01

        # k-ε
        ke = StandardKEpsilon()
        bc_ke = turbulence_inlet_bc(ke, U_mag, intensity, length_scale)
        @test haskey(bc_ke, :k)
        @test haskey(bc_ke, :epsilon)
        @test bc_ke[:k] isa DirichletBC
        @test bc_ke[:epsilon] isa DirichletBC
        @test bc_ke[:k].value > 0
        @test bc_ke[:epsilon].value > 0

        # k-ω
        kw = KOmega()
        bc_kw = turbulence_inlet_bc(kw, U_mag, intensity, length_scale)
        @test haskey(bc_kw, :k)
        @test haskey(bc_kw, :omega)
        @test bc_kw[:k] isa DirichletBC
        @test bc_kw[:omega] isa DirichletBC
        @test bc_kw[:k].value > 0
        @test bc_kw[:omega].value > 0

        # SST
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        sst = KOmegaSSTModel(mesh, [:bottom])
        bc_sst = turbulence_inlet_bc(sst, U_mag, intensity, length_scale)
        @test haskey(bc_sst, :k)
        @test haskey(bc_sst, :omega)
        @test bc_sst[:k] isa DirichletBC
        @test bc_sst[:omega] isa DirichletBC

        # SA
        sa = SpalartAllmaras(mesh, [:bottom])
        bc_sa = turbulence_inlet_bc(sa, U_mag, intensity, length_scale)
        @test haskey(bc_sa, :nu_tilde)
        @test bc_sa[:nu_tilde] isa DirichletBC
        @test bc_sa[:nu_tilde].value > 0
    end

    # ── 11. turbulence_wall_bc ────────────────────────────────────────
    @testset "turbulence_wall_bc" begin
        # k-ε
        ke = StandardKEpsilon()
        wbc_ke = turbulence_wall_bc(ke)
        @test haskey(wbc_ke, :k)
        @test haskey(wbc_ke, :epsilon)
        @test wbc_ke[:k] isa NeumannBC
        @test wbc_ke[:epsilon] isa NeumannBC

        # k-ω
        kw = KOmega()
        wbc_kw = turbulence_wall_bc(kw)
        @test haskey(wbc_kw, :k)
        @test haskey(wbc_kw, :omega)
        @test wbc_kw[:k] isa NeumannBC
        @test wbc_kw[:omega] isa NeumannBC

        # SST (dispatches via Union{KOmega, KOmegaSSTModel})
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        sst = KOmegaSSTModel(mesh, [:bottom])
        wbc_sst = turbulence_wall_bc(sst)
        @test haskey(wbc_sst, :k)
        @test haskey(wbc_sst, :omega)

        # SA
        sa = SpalartAllmaras(mesh, [:bottom])
        wbc_sa = turbulence_wall_bc(sa)
        @test haskey(wbc_sa, :nu_tilde)
        @test wbc_sa[:nu_tilde] isa DirichletBC
        @test wbc_sa[:nu_tilde].value == 0.0
    end

    # Shared helper for the new regression testsets below
    function _full_ke_bcs()
        inner(v_k, v_e) = Dict{Symbol, AbstractBoundaryCondition}(
            :left => DirichletBC(v_k),
            :right => NeumannBC(0.0),
            :bottom => NeumannBC(0.0),
            :top => NeumannBC(0.0),
        )
        return Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(
            :k => inner(1.0e-4, 0.0),
            :epsilon => inner(1.0e-5, 0.0),
        )
    end

    # ── Durbin realizability cap must survive to momentum's nu_t ──────
    @testset "Durbin cap survives nu_t recompute" begin
        mesh = build_cartesian_unstructured_mesh(8, 4, 2.0, 1.0)
        nc = length(mesh.cell_volumes)
        ke = StandardKEpsilon()

        # High-shear velocity field with large k and tiny epsilon so the
        # uncapped nu_t = C_mu k^2/eps grossly violates the Durbin bound.
        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((1.0, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => NoSlipWallBC(),
            :top => NoSlipWallBC(),
        )
        prob = IncompressibleProblem(mesh, bcs, SIMPLE(); nu = 1.0e-3)
        state = IncompressibleState(mesh)
        for c in 1:nc
            y = mesh.cell_centers[2, c]
            state.U.internal[c] = SVector(4.0 * y, 0.0)  # strong shear
        end
        FiniteVolumeMethod.update_boundary_velocity!(state, bcs, mesh)

        turb_state = RANSTurbulenceState(ke, mesh; k = 0.5, epsilon = 1.0e-4)
        FiniteVolumeMethod._update_turbulence!(
            turb_state, ke, state, prob, mesh, _full_ke_bcs(),
        )

        # After the full update (transport solve + nu_t recompute + cap
        # re-application), the momentum-visible nu_t must satisfy the
        # Durbin bound wherever |S| is significant.
        S_mag = FiniteVolumeMethod.compute_strain_rate(state.U, mesh)
        C_T = 0.6
        k_int = turb_state.fields[:k].internal
        n_checked = 0
        for c in 1:nc
            if S_mag[c] > 1.0e-8
                cap = C_T * max(k_int[c], 1.0e-10) / S_mag[c]
                @test turb_state.nu_t[c] <= cap * (1.0 + 1.0e-12)
                n_checked += 1
            end
        end
        @test n_checked > 0
    end

    # ── Missing turbulence BCs must fail fast with a clear message ────
    @testset "Turbulence BC validation up front" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        ke = StandardKEpsilon()
        state = IncompressibleState(mesh)
        turb_state = RANSTurbulenceState(ke, mesh; k = 1.0e-3, epsilon = 1.0e-4)
        phi = FiniteVolumeMethod.FaceFluxField(:phi, mesh; value = 0.0)

        # Entirely missing bcs dict
        err = try
            solve_turbulence!(
                turb_state, ke, state.U, phi, 1.0e-3, mesh,
                Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
            )
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("epsilon", sprint(showerror, err))
        @test occursin(":k", sprint(showerror, err))

        # Partially missing patches: message must name the patches
        partial = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(
            :k => Dict{Symbol, AbstractBoundaryCondition}(
                :left => NeumannBC(0.0),
            ),
            :epsilon => Dict{Symbol, AbstractBoundaryCondition}(
                :left => NeumannBC(0.0),
                :right => NeumannBC(0.0),
                :bottom => NeumannBC(0.0),
                :top => NeumannBC(0.0),
            ),
        )
        err2 = try
            solve_turbulence!(
                turb_state, ke, state.U, phi, 1.0e-3, mesh, partial,
            )
            nothing
        catch e
            e
        end
        @test err2 isa ErrorException
        msg = sprint(showerror, err2)
        @test occursin("right", msg) && occursin("top", msg) && occursin("bottom", msg)
    end

    # ── WallFunctionBC produces wall drag in the momentum equation ────
    @testset "WallFunctionBC adds wall shear (not stress-free)" begin
        mesh = build_cartesian_unstructured_mesh(8, 4, 2.0, 1.0)
        nc = length(mesh.cell_volumes)

        # Channel with wall-function walls: velocity expansion must be
        # no-slip Dirichlet so the boundary diffusion term produces drag.
        @test FiniteVolumeMethod.expand_velocity_bc(WallFunctionBC(), 1) isa
            DirichletBC
        @test FiniteVolumeMethod.expand_velocity_bc(WallFunctionBC(), 1).value == 0.0

        bcs_wf = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.5, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => WallFunctionBC(),
            :top => WallFunctionBC(),
        )
        bcs_free = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.5, 0.0)),
            :right => FixedPressureBC(0.0),
            :bottom => ZeroGradientBC(),  # stress-free comparison
            :top => ZeroGradientBC(),
        )
        prob_wf = IncompressibleProblem(mesh, bcs_wf, SIMPLE(); nu = 1.0e-3)
        prob_free = IncompressibleProblem(mesh, bcs_free, SIMPLE(); nu = 1.0e-3)
        state = IncompressibleState(mesh)
        for c in 1:nc
            state.U.internal[c] = SVector(0.5, 0.0)
        end

        eq_wf = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_momentum!(eq_wf, state, prob_wf, 1)
        eq_free = FiniteVolumeMethod.CollocatedEquation(mesh)
        FiniteVolumeMethod.assemble_momentum!(eq_free, state, prob_free, 1)

        # Wall-adjacent cells must carry a strictly larger diagonal with
        # the wall function than with the stress-free wall.
        n_larger = 0
        for c in 1:nc
            y = mesh.cell_centers[2, c]
            if y < 0.3 || y > 0.7  # wall-adjacent rows
                if eq_wf.A[c, c] > eq_free.A[c, c] + 1.0e-12
                    n_larger += 1
                end
            end
        end
        @test n_larger == 16  # all wall-adjacent cells see the drag term

        # Boundary velocity update zeroes the wall value (wall does not
        # inherit the cell velocity → zero wall flux)
        FiniteVolumeMethod.update_boundary_velocity!(state, bcs_wf, mesh)
        for (i, f) in enumerate(state.U.boundary_face_indices)
            tag = FiniteVolumeMethod._face_tag(mesh, f)
            if tag === :bottom || tag === :top
                @test state.U.boundary[i] == zero(SVector{2, Float64})
            end
        end

        # End-to-end: turbulent channel with wall functions develops a
        # velocity DEFICIT at the walls relative to the centerline
        ke = StandardKEpsilon()
        result, turb_state = solve_simple_turbulent(
            IncompressibleProblem(
                mesh, bcs_wf, SIMPLE(; max_iterations = 20); nu = 1.0e-3,
            ),
            ke; turb_bcs = _full_ke_bcs(),
        )
        u_wall_cells = Float64[]
        u_center_cells = Float64[]
        for c in 1:nc
            y = mesh.cell_centers[2, c]
            u = result.state.U.internal[c][1]
            if y < 0.3 || y > 0.7
                push!(u_wall_cells, u)
            else
                push!(u_center_cells, u)
            end
        end
        # Drag at the wall: near-wall velocity below centerline velocity
        @test maximum(u_wall_cells) < maximum(u_center_cells)
    end
end
