using FiniteVolumeMethod
using FiniteVolumeMethod: PorousZone
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC
using Test
using LinearAlgebra
using LinearSolve
using StaticArrays
using SciMLBase
import SciMLStructures as SS

# ── Mesh builder (shared helper) ─────────────────────────────────────
include("TestHelpers.jl")

# ── Helper: build a standard channel problem ─────────────────────────

function build_channel_problem(; max_iterations = 5, tolerance = 1.0e-10)
    mesh = build_cartesian_unstructured_mesh(8, 4, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => FixedVelocityBC((0.1, 0.0)),
        :right => FixedPressureBC(0.0),
        :bottom => NoSlipWallBC(),
        :top => NoSlipWallBC(),
    )
    algo = SIMPLE(; max_iterations = max_iterations, tolerance = tolerance)
    return SteadyIncompressibleProblem(mesh, bcs, algo; nu = 0.1)
end

# ── Tests ─────────────────────────────────────────────────────────────

@testset "Incompressible SciML Compliance" begin

    # ── 1. CommonSolve.solve dispatch (B2) ────────────────────────────
    @testset "CommonSolve.solve dispatch (B2)" begin
        prob = build_channel_problem()

        sol = solve(prob, SIMPLE(; max_iterations = 5, tolerance = 1.0e-10))
        @test sol isa IncompressibleSolution
        @test sol.iterations > 0
        @test sol.retcode in (SciMLBase.ReturnCode.Success, SciMLBase.ReturnCode.MaxIters)

        sol2 = solve(prob)
        @test sol2 isa IncompressibleSolution
        @test sol2.iterations > 0
        @test sol2.retcode in (SciMLBase.ReturnCode.Success, SciMLBase.ReturnCode.MaxIters)
    end

    # ── 2. Symbolic indexing (B3) ─────────────────────────────────────
    @testset "Symbolic indexing (B3)" begin
        prob = build_channel_problem()
        sol = solve(prob, SIMPLE(; max_iterations = 5, tolerance = 1.0e-10))
        ncells = length(prob.mesh.cell_volumes)
        nfaces = size(prob.mesh.face_cells, 2)

        U = sol[:U]
        @test U isa Vector{<:SVector}
        @test length(U) == ncells

        p = sol[:p]
        @test p isa Vector{Float64}
        @test length(p) == ncells

        ux = sol[:Ux]
        @test ux isa Vector{Float64}
        @test length(ux) == ncells

        uy = sol[:Uy]
        @test uy isa Vector{Float64}
        @test length(uy) == ncells

        phi = sol[:phi]
        @test phi isa AbstractVector
        @test length(phi) == nfaces

        @test keys(sol) == (:U, :p, :phi, :Ux, :Uy)

        @test_throws ErrorException sol[:nonexistent]
    end

    # ── 3. Solution properties ────────────────────────────────────────
    @testset "Solution properties" begin
        prob = build_channel_problem()
        sol = solve(prob, SIMPLE(; max_iterations = 5, tolerance = 1.0e-10))

        @test sol.converged isa Bool
        @test sol.iterations isa Int
        @test sol.iterations > 0
        @test sol.residuals isa Dict
        @test haskey(sol.residuals, :Ux)
        @test haskey(sol.residuals, :Uy)
        @test haskey(sol.residuals, :continuity)
        @test sol.retcode isa SciMLBase.ReturnCode.T
    end

    # ── 4. remake (B1) ────────────────────────────────────────────────
    @testset "remake (B1)" begin
        prob = build_channel_problem()

        prob2 = remake(prob; nu = 0.05)
        @test prob2.nu == 0.05
        @test prob.nu == 0.1  # original unchanged

        prob3 = remake(prob; density = 2.0)
        @test prob3.density == 2.0
        @test prob.density == 1.0  # original unchanged

        new_algo = SIMPLE(; max_iterations = 10)
        prob4 = remake(prob; algorithm = new_algo)
        @test prob4.algorithm.max_iterations == 10
        @test prob.algorithm.max_iterations == 5  # original unchanged
    end

    # ── 5. SciMLStructures (B4) ───────────────────────────────────────
    @testset "SciMLStructures (B4)" begin
        prob = build_channel_problem()

        @test SS.isscimlstructure(prob)
        @test SS.hasportion(SS.Tunable(), prob)

        vals, repack, aliased = SS.canonicalize(SS.Tunable(), prob)
        @test !aliased

        @test vals[1] ≈ prob.nu
        @test vals[2] ≈ prob.density

        new_prob = repack([0.05, 2.0, vals[3:end]...])
        @test new_prob.nu ≈ 0.05
        @test new_prob.density ≈ 2.0

        replaced = SS.replace(SS.Tunable(), prob, [0.05, 2.0, vals[3:end]...])
        @test replaced.nu ≈ 0.05
        @test replaced.density ≈ 2.0
    end

    # ── 6. New BC types (C1) ──────────────────────────────────────────
    @testset "New BC types (C1)" begin
        # ZeroGradientBC
        zg = ZeroGradientBC()
        @test FiniteVolumeMethod.expand_velocity_bc(zg, 1) isa NeumannBC
        @test FiniteVolumeMethod.expand_velocity_bc(zg, 2) isa NeumannBC
        @test FiniteVolumeMethod.expand_pressure_bc(zg) isa NeumannBC

        # TotalPressureBC
        tp = TotalPressureBC(101325.0)
        @test FiniteVolumeMethod.expand_pressure_bc(tp) isa DirichletBC
        @test FiniteVolumeMethod.expand_pressure_bc(tp).value ≈ 101325.0
        @test FiniteVolumeMethod.expand_velocity_bc(tp, 1) isa NeumannBC

        # SymmetryBC
        sym = SymmetryBC()
        @test FiniteVolumeMethod.expand_velocity_bc(sym, 1) isa NeumannBC
        @test FiniteVolumeMethod.expand_velocity_bc(sym, 2) isa NeumannBC
        @test FiniteVolumeMethod.expand_pressure_bc(sym) isa NeumannBC

        # FlowRateInletBC
        fr = FlowRateInletBC((0.5, 0.0))
        vel_bc = FiniteVolumeMethod.expand_velocity_bc(fr, 1)
        @test vel_bc isa DirichletBC
        @test vel_bc.value ≈ 0.5
        vel_bc_y = FiniteVolumeMethod.expand_velocity_bc(fr, 2)
        @test vel_bc_y.value ≈ 0.0
        @test FiniteVolumeMethod.expand_pressure_bc(fr) isa NeumannBC

        # TimeDependentVelocityBC
        td = TimeDependentVelocityBC{2, Float64}(t -> SVector(t, 0.0))
        vel_td = FiniteVolumeMethod.expand_velocity_bc(td, 1)
        @test vel_td isa DirichletBC
        @test vel_td.value ≈ 0.0  # t_ref = 0
        vel_td_y = FiniteVolumeMethod.expand_velocity_bc(td, 2)
        @test vel_td_y.value ≈ 0.0
        @test FiniteVolumeMethod.expand_pressure_bc(td) isa NeumannBC
    end

    # ── 8. IncompressibleModel physics composition (Stage 5d) ─────────
    #
    # The physics dispatch in sciml_interface.jl had no coverage before
    # 5d: every consumer called the solve_simple_* internals directly, so
    # none of the branches or dependency checks were ever exercised.
    @testset "IncompressibleModel composition (5d)" begin
        @testset "default model is plain flow" begin
            prob = build_channel_problem()
            @test prob.model isa IncompressibleModel
            @test is_plain_flow(prob)
            @test !has_thermal(prob)
            @test !has_turbulence(prob)
            @test !has_radiation(prob)
            @test !has_combustion(prob)
            @test !has_porous_zones(prob)
            @test !has_mrf_zones(prob)
        end

        @testset "components require their boundary conditions" begin
            props = FluidThermalProperties{2}()
            @test_throws UndefKeywordError ThermalComponent(props)
            @test_throws UndefKeywordError RadiationComponent(P1Model())

            thermal = ThermalComponent(props; bcs = Dict{Symbol, Any}())
            @test thermal.T_init == props.T_ref
            @test ThermalComponent(
                props; bcs = Dict{Symbol, Any}(), T_init = 350.0,
            ).T_init == 350.0
        end

        @testset "model validates component dependencies" begin
            props = FluidThermalProperties{2}()
            thermal = ThermalComponent(props; bcs = Dict{Symbol, Any}())
            rad = RadiationComponent(P1Model(); bcs = Dict{Symbol, Any}())
            comb = CombustionComponent(1, 2; bcs = Dict{Symbol, Any}())

            # Radiation and combustion both enter the energy equation.
            @test_throws ArgumentError IncompressibleModel(radiation = rad)
            @test_throws ArgumentError IncompressibleModel(combustion = comb)

            # No reacting-flow solver accepts a radiation model, so this
            # combination used to drop radiation silently.
            @test_throws ArgumentError IncompressibleModel(
                thermal = thermal, radiation = rad, combustion = comb,
            )

            # Zones are threaded through the plain/turbulent/thermal paths only.
            @test_throws ArgumentError IncompressibleModel(
                thermal = thermal, radiation = rad,
                porous_zones = PorousZone{Float64}[],
            )

            model = IncompressibleModel(thermal = thermal, radiation = rad)
            @test has_thermal(model)
            @test has_radiation(model)
            @test !is_plain_flow(model)
        end

        @testset "transient solve rejects steady-only physics" begin
            props = FluidThermalProperties{2}()
            model = IncompressibleModel(
                thermal = ThermalComponent(props; bcs = Dict{Symbol, Any}()),
                radiation = RadiationComponent(P1Model(); bcs = Dict{Symbol, Any}()),
            )
            mesh = build_cartesian_unstructured_mesh(8, 4, 1.0, 1.0)
            bcs = Dict{Symbol, AbstractBoundaryCondition}(
                :left => FixedVelocityBC((0.1, 0.0)),
                :right => FixedPressureBC(0.0),
                :bottom => NoSlipWallBC(),
                :top => NoSlipWallBC(),
            )
            prob = IncompressibleProblem(mesh, bcs, PISO(); nu = 0.1, model = model)
            @test_throws ArgumentError solve(
                prob, PISO(); tspan = (0.0, 0.01), dt = 0.01,
            )
        end

        @testset "model survives remake and the tunable round-trip" begin
            zone = PorousZone(collect(1:4); K = 1.0e-4, F = 0.0)
            prob = build_channel_problem()
            prob = remake(prob; model = IncompressibleModel(porous_zones = [zone]))
            @test has_porous_zones(prob)

            prob_nu = remake(prob; nu = 0.2)
            @test prob_nu.nu == 0.2
            @test has_porous_zones(prob_nu)

            vals, repack, _ = SS.canonicalize(SS.Tunable(), prob)
            @test has_porous_zones(repack(vals))
        end

        @testset "zones in the model reach the internal entry point" begin
            zone = PorousZone(collect(1:4); K = 1.0e-4, F = 0.0)
            prob = build_channel_problem(; max_iterations = 2)
            prob = remake(prob; model = IncompressibleModel(porous_zones = [zone]))
            result = FiniteVolumeMethod.solve_simple(
                prob; linear_solver = LUFactorization(),
            )
            @test all(isfinite, result.state.p.internal)
        end

        @testset "solve-time model override is recorded in the solution" begin
            zone = PorousZone(collect(1:4); K = 1.0e-4, F = 0.0)
            prob = build_channel_problem()
            sol = solve(
                prob, SIMPLE(; max_iterations = 2);
                model = IncompressibleModel(porous_zones = [zone]),
                linear_solver = LUFactorization(),
            )
            @test has_porous_zones(sol.prob)
            @test !has_porous_zones(prob)
        end
    end
end
