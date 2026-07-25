# test/incompressible_integrator.jl — Stage 5f-3
#
# Covers the CommonSolve integrator (`init`/`step!`/`solve!`) and the
# SymbolicIndexingInterface traits for the collocated incompressible family.

using FiniteVolumeMethod
using FiniteVolumeMethod: CollocatedSymbolicIndex
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC
using Test
using LinearSolve
using StaticArrays

# CommonSolve and SymbolicIndexingInterface are dependencies of the package but
# not of the test environment: `init`/`step!`/`solve!` come from the package's
# own export surface, and SII is reached through it.
const SII = FiniteVolumeMethod.SII

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

const LS = LUFactorization()

function _channel_bcs()
    return Dict{Symbol, AbstractBoundaryCondition}(
        :left => FixedVelocityBC((0.1, 0.0)),
        :right => FixedPressureBC(0.0),
        :bottom => NoSlipWallBC(),
        :top => NoSlipWallBC(),
    )
end

_steady_prob(; max_iterations = 6) = SteadyIncompressibleProblem(
    build_cartesian_unstructured_mesh(8, 4, 1.0, 1.0), _channel_bcs(),
    SIMPLE(; max_iterations = max_iterations, tolerance = 1.0e-12); nu = 0.1,
)

_transient_prob() = IncompressibleProblem(
    build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0), _channel_bcs(),
    PISO(; n_correctors = 2); nu = 0.1,
)

@testset "Incompressible integrator + SII (5f-3)" begin

    # ── 1. Steady: init / step! / solve! ──────────────────────────────
    @testset "SIMPLE integrator" begin
        integrator = init(_steady_prob(); linear_solver = LS)
        @test integrator isa IncompressibleIntegrator
        @test integrator.iter == 0
        @test integrator.u === integrator.state.u   # `u` aliases the flat state

        step!(integrator)
        @test integrator.iter == 1
        @test length(integrator.residuals[:continuity]) == 1
        # The first step must move the solution off the zero initial field.
        @test integrator.u != integrator.uprev
        @test all(isfinite, integrator.u)

        sol = solve!(integrator)
        @test sol isa IncompressibleSolution
        @test integrator.iter > 1
        @test sol.iterations == integrator.iter
    end

    # ── 2. Transient: one step per `step!` ────────────────────────────
    @testset "PISO integrator" begin
        integrator = init(
            _transient_prob(); tspan = (0.0, 0.05), dt = 0.01, linear_solver = LS,
        )
        @test integrator.t == 0.0
        step!(integrator)
        @test integrator.t ≈ 0.01
        @test integrator.iter == 1

        sol = solve!(integrator)
        @test integrator.t ≈ 0.05
        @test integrator.iter == 5
        @test sol.converged            # finite residuals
        @test all(isfinite, sol[:p])
    end

    # ── 3. Integrator and batch solve agree bitwise ───────────────────
    #
    # `step!` calls the same Stage-5e cores the batch solvers use, so the two
    # paths must not merely agree to a tolerance — they must be identical.
    @testset "integrator matches solve bitwise" begin
        batch = solve(_steady_prob(), SIMPLE(; max_iterations = 6, tolerance = 1.0e-12);
            linear_solver = LS)
        stepped = solve!(init(_steady_prob(); linear_solver = LS))
        @test stepped.iterations == batch.iterations
        @test stepped[:p] == batch[:p]
        @test stepped[:U] == batch[:U]

        batch_t = solve(_transient_prob(), PISO(; n_correctors = 2);
            tspan = (0.0, 0.05), dt = 0.01, linear_solver = LS)
        stepped_t = solve!(init(_transient_prob();
            tspan = (0.0, 0.05), dt = 0.01, linear_solver = LS))
        @test stepped_t[:p] == batch_t[:p]
        @test stepped_t[:U] == batch_t[:U]
    end

    # ── 4. Manual stepping reaches the same state as solve! ───────────
    @testset "manual stepping == solve!" begin
        manual = init(_transient_prob(); tspan = (0.0, 0.03), dt = 0.01, linear_solver = LS)
        for _ in 1:3
            step!(manual)
        end
        auto = solve!(init(_transient_prob();
            tspan = (0.0, 0.03), dt = 0.01, linear_solver = LS))
        # Same flat solution vector, reached by hand-stepping vs `solve!`.
        @test manual.u == auto.state.u
        @test manual.t ≈ 0.03
    end

    # ── 5. Physics-carrying problems are rejected, not silently ignored ─
    @testset "integrator rejects coupled physics" begin
        props = FluidThermalProperties{2}()
        model = IncompressibleModel(
            thermal = ThermalComponent(props; bcs = Dict{Symbol, Any}()),
        )
        prob = SteadyIncompressibleProblem(
            build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0), _channel_bcs(),
            SIMPLE(; max_iterations = 3); nu = 0.1, model = model,
        )
        @test_throws ArgumentError init(prob)
    end

    # ── 6. Displaying a problem must not throw ────────────────────────
    #
    # Both problem types root at a SciMLBase problem supertype whose generic
    # `show` reaches for `prob.u0` via `state_values`. These problems build
    # their initial state from the mesh instead of storing it, so without our
    # own `show` even typing `prob` at the REPL threw a FieldError. No test
    # displayed a problem, so only the docs build caught it.
    @testset "problems display without error" begin
        for prob in (_steady_prob(), _transient_prob())
            str = sprint(show, MIME"text/plain"(), prob)
            @test occursin("cells", str)
            @test occursin("nu = ", str)
            @test occursin("physics: ", str)
        end
        # A problem is also an SII value provider (generic ecosystem code asks).
        prob = _steady_prob()
        ncells = length(prob.mesh.cell_volumes)
        @test length(SII.state_values(prob)) == ncells * 3
        @test SII.parameter_values(prob) === prob
    end

    # ── 7. SymbolicIndexingInterface traits ───────────────────────────
    @testset "SII traits" begin
        sol = solve!(init(_steady_prob(); linear_solver = LS))
        ncells = length(sol.prob.mesh.cell_volumes)

        sys = SII.symbolic_container(sol)
        @test sys isa CollocatedSymbolicIndex
        @test SII.constant_structure(sys)
        @test !SII.is_time_dependent(sys)
        @test Set(SII.all_variable_symbols(sys)) == Set([:U, :p, :Ux, :Uy])
        @test SII.is_observed(sys, :p)
        @test !SII.is_observed(sys, :nonexistent)

        # State/parameter providers.
        @test SII.state_values(sol) === sol.state.u
        @test SII.parameter_values(sol) === sol.prob
        @test length(SII.state_values(sol)) == ncells * 3   # U (2) + p (1)

        # The extractor reads the correct block of the flat vector.
        u = SII.state_values(sol)
        @test collect(SII.observed(sys, :p)(u, sol.prob)) == sol[:p]
        @test collect(SII.observed(sys, :Ux)(u, sol.prob)) == sol[:Ux]

        # `sol[...]` routes through SII but materialises independent copies, so
        # it never aliases live solver state.
        p1 = sol[:p]
        p1[1] += 1.0
        @test sol[:p][1] != p1[1]
        @test sol[:U] isa Vector{<:SVector}
        @test sol[:p] isa Vector{Float64}
        @test keys(sol) == (:U, :p, :phi, :Ux, :Uy)
        @test_throws ErrorException sol[:nonexistent]

        # `:phi` is derived face state, not part of `u`, and stays reachable.
        @test length(sol[:phi]) == size(sol.prob.mesh.face_cells, 2)
    end
end
