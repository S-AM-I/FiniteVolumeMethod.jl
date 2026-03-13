using DelaunayTriangulation
using FiniteVolumeMethod
using OrdinaryDiffEq
using SciMLBase: ODEProblem, ReturnCode, SteadyStateProblem
using StaticArrays
using Test

@testset "Canonical SciML problem for node-based FVM problems" begin
    tri = triangulate_rectangle(0.0, 1.0, 0.0, 1.0, 4, 4; single_boundary = true)
    mesh = FVMGeometry(tri)
    bc = (x, y, t, u, p) -> zero(u)
    BCs = BoundaryConditions(mesh, bc, Dirichlet)
    initial_condition = [sin(pi * x) * sin(pi * y) for (x, y) in DelaunayTriangulation.each_point(tri)]
    D = (x, y, t, u, p) -> 1.0

    prob = FVMProblem(mesh, BCs; diffusion_function = D, initial_condition, final_time = 0.01)

    ode_prob = sciml_problem(prob)
    @test ode_prob isa ODEProblem
    @test ode_prob.tspan == (0.0, 0.01)

    steady_prob = SteadyFVMProblem(prob)
    steady_sciml = sciml_problem(steady_prob)
    @test steady_sciml isa SteadyStateProblem

    integrator = init(prob, Tsit5(); adaptive = false, dt = 1.0e-3, saveat = [0.0, 0.01])
    @test integrator.sol.prob isa ODEProblem

    sol = solve(prob, Tsit5(); adaptive = false, dt = 1.0e-3, saveat = [0.0, 0.01])
    @test sol.retcode == ReturnCode.Success

    accessor = solution_accessor(prob)
    @test accessor isa FVMSolutionAccessor
    @test solution_state_layout(accessor) == :node_values
    @test solution_variables(accessor) == ["u"]
    @test length(solution_coordinates(accessor)) == length(initial_condition)

    snapshot = solution_snapshot(prob, sol, length(sol.t))
    @test snapshot.layout == :node_values
    @test snapshot.variables == ["u"]
    @test length(snapshot.coordinates) == length(initial_condition)
    @test length(snapshot.values) == length(initial_condition)
end

@testset "Direct semidiscrete solve/init for hyperbolic problems" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 40)

    function sod_ic(x)
        if x < 0.5
            return SVector(1.0, 0.0, 1.0)
        else
            return SVector(0.125, 0.0, 0.1)
        end
    end

    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), sod_ic;
        final_time = 0.05, cfl = 0.4
    )

    ode_prob = sciml_problem(prob)
    @test ode_prob isa ODEProblem

    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    integrator = init(prob, SSPRK33(); adaptive = false, dt = dt0)
    @test integrator.sol.prob isa ODEProblem

    sol_direct = solve(prob, SSPRK33(); adaptive = false, dt = dt0)
    sol_ref = solve(ode_prob, SSPRK33(); adaptive = false, dt = dt0)
    @test sol_direct.retcode == ReturnCode.Success
    @test sol_ref.retcode == ReturnCode.Success
    @test sol_direct.t[end] ≈ sol_ref.t[end] atol = 1.0e-12

    U_direct = reinterpret(SVector{3, Float64}, copy(sol_direct.u[end]))
    U_ref = reinterpret(SVector{3, Float64}, copy(sol_ref.u[end]))
    max_diff = maximum(maximum(abs.(U_direct[i] - U_ref[i])) for i in eachindex(U_direct))
    @test max_diff < 1.0e-10

    accessor = solution_accessor(prob)
    @test accessor isa HyperbolicSolutionAccessor
    @test solution_state_layout(accessor) == :cell_centered_conserved
    @test solution_variables(accessor) == ["rho", "rho_v", "E"]
    @test length(solution_coordinates(accessor)) == 40

    snapshot = solution_snapshot(prob, sol_direct, length(sol_direct.t))
    @test snapshot.layout == :cell_centered_conserved
    @test snapshot.variables == ["rho", "rho_v", "E"]
    @test length(snapshot.conserved) == 40
    @test length(snapshot.primitive) == 40
end

@testset "Canonical split SciML problem for semidiscrete hyperbolic problems" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 32)
    wL = SVector(1.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.1)

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        x -> x < 0.5 ? wL : wR;
        final_time = 0.05, cfl = 0.4
    )

    split_prob = sciml_problem(prob, NullSource())
    @test split_prob isa ODEProblem

    dt0 = compute_initial_dt(split_prob.p, split_prob.u0)
    sol_direct = solve(prob, NullSource(), SSPRK33(); adaptive = false, dt = dt0)
    sol_ref = solve(split_prob, SSPRK33(); adaptive = false, dt = dt0)
    @test sol_direct.retcode == ReturnCode.Success
    @test sol_ref.retcode == ReturnCode.Success
    @test sol_direct.t[end] ≈ sol_ref.t[end] atol = 1.0e-12
end

@testset "AMR direct solve and snapshot accessor" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    criterion = GradientRefinement(; refine_threshold = 0.5, coarsen_threshold = 0.05)
    block_size = (4, 4)
    grid = AMRGrid(law, criterion, block_size, 2, (0.0, 0.0), (1.0, 1.0), Val(4))

    gamma = 1.4
    rho = 1.0
    P = 1.0
    E = P / (gamma - 1)
    state = SVector(rho, 0.0, 0.0, E)
    root = grid.blocks[1]
    for j in 1:4, i in 1:4
        root.U[i, j] = state
    end

    prob = AMRProblem(
        grid,
        LaxFriedrichsSolver(),
        NoReconstruction(),
        (TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC());
        final_time = 0.01,
        cfl = 0.4,
    )

    ode_prob = sciml_problem(prob)
    @test ode_prob isa ODEProblem

    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    sol = solve(prob, SSPRK33(); adaptive = false, dt = dt0)
    @test sol.retcode == ReturnCode.Success

    accessor = solution_accessor(prob)
    @test accessor isa AMRODESolutionAccessor
    @test solution_state_layout(accessor) == :block_cell_centered_conserved
    @test solution_variables(accessor) == ["rho", "rho_vx", "rho_vy", "E"]

    snapshot = solution_snapshot(prob, sol, length(sol.t))
    @test snapshot.layout == :block_cell_centered_conserved
    @test Set(keys(snapshot.conserved)) == Set([1])
    @test size(snapshot.conserved[1]) == (4, 4)
    @test size(snapshot.coordinates[1]) == (4, 4)
end
