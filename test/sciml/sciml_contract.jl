using DelaunayTriangulation
using FiniteVolumeMethod
using FiniteVolumeMethod: AMRODESolutionAccessor, CTData3D, FVMSolutionAccessor, HyperbolicSolutionAccessor, MHD3DSolutionAccessor, MHDCTCache3D, max_divB_3d, solution_state_layout, solution_variables
using OrdinaryDiffEq
using OrdinaryDiffEqSSPRK: SSPRK33
using SciMLBase: DiscreteCallback, ODEProblem, ReturnCode, SteadyStateProblem, remake
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

@testset "Semidiscrete callback merging stays SciML-compatible" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 32)

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        x -> x < 0.5 ? SVector(1.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.1);
        final_time = 0.02, cfl = 0.4
    )

    ode_prob = sciml_problem(prob)
    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)

    callback_hits = Ref(0)
    cb = DiscreteCallback(
        (u, t, integrator) -> true,
        integrator -> (callback_hits[] += 1),
        save_positions = (false, false),
    )

    sol = solve(prob, SSPRK33(); adaptive = false, dt = dt0, callback = cb)
    @test sol.retcode == ReturnCode.Success
    @test callback_hits[] > 0
    @test sol.t[end] ≈ prob.final_time atol = 1.0e-12
end

@testset "Split semidiscrete callback merging stays SciML-compatible" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 24)

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        x -> x < 0.5 ? SVector(1.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.1);
        final_time = 0.02, cfl = 0.35
    )

    split_prob = sciml_problem(prob, NullSource())
    dt0 = compute_initial_dt(split_prob.p, split_prob.u0)

    callback_hits = Ref(0)
    cb = DiscreteCallback(
        (u, t, integrator) -> true,
        integrator -> (callback_hits[] += 1),
        save_positions = (false, false),
    )

    sol = solve(prob, NullSource(), SSPRK33(); adaptive = false, dt = dt0, callback = cb)
    @test sol.retcode == ReturnCode.Success
    @test callback_hits[] > 0
end

@testset "3D MHD canonical SciML contract and remake" begin
    eos = IdealGasEOS(5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)
    nx, ny, nz = 4, 3, 2
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.75, 0.0, 0.5, nx, ny, nz)

    function uniform_mhd_ic(x, y, z)
        return SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.2, -0.1, 0.05)
    end

    prob = HyperbolicProblem3D(
        law, mesh, HLLDSolver(), NoReconstruction(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        uniform_mhd_ic; final_time = 0.01, cfl = 0.25
    )

    ode_prob = sciml_problem(prob)
    @test ode_prob isa ODEProblem
    @test ode_prob.p isa MHDCTCache3D

    expected_len = nx * ny * nz * 8 + (nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * (nz + 1)
    @test length(ode_prob.u0) == expected_len

    remade = remake(ode_prob; final_time = 0.02)
    @test remade.tspan == (0.0, 0.02)
    @test remade.p isa MHDCTCache3D

    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    limiter = mhd_stage_limiter(ode_prob.p)
    sol = solve(prob, SSPRK33(; stage_limiter! = limiter); adaptive = false, dt = dt0)
    @test sol.retcode == ReturnCode.Success

    accessor = solution_accessor(prob)
    @test accessor isa MHD3DSolutionAccessor
    @test solution_state_layout(accessor) == :cell_centered_conserved_with_ct
    @test solution_variables(accessor) == ["rho", "rho_vx", "rho_vy", "rho_vz", "E", "Bx", "By", "Bz"]

    snapshot = solution_snapshot(prob, sol, length(sol.t))
    @test size(snapshot.conserved) == (nx, ny, nz)
    @test size(snapshot.primitive) == (nx, ny, nz)
    @test size(snapshot.coordinates) == (nx, ny, nz)
    @test snapshot.ct_state isa CTData3D
    @test max_divB_3d(snapshot.ct_state, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz) < 1.0e-12
end
