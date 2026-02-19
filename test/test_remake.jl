using FiniteVolumeMethod
using Test
using StaticArrays
using DelaunayTriangulation

# ============================================================
# FVMProblem
# ============================================================
@testset "remake FVMProblem" begin
    tri = triangulate_rectangle(0, 1, 0, 1, 5, 5; single_boundary = true)
    mesh = FVMGeometry(tri)
    bc = (x, y, t, u, p) -> zero(u)
    BCs = BoundaryConditions(mesh, bc, Neumann)
    D = (x, y, t, u, p) -> one(u)
    ic = [0.0 for _ in DelaunayTriangulation.each_point(tri)]
    prob = FVMProblem(mesh, BCs; diffusion_function = D, initial_condition = ic, final_time = 1.0)

    # Change final_time only
    prob2 = remake(prob; final_time = 2.0)
    @test prob2.final_time == 2.0
    @test prob2.initial_time === prob.initial_time
    @test prob2.mesh === prob.mesh
    @test prob2.conditions === prob.conditions
    @test prob2.flux_function === prob.flux_function
    @test prob2.initial_condition === prob.initial_condition

    # No-kwargs round-trip
    prob3 = remake(prob)
    @test prob3.final_time === prob.final_time
    @test prob3.mesh === prob.mesh
end

# ============================================================
# SteadyFVMProblem
# ============================================================
@testset "remake SteadyFVMProblem" begin
    tri = triangulate_rectangle(0, 1, 0, 1, 5, 5; single_boundary = true)
    mesh = FVMGeometry(tri)
    bc = (x, y, t, u, p) -> zero(u)
    BCs = BoundaryConditions(mesh, bc, Dirichlet)
    D = (x, y, t, u, p) -> one(u)
    ic = [0.0 for _ in DelaunayTriangulation.each_point(tri)]
    prob = FVMProblem(mesh, BCs; diffusion_function = D, initial_condition = ic, final_time = Inf)
    steady = SteadyFVMProblem(prob)

    # No-kwargs round-trip
    steady2 = remake(steady)
    @test steady2.problem.final_time === prob.final_time

    # Replace inner problem
    prob_new = remake(prob; final_time = 10.0)
    steady3 = remake(steady; problem = prob_new)
    @test steady3.problem.final_time == 10.0
end

# ============================================================
# FVMSystem
# ============================================================
@testset "remake FVMSystem" begin
    tri = triangulate_rectangle(0, 1, 0, 1, 5, 5; single_boundary = true)
    mesh = FVMGeometry(tri)
    bc_u = (x, y, t, (u, v), p) -> zero(u)
    bc_v = (x, y, t, (u, v), p) -> zero(v)
    BCs_u = BoundaryConditions(mesh, bc_u, Neumann)
    BCs_v = BoundaryConditions(mesh, bc_v, Neumann)
    q_u = (x, y, t, (αu, αv), (βu, βv), (γu, γv), p) -> begin
        (-(αu), -(βu))
    end
    q_v = (x, y, t, (αu, αv), (βu, βv), (γu, γv), p) -> begin
        (-(αv), -(βv))
    end
    npts = DelaunayTriangulation.num_points(tri)
    u_ic = zeros(npts)
    v_ic = zeros(npts)
    u_prob = FVMProblem(
        mesh, BCs_u;
        flux_function = q_u,
        initial_condition = u_ic, final_time = 1.0,
    )
    v_prob = FVMProblem(
        mesh, BCs_v;
        flux_function = q_v,
        initial_condition = v_ic, final_time = 1.0,
    )
    sys = FVMSystem(u_prob, v_prob)

    # Change final_time
    sys2 = remake(sys; final_time = 5.0)
    @test sys2.final_time == 5.0
    @test sys2.initial_time === sys.initial_time
    @test sys2.mesh === sys.mesh

    # No-kwargs round-trip
    sys3 = remake(sys)
    @test sys3.final_time === sys.final_time
end

# ============================================================
# HyperbolicProblem
# ============================================================
@testset "remake HyperbolicProblem" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 100)
    wL = SVector(1.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.1)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(),
        x -> x < 0.5 ? wL : wR;
        final_time = 0.2, cfl = 0.5,
    )

    # Change final_time and cfl
    prob2 = remake(prob; final_time = 0.4, cfl = 0.3)
    @test prob2.final_time == 0.4
    @test prob2.cfl == 0.3
    @test prob2.law === prob.law
    @test prob2.mesh === prob.mesh
    @test prob2.riemann_solver === prob.riemann_solver
    @test prob2.reconstruction === prob.reconstruction
    @test prob2.initial_condition === prob.initial_condition

    # Change reconstruction
    prob3 = remake(prob; reconstruction = NoReconstruction())
    @test prob3.reconstruction isa NoReconstruction
    @test prob3.final_time === prob.final_time

    # No-kwargs round-trip
    prob4 = remake(prob)
    @test prob4.final_time === prob.final_time
    @test prob4.cfl === prob.cfl
end

# ============================================================
# HyperbolicProblem2D
# ============================================================
@testset "remake HyperbolicProblem2D" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 10, 10)
    w0 = SVector(1.0, 0.0, 0.0, 1.0)
    prob = HyperbolicProblem2D(
        law, mesh, HLLCSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        (x, y) -> w0;
        final_time = 0.1, cfl = 0.4,
    )

    prob2 = remake(prob; final_time = 0.5)
    @test prob2.final_time == 0.5
    @test prob2.mesh === prob.mesh
    @test prob2.bc_bottom === prob.bc_bottom

    # No-kwargs round-trip
    prob3 = remake(prob)
    @test prob3.final_time === prob.final_time
end

# ============================================================
# HyperbolicProblem3D
# ============================================================
@testset "remake HyperbolicProblem3D" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 4, 4, 4)
    w0 = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
    prob = HyperbolicProblem3D(
        law, mesh, HLLCSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        (x, y, z) -> w0;
        final_time = 0.1, cfl = 0.3,
    )

    prob2 = remake(prob; cfl = 0.2)
    @test prob2.cfl == 0.2
    @test prob2.final_time === prob.final_time
    @test prob2.bc_front === prob.bc_front

    # No-kwargs round-trip
    prob3 = remake(prob)
    @test prob3.cfl === prob.cfl
end

# ============================================================
# UnstructuredHyperbolicProblem
# ============================================================
@testset "remake UnstructuredHyperbolicProblem" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    # Build a simple triangular mesh via DelaunayTriangulation
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    bnodes = [[1, 2], [2, 3], [3, 4], [4, 1]]
    tri = triangulate(pts; boundary_nodes = bnodes)
    refine!(tri; max_area = 0.05)
    umesh = UnstructuredHyperbolicMesh(tri)
    w0 = SVector(1.0, 0.0, 0.0, 1.0)
    prob = UnstructuredHyperbolicProblem(
        law, umesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), (x, y) -> w0;
        final_time = 0.1, cfl = 0.3,
    )

    prob2 = remake(prob; final_time = 0.5)
    @test prob2.final_time == 0.5
    @test prob2.mesh === prob.mesh
    @test prob2.default_bc === prob.default_bc

    # No-kwargs round-trip
    prob3 = remake(prob)
    @test prob3.final_time === prob.final_time
end

# ============================================================
# AMRProblem
# ============================================================
@testset "remake AMRProblem" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    criterion = GradientRefinement(; variable_index = 1, refine_threshold = 0.3, coarsen_threshold = 0.05)
    grid = AMRGrid(law, criterion, (8, 8), 2, (0.0, 0.0), (1.0, 1.0), Val(4))
    # Initialize with uniform state
    w0 = SVector(1.0, 0.0, 0.0, 1.0)
    for block in values(grid.blocks)
        for j in 1:block.dims[2], i in 1:block.dims[1]
            block.U[i, j] = primitive_to_conserved(law, w0)
        end
    end

    bcs = (TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC())
    prob = AMRProblem(
        grid, HLLCSolver(), NoReconstruction(),
        bcs;
        final_time = 0.1, cfl = 0.4, regrid_interval = 4,
    )

    prob2 = remake(prob; cfl = 0.2, regrid_interval = 8)
    @test prob2.cfl == 0.2
    @test prob2.regrid_interval == 8
    @test prob2.grid === prob.grid
    @test prob2.final_time === prob.final_time

    # No-kwargs round-trip
    prob3 = remake(prob)
    @test prob3.cfl === prob.cfl
    @test prob3.regrid_interval === prob.regrid_interval
end
