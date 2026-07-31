# test/mpi_parity.jl — Serial ↔ parallel parity test for the collocated
# incompressible solver.
#
# NOT part of the default `runtests.jl` loop. Invoke manually:
#
#     mpiexec -n 2 julia --project=test test/mpi_parity.jl
#     mpiexec -n 4 julia --project=test test/mpi_parity.jl
#
# The test builds a canonical lid-driven-cavity problem, runs the serial
# SIMPLE solver on rank 0 for a reference, then runs the distributed
# SIMPLE solver on every rank. Rank 0 compares the distributed solution
# (gathered via owned-cell → global lookup) against the serial reference
# after both have reached their stationary states (2000 outer iterations
# — measured bit-stationary well before that on this problem).
#
# Acceptance criterion (measured 2026-07-31, after the Dirichlet-
# transmission halo pinning + relaxed-operator-extraction fixes): the
# Schwarz fixed point agrees with the serial fixed point to
# L∞(U) ≈ 7.6e-5, L∞(p) ≈ 2.8e-5 on the 16×16 cavity at 2 ranks, with
# the residual difference concentrated on the subdomain interface — the
# intrinsic transmission error of one-cell-overlap additive Schwarz.
# Gates are set at ~3× the measured values. Iterate-parity at 1e-6 or
# tighter would require a genuinely distributed matrix (PSparseMatrix
# path; Stage 2 follow-up in the roadmap).

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: distribute_mesh, solve_simple_distributed
using Test
using MPI
using PartitionedArrays # FVMMPIExt trigger (with MPI) — provides distribute_mesh methods
using LinearSolve
using LinearAlgebra: norm
using StaticArrays: SVector

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

nx, ny = 16, 16
Lx, Ly = 1.0, 1.0
U_lid = 1.0

function build_problem()
    mesh = build_cartesian_unstructured_mesh(nx, ny, Lx, Ly)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(),
        :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(),
        :top => FixedVelocityBC(SVector(U_lid, 0.0)),
    )
    prob = SteadyIncompressibleProblem(
        mesh, bcs, SIMPLE(; max_iterations = 2000, tolerance = 1.0e-8);
        nu = 1.0e-2, density = 1.0,
    )
    return prob
end

# Rank 0 runs the serial reference.
serial_state = nothing
if rank == 0
    prob = build_problem()
    ref_sol = solve(prob, prob.algorithm)
    serial_state = ref_sol.result.state
    println("[serial] converged=$(ref_sol.result.converged) iters=$(ref_sol.result.iterations)")
end
MPI.Barrier(comm)

# Every rank constructs the distributed problem.
prob = build_problem()
dmesh = distribute_mesh(prob.mesh, comm)
dist_result = solve_simple_distributed(prob, dmesh; verbose = (rank == 0))

# Rank 0 gathers everyone's owned-cell solutions and checks L∞ against
# the serial reference.
MPI.Barrier(comm)
local_U = [dist_result.state.U.internal[i] for i in 1:dmesh.n_owned]
local_p = [dist_result.state.p.internal[i] for i in 1:dmesh.n_owned]
local_globals = dmesh.local_to_global[1:dmesh.n_owned]

all_Ux = MPI.gather([u[1] for u in local_U], comm; root = 0)
all_Uy = MPI.gather([u[2] for u in local_U], comm; root = 0)
all_p = MPI.gather(local_p, comm; root = 0)
all_globals = MPI.gather(local_globals, comm; root = 0)

if rank == 0
    nc = nx * ny
    U_dist = Vector{SVector{2, Float64}}(undef, nc)
    p_dist = Vector{Float64}(undef, nc)
    for r in 1:nranks
        globals = all_globals[r]
        Ux_r = all_Ux[r]
        Uy_r = all_Uy[r]
        p_r = all_p[r]
        for (i, g) in pairs(globals)
            U_dist[g] = SVector(Ux_r[i], Uy_r[i])
            p_dist[g] = p_r[i]
        end
    end

    # Compare against serial reference.
    U_ser = serial_state.U.internal
    p_ser = serial_state.p.internal

    linf_U = maximum(norm(U_dist[i] - U_ser[i]) for i in 1:nc)
    linf_p = maximum(abs(p_dist[i] - p_ser[i]) for i in 1:nc)

    println("[parity] L∞(U) = $linf_U")
    println("[parity] L∞(p) = $linf_p")

    atol_U = 2.5e-4
    atol_p = 1.0e-4
    @testset "MPI parity (lid-driven cavity, $nranks ranks)" begin
        @test linf_U < atol_U
        @test linf_p < atol_p
    end
end

MPI.Barrier(comm)
MPI.Finalize()
