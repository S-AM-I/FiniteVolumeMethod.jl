# test/v_and_v_poiseuille.jl — Poiseuille channel analytical V&V (v3.10)
#
# First steady Navier-Stokes V&V for the incompressible_ns solver.
# Exercises the full SIMPLE loop against a closed-form analytical
# solution with NO boundary singularities (unlike Ghia cavity) and NO
# boundary-driven flow (unlike a moving-lid case) — just pressure- or
# inlet-driven Hagen-Poiseuille flow in a 2D channel.
#
# Analytical solution:
#   Channel: [0, L] × [0, H], no-slip at y = 0 and y = H.
#   For a fully-developed pressure-driven flow with dp/dx = -G:
#     u(y) = G / (2μ) · y · (H - y)
#     v   = 0
#   v3.10 drives this with a SpatialVelocityBC at the inlet matching
#   the analytical profile + FixedPressureBC at the outlet — the
#   interior naturally develops the same parabolic profile.
#
# This is the first test of the full pressure-velocity coupling
# against an analytical benchmark. A passing test at <5% accuracy is
# a meaningful step toward promoting `incompressible_ns` in the
# validation manifest.

using FiniteVolumeMethod
using FiniteVolumeMethod: SpatialVelocityBC
using LinearSolve
using StaticArrays: SVector
using Test

include("TestHelpers.jl")

@testset "V&V: Poiseuille channel — analytical parabolic profile" begin
    H = 1.0
    L = 5.0
    mu = 1.0
    G = 2.0   # pressure-gradient magnitude → u_max = G H² / (8μ) = 0.25

    N_x = 50
    N_y = 20
    mesh = build_cartesian_unstructured_mesh(N_x, N_y, L, H)

    u_inlet = x -> SVector(G / (2 * mu) * x[2] * (H - x[2]), 0.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => SpatialVelocityBC(u_inlet, Val(2), Float64),
        :right => FixedPressureBC(0.0),
        :bottom => NoSlipWallBC(),
        :top => NoSlipWallBC(),
    )

    algo = SIMPLE(0.5, 0.2, 500, 1.0e-6)
    prob = IncompressibleProblem(mesh, bcs, algo; nu = mu, density = 1.0)
    sol = solve(prob, algo)

    # Solver either converged or hit iter cap — doesn't matter for this
    # test as we compare the final state against the analytical profile.
    @test sol.result.iterations > 0

    # Sample centerline at x = L/2 (fully-developed region).
    i_mid = 25
    u_num = [sol.result.state.U.internal[(j - 1) * N_x + i_mid][1] for j in 1:N_y]
    v_num = [sol.result.state.U.internal[(j - 1) * N_x + i_mid][2] for j in 1:N_y]
    y_mesh = [mesh.cell_centers[2, (j - 1) * N_x + i_mid] for j in 1:N_y]

    # Point-wise comparison. Analytical peak is 0.25 at y = 0.5.
    max_rel_u = 0.0
    for (y, u) in zip(y_mesh, u_num)
        u_ex = G / (2 * mu) * y * (H - y)
        if u_ex > 0.05
            max_rel_u = max(max_rel_u, abs(u - u_ex) / u_ex)
        end
    end
    @test max_rel_u < 0.05   # 5% agreement on 50×20 mesh in fully-developed region

    # Peak velocity location: argmax of u_num should land in the middle
    # of the channel.
    _, j_peak = findmax(u_num)
    y_peak = y_mesh[j_peak]
    @test 0.4 < y_peak < 0.6
    u_peak = u_num[j_peak]
    @test 0.22 < u_peak < 0.26    # analytical: 0.25 exactly

    # v component should be essentially zero in fully-developed flow.
    max_v = maximum(abs, v_num)
    @test max_v < 0.05   # small spurious transverse velocity tolerated
end
