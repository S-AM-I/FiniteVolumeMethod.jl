# test/v_and_v_ghia_cavity.jl — Ghia 1982 lid-driven-cavity V&V (v3.1)
#
# Reference: Ghia, Ghia, Shin (1982), "High-Re Solutions for
# Incompressible Flow Using the Navier-Stokes Equations and a Multigrid
# Method", J. Comput. Phys. 48, 387-411. Tabulated centerline u(y) at
# x = 0.5 for Re = 100 at 17 y-stations reproduced below from Table I.
#
# State in v3.2.0 (after OpenFOAM-style residual normalization landed):
# residuals drop from the v3.1 ~2% plateau to ~0.4%, and all Ghia
# reference points (except the zero-crossing at y≈0.73) agree to within
# 4% on an 80×80 mesh. Zero-crossing absolute error is 0.013 — small
# in absolute terms but relative to Ghia's +0.003 it becomes large, so
# we gate that point on absolute error.

using FiniteVolumeMethod
using FiniteVolumeMethod: continuity_residual_interior
using LinearSolve
using StaticArrays: SVector
using Test

include("TestHelpers.jl")

# Ghia 1982 Table I: Re=100 u(y) at x=0.5 (17 points, here selected).
const GHIA_Y_RE100 = [
    0.0, 0.0547, 0.1719, 0.2813, 0.5,
    0.7344, 0.8516, 0.9531, 0.9688, 1.0,
]
const GHIA_U_RE100 = [
    0.0, -0.03717, -0.1015, -0.15662, -0.20581,
    0.00332, 0.23151, 0.68717, 0.78871, 1.0,
]

@testset "V&V: Ghia lid-driven cavity Re=100 (80x80 mesh, qualitative gate)" begin
    N = 80
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(),
        :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(),
        :top => FixedVelocityBC(SVector(1.0, 0.0)),
    )
    algo = SIMPLE(; max_iterations = 2500, tolerance = 1.0e-5)
    prob = SteadyIncompressibleProblem(mesh, bcs, algo; nu = 0.01, density = 1.0)
    sol = solve(prob, algo)

    # The solver is not guaranteed to hit tolerance on this mesh (v3.0
    # known-plateau issue). We validate the OUTPUT, not the retcode.
    @test sol.result.iterations > 0

    # Extract centerline u(y) at x ≈ 0.5 — cells in column i=N/2.
    i_mid = N ÷ 2
    centerline = [
        (
                mesh.cell_centers[2, (j - 1) * N + i_mid],
                sol.result.state.U.internal[(j - 1) * N + i_mid][1],
            ) for j in 1:N
    ]

    # Peak negative u (primary vortex) should appear around y = 0.5 per Ghia.
    (peak_u, peak_i) = findmin(last.(centerline))
    peak_y = first(centerline[peak_i])

    # Peak primary-vortex u is ~ -0.206 per Ghia; post-v3.2 solver
    # produces -0.198 ± noise on 80×80.
    @test -0.22 < peak_u < -0.18
    @test 0.4 < peak_y < 0.55
    @test 0.95 < maximum(last.(centerline)) <= 1.01

    # v3.3: the true interior continuity residual (excluding the 0.1L band
    # where the lid/wall BC corner singularity concentrates the defect) is
    # a cleaner metric. On this mesh with the current under-relaxation it
    # should be < 1e-4 — two orders of magnitude below the total residual.
    interior_div = continuity_residual_interior(sol.result.state, mesh)
    @test interior_div < 1.0e-4

    # Point-wise Ghia agreement (tightened from v3.1's 30% qualitative
    # gate after the OpenFOAM-style residual normalization landed).
    # Interior points: 8% relative. Near-lid (y > 0.9): 5%.
    # Zero-crossing points (y ≈ 0.73 where |u_t| < 0.05): absolute gate.
    tol_interior = 0.08
    tol_near_lid = 0.05
    abs_zero_crossing = 0.025
    for (y_t, u_t) in zip(GHIA_Y_RE100, GHIA_U_RE100)
        _, idx = findmin(abs.(first.(centerline) .- y_t))
        _, u_c = centerline[idx]
        if abs(u_t) < 0.05
            # Zero-crossing: absolute tolerance is fairer than relative.
            @test abs(u_c - u_t) <= abs_zero_crossing
        else
            tol = y_t > 0.9 ? tol_near_lid : tol_interior
            @test abs(u_c - u_t) / abs(u_t) <= tol
        end
    end
end
