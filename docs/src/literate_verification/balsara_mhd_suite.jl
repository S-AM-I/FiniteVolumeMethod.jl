using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Balsara MHD Riemann Test Suite
# This example runs all five standard Balsara (2001) MHD Riemann problems
# and verifies structural properties against known solution features.
# These tests exercise compound MHD wave structures including fast/slow
# shocks, rotational discontinuities, and contact discontinuities.
#
# ## Reference
# - Balsara, D.S. (2001). Total Variation Diminishing Scheme for Adiabatic
#   and Isothermal Magnetohydrodynamics. J. Comput. Phys., 174, 614-648.
# - Brio, M. & Wu, C.C. (1988). An Upwind Differencing Scheme for the
#   Equations of Ideal Magnetohydrodynamics. J. Comput. Phys., 75, 400-422.

using FiniteVolumeMethod
using OrdinaryDiffEqSSPRK: SSPRK33
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

# ## Test Problem Definitions
# Embedded directly to avoid JSON dependency in verification scripts.
balsara_tests = (
    test1 = (
        name = "Brio-Wu shock tube",
        gamma = 2.0, Bx = 0.75,
        left = (rho = 1.0, vx = 0.0, vy = 0.0, vz = 0.0, p = 1.0, By = 1.0, Bz = 0.0),
        right = (rho = 0.125, vx = 0.0, vy = 0.0, vz = 0.0, p = 0.1, By = -1.0, Bz = 0.0),
        x0 = 0.5, t_final = 0.1,
    ),
    test2 = (
        name = "Balsara 2",
        gamma = 5.0 / 3.0, Bx = 5.0 / sqrt(4 * pi),
        left = (rho = 1.0, vx = 0.0, vy = 0.0, vz = 0.0, p = 1.0, By = 5.0 / sqrt(4 * pi), Bz = 0.0),
        right = (rho = 0.125, vx = 0.0, vy = 0.0, vz = 0.0, p = 0.1, By = -5.0 / sqrt(4 * pi), Bz = 0.0),
        x0 = 0.5, t_final = 0.1,
    ),
    test3 = (
        name = "Balsara 3",
        gamma = 5.0 / 3.0, Bx = 10.0 / sqrt(4 * pi),
        left = (rho = 1.0, vx = 10.0, vy = 0.0, vz = 0.0, p = 20.0, By = 5.0 / sqrt(4 * pi), Bz = 0.0),
        right = (rho = 1.0, vx = -10.0, vy = 0.0, vz = 0.0, p = 1.0, By = 5.0 / sqrt(4 * pi), Bz = 0.0),
        x0 = 0.5, t_final = 0.08,
    ),
    test4 = (
        name = "Balsara 4",
        gamma = 5.0 / 3.0, Bx = 0.0,
        left = (rho = 1.0, vx = 0.0, vy = 0.0, vz = 0.0, p = 1.0, By = 1.0, Bz = 0.0),
        right = (rho = 0.2, vx = 0.0, vy = 0.0, vz = 0.0, p = 0.1, By = 0.0, Bz = 0.0),
        x0 = 0.5, t_final = 0.15,
    ),
    test5 = (
        name = "Balsara 5",
        gamma = 5.0 / 3.0, Bx = 5.0 / sqrt(4 * pi),
        left = (rho = 1.08, vx = 1.2, vy = 0.01, vz = 0.5, p = 0.95, By = 3.6 / sqrt(4 * pi), Bz = 2.0 / sqrt(4 * pi)),
        right = (rho = 1.0, vx = 0.0, vy = 0.0, vz = 0.0, p = 1.0, By = 4.0 / sqrt(4 * pi), Bz = 2.0 / sqrt(4 * pi)),
        x0 = 0.5, t_final = 0.2,
    ),
)

# ## Solver Helper
function solve_balsara(test; N = 400)
    eos = IdealGasEOS(test.gamma)
    law = IdealMHDEquations{1}(eos)
    L = test.left
    R = test.right
    wL = SVector(L.rho, L.vx, L.vy, L.vz, L.p, test.Bx, L.By, L.Bz)
    wR = SVector(R.rho, R.vx, R.vy, R.vz, R.p, test.Bx, R.By, R.Bz)
    ic(x) = x < test.x0 ? wL : wR

    mesh = StructuredMesh1D(0.0, 1.0, N)
    prob = HyperbolicProblem(
        law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), ic;
        final_time = test.t_final, cfl = 0.5,
    )
    ode_prob = sciml_problem(prob)
    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    sol = solve(prob, SSPRK33(); adaptive = false, dt = dt0)
    accessor = solution_accessor(prob)
    x = get_coordinates(accessor)
    U = get_conserved(accessor, sol, length(sol.t))
    t = sol.t[end]
    W = [conserved_to_primitive(law, U[i]) for i in eachindex(U)]
    return x, W, t
end

# ## Run All Tests
N_run = 400
all_results = Dict{Symbol, Any}()

for key in keys(balsara_tests)
    test = balsara_tests[key]
    x, W, t = solve_balsara(test; N = N_run)
    rho = [W[i][1] for i in eachindex(W)]
    Bx_vals = [W[i][6] for i in eachindex(W)]
    all_results[key] = (; x, W, rho, Bx_vals, t, test)
end

# ## Visualisation — Density Profiles
fig = Figure(fontsize = 18, size = (1500, 900))
for (idx, key) in enumerate(sort(collect(keys(all_results))))
    res = all_results[key]
    row = div(idx - 1, 3) + 1
    col = mod(idx - 1, 3) + 1
    ax = Axis(fig[row, col], xlabel = "x", ylabel = L"\rho", title = res.test.name)
    lines!(ax, res.x, res.rho, color = :blue, linewidth = 1.5)
end
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "balsara_mhd_suite.png") fig #src

# ## Test Assertions
# 1. Density stays positive in all tests.
for key in keys(all_results)
    res = all_results[key]
    @test all(res.rho .> 0) #src
end
# 2. Bx is conserved (constant) for each test.
for key in keys(all_results)
    res = all_results[key]
    Bx_ref = res.test.Bx
    @test all(b -> abs(b - Bx_ref) < 1.0e-10, res.Bx_vals) #src
end
# 3. Self-convergence: N=400 vs N=800 errors decrease for test1 (Brio-Wu).
x_fine, W_fine, _ = solve_balsara(balsara_tests.test1; N = 800)
rho_fine = [W_fine[i][1] for i in eachindex(W_fine)]
rho_coarse = all_results[:test1].rho
x_coarse = all_results[:test1].x
rho_fine_interp = [rho_fine[argmin(abs.(x_fine .- xc))] for xc in x_coarse]
err_400 = sum(abs(rho_coarse[i] - rho_fine_interp[i]) for i in eachindex(rho_coarse)) / N_run
x_ref, W_ref, _ = solve_balsara(balsara_tests.test1; N = 1600)
rho_ref = [W_ref[i][1] for i in eachindex(W_ref)]
rho_ref_interp = [rho_ref[argmin(abs.(x_ref .- xc))] for xc in x_fine]
err_800 = sum(abs(rho_fine[i] - rho_ref_interp[i]) for i in eachindex(rho_fine)) / 800
@test err_800 < err_400 #src
@assert all(all(all_results[key].rho .> 0) for key in keys(all_results)) #hide
@assert err_800 < err_400 #hide
