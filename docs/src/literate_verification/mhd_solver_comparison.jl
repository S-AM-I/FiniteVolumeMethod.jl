using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # MHD Riemann Solver Comparison: HLL vs HLLD
# This example provides a programmatic comparison of the HLL and HLLD
# Riemann solvers on the circularly polarized Alfvén wave, verifying
# that HLLD achieves lower errors than HLL at matched resolution.
#
# ## Mathematical Setup
# The circularly polarized Alfvén wave is an exact nonlinear solution
# of the ideal MHD equations (Tóth, 2000). We measure the L1 error
# in $B_y$ for both solvers across multiple resolutions.
#
# ## Reference
# - Tóth, G. (2000). The ∇·B = 0 Constraint in Shock-Capturing MHD
#   Codes. J. Comput. Phys., 161, 605-652.
# - Miyoshi, T. & Kusano, K. (2005). A multi-state HLL approximate
#   Riemann solver for ideal MHD. J. Comput. Phys., 208, 315-344.

using FiniteVolumeMethod
using OrdinaryDiffEqSSPRK: SSPRK33
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)
law = IdealMHDEquations{2}(eos)

amp = 0.1
Bx0 = 1.0
rho0 = 1.0
P0 = 0.1
vA = Bx0 / sqrt(rho0)
t_final = 1.0 / vA

function alfven_ic(x, y)
    vy = amp * sin(2 * pi * x)
    vz = amp * cos(2 * pi * x)
    By = amp * sin(2 * pi * x)
    Bz = amp * cos(2 * pi * x)
    return SVector(rho0, 0.0, vy, vz, P0, Bx0, By, Bz)
end

# ## Error Computation for a Given Solver
function compute_alfven_error(N, solver)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N, 4)
    prob = HyperbolicProblem2D(
        law, mesh, solver, CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        alfven_ic; final_time = t_final, cfl = 0.3,
    )
    ode_prob = sciml_problem(prob)
    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    limiter = mhd_stage_limiter(ode_prob.p)
    sol = solve(ode_prob, SSPRK33(; stage_limiter! = limiter); adaptive = false, dt = dt0)
    accessor = solution_accessor(prob)
    coords = get_coordinates(accessor)
    U = get_conserved(accessor, sol, length(sol.t))
    t_end = sol.t[end]
    nx = N
    err = 0.0
    for ix in 1:nx
        x = coords[ix, 1][1]
        x_shifted = mod(x - vA * t_end, 1.0)
        By_exact = amp * sin(2 * pi * x_shifted)
        By_num = conserved_to_primitive(law, U[ix, 1])[7]
        err += abs(By_num - By_exact)
    end
    return err / nx
end

# ## Convergence Study — Both Solvers
resolutions = [16, 32, 64, 128]
errors_hll = [compute_alfven_error(N, HLLSolver()) for N in resolutions]
errors_hlld = [compute_alfven_error(N, HLLDSolver()) for N in resolutions]

function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates_hll = convergence_rates(errors_hll)
rates_hlld = convergence_rates(errors_hlld)

# ## Visualisation — Error Comparison
fig = Figure(fontsize = 24, size = (700, 550))
ax = Axis(
    fig[1, 1], xlabel = "N", ylabel = L"L^1 \text{ error } (B_y)",
    xscale = log2, yscale = log10,
    title = "MHD Solver Comparison: Alfvén Wave",
)
scatterlines!(
    ax, resolutions, errors_hll, color = :orange, marker = :utriangle,
    linewidth = 2, markersize = 12, label = "HLL+MUSCL",
)
scatterlines!(
    ax, resolutions, errors_hlld, color = :blue, marker = :circle,
    linewidth = 2, markersize = 12, label = "HLLD+MUSCL",
)
e_ref = errors_hll[1]
N_ref = resolutions[1]
lines!(
    ax, resolutions, e_ref .* (N_ref ./ resolutions) .^ 1,
    color = :black, linestyle = :dash, linewidth = 1, label = L"O(N^{-1})",
)
lines!(
    ax, resolutions, e_ref .* (N_ref ./ resolutions) .^ 2,
    color = :black, linestyle = :dashdot, linewidth = 1, label = L"O(N^{-2})",
)
axislegend(ax, position = :lb)
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "mhd_solver_comparison.png") fig #src

# ## Test Assertions
# 1. HLLD must produce strictly lower L1 error than HLL at each resolution.
@test all(errors_hlld[i] < errors_hll[i] for i in eachindex(resolutions)) #src
# 2. Both solvers should achieve at least 0.8-order convergence.
@test all(r -> r > 0.8, rates_hll) #src
@test all(r -> r > 0.8, rates_hlld) #src
@assert all(errors_hlld[i] < errors_hll[i] for i in eachindex(resolutions)) #hide
@assert all(r -> r > 0.8, rates_hll) #hide
@assert all(r -> r > 0.8, rates_hlld) #hide
