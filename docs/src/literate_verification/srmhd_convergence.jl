using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # SRMHD Smooth Wave Convergence
# This example measures the convergence rate of the special-relativistic MHD
# solver using a smooth sinusoidal density perturbation advected by uniform
# flow with periodic boundary conditions.
#
# ## Mathematical Setup
# We solve the 1D SRMHD equations with initial condition:
# ```math
# \rho = 1 + 0.01\sin(2\pi x), \quad v_x = 0.5, \quad v_y = v_z = 0
# ```
# ```math
# P = 1, \quad B_x = 0.5, \quad B_y = B_z = 0
# ```
# Since the perturbation is smooth and the background flow is uniform with
# $B_y = B_z = 0$, the density wave advects at $v_x = 0.5$ without distortion.
#
# ## Reference
# - Balsara, D.S. (2001). Total Variation Diminishing Scheme for Adiabatic
#   and Isothermal Magnetohydrodynamics. J. Comput. Phys., 174, 614-648.
# - Mignone, A. & Bodo, G. (2006). An HLLC Riemann solver for relativistic
#   flows. Mon. Not. R. Astron. Soc., 364, 126-136.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)
law = SRMHDEquations{1}(eos)

amp = 0.01
vx0 = 0.5
Bx0 = 0.5
rho0 = 1.0
P0 = 1.0
t_final = 0.5  # wave travels 0.25 of domain

function srmhd_smooth_ic(x)
    rho = rho0 + amp * sin(2 * pi * x)
    return SVector(rho, vx0, 0.0, 0.0, P0, Bx0, 0.0, 0.0)
end

# ## Convergence Measurement
function compute_srmhd_error(N)
    mesh = StructuredMesh1D(0.0, 1.0, N)
    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), srmhd_smooth_ic;
        final_time = t_final, cfl = 0.3,
    )
    x, U, t_end = solve_hyperbolic(prob)
    err = 0.0
    for i in eachindex(x)
        x_shifted = mod(x[i] - vx0 * t_end, 1.0)
        rho_exact = rho0 + amp * sin(2 * pi * x_shifted)
        rho_num = conserved_to_primitive(law, U[i])[1]
        err += abs(rho_num - rho_exact)
    end
    return err / N
end

resolutions = [32, 64, 128, 256]
errors = [compute_srmhd_error(N) for N in resolutions]

# ## Convergence Rates
function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates = convergence_rates(errors)

# ## Visualisation — Solutions
mesh_lo = StructuredMesh1D(0.0, 1.0, 64)
mesh_hi = StructuredMesh1D(0.0, 1.0, 256)
prob_lo = HyperbolicProblem(
    law, mesh_lo, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), srmhd_smooth_ic;
    final_time = t_final, cfl = 0.3,
)
prob_hi = HyperbolicProblem(
    law, mesh_hi, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), srmhd_smooth_ic;
    final_time = t_final, cfl = 0.3,
)
x_lo, U_lo, t_lo = solve_hyperbolic(prob_lo)
x_hi, U_hi, t_hi = solve_hyperbolic(prob_hi)
rho_lo = [conserved_to_primitive(law, U_lo[i])[1] for i in eachindex(U_lo)]
rho_hi = [conserved_to_primitive(law, U_hi[i])[1] for i in eachindex(U_hi)]
x_exact = range(0.0, 1.0, length = 500)
rho_exact = [rho0 + amp * sin(2 * pi * mod(xi - vx0 * t_lo, 1.0)) for xi in x_exact]

fig1 = Figure(fontsize = 24, size = (1100, 450))
ax1 = Axis(fig1[1, 1], xlabel = "x", ylabel = L"\rho", title = "N = 64")
scatter!(ax1, x_lo, rho_lo, color = :blue, markersize = 6, label = "Numerical")
lines!(ax1, x_exact, rho_exact, color = :black, linewidth = 2, label = "Exact")
axislegend(ax1, position = :rt)
ax2 = Axis(fig1[1, 2], xlabel = "x", ylabel = L"\rho", title = "N = 256")
scatter!(ax2, x_hi, rho_hi, color = :blue, markersize = 3, label = "Numerical")
lines!(ax2, x_exact, rho_exact, color = :black, linewidth = 2, label = "Exact")
axislegend(ax2, position = :rt)
resize_to_layout!(fig1)
fig1
@test_reference joinpath(@__DIR__, "../figures", "srmhd_smooth_solutions.png") fig1 #src

# ## Visualisation — Convergence Plot
fig2 = Figure(fontsize = 24, size = (700, 550))
ax = Axis(
    fig2[1, 1], xlabel = "N", ylabel = L"L^1 \text{ error } (\rho)",
    xscale = log2, yscale = log10,
    title = "SRMHD Smooth Wave Convergence",
)
scatterlines!(ax, resolutions, errors, color = :blue, marker = :circle, linewidth = 2, markersize = 12, label = "HLL+MUSCL")
e_ref = errors[1]
N_ref = resolutions[1]
lines!(
    ax, resolutions, e_ref .* (N_ref ./ resolutions) .^ 2,
    color = :black, linestyle = :dashdot, linewidth = 1, label = L"O(N^{-2})",
)
axislegend(ax, position = :lb)
resize_to_layout!(fig2)
fig2
@test_reference joinpath(@__DIR__, "../figures", "srmhd_smooth_convergence.png") fig2 #src

# ## Test Assertions
@test all(r -> r > 0.8, rates) #src
@assert all(r -> r > 0.8, rates) #hide
