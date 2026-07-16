using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Euler MMS Convergence (Hyperbolic)
# This example verifies the cell-centered hyperbolic Euler solver using a
# smooth advection test that serves as a method of manufactured solutions
# (MMS) convergence study. A smooth density perturbation is advected by
# uniform flow on a periodic domain and compared against the exact solution
# after one full period.
#
# ## Mathematical Setup
# We solve the 1D Euler equations with initial condition:
# ```math
# \rho = 1 + 0.2\sin(2\pi x), \quad v = 0.5, \quad P = 1.0
# ```
# on the periodic domain $[0,1]$. After one full advection period
# $t = L/v = 2.0$, the exact solution equals the initial condition.
# We measure L1 errors in density and pressure at four resolutions
# and compute the ASME V&V 20-2009 Grid Convergence Index (GCI).
#
# ## Reference
# - Roache, P.J. (1998). Verification and Validation in Computational
#   Science and Engineering. Hermosa Publishers.
# - ASME V&V 20-2009 Standard for Verification and Validation in
#   Computational Fluid Dynamics and Heat Transfer.

using FiniteVolumeMethod
using OrdinaryDiffEq
using OrdinaryDiffEqSSPRK: SSPRK33
using SciMLBase: ReturnCode
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)

rho0 = 1.0
v0 = 0.5
P0 = 1.0
amp = 0.2
t_final = 1.0 / v0  # one full period on [0,1]

function euler_mms_ic(x)
    rho = rho0 + amp * sin(2 * pi * x)
    return SVector(rho, v0, P0)
end

function solve_euler_mms_state(N)
    mesh = StructuredMesh1D(0.0, 1.0, N)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), euler_mms_ic;
        final_time = t_final, cfl = 0.4,
    )
    ode_prob = sciml_problem(prob)
    dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
    sol = solve(prob, SSPRK33(); adaptive = false, dt = dt0)
    @test sol.retcode == ReturnCode.Success #src
    accessor = solution_accessor(prob)
    return solution_coordinates(accessor), get_primitive(accessor, sol, length(sol.t)), sol.t[end]
end

# ## Convergence Measurement
# After one full period the exact solution equals the IC.
function compute_euler_mms_error(N)
    x, W, t_end = solve_euler_mms_state(N)
    err_rho = 0.0
    err_P = 0.0
    for i in eachindex(x)
        x_shifted = mod(x[i] - v0 * t_end, 1.0)
        w_exact = euler_mms_ic(x_shifted)
        err_rho += abs(W[i][1] - w_exact[1])
        err_P += abs(W[i][3] - w_exact[3])
    end
    return err_rho / N, err_P / N
end

resolutions = [32, 64, 128, 256]
results = [compute_euler_mms_error(N) for N in resolutions]
errors_rho = [r[1] for r in results]
errors_P = [r[2] for r in results]

# ## Convergence Rates
function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates_rho = convergence_rates(errors_rho)
rates_P = convergence_rates(errors_P)

# ## Grid Convergence Index
# Using the three finest grids (N = 64, 128, 256) with refinement ratio r = 2:
gci_rho = let e1 = errors_rho[4], e2 = errors_rho[3], e3 = errors_rho[2], r = 2.0
    p = log(e3 / e2) / log(r)
    gci_fine = 1.25 * abs(e2 - e1) / (r^p - 1)
    gci_coarse = 1.25 * abs(e3 - e2) / (r^p - 1)
    asymptotic_ratio = gci_coarse / (r^p * gci_fine)
    (; p, gci_fine, gci_coarse, asymptotic_ratio)
end

# ## Visualisation -- Solution Comparison
x_fine, W_fine, t_fine = solve_euler_mms_state(256)
rho_num = [W_fine[i][1] for i in eachindex(W_fine)]
rho_exact = [euler_mms_ic(mod(x_fine[i] - v0 * t_fine, 1.0))[1] for i in eachindex(x_fine)]

fig1 = Figure(fontsize = 24, size = (1100, 450))
ax1 = Axis(fig1[1, 1], xlabel = "x", ylabel = L"\rho", title = "N = 256")
scatter!(ax1, x_fine, rho_num, color = :blue, markersize = 4, label = "Numerical")
lines!(ax1, x_fine, rho_exact, color = :black, linewidth = 2, label = "Exact")
axislegend(ax1, position = :rt)
ax2 = Axis(fig1[1, 2], xlabel = "x", ylabel = L"|\rho - \rho_\mathrm{exact}|", title = "Pointwise Error")
scatter!(ax2, x_fine, abs.(rho_num .- rho_exact), color = :red, markersize = 4)
resize_to_layout!(fig1)
fig1
@test_reference joinpath(@__DIR__, "../figures", "euler_mms_solution.png") fig1 #src
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("euler_mms_solution.png"), fig1)
end

# ## Visualisation -- Convergence Plot
fig2 = Figure(fontsize = 24, size = (700, 550))
ax = Axis(
    fig2[1, 1], xlabel = "N", ylabel = L"L^1 \text{ error}",
    xscale = log2, yscale = log10,
    title = "Euler MMS Convergence (HLLC+MUSCL)",
)
scatterlines!(
    ax, resolutions, errors_rho, color = :blue, marker = :circle,
    linewidth = 2, markersize = 12, label = L"\rho",
)
scatterlines!(
    ax, resolutions, errors_P, color = :red, marker = :diamond,
    linewidth = 2, markersize = 12, label = "P",
)
e_ref = errors_rho[1]
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
resize_to_layout!(fig2)
fig2
@test_reference joinpath(@__DIR__, "../figures", "euler_mms_convergence.png") fig2 #src
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("euler_mms_convergence.png"), fig2)
end

# ## Test Assertions
# MUSCL with HLLC should achieve at least first-order convergence on smooth data.
@test all(r -> r > 0.8, rates_rho) #src
# Pressure is constant in the exact entropy wave, so its error stays at machine precision.
@test errors_P[end] < 1.0e-10 #src
# GCI asymptotic ratio should be near 1.0 (within tolerance for practical grids).
@test abs(gci_rho.asymptotic_ratio - 1.0) < 0.5 #src
@assert all(r -> r > 0.8, rates_rho) #hide
@assert errors_P[end] < 1.0e-10 #hide
@assert abs(gci_rho.asymptotic_ratio - 1.0) < 0.5 #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "min_density_rate" => minimum(rates_rho),
            "finest_density_error" => errors_rho[end],
            "finest_pressure_error" => errors_P[end],
            "gci_asymptotic_ratio" => gci_rho.asymptotic_ratio,
        ),
        artifacts = ["euler_mms_solution.png", "euler_mms_convergence.png"],
        notes = [
            "Canonical SciML execution path via sciml_problem(prob) and solve(prob, SSPRK33()).",
            "This evidence entry is the hyperbolic verification-stage manufactured-solution case.",
        ],
        summary = Dict(
            "resolutions" => resolutions,
            "density_errors" => errors_rho,
            "pressure_errors" => errors_P,
            "density_rates" => rates_rho,
            "pressure_rates" => rates_P,
        ),
    )
end
