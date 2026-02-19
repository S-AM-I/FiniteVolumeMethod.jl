using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Acoustic Wave Convergence
# This example verifies the 1D Euler solver's convergence on a smooth acoustic
# wave with a known exact solution. A small-amplitude right-traveling sound
# wave propagates on a uniform background. After one full period, the wave
# returns to its initial position on the periodic domain.
#
# ## Mathematical Setup
# The linearised Euler equations admit a right-traveling acoustic wave:
# ```math
# \rho = \rho_0 + \varepsilon\sin(2\pi x), \quad
# v = \frac{\varepsilon c_s}{\rho_0}\sin(2\pi x), \quad
# P = P_0 + \varepsilon c_s^2 \sin(2\pi x)
# ```
# where $c_s = \sqrt{\gamma P_0/\rho_0}$ is the sound speed. The wave
# propagates with speed $c_s$ and returns to the initial condition at
# $t = 1/c_s$ on the domain $[0,1]$.
#
# ## Reference
# - LeVeque, R.J. (2002). Finite Volume Methods for Hyperbolic Problems.
#   Cambridge University Press. Chapter 14.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)

rho0 = 1.0
P0 = 1.0
cs = sqrt(gamma * P0 / rho0)  # sound speed
eps_amp = 1.0e-4               # small amplitude for linearity
t_final = 1.0 / cs            # one full period

function acoustic_ic(x)
    rho = rho0 + eps_amp * sin(2 * pi * x)
    v = eps_amp * cs / rho0 * sin(2 * pi * x)
    P = P0 + eps_amp * cs^2 * sin(2 * pi * x)
    return SVector(rho, v, P)
end

# ## Convergence Measurement
# After one full period, the exact solution equals the IC.
function compute_acoustic_error(N)
    mesh = StructuredMesh1D(0.0, 1.0, N)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), acoustic_ic;
        final_time = t_final, cfl = 0.4,
    )
    x, U, t_end = solve_hyperbolic(prob)

    err_rho = 0.0
    err_P = 0.0
    for i in eachindex(x)
        w_num = conserved_to_primitive(law, U[i])
        w_exact = acoustic_ic(x[i])
        err_rho += abs(w_num[1] - w_exact[1])
        err_P += abs(w_num[3] - w_exact[3])
    end
    return err_rho / N, err_P / N
end

resolutions = [32, 64, 128, 256]
results = [compute_acoustic_error(N) for N in resolutions]
errors_rho = [r[1] for r in results]
errors_P = [r[2] for r in results]

# ## Convergence Rates
function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates_rho = convergence_rates(errors_rho)
rates_P = convergence_rates(errors_P)

# ## Visualisation
fig = Figure(fontsize = 24, size = (700, 550))
ax = Axis(
    fig[1, 1], xlabel = "N", ylabel = L"L^1 \text{ error}",
    xscale = log2, yscale = log10,
    title = "Acoustic Wave Convergence (Euler)",
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
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "source_term_convergence.png") fig #src

# ## Test Assertions
# MUSCL with HLLC should achieve at least first-order convergence on
# smooth acoustic data.
@test all(r -> r > 0.8, rates_rho) #src
@test all(r -> r > 0.8, rates_P) #src
@assert all(r -> r > 0.8, rates_rho) #hide
@assert all(r -> r > 0.8, rates_P) #hide
