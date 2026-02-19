using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Passive Scalar Convergence
# This example measures the grid convergence rate for passive species
# advection using the `ReactiveEulerEquations` without chemistry.
#
# ## Mathematical Setup
# A Gaussian species profile is advected by uniform flow on a
# periodic domain. The exact solution at time $t$ is a simple
# translation: $Y(x,t) = Y_0(x - v t)$.
#
# ## Inputs
# - **Resolutions**: $N \in \{50, 100, 200, 400\}$
# - **Reconstruction**: `MUSCL(Minmod)` (expect ~2nd order on smooth data)
# - **Riemann solver**: `HLLCSolver` (species-aware extension)
# - **Final time**: $t_f = 0.1$

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)
law = ReactiveEulerEquations{1}(eos, (:tracer, :background))

v0 = 1.0
P0 = 1.0
rho0 = 1.0
sigma = 0.05
t_final = 0.1

function scalar_ic(x)
    Y_tracer = 0.5 * exp(-(x - 0.3)^2 / (2 * sigma^2))
    Y_bg = 1.0 - Y_tracer
    return SVector(rho0, v0, P0, Y_tracer, Y_bg)
end

# ## Convergence Measurement
function compute_scalar_error(N)
    mesh = StructuredMesh1D(0.0, 1.0, N)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), scalar_ic;
        final_time = t_final, cfl = 0.4
    )
    x, U, t_end = solve_hyperbolic(prob)
    err_L1 = 0.0
    err_L2 = 0.0
    dx = 1.0 / N
    for i in eachindex(x)
        x_shifted = mod(x[i] - v0 * t_end, 1.0)
        Y_exact = 0.5 * exp(-(x_shifted - 0.3)^2 / (2 * sigma^2))
        w = conserved_to_primitive(law, U[i])
        Y_num = w[4]
        err_L1 += abs(Y_num - Y_exact) * dx
        err_L2 += (Y_num - Y_exact)^2 * dx
    end
    return err_L1, sqrt(err_L2)
end

resolutions = [50, 100, 200, 400]
errors_L1 = Float64[]
errors_L2 = Float64[]
for N in resolutions
    e1, e2 = compute_scalar_error(N)
    push!(errors_L1, e1)
    push!(errors_L2, e2)
end

# ## Convergence Rates
function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates_L1 = convergence_rates(errors_L1)
rates_L2 = convergence_rates(errors_L2)

# ## Visualisation
fig = Figure(fontsize = 24, size = (700, 550))
ax = Axis(
    fig[1, 1], xlabel = "N", ylabel = "Error",
    xscale = log2, yscale = log10,
    title = "Passive scalar convergence"
)
scatterlines!(
    ax, resolutions, errors_L1, label = "L1", marker = :circle,
    color = :blue, linewidth = 2, markersize = 12
)
scatterlines!(
    ax, resolutions, errors_L2, label = "L2", marker = :utriangle,
    color = :red, linewidth = 2, markersize = 12
)

## Reference slopes
e_ref = errors_L1[1]
N_ref = resolutions[1]
lines!(
    ax, resolutions, e_ref .* (N_ref ./ resolutions) .^ 1,
    color = :black, linestyle = :dash, linewidth = 1, label = L"O(N^{-1})"
)
lines!(
    ax, resolutions, e_ref .* (N_ref ./ resolutions) .^ 2,
    color = :black, linestyle = :dashdot, linewidth = 1, label = L"O(N^{-2})"
)
axislegend(ax, position = :lb)
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "passive_scalar_convergence.png") fig #src

# ## Test Assertions
@test all(r -> r > 1.0, rates_L1) #src
@test all(r -> r > 1.0, rates_L2) #src
@assert all(r -> r > 1.0, rates_L1) #hide
@assert all(r -> r > 1.0, rates_L2) #hide
