using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # WENO Convergence Study
# This tutorial compares the accuracy of MUSCL, WENO-3, and characteristic
# WENO reconstruction schemes. We first measure convergence rates on a
# smooth density wave, then apply WENO to the Sod shock tube to
# demonstrate shock-capturing.
#
# ## Smooth Density Wave
# A small-amplitude sinusoidal density perturbation is advected with
# uniform velocity and periodic boundary conditions:
# ```math
# \rho(x, 0) = 1 + 0.01\sin(2\pi x), \qquad v = 1, \qquad P = 1.
# ```

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{1}(eos)

rho0 = 1.0
v0 = 1.0
P0 = 1.0
amp = 0.01
t_final = 0.05

function advection_ic(x)
    rho = rho0 + amp * sin(2 * pi * x)
    return SVector(rho, v0, P0)
end

# ## Convergence Measurement
# We compute the $L^1$ density error at multiple resolutions for
# each reconstruction scheme.
function compute_error(N, recon)
    mesh = StructuredMesh1D(0.0, 1.0, N)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), recon,
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), advection_ic;
        final_time = t_final, cfl = 0.4
    )
    x, U, t_end = solve_hyperbolic(prob)
    err = 0.0
    for i in eachindex(x)
        x_shifted = mod(x[i] - v0 * t_end, 1.0)
        rho_exact = rho0 + amp * sin(2 * pi * x_shifted)
        rho_num = conserved_to_primitive(law, U[i])[1]
        err += abs(rho_num - rho_exact)
    end
    return err / N
end

resolutions = [32, 64, 128]

## MUSCL with minmod limiter
err_muscl = [compute_error(N, CellCenteredMUSCL(MinmodLimiter())) for N in resolutions]

## WENO-3
err_weno3 = [compute_error(N, WENO3()) for N in resolutions]

## Characteristic WENO-3
err_char_weno3 = [compute_error(N, CharacteristicWENO(WENO3())) for N in resolutions]

# ## Convergence Rates
# We compute the convergence rate as $\log_2(e_N / e_{2N})$:
using CairoMakie

function convergence_rate(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates_muscl = convergence_rate(err_muscl)
rates_weno3 = convergence_rate(err_weno3)
rates_char = convergence_rate(err_char_weno3)

fig = Figure(fontsize = 24, size = (600, 500))
ax = Axis(
    fig[1, 1], xlabel = "N", ylabel = L"L^1 \text{ error}",
    xscale = log2, yscale = log10, title = "Convergence: smooth density wave"
)
scatterlines!(
    ax, resolutions, err_muscl, label = "MUSCL (minmod)", marker = :circle,
    color = :blue, linewidth = 2, markersize = 12
)
scatterlines!(
    ax, resolutions, err_weno3, label = "WENO-3", marker = :utriangle,
    color = :red, linewidth = 2, markersize = 12
)
scatterlines!(
    ax, resolutions, err_char_weno3, label = "Char. WENO-3", marker = :diamond,
    color = :green, linewidth = 2, markersize = 12
)
## Reference slope
lines!(
    ax, resolutions, err_muscl[1] .* (resolutions[1] ./ resolutions) .^ 2,
    color = :gray, linestyle = :dash, linewidth = 1, label = L"O(N^{-2})"
)
axislegend(ax, position = :lb)
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "weno_convergence_smooth.png") fig #src

# ## Sod Shock Tube with WENO
# We also apply WENO-3 to the Sod shock tube to demonstrate
# shock-capturing capabilities.
wL = SVector(1.0, 0.0, 1.0)
wR = SVector(0.125, 0.0, 0.1)
sod_ic(x) = x < 0.5 ? wL : wR

N_sod = 200
mesh_sod = StructuredMesh1D(0.0, 1.0, N_sod)

prob_muscl = HyperbolicProblem(
    law, mesh_sod, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR), sod_ic;
    final_time = 0.2, cfl = 0.4
)
x_m, U_m, _ = solve_hyperbolic(prob_muscl)

prob_weno = HyperbolicProblem(
    law, mesh_sod, HLLCSolver(), WENO3(),
    DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR), sod_ic;
    final_time = 0.2, cfl = 0.4
)
x_w, U_w, _ = solve_hyperbolic(prob_weno)
x_w |> tc #hide

rho_muscl = [conserved_to_primitive(law, U_m[i])[1] for i in eachindex(U_m)]
rho_weno = [conserved_to_primitive(law, U_w[i])[1] for i in eachindex(U_w)]

fig2 = Figure(fontsize = 24, size = (600, 400))
ax2 = Axis(
    fig2[1, 1], xlabel = "x", ylabel = L"\rho",
    title = "Sod shock tube: MUSCL vs WENO-3"
)
scatter!(ax2, x_m, rho_muscl, color = :blue, markersize = 4, label = "MUSCL")
scatter!(ax2, x_w, rho_weno, color = :red, markersize = 4, label = "WENO-3")
axislegend(ax2, position = :cb)
resize_to_layout!(fig2)
fig2
@test_reference joinpath(@__DIR__, "../figures", "weno_convergence_sod.png") fig2 #src
