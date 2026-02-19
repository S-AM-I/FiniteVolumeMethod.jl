using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Navier-Stokes Taylor-Green Vortex Convergence
# This example measures the convergence rate of the Navier-Stokes solver using
# the 2D Taylor-Green vortex, which has a known analytical solution.
#
# ## Mathematical Setup
# The Taylor-Green vortex is an exact solution of the incompressible
# Navier-Stokes equations. In the low-Mach limit, it also satisfies the
# compressible equations. The velocity field decays exponentially:
# ```math
# v_x = -U_0 \cos(kx)\sin(ky)\,e^{-2\nu k^2 t}, \qquad
# v_y = U_0 \sin(kx)\cos(ky)\,e^{-2\nu k^2 t}
# ```
# where $\nu = \mu/\rho_0$ and $k = 2\pi/L$.
#
# ## Reference
# - Taylor, G.I. & Green, A.E. (1937). Mechanism of the Production of
#   Small Eddies from Large Ones. Proc. R. Soc. A, 158, 499-521.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

rho0 = 1.0
P0 = 100.0     # high pressure → low Mach
U0 = 0.01      # small velocity amplitude
L = 1.0
k = 2 * pi / L
mu = 0.01
nu = mu / rho0
t_final = 0.1

gamma = 1.4
eos = IdealGasEOS(gamma)

function tgv_ic(x, y)
    vx = -U0 * cos(k * x) * sin(k * y)
    vy = U0 * sin(k * x) * cos(k * y)
    P = P0 - rho0 * U0^2 / 4.0 * (cos(2 * k * x) + cos(2 * k * y))
    return SVector(rho0, vx, vy, P)
end

# ## Convergence Measurement
# We compute the $L^\infty$ velocity error at each resolution.
function compute_ns_error(N)
    ns = NavierStokesEquations{2}(eos, mu = mu, Pr = 0.72)
    mesh = StructuredMesh2D(0.0, L, 0.0, L, N, N)
    prob = HyperbolicProblem2D(
        ns, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        tgv_ic; final_time = t_final, cfl = 0.3,
    )
    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(ns, U)
    decay = exp(-2 * nu * k^2 * t)

    max_err = 0.0
    for iy in 1:N, ix in 1:N
        x, y = coords[ix, iy]
        vx_exact = -U0 * cos(k * x) * sin(k * y) * decay
        vy_exact = U0 * sin(k * x) * cos(k * y) * decay
        err = max(abs(W[ix, iy][2] - vx_exact), abs(W[ix, iy][3] - vy_exact))
        max_err = max(max_err, err)
    end
    return max_err
end

resolutions = [16, 32, 64]
errors = [compute_ns_error(N) for N in resolutions]

# ## Convergence Rates
function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates = convergence_rates(errors)

# ## Visualisation — Solution at N=64
ns_fine = NavierStokesEquations{2}(eos, mu = mu, Pr = 0.72)
mesh_fine = StructuredMesh2D(0.0, L, 0.0, L, 64, 64)
prob_fine = HyperbolicProblem2D(
    ns_fine, mesh_fine, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    tgv_ic; final_time = t_final, cfl = 0.3,
)
coords_fine, U_fine, t_fine = solve_hyperbolic(prob_fine)
W_fine = to_primitive(ns_fine, U_fine)
decay_fine = exp(-2 * nu * k^2 * t_fine)

xc = [coords_fine[i, 1][1] for i in 1:64]
yc = [coords_fine[1, j][2] for j in 1:64]
vmag = [sqrt(W_fine[i, j][2]^2 + W_fine[i, j][3]^2) for i in 1:64, j in 1:64]
vmag_exact = [
    sqrt(
            (-U0 * cos(k * xc[i]) * sin(k * yc[j]) * decay_fine)^2 +
            (U0 * sin(k * xc[i]) * cos(k * yc[j]) * decay_fine)^2
        ) for i in 1:64, j in 1:64
]

fig1 = Figure(fontsize = 24, size = (1100, 500))
ax1 = Axis(fig1[1, 1], xlabel = "x", ylabel = "y", title = "|v| numerical", aspect = DataAspect())
hm1 = heatmap!(ax1, xc, yc, vmag, colormap = :viridis)
Colorbar(fig1[1, 2], hm1)
ax2 = Axis(fig1[1, 3], xlabel = "x", ylabel = "y", title = "|v| exact", aspect = DataAspect())
hm2 = heatmap!(ax2, xc, yc, vmag_exact, colormap = :viridis)
Colorbar(fig1[1, 4], hm2)
resize_to_layout!(fig1)
fig1
@test_reference joinpath(@__DIR__, "../figures", "ns_tgv_solution.png") fig1 #src

# ## Visualisation — Convergence Plot
fig2 = Figure(fontsize = 24, size = (700, 550))
ax = Axis(
    fig2[1, 1], xlabel = "N", ylabel = L"L^\infty \text{ velocity error}",
    xscale = log2, yscale = log10,
    title = "Taylor-Green Vortex Convergence",
)
scatterlines!(ax, resolutions, errors, color = :blue, marker = :circle, linewidth = 2, markersize = 12, label = "HLLC+MUSCL")

## Reference slopes
e_ref = errors[1]
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
@test_reference joinpath(@__DIR__, "../figures", "ns_tgv_convergence.png") fig2 #src

# ## Test Assertions
@test all(r -> r > 0.8, rates) #src
@assert all(r -> r > 0.8, rates) #hide
