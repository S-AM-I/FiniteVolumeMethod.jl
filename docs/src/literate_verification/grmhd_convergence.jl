using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # GRMHD Minkowski Limit Convergence
# This example verifies that the GRMHD solver reproduces special-relativistic
# results when using a Minkowski metric, and measures convergence on a smooth
# density wave.
#
# ## Mathematical Setup
# We solve the 2D GRMHD equations with a `MinkowskiMetric` (flat spacetime),
# which should reproduce SRMHD results exactly. The test uses a smooth
# density perturbation advected by uniform flow:
# ```math
# \rho = 1 + 0.01\sin(2\pi x), \quad v_x = 0.3, \quad P = 1
# ```
# with constant $B_x = 0.5$ and $B_y = B_z = 0$.
#
# ## Reference
# - Del Zanna, L. et al. (2007). ECHO: a Eulerian conservative high-order
#   scheme for general relativistic MHD. A&A, 473, 11-30.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)
metric = MinkowskiMetric{2}()
law = GRMHDEquations{2}(eos, metric)

amp = 0.01
vx0 = 0.3
Bx0 = 0.5
rho0 = 1.0
P0 = 1.0
t_final = 0.5

function grmhd_ic(x, y)
    rho = rho0 + amp * sin(2 * pi * x)
    return SVector(rho, vx0, 0.0, 0.0, P0, Bx0, 0.0, 0.0)
end

# ## Convergence Measurement
function compute_grmhd_error(N)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N, 4)
    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        grmhd_ic; final_time = t_final, cfl = 0.25,
    )
    coords, U, t_end, _ = solve_hyperbolic(prob; vector_potential = nothing)
    nx = N
    err = 0.0
    for ix in 1:nx
        x = coords[ix, 1][1]
        x_shifted = mod(x - vx0 * t_end, 1.0)
        rho_exact = rho0 + amp * sin(2 * pi * x_shifted)
        rho_num = conserved_to_primitive(law, U[ix, 1])[1]
        err += abs(rho_num - rho_exact)
    end
    return err / nx
end

resolutions = [16, 32, 64, 128]
errors = [compute_grmhd_error(N) for N in resolutions]

# ## Convergence Rates
function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates = convergence_rates(errors)

# ## Visualisation — Convergence Plot
fig = Figure(fontsize = 24, size = (700, 550))
ax = Axis(
    fig[1, 1], xlabel = "N", ylabel = L"L^1 \text{ error } (\rho)",
    xscale = log2, yscale = log10,
    title = "GRMHD (Minkowski) Smooth Wave Convergence",
)
scatterlines!(ax, resolutions, errors, color = :red, marker = :circle, linewidth = 2, markersize = 12, label = "GRMHD HLL+MUSCL")
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
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "grmhd_smooth_convergence.png") fig #src

# ## Test Assertions
# With MUSCL reconstruction, expect at least first-order convergence.
# The Minkowski limit should not degrade accuracy relative to SRMHD.
@test all(r -> r > 1.0, rates) #src
@assert all(r -> r > 1.0, rates) #hide
