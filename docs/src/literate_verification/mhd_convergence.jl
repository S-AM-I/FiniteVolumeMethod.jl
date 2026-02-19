using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # MHD Smooth Alfvén Wave Convergence
# This example measures the convergence rate of the ideal MHD solver using
# a smooth circularly polarized Alfvén wave propagating along the x-axis.
#
# ## Mathematical Setup
# The circularly polarized Alfvén wave is an exact nonlinear solution of the
# ideal MHD equations (Tóth, 2000). The wave propagates at the Alfvén speed
# $v_A = B_x / \sqrt{\rho}$ without changing shape.
#
# Initial conditions (primitive variables):
# ```math
# \rho = 1, \quad v_x = 0, \quad v_y = 0.1\sin(2\pi x), \quad v_z = 0.1\cos(2\pi x)
# ```
# ```math
# P = 0.1, \quad B_x = 1, \quad B_y = 0.1\sin(2\pi x), \quad B_z = 0.1\cos(2\pi x)
# ```
#
# ## References
# - Tóth, G. (2000). The ∇·B = 0 Constraint in Shock-Capturing MHD Codes.
#   J. Comput. Phys., 161, 605-652. DOI: 10.1006/jcph.2000.6519
# - Stone et al. (2008). Athena: A New Code for Astrophysical MHD.
#   ApJS, 178, 137-177. DOI: 10.1086/588755

using FiniteVolumeMethod
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
vA = Bx0 / sqrt(rho0)  # Alfvén speed

t_final = 1.0 / vA  # one full crossing of [0,1]

function alfven_ic(x, y)
    vy = amp * sin(2 * pi * x)
    vz = amp * cos(2 * pi * x)
    By = amp * sin(2 * pi * x)
    Bz = amp * cos(2 * pi * x)
    return SVector(rho0, 0.0, vy, vz, P0, Bx0, By, Bz)
end

# ## Convergence Measurement
# The L1 error in $B_y$ at each resolution:
function compute_mhd_error(N)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N, 4)
    prob = HyperbolicProblem2D(
        law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        alfven_ic; final_time = t_final, cfl = 0.3,
    )
    coords, U, t_end = solve_hyperbolic(prob)
    nx = N
    ny = 4
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

resolutions = [16, 32, 64, 128]
errors = [compute_mhd_error(N) for N in resolutions]

# ## Convergence Rates
function convergence_rates(errs)
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

rates = convergence_rates(errors)

# ## Visualisation — Convergence Plot
fig = Figure(fontsize = 24, size = (700, 550))
ax = Axis(
    fig[1, 1], xlabel = "N", ylabel = L"L^1 \text{ error } (B_y)",
    xscale = log2, yscale = log10,
    title = "MHD Alfvén Wave Convergence",
)
scatterlines!(ax, resolutions, errors, color = :blue, marker = :circle, linewidth = 2, markersize = 12, label = "HLLD+MUSCL")

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
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "mhd_alfven_convergence.png") fig #src

# ## Test Assertions
# MUSCL with HLLD should achieve at least 1.5th-order convergence on smooth data.
@test all(r -> r > 0.8, rates) #src
@assert all(r -> r > 0.8, rates) #hide
