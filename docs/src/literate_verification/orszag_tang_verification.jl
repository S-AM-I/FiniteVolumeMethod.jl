using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Orszag-Tang Vortex Verification
# This example performs a grid convergence study of the Orszag-Tang vortex,
# a standard 2D MHD test problem. We verify that the solver produces
# physically reasonable results and that the constrained transport algorithm
# maintains $\nabla \cdot \mathbf{B} = 0$ to machine precision.
#
# ## Mathematical Setup
# Domain $[0,1]^2$ with periodic BCs, $\gamma = 5/3$. Initial conditions:
# ```math
# \rho = \gamma^2, \quad P = \gamma, \quad
# v_x = -\sin(2\pi y), \quad v_y = \sin(2\pi x)
# ```
# ```math
# B_x = -\frac{\sin(2\pi y)}{\sqrt{4\pi}}, \quad
# B_y = \frac{\sin(4\pi x)}{\sqrt{4\pi}}
# ```
#
# ## References
# - Orszag, S.A. & Tang, C.M. (1979). Small-scale structure of two-dimensional
#   magnetohydrodynamic turbulence. J. Fluid Mech., 90, 129-143.
# - Londrillo, P. & Del Zanna, L. (2000). High-Order Upwind Schemes for
#   Multidimensional MHD. ApJ, 530, 508-524.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src
using CairoMakie

gamma_val = 5.0 / 3.0
eos = IdealGasEOS(gamma_val)
law = IdealMHDEquations{2}(eos)

function ot_ic(x, y)
    rho = gamma_val^2
    P = gamma_val
    vx = -sin(2 * pi * y)
    vy = sin(2 * pi * x)
    Bx = -sin(2 * pi * y) / sqrt(4 * pi)
    By = sin(4 * pi * x) / sqrt(4 * pi)
    return SVector(rho, vx, vy, 0.0, P, Bx, By, 0.0)
end

function Az_ot(x, y)
    return cos(2 * pi * y) / (2 * pi * sqrt(4 * pi)) +
        cos(4 * pi * x) / (4 * pi * sqrt(4 * pi))
end

t_final = 0.5

# ## Grid Convergence Study
# Run at increasing resolutions and compare pressure along y=0.5 midline.
function solve_ot(N)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N, N)
    prob = HyperbolicProblem2D(
        law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ot_ic; final_time = t_final, cfl = 0.4,
    )
    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = Az_ot)
    return coords, U, t, ct, mesh
end

# Solve at three resolutions
N_lo, N_med, N_hi = 32, 64, 128
coords_lo, U_lo, t_lo, ct_lo, mesh_lo = solve_ot(N_lo)
coords_med, U_med, t_med, ct_med, mesh_med = solve_ot(N_med)
coords_hi, U_hi, t_hi, ct_hi, mesh_hi = solve_ot(N_hi)

# ## Divergence-Free Verification
divB_lo = max_divB(ct_lo, mesh_lo)
divB_med = max_divB(ct_med, mesh_med)
divB_hi = max_divB(ct_hi, mesh_hi)

# ## Midline Pressure Comparison
# Extract pressure along y ≈ 0.5 (the closest row to midline).
function midline_pressure(U, coords, N)
    jmid = div(N, 2)
    x_mid = [coords[i, jmid][1] for i in 1:N]
    P_mid = [conserved_to_primitive(law, U[i, jmid])[5] for i in 1:N]
    return x_mid, P_mid
end

x_lo_mid, P_lo_mid = midline_pressure(U_lo, coords_lo, N_lo)
x_med_mid, P_med_mid = midline_pressure(U_med, coords_med, N_med)
x_hi_mid, P_hi_mid = midline_pressure(U_hi, coords_hi, N_hi)

# ## Visualisation — Density Fields
xc_hi = [coords_hi[i, 1][1] for i in 1:N_hi]
yc_hi = [coords_hi[1, j][2] for j in 1:N_hi]
rho_hi = [conserved_to_primitive(law, U_hi[i, j])[1] for i in 1:N_hi, j in 1:N_hi]
P_hi = [conserved_to_primitive(law, U_hi[i, j])[5] for i in 1:N_hi, j in 1:N_hi]

fig1 = Figure(fontsize = 24, size = (1100, 500))
ax1 = Axis(
    fig1[1, 1], xlabel = "x", ylabel = "y",
    title = "Density (N=$(N_hi))", aspect = DataAspect()
)
hm1 = heatmap!(ax1, xc_hi, yc_hi, rho_hi, colormap = :viridis)
Colorbar(fig1[1, 2], hm1)
ax2 = Axis(
    fig1[1, 3], xlabel = "x", ylabel = "y",
    title = "Pressure (N=$(N_hi))", aspect = DataAspect()
)
hm2 = heatmap!(ax2, xc_hi, yc_hi, P_hi, colormap = :magma)
Colorbar(fig1[1, 4], hm2)
resize_to_layout!(fig1)
fig1
@test_reference joinpath(@__DIR__, "../figures", "orszag_tang_verification_fields.png") fig1 #src

# ## Visualisation — Midline Pressure Convergence
fig2 = Figure(fontsize = 24, size = (700, 550))
ax = Axis(
    fig2[1, 1], xlabel = "x", ylabel = "P(x, y=0.5)",
    title = "Orszag-Tang Midline Pressure"
)
lines!(
    ax, x_lo_mid, P_lo_mid, color = :red, linewidth = 1.5,
    label = "N=$(N_lo)"
)
lines!(
    ax, x_med_mid, P_med_mid, color = :blue, linewidth = 1.5,
    label = "N=$(N_med)"
)
lines!(
    ax, x_hi_mid, P_hi_mid, color = :black, linewidth = 2,
    label = "N=$(N_hi)"
)
axislegend(ax, position = :rt)
resize_to_layout!(fig2)
fig2
@test_reference joinpath(@__DIR__, "../figures", "orszag_tang_verification_midline.png") fig2 #src

# ## Test Assertions
# 1. div(B) stays at machine precision (constrained transport)
@test divB_lo < 1.0e-10 #src
@test divB_med < 1.0e-10 #src
@test divB_hi < 1.0e-10 #src
# 2. Density stays positive and bounded
rho_all_hi = [conserved_to_primitive(law, U_hi[i, j])[1] for i in 1:N_hi, j in 1:N_hi]
@test minimum(rho_all_hi) > 0 #src
@test maximum(rho_all_hi) < 10 * gamma_val^2 #src
# 3. Pressure stays positive
@test minimum(P_hi) > 0 #src
# 4. Simulation reaches final time
@test t_hi ≈ t_final #src
@assert divB_lo < 1.0e-10 #hide
@assert divB_med < 1.0e-10 #hide
@assert divB_hi < 1.0e-10 #hide
@assert minimum(rho_all_hi) > 0 #hide
@assert maximum(rho_all_hi) < 10 * gamma_val^2 #hide
@assert minimum(P_hi) > 0 #hide
@assert t_hi ≈ t_final #hide
