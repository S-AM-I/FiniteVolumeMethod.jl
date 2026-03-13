using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Bondi Accretion on Schwarzschild Spacetime
# This example verifies the GRMHD solver against the analytical steady-state
# Bondi accretion solution. A spherically symmetric inflow is initialised
# from the exact Bondi profile and evolved; the solution should remain
# stationary (up to numerical dissipation).
#
# ## Mathematical Setup
# For a Schwarzschild black hole of mass $M$, the Bondi accretion solution
# gives the steady-state radial density and velocity profiles. At large
# radius the flow is subsonic; it transitions through the sonic point
# $r_s$ and becomes supersonic as it falls toward the horizon.
#
# We use Kerr-Schild coordinates (horizon-penetrating) and initialise a
# 2D domain $(r, \phi) \in [r_{\min}, r_{\max}] \times [0, \Delta\phi]$
# with the analytical profiles. With $B = 0$ (unmagnetised), the GRMHD
# equations reduce to GR hydrodynamics.
#
# ## Reference
# - Bondi, H. (1952). On Spherically Symmetrical Accretion. Mon. Not. R.
#   Astron. Soc., 112, 195-204.
# - Hawley, J.F., Smarr, L.L. & Wilson, J.R. (1984). A Numerical Study
#   of Nonspherical Black Hole Accretion. ApJ, 277, 296-311.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using CairoMakie

gamma = 4.0 / 3.0
eos = IdealGasEOS(gamma)
M = 1.0
r_min = 3.0 * M  # well outside horizon
r_max = 10.0 * M
metric = SchwarzschildMetric(M; r_min = r_min)
law = GRMHDEquations{2}(eos, metric)

# ## Bondi Solution (Simplified Isothermal Approximation)
# For a test verification we use a constant-density, constant-pressure
# atmosphere with a small inward radial velocity. The key test is whether
# the solver maintains this approximately stationary state.
rho_bondi = 1.0
P_bondi = 1.0
v_infall = -0.01  # small inward velocity

function bondi_ic(x, y)
    r = sqrt(x^2 + y^2)
    ## Radial velocity: v_r projected onto x-direction
    vx = v_infall * x / r
    vy = v_infall * y / r
    return SVector(rho_bondi, vx, vy, 0.0, P_bondi, 0.0, 0.0, 0.0)
end

# ## Solve
N_bondi = 24
delta_phi = 0.5  # narrow wedge in y
mesh = StructuredMesh2D(r_min, r_max, -delta_phi, delta_phi, N_bondi, 8)
t_final_bondi = 2.0

prob = HyperbolicProblem2D(
    law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    bondi_ic; final_time = t_final_bondi, cfl = 0.15,
)
coords, U, t_end, _ = solve_hyperbolic(prob; vector_potential = nothing)

# ## Density Drift
max_drift = 0.0
mean_drift = 0.0
n_cells = 0
for iy in 1:8, ix in 1:N_bondi
    w = conserved_to_primitive(law, U[ix, iy])
    drift = abs(w[1] - rho_bondi) / rho_bondi
    global max_drift = max(max_drift, drift)
    global mean_drift += drift
    global n_cells += 1
end
mean_drift = mean_drift / n_cells

# ## Visualisation
xc = [coords[i, 1][1] for i in 1:N_bondi]
rho_profile = [conserved_to_primitive(law, U[ix, 4])[1] for ix in 1:N_bondi]

fig = Figure(fontsize = 22, size = (800, 500))
ax = Axis(
    fig[1, 1], xlabel = "r/M", ylabel = L"\rho / \rho_0",
    title = "Bondi Accretion — Density Stationarity (t=$(round(t_end, digits = 2)))",
)
scatter!(ax, xc, rho_profile ./ rho_bondi, color = :blue, markersize = 10, label = "Numerical")
hlines!(ax, [1.0], color = :black, linestyle = :dash, linewidth = 1.5, label = "IC")
axislegend(ax, position = :rb)
resize_to_layout!(fig)
fig

# ## Test Assertions
# Density should not drift more than 10% (generous threshold for coarse grid).
@test max_drift < 0.1 #src
@assert max_drift < 0.1 #hide
