using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # GRMHD Newtonian Limit — Low-Velocity Con2Prim Stability
# This example verifies that the GRMHD conserved-to-primitive (con2prim)
# variable inversion remains stable and convergent in the Newtonian limit
# (low velocity). Two checks are performed:
#
# 1. **Con2Prim round-trip at low velocities**: Verify that
#    `conserved_to_primitive(primitive_to_conserved(W))` recovers `W`
#    at velocities $v \sim 10^{-6}, 10^{-8}, 10^{-10}$ in Schwarzschild.
# 2. **Static atmosphere stationarity**: A uniform atmosphere at rest in
#    Schwarzschild spacetime should remain stationary (within numerical
#    drift tolerance) after many timesteps.
#
# ## Reference
# - Noble, S.C. et al. (2006). Primitive Variable Solvers for Conservative
#   General Relativistic Magnetohydrodynamics. ApJ, 641, 626-637.
# - Font, J.A. (2008). Numerical Hydrodynamics and Magnetohydrodynamics in
#   General Relativity. Living Rev. Relativity, 11, 7.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using CairoMakie

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)

# ## Check 1 — Con2Prim Round-Trip at Low Velocities
M = 1.0
metric_s = SchwarzschildMetric(M; r_min = 3.0)
law_s = GRMHDEquations{2}(eos, metric_s)

velocities = [1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10]
roundtrip_errors = Float64[]

for v in velocities
    w = SVector(1.0, v, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0)
    u = primitive_to_conserved(law_s, w)
    w_rec = conserved_to_primitive(law_s, u)
    err = maximum(abs.(w_rec .- w) ./ (abs.(w) .+ 1.0e-15))
    push!(roundtrip_errors, err)
end

# ## Check 2 — Static Atmosphere Stationarity
# A uniform atmosphere at rest in Schwarzschild. We run a short simulation
# and verify that the solution does not drift significantly.
metric_atm = SchwarzschildMetric(1.0; r_min = 4.0)
law_atm = GRMHDEquations{2}(eos, metric_atm)

rho_atm = 1.0
P_atm = 1.0
N_atm = 16
t_final_atm = 0.5

function static_atm_ic(x, y)
    return SVector(rho_atm, 0.0, 0.0, 0.0, P_atm, 0.0, 0.0, 0.0)
end

mesh_atm = StructuredMesh2D(4.0, 8.0, 0.0, 1.0, N_atm, N_atm)
prob_atm = HyperbolicProblem2D(
    law_atm, mesh_atm, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    static_atm_ic; final_time = t_final_atm, cfl = 0.2,
)
coords_atm, U_atm, t_atm, _ = solve_hyperbolic(prob_atm; vector_potential = nothing)

## Compute maximum density drift.
## On the curved (non-Minkowski) path the returned state is the
## DENSITIZED Valencia state sqrt(gamma)*[D, S_j, tau, B], so primitives
## must be recovered with the metric-aware con2prim; the flat
## `conserved_to_primitive` would misread the sqrt(gamma) factor
## (~0.22 at the inner boundary) as a spurious density drift.
## Note also that a uniform-pressure fluid at rest is only a
## *near*-equilibrium in Schwarzschild (the geometric sources drive a
## weak physical infall, |v| ~ 1e-2 by t = 0.5 at these radii); the
## measured drift is that genuine weak-field response plus truncation
## error, about 6.5e-3 on this grid — well within the 0.05 gate.
W_atm = FiniteVolumeMethod.grmhd_recover_primitive_field(law_atm, U_atm, mesh_atm)
max_drift = 0.0
for iy in 1:N_atm, ix in 1:N_atm
    drift = abs(W_atm[ix, iy][1] - rho_atm) / rho_atm
    global max_drift = max(max_drift, drift)
end

# ## Visualisation
fig = Figure(fontsize = 22, size = (1100, 450))
ax1 = Axis(
    fig[1, 1], xlabel = "Velocity", ylabel = "Relative round-trip error",
    xscale = log10, yscale = log10,
    title = "Con2Prim Stability (Schwarzschild)",
)
scatterlines!(
    ax1, velocities, max.(roundtrip_errors, 1.0e-16),
    color = :blue, marker = :circle, linewidth = 2, markersize = 12,
)
hlines!(ax1, [1.0e-10], color = :red, linestyle = :dash, linewidth = 1.5, label = "Threshold")
axislegend(ax1, position = :rt)

ax2 = Axis(
    fig[1, 2], xlabel = "x", ylabel = "y",
    title = "Density Drift (static atm, t=$(round(t_atm, digits = 2)))",
)
drift_map = [
    abs(W_atm[ix, iy][1] - rho_atm) / rho_atm
        for ix in 1:N_atm, iy in 1:N_atm
]
xc = [coords_atm[i, 1][1] for i in 1:N_atm]
yc = [coords_atm[1, j][2] for j in 1:N_atm]
hm = heatmap!(ax2, xc, yc, drift_map, colormap = :hot)
Colorbar(fig[1, 3], hm, label = "Relative drift")
resize_to_layout!(fig)
fig

# ## Test Assertions
# Con2Prim should converge for all tested velocities.
@test all(roundtrip_errors .< 1.0e-6) #src
# Static atmosphere drift should be small.
@test max_drift < 0.05 #src
@assert all(roundtrip_errors .< 1.0e-6) #hide
@assert max_drift < 0.05 #hide
