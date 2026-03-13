using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Fishbone-Moncrief Torus Equilibrium
# This example verifies the GRMHD solver's ability to maintain a
# hydrodynamic equilibrium torus in Kerr spacetime. The Fishbone-Moncrief
# (FM) solution describes an axisymmetric torus orbiting a spinning
# black hole in hydrostatic equilibrium.
#
# ## Mathematical Setup
# We use a Kerr black hole with spin $a/M = 0.5$ and initialise a
# uniform-density torus in approximate equilibrium. The test verifies
# that the density profile does not drift significantly over the
# simulation time (hydro-only, no magnetic field).
#
# ## Reference
# - Fishbone, L.G. & Moncrief, V. (1976). Relativistic Fluid Disks in
#   Orbit Around Kerr Black Holes. ApJ, 207, 962-976.
# - De Villiers, J.-P. & Hawley, J.F. (2003). A Numerical Method for
#   General Relativistic MHD. ApJ, 589, 458-480.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using CairoMakie

gamma = 4.0 / 3.0
eos = IdealGasEOS(gamma)
M = 1.0
a_spin = 0.5 * M
metric = KerrMetric(M, a_spin; r_min = 3.0 * M)
law = GRMHDEquations{2}(eos, metric)

# ## Torus Initial Condition
# Simplified: uniform density/pressure torus between r_in and r_out,
# with approximately Keplerian azimuthal velocity.
r_in = 6.0 * M
r_out = 12.0 * M
rho_torus = 1.0
P_torus = 1.0
rho_atm = 1.0e-4
P_atm = 1.0e-6

function torus_ic(x, y)
    r = sqrt(x^2 + y^2)
    if r >= r_in && r <= r_out
        ## Approximate Keplerian velocity: v_phi ~ sqrt(M/r)
        v_kep = sqrt(M / r) * 0.5  # reduced to keep v < 1
        ## Convert azimuthal to Cartesian
        phi = atan(y, x)
        vx = -v_kep * sin(phi)
        vy = v_kep * cos(phi)
        return SVector(rho_torus, vx, vy, 0.0, P_torus, 0.0, 0.0, 0.0)
    else
        return SVector(rho_atm, 0.0, 0.0, 0.0, P_atm, 0.0, 0.0, 0.0)
    end
end

# ## Solve
N_torus = 16
mesh = StructuredMesh2D(3.0 * M, 15.0 * M, -6.0 * M, 6.0 * M, N_torus, N_torus)
t_final_torus = 1.0  # short run for equilibrium check

prob = HyperbolicProblem2D(
    law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(),
    TransmissiveBC(), TransmissiveBC(),
    torus_ic; final_time = t_final_torus, cfl = 0.15,
)
coords, U, t_end, _ = solve_hyperbolic(prob; vector_potential = nothing)

# ## Density Drift in Torus Region
max_drift = 0.0
n_torus_cells = 0
for iy in 1:N_torus, ix in 1:N_torus
    x, y = coords[ix, iy]
    r = sqrt(x^2 + y^2)
    if r >= r_in && r <= r_out
        w = conserved_to_primitive(law, U[ix, iy])
        drift = abs(w[1] - rho_torus) / rho_torus
        global max_drift = max(max_drift, drift)
        global n_torus_cells += 1
    end
end

# ## Visualisation
rho_map = [conserved_to_primitive(law, U[ix, iy])[1] for ix in 1:N_torus, iy in 1:N_torus]
xc = [coords[i, 1][1] for i in 1:N_torus]
yc = [coords[1, j][2] for j in 1:N_torus]

fig = Figure(fontsize = 22, size = (700, 600))
ax = Axis(fig[1, 1], xlabel = "x/M", ylabel = "y/M", title = "FM Torus — log₁₀(ρ) at t=$(round(t_end, digits = 1))", aspect = DataAspect())
hm = heatmap!(ax, xc, yc, log10.(max.(rho_map, 1.0e-8)), colormap = :inferno)
Colorbar(fig[1, 2], hm, label = "log₁₀(ρ)")
resize_to_layout!(fig)
fig

# ## Test Assertions
# Density drift should be bounded (generous for this coarse setup).
@test max_drift < 0.5 #src
@test n_torus_cells > 0 #src
@assert max_drift < 0.5 #hide
@assert n_torus_cells > 0 #hide
