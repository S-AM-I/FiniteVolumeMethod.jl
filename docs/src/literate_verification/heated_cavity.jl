using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Heated Cavity — De Vahl Davis (1983)
# This example implements a simplified natural convection verification
# using the compressible Navier-Stokes solver. While the classical
# De Vahl Davis benchmark uses the Boussinesq approximation, we test
# the compressible solver's ability to capture buoyancy-driven flow
# through density variations.
#
# ## Mathematical Setup
# A differentially heated square cavity $[0,1]^2$:
# - Left wall: hot ($T_H$)
# - Right wall: cold ($T_C$)
# - Top/bottom: adiabatic (no-slip BCs)
#
# The temperature difference drives circulation through density
# variations in the compressible framework.
#
# ## Reference
# - De Vahl Davis, G. (1983). Natural Convection of Air in a Square
#   Cavity: A Bench Mark Numerical Solution. Int. J. Numer. Methods
#   Fluids, 3, 249-264.

using FiniteVolumeMethod
using OrdinaryDiffEqSSPRK: SSPRK33
using StaticArrays
using Test #src
using CairoMakie

gamma = 1.4
eos = IdealGasEOS(gamma)

# ## Parameters
# Use temperature-driven density variation via the ideal gas law.
L = 1.0
N = 32
T_hot = 1.1
T_cold = 0.9
T_mean = 1.0
rho0 = 1.0
P0 = rho0 * T_mean  # ideal gas: P = rho * T (with R = 1)
mu = 0.01
Pr = 0.72

ns = NavierStokesEquations{2}(eos, mu = mu, Pr = Pr)

# ## Initial Condition
# Start with uniform state; the temperature difference at boundaries
# will drive convection.
function heated_cavity_ic(x, y)
    ## Linear temperature profile from hot to cold
    T = T_hot + (T_cold - T_hot) * x / L
    rho = P0 / T  # ideal gas
    return SVector(rho, 0.0, 0.0, P0)
end

# ## Solve
t_final = 5.0
mesh = StructuredMesh2D(0.0, L, 0.0, L, N, N)
prob = HyperbolicProblem2D(
    ns, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    NoSlipBC(), NoSlipBC(),
    NoSlipBC(), NoSlipBC(),
    heated_cavity_ic; final_time = t_final, cfl = 0.3,
)
ode_prob = sciml_problem(prob)
dt0 = compute_initial_dt(ode_prob.p, ode_prob.u0)
sol = solve(prob, SSPRK33(); adaptive = false, dt = dt0)
accessor = solution_accessor(prob)
coords = get_coordinates(accessor)
U = reshape(get_conserved(accessor, sol, length(sol.t)), N, N)
t_end = sol.t[end]
W = to_primitive(ns, U)

# ## Extract Temperature Field
# T = P / rho for ideal gas with R = 1
T_field = [W[ix, iy][4] / W[ix, iy][1] for ix in 1:N, iy in 1:N]
vx_field = [W[ix, iy][2] for ix in 1:N, iy in 1:N]
vy_field = [W[ix, iy][3] for ix in 1:N, iy in 1:N]
vmag = sqrt.(vx_field .^ 2 .+ vy_field .^ 2)

xc = [coords[i, 1][1] for i in 1:N]
yc = [coords[1, j][2] for j in 1:N]

# ## Visualisation
fig = Figure(fontsize = 20, size = (1200, 500))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Temperature", aspect = DataAspect())
hm1 = heatmap!(ax1, xc, yc, T_field, colormap = :thermal)
Colorbar(fig[1, 2], hm1, label = "T")
ax2 = Axis(fig[1, 3], xlabel = "x", ylabel = "y", title = "|v|", aspect = DataAspect())
hm2 = heatmap!(ax2, xc, yc, vmag, colormap = :viridis)
Colorbar(fig[1, 4], hm2, label = "|v|")
resize_to_layout!(fig)
fig

# ## Test Assertions
# Temperature should be higher on the left than the right (heat transfer).
T_left_avg = sum(T_field[1, :]) / N
T_right_avg = sum(T_field[N, :]) / N
@test T_left_avg > T_right_avg #src
# Some velocity should develop (buoyancy-driven convection).
@test maximum(vmag) > 0 #src
@assert T_left_avg > T_right_avg #hide
@assert maximum(vmag) > 0 #hide
