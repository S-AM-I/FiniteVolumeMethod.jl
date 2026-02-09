using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Taylor-Green Vortex Decay
# The Taylor-Green vortex is a canonical test for viscous flow solvers.
# In the incompressible limit, the vortex decays exponentially due to
# viscosity, with a known exact solution. We solve it using the
# compressible Navier-Stokes equations at low Mach number.
#
# ## Problem Setup
# The exact solution for the velocity field is:
# ```math
# v_x = -U_0\cos(kx)\sin(ky)\,\mathrm{e}^{-2\nu k^2 t}, \qquad
# v_y = U_0\sin(kx)\cos(ky)\,\mathrm{e}^{-2\nu k^2 t},
# ```
# where $k = 2\pi/L$, $\nu = \mu/\rho_0$ is the kinematic viscosity.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

gamma = 1.4
eos = IdealGasEOS(gamma)
mu = 0.01
Pr = 0.72
ns = NavierStokesEquations{2}(eos, mu=mu, Pr=Pr)

# Physical parameters (low Mach number: $U_0 \ll c_s$):
rho0 = 1.0
P0 = 100.0   ## high pressure ensures low Mach
U0 = 0.01
L = 1.0
k = 2 * pi / L
nu = mu / rho0

function ic_tgv(x, y)
    vx = -U0 * cos(k * x) * sin(k * y)
    vy = U0 * sin(k * x) * cos(k * y)
    P = P0 - rho0 * U0^2 / 4.0 * (cos(2 * k * x) + cos(2 * k * y))
    return SVector(rho0, vx, vy, P)
end

# ## Solving
N = 32
mesh = StructuredMesh2D(0.0, L, 0.0, L, N, N)
t_final = 0.5

prob = HyperbolicProblem2D(
    ns, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    ic_tgv; final_time=t_final, cfl=0.3
)
coords, U, t_end = solve_hyperbolic(prob)
coords |> tc #hide

# ## Comparison with Exact Solution
# The exact solution decays exponentially:
decay = exp(-2 * nu * k^2 * t_end)
nx, ny = N, N

max_err_vx = 0.0
max_err_vy = 0.0
for i in 1:nx, j in 1:ny
    w = conserved_to_primitive(ns, U[i, j])
    vx_exact = -U0 * cos(k * coords[1][i]) * sin(k * coords[2][j]) * decay
    vy_exact = U0 * sin(k * coords[1][i]) * cos(k * coords[2][j]) * decay
    global max_err_vx = max(max_err_vx, abs(w[2] - vx_exact))
    global max_err_vy = max(max_err_vy, abs(w[3] - vy_exact))
end

@assert max_err_vx < 0.5 * U0 #hide
@assert max_err_vy < 0.5 * U0 #hide

# ## Visualisation
using CairoMakie

xc = [coords[1][i] for i in 1:nx]
yc = [coords[2][j] for j in 1:ny]
vx_num = [conserved_to_primitive(ns, U[i, j])[2] for i in 1:nx, j in 1:ny]
vy_num = [conserved_to_primitive(ns, U[i, j])[3] for i in 1:nx, j in 1:ny]
vx_ex = [-U0 * cos(k * xc[i]) * sin(k * yc[j]) * decay for i in 1:nx, j in 1:ny]

fig = Figure(fontsize=24, size=(1200, 500))
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y",
           title=L"v_x \text{ (numerical)}", aspect=DataAspect())
hm1 = heatmap!(ax1, xc, yc, vx_num, colormap=:RdBu)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(fig[1, 3], xlabel="x", ylabel="y",
           title=L"v_x \text{ (exact)}", aspect=DataAspect())
hm2 = heatmap!(ax2, xc, yc, vx_ex, colormap=:RdBu)
Colorbar(fig[1, 4], hm2)
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "taylor_green_vortex.png") fig #src

# The numerical solution matches the exact viscous decay very well.
# The maximum $v_x$ error is $(round(max_err_vx, sigdigits=3)),
# which is small compared to the velocity amplitude $U_0 = $(U0)$.
