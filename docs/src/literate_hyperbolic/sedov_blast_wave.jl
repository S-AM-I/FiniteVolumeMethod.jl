using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Sedov Blast Wave
# The Sedov blast wave is a 2D point-explosion problem with a known
# self-similar solution. A concentrated energy release at the origin
# drives a cylindrical shock wave whose radius evolves as
# $r_s(t) \propto t^{2/5}$.
#
# ## Problem Setup
# We solve the 2D Euler equations on $[-1, 1]^2$ with transmissive
# boundary conditions. The initial condition is a uniform density field
# with a small, high-pressure region near the origin.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

gamma = 1.4
eos = IdealGasEOS(gamma)
law = EulerEquations{2}(eos)

N = 50
mesh = StructuredMesh2D(-1.0, 1.0, -1.0, 1.0, N, N)

# The background pressure is very low, with a strong blast in the central cells:
P_bg = 1e-5
P_blast = 1.0
r_blast = 3.0 * mesh.dx

function sedov_ic(x, y)
    r = sqrt(x^2 + y^2)
    P = r < r_blast ? P_blast : P_bg
    return SVector(1.0, 0.0, 0.0, P)
end

# We use transmissive boundary conditions on all four sides:
prob = HyperbolicProblem2D(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(),
    TransmissiveBC(), TransmissiveBC(),
    sedov_ic; final_time=0.1, cfl=0.3
)
coords, U, t_final = solve_hyperbolic(prob)
coords |> tc #hide

# ## Visualisation
# We extract the density field and plot it as a 2D heatmap.
using CairoMakie

nx, ny = N, N
xc = [coords[1][i] for i in 1:nx]
yc = [coords[2][j] for j in 1:ny]
rho = [conserved_to_primitive(law, U[i, j])[1] for i in 1:nx, j in 1:ny]

fig = Figure(fontsize=24, size=(1000, 500))
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Density at t = $(round(t_final, digits=3))",
           aspect=DataAspect())
hm = heatmap!(ax1, xc, yc, rho, colormap=:inferno)
Colorbar(fig[1, 2], hm)

# We also plot the radial density profile to check cylindrical symmetry:
ax2 = Axis(fig[1, 3], xlabel="r", ylabel=L"\rho", title="Radial profile")
r_vals = Float64[]
rho_vals = Float64[]
for i in 1:nx, j in 1:ny
    r = sqrt(coords[1][i]^2 + coords[2][j]^2)
    push!(r_vals, r)
    push!(rho_vals, rho[i, j])
end
scatter!(ax2, r_vals, rho_vals, color=:blue, markersize=3)
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "sedov_blast_wave.png") fig #src

# The density profile shows a clear cylindrical shock front with a thin
# shell of compressed gas. The radial profile confirms the approximate
# cylindrical symmetry of the solution.
@assert all(rho .> 0) #hide
