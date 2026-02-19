using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Hall MHD Whistler Waves
# This tutorial demonstrates the `HallMHDEquations` type, which extends
# ideal MHD with the Hall term from the generalized Ohm's law. The
# Hall effect introduces dispersive whistler waves with phase speed
# proportional to $d_i k v_A$ (resolution-dependent).
#
# The `solve_hyperbolic` solver evolves only the ideal (hyperbolic) MHD
# part. The Hall flux corrections (`hall_flux_x`, `whistler_speed`) are
# provided as standalone utility functions. Here we show the ideal
# evolution of a circularly polarized Alfvén wave and demonstrate the
# Hall utility API.
#
# ## Problem Setup
# We initialise a circularly polarized Alfvén wave on a uniform
# background with density $\rho = 1$, pressure $P = 1$, and guide
# field $B_x = 1$. The transverse components $B_y$ and $B_z$ are
# initialised as a sinusoidal perturbation.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

eos = IdealGasEOS(5.0 / 3.0)
law = HallMHDEquations{1}(eos; di = 0.1)

# Initial condition: circularly polarized Alfvén wave.
function alfven_wave_ic(x)
    rho = 1.0
    P = 1.0
    Bx = 1.0
    amp = 0.1
    k = 2 * pi
    By = amp * sin(k * x)
    Bz = amp * cos(k * x)
    ## Alfvén wave: velocity perturbation anti-correlated with B
    vA = Bx / sqrt(rho)
    vy = -By / sqrt(rho)
    vz = -Bz / sqrt(rho)
    vx = 0.0
    return SVector(rho, vx, vy, vz, P, Bx, By, Bz)
end

N = 200
mesh = StructuredMesh1D(0.0, 1.0, N)

prob = HyperbolicProblem(
    law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), alfven_wave_ic;
    final_time = 0.5, cfl = 0.3,
)
x, U, t = solve_hyperbolic(prob)
x |> tc #hide

# ## Computing Whistler Speed Profile
# The whistler speed $c_w = d_i |B| / (\sqrt{\rho} \, \Delta x)$
# is computed at each cell to illustrate the resolution-dependent
# wave speed introduced by the Hall term.
W = to_primitive(law, U)
dx = 1.0 / N

By_vals = [W[i][7] for i in eachindex(W)]
Bz_vals = [W[i][8] for i in eachindex(W)]
cw_vals = [
    begin
            rho = W[i][1]
            B_mag = sqrt(W[i][6]^2 + W[i][7]^2 + W[i][8]^2)
            whistler_speed(law, rho, B_mag, dx)
        end
        for i in eachindex(W)
]

## Demonstrate hall_flux_x between two adjacent cells
i_mid = N ÷ 2
flux_demo = hall_flux_x(law, U[i_mid], U[i_mid + 1], dx)

# ## Visualisation
using CairoMakie

fig = Figure(fontsize = 24, size = (1200, 400))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"B_y", title = "Transverse Field By")
ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = L"B_z", title = "Transverse Field Bz")
ax3 = Axis(fig[1, 3], xlabel = "x", ylabel = L"c_w", title = "Whistler Speed")
scatter!(ax1, x, By_vals, color = :blue, markersize = 4)
scatter!(ax2, x, Bz_vals, color = :red, markersize = 4)
scatter!(ax3, x, cw_vals, color = :purple, markersize = 4)
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "hall_mhd_whistler.png") fig #src

# The Alfvén wave propagates along the domain with the ideal MHD
# solver. The whistler speed profile shows the resolution-dependent
# phase speed that the Hall term would introduce — at this grid
# spacing, $c_w \approx d_i |B| / (\sqrt{\rho} \Delta x)$.

# ## Physical Checks
@test all(W[i][1] > 0.0 for i in eachindex(W)) #src
B_mag_max = maximum(sqrt(W[i][6]^2 + W[i][7]^2 + W[i][8]^2) for i in eachindex(W)) #src
@test B_mag_max < 2.0 #src
