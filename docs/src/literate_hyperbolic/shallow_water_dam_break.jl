using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Shallow Water Dam Break
# The dam break problem is a standard test for the shallow water equations.
# A column of deep water (height $h_L$) is released into a shallower
# region ($h_R$), producing a rarefaction wave propagating upstream and
# a bore (shock) propagating downstream.
#
# ## Problem Setup
# The 1D shallow water equations are:
# ```math
# \pdv{}{t}\begin{pmatrix}h \\ hu\end{pmatrix} + \pdv{}{x}\begin{pmatrix}hu \\ hu^2 + \tfrac{1}{2}g h^2\end{pmatrix} = 0,
# ```
# with initial conditions:
# ```math
# (h, u) = \begin{cases}(2, 0) & x < 0.5,\\(1, 0) & x \geq 0.5.\end{cases}
# ```

# We begin by loading the package and defining the conservation law.
using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

law = ShallowWaterEquations{1}(g = 9.81)

# Define left and right primitive states $(h, u)$:
wL = SVector(2.0, 0.0)
wR = SVector(1.0, 0.0)

# Set up the mesh, boundary conditions, and initial condition:
N = 200
mesh = StructuredMesh1D(0.0, 1.0, N)

ic(x) = x < 0.5 ? wL : wR

# ## Solving with HLLC + MUSCL
# We use the HLLC Riemann solver with MUSCL reconstruction
# using the minmod limiter.
prob = HyperbolicProblem(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.15, cfl = 0.4,
)
x, U, t = solve_hyperbolic(prob)
x |> tc #hide

# ## Visualisation
# Extract the primitive variables (water height $h$ and velocity $u$).
using CairoMakie

W = to_primitive(law, U)
h_vals = [W[i][1] for i in eachindex(W)]
u_vals = [W[i][2] for i in eachindex(W)]

fig = Figure(fontsize = 24, size = (900, 400))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "h", title = "Water Height")
ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = "u", title = "Velocity")
scatter!(ax1, x, h_vals, color = :blue, markersize = 4)
scatter!(ax2, x, u_vals, color = :red, markersize = 4)
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "shallow_water_dam_break.png") fig #src

# The left rarefaction fan smoothly connects $h = 2$ to the
# intermediate plateau, while the right-moving bore sharply
# transitions the intermediate state to $h = 1$.

# ## Physical Checks
# Water height must be positive everywhere, and mass should be
# approximately conserved.
@test all(W[i][1] > 0.0 for i in eachindex(W)) #src
dx = 1.0 / N #src
mass_initial = 0.5 * 2.0 + 0.5 * 1.0 #src
mass_final = sum(W[i][1] for i in eachindex(W)) * dx #src
@test mass_final ≈ mass_initial atol = 0.02 #src
