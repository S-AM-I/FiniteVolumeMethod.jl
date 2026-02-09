using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Brio-Wu MHD Shock Tube
# The Brio-Wu shock tube is the canonical test for MHD Riemann solvers.
# It is the MHD analogue of the Sod shock tube and produces a rich wave
# structure including fast and slow magnetosonic shocks, a compound wave,
# and a contact discontinuity.
#
# ## Problem Setup
# The 1D ideal MHD equations with 8 conserved variables
# $(\rho, \rho v_x, \rho v_y, \rho v_z, E, B_x, B_y, B_z)$ are solved
# with $\gamma = 2$ and a constant normal magnetic field $B_x = 0.75$.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using ReferenceTests #src

gamma = 2.0
eos = IdealGasEOS(gamma)
law = IdealMHDEquations{1}(eos)

# The left and right primitive states are
# $(\rho, v_x, v_y, v_z, P, B_x, B_y, B_z)$:
Bx = 0.75
wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx, 1.0, 0.0)
wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, Bx, -1.0, 0.0)

# Note the sign change in $B_y$: this is the key feature that produces
# compound MHD waves.

N = 800
mesh = StructuredMesh1D(0.0, 1.0, N)
bw_ic(x) = x < 0.5 ? wL : wR

# ## Solving with HLLD
# The HLLD solver (Miyoshi & Kusano 2005) resolves all five MHD wave
# families: two fast magnetosonic, two Alfven/rotational, and the contact.
prob = HyperbolicProblem(
    law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), bw_ic;
    final_time=0.1, cfl=0.8
)
x, U, t_final = solve_hyperbolic(prob)
x |> tc #hide

# ## Visualisation
# The Brio-Wu problem is best visualised with multiple panels showing
# different variables.
using CairoMakie

rho = [conserved_to_primitive(law, U[i])[1] for i in eachindex(U)]
vx = [conserved_to_primitive(law, U[i])[2] for i in eachindex(U)]
vy = [conserved_to_primitive(law, U[i])[3] for i in eachindex(U)]
P = [conserved_to_primitive(law, U[i])[5] for i in eachindex(U)]
By = [conserved_to_primitive(law, U[i])[7] for i in eachindex(U)]

fig = Figure(fontsize=20, size=(1200, 800))
titles = [L"\rho", L"v_x", L"v_y", "P", L"B_y"]
data = [rho, vx, vy, P, By]
for (idx, (title, d)) in enumerate(zip(titles, data))
    row = (idx - 1) ÷ 3 + 1
    col = (idx - 1) % 3 + 1
    ax = Axis(fig[row, col], xlabel="x", ylabel=title, title=title)
    lines!(ax, x, d, color=:blue, linewidth=1.5)
end
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "brio_wu_shock_tube.png") fig #src

# The solution clearly shows the compound wave structure characteristic
# of MHD: the fast rarefaction on the left, slow compound wave,
# contact discontinuity, slow shock, and fast rarefaction on the right.
# Note that $B_x$ remains constant at $0.75$ throughout (not shown).
Bx_vals = [U[i][6] for i in eachindex(U)] #hide
@assert all(b -> abs(b - 0.75) < 1e-10, Bx_vals) #hide
@assert all(rho .> 0) #hide
