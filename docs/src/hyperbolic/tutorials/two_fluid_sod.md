```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/two_fluid_sod.jl"
```

````julia
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# Two-Fluid Plasma Sod Shock Tube
The two-fluid model treats ions and electrons as separate compressible
fluids. Each species evolves under its own Euler equations; coupling
(Lorentz force) is applied via source terms. In this tutorial we use
a symmetric setup with equal mass ratio ($m_i = m_e$) so that both
species produce identical Sod shock tube solutions.

## Problem Setup
The 1D two-fluid system has 6 conserved variables:
$U = [\rho_i, \rho_i v_i, E_i, \rho_e, \rho_e v_e, E_e]$

Initial conditions (primitive $[\rho_i, v_i, P_i, \rho_e, v_e, P_e]$):
```math
W = \begin{cases}[1, 0, 1, 1, 0, 1] & x < 0.5,\\[0.125, 0, 0.1, 0.125, 0, 0.1] & x \geq 0.5.\end{cases}
```

We begin by loading the package and defining the conservation law.

````julia
using FiniteVolumeMethod
using StaticArrays

eos = IdealGasEOS(1.4)
law = TwoFluidEquations{1}(eos, eos; mass_ratio = 1.0)
````

Define left and right primitive states:

````julia
wL = SVector(1.0, 0.0, 1.0, 1.0, 0.0, 1.0)
wR = SVector(0.125, 0.0, 0.1, 0.125, 0.0, 0.1)
````

Set up the mesh and solver:

````julia
N = 200
mesh = StructuredMesh1D(0.0, 1.0, N)

ic(x) = x < 0.5 ? wL : wR
````

## Solving with Lax-Friedrichs
We use the global Lax-Friedrichs solver (the most diffusive but
most robust option) with first-order reconstruction.

````julia
prob = HyperbolicProblem(
    law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.2, cfl = 0.4,
)
x, U, t = solve_hyperbolic(prob)
x |> tc #hide
````

## Visualisation
Extract primitive variables for each species.

````julia
using CairoMakie

W = to_primitive(law, U)
rho_i = [W[i][1] for i in eachindex(W)]
v_i = [W[i][2] for i in eachindex(W)]
rho_e = [W[i][4] for i in eachindex(W)]
v_e = [W[i][5] for i in eachindex(W)]

fig = Figure(fontsize = 24, size = (1200, 800))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"\rho_i", title = "Ion Density")
ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = L"v_i", title = "Ion Velocity")
ax3 = Axis(fig[2, 1], xlabel = "x", ylabel = L"\rho_e", title = "Electron Density")
ax4 = Axis(fig[2, 2], xlabel = "x", ylabel = L"v_e", title = "Electron Velocity")
scatter!(ax1, x, rho_i, color = :blue, markersize = 4)
scatter!(ax2, x, v_i, color = :red, markersize = 4)
scatter!(ax3, x, rho_e, color = :teal, markersize = 4)
scatter!(ax4, x, v_e, color = :orange, markersize = 4)
resize_to_layout!(fig)
fig
````

With equal mass ratio and identical initial conditions, both species
produce the same Sod solution. In realistic applications the mass
ratio would be $m_i / m_e \approx 1836$, leading to very different
electron and ion dynamics.

## Physical Checks
All densities and pressures must be positive. With identical ICs
and no coupling, ion and electron solutions should match.

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/two_fluid_sod.jl).

```julia
using FiniteVolumeMethod
using StaticArrays

eos = IdealGasEOS(1.4)
law = TwoFluidEquations{1}(eos, eos; mass_ratio = 1.0)

wL = SVector(1.0, 0.0, 1.0, 1.0, 0.0, 1.0)
wR = SVector(0.125, 0.0, 0.1, 0.125, 0.0, 0.1)

N = 200
mesh = StructuredMesh1D(0.0, 1.0, N)

ic(x) = x < 0.5 ? wL : wR

prob = HyperbolicProblem(
    law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.2, cfl = 0.4,
)
x, U, t = solve_hyperbolic(prob)

using CairoMakie

W = to_primitive(law, U)
rho_i = [W[i][1] for i in eachindex(W)]
v_i = [W[i][2] for i in eachindex(W)]
rho_e = [W[i][4] for i in eachindex(W)]
v_e = [W[i][5] for i in eachindex(W)]

fig = Figure(fontsize = 24, size = (1200, 800))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"\rho_i", title = "Ion Density")
ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = L"v_i", title = "Ion Velocity")
ax3 = Axis(fig[2, 1], xlabel = "x", ylabel = L"\rho_e", title = "Electron Density")
ax4 = Axis(fig[2, 2], xlabel = "x", ylabel = L"v_e", title = "Electron Velocity")
scatter!(ax1, x, rho_i, color = :blue, markersize = 4)
scatter!(ax2, x, v_i, color = :red, markersize = 4)
scatter!(ax3, x, rho_e, color = :teal, markersize = 4)
scatter!(ax4, x, v_e, color = :orange, markersize = 4)
resize_to_layout!(fig)
fig
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

