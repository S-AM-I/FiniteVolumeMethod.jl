```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/shallow_water_dam_break.jl"
```

````julia
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# Shallow Water Dam Break
The dam break problem is a standard test for the shallow water equations.
A column of deep water (height $h_L$) is released into a shallower
region ($h_R$), producing a rarefaction wave propagating upstream and
a bore (shock) propagating downstream.

## Problem Setup
The 1D shallow water equations are:
```math
\pdv{}{t}\begin{pmatrix}h \\ hu\end{pmatrix} + \pdv{}{x}\begin{pmatrix}hu \\ hu^2 + \tfrac{1}{2}g h^2\end{pmatrix} = 0,
```
with initial conditions:
```math
(h, u) = \begin{cases}(2, 0) & x < 0.5,\\(1, 0) & x \geq 0.5.\end{cases}
```

We begin by loading the package and defining the conservation law.

````julia
using FiniteVolumeMethod
using OrdinaryDiffEqSSPRK: SSPRK33
using StaticArrays

law = ShallowWaterEquations{1}(g = 9.81)
````

Define left and right primitive states $(h, u)$:

````julia
wL = SVector(2.0, 0.0)
wR = SVector(1.0, 0.0)
````

Set up the mesh, boundary conditions, and initial condition:

````julia
N = 200
mesh = StructuredMesh1D(0.0, 1.0, N)

ic(x) = x < 0.5 ? wL : wR
````

## Solving with HLLC + MUSCL
We use the HLLC Riemann solver with MUSCL reconstruction
using the minmod limiter.

````julia
prob = HyperbolicProblem(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.15, cfl = 0.4,
)
ode = sciml_problem(prob)
dt0 = compute_initial_dt(ode.p, ode.u0)
sol = solve(ode, SSPRK33(); adaptive = false, dt = dt0)
acc = solution_accessor(prob)
U = get_conserved(acc, sol, length(sol.t))
x = get_coordinates(acc)
t = sol.t[end]
x |> tc #hide
````

## Visualisation
Extract the primitive variables (water height $h$ and velocity $u$).

````julia
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
````

The left rarefaction fan smoothly connects $h = 2$ to the
intermediate plateau, while the right-moving bore sharply
transitions the intermediate state to $h = 1$.

## Physical Checks
Water height must be positive everywhere, and mass should be
approximately conserved.

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/shallow_water_dam_break.jl).

```julia
using FiniteVolumeMethod
using OrdinaryDiffEqSSPRK: SSPRK33
using StaticArrays

law = ShallowWaterEquations{1}(g = 9.81)

wL = SVector(2.0, 0.0)
wR = SVector(1.0, 0.0)

N = 200
mesh = StructuredMesh1D(0.0, 1.0, N)

ic(x) = x < 0.5 ? wL : wR

prob = HyperbolicProblem(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(), ic;
    final_time = 0.15, cfl = 0.4,
)
ode = sciml_problem(prob)
dt0 = compute_initial_dt(ode.p, ode.u0)
sol = solve(ode, SSPRK33(); adaptive = false, dt = dt0)
acc = solution_accessor(prob)
U = get_conserved(acc, sol, length(sol.t))
x = get_coordinates(acc)
t = sol.t[end]

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
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

