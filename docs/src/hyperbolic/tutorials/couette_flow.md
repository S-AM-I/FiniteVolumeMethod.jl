```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/couette_flow.jl"
```

````@example couette_flow
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# Couette Flow
Couette flow is the steady viscous flow between two parallel plates
where one plate is stationary and the other moves at a constant
velocity. The exact solution is a linear velocity profile:
$v_x(y) = U_w y / H$.

This tutorial demonstrates the `NavierStokesEquations` solver with
`NoSlipBC` (stationary wall) and `DirichletHyperbolicBC` (moving wall).

## Problem Setup
We solve in a channel $[0, 1] \times [0, H]$ with $H = 1$,
periodic in $x$. The bottom wall is stationary and the top wall
moves at velocity $U_w = 0.01$.

````@example couette_flow
using FiniteVolumeMethod
using StaticArrays

gamma = 1.4
eos = IdealGasEOS(gamma)
mu = 0.01
Pr = 0.72
ns = NavierStokesEquations{2}(eos, mu = mu, Pr = Pr)

# Physical parameters (low Mach number)
rho0 = 1.0
P0 = 100.0
U_wall = 0.01
H = 1.0
````

We initialise with the exact Couette profile so the simulation
remains near steady state:

````@example couette_flow
w_top = SVector(rho0, U_wall, 0.0, P0)

function ic_couette(x, y)
    vx = U_wall * y / H
    return SVector(rho0, vx, 0.0, P0)
end
````

## Boundary Conditions
- Left/Right: periodic (the flow is uniform in $x$)
- Bottom: `NoSlipBC` — negates all velocity components
- Top: `DirichletHyperbolicBC` — prescribes the moving wall state

````@example couette_flow
nx, ny = 4, 16
mesh = StructuredMesh2D(0.0, 1.0, 0.0, H, nx, ny)

prob = HyperbolicProblem2D(
    ns, mesh, HLLCSolver(), NoReconstruction(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    NoSlipBC(), DirichletHyperbolicBC(w_top),
    ic_couette; final_time = 0.5, cfl = 0.3
)
coords, U, t_final = solve_hyperbolic(prob)
coords |> tc #hide
````

## Comparison with Exact Solution

````@example couette_flow
yc = [coords[2][j] for j in 1:ny]
# Take a slice at the middle x-cell
ix_mid = nx ÷ 2 + 1
vx_num = [conserved_to_primitive(ns, U[ix_mid, j])[2] for j in 1:ny]
vx_exact = [U_wall * y / H for y in yc]

max_err = maximum(abs.(vx_num .- vx_exact))
@assert max_err < 0.1 * U_wall #hide
````

## Visualisation

````@example couette_flow
using CairoMakie

fig = Figure(fontsize = 24, size = (600, 500))
ax = Axis(
    fig[1, 1], xlabel = L"v_x", ylabel = "y",
    title = "Couette Flow: velocity profile"
)
lines!(ax, vx_exact, yc, color = :black, linewidth = 2, label = "Exact")
scatter!(ax, vx_num, yc, color = :blue, markersize = 10, label = "Numerical")
axislegend(ax, position = :lt)
resize_to_layout!(fig)
fig
````

The numerical solution closely matches the exact linear Couette
profile. The maximum error is $(round(max_err, sigdigits=3)),
which is small compared to the wall velocity $U_w = $(U_wall)$.

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/couette_flow.jl).

```julia
using FiniteVolumeMethod
using StaticArrays

gamma = 1.4
eos = IdealGasEOS(gamma)
mu = 0.01
Pr = 0.72
ns = NavierStokesEquations{2}(eos, mu = mu, Pr = Pr)

# Physical parameters (low Mach number)
rho0 = 1.0
P0 = 100.0
U_wall = 0.01
H = 1.0

w_top = SVector(rho0, U_wall, 0.0, P0)

function ic_couette(x, y)
    vx = U_wall * y / H
    return SVector(rho0, vx, 0.0, P0)
end

nx, ny = 4, 16
mesh = StructuredMesh2D(0.0, 1.0, 0.0, H, nx, ny)

prob = HyperbolicProblem2D(
    ns, mesh, HLLCSolver(), NoReconstruction(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    NoSlipBC(), DirichletHyperbolicBC(w_top),
    ic_couette; final_time = 0.5, cfl = 0.3
)
coords, U, t_final = solve_hyperbolic(prob)

yc = [coords[2][j] for j in 1:ny]
# Take a slice at the middle x-cell
ix_mid = nx ÷ 2 + 1
vx_num = [conserved_to_primitive(ns, U[ix_mid, j])[2] for j in 1:ny]
vx_exact = [U_wall * y / H for y in yc]

max_err = maximum(abs.(vx_num .- vx_exact))

using CairoMakie

fig = Figure(fontsize = 24, size = (600, 500))
ax = Axis(
    fig[1, 1], xlabel = L"v_x", ylabel = "y",
    title = "Couette Flow: velocity profile"
)
lines!(ax, vx_exact, yc, color = :black, linewidth = 2, label = "Exact")
scatter!(ax, vx_num, yc, color = :blue, markersize = 10, label = "Numerical")
axislegend(ax, position = :lt)
resize_to_layout!(fig)
fig
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

