```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/orszag_tang_vortex.jl"
```

````julia
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# Orszag-Tang Vortex
The Orszag-Tang vortex is an iconic 2D MHD test problem that demonstrates
the transition to MHD turbulence. Starting from smooth initial conditions,
the flow develops a complex pattern of interacting shocks and current sheets.

## Problem Setup
The domain is $[0, 1]^2$ with periodic boundary conditions and $\gamma = 5/3$.
The initial condition is a superposition of velocity and magnetic field
vortices:
```math
\rho = \gamma^2, \quad P = \gamma, \quad v_x = -\sin(2\pi y), \quad v_y = \sin(2\pi x),
```
```math
B_x = -\frac{\sin(2\pi y)}{\sqrt{4\pi}}, \quad B_y = \frac{\sin(4\pi x)}{\sqrt{4\pi}}.
```

````julia
using FiniteVolumeMethod
using OrdinaryDiffEqSSPRK: SSPRK33
using StaticArrays

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)
law = IdealMHDEquations{2}(eos)

N = 50
mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N, N)

function ot_ic(x, y)
    rho = gamma^2
    P = gamma
    vx = -sin(2 * pi * y)
    vy = sin(2 * pi * x)
    vz = 0.0
    Bx = -sin(2 * pi * y) / sqrt(4 * pi)
    By = sin(4 * pi * x) / sqrt(4 * pi)
    Bz = 0.0
    return SVector(rho, vx, vy, vz, P, Bx, By, Bz)
end
````

## Vector Potential Initialisation
For MHD with constrained transport, we initialise the magnetic field
using a vector potential $A_z(x,y)$ to guarantee $\nabla\cdot\vb B = 0$
to machine precision. The vector potential satisfying
$B_x = \partial A_z/\partial y$ and $B_y = -\partial A_z/\partial x$ is:

````julia
function Az_ot(x, y)
    return cos(2 * pi * y) / (2 * pi * sqrt(4 * pi)) +
        cos(4 * pi * x) / (4 * pi * sqrt(4 * pi))
end
````

## Solving
We use the HLLD Riemann solver with MUSCL reconstruction and periodic BCs.

````julia
prob = HyperbolicProblem2D(
    law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    ot_ic; final_time = 0.5, cfl = 0.4
)
````

The `vector_potential` keyword tells `sciml_problem` to initialise
the face-centred magnetic field from $A_z$ via Stokes' theorem. The
`mhd_stage_limiter` keeps the cell-centred $B$ consistent with the
face-centred field after every Runge-Kutta stage:

````julia
ode = sciml_problem(prob; vector_potential = Az_ot)
dt0 = compute_initial_dt(ode.p, ode.u0)
sol = solve(
    ode, SSPRK33(; stage_limiter! = mhd_stage_limiter(ode.p));
    adaptive = false, dt = dt0, save_everystep = false
)
acc = solution_accessor(prob)
U = get_conserved(acc, sol, length(sol.t))
ct = get_ct_state(acc, sol, length(sol.t))
coords = get_coordinates(acc)
t_final = sol.t[end]
coords |> tc #hide
````

## Checking $\nabla\cdot\vb B$
The constrained transport algorithm should keep $|\nabla\cdot\vb B|$
at machine precision:

````julia
divB_max = max_divB(ct, mesh)

divB_max < 1.0e-10 || @warn("divB exceeds tolerance: $divB_max") #hide
````

## Visualisation

````julia
using CairoMakie

nx, ny = N, N
xc = [coords[i, 1][1] for i in 1:nx]
yc = [coords[1, j][2] for j in 1:ny]
rho = [conserved_to_primitive(law, U[i, j])[1] for i in 1:nx, j in 1:ny]
P_vals = [conserved_to_primitive(law, U[i, j])[5] for i in 1:nx, j in 1:ny]

fig = Figure(fontsize = 24, size = (1100, 500))
ax1 = Axis(
    fig[1, 1], xlabel = "x", ylabel = "y",
    title = "Density at t = $(round(t_final, digits = 2))", aspect = DataAspect()
)
hm1 = heatmap!(ax1, xc, yc, rho, colormap = :viridis)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(
    fig[1, 3], xlabel = "x", ylabel = "y",
    title = "Pressure", aspect = DataAspect()
)
hm2 = heatmap!(ax2, xc, yc, P_vals, colormap = :magma)
Colorbar(fig[1, 4], hm2)
resize_to_layout!(fig)
fig
````

The density and pressure fields show the characteristic pattern of
interacting shocks and current sheets. The maximum $|\nabla\cdot\vb B|$
is $(round(divB_max, sigdigits=2)), confirming that constrained
transport maintains the divergence-free constraint to machine precision.

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/orszag_tang_vortex.jl).

```julia
using FiniteVolumeMethod
using OrdinaryDiffEqSSPRK: SSPRK33
using StaticArrays

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)
law = IdealMHDEquations{2}(eos)

N = 50
mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N, N)

function ot_ic(x, y)
    rho = gamma^2
    P = gamma
    vx = -sin(2 * pi * y)
    vy = sin(2 * pi * x)
    vz = 0.0
    Bx = -sin(2 * pi * y) / sqrt(4 * pi)
    By = sin(4 * pi * x) / sqrt(4 * pi)
    Bz = 0.0
    return SVector(rho, vx, vy, vz, P, Bx, By, Bz)
end

function Az_ot(x, y)
    return cos(2 * pi * y) / (2 * pi * sqrt(4 * pi)) +
        cos(4 * pi * x) / (4 * pi * sqrt(4 * pi))
end

prob = HyperbolicProblem2D(
    law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    ot_ic; final_time = 0.5, cfl = 0.4
)

ode = sciml_problem(prob; vector_potential = Az_ot)
dt0 = compute_initial_dt(ode.p, ode.u0)
sol = solve(
    ode, SSPRK33(; stage_limiter! = mhd_stage_limiter(ode.p));
    adaptive = false, dt = dt0, save_everystep = false
)
acc = solution_accessor(prob)
U = get_conserved(acc, sol, length(sol.t))
ct = get_ct_state(acc, sol, length(sol.t))
coords = get_coordinates(acc)
t_final = sol.t[end]

divB_max = max_divB(ct, mesh)


using CairoMakie

nx, ny = N, N
xc = [coords[i, 1][1] for i in 1:nx]
yc = [coords[1, j][2] for j in 1:ny]
rho = [conserved_to_primitive(law, U[i, j])[1] for i in 1:nx, j in 1:ny]
P_vals = [conserved_to_primitive(law, U[i, j])[5] for i in 1:nx, j in 1:ny]

fig = Figure(fontsize = 24, size = (1100, 500))
ax1 = Axis(
    fig[1, 1], xlabel = "x", ylabel = "y",
    title = "Density at t = $(round(t_final, digits = 2))", aspect = DataAspect()
)
hm1 = heatmap!(ax1, xc, yc, rho, colormap = :viridis)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(
    fig[1, 3], xlabel = "x", ylabel = "y",
    title = "Pressure", aspect = DataAspect()
)
hm2 = heatmap!(ax2, xc, yc, P_vals, colormap = :magma)
Colorbar(fig[1, 4], hm2)
resize_to_layout!(fig)
fig
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

