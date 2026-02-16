```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/srmhd_cylindrical_blast.jl"
```

````@example srmhd_cylindrical_blast
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# SRMHD Cylindrical Blast Wave
This tutorial demonstrates the 2D special relativistic MHD (SRMHD)
solver on a cylindrical blast wave problem. A high-pressure region
is placed at the centre of the domain, driving a relativistic
expansion into a magnetised ambient medium. The magnetic field
breaks the cylindrical symmetry, producing an elongated shock
structure along the field direction.

## Problem Setup
The domain is $[-0.5, 0.5]^2$ with transmissive boundaries.
A uniform magnetic field $B_x = 0.5$ threads the domain.
The initial pressure is $P = 1$ inside a circle of radius $r = 0.1$
and $P = 0.01$ outside.

````@example srmhd_cylindrical_blast
using FiniteVolumeMethod
using StaticArrays

gamma = 4.0 / 3.0
eos = IdealGasEOS(gamma)
law = SRMHDEquations{2}(eos)

P_in = 1.0
P_out = 0.01
rho0 = 0.01
r_blast = 0.1
Bx0 = 0.5

function blast_ic(x, y)
    r = sqrt(x^2 + y^2)
    P = r < r_blast ? P_in : P_out
    return SVector(rho0, 0.0, 0.0, 0.0, P, Bx0, 0.0, 0.0)
end
````

Vector potential for uniform $B_x$:

````@example srmhd_cylindrical_blast
Az_blast(x, y) = Bx0 * y
````

## Solving

````@example srmhd_cylindrical_blast
N = 100
mesh = StructuredMesh2D(-0.5, 0.5, -0.5, 0.5, N, N)

prob = HyperbolicProblem2D(
    law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(),
    TransmissiveBC(), TransmissiveBC(),
    blast_ic; final_time = 0.4, cfl = 0.25
)

coords, U, t_final, ct = solve_hyperbolic(prob; vector_potential = Az_blast)
coords |> tc #hide
````

## Divergence Check

````@example srmhd_cylindrical_blast
divB_max = max_divB(ct, mesh)
@assert divB_max < 1.0e-12 #hide
````

## Visualisation

````@example srmhd_cylindrical_blast
using CairoMakie

nx, ny = N, N
xc = [coords[i, 1][1] for i in 1:nx]
yc = [coords[1, j][2] for j in 1:ny]

rho = [conserved_to_primitive(law, U[i, j])[1] for i in 1:nx, j in 1:ny]
P = [conserved_to_primitive(law, U[i, j])[5] for i in 1:nx, j in 1:ny]

fig = Figure(fontsize = 20, size = (1000, 500))
ax1 = Axis(
    fig[1, 1], xlabel = "x", ylabel = "y",
    title = "Density at t = $(round(t_final, digits = 3))",
    aspect = DataAspect()
)
hm1 = heatmap!(ax1, xc, yc, rho, colormap = :viridis)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(
    fig[1, 3], xlabel = "x", ylabel = "y",
    title = "Pressure",
    aspect = DataAspect()
)
hm2 = heatmap!(ax2, xc, yc, P, colormap = :inferno)
Colorbar(fig[1, 4], hm2)

resize_to_layout!(fig)
fig
````

The blast wave is clearly elongated along the $x$-direction due to
the uniform $B_x$ field, which channels the expansion. The
$\nabla\cdot\vb B$ constraint is maintained at machine precision
($|\nabla\cdot\vb B|_{\max} = $ $(round(divB_max, sigdigits=2)))
thanks to constrained transport.

````@example srmhd_cylindrical_blast
@assert all(isfinite, rho) #hide
@assert all(isfinite, P) #hide
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_hyperbolic/srmhd_cylindrical_blast.jl).

```julia
using FiniteVolumeMethod
using StaticArrays

gamma = 4.0 / 3.0
eos = IdealGasEOS(gamma)
law = SRMHDEquations{2}(eos)

P_in = 1.0
P_out = 0.01
rho0 = 0.01
r_blast = 0.1
Bx0 = 0.5

function blast_ic(x, y)
    r = sqrt(x^2 + y^2)
    P = r < r_blast ? P_in : P_out
    return SVector(rho0, 0.0, 0.0, 0.0, P, Bx0, 0.0, 0.0)
end

Az_blast(x, y) = Bx0 * y

N = 100
mesh = StructuredMesh2D(-0.5, 0.5, -0.5, 0.5, N, N)

prob = HyperbolicProblem2D(
    law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(),
    TransmissiveBC(), TransmissiveBC(),
    blast_ic; final_time = 0.4, cfl = 0.25
)

coords, U, t_final, ct = solve_hyperbolic(prob; vector_potential = Az_blast)

divB_max = max_divB(ct, mesh)

using CairoMakie

nx, ny = N, N
xc = [coords[i, 1][1] for i in 1:nx]
yc = [coords[1, j][2] for j in 1:ny]

rho = [conserved_to_primitive(law, U[i, j])[1] for i in 1:nx, j in 1:ny]
P = [conserved_to_primitive(law, U[i, j])[5] for i in 1:nx, j in 1:ny]

fig = Figure(fontsize = 20, size = (1000, 500))
ax1 = Axis(
    fig[1, 1], xlabel = "x", ylabel = "y",
    title = "Density at t = $(round(t_final, digits = 3))",
    aspect = DataAspect()
)
hm1 = heatmap!(ax1, xc, yc, rho, colormap = :viridis)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(
    fig[1, 3], xlabel = "x", ylabel = "y",
    title = "Pressure",
    aspect = DataAspect()
)
hm2 = heatmap!(ax2, xc, yc, P, colormap = :inferno)
Colorbar(fig[1, 4], hm2)

resize_to_layout!(fig)
fig
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

