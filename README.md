# FiniteVolumeMethod

[![DOI](https://zenodo.org/badge/561533716.svg)](https://zenodo.org/badge/latestdoi/561533716)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cx-xd.github.io/FiniteVolumeMethod.jl/dev)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cx-xd.github.io/FiniteVolumeMethod.jl/stable)
[![Coverage](https://codecov.io/gh/cx-xd/FiniteVolumeMethod.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cx-xd/FiniteVolumeMethod.jl)

FiniteVolumeMethod.jl is a Julia package for solving partial differential equations (PDEs) with two main solver families:

- a cell-vertex parabolic/elliptic solver on unstructured triangular meshes
- a cell-centered hyperbolic solver on structured 1D/2D/3D meshes

The repository also contains more advanced capabilities such as constrained-transport MHD, relativistic solvers, AMR, coupling infrastructure, and research-support tooling. These capabilities do **not** all share the same scientific maturity.

The authoritative public contract for what is publication-grade, provisional, or experimental is the documentation's capability matrix and verification material. Only features marked `stable` in that contract should be used for publication-grade scientific claims.

The parabolic solver handles PDEs of the form

$$
\dfrac{\partial u(\boldsymbol x, t)}{\partial t} + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}(\boldsymbol x, t, u) = S(\boldsymbol x, t, u), \quad (x, y)^{\mkern-1.5mu\mathsf{T}} \in \Omega \subset \mathbb R^2,t>0,
$$

with support for steady-state problems and for systems of PDEs of the above form. In addition to this generic form above, we also provide support for specific problems that can be solved in a more efficient manner, namely:

 1. `DiffusionEquation`s: $\partial_tu = \boldsymbol\nabla\boldsymbol\cdot[D(\boldsymbol x)\boldsymbol\nabla u]$.
 2. `MeanExitTimeProblem`s: $\boldsymbol\nabla\boldsymbol\cdot[D(\boldsymbol x)\boldsymbol\nabla T(\boldsymbol x)] = -1$.
 3. `LinearReactionDiffusionEquation`s: $\partial_tu = \boldsymbol\nabla\boldsymbol\cdot[D(\boldsymbol x)\boldsymbol\nabla u] + f(\boldsymbol x)u$.
 4. `PoissonsEquation`: $\boldsymbol\nabla\boldsymbol\cdot[D(\boldsymbol x)\boldsymbol\nabla u] = f(\boldsymbol x)$.
 5. `LaplacesEquation`: $\boldsymbol\nabla\boldsymbol\cdot[D(\boldsymbol x)\boldsymbol\nabla u] = 0$.

See the documentation for the capability matrix, verification evidence, and detailed interface notes.

## Research Support Policy

This repository treats only the current Julia stable release and the current Julia LTS release as release-supported targets. Pre-release Julia versions may still be tested opportunistically, but they are not part of the scientific support contract.

If this package doesn't suit what you need, you may like to review some of the other PDE packages shown [here](https://github.com/JuliaPDE/SurveyofPDEPackages).

As a very quick demonstration, here is how we could solve a diffusion equation with Dirichlet boundary conditions on a square domain using the standard `FVMProblem` formulation; please see the docs for more information.

```julia
using FiniteVolumeMethod, DelaunayTriangulation, CairoMakie, OrdinaryDiffEq
a, b, c, d = 0.0, 2.0, 0.0, 2.0
nx, ny = 50, 50
tri = triangulate_rectangle(a, b, c, d, nx, ny, single_boundary = true)
mesh = FVMGeometry(tri)
bc = (x, y, t, u, p) -> zero(u)
BCs = BoundaryConditions(mesh, bc, Dirichlet)
f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0
initial_condition = [f(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
D = (x, y, t, u, p) -> 1 / 9
final_time = 0.5
prob = FVMProblem(mesh, BCs; diffusion_function = D, initial_condition, final_time)
sol = solve(prob, Tsit5(), saveat = 0.001)
u = Observable(sol.u[1])
fig, ax, sc = tricontourf(tri, u, levels = 0:5:50, colormap = :matter)
tightlimits!(ax)
record(fig, "anim.gif", eachindex(sol)) do i
    u[] = sol.u[i]
end
```

![Animation of a solution](https://github.com/SciML/FiniteVolumeMethod.jl/blob/main/anim.gif)

We could have equivalently used the `DiffusionEquation` template, so that `prob` could have also been defined by

```julia
prob = DiffusionEquation(mesh, BCs; diffusion_function = D, initial_condition, final_time)
```

and be solved much more efficiently. See the documentation for more information.
