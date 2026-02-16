```@meta
CurrentModule = FiniteVolumeMethod
```

# Introduction

This is the documentation for FiniteVolumeMethod.jl. [Click here to go back to the GitHub repository](https://github.com/SciML/FiniteVolumeMethod.jl).

FiniteVolumeMethod.jl is a Julia package for solving partial differential equations (PDEs) using the finite volume method. The package provides two complementary solvers for different classes of PDEs.

## Parabolic and Elliptic PDEs (Cell-Vertex Solver)

The cell-vertex solver handles PDEs of the form

```math
\pdv{u(\vb x, t)}{t} + \div \vb q(\vb x, t, u) = S(\vb x, t, u), \quad (x, y)^{\mkern-1.5mu\mathsf{T}} \in \Omega \subset \mathbb R^2,\,t>0,
```

using the finite volume method on triangular meshes, with additional support for steady-state problems and for systems of PDEs of the above form. We support Neumann, Dirichlet, and boundary conditions on $\mathrm du/\mathrm dt$, as well as internal conditions and custom constraints. We also provide an interface for solving special cases of the above PDE, namely reaction-diffusion equations

```math
\pdv{u(\vb x, t)}{t} = \div\left[D(\vb x, t, u)\grad u(\vb x, t)\right] + S(\vb x, t, u).
```

The [tutorials](tutorials/overview.md) in the sidebar demonstrate the many possibilities of this solver. In addition to these two generic forms, we also provide support for specific problems that can be solved in a more efficient manner, namely:

 1. `DiffusionEquation`s: $\partial_tu = \div[D(\vb x)\grad u]$.
 2. `MeanExitTimeProblem`s: $\div[D(\vb x)\grad T(\vb x)] = -1$.
 3. `LinearReactionDiffusionEquation`s: $\partial_tu = \div[D(\vb x)\grad u] + f(\vb x)u$.
 4. `PoissonsEquation`: $\div[D(\vb x)\grad u] = f(\vb x)$.
 5. `LaplacesEquation`: $\div[D(\vb x)\grad u] = 0$.

See the [Solvers for Specific Problems, and Writing Your Own](wyos/overview.md) section for more information on these templates.

## Hyperbolic Conservation Laws (Cell-Centered Solver)

The cell-centered solver targets hyperbolic conservation laws of the form

```math
\pdv{\vb U}{t} + \nabla \cdot \vb F(\vb U) = \vb S(\vb U)
```

on structured Cartesian meshes in 1D, 2D, and 3D. Supported physics includes:

- **Euler equations** for compressible gas dynamics
- **Ideal MHD** with divergence-free constrained transport
- **Compressible Navier-Stokes** with viscous stress and heat conduction
- **Special relativistic MHD** (SRMHD) with iterative conserved-to-primitive conversion
- **General relativistic MHD** (GRMHD) in the Valencia formulation with Schwarzschild and Kerr metrics

Numerical features include MUSCL and WENO (3rd/5th order) reconstruction, Lax-Friedrichs/HLL/HLLC/HLLD Riemann solvers, SSP-RK3 and IMEX time integration, and block-structured adaptive mesh refinement (AMR) with Berger-Colella flux correction.

See the [hyperbolic tutorials](hyperbolic/overview.md) for examples and the [hyperbolic interface](hyperbolic/interface.md) for the full API reference.
