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

## v2 Transition

The repository now follows an explicit research-grade `v2` contract.

- Start with the [capability matrix](docs/src/capability_matrix.md) and the [v2 migration guide](docs/src/v2_migration.md) before treating a feature as publication-grade.
- Review the proposed [v2.0.0-rc1 changelog](CHANGELOG.md) for the current release-candidate contract and migration summary.
- CPU `Float64` runs remain the publication baseline. CUDA parity is currently audited only for the supported 2D hyperbolic extension path, so other GPU usage should be treated as experimental.
- GitHub Actions CI is enabled with four lanes (environment-integrity, unit-interop, scientific-smoke, docs). For local iteration, use `make ci-fast`, `make ci-smoke`, `make ci-full-evidence`, `make ci-performance`, or `make ci-release-audit`.

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

![Animation of a solution](https://github.com/cx-xd/FiniteVolumeMethod.jl/blob/main/anim.gif)

We could have equivalently used the `DiffusionEquation` template, so that `prob` could have also been defined by

```julia
prob = DiffusionEquation(mesh, BCs; diffusion_function = D, initial_condition, final_time)
```

and be solved much more efficiently. See the documentation for more information.

## Hyperbolic Solver

The cell-centered hyperbolic solver handles conservation laws on structured 1D/2D/3D meshes using a method-of-lines approach with explicit time stepping. Here is a 1D Sod shock tube solved with the HLLC Riemann solver and MUSCL reconstruction:

```julia
using FiniteVolumeMethod, OrdinaryDiffEq, StaticArrays, CairoMakie

law = EulerEquations{1}(IdealGasEOS(1.4))
mesh = StructuredMesh1D(0.0, 1.0, 200)

prob = HyperbolicProblem(
    law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
    TransmissiveBC(), TransmissiveBC(),
    x -> x < 0.5 ? SVector(1.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.1);
    final_time = 0.2, cfl = 0.5
)

dt0 = compute_initial_dt(sciml_problem(prob).p, sciml_problem(prob).u0)
sol = solve(prob, SSPRK33(); adaptive = false, dt = dt0)

# Access fields by name via SymbolicIndexingInterface
rho = sol[:rho]       # density at each cell, for each saved time step

# Animate the density profile
N = nvariables(law)
xs = mesh.cell_centers
rho_obs = Observable([sol.u[1][(i - 1) * N + 1] for i in 1:200])
fig, ax, _ = lines(xs, rho_obs; axis = (; xlabel = "x", ylabel = "density", title = "Sod Shock Tube"))
ylims!(ax, 0.0, 1.15)
record(fig, "sod_shock.gif", eachindex(sol); framerate = 20) do idx
    rho_obs[] = [sol.u[idx][(i - 1) * N + 1] for i in 1:200]
end
```

![Sod shock tube animation](https://github.com/cx-xd/FiniteVolumeMethod.jl/blob/main/sod_shock.gif)

### Riemann Solvers

| Solver | Waves | Use case |
|--------|-------|----------|
| `LaxFriedrichsSolver()` | 1 | Most diffusive, always stable |
| `HLLSolver()` | 2 | Robust baseline |
| `HLLCSolver()` | 3 | Default for Euler (resolves contacts) |
| `HLLDSolver()` | 5 | MHD (resolves Alfven waves) |

### Reconstruction Schemes

| Scheme | Order | Notes |
|--------|-------|-------|
| `NoReconstruction()` | 1st | Piecewise constant |
| `CellCenteredMUSCL(limiter)` | 2nd | Default; limiters: `MinmodLimiter`, `SuperbeeLimiter`, `VanLeerLimiter`, `KorenLimiter`, `OspreLimiter`, `VenkatakrishnanLimiter` |
| `PPMReconstruction()` | 3rd | Piecewise parabolic (sharp contacts) |
| `WENO3()` | 3rd | Weighted ENO |
| `WENO5()` | 5th | High-order for smooth regions |

Characteristic-variable projection is available via `CharacteristicWENO(WENO3())` or `CharacteristicWENO(WENO5())`.

## MHD with Constrained Transport

The MHD solver preserves $\nabla \cdot \mathbf{B} = 0$ to machine precision using constrained transport on face-centered magnetic fields. Here is the Orszag-Tang MHD vortex, a standard test for MHD turbulence:

```julia
using FiniteVolumeMethod, OrdinaryDiffEq, StaticArrays, CairoMakie

gamma = 5.0 / 3.0
law = IdealMHDEquations{2}(IdealGasEOS(gamma))
mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 64, 64)

function orszag_tang_ic(x, y)
    rho = gamma^2;  P = gamma
    vx = -sin(2pi * y);  vy = sin(2pi * x);  vz = 0.0
    Bx = -sin(2pi * y) / sqrt(4pi);  By = sin(4pi * x) / sqrt(4pi);  Bz = 0.0
    E = P / (gamma - 1) + 0.5 * rho * (vx^2 + vy^2 + vz^2) + 0.5 * (Bx^2 + By^2 + Bz^2)
    return SVector(rho, rho * vx, rho * vy, rho * vz, E, Bx, By, Bz)
end

# Vector potential ensures ∇·B = 0 at initialization
Az(x, y) = cos(2pi * y) / (2pi * sqrt(4pi)) + cos(4pi * x) / (4pi * sqrt(4pi))

prob = HyperbolicProblem2D(
    law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
    orszag_tang_ic; final_time = 0.5, cfl = 0.4
)

ode = ODEProblem(prob; vector_potential = Az)
limiter = mhd_stage_limiter(ode.p)
dt0 = compute_initial_dt(ode.p, ode.u0)
sol = solve(ode, SSPRK33(; stage_limiter! = limiter); adaptive = false, dt = dt0, saveat = 0.01)
```

![Orszag-Tang vortex animation](https://github.com/cx-xd/FiniteVolumeMethod.jl/blob/main/orszag_tang.gif)

## Conservation Laws

**Gas dynamics:** `EulerEquations{1,2,3}`, `NavierStokesEquations{1,2}`, `ShallowWaterEquations{1,2}`

**Magnetohydrodynamics:** `IdealMHDEquations{2,3}`, `ResistiveMHDEquations`, `HallMHDEquations`

**Relativistic:** `SRHydroEquations{1,2}`, `SRMHDEquations`, `GRMHDEquations`

**Multi-species / multi-fluid:** `ReactiveEulerEquations{Dim,NSpecies}`, `TwoFluidEquations{1,2}`

New conservation laws can be added by subtyping `AbstractConservationLaw{Dim}` and implementing `nvariables`, `physical_flux`, `max_wave_speed`, `conserved_to_primitive`, and `primitive_to_conserved`.

## SciML Ecosystem Integration

All solver families produce standard `SciMLBase.ODEProblem` objects, compatible with any ODE solver from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl):

- **CommonSolve**: `solve(prob, alg; kwargs...)` and `init(prob, alg; kwargs...)` work directly on FVM problem types
- **`remake`**: Full support for parameter studies — `remake(ode_prob; cfl = 0.3, final_time = 1.0)`
- **SymbolicIndexingInterface**: Access solution fields by name — `sol[:rho]`, `sol[:E]`, `sol[:Bx]`
- **SciMLStructures**: Parameter partitioning for optimization and sensitivity analysis
- **RecipesBase**: Plot recipes for 1D line plots and 2D heatmaps (via `FVMRecipesExt`)
