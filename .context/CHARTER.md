# CHARTER — FiniteVolumeMethod.jl

## Purpose
FiniteVolumeMethod.jl is a comprehensive Julia package for solving 2D/3D partial differential equations using the finite volume method. It provides two solver families: a triangular (vertex-centered) FVM for parabolic/elliptic PDEs on unstructured meshes via DelaunayTriangulation.jl, and a cell-centered FVM for hyperbolic conservation laws on structured/unstructured grids. The package integrates with the SciML ecosystem for time integration and linear algebra.

## Scientific Foundation
**Parabolic/Elliptic (Triangular FVM):**
- Generic form: ∂u/∂t + ∇·q(x, t, u) = S(x, t, u)
- Control volumes around mesh vertices; flux evaluation at CV edges via linear shape functions
- Coordinate systems: Cartesian, Cylindrical (axisymmetric r,z), Spherical (r,θ) — Jacobian weighting on volumes and fluxes
- Included templates: AdvectionDiffusionEquation, AnisotropicDiffusionEquation
- Reference implementations (wyos, not module-included): DiffusionEquation, PoissonsEquation, LaplacesEquation, LinearReactionDiffusionEquation, MeanExitTimeProblem
- BCs: Dirichlet, Neumann, Robin, Dudt, Constrained, plus nonlinear, periodic, and coupled variants

**Hyperbolic (Cell-Centered FVM):**
- Generic form: ∂U/∂t + ∇·F(U) = S(U)
- Conservation laws: Euler (1D/2D/3D), Ideal MHD, Navier-Stokes, Shallow Water, SR Hydro, SRMHD, GRMHD, Two-Fluid, Hall MHD, Resistive MHD
- Riemann solvers: Lax-Friedrichs, HLL, HLLC, HLLD
- Reconstruction: MUSCL (with 7 limiters), PPM (Colella & Woodward), WENO3, WENO5, Characteristic WENO
- Constrained transport for ∇·B = 0 (face-centered B, edge-centered EMF)
- Positivity limiter (Zhang & Shu 2010)
- Block-structured AMR with Berger-Colella flux correction and multi-rate subcycling
- IMEX time integration: SSP3_433, ARS222, Midpoint
- Spacetime metrics: Minkowski, Schwarzschild, Kerr (for GRMHD)
- Multi-physics coupling via operator splitting (Lie-Trotter, Strang)

**Gradient Reconstruction:** Green-Gauss, Least-Squares

## Technical Constraints
- Julia 1.0+. Multiple dispatch, not OOP. All arrays 1-indexed.
- Triangular FVM depends on DelaunayTriangulation.jl for mesh generation.
- SciML integration: SciMLBase for ODEProblem/LinearProblem, CommonSolve for `solve()` interface.
- StaticArrays (SVector) for conserved variable tuples in the hyperbolic solver.
- PreallocationTools (DiffCache) for automatic differentiation compatibility.
- Threading via ChunkSplitters for parallel 2D hyperbolic solves.
- Formatting: Runic.jl (enforced by CI).
- Current version: 1.2.0.

## Key References
- Colella & Woodward (1984) — PPM reconstruction
- Zhang & Shu (2010) — Positivity-preserving limiters
- Berger & Colella (1989) — AMR with flux correction
- Valencia formulation for GRMHD
