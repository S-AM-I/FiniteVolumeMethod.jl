# Section Overview

```@contents
Pages = ["overview.md"]
```

This section provides tutorials for the cell-centered finite volume solver for hyperbolic conservation laws on structured Cartesian meshes. Each tutorial is self-contained and demonstrates a different aspect of the solver. At the end of each tutorial we show the uncommented code; click the `Edit on GitHub` link at the top right of each page to see the source script.

For the mathematical and implementation details of the solver, see the [Mathematical Details](math.md) page. For the complete API reference, see the [Interface](interface.md) page.

# Sod Shock Tube

[This tutorial](tutorials/sod_shock_tube.md) introduces the hyperbolic solver by solving the classic Sod shock tube problem for the 1D Euler equations:

```math
\pdv{}{t}\begin{pmatrix}\rho \\ \rho v \\ E\end{pmatrix} + \pdv{}{x}\begin{pmatrix}\rho v \\ \rho v^2 + P \\ (E+P)v\end{pmatrix} = 0,
```

with left and right states $(\rho_L, v_L, P_L) = (1, 0, 1)$ and $(\rho_R, v_R, P_R) = (0.125, 0, 0.1)$, using $\gamma = 1.4$. We compare HLL and HLLC Riemann solvers, demonstrate MUSCL reconstruction with the minmod limiter, and plot the solution against the exact Riemann solution at $t = 0.2$.

# Sedov Blast Wave

[This tutorial](tutorials/sedov_blast_wave.md) solves the 2D Sedov blast wave problem on $[-1, 1]^2$. A point-like energy release drives a cylindrical shock wave whose radius evolves self-similarly as $r_s(t) \propto t^{2/5}$:

```math
\pdv{\vb U}{t} + \pdv{\vb F_x}{x} + \pdv{\vb F_y}{y} = 0, \qquad \vb U = (\rho, \rho v_x, \rho v_y, E)^{\mathrm T}.
```

We demonstrate `HyperbolicProblem2D`, transmissive boundary conditions, and the HLLC solver. The result is visualised as a 2D density heatmap and a radial profile.

# Brio-Wu MHD Shock Tube

[This tutorial](tutorials/brio_wu_shock_tube.md) solves the Brio-Wu MHD shock tube, the canonical test for MHD Riemann solvers. The 1D MHD equations with $\gamma = 2$ and constant $B_x = 0.75$ produce a rich wave structure including fast and slow shocks, a compound wave, and a contact discontinuity:

```math
\pdv{\vb U}{t} + \pdv{\vb F}{x} = 0, \qquad \vb U = (\rho, \rho v_x, \rho v_y, \rho v_z, E, B_x, B_y, B_z)^{\mathrm T},
```

with $(\rho_L, P_L, B_{y,L}) = (1, 1, 1)$ and $(\rho_R, P_R, B_{y,R}) = (0.125, 0.1, -1)$. We use the HLLD Riemann solver.

# Orszag-Tang Vortex

[This tutorial](tutorials/orszag_tang_vortex.md) simulates the Orszag-Tang vortex, an iconic 2D MHD turbulence problem on $[0, 1]^2$ with periodic boundary conditions and $\gamma = 5/3$:

```math
\rho = \gamma^2, \quad P = \gamma, \quad v_x = -\sin(2\pi y), \quad v_y = \sin(2\pi x),
```

```math
B_x = -\frac{\sin(2\pi y)}{\sqrt{4\pi}}, \quad B_y = \frac{\sin(4\pi x)}{\sqrt{4\pi}}.
```

We demonstrate constrained transport with `vector_potential` initialisation and verify $\max|\nabla\cdot\vb B| < 10^{-12}$.

# Taylor-Green Vortex Decay

[This tutorial](tutorials/taylor_green_vortex.md) solves the incompressible Taylor-Green vortex using the compressible Navier-Stokes equations in the low-Mach limit:

```math
v_x = -U_0\cos(kx)\sin(ky)\,\mathrm{e}^{-2\nu k^2 t}, \qquad v_y = U_0\sin(kx)\cos(ky)\,\mathrm{e}^{-2\nu k^2 t},
```

where $\nu = \mu/\rho_0$ is the kinematic viscosity. We demonstrate `NavierStokesEquations`, periodic boundary conditions, and convergence of the numerical solution to the analytical viscous decay.

# Field Loop Advection

[This tutorial](tutorials/field_loop_advection.md) advects a magnetic field loop across a periodic 2D domain, testing the constrained transport algorithm. A weak circular magnetic field is initialised using the vector potential

```math
A_z(x, y) = \begin{cases} A_0(R_0 - r) & r < R_0, \\ 0 & r \geq R_0, \end{cases} \qquad r = \sqrt{(x-0.5)^2 + (y-0.5)^2},
```

which guarantees $\nabla\cdot\vb B = 0$ by construction. The tutorial demonstrates `initialize_ct_from_potential!` and verifies that $\nabla\cdot\vb B$ remains at machine precision.

# Kelvin-Helmholtz Instability

[This tutorial](tutorials/kelvin_helmholtz_instability.md) simulates the Kelvin-Helmholtz instability, a shear-driven hydrodynamic instability in 2D Euler:

```math
v_x = \begin{cases} v_{\mathrm{shear}} & |y - 0.5| < 0.25, \\ -v_{\mathrm{shear}} & \text{otherwise}, \end{cases}
```

with a small sinusoidal $v_y$ perturbation to seed the instability. We use periodic boundary conditions and demonstrate the effect of resolution on the development of the roll-up vortices.

# Balsara SRMHD Shock Tube

[This tutorial](tutorials/balsara_srmhd_shock_tube.md) solves the Balsara 1 shock tube for special relativistic MHD, the relativistic analogue of the Brio-Wu test:

```math
(\rho_L, P_L, B_{y,L}) = (1, 1, 1), \qquad (\rho_R, P_R, B_{y,R}) = (0.125, 0.1, -1), \qquad B_x = 0.5,
```

with $\gamma = 5/3$. We demonstrate `SRMHDEquations`, the iterative conserved-to-primitive conversion, and the role of the Lorentz factor.

# WENO Convergence Study

[This tutorial](tutorials/weno_convergence.md) compares MUSCL, WENO-3, and characteristic WENO reconstruction schemes on a smooth density wave:

```math
\rho(x, 0) = 1 + 0.01\sin(2\pi x), \qquad v = 1, \qquad P = 1,
```

with periodic boundary conditions. We measure $L^1$ errors at multiple resolutions and compare convergence rates. We also apply WENO reconstruction to the Sod shock tube to demonstrate shock-capturing.

# Couette Flow

[This tutorial](tutorials/couette_flow.md) simulates steady Couette flow between two parallel plates using the Navier-Stokes solver. The bottom wall is stationary (`NoSlipBC`) and the top wall moves at velocity $U_w$:

```math
v_x(y) = U_w\frac{y}{H},
```

which is the exact linear profile for viscous flow between parallel plates. We demonstrate `NoSlipBC` and `DirichletHyperbolicBC` for viscous wall boundary conditions.

# IMEX Stiff Relaxation

[This tutorial](tutorials/imex_stiff_relaxation.md) demonstrates implicit-explicit time integration for a 1D Euler system with stiff radiative cooling:

```math
\pdv{\vb U}{t} + \pdv{\vb F}{x} = \vb S_{\mathrm{stiff}}(\vb U),
```

where the source term drives the gas toward an equilibrium temperature on a time scale much shorter than the CFL time step. We use `CoolingSource`, `solve_hyperbolic_imex`, and the `IMEX_SSP3_433` scheme to demonstrate stable time integration without resolving the stiff time scale.

# AMR Sedov Blast

[This tutorial](tutorials/amr_sedov_blast.md) solves the Sedov blast wave with block-structured adaptive mesh refinement (AMR). We demonstrate `AMRGrid`, `GradientRefinement`, `solve_amr`, and the Berger-Colella flux correction that maintains conservation at coarse-fine boundaries. The AMR automatically refines around the shock front while keeping the smooth interior at coarse resolution.
