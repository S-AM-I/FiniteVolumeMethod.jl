# Mathematical and Implementation Details: Hyperbolic Solver

```@contents
Pages = ["math.md"]
```

This page describes the mathematical framework and implementation details of the cell-centered finite volume solver for hyperbolic conservation laws. For the parabolic (cell-vertex) solver, see the [main mathematical details page](../math.md).

## Hyperbolic Conservation Laws

The solver targets systems of hyperbolic conservation laws in $d$ spatial dimensions:

```math
\pdv{\vb U}{t} + \sum_{k=1}^{d} \pdv{\vb F_k(\vb U)}{x_k} = \vb S(\vb U),
```

where $\vb U \in \mathbb R^m$ is the vector of conserved variables, $\vb F_k(\vb U)$ is the physical flux in the $k$th coordinate direction, and $\vb S(\vb U)$ is a source term. In one dimension this reduces to

```math
\pdv{\vb U}{t} + \pdv{\vb F(\vb U)}{x} = \vb S(\vb U).
```

### Euler Equations

The compressible Euler equations describe inviscid gas dynamics. In 1D ($m=3$):

```math
\vb U = \begin{pmatrix}\rho \\ \rho v \\ E\end{pmatrix},\quad
\vb F = \begin{pmatrix}\rho v \\ \rho v^2 + P \\ (E+P)v\end{pmatrix},
```

where $\rho$ is density, $v$ is velocity, $E = \rho\varepsilon + \tfrac{1}{2}\rho v^2$ is total energy, $P$ is pressure, and $\varepsilon$ is specific internal energy. The system is closed by an equation of state $P = P(\rho, \varepsilon)$.

In 2D ($m=4$) and 3D ($m=5$), the conserved vector includes momentum components $\rho v_x, \rho v_y$ (and $\rho v_z$), with corresponding fluxes in each direction.

### Ideal Gas Equation of State

For an ideal gas with adiabatic index $\gamma$:

```math
P = (\gamma - 1)\rho\varepsilon, \qquad c_s = \sqrt{\frac{\gamma P}{\rho}},
```

where $c_s$ is the sound speed. The stiffened gas equation of state generalises this to

```math
P = (\gamma - 1)\rho\varepsilon - \gamma P_\infty,
```

which reduces to the ideal gas when $P_\infty = 0$.

### Ideal MHD Equations

The ideal magnetohydrodynamics (MHD) equations extend the Euler equations with a magnetic field $\vb B$. The system has $m=8$ variables in all dimensions:

```math
\vb U = \begin{pmatrix}\rho \\ \rho\vb v \\ E \\ \vb B\end{pmatrix},\qquad
\vb F_k = \begin{pmatrix}\rho v_k \\ \rho\vb v v_k + P_{\mathrm{tot}}\vb e_k - B_k\vb B \\ (E + P_{\mathrm{tot}})v_k - (\vb v \cdot \vb B)B_k \\ \vb B v_k - B_k \vb v\end{pmatrix},
```

where $P_{\mathrm{tot}} = P + \tfrac{1}{2}|\vb B|^2$ is the total (gas + magnetic) pressure and the total energy is

```math
E = \frac{P}{\gamma - 1} + \frac{1}{2}\rho|\vb v|^2 + \frac{1}{2}|\vb B|^2.
```

The system must satisfy the divergence-free constraint $\nabla \cdot \vb B = 0$, which is enforced through constrained transport (see below).

### Navier-Stokes Equations

The compressible Navier-Stokes equations add viscous stress and heat conduction to the Euler equations:

```math
\pdv{\vb U}{t} + \nabla\cdot\vb F^{\mathrm{inv}}(\vb U) = \nabla\cdot\vb F^{\mathrm{visc}}(\vb U, \nabla\vb U),
```

where $\vb F^{\mathrm{inv}}$ is the inviscid (Euler) flux and $\vb F^{\mathrm{visc}}$ contains the viscous stress tensor $\tau_{ij}$ and heat flux $q_i$:

```math
\tau_{ij} = \mu\left(\pdv{v_i}{x_j} + \pdv{v_j}{x_i} - \frac{2}{3}\delta_{ij}\nabla\cdot\vb v\right), \qquad q_i = -\kappa\pdv{T}{x_i},
```

with dynamic viscosity $\mu$ and thermal conductivity $\kappa = \mu\gamma/[\mathrm{Pr}(\gamma-1)]$, where $\mathrm{Pr}$ is the Prandtl number.

### Special Relativistic MHD

The SRMHD equations use $m=8$ conserved variables:

```math
\vb U = \begin{pmatrix} D \\ \vb S \\ \tau \\ \vb B \end{pmatrix} = \begin{pmatrix} \rho W \\ (\rho h W^2 + |\vb B|^2)\vb v - (\vb v\cdot\vb B)\vb B \\ \rho h W^2 + |\vb B|^2 - P_{\mathrm{tot}} - D \\ \vb B \end{pmatrix},
```

where $W = 1/\sqrt{1-|\vb v|^2}$ is the Lorentz factor, $h = 1 + \varepsilon + P/\rho$ is the specific enthalpy, and $P_{\mathrm{tot}} = P + \tfrac{1}{2}(|\vb B|^2/W^2 + (\vb v\cdot\vb B)^2)$. The conserved-to-primitive conversion is performed iteratively using the Palenzuela method, solving

```math
f(\xi) = \xi + |\vb B|^2 - P_{\mathrm{tot}} - (\tau + D) = 0, \qquad \xi = \rho h W^2.
```

### General Relativistic MHD

The GRMHD equations use the Valencia formulation, evolving densitised conserved variables $\sqrt{\gamma}\,\vb U$ on a curved spacetime with lapse $\alpha$, shift $\beta^i$, and spatial metric $\gamma_{ij}$. Geometric source terms arise from spacetime curvature:

```math
\pdv{(\sqrt{\gamma}\,\vb U)}{t} + \pdv{(\sqrt{\gamma}\,\vb F^i)}{x^i} = \sqrt{\gamma}\,\vb S_{\mathrm{geom}}.
```

## Cell-Centered FVM Discretisation

### Structured Mesh

The domain is discretised on a uniform Cartesian mesh. In 1D, the domain $[x_L, x_R]$ is divided into $N$ cells of width $\Delta x = (x_R - x_L)/N$. Cell $i$ has centre $x_i = x_L + (i - \tfrac{1}{2})\Delta x$. In 2D and 3D the mesh is the tensor product of 1D meshes.

### Integral Form

Integrating the conservation law over cell $i$ and applying the divergence theorem:

```math
\dv{\bar{\vb U}_i}{t} = -\frac{1}{\Delta x}\left[\hat{\vb F}_{i+1/2} - \hat{\vb F}_{i-1/2}\right] + \bar{\vb S}_i,
```

where $\bar{\vb U}_i$ is the cell average of $\vb U$ and $\hat{\vb F}_{i+1/2}$ is the numerical flux at the interface between cells $i$ and $i+1$. In multiple dimensions, the update includes flux differences in each coordinate direction (dimension-by-dimension splitting).

### Ghost Cells

Boundary conditions are imposed through ghost cells that pad the computational domain. By default, two layers of ghost cells are placed on each side of the domain (more for higher-order schemes such as WENO-5, which requires three). Interior cell index $i$ maps to storage index $i + n_g$ in the padded array, where $n_g$ is the number of ghost layers.

The ghost-cell values are filled before each flux evaluation according to the selected boundary condition type:

| BC Type | Ghost-Cell Strategy |
|:--------|:-------------------|
| `TransmissiveBC` | Copy nearest interior cell (zero-gradient extrapolation) |
| `ReflectiveBC` | Mirror interior cells, negate wall-normal velocity |
| `NoSlipBC` | Mirror interior cells, negate all velocity components |
| `DirichletHyperbolicBC` | Set ghost cells to a prescribed state |
| `InflowBC` | Set ghost cells to a prescribed inflow state |
| `PeriodicHyperbolicBC` | Copy from opposite side of domain |

## Approximate Riemann Solvers

The numerical flux $\hat{\vb F}_{i+1/2}$ is computed by solving an approximate Riemann problem at each cell interface with left state $\vb U_L$ and right state $\vb U_R$.

### Lax-Friedrichs (Rusanov)

The simplest solver, using a single wave speed $\lambda_{\max} = \max(|v_L| + c_L,\, |v_R| + c_R)$:

```math
\hat{\vb F}^{\mathrm{LF}} = \frac{1}{2}\left[\vb F(\vb U_L) + \vb F(\vb U_R) - \lambda_{\max}(\vb U_R - \vb U_L)\right].
```

### HLL (Harten-Lax-van Leer)

Uses two wave speeds $S_L \leq 0$ and $S_R \geq 0$ bounding the Riemann fan:

```math
\hat{\vb F}^{\mathrm{HLL}} = \begin{cases}
\vb F_L & \text{if } S_L \geq 0, \\[4pt]
\dfrac{S_R\vb F_L - S_L\vb F_R + S_LS_R(\vb U_R - \vb U_L)}{S_R - S_L} & \text{if } S_L < 0 < S_R, \\[4pt]
\vb F_R & \text{if } S_R \leq 0.
\end{cases}
```

### HLLC

The HLLC solver introduces a third wave speed $S_*$ representing the contact discontinuity, giving improved resolution of contact waves and shear layers:

```math
S_* = \frac{P_R - P_L + \rho_L v_L(S_L - v_L) - \rho_R v_R(S_R - v_R)}{\rho_L(S_L - v_L) - \rho_R(S_R - v_R)}.
```

The intermediate states $\vb U_L^*$ and $\vb U_R^*$ satisfy the Rankine-Hugoniot conditions across $S_L$ and $S_R$.

### HLLD

For ideal MHD, the HLLD solver (Miyoshi & Kusano 2005) resolves five waves: two fast magnetosonic waves ($S_L$, $S_R$), two rotational/Alfven discontinuities ($S_L^*$, $S_R^*$), and the contact discontinuity ($S_M$). This gives superior resolution of the Alfven and contact waves compared to HLL.

## Reconstruction and Limiters

### First Order (No Reconstruction)

The simplest approach uses piecewise-constant data: $\vb U_L = \bar{\vb U}_i$ and $\vb U_R = \bar{\vb U}_{i+1}$ at interface $i+1/2$. This is first-order accurate and very diffusive.

### MUSCL Reconstruction

The Monotone Upstream-centred Scheme for Conservation Laws (MUSCL) achieves second-order accuracy by linearly reconstructing states at cell interfaces:

```math
\vb U_{i+1/2}^L = \bar{\vb U}_i + \frac{1}{2}\phi(r_i)\,(\bar{\vb U}_{i+1} - \bar{\vb U}_i), \qquad r_i = \frac{\bar{\vb U}_i - \bar{\vb U}_{i-1}}{\bar{\vb U}_{i+1} - \bar{\vb U}_i},
```

where $\phi(r)$ is a slope limiter function. The following limiters are available:

| Limiter | $\phi(r)$ |
|:--------|:----------|
| Minmod | $\max(0, \min(1, r))$ |
| Van Leer | $\dfrac{r + |r|}{1 + |r|}$ |
| MC (Monotonised Central) | $\max\!\bigl(0,\, \min(2r,\, \tfrac{1+r}{2},\, 2)\bigr)$ |
| Superbee | $\max\!\bigl(0,\, \min(2r, 1),\, \min(r, 2)\bigr)$ |

### WENO Reconstruction

Weighted Essentially Non-Oscillatory (WENO) schemes use nonlinear combinations of candidate stencils, weighted by smoothness to avoid oscillations near discontinuities.

**WENO-3** uses two candidate stencils, each providing a linear reconstruction. At interface $i+1/2$, the left-biased reconstruction is:

```math
\vb U_{i+1/2}^{(0)} = \tfrac{1}{2}\bar{\vb U}_{i-1} + \tfrac{1}{2}\bar{\vb U}_i, \qquad
\vb U_{i+1/2}^{(1)} = -\tfrac{1}{2}\bar{\vb U}_{i} + \tfrac{3}{2}\bar{\vb U}_{i+1},
```

with smoothness indicators $\beta_k$ and nonlinear weights $\omega_k$ satisfying $\sum_k \omega_k = 1$. Requires 2 ghost cells per side.

**WENO-5** (WENO-Z variant) uses three candidate stencils (6 points total) to achieve fifth-order accuracy in smooth regions. Requires 3 ghost cells per side.

### Characteristic WENO

The `CharacteristicWENO` wrapper projects the conserved variables into characteristic space before applying WENO reconstruction, then projects back. This avoids spurious oscillations that can arise when reconstructing conserved or primitive variables directly, especially for systems with waves of very different speeds:

```math
\vb W = \vb L \cdot \vb U, \qquad \text{reconstruct } \vb W, \qquad \vb U^{\mathrm{recon}} = \vb R \cdot \vb W^{\mathrm{recon}},
```

where $\vb L$ and $\vb R$ are the left and right eigenvector matrices of the flux Jacobian $\partial\vb F/\partial\vb U$.

## Time Integration

### CFL Condition

The Courant-Friedrichs-Lewy (CFL) condition restricts the time step for stability:

```math
\Delta t \leq C_{\mathrm{CFL}} \cdot \frac{\Delta x}{\max_i \lambda_{\max,i}},
```

where $\lambda_{\max,i}$ is the maximum wave speed in cell $i$ and $C_{\mathrm{CFL}} \in (0, 1]$. In multiple dimensions:

```math
\Delta t \leq C_{\mathrm{CFL}} \cdot \left(\frac{\max_i\lambda_{\max,i}^x}{\Delta x} + \frac{\max_i\lambda_{\max,i}^y}{\Delta y} + \cdots\right)^{-1}.
```

For viscous flows (Navier-Stokes), an additional viscous CFL constraint applies:

```math
\Delta t_{\mathrm{visc}} \leq \frac{1}{2}\frac{\rho_{\min}}{\mu}\left(\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}\right)^{-1}.
```

### Forward Euler

The simplest explicit time integrator:

```math
\vb U^{n+1} = \vb U^n + \Delta t\,\mathcal L(\vb U^n),
```

where $\mathcal L$ is the spatial discretisation operator (right-hand side).

### SSP-RK3 (Strong Stability Preserving Runge-Kutta)

The default time integrator is the third-order SSP-RK3 scheme of Shu and Osher:

```math
\begin{aligned}
\vb U^{(1)} &= \vb U^n + \Delta t\,\mathcal L(\vb U^n), \\
\vb U^{(2)} &= \tfrac{3}{4}\vb U^n + \tfrac{1}{4}\vb U^{(1)} + \tfrac{1}{4}\Delta t\,\mathcal L(\vb U^{(1)}), \\
\vb U^{n+1} &= \tfrac{1}{3}\vb U^n + \tfrac{2}{3}\vb U^{(2)} + \tfrac{2}{3}\Delta t\,\mathcal L(\vb U^{(2)}).
\end{aligned}
```

This scheme preserves the TVD (total variation diminishing) property of the forward Euler method and is the recommended choice for hyperbolic problems.

### IMEX Runge-Kutta

For problems with stiff source terms (e.g., radiative cooling, resistive MHD), an implicit-explicit (IMEX) Runge-Kutta scheme treats the hyperbolic fluxes explicitly and the stiff sources implicitly. The general form with $s$ stages is:

```math
\begin{aligned}
\vb U^{(i)} &= \vb U^n + \Delta t\sum_{j=1}^{i-1}\tilde a_{ij}\,\mathcal L(\vb U^{(j)}) + \Delta t\sum_{j=1}^{i}a_{ij}\,\vb S(\vb U^{(j)}), \\
\vb U^{n+1} &= \vb U^n + \Delta t\sum_{i=1}^{s}\tilde b_i\,\mathcal L(\vb U^{(i)}) + \Delta t\sum_{i=1}^{s}b_i\,\vb S(\vb U^{(i)}),
\end{aligned}
```

where $\tilde a_{ij}$ and $\tilde b_i$ are the explicit Butcher coefficients and $a_{ij}$ and $b_i$ are the implicit (diagonally implicit) coefficients. The implicit stage solve is performed with Newton's method.

Available IMEX schemes:
- **`IMEX_SSP3_433`**: 4-stage, 3rd-order SSP scheme (Pareschi & Russo 2005).
- **`IMEX_ARS222`**: 3-stage, 2nd-order L-stable scheme (Ascher, Ruuth & Spiteri 1997).
- **`IMEX_Midpoint`**: 2-stage, 2nd-order midpoint scheme.

## Constrained Transport

### The Divergence-Free Constraint

Maxwell's equations require $\nabla\cdot\vb B = 0$. Numerically, violating this constraint leads to unphysical forces parallel to the magnetic field. The constrained transport (CT) method maintains $\nabla\cdot\vb B = 0$ to machine precision by evolving **face-centred** magnetic field components.

### Face-Centred B

On a 2D mesh, the magnetic field components are stored at face centres:

```math
B_x^{i+1/2,j} \quad\text{(at the right face of cell $(i,j)$)}, \qquad
B_y^{i,j+1/2} \quad\text{(at the top face of cell $(i,j)$)}.
```

Cell-centred values are obtained by averaging: $B_{x,ij} = \tfrac{1}{2}(B_x^{i-1/2,j} + B_x^{i+1/2,j})$.

### Electric Field (EMF)

The induction equation $\partial\vb B/\partial t = -\nabla\times\vb E$ is discretised using corner-centred electromotive forces (EMFs). In 2D with a single EMF component $\mathcal E_z$:

```math
\mathcal E_z^{i+1/2,j+1/2} = \frac{1}{4}\left(\mathcal E_{z,ij} + \mathcal E_{z,i+1,j} + \mathcal E_{z,i,j+1} + \mathcal E_{z,i+1,j+1}\right),
```

where $\mathcal E_{z,ij} = v_{x,ij}B_{y,ij} - v_{y,ij}B_{x,ij}$ or equivalently obtained from the Riemann solver fluxes.

### CT Update

The face-centred fields are updated using a discrete curl of the EMF:

```math
\begin{aligned}
B_x^{i+1/2,j} &\leftarrow B_x^{i+1/2,j} - \frac{\Delta t}{\Delta y}\left(\mathcal E_z^{i+1/2,j+1/2} - \mathcal E_z^{i+1/2,j-1/2}\right), \\
B_y^{i,j+1/2} &\leftarrow B_y^{i,j+1/2} + \frac{\Delta t}{\Delta x}\left(\mathcal E_z^{i+1/2,j+1/2} - \mathcal E_z^{i-1/2,j+1/2}\right).
\end{aligned}
```

This update preserves the discrete divergence

```math
(\nabla\cdot\vb B)_{ij} = \frac{B_x^{i+1/2,j} - B_x^{i-1/2,j}}{\Delta x} + \frac{B_y^{i,j+1/2} - B_y^{i,j-1/2}}{\Delta y}
```

exactly: if $(\nabla\cdot\vb B)_{ij} = 0$ initially, it remains zero to machine precision at all subsequent times.

### Vector Potential Initialisation

For problems with discontinuous magnetic fields (e.g., field loop advection), initialising $\vb B$ by point evaluation can violate the discrete divergence constraint. Instead, one should initialise from a vector potential $\vb A$ via Stokes' theorem. In 2D with $A_z(x,y)$:

```math
B_x^{i+1/2,j} = \frac{1}{\Delta y}\int_{y_{j-1/2}}^{y_{j+1/2}} \pdv{A_z}{y}\bigg|_{x_{i+1/2}} \dd y \approx \frac{A_z(x_{i+1/2}, y_{j+1/2}) - A_z(x_{i+1/2}, y_{j-1/2})}{\Delta y},
```

```math
B_y^{i,j+1/2} = -\frac{1}{\Delta x}\int_{x_{i-1/2}}^{x_{i+1/2}} \pdv{A_z}{x}\bigg|_{y_{j+1/2}} \dd x \approx -\frac{A_z(x_{i+1/2}, y_{j+1/2}) - A_z(x_{i-1/2}, y_{j+1/2})}{\Delta x}.
```

This guarantees $(\nabla\cdot\vb B)_{ij} = 0$ by construction via the exactness of the discrete curl-grad identity.

## Viscous Terms (Navier-Stokes)

The viscous fluxes are computed using central finite differences. In 1D, the viscous stress is

```math
\tau_{xx} = \frac{4}{3}\mu\pdv{v_x}{x} \approx \frac{4}{3}\mu\frac{v_{x,i+1} - v_{x,i-1}}{2\Delta x},
```

and the heat flux is

```math
q_x = -\kappa\pdv{T}{x} \approx -\kappa\frac{T_{i+1} - T_{i-1}}{2\Delta x}.
```

In 2D, the stress tensor includes off-diagonal terms $\tau_{xy}$, which require cross-derivative estimates obtained by four-cell averages:

```math
\left.\pdv{v_x}{y}\right|_{i,j} \approx \frac{1}{4}\left(\frac{v_{x,i,j+1} - v_{x,i,j-1}}{2\Delta y} + \frac{v_{x,i+1,j+1} - v_{x,i+1,j-1}}{2\Delta y} + \cdots\right).
```

The viscous time step constraint is $\Delta t_{\mathrm{visc}} = \tfrac{1}{2}\rho_{\min}\Delta x^2/\mu$ (1D) or $\Delta t_{\mathrm{visc}} = \tfrac{1}{2}\rho_{\min}/[\mu(1/\Delta x^2 + 1/\Delta y^2)]$ (2D).

## Relativistic MHD

### SRMHD Conserved Variables

The SRMHD system uses conserved variables $\vb U = (D, S_x, S_y, S_z, \tau, B_x, B_y, B_z)^{\mathrm T}$, where:

- $D = \rho W$ is the relativistic mass density,
- $S_j = (\rho h W^2 + |\vb B|^2)v_j - (\vb v\cdot\vb B)B_j$ is the momentum density,
- $\tau = \rho h W^2 + |\vb B|^2 - P_{\mathrm{tot}} - D$ is the energy density minus rest mass.

The key identity $(\rho h + b^2)W^2 = \rho h W^2 + |\vb B|^2 + (b^0)^2$ (where $b^0 = W(\vb v\cdot\vb B)$) means the energy flux is $F_\tau = S_n - D v_n = (w_{\mathrm{tot}} - D)v_n - b^0 b_n$, not $(\\tau + P_{\mathrm{tot}})v_n - b^0 b_n$.

### Conservative-to-Primitive Conversion

Since the SRMHD equations are implicit in the primitive variables $(\rho, \vb v, P, \vb B)$, an iterative procedure (Newton's method with the Palenzuela approach) is required:

1. Define $\xi = \rho h W^2$.
2. Compute $v^2 = \bigl[|\vb S|^2\xi^2 + (\vb S\cdot\vb B)^2(2\xi + |\vb B|^2)\bigr] / \bigl[\xi^2(\xi + |\vb B|^2)^2\bigr]$.
3. Compute $W = 1/\sqrt{1 - v^2}$, then $\rho = D/W$, then $P$ from the EOS.
4. Solve $f(\xi) = \xi + |\vb B|^2 - P_{\mathrm{tot}} - (\tau + D) = 0$ iteratively.

### GRMHD Valencia Formulation

The GRMHD equations in the Valencia formulation evolve densitised conserved variables $\sqrt{\gamma}\,\vb U$ with:

- Fluxes modified by the lapse $\alpha$ and shift $\beta^i$: $\vb F^i = \alpha\vb F^i_{\mathrm{phys}} - \beta^i\vb U$.
- Geometric source terms $\vb S_{\mathrm{geom}}$ proportional to Christoffel symbols (spacetime curvature).

Available metrics include Minkowski (flat), Schwarzschild (non-rotating black hole), and Kerr (rotating black hole).

## Block-Structured AMR

### Tree Hierarchy

The block-structured adaptive mesh refinement (AMR) uses a tree of `AMRBlock`s. Each block is a fixed-size patch (e.g., $8\times 8$ cells) that can be refined into 4 child blocks (2D) or 8 child blocks (3D), each with the same number of cells but half the grid spacing.

### Prolongation and Restriction

**Prolongation** (coarse $\to$ fine): Each coarse cell is subdivided into $2^d$ fine cells. Conservative injection is used by default; for MHD, divergence-preserving prolongation maintains $\nabla\cdot\vb B = 0$.

**Restriction** (fine $\to$ coarse): Fine-grid data is volume-averaged back to the parent block:

```math
\bar{\vb U}_{\mathrm{coarse}} = \frac{1}{2^d}\sum_{k=1}^{2^d}\bar{\vb U}_{\mathrm{fine},k}.
```

### Flux Correction

At coarse-fine boundaries, the coarse-grid flux is replaced by the sum of fine-grid fluxes (Berger-Colella algorithm) to maintain strict conservation:

```math
\bar{\vb U}_{\mathrm{coarse}} \leftarrow \bar{\vb U}_{\mathrm{coarse}} - \frac{\Delta t}{\Delta x_{\mathrm{coarse}}}\left(\hat{\vb F}_{\mathrm{coarse}} - \frac{1}{r}\sum_{k=1}^{r}\hat{\vb F}_{\mathrm{fine},k}\right),
```

where $r$ is the refinement ratio.

### Refinement Criteria

Two built-in criteria control when blocks are refined or coarsened:

- **`GradientRefinement`**: Refines when $|\nabla \vb U_k| > \theta_r$ for a chosen variable index $k$; coarsens when $|\nabla \vb U_k| < \theta_c$.
- **`CurrentSheetRefinement`**: Refines near current sheets where $|\nabla\times\vb B|$ exceeds a threshold.
