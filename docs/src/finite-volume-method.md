# The Finite Volume Method

```@contents
Pages = ["finite-volume-method.md"]
```

This document provides a treatment of the Finite Volume Method (FVM), covering foundational continuum mechanics, classical CFD discretization, and advanced topics. For the implementation details of the two solvers in this package, see [Parabolic Solver (Cell-Vertex)](math.md) and [Hyperbolic Solver (Cell-Centered)](hyperbolic/math.md).

## 1. Foundations of Continuum Mechanics and Conservation Laws

The mathematical modeling of fluid mechanics and heat transfer rests upon conservation laws. The **Finite Volume Method (FVM)** is distinguished by its direct derivation from **integral conservation laws**, rather than differential forms. This endows FVM with intrinsic robustness in preserving physical quantities locally and globally, making it the dominant framework in modern Computational Fluid Dynamics (CFD).

### 1.1 The Reynolds Transport Theorem

The bridge between a **system** (material volume) and a **control volume** (fixed in space) is the **Reynolds Transport Theorem (RTT)**. For a scalar property $\phi$ per unit mass:

```math
\frac{D}{Dt} \int_{V_{sys}} \rho \phi \, dV = \frac{\partial}{\partial t} \int_{CV} \rho \phi \, dV + \oint_{CS} \rho \phi (\mathbf{v} \cdot \mathbf{n}) \, dA
```

FVM starts directly from this integral form, ensuring discrete conservation even on coarse meshes.

### 1.2 The Navier-Stokes Equations in Integral Form

#### 1.2.1 Conservation of Mass (Continuity)

Setting $\phi = 1$:

```math
\frac{\partial}{\partial t} \int_{\Omega} \rho \, d\Omega + \oint_{\Gamma} \rho (\mathbf{v} \cdot \mathbf{n}) \, d\Gamma = 0
```

For incompressible flow ($\rho = \text{const}$), this enforces a divergence-free velocity field.

#### 1.2.2 Conservation of Momentum

Setting $\phi = \mathbf{v}$:

```math
\frac{\partial}{\partial t} \int_{\Omega} \rho \mathbf{v} \, d\Omega + \oint_{\Gamma} \rho \mathbf{v} (\mathbf{v} \cdot \mathbf{n}) \, d\Gamma = \oint_{\Gamma} \boldsymbol{\tau} \cdot \mathbf{n} \, d\Gamma - \oint_{\Gamma} p \mathbf{n} \, d\Gamma + \int_{\Omega} \rho \mathbf{g} \, d\Omega
```

Momentum transport is both **convective** and **diffusive**.

#### 1.2.3 General Scalar Transport Equation

All conservation laws may be written in unified form:

```math
\underbrace{\frac{\partial}{\partial t} \int_{\Omega} \rho \phi \, d\Omega}_{\text{Transient}} + \underbrace{\oint_{\Gamma} \rho \phi (\mathbf{v} \cdot \mathbf{n}) \, d\Gamma}_{\text{Convection}} = \underbrace{\oint_{\Gamma} \Gamma_\phi \nabla \phi \cdot \mathbf{n} \, d\Gamma}_{\text{Diffusion}} + \underbrace{\int_{\Omega} S_\phi \, d\Omega}_{\text{Source}}
```

The terms represent:
- **Transient**: Rate of change of $\phi$ inside the control volume. Set to 0 for steady-state.
- **Convection**: Transport of $\phi$ due to fluid velocity. Set to 0 for pure diffusion (e.g., solid conduction).
- **Diffusion**: Transport of $\phi$ due to gradients. Set to 0 for inviscid flows (Euler equations).
- **Source**: Generation or destruction of $\phi$ (e.g., chemical reaction, gravity, heat generation).

#### 1.2.4 Application to Specific Conservation Laws

| Conservation Law | $\phi$ | $\Gamma_\phi$ | $S_\phi$ |
|-----------------|--------|---------------|----------|
| Mass (Continuity) | 1 | 0 | 0 |
| $x$-Momentum | $u$ | $\mu$ (Viscosity) | $-\partial p/\partial x + F_x$ |
| $y$-Momentum | $v$ | $\mu$ (Viscosity) | $-\partial p/\partial y + F_y$ |
| Energy | $h$ (Enthalpy) | $k/c_p$ | Viscous dissipation, radiation |
| Species | $c$ (Concentration) | $D$ (Diffusivity) | Reaction rate |
| Turbulence ($k$-$\varepsilon$) | $k, \varepsilon$ | $\mu_t/\sigma_k, \mu_t/\sigma_\varepsilon$ | Production/Dissipation |

## 2. Finite Volume Discretization Framework

FVM discretizes the **control volume**, ensuring that flux entering one cell exits its neighbor exactly ("telescoping flux"), guaranteeing global conservation.

### 2.1 Mesh Topologies and Geometric Definitions

#### 2.1.1 Structured Grids

Implicit connectivity with efficient memory access, but difficult to generate for complex geometries.

#### 2.1.2 Unstructured and Polyhedral Grids

Arbitrary polygons/polyhedra with connectivity stored explicitly:
- Each face has **Owner / Neighbor**
- High geometric flexibility
- Higher memory and cache cost

#### 2.1.3 Cell-Centered vs Cell-Vertex Formulations

1. **Cell-Centered FVM (CC-FVM)**: Variables at cell centroids (most commercial solvers)
2. **Cell-Vertex FVM (CV-FVM)**: Variables at vertices; dual mesh control volumes

### 2.2 Geometric Parameters and Surface Approximation

Surface integrals are approximated as:

```math
\oint_{\Gamma} \mathbf{J} \cdot \mathbf{n} \, d\Gamma \approx \sum_f \mathbf{J}_f \cdot \mathbf{S}_f
```

Key mesh quality metrics:
- **Orthogonality**: Alignment of face normal with cell-center connection
- **Skewness**: Deviation of face center from intersection point

## 3. Convective Transport: High-Resolution Spatial Reconstruction

The key challenge is determining the face value $\phi_f$.

### 3.1 Upwind Differencing Scheme (UDS)

```math
\phi_f = \begin{cases} \phi_P & F_f \ge 0 \\ \phi_N & F_f < 0 \end{cases}
```

- First-order accurate
- Unconditionally stable
- Strong numerical diffusion

### 3.2 Central Differencing Scheme (CDS)

```math
\phi_f = f_x \phi_P + (1 - f_x)\phi_N
```

- Second-order accurate
- Unbounded in convection-dominated flows

### 3.3 Godunov's Theorem

No **linear**, **second-order**, **monotone** scheme exists.

### 3.4 TVD Schemes and Flux Limiters

```math
\phi_f = \phi_{UDS} + \frac{1}{2}\psi(r)\left(\phi_{CDS} - \phi_{UDS}\right)
```

Gradient ratio:

```math
r = \frac{\phi_P - \phi_{up}}{\phi_N - \phi_P}
```

#### Common Limiters

| Limiter | Formula |
|---------|---------|
| Minmod | $\psi(r) = \max(0, \min(1, r))$ |
| Superbee | $\psi(r) = \max(0, \min(2r, 1), \min(r, 2))$ |
| Van Leer | $\psi(r) = (r + |r|)/(1 + |r|)$ |

## 4. Diffusive Transport and Gradient Computation

### 4.1 Gradient Reconstruction Methods

#### 4.1.1 Green-Gauss Method

```math
(\nabla \phi)_P \approx \frac{1}{V_P} \sum_f \phi_f \mathbf{S}_f
```

Variants: Cell-based, Node-based

#### 4.1.2 Least Squares Method

Assume linear variation:

```math
\phi_{nb} \approx \phi_P + (\nabla \phi)_P \cdot \mathbf{d}_{nb}
```

Minimize:

```math
E^2 = \sum_{nb} w_{nb} \left[\phi_{nb} - (\phi_P + \nabla \phi_P \cdot \mathbf{d}_{nb})\right]^2
```

### 4.2 Non-Orthogonal Correction

Decompose face area vector:

```math
\mathbf{S}_f = \mathbf{E} + \mathbf{T}
```

Diffusive flux:

```math
\nabla \phi \cdot \mathbf{S}_f \approx |\mathbf{E}|\frac{\phi_N - \phi_P}{|\mathbf{d}|} + (\nabla \phi)_f \cdot \mathbf{T}
```

## 5. Temporal Evolution and Stability Analysis

### 5.1 Explicit Time Integration

Forward Euler:

```math
\frac{\rho V (\phi_P^{n+1} - \phi_P^n)}{\Delta t} = -\sum_f F_f^n + S_\phi^n V
```

**CFL condition**:

```math
\text{CFL} = \frac{u \Delta t}{\Delta x} \le 1
```

Runge-Kutta methods provide multi-stage explicit schemes with larger stability regions, favored in compressible flows. For a detailed treatment of SSP-RK3 and IMEX schemes as implemented in this package, see [Hyperbolic Solver Mathematical Details](hyperbolic/math.md).

### 5.2 Implicit Time Integration

```math
\frac{\rho V (\phi_P^{n+1} - \phi_P^n)}{\Delta t} = -\sum_f F_f^{n+1} + S_\phi^{n+1} V
```

- Unconditionally stable
- Requires matrix solvers (AMG)

## 6. Pressure-Velocity Coupling Algorithms

### 6.1 Rhie-Chow Interpolation

Prevents checkerboard pressure on collocated grids:

```math
u_f = \overline{u_f} + \overline{D_f}\left(\overline{\nabla p}_f - (\nabla p)_f\right)
```

### 6.2 Segregated Solvers

#### SIMPLE (Semi-Implicit Method for Pressure-Linked Equations)

- Predictor-corrector loop
- Requires under-relaxation

#### PISO (Pressure-Implicit with Splitting of Operators)

- Two pressure correctors
- Efficient for transient flows

## 7. Boundary Condition Implementation

General discretized equation:

```math
a_P \phi_P + \sum_{nb} a_{nb} \phi_{nb} = S_u
```

The three fundamental boundary condition types (Dirichlet, Neumann, Robin) are implemented differently depending on the solver. For full details, see [Parabolic Solver](math.md) and [Hyperbolic Solver](hyperbolic/math.md).

## 8. The Arbitrary Lagrangian-Eulerian (ALE) Formulation

For problems involving moving boundaries (fluid-structure interaction, free surfaces), the formulation must account for control volume movement:

```math
\frac{d}{dt}\int_{\Omega(t)} \rho\phi \, d\Omega + \oint_{\partial\Omega(t)} \rho\phi(\mathbf{v} - \mathbf{v}_g) \cdot \mathbf{n} \, dS = \oint_{\partial\Omega(t)} (\Gamma\nabla\phi) \cdot \mathbf{n} \, dS + \int_{\Omega(t)} S_\phi \, d\Omega
```

Where $\mathbf{v}_g$ is the mesh velocity:
- If $\mathbf{v}_g = 0$: Eulerian (fixed grid)
- If $\mathbf{v}_g = \mathbf{v}$: Lagrangian (grid moves with fluid)

## 9. Hyperbolic Conservation Laws and Beyond

The finite volume method extends naturally to hyperbolic conservation laws including compressible Euler equations, ideal and relativistic MHD, and the full GRMHD system in curved spacetime. These formulations introduce additional challenges: approximate Riemann solvers for flux computation, high-order reconstruction (MUSCL, WENO), the divergence-free constraint on the magnetic field (constrained transport), conservative-to-primitive variable recovery, and adaptive mesh refinement. For a complete treatment of these topics as implemented in this package, see [Hyperbolic Solver Mathematical Details](hyperbolic/math.md).

## 10. Advanced Topics

### 10.1 WENO on Unstructured Meshes

- Arbitrarily high order
- Nonlinear stencil weighting
- Shock-capturing with low dissipation

### 10.2 Machine Learning Accelerated FVM

- ML replaces turbulence closures
- Conservation laws enforced by FVM structure
- Neural networks as constitutive models

### 10.3 The Israel-Stewart Formulation (Viscous Relativistic Fluids)

Standard viscosity violates causality in relativity. The Israel-Stewart formulation elevates the viscous stress tensor to a dynamic variable:

```math
\tau_\pi \dot{\pi}^{\mu\nu} + \pi^{\mu\nu} = -2\eta \sigma^{\mu\nu}
```

Taking $\tau_\pi \to 0$ recovers the relativistic Navier-Stokes structure.
