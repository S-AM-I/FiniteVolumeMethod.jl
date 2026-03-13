# Verification & Validation

## Verification vs. Validation

**Verification** asks: *"Are we solving the equations correctly?"* It checks that the numerical
implementation converges at the expected rate to a known exact or manufactured solution.

**Validation** asks: *"Are we solving the correct equations?"* It compares simulation results
against experimental data or benchmark results from the literature.

This section is organised following the ASME V&V 20-2009 framework into three tiers:

1. **Code Verification** — manufactured-solution convergence, operator consistency, and
   asymptotic-reduction checks that prove the discretisation is implemented correctly.
2. **Analytical Benchmarks** — comparison against exact, semi-analytical, or literature-grade
   reference solutions for the physical models the package implements.
3. **Experimental Validation** — comparison against published experimental data (stretch goals).

## Code Verification

| Example | Solver | Property Verified | Expected Order |
|:--------|:-------|:-------------------|:---------------|
| [MMS Convergence](@ref) | Parabolic (vertex-centred) | Code correctness via manufactured solution | ``O(h^2)`` |
| [Decoupled MMS Convergence](@ref) | Parabolic (vertex-centred) | Spatial vs temporal error separation | ``O(h^2)`` spatial, ``O(\Delta t^{4\text{--}5})`` temporal |
| [Euler MMS Convergence](@ref) | Hyperbolic (cell-centred) | Cell-centred Euler MMS convergence | ``O(h^{1.5+})`` |
| [Poisson Convergence](@ref) | Parabolic (steady-state) | Steady-state solver accuracy | ``O(h^2)`` |
| [Smooth Advection](@ref) | Hyperbolic (cell-centred) | Reconstruction scheme accuracy | ``O(h^1)`` to ``O(h^5)`` |
| [Source Term Convergence](@ref) | Hyperbolic (cell-centred) | Source term integration accuracy | ``O(h^2)`` |
| [Flux Balance](@ref) | Parabolic (vertex-centred) | Discrete flux balance verification | Machine epsilon |
| [Conservation Verification](@ref) | Hyperbolic (cell-centred) | Discrete conservation properties | Machine epsilon |
| [Species Conservation](@ref) | Hyperbolic (cell-centred) | Multi-species conservation | Machine epsilon |
| [Passive Scalar Convergence](@ref) | Hyperbolic (cell-centred) | Passive scalar transport accuracy | ``O(h^2)`` |
| [MHD Solver Comparison](@ref) | Hyperbolic (MHD) | HLL vs HLLD accuracy comparison | HLLD < HLL in L1 |
| [GRMHD Flat-Space Reduction](@ref) | Hyperbolic (GRMHD) | GRMHD → SRMHD asymptotic limit | Machine epsilon |
| [GRMHD Newtonian Limit](@ref) | Hyperbolic (GRMHD) | Low-velocity Con2Prim stability | Machine epsilon |

## Analytical Benchmarks

| Example | Solver | Property Verified | Expected Order |
|:--------|:-------|:-------------------|:---------------|
| [Sod Grid Convergence](@ref) | Hyperbolic (cell-centred) | Shock-capturing convergence | ``O(h^{0.5\text{--}1})`` |
| [Toro Riemann Tests](@ref) | Hyperbolic (cell-centred) | All five Toro test problems | Reference profiles |
| [Balsara MHD Suite](@ref) | Hyperbolic (MHD) | Full Balsara MHD test battery | Reference profiles |
| [Brio-Wu Verification](@ref) | Hyperbolic (MHD) | MHD Riemann problem | Reference profiles |
| [Orszag-Tang Verification](@ref) | Hyperbolic (MHD + CT) | MHD turbulence transition | Reference profiles |
| [MHD div(B) Preservation](@ref) | Hyperbolic (MHD + CT) | Constraint preservation | Machine epsilon |
| [MHD Convergence](@ref) | Hyperbolic (MHD + CT) | Circularly polarised Alfvén wave | ``O(h^2)`` |
| [AMR Convergence](@ref) | Hyperbolic (AMR) | Refinement convergence | ``O(h^1)`` |
| [Navier-Stokes Convergence](@ref) | Hyperbolic (NS) | Viscous flow convergence | ``O(h^2)`` |
| [Taylor-Green KE Decay](@ref) | Hyperbolic (NS) | Viscous kinetic energy decay rate | Analytical ``e^{-4\nu k^2 t}`` |
| [Porous Medium (Barenblatt)](@ref) | Parabolic (vertex-centred) | Self-similar Barenblatt solution | Compact support match |
| [Premixed Flame 1D](@ref) | Hyperbolic (reactive) | Flame speed and profile | Reference profiles |
| [SRMHD Convergence](@ref) | Hyperbolic (SRMHD) | Relativistic MHD convergence | ``O(h^2)`` |
| [SRMHD Eigenmode Convergence](@ref) | Hyperbolic (SRMHD) | All SRMHD wave families | ``O(h^{0.8+})`` per mode |
| [GRMHD Convergence](@ref) | Hyperbolic (GRMHD) | GR MHD convergence | ``O(h^2)`` |
| [Bondi Accretion](@ref) | Hyperbolic (GRMHD) | Steady-state spherical accretion | Stationarity < 1% drift |

## Experimental Validation

| Example | Solver | Property Verified | Reference |
|:--------|:-------|:-------------------|:----------|
| [Lid-Driven Cavity](@ref) | Hyperbolic (NS) | Vortex centre and velocity profiles | Ghia et al. (1982) |
| [Fishbone-Moncrief Torus](@ref) | Hyperbolic (GRMHD) | Hydrostatic equilibrium in Kerr spacetime | Fishbone & Moncrief (1976) |
| [Heated Cavity](@ref) | Hyperbolic (NS) | Natural convection Nusselt number | De Vahl Davis (1983) |

## Error Norms

The following norms are used throughout:

```math
\|e\|_1 = \frac{1}{N}\sum_{i=1}^N |u_i - u_{\text{exact},i}|, \qquad
\|e\|_2 = \sqrt{\frac{1}{N}\sum_{i=1}^N (u_i - u_{\text{exact},i})^2}, \qquad
\|e\|_\infty = \max_{i} |u_i - u_{\text{exact},i}|.
```

The **convergence rate** between two meshes with ``N`` and ``2N`` cells is:

```math
p = \log_2\!\left(\frac{\|e\|_N}{\|e\|_{2N}}\right).
```

The **Grid Convergence Index** (GCI) follows the ASME V&V 20-2009 standard using three-grid
Richardson extrapolation with a safety factor of 1.25 for three or more grids:

```math
\text{GCI}_{\text{fine}} = \frac{F_s \left| \frac{f_2 - f_1}{f_1} \right|}{r^p - 1}
```

where ``r`` is the refinement ratio, ``p`` is the observed order, and ``F_s = 1.25``.
The asymptotic ratio ``\text{GCI}_{\text{coarse}} / (r^p \cdot \text{GCI}_{\text{fine}})``
should approach 1.0 in the asymptotic convergence range.

## References

- P. J. Roache, *Verification and Validation in Computational Science and Engineering*, Hermosa Publishers, 1998.
- ASME V&V 20-2009, *Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer*, 2009.
- W. L. Oberkampf and T. G. Trucano, "Verification and validation in computational fluid dynamics," *Progress in Aerospace Sciences*, 38(3):209–272, 2002.
- C. J. Roy, "Review of code and solution verification procedures for computational simulation," *Journal of Computational Physics*, 205(1):131–156, 2005.
- E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., Springer, 2009.
