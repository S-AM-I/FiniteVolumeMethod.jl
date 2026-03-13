# Verification & Validation

This section only includes cases that are currently treated as scientific verification evidence in this repository.
Examples that are merely smoke tests, qualitative regressions, or surrogate setups are intentionally excluded from the
tables below even if related scripts still exist elsewhere in the tree.

The repository's capability matrix determines which solver families are currently allowed to support publication-grade
claims. Stable claim-bearing solver features must have automated evidence in this section; provisional features may
still appear elsewhere in the repository, but they are not elevated to the same claim status until their evidence ladder
is complete.

## Verification vs. Validation

**Verification** asks whether the discretisation and implementation solve the intended equations correctly.
These cases use manufactured solutions, exact solutions, or discrete invariants.

**Validation** asks whether the chosen equations reproduce external physical measurements.
No experimental validation claims are currently made in this section, because the available cavity/torus surrogate
examples are not yet rigorous literature-backed reproductions.

## Code Verification

| Example | Evidence Basis | Metric | Acceptance |
|:--------|:---------------|:-------|:-----------|
| [MMS Convergence](@ref) | Manufactured solution | Observed convergence rates in multiple norms | Meets script tolerance for second-order behaviour |
| [Decoupled MMS Convergence](@ref) | Manufactured solution | Separate spatial and temporal convergence rates | Meets script tolerances for each refinement study |
| [Euler MMS Convergence](@ref) | Manufactured solution | L1 density error, pressure error, GCI ratio | Density rates and GCI remain within script tolerances |
| [Poisson Convergence](@ref) | Exact solution | L2 error and observed convergence rate | Meets script convergence tolerance |
| [Smooth Advection](@ref) | Exact solution | L1 transport error and observed convergence rate | Meets scheme-specific tolerance |
| [Source Term Convergence](@ref) | Manufactured solution | Convergence with source forcing | Meets script tolerance |
| [Flux Balance](@ref) | Discrete invariant | Residual in assembled flux balance | Machine-precision imbalance |
| [Conservation Verification](@ref) | Discrete invariant | Drift in conserved totals | Remains within script tolerance |
| [Species Conservation](@ref) | Discrete invariant | Drift in total and per-species quantities | Remains within script tolerance |
| [Passive Scalar Convergence](@ref) | Exact solution | L1 passive-scalar error and observed rate | Meets script tolerance |
| [GRMHD Flat-Space Reduction](@ref) | Asymptotic reduction | GRMHD/SRMHD flux mismatch and round-trip error | Remains below stated thresholds |
| [GRMHD Newtonian Limit](@ref) | Asymptotic reduction | Primitive-recovery error and static-atmosphere drift | Remains below stated thresholds |
| [MHD div(B) Preservation](@ref) | Discrete invariant | Maximum cellwise ``\nabla\cdot B`` | Remains below stated threshold |

## Analytical Benchmarks

| Example | Evidence Basis | Metric | Acceptance |
|:--------|:---------------|:-------|:-----------|
| [Sod Grid Convergence](@ref) | Exact Riemann solution | L1 density error and convergence trend | Errors decrease monotonically |
| [Toro Riemann Tests](@ref) | Published star-state values | Star-region pressure error | Each case meets the script tolerance |
| [MHD Convergence](@ref) | Exact Alfvén-wave solution | L1 magnetic-field error and observed rate | Rates exceed the script threshold |
| [Navier-Stokes Convergence](@ref) | Exact Taylor-Green vortex | Linf velocity error and observed rate | Rates exceed the script threshold |
| [Taylor-Green KE Decay](@ref) | Exact kinetic-energy decay law | Relative decay error | Meets the script tolerance |
| [Porous Medium (Barenblatt)](@ref) | Exact self-similar solution | L2 error trend and minimum observed rate | Errors decrease monotonically |
| [SRMHD Convergence](@ref) | Exact smooth-wave solution | L1 density error and observed rate | Rates exceed the script threshold |
| [SRMHD Eigenmode Convergence](@ref) | Linearised exact eigenmodes | Self-convergence rate for each mode | All mode-wise rates exceed the script threshold |
| [GRMHD Convergence](@ref) | Exact smooth-wave solution in Minkowski spacetime | L1 density error and observed rate | Rates exceed the script threshold |

## Scope Notes

- Tutorials and smoke tests are validated separately as executable documentation, not as scientific evidence.
- Regression-style scripts that compare one numerical method to another without an external truth model are not listed here.
- Experimental validation will be added back only when the implementation reproduces the referenced benchmark physics and
  compares against published data with explicit quantitative acceptance criteria.
- Demoted or manually reviewed cases are tracked in the validation manifest and release report rather than being silently
  left in the tree as implied evidence.

## References

- P. J. Roache, *Verification and Validation in Computational Science and Engineering*, 1998.
- ASME V&V 20-2009, *Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer*, 2009.
- W. L. Oberkampf and T. G. Trucano, "Verification and validation in computational fluid dynamics," *Progress in Aerospace Sciences*, 2002.
- C. J. Roy, "Review of code and solution verification procedures for computational simulation," *Journal of Computational Physics*, 2005.
- E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., 2009.
