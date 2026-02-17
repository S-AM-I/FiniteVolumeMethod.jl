# Verification & Validation

```@contents
Pages = [
    "mms_convergence.md",
    "poisson_convergence.md",
    "smooth_advection_convergence.md",
    "sod_grid_convergence.md",
    "conservation_verification.md",
    "mhd_divb_verification.md",
]
Depth = 1
```

## Verification vs. Validation

**Verification** asks: *"Are we solving the equations correctly?"* It checks that the numerical
implementation converges at the expected rate to a known exact or manufactured solution.

**Validation** asks: *"Are we solving the correct equations?"* It compares simulation results
against experimental data or benchmark results from the literature.

This section focuses primarily on **code verification** — confirming that the discretisation
and implementation are correct via convergence studies and conservation checks.

## Summary of Verification Cases

| Example | Solver | Property Verified | Expected Order |
|:--------|:-------|:-------------------|:---------------|
| [MMS Convergence](@ref) | Parabolic (vertex-centred) | Code correctness via manufactured solution | ``O(h^2)`` |
| [Poisson Convergence](@ref) | Parabolic (steady-state) | Steady-state solver accuracy | ``O(h^2)`` |
| [Smooth Advection](@ref) | Hyperbolic (cell-centred) | Reconstruction scheme accuracy | ``O(h^1)`` to ``O(h^5)`` |
| [Sod Grid Convergence](@ref) | Hyperbolic (cell-centred) | Shock-capturing convergence | ``O(h^{0.5\text{--}1})`` |
| [Conservation Verification](@ref) | Hyperbolic (cell-centred) | Discrete conservation properties | Machine epsilon |
| [MHD div(B) Preservation](@ref) | Hyperbolic (MHD + CT) | Constraint preservation | Machine epsilon |

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

## References

- P. J. Roache, *Verification and Validation in Computational Science and Engineering*, Hermosa Publishers, 1998.
- W. L. Oberkampf and T. G. Trucano, "Verification and validation in computational fluid dynamics," *Progress in Aerospace Sciences*, 38(3):209–272, 2002.
- C. J. Roy, "Review of code and solution verification procedures for computational simulation," *Journal of Computational Physics*, 205(1):131–156, 2005.
