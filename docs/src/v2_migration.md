# v2 Migration

FiniteVolumeMethod.jl v2 significantly expands the solver capabilities and
adopts SciML as the time-integration interface. This page summarises what
changed and how to update existing code.

## New Solver Capabilities

v2 adds a cell-centred hyperbolic solver family alongside the original
cell-vertex parabolic/elliptic solver:

| Solver family   | PDE class                          | Validation status |
|-----------------|------------------------------------|-------------------|
| `parabolic`     | Diffusion, advection-diffusion     | Stable            |
| `hyperbolic`    | Compressible Euler equations       | Stable            |
| `mhd_ct`        | Ideal MHD with constrained transport | Stable          |
| `relativistic`  | SRMHD, GRMHD                      | Stable            |
| `amr`           | Block-structured AMR (Euler, MHD)  | Provisional       |
| `coupling`      | Multi-physics operator splitting   | Provisional       |

See the [Capability Matrix](capability_matrix.md) for detailed V&V coverage.

## SciML Integration

The recommended workflow is now the SciML method-of-lines interface:

```julia
prob = HyperbolicProblem2D(law, mesh, ic, bcs, tspan)
ode  = sciml_problem(prob)          # or ODEProblem(prob)
sol  = solve(ode, SSPRK33(); dt = 1e-3, adaptive = false)
```

This enables:
- **Adaptive timestepping** via the CFL callback
- **Implicit-explicit (IMEX) splitting** via `SplitODEProblem`
- **`remake`** for parameter studies without re-allocating
- Access to the full DifferentialEquations.jl solver library

Older `solve_*` entry points still work but are not recommended for new code.

## GPU Support

A CUDA backend is available for the 2D hyperbolic solver via the `FVMCUDAExt`
package extension. GPU results must demonstrate parity against the CPU
`Float64` baseline before they are considered at the same validation level.
Currently only the 2D Euler/MHD paths have a parity check.

## Checklist for Existing Users

- Check the [Capability Matrix](capability_matrix.md) before relying on a
  solver family for publication-grade results.
- Prefer `sciml_problem(prob)` for new workflows.
- Provisional and experimental solvers are suitable for development and
  exploration, not for final published results.
