# Experimental — Scope and Caveats

Everything under the `Experimental` module is research scaffolding. It is
included because it is useful to develop against, not because it is ready to
support scientific claims. Entry points emit a one-time warning per session.

**Nothing in this section is covered by the package's validation claims.**
Consult the [Capability Matrix](../capability_matrix.md) and the validation
manifest before relying on any of it.

## Honest per-module status

The following is deliberately more conservative than the module names suggest.

| Module | What actually exists |
|---|---|
| `pressure_based` | A real subsonic compressible pressure equation (ρ_f mass fluxes, implicit `ψ = ∂ρ/∂p` diagonal, `(1/ρ)∇p` momentum). Closed-box mass is conserved to machine precision and the low-Mach limit matches the incompressible solver. Subsonic only — there is no `div(phid, p)` shock treatment — and the momentum time derivative neglects `∂ρ/∂t`. |
| `aeroacoustics` | `fwh_farassat1a` is a genuine retarded-time Farassat 1A implementation for static surfaces, validated against analytic monopole and dipole sources. The remaining static-sum functions are near-field snapshot approximations, the Lighthill quadrupole term is a stub, and the "PML" is a plain damping sponge. |
| `solid_mechanics` | A decoupled per-component Poisson solve, not full coupled elasticity. `traction_bcs` are unused by the solvers and throw if supplied. |
| `fsi` | A generic Aitken-Δ² fixed-point accelerator over user-supplied callbacks. There are no adapters to this package's PISO or elasticity solvers; it has only been exercised against a mock one-degree-of-freedom spring-damper. Interface transfer supports matching meshes only. |
| `adjoint` | Dense linear adjoint identities only. Not wired into SIMPLE/PIMPLE, and there is no SciMLSensitivity integration. |
| `mesh_generation` | Octree castellated refinement plus an STL snapping prototype. There is **no** extraction path from the octree to a solver-usable mesh, and no layer addition. |
| `population_balance` | A zero-dimensional moment/class kernel library (QMoM/DQMoM/class method), not coupled to transport. |
| `parallel` | Per-rank local solves with halo exchange between outer iterations, in the manner of additive Schwarz. No distributed matrix is ever assembled. |

## Tutorials

The tutorials in this section run, and their printed results are real, but they
demonstrate usage rather than establish validation evidence.
