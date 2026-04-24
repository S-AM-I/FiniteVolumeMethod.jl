# adjoint/types.jl — Discrete adjoint algorithm types (Wave 4).
#
# v3.0 fast-path: only the steady (SIMPLE) discrete adjoint is implemented.
# Transient PIMPLE adjoint requires checkpointing and is deferred to v3.1
# — `TransientAdjoint` is shipped as a marker type whose `solve` path
# emits a warn + error.
#
# Tier: experimental — math identity only (A^T λ = (∂J/∂u)^T); the adjoint
# is not yet wired into the full SIMPLE outer loop (that remains a v3.1 task).

"""
    AbstractAdjointAlgorithm

Supertype for discrete-adjoint algorithms. Concrete subtypes select between
steady (linearized about a converged SIMPLE state) and transient
(checkpointed PIMPLE, research-grade, not yet implemented).
"""
abstract type AbstractAdjointAlgorithm end

"""
    SteadyAdjoint

Discrete adjoint for a steady problem `R(u, p) = A(p) · u − b(p) = 0` with
cost functional `J(u, p)`. Solves the linear adjoint equation

```
A(p)^T · λ = (∂J/∂u)^T
```

and evaluates the total derivative

```
dJ/dp = ∂J/∂p + λ^T · ∂R/∂p
```

For the V&V identity `J(u) = c^T · u` with fixed `A(p) = A`, the partial
`∂J/∂p` vanishes and `∂R/∂p = ∂b/∂p`, so `dJ/dp = λ^T · ∂b/∂p`.

# Fields
- `linear_solver::Any` — backend for the adjoint linear solve. `nothing`
  uses Julia's backslash; pass any callable `(A, b) -> u` or a LinearSolve
  algorithm when wired through `_dispatch_solve`. Tier: experimental.
"""
struct SteadyAdjoint{S} <: AbstractAdjointAlgorithm
    linear_solver::S
end

SteadyAdjoint(; linear_solver = nothing) = SteadyAdjoint{typeof(linear_solver)}(linear_solver)

"""
    TransientAdjoint

Marker type for the transient PIMPLE adjoint. The implementation is
deferred to v3.1 because it requires full-trajectory checkpointing and
a time-reverse sweep over the nonlinear sub-iterations. Calling any
`solve_*` routine on this type warns and throws.

Tier: experimental (deferred).
"""
struct TransientAdjoint <: AbstractAdjointAlgorithm end
