# adjoint/types.jl — Discrete adjoint algorithm types (Wave 4).
#
# Scope: dense linear adjoint identities only. `SteadyAdjoint` solves the
# transposed system for a given (A, b, u); `TransientAdjoint` dispatches
# to the checkpointed linear-transient sweep (`solve_transient_adjoint_linear`,
# v3.107). Neither is wired into the SIMPLE / PIMPLE outer loops, and no
# SciMLSensitivity integration exists.
#
# Tier: experimental — math identity only (A^T λ = (∂J/∂u)^T).

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
dJ/dp = ∂J/∂p − λ^T · ∂R/∂p
```

For the V&V identity `J(u) = c^T · u` with fixed `A(p) = A`, the partial
`∂J/∂p` vanishes and `∂R/∂p = −∂b/∂p`, so `dJ/dp = λ^T · ∂b/∂p`.

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

Algorithm marker for the checkpointed LINEAR transient adjoint
(`solve_transient_adjoint_linear`, backward-Euler time discretisation,
uniform checkpointing). This is a linear-system identity, not a PIMPLE
adjoint: the nonlinear PIMPLE outer loop is not differentiated and no
solver adapters exist yet.

Tier: experimental (linear identity only).
"""
struct TransientAdjoint <: AbstractAdjointAlgorithm end
