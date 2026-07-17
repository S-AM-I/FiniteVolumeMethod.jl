# ============================================================
# Multi-Physics Coupling: Abstract Types and Splitting Schemes
# ============================================================
#
# Operator splitting framework for coupling multiple physics
# operators (e.g., hyperbolic + stiff source, advection + diffusion).
#
# Each operator advances the shared state U by a time step dt.
# Splitting schemes compose operators to achieve desired accuracy:
#   - Lie-Trotter: 1st order, sequential application
#   - Strang: 2nd order, symmetric sweep

"""
    AbstractOperator

Abstract supertype for physics operators in operator splitting.

Subtypes must implement:
- `advance!(U, op, dt, t, workspace)`: Advance state `U` in-place by time `dt`.
- `compute_operator_dt(op, U, t)`: Compute the maximum stable time step.
"""
abstract type AbstractOperator end

"""
    AbstractSplittingScheme

Abstract supertype for operator splitting methods.
"""
abstract type AbstractSplittingScheme end

"""
    LieTrotterSplitting <: AbstractSplittingScheme

First-order Lie-Trotter operator splitting. Operators are applied
sequentially, each for the full time step:

    U^{n+1} = Lₙ(Δt) ∘ ⋯ ∘ L₂(Δt) ∘ L₁(Δt) [Uⁿ]
"""
struct LieTrotterSplitting <: AbstractSplittingScheme end

"""
    StrangSplitting <: AbstractSplittingScheme

Second-order Strang operator splitting. For two operators:

    U^{n+1} = L₁(Δt/2) ∘ L₂(Δt) ∘ L₁(Δt/2) [Uⁿ]

For N operators, a symmetric forward-backward sweep is used:
forward half-steps for operators 1 to N-1, a full step for
operator N, then backward half-steps for N-1 to 1.
"""
struct StrangSplitting <: AbstractSplittingScheme end

"""
    CoupledProblem{Ops, Scheme, FT}

A coupled multi-physics problem solved via operator splitting.

# Fields
- `operators::Ops`: Tuple of `AbstractOperator` instances.
- `splitting::Scheme`: Splitting scheme.
- `initial_time::FT`: Start time.
- `final_time::FT`: End time.
"""
struct CoupledProblem{Ops <: Tuple, Scheme <: AbstractSplittingScheme, FT}
    operators::Ops
    splitting::Scheme
    initial_time::FT
    final_time::FT
end
