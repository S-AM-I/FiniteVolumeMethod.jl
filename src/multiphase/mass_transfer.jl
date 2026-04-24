# multiphase/mass_transfer.jl — Inter-phase mass transfer placeholder
#
# Provides the abstract `AbstractMassTransferModel` interface and a
# no-op `NoMassTransfer` model used as the default in
# `TwoFluidProblem`. Real models (Schnerr-Sauer cavitation, Merkle
# cavitation, HEM phase-change) plug in by subtyping and overloading
# `mass_transfer_source_alpha` / `mass_transfer_source_energy`.
#
# This file deliberately does not couple to cavitation models yet —
# those ship as part of v3.1 Agent B's scope.

"""
    AbstractMassTransferModel

Root type for interphase mass-transfer models used by the Eulerian
two-fluid solver. Concrete subtypes implement `mass_transfer_source_alpha`
(volume-fraction source, in mass per volume per time at the gas phase)
and optionally `mass_transfer_source_energy` (enthalpy source).
"""
abstract type AbstractMassTransferModel end

"""
    NoMassTransfer <: AbstractMassTransferModel

Null mass-transfer model. Returns zero source per cell, which decouples
the two phases' continuity equations entirely (no cavitation,
evaporation, or condensation). This is the default mass-transfer model
assigned to `TwoFluidProblem` by the keyword constructor.
"""
struct NoMassTransfer <: AbstractMassTransferModel end

"""
    mass_transfer_source_alpha(model, state, prob) -> Vector{T}

Return the interphase mass-transfer source per cell `S_Γ` in units of
`kg / (m³ · s)`. Positive values add mass to the gas phase (and remove
it from the liquid); negative values do the reverse.

For the default [`NoMassTransfer`](@ref) model this returns a zero
vector. Custom models override this method to couple in cavitation,
evaporation, or condensation kinetics.

The vector length equals `length(state.alpha_g.internal)` —
interior-only; boundary values are handled via the alpha BCs.
"""
function mass_transfer_source_alpha(
        ::NoMassTransfer,
        state::TwoFluidState{Dim, T},
        prob::TwoFluidProblem{Dim, T},
    ) where {Dim, T}
    return zeros(T, length(state.alpha_g.internal))
end

"""
    mass_transfer_source_energy(model, state, prob) -> Vector{T}

Return the enthalpy source per cell from interphase mass transfer. The
default `NoMassTransfer` model returns zero. Reserved for future
coupling with the thermal / energy equation.
"""
function mass_transfer_source_energy(
        ::NoMassTransfer,
        state::TwoFluidState{Dim, T},
        prob::TwoFluidProblem{Dim, T},
    ) where {Dim, T}
    return zeros(T, length(state.alpha_g.internal))
end
