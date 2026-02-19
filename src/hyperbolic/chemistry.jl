# ============================================================
# Arrhenius Chemistry Source Terms
# ============================================================
#
# Single-step and multi-reaction chemistry models implemented
# as AbstractStiffSource for use with IMEX or operator splitting.
#
# Reaction rate:  k = A * T^n * exp(-Ea / T)
# Species production:  ω_k = Σ_i (ν"_ki - ν'_ki) * R_i
# where R_i = k_i * Π_j (ρ Y_j)^{ν'_ji}
# Energy source:  S_E = -Σ_i q_i * R_i

"""
    ArrheniusReaction{NS}

A single irreversible reaction with Arrhenius kinetics.

The reaction rate is `k = A * T^n * exp(-Ea / T)` where `T` is the
temperature and `Ea` is the activation energy divided by the gas constant.

# Fields
- `A::Float64`: Pre-exponential factor.
- `n::Float64`: Temperature exponent.
- `Ea::Float64`: Activation energy / R_gas (units of temperature).
- `nu_reactant::NTuple{NS, Float64}`: Reactant stoichiometric coefficients ν'.
- `nu_product::NTuple{NS, Float64}`: Product stoichiometric coefficients ν".
- `heat_release::Float64`: Energy released per unit mass of fuel consumed (q > 0 for exothermic).
"""
struct ArrheniusReaction{NS}
    A::Float64
    n::Float64
    Ea::Float64
    nu_reactant::NTuple{NS, Float64}
    nu_product::NTuple{NS, Float64}
    heat_release::Float64
end

"""
    ReactionMechanism{NS, NR}

A collection of `NR` reactions involving `NS` species.

# Fields
- `reactions::NTuple{NR, ArrheniusReaction{NS}}`: The reactions.
- `molecular_weights::NTuple{NS, Float64}`: Molecular weights of each species.
"""
struct ReactionMechanism{NS, NR}
    reactions::NTuple{NR, ArrheniusReaction{NS}}
    molecular_weights::NTuple{NS, Float64}
end

"""
    ChemistrySource{NS, NR} <: AbstractStiffSource

Stiff source term for Arrhenius chemistry in reactive Euler equations.

Evaluates species production rates and the corresponding energy source
from chemical heat release. Designed for use with `SourceOperator` /
`StrangSplitting` or the IMEX framework.

# Fields
- `mechanism::ReactionMechanism{NS, NR}`: The reaction mechanism.
- `mu_mol::Float64`: Mean molecular weight for temperature computation
  (`T = P * μ / ρ` in non-dimensional form, or `T = P / ρ` when `μ = 1`).
"""
struct ChemistrySource{NS, NR} <: AbstractStiffSource
    mechanism::ReactionMechanism{NS, NR}
    mu_mol::Float64
end

function ChemistrySource(mechanism::ReactionMechanism{NS, NR}; mu_mol = 1.0) where {NS, NR}
    return ChemistrySource{NS, NR}(mechanism, mu_mol)
end

# ============================================================
# Rate evaluation helpers
# ============================================================

"""
    _arrhenius_rate(rxn::ArrheniusReaction, T) -> k

Compute the Arrhenius rate constant `k = A * T^n * exp(-Ea / T)`.
"""
@inline function _arrhenius_rate(rxn::ArrheniusReaction, T)
    T_safe = max(T, 1.0e-30)
    return rxn.A * T_safe^rxn.n * exp(-rxn.Ea / T_safe)
end

"""
    _reaction_rate(rxn::ArrheniusReaction{NS}, k, ρ, Y::NTuple{NS}) -> R

Compute the mass-specific reaction rate `R = k * Π_j (ρ Y_j)^{ν'_j}`.
"""
@inline function _reaction_rate(rxn::ArrheniusReaction{NS}, k, ρ, Y::NTuple{NS}) where {NS}
    R = k
    for j in 1:NS
        if rxn.nu_reactant[j] > 0.0
            conc = ρ * max(Y[j], 0.0)
            if rxn.nu_reactant[j] == 1.0
                R *= conc
            else
                R *= conc^rxn.nu_reactant[j]
            end
        end
    end
    return R
end

# ============================================================
# 1D evaluate_stiff_source
# ============================================================

@inline function evaluate_stiff_source(
        src::ChemistrySource{NS, NR},
        law::ReactiveEulerEquations{1, NS},
        w::SVector{NV}, u::SVector{NV},
    ) where {NS, NR, NV}
    ρ = w[1]
    P = w[3]
    T = P / max(ρ, 1.0e-30) * src.mu_mol

    Y = species_mass_fractions(law, w)

    # Accumulate species production rates and energy source
    omega = ntuple(_ -> zero(ρ), Val(NS))
    S_E = zero(ρ)

    for i in 1:NR
        rxn = src.mechanism.reactions[i]
        k = _arrhenius_rate(rxn, T)
        R = _reaction_rate(rxn, k, ρ, Y)

        # Species production: ω_k += (ν"_k - ν'_k) * R
        omega = ntuple(
            j -> omega[j] + (rxn.nu_product[j] - rxn.nu_reactant[j]) * R,
            Val(NS),
        )

        # Energy source: -q * R (heat release is positive for exothermic)
        S_E -= rxn.heat_release * R
    end

    # Source vector: [0, 0, S_E, ω₁, …, ω_NS]
    return SVector(zero(ρ), zero(ρ), S_E, omega...)
end

# ============================================================
# 2D evaluate_stiff_source
# ============================================================

@inline function evaluate_stiff_source(
        src::ChemistrySource{NS, NR},
        law::ReactiveEulerEquations{2, NS},
        w::SVector{NV}, u::SVector{NV},
    ) where {NS, NR, NV}
    ρ = w[1]
    P = w[4]
    T = P / max(ρ, 1.0e-30) * src.mu_mol

    Y = species_mass_fractions(law, w)

    omega = ntuple(_ -> zero(ρ), Val(NS))
    S_E = zero(ρ)

    for i in 1:NR
        rxn = src.mechanism.reactions[i]
        k = _arrhenius_rate(rxn, T)
        R = _reaction_rate(rxn, k, ρ, Y)

        omega = ntuple(
            j -> omega[j] + (rxn.nu_product[j] - rxn.nu_reactant[j]) * R,
            Val(NS),
        )
        S_E -= rxn.heat_release * R
    end

    # Source vector: [0, 0, 0, S_E, ω₁, …, ω_NS]
    return SVector(zero(ρ), zero(ρ), zero(ρ), S_E, omega...)
end

# ============================================================
# Jacobian — diagonal approximation
# ============================================================

@inline function stiff_source_jacobian(
        src::ChemistrySource{NS, NR},
        law::ReactiveEulerEquations{1, NS},
        w::SVector{NV}, u::SVector{NV},
    ) where {NS, NR, NV}
    # Diagonal approximation: ∂S_k/∂U_k ≈ S_k / U_k for species,
    # ∂S_E/∂E ≈ S_E / E for energy.
    S = evaluate_stiff_source(src, law, w, u)
    z = zero(eltype(u))
    return _diagonal_jacobian(S, u, z, Val(3 + NS), Val(2))
end

@inline function stiff_source_jacobian(
        src::ChemistrySource{NS, NR},
        law::ReactiveEulerEquations{2, NS},
        w::SVector{NV}, u::SVector{NV},
    ) where {NS, NR, NV}
    S = evaluate_stiff_source(src, law, w, u)
    z = zero(eltype(u))
    return _diagonal_jacobian(S, u, z, Val(4 + NS), Val(3))
end

"""
    _diagonal_jacobian(S, u, z, ::Val{N}, ::Val{NZ}) -> SMatrix{N,N}

Build a diagonal Jacobian SMatrix where the first `NZ` diagonal entries
are zero (mass, momentum) and the rest use `S[i]/u[i]`.
"""
@inline function _diagonal_jacobian(
        S::SVector{N}, u::SVector{N}, z, ::Val{N}, ::Val{NZ},
    ) where {N, NZ}
    # Column-major: entry (row, col) is at index (col-1)*N + row
    mat_entries = ntuple(Val(N * N)) do idx
        row = mod1(idx, N)
        col = (idx - 1) ÷ N + 1
        if row == col && row > NZ
            abs(u[row]) > eps(typeof(z)) ? S[row] / u[row] : z
        else
            z
        end
    end
    return StaticArrays.SMatrix{N, N}(mat_entries...)
end
