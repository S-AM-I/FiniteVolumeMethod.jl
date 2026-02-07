"""
    IdealGasEOS{FT} <: AbstractEOS

Ideal gas (gamma-law) equation of state: `P = (γ - 1) ρ ε`.

# Fields
- `gamma::FT`: Adiabatic index (ratio of specific heats). Common values: 5/3 (monatomic), 7/5 (diatomic), 4/3 (relativistic).
"""
struct IdealGasEOS{FT} <: AbstractEOS
    gamma::FT
end

IdealGasEOS(; gamma = 1.4) = IdealGasEOS(gamma)

@inline function pressure(eos::IdealGasEOS, ρ, ε)
    return (eos.gamma - 1) * ρ * ε
end

@inline function sound_speed(eos::IdealGasEOS, ρ, P)
    return sqrt(eos.gamma * P / ρ)
end

@inline function internal_energy(eos::IdealGasEOS, ρ, P)
    return P / ((eos.gamma - 1) * ρ)
end

"""
    total_energy(eos::IdealGasEOS, ρ, v, P) -> E

Compute the total energy density `E = P/(γ-1) + ½ρv²` for 1D.
"""
@inline function total_energy(eos::IdealGasEOS, ρ, v, P)
    return P / (eos.gamma - 1) + 0.5 * ρ * v^2
end

"""
    total_energy(eos::IdealGasEOS, ρ, vx, vy, P) -> E

Compute the total energy density `E = P/(γ-1) + ½ρ(vx² + vy²)` for 2D.
"""
@inline function total_energy(eos::IdealGasEOS, ρ, vx, vy, P)
    return P / (eos.gamma - 1) + 0.5 * ρ * (vx^2 + vy^2)
end
