"""
    StiffenedGasEOS{FT} <: AbstractEOS

Stiffened gas equation of state: `P = (γ - 1) ρ ε - γ P∞`.

This generalizes the ideal gas EOS with a stiffening pressure `P∞` that
models the repulsive molecular interactions in dense fluids (e.g., water).
Setting `P∞ = 0` recovers the ideal gas law.

# Fields
- `gamma::FT`: Adiabatic index.
- `P_inf::FT`: Stiffening pressure. Typical values: 0 (ideal gas), ~6.08e8 Pa (water).
"""
struct StiffenedGasEOS{FT} <: AbstractEOS
    gamma::FT
    P_inf::FT
end

StiffenedGasEOS(; gamma = 1.4, P_inf = 0.0) = StiffenedGasEOS(gamma, P_inf)

@inline function pressure(eos::StiffenedGasEOS, ρ, ε)
    return (eos.gamma - 1) * ρ * ε - eos.gamma * eos.P_inf
end

@inline function sound_speed(eos::StiffenedGasEOS, ρ, P)
    return sqrt(eos.gamma * (P + eos.P_inf) / ρ)
end

@inline function internal_energy(eos::StiffenedGasEOS, ρ, P)
    return (P + eos.gamma * eos.P_inf) / ((eos.gamma - 1) * ρ)
end

"""
    total_energy(eos::StiffenedGasEOS, ρ, v, P) -> E

Compute the total energy density for 1D:
`E = (P + γP∞)/(γ-1) + ½ρv²`
"""
@inline function total_energy(eos::StiffenedGasEOS, ρ, v, P)
    return (P + eos.gamma * eos.P_inf) / (eos.gamma - 1) + 0.5 * ρ * v^2
end

"""
    total_energy(eos::StiffenedGasEOS, ρ, vx, vy, P) -> E

Compute the total energy density for 2D:
`E = (P + γP∞)/(γ-1) + ½ρ(vx² + vy²)`
"""
@inline function total_energy(eos::StiffenedGasEOS, ρ, vx, vy, P)
    return (P + eos.gamma * eos.P_inf) / (eos.gamma - 1) + 0.5 * ρ * (vx^2 + vy^2)
end
