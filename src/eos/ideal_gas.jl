"""
    IdealGasEOS{FT} <: AbstractEOS

Ideal (gamma-law) equation of state: `P = (γ - 1) ρ ε`, where `ε` is the
specific internal energy. Equivalent to `P V = n R T` for a calorically
perfect gas with constant ratio of specific heats `γ = cₚ/cᵥ`.

Sound speed: `c = √(γ P / ρ)`.

# Fields
- `gamma::FT`: Adiabatic index. Common values: 5/3 (monatomic), 7/5 (diatomic),
  4/3 (relativistic gas or radiation-dominated).
"""
struct IdealGasEOS{FT} <: AbstractEOS
    gamma::FT
end

IdealGasEOS(; gamma = 1.4) = IdealGasEOS(gamma)

@inline function pressure(eos::IdealGasEOS, ρ, ε)
    return (eos.gamma - 1) * ρ * ε
end

@inline function sound_speed(eos::IdealGasEOS, ρ, P)
    return sqrt(eos.gamma * max(P, zero(P)) / max(ρ, 1.0e-30))
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
