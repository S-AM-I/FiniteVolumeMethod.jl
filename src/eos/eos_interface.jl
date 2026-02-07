"""
    AbstractEOS

Abstract supertype for equations of state.

All subtypes must implement:
- `pressure(eos, ρ, ε)`: Compute pressure from density and specific internal energy.
- `sound_speed(eos, ρ, P)`: Compute sound speed from density and pressure.
- `internal_energy(eos, ρ, P)`: Compute specific internal energy from density and pressure.
"""
abstract type AbstractEOS end

"""
    pressure(eos::AbstractEOS, ρ, ε)

Compute the thermodynamic pressure given density `ρ` and specific internal energy `ε`.
"""
function pressure end

"""
    sound_speed(eos::AbstractEOS, ρ, P)

Compute the adiabatic sound speed given density `ρ` and pressure `P`.
"""
function sound_speed end

"""
    internal_energy(eos::AbstractEOS, ρ, P)

Compute the specific internal energy given density `ρ` and pressure `P`.
"""
function internal_energy end
