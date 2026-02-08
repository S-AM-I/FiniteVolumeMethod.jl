# ============================================================
# 3D Euler Equations
# ============================================================
#
# Conserved variables: U = [ρ, ρvx, ρvy, ρvz, E]
# Primitive variables: W = [ρ, vx, vy, vz, P]
#
# Fluxes:
#   Fx = [ρvx, ρvx²+P, ρvx·vy, ρvx·vz, (E+P)vx]
#   Fy = [ρvy, ρvx·vy, ρvy²+P, ρvy·vz, (E+P)vy]
#   Fz = [ρvz, ρvx·vz, ρvy·vz, ρvz²+P, (E+P)vz]

nvariables(::EulerEquations{3}) = 5

"""
    total_energy(eos::IdealGasEOS, ρ, vx, vy, vz, P) -> E

Compute the total energy density `E = P/(γ-1) + ½ρ(vx² + vy² + vz²)` for 3D.
"""
@inline function total_energy(eos::IdealGasEOS, ρ, vx, vy, vz, P)
    return P / (eos.gamma - 1) + 0.5 * ρ * (vx^2 + vy^2 + vz^2)
end

"""
    total_energy(eos::StiffenedGasEOS, ρ, vx, vy, vz, P) -> E

Compute the total energy density for 3D stiffened gas:
`E = (P + γP∞)/(γ-1) + ½ρ(vx² + vy² + vz²)`
"""
@inline function total_energy(eos::StiffenedGasEOS, ρ, vx, vy, vz, P)
    return (P + eos.gamma * eos.P_inf) / (eos.gamma - 1) + 0.5 * ρ * (vx^2 + vy^2 + vz^2)
end

"""
    conserved_to_primitive(law::EulerEquations{3}, u::SVector{5}) -> SVector{5}

Convert 3D conserved `[ρ, ρvx, ρvy, ρvz, E]` to primitive `[ρ, vx, vy, vz, P]`.
"""
@inline function conserved_to_primitive(law::EulerEquations{3}, u::SVector{5})
    ρ = u[1]
    vx = u[2] / ρ
    vy = u[3] / ρ
    vz = u[4] / ρ
    E = u[5]
    ε = (E - 0.5 * ρ * (vx^2 + vy^2 + vz^2)) / ρ
    P = pressure(law.eos, ρ, ε)
    return SVector(ρ, vx, vy, vz, P)
end

"""
    primitive_to_conserved(law::EulerEquations{3}, w::SVector{5}) -> SVector{5}

Convert 3D primitive `[ρ, vx, vy, vz, P]` to conserved `[ρ, ρvx, ρvy, ρvz, E]`.
"""
@inline function primitive_to_conserved(law::EulerEquations{3}, w::SVector{5})
    ρ, vx, vy, vz, P = w
    E = total_energy(law.eos, ρ, vx, vy, vz, P)
    return SVector(ρ, ρ * vx, ρ * vy, ρ * vz, E)
end

"""
    physical_flux(law::EulerEquations{3}, w::SVector{5}, dir::Int) -> SVector{5}

Compute the 3D Euler flux in direction `dir` (1=x, 2=y, 3=z) from primitive variables
`[ρ, vx, vy, vz, P]`.
"""
@inline function physical_flux(law::EulerEquations{3}, w::SVector{5}, dir::Int)
    ρ, vx, vy, vz, P = w
    E = total_energy(law.eos, ρ, vx, vy, vz, P)
    if dir == 1
        return SVector(ρ * vx, ρ * vx^2 + P, ρ * vx * vy, ρ * vx * vz, (E + P) * vx)
    elseif dir == 2
        return SVector(ρ * vy, ρ * vx * vy, ρ * vy^2 + P, ρ * vy * vz, (E + P) * vy)
    else  # dir == 3
        return SVector(ρ * vz, ρ * vx * vz, ρ * vy * vz, ρ * vz^2 + P, (E + P) * vz)
    end
end

"""
    max_wave_speed(law::EulerEquations{3}, w::SVector{5}, dir::Int) -> Real

Maximum wave speed in direction `dir` from primitive variables.
"""
@inline function max_wave_speed(law::EulerEquations{3}, w::SVector{5}, dir::Int)
    ρ, vx, vy, vz, P = w
    c = sound_speed(law.eos, ρ, P)
    if dir == 1
        v_n = vx
    elseif dir == 2
        v_n = vy
    else
        v_n = vz
    end
    return abs(v_n) + c
end

"""
    wave_speeds(law::EulerEquations{3}, w::SVector{5}, dir::Int) -> (λ_min, λ_max)

Return the minimum and maximum wave speeds in direction `dir` from primitive variables.
"""
@inline function wave_speeds(law::EulerEquations{3}, w::SVector{5}, dir::Int)
    ρ, vx, vy, vz, P = w
    c = sound_speed(law.eos, ρ, P)
    if dir == 1
        v_n = vx
    elseif dir == 2
        v_n = vy
    else
        v_n = vz
    end
    return v_n - c, v_n + c
end
