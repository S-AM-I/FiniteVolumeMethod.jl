using StaticArrays: SVector

"""
    EulerEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}

The compressible Euler equations in `Dim` spatial dimensions.

## 1D (Dim=1)
Conserved variables: `U = [ρ, ρv, E]`
Primitive variables: `W = [ρ, v, P]`
Flux: `F = [ρv, ρv² + P, (E + P)v]`

## 2D (Dim=2)
Conserved variables: `U = [ρ, ρvx, ρvy, E]`
Primitive variables: `W = [ρ, vx, vy, P]`
Fluxes:
  `Fx = [ρvx, ρvx² + P, ρvx*vy, (E + P)vx]`
  `Fy = [ρvy, ρvx*vy, ρvy² + P, (E + P)vy]`

# Fields
- `eos::EOS`: Equation of state.
"""
struct EulerEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}
    eos::EOS
end

EulerEquations{Dim}(eos::EOS) where {Dim, EOS <: AbstractEOS} = EulerEquations{Dim, EOS}(eos)

nvariables(::EulerEquations{1}) = 3
nvariables(::EulerEquations{2}) = 4

# ============================================================
# 1D Euler
# ============================================================

"""
    conserved_to_primitive(law::EulerEquations{1}, u::SVector{3}) -> SVector{3}

Convert 1D conserved `[ρ, ρv, E]` to primitive `[ρ, v, P]`.
"""
@inline function conserved_to_primitive(law::EulerEquations{1}, u::SVector{3})
    ρ = u[1]
    v = u[2] / ρ
    E = u[3]
    ε = (E - 0.5 * ρ * v^2) / ρ
    P = pressure(law.eos, ρ, ε)
    return SVector(ρ, v, P)
end

"""
    primitive_to_conserved(law::EulerEquations{1}, w::SVector{3}) -> SVector{3}

Convert 1D primitive `[ρ, v, P]` to conserved `[ρ, ρv, E]`.
"""
@inline function primitive_to_conserved(law::EulerEquations{1}, w::SVector{3})
    ρ, v, P = w
    E = total_energy(law.eos, ρ, v, P)
    return SVector(ρ, ρ * v, E)
end

"""
    physical_flux(law::EulerEquations{1}, w::SVector{3}, ::Int) -> SVector{3}

Compute the 1D Euler flux from primitive variables `[ρ, v, P]`.
"""
@inline function physical_flux(law::EulerEquations{1}, w::SVector{3}, ::Int)
    ρ, v, P = w
    E = total_energy(law.eos, ρ, v, P)
    return SVector(ρ * v, ρ * v^2 + P, (E + P) * v)
end

"""
    max_wave_speed(law::EulerEquations{1}, w::SVector{3}, ::Int) -> Real

Maximum wave speed `|v| + c` from primitive variables.
"""
@inline function max_wave_speed(law::EulerEquations{1}, w::SVector{3}, ::Int)
    ρ, v, P = w
    c = sound_speed(law.eos, ρ, P)
    return abs(v) + c
end

"""
    wave_speeds(law::EulerEquations{1}, w::SVector{3}, ::Int) -> (λ_min, λ_max)

Return the minimum and maximum wave speeds from primitive variables.
"""
@inline function wave_speeds(law::EulerEquations{1}, w::SVector{3}, ::Int)
    ρ, v, P = w
    c = sound_speed(law.eos, ρ, P)
    return v - c, v + c
end

# ============================================================
# 2D Euler
# ============================================================

"""
    conserved_to_primitive(law::EulerEquations{2}, u::SVector{4}) -> SVector{4}

Convert 2D conserved `[ρ, ρvx, ρvy, E]` to primitive `[ρ, vx, vy, P]`.
"""
@inline function conserved_to_primitive(law::EulerEquations{2}, u::SVector{4})
    ρ = u[1]
    vx = u[2] / ρ
    vy = u[3] / ρ
    E = u[4]
    ε = (E - 0.5 * ρ * (vx^2 + vy^2)) / ρ
    P = pressure(law.eos, ρ, ε)
    return SVector(ρ, vx, vy, P)
end

"""
    primitive_to_conserved(law::EulerEquations{2}, w::SVector{4}) -> SVector{4}

Convert 2D primitive `[ρ, vx, vy, P]` to conserved `[ρ, ρvx, ρvy, E]`.
"""
@inline function primitive_to_conserved(law::EulerEquations{2}, w::SVector{4})
    ρ, vx, vy, P = w
    E = total_energy(law.eos, ρ, vx, vy, P)
    return SVector(ρ, ρ * vx, ρ * vy, E)
end

"""
    physical_flux(law::EulerEquations{2}, w::SVector{4}, dir::Int) -> SVector{4}

Compute the 2D Euler flux in direction `dir` (1=x, 2=y) from primitive variables `[ρ, vx, vy, P]`.
"""
@inline function physical_flux(law::EulerEquations{2}, w::SVector{4}, dir::Int)
    ρ, vx, vy, P = w
    E = total_energy(law.eos, ρ, vx, vy, P)
    if dir == 1
        return SVector(ρ * vx, ρ * vx^2 + P, ρ * vx * vy, (E + P) * vx)
    else
        return SVector(ρ * vy, ρ * vx * vy, ρ * vy^2 + P, (E + P) * vy)
    end
end

"""
    max_wave_speed(law::EulerEquations{2}, w::SVector{4}, dir::Int) -> Real

Maximum wave speed in direction `dir` from primitive variables.
"""
@inline function max_wave_speed(law::EulerEquations{2}, w::SVector{4}, dir::Int)
    ρ, vx, vy, P = w
    c = sound_speed(law.eos, ρ, P)
    v_n = dir == 1 ? vx : vy
    return abs(v_n) + c
end

"""
    wave_speeds(law::EulerEquations{2}, w::SVector{4}, dir::Int) -> (λ_min, λ_max)

Return the minimum and maximum wave speeds in direction `dir` from primitive variables.
"""
@inline function wave_speeds(law::EulerEquations{2}, w::SVector{4}, dir::Int)
    ρ, vx, vy, P = w
    c = sound_speed(law.eos, ρ, P)
    v_n = dir == 1 ? vx : vy
    return v_n - c, v_n + c
end
