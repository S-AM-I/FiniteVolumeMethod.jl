# ============================================================
# Reactive Euler Equations — Multi-Species Conservation Law
# ============================================================
#
# Extends the compressible Euler equations with NS passive species
# (mass fractions) for reactive flow simulations.
#
# 1D (3 + NS variables):
#   Conserved: U = [ρ, ρv, E, ρY₁, …, ρY_NS]
#   Primitive: W = [ρ, v, P, Y₁, …, Y_NS]
#
# 2D (4 + NS variables):
#   Conserved: U = [ρ, ρvx, ρvy, E, ρY₁, …, ρY_NS]
#   Primitive: W = [ρ, vx, vy, P, Y₁, …, Y_NS]
#
# Species do not affect the acoustic structure — wave speeds
# delegate to the underlying EulerEquations.

using StaticArrays: SVector

"""
    ReactiveEulerEquations{Dim, NSpecies, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}

The compressible Euler equations with `NSpecies` passive species
(mass fractions) for reactive flow simulations.

Species are advected with the flow velocity and do not affect
the acoustic wave structure. Chemical reactions are handled
via operator splitting (see `ChemistrySource`).

## 1D (Dim=1, 3 + NSpecies variables)
- Conserved: `U = [ρ, ρv, E, ρY₁, …, ρY_NS]`
- Primitive: `W = [ρ, v, P, Y₁, …, Y_NS]`

## 2D (Dim=2, 4 + NSpecies variables)
- Conserved: `U = [ρ, ρvx, ρvy, E, ρY₁, …, ρY_NS]`
- Primitive: `W = [ρ, vx, vy, P, Y₁, …, Y_NS]`

# Fields
- `euler::EulerEquations{Dim, EOS}`: The underlying Euler equations.
- `species_names::NTuple{NSpecies, Symbol}`: Names of species (e.g., `(:fuel, :product)`).
"""
struct ReactiveEulerEquations{Dim, NSpecies, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}
    euler::EulerEquations{Dim, EOS}
    species_names::NTuple{NSpecies, Symbol}
end

function ReactiveEulerEquations{Dim}(
        eos::EOS, species_names::NTuple{NS, Symbol},
    ) where {Dim, EOS <: AbstractEOS, NS}
    return ReactiveEulerEquations{Dim, NS, EOS}(EulerEquations{Dim}(eos), species_names)
end

nvariables(::ReactiveEulerEquations{1, NS}) where {NS} = 3 + NS
nvariables(::ReactiveEulerEquations{2, NS}) where {NS} = 4 + NS

# ============================================================
# Helper accessors
# ============================================================

"""
    euler_primitive(law::ReactiveEulerEquations{1}, w) -> SVector{3}

Extract the Euler primitive variables `[ρ, v, P]` from the full primitive vector.
"""
@inline function euler_primitive(::ReactiveEulerEquations{1, NS}, w::SVector{N}) where {NS, N}
    return SVector(w[1], w[2], w[3])
end

"""
    euler_primitive(law::ReactiveEulerEquations{2}, w) -> SVector{4}

Extract the Euler primitive variables `[ρ, vx, vy, P]` from the full primitive vector.
"""
@inline function euler_primitive(::ReactiveEulerEquations{2, NS}, w::SVector{N}) where {NS, N}
    return SVector(w[1], w[2], w[3], w[4])
end

"""
    euler_conserved(law::ReactiveEulerEquations{1}, u) -> SVector{3}

Extract the Euler conserved variables `[ρ, ρv, E]` from the full conserved vector.
"""
@inline function euler_conserved(::ReactiveEulerEquations{1, NS}, u::SVector{N}) where {NS, N}
    return SVector(u[1], u[2], u[3])
end

"""
    euler_conserved(law::ReactiveEulerEquations{2}, u) -> SVector{4}

Extract the Euler conserved variables `[ρ, ρvx, ρvy, E]` from the full conserved vector.
"""
@inline function euler_conserved(::ReactiveEulerEquations{2, NS}, u::SVector{N}) where {NS, N}
    return SVector(u[1], u[2], u[3], u[4])
end

"""
    species_mass_fractions(law::ReactiveEulerEquations{1}, w) -> NTuple{NS, Float64}

Extract the species mass fractions `(Y₁, …, Y_NS)` from the primitive vector.
"""
@inline function species_mass_fractions(::ReactiveEulerEquations{1, NS}, w::SVector{N}) where {NS, N}
    return ntuple(k -> w[3 + k], Val(NS))
end

"""
    species_mass_fractions(law::ReactiveEulerEquations{2}, w) -> NTuple{NS, Float64}

Extract the species mass fractions `(Y₁, …, Y_NS)` from the primitive vector.
"""
@inline function species_mass_fractions(::ReactiveEulerEquations{2, NS}, w::SVector{N}) where {NS, N}
    return ntuple(k -> w[4 + k], Val(NS))
end

"""
    species_partial_densities(law::ReactiveEulerEquations{1}, u) -> NTuple{NS, Float64}

Extract the species partial densities `(ρY₁, …, ρY_NS)` from the conserved vector.
"""
@inline function species_partial_densities(::ReactiveEulerEquations{1, NS}, u::SVector{N}) where {NS, N}
    return ntuple(k -> u[3 + k], Val(NS))
end

"""
    species_partial_densities(law::ReactiveEulerEquations{2}, u) -> NTuple{NS, Float64}

Extract the species partial densities `(ρY₁, …, ρY_NS)` from the conserved vector.
"""
@inline function species_partial_densities(::ReactiveEulerEquations{2, NS}, u::SVector{N}) where {NS, N}
    return ntuple(k -> u[4 + k], Val(NS))
end

"""
    temperature(law::ReactiveEulerEquations, ρ, P)

Compute the temperature from the ideal gas law: `T = P / (ρ (γ-1) cᵥ)`.
For a gamma-law gas with non-dimensional specific heats, `T ≈ P / ρ`.
"""
@inline function temperature(law::ReactiveEulerEquations, ρ, P)
    return P / max(ρ, 1.0e-30)
end

# ============================================================
# 1D Conserved <-> Primitive Conversion
# ============================================================

"""
    conserved_to_primitive(law::ReactiveEulerEquations{1,NS}, u::SVector{3+NS}) -> SVector{3+NS}

Convert 1D conserved `[ρ, ρv, E, ρY₁, …]` to primitive `[ρ, v, P, Y₁, …]`.
"""
@inline function conserved_to_primitive(
        law::ReactiveEulerEquations{1, NS}, u::SVector{N},
    ) where {NS, N}
    ρ = u[1]
    v = u[2] / ρ
    E = u[3]
    ε = (E - 0.5 * ρ * v^2) / ρ
    P = pressure(law.euler.eos, ρ, ε)
    # Species mass fractions: Y_k = ρY_k / ρ
    species = ntuple(k -> u[3 + k] / ρ, Val(NS))
    return SVector(ρ, v, P, species...)
end

"""
    primitive_to_conserved(law::ReactiveEulerEquations{1,NS}, w::SVector{3+NS}) -> SVector{3+NS}

Convert 1D primitive `[ρ, v, P, Y₁, …]` to conserved `[ρ, ρv, E, ρY₁, …]`.
"""
@inline function primitive_to_conserved(
        law::ReactiveEulerEquations{1, NS}, w::SVector{N},
    ) where {NS, N}
    ρ, v, P = w[1], w[2], w[3]
    E = total_energy(law.euler.eos, ρ, v, P)
    # Species partial densities: ρY_k = ρ * Y_k
    species = ntuple(k -> ρ * w[3 + k], Val(NS))
    return SVector(ρ, ρ * v, E, species...)
end

# ============================================================
# 2D Conserved <-> Primitive Conversion
# ============================================================

"""
    conserved_to_primitive(law::ReactiveEulerEquations{2,NS}, u::SVector{4+NS}) -> SVector{4+NS}

Convert 2D conserved `[ρ, ρvx, ρvy, E, ρY₁, …]` to primitive `[ρ, vx, vy, P, Y₁, …]`.
"""
@inline function conserved_to_primitive(
        law::ReactiveEulerEquations{2, NS}, u::SVector{N},
    ) where {NS, N}
    ρ = u[1]
    vx = u[2] / ρ
    vy = u[3] / ρ
    E = u[4]
    ε = (E - 0.5 * ρ * (vx^2 + vy^2)) / ρ
    P = pressure(law.euler.eos, ρ, ε)
    species = ntuple(k -> u[4 + k] / ρ, Val(NS))
    return SVector(ρ, vx, vy, P, species...)
end

"""
    primitive_to_conserved(law::ReactiveEulerEquations{2,NS}, w::SVector{4+NS}) -> SVector{4+NS}

Convert 2D primitive `[ρ, vx, vy, P, Y₁, …]` to conserved `[ρ, ρvx, ρvy, E, ρY₁, …]`.
"""
@inline function primitive_to_conserved(
        law::ReactiveEulerEquations{2, NS}, w::SVector{N},
    ) where {NS, N}
    ρ, vx, vy, P = w[1], w[2], w[3], w[4]
    E = total_energy(law.euler.eos, ρ, vx, vy, P)
    species = ntuple(k -> ρ * w[4 + k], Val(NS))
    return SVector(ρ, ρ * vx, ρ * vy, E, species...)
end

# ============================================================
# 1D Physical Flux
# ============================================================

"""
    physical_flux(law::ReactiveEulerEquations{1,NS}, w::SVector{3+NS}, ::Int) -> SVector{3+NS}

Compute the 1D physical flux from primitive variables.
Euler part is identical to `EulerEquations`; species flux is `ρY_k * v`.
"""
@inline function physical_flux(
        law::ReactiveEulerEquations{1, NS}, w::SVector{N}, ::Int,
    ) where {NS, N}
    ρ, v, P = w[1], w[2], w[3]
    E = total_energy(law.euler.eos, ρ, v, P)
    # Species flux: ρY_k * v = ρ * Y_k * v
    species_flux = ntuple(k -> ρ * w[3 + k] * v, Val(NS))
    return SVector(ρ * v, ρ * v^2 + P, (E + P) * v, species_flux...)
end

# ============================================================
# 2D Physical Flux
# ============================================================

"""
    physical_flux(law::ReactiveEulerEquations{2,NS}, w::SVector{4+NS}, dir::Int) -> SVector{4+NS}

Compute the 2D physical flux in direction `dir` (1=x, 2=y) from primitive variables.
"""
@inline function physical_flux(
        law::ReactiveEulerEquations{2, NS}, w::SVector{N}, dir::Int,
    ) where {NS, N}
    ρ, vx, vy, P = w[1], w[2], w[3], w[4]
    E = total_energy(law.euler.eos, ρ, vx, vy, P)
    if dir == 1
        species_flux = ntuple(k -> ρ * w[4 + k] * vx, Val(NS))
        return SVector(ρ * vx, ρ * vx^2 + P, ρ * vx * vy, (E + P) * vx, species_flux...)
    else
        species_flux = ntuple(k -> ρ * w[4 + k] * vy, Val(NS))
        return SVector(ρ * vy, ρ * vx * vy, ρ * vy^2 + P, (E + P) * vy, species_flux...)
    end
end

# ============================================================
# Wave Speeds — delegate to underlying Euler
# ============================================================

"""
    max_wave_speed(law::ReactiveEulerEquations{1}, w, ::Int) -> Real

Maximum wave speed `|v| + c` — species do not affect acoustics.
"""
@inline function max_wave_speed(
        law::ReactiveEulerEquations{1, NS}, w::SVector{N}, dir::Int,
    ) where {NS, N}
    w_euler = euler_primitive(law, w)
    return max_wave_speed(law.euler, w_euler, dir)
end

"""
    max_wave_speed(law::ReactiveEulerEquations{2}, w, dir::Int) -> Real

Maximum wave speed in direction `dir` — species do not affect acoustics.
"""
@inline function max_wave_speed(
        law::ReactiveEulerEquations{2, NS}, w::SVector{N}, dir::Int,
    ) where {NS, N}
    w_euler = euler_primitive(law, w)
    return max_wave_speed(law.euler, w_euler, dir)
end

"""
    wave_speeds(law::ReactiveEulerEquations{1}, w, ::Int) -> (λ_min, λ_max)

Wave speed bounds — delegate to Euler.
"""
@inline function wave_speeds(
        law::ReactiveEulerEquations{1, NS}, w::SVector{N}, dir::Int,
    ) where {NS, N}
    w_euler = euler_primitive(law, w)
    return wave_speeds(law.euler, w_euler, dir)
end

"""
    wave_speeds(law::ReactiveEulerEquations{2}, w, dir::Int) -> (λ_min, λ_max)

Wave speed bounds — delegate to Euler.
"""
@inline function wave_speeds(
        law::ReactiveEulerEquations{2, NS}, w::SVector{N}, dir::Int,
    ) where {NS, N}
    w_euler = euler_primitive(law, w)
    return wave_speeds(law.euler, w_euler, dir)
end

# ============================================================
# HLLC Riemann Solver — species-aware extension
# ============================================================
# Species passively follow the contact wave in the star state:
#   (ρY_k)* = ρ_K * Y_k * (S_K - v_K) / (S_K - S*)
# This is identical to the density star state times the upwind Y_k.

"""
    solve_riemann(::HLLCSolver, law::ReactiveEulerEquations{1,NS}, wL, wR, dir) -> SVector

Species-aware 1D HLLC solver. Wave speeds from Euler part; species
follow the contact discontinuity.
"""
function solve_riemann(
        ::HLLCSolver, law::ReactiveEulerEquations{1, NS},
        wL::SVector{NV}, wR::SVector{NV}, dir::Int,
    ) where {NS, NV}
    ρL, vL, PL = wL[1], wL[2], wL[3]
    ρR, vR, PR = wR[1], wR[2], wR[3]
    γ = law.euler.eos.gamma

    # Conserved states
    uL = primitive_to_conserved(law, wL)
    uR = primitive_to_conserved(law, wR)

    # Sound speeds
    cL = sound_speed(law.euler.eos, ρL, PL)
    cR = sound_speed(law.euler.eos, ρR, PR)

    # Wave speed estimates (PVRS)
    ρ_avg = 0.5 * (ρL + ρR)
    c_avg = 0.5 * (cL + cR)
    P_pvrs = 0.5 * (PL + PR) - 0.5 * (vR - vL) * ρ_avg * c_avg
    P_star = max(P_pvrs, zero(P_pvrs))

    qL = _pressure_wave_factor(P_star, PL, γ)
    qR = _pressure_wave_factor(P_star, PR, γ)

    SL = vL - cL * qL
    SR = vR + cR * qR

    # Contact wave speed
    S_star = (PR - PL + ρL * vL * (SL - vL) - ρR * vR * (SR - vR)) /
        (ρL * (SL - vL) - ρR * (SR - vR))

    if SL >= zero(SL)
        return physical_flux(law, wL, dir)
    elseif SR <= zero(SR)
        return physical_flux(law, wR, dir)
    elseif S_star >= zero(S_star)
        # Star-left region
        fL = physical_flux(law, wL, dir)
        factor = ρL * (SL - vL) / (SL - S_star)
        E_star = uL[3] / ρL + (S_star - vL) * (S_star + PL / (ρL * (SL - vL)))
        euler_star = SVector(factor, factor * S_star, factor * E_star)
        species_star = ntuple(k -> factor * wL[3 + k], Val(NS))
        u_star_L = SVector(euler_star..., species_star...)
        return fL + SL * (u_star_L - uL)
    else
        # Star-right region
        fR = physical_flux(law, wR, dir)
        factor = ρR * (SR - vR) / (SR - S_star)
        E_star = uR[3] / ρR + (S_star - vR) * (S_star + PR / (ρR * (SR - vR)))
        euler_star = SVector(factor, factor * S_star, factor * E_star)
        species_star = ntuple(k -> factor * wR[3 + k], Val(NS))
        u_star_R = SVector(euler_star..., species_star...)
        return fR + SR * (u_star_R - uR)
    end
end

"""
    solve_riemann(::HLLCSolver, law::ReactiveEulerEquations{2,NS}, wL, wR, dir) -> SVector

Species-aware 2D HLLC solver. Wave speeds from Euler part; species
follow the contact discontinuity.
"""
function solve_riemann(
        ::HLLCSolver, law::ReactiveEulerEquations{2, NS},
        wL::SVector{NV}, wR::SVector{NV}, dir::Int,
    ) where {NS, NV}
    ρL, vxL, vyL, PL = wL[1], wL[2], wL[3], wL[4]
    ρR, vxR, vyR, PR = wR[1], wR[2], wR[3], wR[4]
    γ = law.euler.eos.gamma

    # Normal and tangential velocities
    if dir == 1
        vnL, vtL = vxL, vyL
        vnR, vtR = vxR, vyR
    else
        vnL, vtL = vyL, vxL
        vnR, vtR = vyR, vxR
    end

    # Conserved states
    uL = primitive_to_conserved(law, wL)
    uR = primitive_to_conserved(law, wR)
    EL = uL[4]
    ER = uR[4]

    # Sound speeds
    cL = sound_speed(law.euler.eos, ρL, PL)
    cR = sound_speed(law.euler.eos, ρR, PR)

    # Wave speed estimates (PVRS)
    ρ_avg = 0.5 * (ρL + ρR)
    c_avg = 0.5 * (cL + cR)
    P_pvrs = 0.5 * (PL + PR) - 0.5 * (vnR - vnL) * ρ_avg * c_avg
    P_star = max(P_pvrs, zero(P_pvrs))

    qL = _pressure_wave_factor(P_star, PL, γ)
    qR = _pressure_wave_factor(P_star, PR, γ)

    SL = vnL - cL * qL
    SR = vnR + cR * qR

    # Contact wave speed
    S_star = (PR - PL + ρL * vnL * (SL - vnL) - ρR * vnR * (SR - vnR)) /
        (ρL * (SL - vnL) - ρR * (SR - vnR))

    if SL >= zero(SL)
        return physical_flux(law, wL, dir)
    elseif SR <= zero(SR)
        return physical_flux(law, wR, dir)
    elseif S_star >= zero(S_star)
        # Star-left region
        fL = physical_flux(law, wL, dir)
        factor = ρL * (SL - vnL) / (SL - S_star)
        E_star = EL / ρL + (S_star - vnL) * (S_star + PL / (ρL * (SL - vnL)))
        if dir == 1
            euler_star = SVector(factor, factor * S_star, factor * vtL, factor * E_star)
        else
            euler_star = SVector(factor, factor * vtL, factor * S_star, factor * E_star)
        end
        species_star = ntuple(k -> factor * wL[4 + k], Val(NS))
        u_star_L = SVector(euler_star..., species_star...)
        return fL + SL * (u_star_L - uL)
    else
        # Star-right region
        fR = physical_flux(law, wR, dir)
        factor = ρR * (SR - vnR) / (SR - S_star)
        E_star = ER / ρR + (S_star - vnR) * (S_star + PR / (ρR * (SR - vnR)))
        if dir == 1
            euler_star = SVector(factor, factor * S_star, factor * vtR, factor * E_star)
        else
            euler_star = SVector(factor, factor * vtR, factor * S_star, factor * E_star)
        end
        species_star = ntuple(k -> factor * wR[4 + k], Val(NS))
        u_star_R = SVector(euler_star..., species_star...)
        return fR + SR * (u_star_R - uR)
    end
end
