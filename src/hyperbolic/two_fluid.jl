# ============================================================
# Two-Fluid Plasma Equations
# ============================================================
#
# Two independent Euler systems (ions + electrons) stacked together.
# Coupling between species (Lorentz force, collisions) is handled
# via source terms and operator splitting, not in the hyperbolic flux.
#
# 1D (6 variables):
#   Conserved: U = [ρ_i, ρ_i v_i, E_i, ρ_e, ρ_e v_e, E_e]
#   Primitive: W = [ρ_i, v_i, P_i, ρ_e, v_e, P_e]
#
# 2D (8 variables):
#   Conserved: U = [ρ_i, ρ_i vx_i, ρ_i vy_i, E_i,
#                   ρ_e, ρ_e vx_e, ρ_e vy_e, E_e]
#   Primitive: W = [ρ_i, vx_i, vy_i, P_i,
#                   ρ_e, vx_e, vy_e, P_e]

"""
    TwoFluidEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}

Two-fluid plasma equations in `Dim` spatial dimensions.

Each species (ion and electron) is governed by an independent Euler system.
The hyperbolic fluxes are decoupled; coupling through electromagnetic
forces is applied via source terms (operator splitting).

## 1D (Dim=1, 6 variables)
- Conserved: `U = [ρ_i, ρ_i v_i, E_i, ρ_e, ρ_e v_e, E_e]`
- Primitive: `W = [ρ_i, v_i, P_i, ρ_e, v_e, P_e]`

## 2D (Dim=2, 8 variables)
- Conserved: `U = [ρ_i, ρ_i vx_i, ρ_i vy_i, E_i, ρ_e, ρ_e vx_e, ρ_e vy_e, E_e]`
- Primitive: `W = [ρ_i, vx_i, vy_i, P_i, ρ_e, vx_e, vy_e, P_e]`

# Fields
- `eos_ion::EOS`: Equation of state for ions.
- `eos_electron::EOS`: Equation of state for electrons.
- `mass_ratio::Float64`: Ion-to-electron mass ratio `m_i / m_e` (default 1836.0).
- `charge_ratio::Float64`: Ion-to-electron charge ratio `q_i / q_e` (default -1.0).
"""
struct TwoFluidEquations{Dim, EOS <: AbstractEOS} <: AbstractConservationLaw{Dim}
    eos_ion::EOS
    eos_electron::EOS
    mass_ratio::Float64
    charge_ratio::Float64
end

function TwoFluidEquations{Dim}(
        eos_ion::EOS, eos_electron::EOS;
        mass_ratio = 1836.0, charge_ratio = -1.0,
    ) where {Dim, EOS <: AbstractEOS}
    return TwoFluidEquations{Dim, EOS}(eos_ion, eos_electron, mass_ratio, charge_ratio)
end

nvariables(::TwoFluidEquations{1}) = 6
nvariables(::TwoFluidEquations{2}) = 8

# ============================================================
# Helper accessors — extract per-species primitive sub-vectors
# ============================================================

"""
    ion_primitive(law::TwoFluidEquations{1}, w::SVector{6}) -> SVector{3}

Extract ion primitive variables `[ρ_i, v_i, P_i]` from the full primitive vector.
"""
@inline ion_primitive(::TwoFluidEquations{1}, w::SVector{6}) = SVector(w[1], w[2], w[3])

"""
    electron_primitive(law::TwoFluidEquations{1}, w::SVector{6}) -> SVector{3}

Extract electron primitive variables `[ρ_e, v_e, P_e]` from the full primitive vector.
"""
@inline electron_primitive(::TwoFluidEquations{1}, w::SVector{6}) = SVector(w[4], w[5], w[6])

"""
    ion_primitive(law::TwoFluidEquations{2}, w::SVector{8}) -> SVector{4}

Extract ion primitive variables `[ρ_i, vx_i, vy_i, P_i]` from the full primitive vector.
"""
@inline ion_primitive(::TwoFluidEquations{2}, w::SVector{8}) = SVector(w[1], w[2], w[3], w[4])

"""
    electron_primitive(law::TwoFluidEquations{2}, w::SVector{8}) -> SVector{4}

Extract electron primitive variables `[ρ_e, vx_e, vy_e, P_e]` from the full primitive vector.
"""
@inline electron_primitive(::TwoFluidEquations{2}, w::SVector{8}) = SVector(w[5], w[6], w[7], w[8])

"""
    ion_conserved(law::TwoFluidEquations{1}, u::SVector{6}) -> SVector{3}

Extract ion conserved variables `[ρ_i, ρ_i v_i, E_i]` from the full conserved vector.
"""
@inline ion_conserved(::TwoFluidEquations{1}, u::SVector{6}) = SVector(u[1], u[2], u[3])

"""
    electron_conserved(law::TwoFluidEquations{1}, u::SVector{6}) -> SVector{3}

Extract electron conserved variables `[ρ_e, ρ_e v_e, E_e]` from the full conserved vector.
"""
@inline electron_conserved(::TwoFluidEquations{1}, u::SVector{6}) = SVector(u[4], u[5], u[6])

"""
    ion_conserved(law::TwoFluidEquations{2}, u::SVector{8}) -> SVector{4}

Extract ion conserved variables `[ρ_i, ρ_i vx_i, ρ_i vy_i, E_i]` from the full conserved vector.
"""
@inline ion_conserved(::TwoFluidEquations{2}, u::SVector{8}) = SVector(u[1], u[2], u[3], u[4])

"""
    electron_conserved(law::TwoFluidEquations{2}, u::SVector{8}) -> SVector{4}

Extract electron conserved variables `[ρ_e, ρ_e vx_e, ρ_e vy_e, E_e]` from the full conserved vector.
"""
@inline electron_conserved(::TwoFluidEquations{2}, u::SVector{8}) = SVector(u[5], u[6], u[7], u[8])

# ============================================================
# 1D Conserved <-> Primitive Conversion
# ============================================================

"""
    conserved_to_primitive(law::TwoFluidEquations{1}, u::SVector{6}) -> SVector{6}

Convert 1D conserved `[ρ_i, ρ_i v_i, E_i, ρ_e, ρ_e v_e, E_e]`
to primitive `[ρ_i, v_i, P_i, ρ_e, v_e, P_e]`.
"""
@inline function conserved_to_primitive(law::TwoFluidEquations{1}, u::SVector{6})
    # Ion species
    ρ_i = u[1]
    v_i = u[2] / ρ_i
    E_i = u[3]
    ε_i = (E_i - 0.5 * ρ_i * v_i^2) / ρ_i
    P_i = pressure(law.eos_ion, ρ_i, ε_i)

    # Electron species
    ρ_e = u[4]
    v_e = u[5] / ρ_e
    E_e = u[6]
    ε_e = (E_e - 0.5 * ρ_e * v_e^2) / ρ_e
    P_e = pressure(law.eos_electron, ρ_e, ε_e)

    return SVector(ρ_i, v_i, P_i, ρ_e, v_e, P_e)
end

"""
    primitive_to_conserved(law::TwoFluidEquations{1}, w::SVector{6}) -> SVector{6}

Convert 1D primitive `[ρ_i, v_i, P_i, ρ_e, v_e, P_e]`
to conserved `[ρ_i, ρ_i v_i, E_i, ρ_e, ρ_e v_e, E_e]`.
"""
@inline function primitive_to_conserved(law::TwoFluidEquations{1}, w::SVector{6})
    ρ_i, v_i, P_i = w[1], w[2], w[3]
    E_i = total_energy(law.eos_ion, ρ_i, v_i, P_i)

    ρ_e, v_e, P_e = w[4], w[5], w[6]
    E_e = total_energy(law.eos_electron, ρ_e, v_e, P_e)

    return SVector(ρ_i, ρ_i * v_i, E_i, ρ_e, ρ_e * v_e, E_e)
end

# ============================================================
# 2D Conserved <-> Primitive Conversion
# ============================================================

"""
    conserved_to_primitive(law::TwoFluidEquations{2}, u::SVector{8}) -> SVector{8}

Convert 2D conserved `[ρ_i, ρ_i vx_i, ρ_i vy_i, E_i, ρ_e, ρ_e vx_e, ρ_e vy_e, E_e]`
to primitive `[ρ_i, vx_i, vy_i, P_i, ρ_e, vx_e, vy_e, P_e]`.
"""
@inline function conserved_to_primitive(law::TwoFluidEquations{2}, u::SVector{8})
    # Ion species
    ρ_i = u[1]
    vx_i = u[2] / ρ_i
    vy_i = u[3] / ρ_i
    E_i = u[4]
    ε_i = (E_i - 0.5 * ρ_i * (vx_i^2 + vy_i^2)) / ρ_i
    P_i = pressure(law.eos_ion, ρ_i, ε_i)

    # Electron species
    ρ_e = u[5]
    vx_e = u[6] / ρ_e
    vy_e = u[7] / ρ_e
    E_e = u[8]
    ε_e = (E_e - 0.5 * ρ_e * (vx_e^2 + vy_e^2)) / ρ_e
    P_e = pressure(law.eos_electron, ρ_e, ε_e)

    return SVector(ρ_i, vx_i, vy_i, P_i, ρ_e, vx_e, vy_e, P_e)
end

"""
    primitive_to_conserved(law::TwoFluidEquations{2}, w::SVector{8}) -> SVector{8}

Convert 2D primitive `[ρ_i, vx_i, vy_i, P_i, ρ_e, vx_e, vy_e, P_e]`
to conserved `[ρ_i, ρ_i vx_i, ρ_i vy_i, E_i, ρ_e, ρ_e vx_e, ρ_e vy_e, E_e]`.
"""
@inline function primitive_to_conserved(law::TwoFluidEquations{2}, w::SVector{8})
    ρ_i, vx_i, vy_i, P_i = w[1], w[2], w[3], w[4]
    E_i = total_energy(law.eos_ion, ρ_i, vx_i, vy_i, P_i)

    ρ_e, vx_e, vy_e, P_e = w[5], w[6], w[7], w[8]
    E_e = total_energy(law.eos_electron, ρ_e, vx_e, vy_e, P_e)

    return SVector(ρ_i, ρ_i * vx_i, ρ_i * vy_i, E_i, ρ_e, ρ_e * vx_e, ρ_e * vy_e, E_e)
end

# ============================================================
# 1D Physical Flux
# ============================================================

"""
    physical_flux(law::TwoFluidEquations{1}, w::SVector{6}, ::Int) -> SVector{6}

Compute the 1D physical flux from primitive variables `[ρ_i, v_i, P_i, ρ_e, v_e, P_e]`.
The flux is two independent Euler fluxes stacked together.
"""
@inline function physical_flux(law::TwoFluidEquations{1}, w::SVector{6}, ::Int)
    # Ion flux
    ρ_i, v_i, P_i = w[1], w[2], w[3]
    E_i = total_energy(law.eos_ion, ρ_i, v_i, P_i)
    f1 = ρ_i * v_i
    f2 = ρ_i * v_i^2 + P_i
    f3 = (E_i + P_i) * v_i

    # Electron flux
    ρ_e, v_e, P_e = w[4], w[5], w[6]
    E_e = total_energy(law.eos_electron, ρ_e, v_e, P_e)
    f4 = ρ_e * v_e
    f5 = ρ_e * v_e^2 + P_e
    f6 = (E_e + P_e) * v_e

    return SVector(f1, f2, f3, f4, f5, f6)
end

# ============================================================
# 2D Physical Flux
# ============================================================

"""
    physical_flux(law::TwoFluidEquations{2}, w::SVector{8}, dir::Int) -> SVector{8}

Compute the 2D physical flux in direction `dir` (1=x, 2=y) from primitive variables
`[ρ_i, vx_i, vy_i, P_i, ρ_e, vx_e, vy_e, P_e]`.
The flux is two independent Euler fluxes stacked together.
"""
@inline function physical_flux(law::TwoFluidEquations{2}, w::SVector{8}, dir::Int)
    # Ion species
    ρ_i, vx_i, vy_i, P_i = w[1], w[2], w[3], w[4]
    E_i = total_energy(law.eos_ion, ρ_i, vx_i, vy_i, P_i)

    # Electron species
    ρ_e, vx_e, vy_e, P_e = w[5], w[6], w[7], w[8]
    E_e = total_energy(law.eos_electron, ρ_e, vx_e, vy_e, P_e)

    if dir == 1  # x-flux
        return SVector(
            ρ_i * vx_i,
            ρ_i * vx_i^2 + P_i,
            ρ_i * vx_i * vy_i,
            (E_i + P_i) * vx_i,
            ρ_e * vx_e,
            ρ_e * vx_e^2 + P_e,
            ρ_e * vx_e * vy_e,
            (E_e + P_e) * vx_e,
        )
    else  # y-flux (dir == 2)
        return SVector(
            ρ_i * vy_i,
            ρ_i * vx_i * vy_i,
            ρ_i * vy_i^2 + P_i,
            (E_i + P_i) * vy_i,
            ρ_e * vy_e,
            ρ_e * vx_e * vy_e,
            ρ_e * vy_e^2 + P_e,
            (E_e + P_e) * vy_e,
        )
    end
end

# ============================================================
# Wave Speeds
# ============================================================

"""
    max_wave_speed(law::TwoFluidEquations{1}, w::SVector{6}, ::Int) -> Real

Maximum wave speed across both species: `max(|v_i| + c_i, |v_e| + c_e)`.
"""
@inline function max_wave_speed(law::TwoFluidEquations{1}, w::SVector{6}, ::Int)
    ρ_i, v_i, P_i = w[1], w[2], w[3]
    c_i = sound_speed(law.eos_ion, ρ_i, P_i)

    ρ_e, v_e, P_e = w[4], w[5], w[6]
    c_e = sound_speed(law.eos_electron, ρ_e, P_e)

    return max(abs(v_i) + c_i, abs(v_e) + c_e)
end

"""
    max_wave_speed(law::TwoFluidEquations{2}, w::SVector{8}, dir::Int) -> Real

Maximum wave speed in direction `dir` across both species.
"""
@inline function max_wave_speed(law::TwoFluidEquations{2}, w::SVector{8}, dir::Int)
    ρ_i, vx_i, vy_i, P_i = w[1], w[2], w[3], w[4]
    c_i = sound_speed(law.eos_ion, ρ_i, P_i)
    vn_i = dir == 1 ? vx_i : vy_i

    ρ_e, vx_e, vy_e, P_e = w[5], w[6], w[7], w[8]
    c_e = sound_speed(law.eos_electron, ρ_e, P_e)
    vn_e = dir == 1 ? vx_e : vy_e

    return max(abs(vn_i) + c_i, abs(vn_e) + c_e)
end

"""
    wave_speeds(law::TwoFluidEquations{1}, w::SVector{6}, ::Int) -> (λ_min, λ_max)

Return the envelope of wave speeds across both species:
`(min(v_i - c_i, v_e - c_e), max(v_i + c_i, v_e + c_e))`.
"""
@inline function wave_speeds(law::TwoFluidEquations{1}, w::SVector{6}, ::Int)
    ρ_i, v_i, P_i = w[1], w[2], w[3]
    c_i = sound_speed(law.eos_ion, ρ_i, P_i)

    ρ_e, v_e, P_e = w[4], w[5], w[6]
    c_e = sound_speed(law.eos_electron, ρ_e, P_e)

    λ_min = min(v_i - c_i, v_e - c_e)
    λ_max = max(v_i + c_i, v_e + c_e)
    return λ_min, λ_max
end

"""
    wave_speeds(law::TwoFluidEquations{2}, w::SVector{8}, dir::Int) -> (λ_min, λ_max)

Return the envelope of wave speeds in direction `dir` across both species.
"""
@inline function wave_speeds(law::TwoFluidEquations{2}, w::SVector{8}, dir::Int)
    ρ_i, vx_i, vy_i, P_i = w[1], w[2], w[3], w[4]
    c_i = sound_speed(law.eos_ion, ρ_i, P_i)
    vn_i = dir == 1 ? vx_i : vy_i

    ρ_e, vx_e, vy_e, P_e = w[5], w[6], w[7], w[8]
    c_e = sound_speed(law.eos_electron, ρ_e, P_e)
    vn_e = dir == 1 ? vx_e : vy_e

    λ_min = min(vn_i - c_i, vn_e - c_e)
    λ_max = max(vn_i + c_i, vn_e + c_e)
    return λ_min, λ_max
end

# ============================================================
# Lorentz Force Source Terms (for operator splitting)
# ============================================================

"""
    lorentz_source_1d(law::TwoFluidEquations{1}, w::SVector{6},
                      Bx, By, Bz, Ex) -> SVector{6}

Compute the Lorentz force source term for both species in 1D.

For each species `s` with number density `n_s = ρ_s / m_s`,
charge `q_s`, and velocity `v_s`:
  `F_s = n_s q_s (E + v_s × B)`

The charge-to-mass ratios are determined by `law.mass_ratio` and
`law.charge_ratio`. Ions are taken as reference with `q_i/m_i = 1`;
electrons have `q_e/m_e = charge_ratio / mass_ratio⁻¹ = charge_ratio * mass_ratio`.

Returns source `S = [0, F_ix, F_ix v_i, 0, F_ex, F_ex v_e]` where the
momentum and energy sources for each species are the Lorentz force
and its work, respectively.

# Arguments
- `w`: Primitive variables `[ρ_i, v_i, P_i, ρ_e, v_e, P_e]`.
- `Bx, By, Bz`: Magnetic field components.
- `Ex`: Electric field x-component.
"""
@inline function lorentz_source_1d(
        law::TwoFluidEquations{1}, w::SVector{6},
        Bx, By, Bz, Ex,
    )
    ρ_i, v_i = w[1], w[2]
    ρ_e, v_e = w[4], w[5]

    # Charge-to-mass ratios (normalised so q_i/m_i = 1)
    qm_i = 1.0
    qm_e = law.charge_ratio * law.mass_ratio

    # Lorentz force per unit volume: ρ (q/m) (E + v × B)
    # In 1D with v = (v, 0, 0), v × B = (0, v Bz, -v By)
    # x-component of force: ρ (q/m) Ex  (the cross product has no x-component
    # for v along x, but E contributes)
    # Actually: v × B for v = (vx,0,0), B = (Bx,By,Bz) gives
    #   (0*Bz - 0*By, 0*Bx - vx*Bz, vx*By - 0*Bx) = (0, -vx Bz, vx By)
    # So x-component of E + v×B = Ex + 0 = Ex
    # Full force on species: F_x = ρ (q/m) Ex
    # Energy source: F_x * v_x = ρ (q/m) Ex * v

    F_ix = ρ_i * qm_i * Ex
    W_i = F_ix * v_i

    F_ex = ρ_e * qm_e * Ex
    W_e = F_ex * v_e

    return SVector(zero(ρ_i), F_ix, W_i, zero(ρ_e), F_ex, W_e)
end

"""
    lorentz_source_2d(law::TwoFluidEquations{2}, w::SVector{8},
                      Bx, By, Bz, Ex, Ey) -> SVector{8}

Compute the Lorentz force source term for both species in 2D.

For each species `s` with velocity `(vx_s, vy_s)`:
  `F_s = ρ_s (q_s/m_s) (E + v_s × B)`

where `v × B` for `v = (vx, vy, 0)` and `B = (Bx, By, Bz)`:
  `(v × B)_x = vy Bz`
  `(v × B)_y = -vx Bz`

Returns source `S = [0, F_ix, F_iy, F_i·v_i, 0, F_ex, F_ey, F_e·v_e]`.

# Arguments
- `w`: Primitive variables `[ρ_i, vx_i, vy_i, P_i, ρ_e, vx_e, vy_e, P_e]`.
- `Bx, By, Bz`: Magnetic field components.
- `Ex, Ey`: Electric field components.
"""
@inline function lorentz_source_2d(
        law::TwoFluidEquations{2}, w::SVector{8},
        Bx, By, Bz, Ex, Ey,
    )
    ρ_i, vx_i, vy_i = w[1], w[2], w[3]
    ρ_e, vx_e, vy_e = w[5], w[6], w[7]

    # Charge-to-mass ratios (normalised so q_i/m_i = 1)
    qm_i = 1.0
    qm_e = law.charge_ratio * law.mass_ratio

    # v × B for v = (vx, vy, 0), B = (Bx, By, Bz):
    #   x-component: vy * Bz - 0 * By = vy Bz
    #   y-component: 0 * Bx - vx * Bz = -vx Bz

    # Ion Lorentz force per unit volume
    F_ix = ρ_i * qm_i * (Ex + vy_i * Bz)
    F_iy = ρ_i * qm_i * (Ey - vx_i * Bz)
    W_i = F_ix * vx_i + F_iy * vy_i

    # Electron Lorentz force per unit volume
    F_ex = ρ_e * qm_e * (Ex + vy_e * Bz)
    F_ey = ρ_e * qm_e * (Ey - vx_e * Bz)
    W_e = F_ex * vx_e + F_ey * vy_e

    return SVector(zero(ρ_i), F_ix, F_iy, W_i, zero(ρ_e), F_ex, F_ey, W_e)
end
