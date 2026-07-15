"""
    HLLCSolver <: AbstractRiemannSolver

The HLLC (Harten-Lax-van Leer-Contact) approximate Riemann solver.

A 3-wave model that resolves:
- Left-going wave (speed `SL`)
- Contact discontinuity (speed `S*`)
- Right-going wave (speed `SR`)

Exactly resolves isolated contact discontinuities and shear waves, which
the 2-wave HLL solver smears. The standard choice for the Euler equations.

## References
- Toro, E. F., Spruce, M., & Speares, W. (1994). "Restoration of the contact
  surface in the HLL-Riemann solver." *Shock Waves*, 4(1), 25–34.
- Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid
  Dynamics*, 3rd ed., Springer. Chapter 10.
"""
struct HLLCSolver <: AbstractRiemannSolver end

# ============================================================
# 1D HLLC
# ============================================================

function solve_riemann(::HLLCSolver, law::EulerEquations{1}, wL::SVector{3}, wR::SVector{3}, dir::Int)
    ρL, vL, PL = wL
    ρR, vR, PR = wR
    γ = law.eos.gamma

    # Compute conserved states
    uL = primitive_to_conserved(law, wL)
    uR = primitive_to_conserved(law, wR)

    # Sound speeds
    cL = sound_speed(law.eos, ρL, PL)
    cR = sound_speed(law.eos, ρR, PR)

    # Wave speed estimates (Einfeldt/PVRS)
    ρ_avg = 0.5 * (ρL + ρR)
    c_avg = 0.5 * (cL + cR)
    P_pvrs = 0.5 * (PL + PR) - 0.5 * (vR - vL) * ρ_avg * c_avg
    P_star = max(P_pvrs, zero(P_pvrs))

    # Pressure-based wave speed corrections
    qL = _pressure_wave_factor(P_star, PL, γ)
    qR = _pressure_wave_factor(P_star, PR, γ)

    SL = vL - cL * qL
    SR = vR + cR * qR

    # Contact wave speed
    S_star = (PR - PL + ρL * vL * (SL - vL) - ρR * vR * (SR - vR)) /
        (ρL * (SL - vL) - ρR * (SR - vR))

    if SL >= zero(SL)
        # Left of all waves
        return physical_flux(law, wL, dir)
    elseif SR <= zero(SR)
        # Right of all waves
        return physical_flux(law, wR, dir)
    elseif S_star >= zero(S_star)
        # Star-left region
        fL = physical_flux(law, wL, dir)
        u_star_L = _hllc_star_state_1d(ρL, vL, uL[3], PL, SL, S_star)
        return fL + SL * (u_star_L - uL)
    else
        # Star-right region
        fR = physical_flux(law, wR, dir)
        u_star_R = _hllc_star_state_1d(ρR, vR, uR[3], PR, SR, S_star)
        return fR + SR * (u_star_R - uR)
    end
end

"""
    _pressure_wave_factor(P_star, P_K, γ) -> q

Compute the pressure-based wave speed factor for HLLC estimates.
If P_star > P_K (shock), q > 1; if P_star ≤ P_K (rarefaction), q = 1.
"""
@inline function _pressure_wave_factor(P_star, P_K, γ)
    if P_star <= P_K
        return one(P_star)
    else
        P_K_safe = max(P_K, 1.0e-30)
        arg = one(P_star) + (γ + 1) / (2γ) * (P_star / P_K_safe - 1)
        return sqrt(max(arg, one(P_star)))
    end
end

"""
    _hllc_star_state_1d(ρ, v, E, P, S_K, S_star) -> SVector{3}

Compute the HLLC star-region conserved state for 1D Euler.
"""
@inline function _hllc_star_state_1d(ρ, v, E, P, S_K, S_star)
    factor = ρ * (S_K - v) / (S_K - S_star)
    return SVector(
        factor,
        factor * S_star,
        factor * (E / ρ + (S_star - v) * (S_star + P / (ρ * (S_K - v))))
    )
end

# ============================================================
# 2D HLLC
# ============================================================

function solve_riemann(::HLLCSolver, law::EulerEquations{2}, wL::SVector{4}, wR::SVector{4}, dir::Int)
    ρL, vxL, vyL, PL = wL
    ρR, vxR, vyR, PR = wR
    γ = law.eos.gamma

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
    cL = sound_speed(law.eos, ρL, PL)
    cR = sound_speed(law.eos, ρR, PR)

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
        u_star_L = _hllc_star_state_2d(ρL, vnL, vtL, EL, PL, SL, S_star, dir)
        return fL + SL * (u_star_L - uL)
    else
        # Star-right region
        fR = physical_flux(law, wR, dir)
        u_star_R = _hllc_star_state_2d(ρR, vnR, vtR, ER, PR, SR, S_star, dir)
        return fR + SR * (u_star_R - uR)
    end
end

"""
    _hllc_star_state_2d(ρ, vn, vt, E, P, S_K, S_star, dir) -> SVector{4}

Compute the HLLC star-region conserved state for 2D Euler.
The tangential velocity is preserved across the contact, only the
normal velocity jumps to `S_star`.
"""
@inline function _hllc_star_state_2d(ρ, vn, vt, E, P, S_K, S_star, dir)
    factor = ρ * (S_K - vn) / (S_K - S_star)
    E_star = E / ρ + (S_star - vn) * (S_star + P / (ρ * (S_K - vn)))
    if dir == 1
        return SVector(
            factor,
            factor * S_star,   # ρ* vx* (normal is x)
            factor * vt,       # ρ* vy* (tangential preserved)
            factor * E_star
        )
    else
        return SVector(
            factor,
            factor * vt,       # ρ* vx* (tangential preserved)
            factor * S_star,   # ρ* vy* (normal is y)
            factor * E_star
        )
    end
end

# ============================================================
# 3D HLLC
# ============================================================

function solve_riemann(::HLLCSolver, law::EulerEquations{3}, wL::SVector{5}, wR::SVector{5}, dir::Int)
    ρL, vxL, vyL, vzL, PL = wL
    ρR, vxR, vyR, vzR, PR = wR
    γ = law.eos.gamma

    # Normal and tangential velocities
    if dir == 1
        vnL, vt1L, vt2L = vxL, vyL, vzL
        vnR, vt1R, vt2R = vxR, vyR, vzR
    elseif dir == 2
        vnL, vt1L, vt2L = vyL, vxL, vzL
        vnR, vt1R, vt2R = vyR, vxR, vzR
    else  # dir == 3
        vnL, vt1L, vt2L = vzL, vxL, vyL
        vnR, vt1R, vt2R = vzR, vxR, vyR
    end

    # Conserved states
    uL = primitive_to_conserved(law, wL)
    uR = primitive_to_conserved(law, wR)
    EL = uL[5]
    ER = uR[5]

    # Sound speeds
    cL = sound_speed(law.eos, ρL, PL)
    cR = sound_speed(law.eos, ρR, PR)

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
        u_star_L = _hllc_star_state_3d(ρL, vnL, vt1L, vt2L, EL, PL, SL, S_star, dir)
        return fL + SL * (u_star_L - uL)
    else
        # Star-right region
        fR = physical_flux(law, wR, dir)
        u_star_R = _hllc_star_state_3d(ρR, vnR, vt1R, vt2R, ER, PR, SR, S_star, dir)
        return fR + SR * (u_star_R - uR)
    end
end

"""
    _hllc_star_state_3d(ρ, vn, vt1, vt2, E, P, S_K, S_star, dir) -> SVector{5}

Compute the HLLC star-region conserved state for 3D Euler.
Both tangential velocities are preserved across the contact, only the
normal velocity jumps to `S_star`. The tangential ordering matches the
decomposition in `solve_riemann` (dir=1: vt1=vy, vt2=vz; dir=2: vt1=vx,
vt2=vz; dir=3: vt1=vx, vt2=vy).
"""
@inline function _hllc_star_state_3d(ρ, vn, vt1, vt2, E, P, S_K, S_star, dir)
    factor = ρ * (S_K - vn) / (S_K - S_star)
    E_star = E / ρ + (S_star - vn) * (S_star + P / (ρ * (S_K - vn)))
    if dir == 1
        return SVector(
            factor,
            factor * S_star,   # ρ* vx* (normal is x)
            factor * vt1,      # ρ* vy* (tangential preserved)
            factor * vt2,      # ρ* vz* (tangential preserved)
            factor * E_star
        )
    elseif dir == 2
        return SVector(
            factor,
            factor * vt1,      # ρ* vx* (tangential preserved)
            factor * S_star,   # ρ* vy* (normal is y)
            factor * vt2,      # ρ* vz* (tangential preserved)
            factor * E_star
        )
    else  # dir == 3
        return SVector(
            factor,
            factor * vt1,      # ρ* vx* (tangential preserved)
            factor * vt2,      # ρ* vy* (tangential preserved)
            factor * S_star,   # ρ* vz* (normal is z)
            factor * E_star
        )
    end
end
