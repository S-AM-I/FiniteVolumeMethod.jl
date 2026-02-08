# ============================================================
# HLLD Riemann Solver for Ideal MHD
# ============================================================
#
# 5-wave approximate Riemann solver resolving:
#   SL  (fast left)
#   SL* (left Alfvén/rotational)
#   SM  (contact/entropy)
#   SR* (right Alfvén/rotational)
#   SR  (fast right)
#
# Reference: Miyoshi & Kusano (2005), JCP 208, 315-344.

"""
    HLLDSolver <: AbstractRiemannSolver

The HLLD (Harten-Lax-van Leer-Discontinuities) approximate Riemann solver
for ideal MHD equations.

Resolves 5 waves: fast shocks (SL, SR), Alfvén/rotational discontinuities
(SL*, SR*), and the contact/entropy wave (SM).

Reference: Miyoshi & Kusano (2005), "A multi-state HLL approximate Riemann
solver for ideal magnetohydrodynamics", JCP 208, 315-344.
"""
struct HLLDSolver <: AbstractRiemannSolver end

function solve_riemann(::HLLDSolver, law::IdealMHDEquations, wL::SVector{8}, wR::SVector{8}, dir::Int)
    ρL, vxL, vyL, vzL, PL, BxL, ByL, BzL = wL
    ρR, vxR, vyR, vzR, PR, BxR, ByR, BzR = wR
    γ = law.eos.gamma

    # Decompose into normal/tangential based on flux direction
    if dir == 1
        vnL, vt1L, vt2L = vxL, vyL, vzL
        vnR, vt1R, vt2R = vxR, vyR, vzR
        BnL, Bt1L, Bt2L = BxL, ByL, BzL
        BnR, Bt1R, Bt2R = BxR, ByR, BzR
    else  # dir == 2
        vnL, vt1L, vt2L = vyL, vxL, vzL
        vnR, vt1R, vt2R = vyR, vxR, vzR
        BnL, Bt1L, Bt2L = ByL, BxL, BzL
        BnR, Bt1R, Bt2R = ByR, BxR, BzR
    end

    # Normal B (continuous across interface in ideal MHD)
    Bn = 0.5 * (BnL + BnR)
    Bn_sq = Bn^2

    # Conserved states and fluxes
    uL = primitive_to_conserved(law, wL)
    uR = primitive_to_conserved(law, wR)
    fL = physical_flux(law, wL, dir)
    fR = physical_flux(law, wR, dir)

    # Total pressures
    BsqL = BxL^2 + ByL^2 + BzL^2
    BsqR = BxR^2 + ByR^2 + BzR^2
    PtotL = PL + 0.5 * BsqL
    PtotR = PR + 0.5 * BsqR

    # Fast magnetosonic wave speed estimates
    cfL = fast_magnetosonic_speed(law, wL, dir)
    cfR = fast_magnetosonic_speed(law, wR, dir)

    SL = min(vnL, vnR) - max(cfL, cfR)
    SR = max(vnL, vnR) + max(cfL, cfR)

    # Quick return for supersonic cases
    if SL >= zero(SL)
        return fL
    elseif SR <= zero(SR)
        return fR
    end

    # Contact wave speed (SM)
    denom_SM = (SR - vnR) * ρR - (SL - vnL) * ρL
    SM = ((SR - vnR) * ρR * vnR - (SL - vnL) * ρL * vnL - PtotR + PtotL) / denom_SM

    # Star region total pressure (from Rankine-Hugoniot across outer wave)
    Ptot_star = PtotL + ρL * (SL - vnL) * (SM - vnL)

    # Star densities
    ρL_star = ρL * (SL - vnL) / (SL - SM)
    ρR_star = ρR * (SR - vnR) / (SR - SM)

    # Guard against negative densities
    ρL_star = max(ρL_star, 1e-20)
    ρR_star = max(ρR_star, 1e-20)

    # Star tangential velocities and B-fields
    vt1L_star, vt2L_star, Bt1L_star, Bt2L_star = _hlld_tangential_star(
        ρL, vnL, vt1L, vt2L, Bt1L, Bt2L, Bn_sq, SL, SM
    )
    vt1R_star, vt2R_star, Bt1R_star, Bt2R_star = _hlld_tangential_star(
        ρR, vnR, vt1R, vt2R, Bt1R, Bt2R, Bn_sq, SR, SM
    )

    # Star energies
    EL = uL[5]
    ER = uR[5]
    vdotB_L = vnL * Bn + vt1L * Bt1L + vt2L * Bt2L
    vdotB_L_star = SM * Bn + vt1L_star * Bt1L_star + vt2L_star * Bt2L_star
    vdotB_R = vnR * Bn + vt1R * Bt1R + vt2R * Bt2R
    vdotB_R_star = SM * Bn + vt1R_star * Bt1R_star + vt2R_star * Bt2R_star

    EL_star = ((SL - vnL) * EL - PtotL * vnL + Ptot_star * SM + Bn * (vdotB_L - vdotB_L_star)) / (SL - SM)
    ER_star = ((SR - vnR) * ER - PtotR * vnR + Ptot_star * SM + Bn * (vdotB_R - vdotB_R_star)) / (SR - SM)

    # Build star conserved states
    uL_star = _build_mhd_conserved(ρL_star, SM, vt1L_star, vt2L_star, EL_star, Bn, Bt1L_star, Bt2L_star, dir)
    uR_star = _build_mhd_conserved(ρR_star, SM, vt1R_star, vt2R_star, ER_star, Bn, Bt1R_star, Bt2R_star, dir)

    # Alfvén wave speeds for star regions
    sqrt_ρL_star = sqrt(ρL_star)
    sqrt_ρR_star = sqrt(ρR_star)
    abs_Bn = abs(Bn)
    SL_star = SM - abs_Bn / sqrt_ρL_star
    SR_star = SM + abs_Bn / sqrt_ρR_star

    if SM >= zero(SM)
        # Left side of contact
        fL_star = fL + SL * (uL_star - uL)
        if SL_star >= zero(SL_star)
            return fL_star
        else
            # Double-star region
            uL_dstar = _hlld_double_star_state(
                ρL_star, ρR_star, sqrt_ρL_star, sqrt_ρR_star,
                SM, Bn,
                vt1L_star, vt2L_star, Bt1L_star, Bt2L_star,
                vt1R_star, vt2R_star, Bt1R_star, Bt2R_star,
                EL_star, dir, :left
            )
            return fL_star + SL_star * (uL_dstar - uL_star)
        end
    else
        # Right side of contact
        fR_star = fR + SR * (uR_star - uR)
        if SR_star <= zero(SR_star)
            return fR_star
        else
            # Double-star region
            uR_dstar = _hlld_double_star_state(
                ρL_star, ρR_star, sqrt_ρL_star, sqrt_ρR_star,
                SM, Bn,
                vt1L_star, vt2L_star, Bt1L_star, Bt2L_star,
                vt1R_star, vt2R_star, Bt1R_star, Bt2R_star,
                ER_star, dir, :right
            )
            return fR_star + SR_star * (uR_dstar - uR_star)
        end
    end
end

# ============================================================
# Helper: tangential star-state components
# ============================================================

"""
Compute the tangential velocity and B-field components in the star region
for side K (left or right), given the outer wave speed SK and contact speed SM.

From Miyoshi & Kusano (2005), eqs (44)-(47).
"""
@inline function _hlld_tangential_star(
        ρ, vn, vt1, vt2, Bt1, Bt2, Bn_sq, SK, SM
    )
    denom = ρ * (SK - vn) * (SK - SM) - Bn_sq

    # Scale for tolerance check
    scale = max(abs(ρ * (SK - vn) * (SK - SM)), Bn_sq, eps(typeof(vn)))
    if abs(denom) > 1e-8 * scale
        factor_v = sqrt(Bn_sq) * (SM - vn) / denom
        factor_B = (ρ * (SK - vn)^2 - Bn_sq) / denom

        vt1_star = vt1 - Bt1 * factor_v
        vt2_star = vt2 - Bt2 * factor_v
        Bt1_star = Bt1 * factor_B
        Bt2_star = Bt2 * factor_B
    else
        # Degenerate case: Bn ≈ 0 or sonic point
        vt1_star = vt1
        vt2_star = vt2
        Bt1_star = Bt1
        Bt2_star = Bt2
    end

    return vt1_star, vt2_star, Bt1_star, Bt2_star
end

# ============================================================
# Helper: build conserved state from normal/tangential decomposition
# ============================================================

"""
Convert (ρ, vn, vt1, vt2, E, Bn, Bt1, Bt2) back to the standard
conserved variable ordering `[ρ, ρvx, ρvy, ρvz, E, Bx, By, Bz]`
based on direction.
"""
@inline function _build_mhd_conserved(ρ, vn, vt1, vt2, E, Bn, Bt1, Bt2, dir)
    if dir == 1
        return SVector(ρ, ρ * vn, ρ * vt1, ρ * vt2, E, Bn, Bt1, Bt2)
    else  # dir == 2
        return SVector(ρ, ρ * vt1, ρ * vn, ρ * vt2, E, Bt1, Bn, Bt2)
    end
end

# ============================================================
# Helper: double-star state (across Alfvén/rotational wave)
# ============================================================

"""
Compute the double-star state from left and right single-star states.
The tangential velocity and B-field are averaged using √ρ* weighting.

From Miyoshi & Kusano (2005), eqs (59)-(63).
"""
@inline function _hlld_double_star_state(
        ρL_star, ρR_star, sqrt_ρL, sqrt_ρR,
        SM, Bn,
        vt1L_star, vt2L_star, Bt1L_star, Bt2L_star,
        vt1R_star, vt2R_star, Bt1R_star, Bt2R_star,
        E_star, dir, side::Symbol
    )
    denom = sqrt_ρL + sqrt_ρR

    if denom < 1e-30
        # Both densities effectively zero — return star state as-is
        ρ_ds = side == :left ? ρL_star : ρR_star
        vt1_ds = side == :left ? vt1L_star : vt1R_star
        vt2_ds = side == :left ? vt2L_star : vt2R_star
        Bt1_ds = side == :left ? Bt1L_star : Bt1R_star
        Bt2_ds = side == :left ? Bt2L_star : Bt2R_star
        return _build_mhd_conserved(ρ_ds, SM, vt1_ds, vt2_ds, E_star, Bn, Bt1_ds, Bt2_ds, dir)
    end

    sign_Bn = Bn >= zero(Bn) ? one(Bn) : -one(Bn)

    # Averaged tangential velocities (eq 59-60)
    vt1_ds = (sqrt_ρL * vt1L_star + sqrt_ρR * vt1R_star +
              (Bt1R_star - Bt1L_star) * sign_Bn) / denom
    vt2_ds = (sqrt_ρL * vt2L_star + sqrt_ρR * vt2R_star +
              (Bt2R_star - Bt2L_star) * sign_Bn) / denom

    # Averaged tangential B-fields (eq 61-62)
    Bt1_ds = (sqrt_ρL * Bt1R_star + sqrt_ρR * Bt1L_star +
              sqrt_ρL * sqrt_ρR * (vt1R_star - vt1L_star) * sign_Bn) / denom
    Bt2_ds = (sqrt_ρL * Bt2R_star + sqrt_ρR * Bt2L_star +
              sqrt_ρL * sqrt_ρR * (vt2R_star - vt2L_star) * sign_Bn) / denom

    # Energy update (eq 63)
    if side == :left
        vdotB_star = SM * Bn + vt1L_star * Bt1L_star + vt2L_star * Bt2L_star
        vdotB_ds = SM * Bn + vt1_ds * Bt1_ds + vt2_ds * Bt2_ds
        E_ds = E_star + sqrt_ρL * sign_Bn * (vdotB_star - vdotB_ds)
        ρ_ds = ρL_star
    else
        vdotB_star = SM * Bn + vt1R_star * Bt1R_star + vt2R_star * Bt2R_star
        vdotB_ds = SM * Bn + vt1_ds * Bt1_ds + vt2_ds * Bt2_ds
        E_ds = E_star - sqrt_ρR * sign_Bn * (vdotB_star - vdotB_ds)
        ρ_ds = ρR_star
    end

    return _build_mhd_conserved(ρ_ds, SM, vt1_ds, vt2_ds, E_ds, Bn, Bt1_ds, Bt2_ds, dir)
end
