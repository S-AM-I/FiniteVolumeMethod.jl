# ============================================================
# Metric-Aware Conservative-to-Primitive Recovery for GRMHD
# ============================================================
#
# In the Valencia formulation, the conserved variables are
# densitized: U_tilde = sqrt(gamma) * U. The con2prim recovery
# must:
#   1. Undensitize: U = U_tilde / sqrt(gamma)
#   2. Raise/lower momentum indices with the spatial metric
#   3. Iterate on xi = rho*h*W^2 using the modified Palenzuela
#      algorithm that accounts for the spatial metric
#
# The key difference from flat-space SRMHD con2prim:
#   - S^2 = gamma^ij S_i S_j  (not just S_x^2 + S_y^2 + S_z^2)
#   - v^2 = gamma_ij v^i v^j  (for the Lorentz factor)
#   - v^i recovery uses gamma^ij
#
# For Minkowski metric (gamma^ij = delta_ij), this reduces to
# the standard srmhd_con2prim exactly.

"""
    grmhd_con2prim(law::GRMHDEquations, u_tilde::SVector{8}, x, y)
        -> (SVector{8}, Con2PrimResult)

Recover primitive variables from densitized conserved variables at
position `(x, y)` using the metric from `law.metric`.

Steps:
1. Undensitize: `u = u_tilde / sqrt(gamma)`
2. Compute `S^2 = gamma^ij S_i S_j`
3. Run the modified Palenzuela Newton-Raphson iteration
4. Return primitives `[rho, vx, vy, vz, P, Bx, By, Bz]`
"""
function grmhd_con2prim(law::GRMHDEquations, u_tilde::SVector{8}, x, y)
    metric = law.metric
    eos = law.eos
    tol = law.con2prim_tol
    maxiter = law.con2prim_maxiter

    # Evaluate metric at this position
    sg = sqrt_gamma(metric, x, y)
    gi = inv_spatial_metric(metric, x, y)
    gm = spatial_metric(metric, x, y)

    return _grmhd_con2prim_with_metric(eos, u_tilde, sg, gi, gm, tol, maxiter)
end

"""
    grmhd_con2prim_cached(law::GRMHDEquations, u_tilde::SVector{8},
                           sg, gixx, gixy, giyy, gxx, gxy, gyy)
        -> (SVector{8}, Con2PrimResult)

Metric-aware con2prim using precomputed (cached) metric quantities.
This is the fast path used in the solver loop.
"""
@inline function grmhd_con2prim_cached(law::GRMHDEquations, u_tilde::SVector{8},
        sg, gixx, gixy, giyy, gxx, gxy, gyy)
    gi = StaticArrays.SMatrix{2, 2}(gixx, gixy, gixy, giyy)
    gm = StaticArrays.SMatrix{2, 2}(gxx, gxy, gxy, gyy)
    return _grmhd_con2prim_with_metric(law.eos, u_tilde, sg, gi, gm,
        law.con2prim_tol, law.con2prim_maxiter)
end

"""
    _grmhd_con2prim_with_metric(eos, u_tilde, sg, gi, gm, tol, maxiter)
        -> (SVector{8}, Con2PrimResult)

Internal metric-aware con2prim implementation.

# Arguments
- `eos`: Equation of state.
- `u_tilde`: Densitized conserved variables sqrt(gamma)*[D, Sx, Sy, Sz, tau, Bx, By, Bz].
- `sg`: sqrt(det(gamma_ij)).
- `gi`: Inverse spatial metric (2x2 SMatrix).
- `gm`: Spatial metric (2x2 SMatrix).
- `tol`: Convergence tolerance.
- `maxiter`: Maximum iterations.
"""
function _grmhd_con2prim_with_metric(eos, u_tilde::SVector{8}, sg,
        gi::StaticArrays.SMatrix{2, 2}, gm::StaticArrays.SMatrix{2, 2},
        tol::Real, maxiter::Int)
    gamma_eos = eos.gamma
    gm1 = gamma_eos - 1

    # Undensitize
    inv_sg = 1 / sg
    D = u_tilde[1] * inv_sg
    Sx = u_tilde[2] * inv_sg
    Sy = u_tilde[3] * inv_sg
    Sz = u_tilde[4] * inv_sg
    tau = u_tilde[5] * inv_sg
    Bx = u_tilde[6] * inv_sg
    By = u_tilde[7] * inv_sg
    Bz = u_tilde[8] * inv_sg

    # Floor conserved density
    D = max(D, 1e-12)

    # Compute S^2 = gamma^ij S_i S_j (raised with inverse spatial metric)
    # In 2D: S^2 = gi[1,1]*Sx^2 + 2*gi[1,2]*Sx*Sy + gi[2,2]*Sy^2 + Sz^2
    # (Sz is the z-component, and gamma^zz = 1 in the equatorial plane assumption)
    S_sq = gi[1, 1] * Sx^2 + 2 * gi[1, 2] * Sx * Sy + gi[2, 2] * Sy^2 + Sz^2

    # B^2 = gamma^ij B^i B^j for contravariant B (our stored B is already contravariant)
    # Actually in the Valencia formulation B^i are coordinate components,
    # and B^2 (the flat-space magnitude) appears in the SRMHD-like formulas.
    # For the con2prim we need B_sq = B^i B^j gamma_ij
    # But wait: in the Valencia formulation the B in the conserved variables
    # are the coordinate B^i, and b^2 uses both the metric and v.
    # For the con2prim iteration, B_sq should use the metric:
    B_sq = gm[1, 1] * Bx^2 + 2 * gm[1, 2] * Bx * By + gm[2, 2] * By^2 + Bz^2

    # S.B = S_i B^i (S is covariant, B is contravariant)
    SdotB = Sx * Bx + Sy * By + Sz * Bz

    # Initial guess: ξ ≈ τ + D + P_guess (non-relativistic estimate)
    SdotB_sq = SdotB^2
    E_approx = tau + D
    P_guess = max(gm1 * (E_approx - 0.5 * B_sq - 0.5 * S_sq / max(E_approx + B_sq, 1e-30)), 1e-16)
    xi = E_approx + P_guess

    FT = typeof(xi)
    converged = false
    iterations = 0
    residual = FT(Inf)

    for iter in 1:maxiter
        iterations = iter

        xi_plus_Bsq = xi + B_sq
        SdotB_over_xi = SdotB / xi

        # v^2 from momentum (metric-corrected S^2)
        # v² = [S²ξ² + (S·B)²(2ξ + B²)] / [ξ²(ξ + B²)²]
        v_sq = (S_sq + SdotB_over_xi * SdotB * (2 + B_sq / xi)) / (xi_plus_Bsq^2)
        v_sq = clamp(v_sq, zero(v_sq), 1 - 1e-10)

        W_sq = 1 / (1 - v_sq)
        W = sqrt(W_sq)

        rho = max(D / W, 1e-12)
        P_val = max(gm1 / gamma_eos * (xi / W_sq - rho), 1e-16)

        # Magnetic pressure in comoving frame
        b_sq_iter = B_sq / W_sq + SdotB^2 / xi^2
        P_tot_iter = P_val + 0.5 * b_sq_iter

        # Residual: energy equation f(ξ) = ξ + B² − P_tot − (τ + D)
        f_val = xi + B_sq - P_tot_iter - tau - D
        residual = abs(f_val)

        if residual < tol * max(abs(xi), one(xi))
            converged = true
            break
        end

        # Numerical derivative
        dxi = max(abs(xi) * 1e-7, 1e-20)
        xi_p = xi + dxi

        SdotB_over_xi_p = SdotB / xi_p
        v_sq_p = (S_sq + SdotB_over_xi_p * SdotB * (2 + B_sq / xi_p)) / ((xi_p + B_sq)^2)
        v_sq_p = clamp(v_sq_p, zero(v_sq_p), 1 - 1e-10)
        W_sq_p = 1 / (1 - v_sq_p)
        W_p = sqrt(W_sq_p)
        rho_p = max(D / W_p, 1e-12)
        P_val_p = max(gm1 / gamma_eos * (xi_p / W_sq_p - rho_p), 1e-16)
        b_sq_p = B_sq / W_sq_p + SdotB^2 / xi_p^2
        P_tot_p = P_val_p + 0.5 * b_sq_p
        f_val_p = xi_p + B_sq - P_tot_p - tau - D

        df_dxi = (f_val_p - f_val) / dxi
        if abs(df_dxi) < 1e-30
            break
        end

        xi_new = xi - f_val / df_dxi
        xi = max(xi_new, D)
    end

    # Final recovery of velocity
    SdotB_over_xi = SdotB / xi
    xi_plus_Bsq = xi + B_sq

    # v_j (covariant) = (S_j + (S.B/xi) B_j) / (xi + B^2)
    # But we want v^i (contravariant). The relation is:
    # v^i = gamma^ij v_j
    # First get the "flat-like" intermediate:
    # For the SRMHD form: v_j = (S_j + (SdotB/xi) B_j) / (xi + B_sq)
    # But S_j is covariant and B_j... actually B^j is contravariant.
    # In the metric-aware version:
    # v^i = gamma^ij (S_j + SdotB/xi * B_j_low) / (xi + B_sq)
    # where B_j_low = gamma_jk B^k

    # Lower B: B_j_low = gamma_jk B^k
    Bx_low = gm[1, 1] * Bx + gm[1, 2] * By
    By_low = gm[1, 2] * Bx + gm[2, 2] * By
    Bz_low = Bz  # gamma_zz = 1

    # Covariant velocity components
    vx_cov = (Sx + SdotB_over_xi * Bx_low) / xi_plus_Bsq
    vy_cov = (Sy + SdotB_over_xi * By_low) / xi_plus_Bsq
    vz_cov = (Sz + SdotB_over_xi * Bz_low) / xi_plus_Bsq

    # Raise to contravariant: v^i = gamma^ij v_j
    vx_val = gi[1, 1] * vx_cov + gi[1, 2] * vy_cov
    vy_val = gi[1, 2] * vx_cov + gi[2, 2] * vy_cov
    vz_val = vz_cov  # gamma^zz = 1

    # Compute proper v^2 = gamma_ij v^i v^j
    v_sq = gm[1, 1] * vx_val^2 + 2 * gm[1, 2] * vx_val * vy_val + gm[2, 2] * vy_val^2 + vz_val^2
    v_sq = min(v_sq, 1 - 1e-10)
    W = 1 / sqrt(1 - v_sq)
    rho = max(D / W, 1e-12)
    P_val = max(gm1 / gamma_eos * (xi / W^2 - rho), 1e-16)

    w = SVector(rho, vx_val, vy_val, vz_val, P_val, Bx, By, Bz)
    result = Con2PrimResult(converged, iterations, residual)

    return w, result
end

"""
    grmhd_primitive_to_conserved_densitized(law::GRMHDEquations, w::SVector{8}, x, y)
        -> SVector{8}

Convert primitive variables to densitized conserved variables at position `(x, y)`.
Returns `sqrt(gamma) * [D, Sx, Sy, Sz, tau, Bx, By, Bz]`.
"""
@inline function grmhd_primitive_to_conserved_densitized(law::GRMHDEquations, w::SVector{8}, x, y)
    sg = sqrt_gamma(law.metric, x, y)
    gm = spatial_metric(law.metric, x, y)
    return _grmhd_prim2con_densitized(law.eos, w, sg, gm)
end

"""
    grmhd_prim2con_densitized_cached(law::GRMHDEquations, w::SVector{8},
                                      sg, gxx, gxy, gyy) -> SVector{8}

Convert primitive to densitized conserved using precomputed metric data.
"""
@inline function grmhd_prim2con_densitized_cached(law::GRMHDEquations, w::SVector{8},
        sg, gxx, gxy, gyy)
    gm = StaticArrays.SMatrix{2, 2}(gxx, gxy, gxy, gyy)
    return _grmhd_prim2con_densitized(law.eos, w, sg, gm)
end

"""
    _grmhd_prim2con_densitized(eos, w, sg, gm) -> SVector{8}

Internal: compute densitized conserved from primitives with metric data.

The Lorentz factor uses v^2 = gamma_ij v^i v^j, and the momentum S_j
is covariant (lowered with gamma_ij).
"""
@inline function _grmhd_prim2con_densitized(eos, w::SVector{8}, sg,
        gm::StaticArrays.SMatrix{2, 2})
    rho, vx, vy, vz, P, Bx, By, Bz = w
    gamma_eos = eos.gamma

    # v^2 = gamma_ij v^i v^j (metric-corrected)
    v_sq = gm[1, 1] * vx^2 + 2 * gm[1, 2] * vx * vy + gm[2, 2] * vy^2 + vz^2
    v_sq = min(v_sq, 1 - 1e-10)

    W = 1 / sqrt(1 - v_sq)
    W_sq = W^2

    eps_val = P / ((gamma_eos - 1) * rho)
    h = 1 + eps_val + P / rho

    # Magnetic quantities use metric-corrected B^2
    # For b^mu: b^0 = W * gamma_ij v^i B^j
    vdotB = gm[1, 1] * vx * Bx + gm[1, 2] * (vx * By + vy * Bx) + gm[2, 2] * vy * By + vz * Bz
    B_sq = gm[1, 1] * Bx^2 + 2 * gm[1, 2] * Bx * By + gm[2, 2] * By^2 + Bz^2
    b0 = W * vdotB
    b_sq = B_sq / W_sq + vdotB^2

    rho_h_W2 = rho * h * W_sq
    Ptot = P + 0.5 * b_sq

    D = rho * W

    # S_j (covariant momentum): S_j = (rho*h*W^2 + b^2)(gamma_jk v^k) - b_j * b^0
    # where b_j = gamma_jk b^k + ... actually b_j_spatial = B_j/W + b0 * v_j_low
    # v_j_low = gamma_jk v^k
    vx_low = gm[1, 1] * vx + gm[1, 2] * vy
    vy_low = gm[1, 2] * vx + gm[2, 2] * vy
    vz_low = vz

    # b_j (covariant spatial magnetic 4-vector)
    Bx_low = gm[1, 1] * Bx + gm[1, 2] * By
    By_low = gm[1, 2] * Bx + gm[2, 2] * By
    Bz_low = Bz

    bx_low = Bx_low / W + b0 * vx_low
    by_low = By_low / W + b0 * vy_low
    bz_low = Bz_low / W + b0 * vz_low

    # S_j = (ρhW² + B²)v_j − (v·B)B_j_low
    Sx_c = (rho_h_W2 + B_sq) * vx_low - vdotB * Bx_low
    Sy_c = (rho_h_W2 + B_sq) * vy_low - vdotB * By_low
    Sz_c = (rho_h_W2 + B_sq) * vz_low - vdotB * Bz_low
    # τ = ρhW² + B² − P_tot − D
    tau = rho_h_W2 + B_sq - Ptot - D

    # Densitize: multiply by sqrt(gamma)
    return SVector(sg * D, sg * Sx_c, sg * Sy_c, sg * Sz_c,
        sg * tau, sg * Bx, sg * By, sg * Bz)
end
