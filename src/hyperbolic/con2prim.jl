# ============================================================
# Conservative-to-Primitive Recovery for Relativistic MHD
# ============================================================
#
# Iterative Newton-Raphson to recover primitive variables from
# conserved variables in special relativistic MHD.
#
# Primary algorithm: Palenzuela et al. (2015) — 1D Newton on
# auxiliary variable ξ = ρhW².
#
# Conserved: [D, Sx, Sy, Sz, τ, Bx, By, Bz]
#   D = ρW
#   S_j = (ρhW² + B²)v_j − (v·B)B_j
#   τ = ρhW² + B² − P_tot − D
#
# Primitive: [ρ, vx, vy, vz, P, Bx, By, Bz]

"""
    Con2PrimResult{FT}

Result of a conservative-to-primitive recovery.

# Fields
- `converged::Bool`: Whether the iteration converged.
- `iterations::Int`: Number of iterations performed.
- `residual::FT`: Final residual.
"""
struct Con2PrimResult{FT}
    converged::Bool
    iterations::Int
    residual::FT
end

"""
    srmhd_con2prim(eos, u::SVector{8}, tol, maxiter) -> (SVector{8}, Con2PrimResult)

Recover primitive variables `[ρ, vx, vy, vz, P, Bx, By, Bz]` from
conserved variables `[D, Sx, Sy, Sz, τ, Bx, By, Bz]` using the
Palenzuela et al. (2015) Newton-Raphson method.

Iterates on auxiliary variable `ξ = ρhW²` until the energy equation
residual `f(ξ) = ξ + B² − P_tot − (τ + D)` converges to zero.
"""
function srmhd_con2prim(eos, u::SVector{8}, tol::Real, maxiter::Int)
    D = u[1]
    Sx = u[2]
    Sy = u[3]
    Sz = u[4]
    tau = u[5]
    Bx = u[6]
    By = u[7]
    Bz = u[8]

    γ = eos.gamma
    gm1 = γ - 1

    # Invariants
    S_sq = Sx^2 + Sy^2 + Sz^2
    B_sq = Bx^2 + By^2 + Bz^2
    SdotB = Sx * Bx + Sy * By + Sz * Bz
    SdotB_sq = SdotB^2

    # Floor conserved variables
    D = max(D, 1.0e-12)

    # Initial guess: from τ + D = ξ + B² - P_tot, with P_tot ≈ P
    # Non-relativistic estimate: P ≈ gm1 * (τ + D - B²/2 - S²/(2(τ+D+B²)))
    E_approx = tau + D
    P_guess = max(gm1 * (E_approx - 0.5 * B_sq - 0.5 * S_sq / max(E_approx + B_sq, 1.0e-30)), 1.0e-16)
    # ξ ≈ τ + D + P_tot - B² ≈ τ + D + P + b²/2 - B² ≈ τ + D + P
    xi = E_approx + P_guess

    FT = typeof(xi)
    converged = false
    iterations = 0
    residual = FT(Inf)

    for iter in 1:maxiter
        iterations = iter

        # From ξ, compute v²
        # v² = [S²ξ² + (S·B)²(2ξ + B²)] / [ξ²(ξ + B²)²]
        xi_plus_Bsq = xi + B_sq
        v_sq_num = S_sq * xi^2 + SdotB_sq * (2 * xi + B_sq)
        v_sq_den = xi^2 * xi_plus_Bsq^2
        v_sq = v_sq_num / v_sq_den

        # Cap v² to prevent superluminal
        v_sq = min(v_sq, 1 - 1.0e-10)
        v_sq = max(v_sq, zero(v_sq))

        # Lorentz factor
        W_sq = 1 / (1 - v_sq)
        W = sqrt(W_sq)

        # Recover density
        rho = D / W
        rho = max(rho, 1.0e-12)

        # Pressure from ξ = ρhW² with ideal gas EOS
        # h = 1 + γ/(γ-1) * P/ρ → ξ = (ρ + γ/(γ-1)*P)*W²
        # → P = (γ-1)/γ * (ξ/W² - ρ)
        P_val = gm1 / γ * (xi / W_sq - rho)
        P_val = max(P_val, 1.0e-16)

        # Magnetic quantities
        b_sq = B_sq / W_sq + SdotB_sq / xi^2
        P_tot = P_val + 0.5 * b_sq

        # Residual: f(ξ) = ξ + B² - P_tot - (τ + D)
        f_val = xi + B_sq - P_tot - tau - D
        residual = abs(f_val)

        if residual < tol * max(abs(xi), one(xi))
            converged = true
            break
        end

        # Numerical derivative
        dxi = max(abs(xi) * 1.0e-7, 1.0e-20)
        xi_p = xi + dxi

        xi_p_plus_Bsq = xi_p + B_sq
        v_sq_p_num = S_sq * xi_p^2 + SdotB_sq * (2 * xi_p + B_sq)
        v_sq_p_den = xi_p^2 * xi_p_plus_Bsq^2
        v_sq_p = v_sq_p_num / v_sq_p_den
        v_sq_p = min(v_sq_p, 1 - 1.0e-10)
        v_sq_p = max(v_sq_p, zero(v_sq_p))
        W_sq_p = 1 / (1 - v_sq_p)
        W_p = sqrt(W_sq_p)
        rho_p = max(D / W_p, 1.0e-12)
        P_val_p = max(gm1 / γ * (xi_p / W_sq_p - rho_p), 1.0e-16)
        b_sq_p = B_sq / W_sq_p + SdotB_sq / xi_p^2
        P_tot_p = P_val_p + 0.5 * b_sq_p
        f_val_p = xi_p + B_sq - P_tot_p - tau - D

        df_dxi = (f_val_p - f_val) / dxi

        if abs(df_dxi) < 1.0e-30
            break
        end

        xi_new = xi - f_val / df_dxi
        xi = max(xi_new, D)  # ξ ≥ D always
    end

    # Final recovery
    SdotB_over_xi = SdotB / xi
    xi_plus_Bsq = xi + B_sq
    vx_val = (Sx + SdotB_over_xi * Bx) / xi_plus_Bsq
    vy_val = (Sy + SdotB_over_xi * By) / xi_plus_Bsq
    vz_val = (Sz + SdotB_over_xi * Bz) / xi_plus_Bsq

    v_sq = vx_val^2 + vy_val^2 + vz_val^2
    v_sq = min(v_sq, 1 - 1.0e-10)
    W = 1 / sqrt(1 - v_sq)
    rho = max(D / W, 1.0e-12)
    P_val = max(gm1 / γ * (xi / W^2 - rho), 1.0e-16)

    w = SVector(rho, vx_val, vy_val, vz_val, P_val, Bx, By, Bz)
    result = Con2PrimResult(converged, iterations, residual)

    return w, result
end
