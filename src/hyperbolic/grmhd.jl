# ============================================================
# General Relativistic MHD Equations (Valencia Formulation)
# ============================================================
#
# The GRMHD system in the Valencia formulation (Banyuls et al. 1997)
# extends SRMHD to curved spacetimes using the 3+1 decomposition.
#
# Primitive variables (same as SRMHD):
#   W = [rho, vx, vy, vz, P, Bx, By, Bz]
#
# where v^i are the 3-velocities measured by the Eulerian observer.
#
# Densitized conserved variables:
#   U = sqrt(gamma) * [D, S_x, S_y, S_z, tau, Bx, By, Bz]
#
# with:
#   D   = rho * W                           (conserved density)
#   S_j = (rho*h*W^2 + b^2) v_j - b^0 b_j  (covariant momentum)
#   tau = rho*h*W^2 - P_tot - (b^0)^2 - D   (energy minus rest mass)
#
# The Lorentz factor uses the spatial metric:
#   W = 1 / sqrt(1 - gamma_ij v^i v^j)
#
# The flux in the Valencia formulation is:
#   F^i = alpha * f^i - beta^i * U
#
# where f^i is the SRMHD-like flux (same functional form as flat space).
#
# Geometric source terms arise from the coupling of the stress-energy
# tensor to spacetime curvature:
#   S = sqrt(gamma) * T^mu_nu * partial g^nu_alpha
#
# These source terms are zero for D and B components.
#
# References:
#   Banyuls et al. (1997), Font et al. (2000),
#   Anton et al. (2006), Gammie et al. (2003).

"""
    GRMHDEquations{Dim, EOS <: AbstractEOS, M <: AbstractMetric{Dim}} <: AbstractConservationLaw{Dim}

The general relativistic magnetohydrodynamics equations in `Dim` spatial dimensions
using the Valencia formulation.

## Variables (8 components)
- Primitive: `W = [rho, vx, vy, vz, P, Bx, By, Bz]`
- Conserved (densitized): `U = sqrt(gamma) * [D, Sx, Sy, Sz, tau, Bx, By, Bz]`

The `physical_flux` method returns the flat-space-like flux `f^i`. The solver
applies the Valencia correction `alpha * F_riemann - beta * U` at each face.

# Fields
- `eos::EOS`: Equation of state.
- `metric::M`: Spacetime metric.
- `con2prim_tol::Float64`: Tolerance for con2prim convergence.
- `con2prim_maxiter::Int`: Maximum con2prim iterations.
"""
struct GRMHDEquations{Dim, EOS <: AbstractEOS, M <: AbstractMetric{Dim}} <: AbstractConservationLaw{Dim}
    eos::EOS
    metric::M
    con2prim_tol::Float64
    con2prim_maxiter::Int
end

function GRMHDEquations{Dim}(
        eos::EOS, metric::M;
        con2prim_tol = 1.0e-12, con2prim_maxiter = 50
    ) where {Dim, EOS <: AbstractEOS, M <: AbstractMetric{Dim}}
    return GRMHDEquations{Dim, EOS, M}(eos, metric, con2prim_tol, con2prim_maxiter)
end

nvariables(::GRMHDEquations) = 8

# ============================================================
# Conserved <-> Primitive Conversion
# ============================================================
#
# For the generic interface (used by Riemann solvers and
# reconstruction which work in flat-space-like variables),
# conserved_to_primitive and primitive_to_conserved operate
# on undensitized variables (i.e., the sqrt(gamma) factor
# is handled by the solver, not these routines).
#
# The metric-aware con2prim (grmhd_con2prim) handles
# undensitization and uses the spatial metric for raising/
# lowering indices.

"""
    conserved_to_primitive(law::GRMHDEquations, u::SVector{8}) -> SVector{8}

Convert GRMHD conserved variables to primitive. This method operates on
undensitized variables (sqrt(gamma) already divided out) and uses flat-space
index operations. For metric-aware recovery, use `grmhd_con2prim`.
"""
@inline function conserved_to_primitive(law::GRMHDEquations, u::SVector{8})
    # Delegate to the SRMHD con2prim since the functional form is identical
    # for undensitized variables with flat-space index raising
    w, _ = srmhd_con2prim(law.eos, u, law.con2prim_tol, law.con2prim_maxiter)
    return w
end

"""
    primitive_to_conserved(law::GRMHDEquations, w::SVector{8}) -> SVector{8}

Convert GRMHD primitive variables to undensitized conserved variables.
The functional form is identical to SRMHD; densitization (multiplication
by sqrt(gamma)) is handled by the solver.
"""
@inline function primitive_to_conserved(law::GRMHDEquations, w::SVector{8})
    rho, vx, vy, vz, P, Bx, By, Bz = w
    gamma_eos = law.eos.gamma

    W = lorentz_factor(vx, vy, vz)
    W_sq = W^2

    eps_val = P / ((gamma_eos - 1) * rho)
    h = 1 + eps_val + P / rho

    B_sq = Bx^2 + By^2 + Bz^2
    vdotB = vx * Bx + vy * By + vz * Bz
    _, b_sq = srmhd_b_quantities(vx, vy, vz, Bx, By, Bz, W)

    rho_h_W2 = rho * h * W_sq
    Ptot = P + 0.5 * b_sq

    D = rho * W
    # S_j = (ρhW² + B²)v_j − (v·B)B_j
    Sx_c = (rho_h_W2 + B_sq) * vx - vdotB * Bx
    Sy_c = (rho_h_W2 + B_sq) * vy - vdotB * By
    Sz_c = (rho_h_W2 + B_sq) * vz - vdotB * Bz
    # τ = ρhW² + B² − P_tot − D
    tau = rho_h_W2 + B_sq - Ptot - D

    return SVector(D, Sx_c, Sy_c, Sz_c, tau, Bx, By, Bz)
end

# ============================================================
# Physical Flux (flat-space-like form)
# ============================================================
#
# Returns f^i, the SRMHD-like flux. The solver constructs the
# full Valencia flux as: F^i = alpha * f^i - beta^i * U.

"""
    physical_flux(law::GRMHDEquations, w::SVector{8}, dir::Int) -> SVector{8}

Compute the flat-space-like SRMHD flux f^i in direction `dir` (1=x, 2=y)
from primitive variables.

The full Valencia flux is `F^i = alpha * f^i - beta^i * U`, but only `f^i`
is returned here. The metric correction is applied by the solver.
"""
@inline function physical_flux(law::GRMHDEquations, w::SVector{8}, dir::Int)
    rho, vx, vy, vz, P, Bx, By, Bz = w
    gamma_eos = law.eos.gamma

    W = lorentz_factor(vx, vy, vz)
    eps_val = P / ((gamma_eos - 1) * rho)
    h = 1 + eps_val + P / rho

    B_sq = Bx^2 + By^2 + Bz^2
    b0, b_sq = srmhd_b_quantities(vx, vy, vz, Bx, By, Bz, W)
    rho_h_W2 = rho * h * W^2
    Ptot = P + 0.5 * b_sq

    D = rho * W
    # τ = ρhW² + B² − P_tot − D  (identity: (ρh+b²)W² = ρhW² + B² + b⁰²)
    tau = rho_h_W2 + B_sq - Ptot - D

    # Magnetic 4-vector spatial components: b_j = B_j/W + b^0 v_j
    bx = Bx / W + b0 * vx
    by = By / W + b0 * vy
    bz = Bz / W + b0 * vz

    # (ρh + b²)W² = ρhW² + B² + b⁰²
    wtot = rho_h_W2 + B_sq + b0^2

    if dir == 1
        vn = vx
        bn = bx
        return SVector(
            D * vn,
            wtot * vx * vn - bx * bn + Ptot,
            wtot * vy * vn - by * bn,
            wtot * vz * vn - bz * bn,
            (wtot - D) * vn - b0 * bn,
            zero(vn),
            By * vn - Bx * vy,
            Bz * vn - Bx * vz
        )
    else  # dir == 2
        vn = vy
        bn = by
        return SVector(
            D * vn,
            wtot * vx * vn - bx * bn,
            wtot * vy * vn - by * bn + Ptot,
            wtot * vz * vn - bz * bn,
            (wtot - D) * vn - b0 * bn,
            Bx * vn - By * vx,
            zero(vn),
            Bz * vn - By * vz
        )
    end
end

# ============================================================
# Wave Speeds (metric-corrected)
# ============================================================
#
# In GR, the coordinate-frame wave speeds are modified by the
# lapse and shift. For a wave speed lambda_flat in flat space,
# the coordinate speed is:
#   lambda_coord = alpha * lambda_flat - beta
#
# For the Riemann solver we need the speeds including the
# metric corrections.

"""
    max_wave_speed(law::GRMHDEquations, w::SVector{8}, dir::Int) -> Real

Maximum wave speed from primitive variables. Returns the flat-space
fast magnetosonic speed (same as SRMHD). The metric correction is
applied in the solver's CFL calculation.
"""
@inline function max_wave_speed(law::GRMHDEquations, w::SVector{8}, dir::Int)
    lam_m, lam_p = _grmhd_wave_speeds(law, w, dir)
    return max(abs(lam_m), abs(lam_p))
end

"""
    wave_speeds(law::GRMHDEquations, w::SVector{8}, dir::Int) -> (lam_min, lam_max)

Return the fastest left-going and right-going wave speeds in flat-space form.
"""
@inline function wave_speeds(law::GRMHDEquations, w::SVector{8}, dir::Int)
    return _grmhd_wave_speeds(law, w, dir)
end

"""
    _grmhd_wave_speeds(law::GRMHDEquations, w::SVector{8}, dir::Int) -> (lam_min, lam_max)

Compute the relativistic fast magnetosonic wave speeds in direction `dir`.
Uses the same formula as SRMHD (flat-space Lorentz factor).
"""
@inline function _grmhd_wave_speeds(law::GRMHDEquations, w::SVector{8}, dir::Int)
    rho, vx, vy, vz, P, Bx, By, Bz = w
    gamma_eos = law.eos.gamma

    v_sq = vx^2 + vy^2 + vz^2
    v_sq = min(v_sq, 1 - 1.0e-10)

    W = 1 / sqrt(1 - v_sq)
    eps_val = P / ((gamma_eos - 1) * rho)
    h = 1 + eps_val + P / rho

    cs_sq = gamma_eos * P / (rho * h)
    cs_sq = min(cs_sq, 1 - 1.0e-10)

    B_sq = Bx^2 + By^2 + Bz^2
    vdotB = vx * Bx + vy * By + vz * Bz
    b_sq = B_sq / W^2 + vdotB^2

    rho_h = rho * h
    ca_sq = b_sq / (rho_h + b_sq)
    ca_sq = min(ca_sq, 1 - 1.0e-10)

    c_ms_sq = cs_sq + ca_sq - cs_sq * ca_sq
    c_ms_sq = clamp(c_ms_sq, zero(c_ms_sq), 1 - 1.0e-10)
    c_ms = sqrt(c_ms_sq)

    vn = dir == 1 ? vx : vy

    denom = 1 - v_sq * c_ms_sq
    discriminant = (1 - v_sq) * (1 - v_sq * c_ms_sq - vn^2 * (1 - c_ms_sq))
    discriminant = max(discriminant, zero(discriminant))
    sqrt_disc = sqrt(discriminant)

    lam_minus = (vn * (1 - c_ms_sq) - c_ms * sqrt_disc) / denom
    lam_plus = (vn * (1 - c_ms_sq) + c_ms * sqrt_disc) / denom

    return lam_minus, lam_plus
end

# ============================================================
# Metric-Corrected Wave Speeds for CFL
# ============================================================

"""
    grmhd_max_wave_speed_coord(law::GRMHDEquations, w::SVector{8}, dir::Int,
                                alpha, beta_dir) -> Real

Maximum wave speed in coordinate frame, accounting for lapse and shift:
  lambda_coord = alpha * lambda_flat - beta^dir

Used for CFL condition in the GRMHD solver.
"""
@inline function grmhd_max_wave_speed_coord(
        law::GRMHDEquations, w::SVector{8}, dir::Int,
        alpha, beta_dir
    )
    lam_m, lam_p = _grmhd_wave_speeds(law, w, dir)
    # Transform to coordinate frame
    lam_m_coord = alpha * lam_m - beta_dir
    lam_p_coord = alpha * lam_p - beta_dir
    return max(abs(lam_m_coord), abs(lam_p_coord))
end

# ============================================================
# Geometric Source Terms
# ============================================================
#
# The GRMHD source terms arise from the contraction of the
# stress-energy tensor with the Christoffel symbols:
#   S^mu = sqrt(gamma) * T^mu_nu * Gamma^nu_alpha_beta * g^alpha_beta
#
# In practice, these can be written as:
#   S^mu = alpha * sqrt(gamma) * T^mu_nu * partial_mu ln(alpha)
#        + ... (shift and metric derivative terms)
#
# For the momentum and energy equations, the source depends on
# derivatives of the metric. The source is zero for D and B.
#
# We use a finite-difference approximation for the metric derivatives
# computed from the precomputed MetricData2D.

"""
    grmhd_source_terms(law::GRMHDEquations{2}, w::SVector{8}, u_densitized::SVector{8},
                        md::MetricData2D, mesh, ix::Int, iy::Int) -> SVector{8}

Compute the geometric source terms for the GRMHD equations at cell `(ix, iy)`.

Returns `S = (0, Sx_src, Sy_src, Sz_src, tau_src, 0, 0, 0)`.

The source terms are computed using the stress-energy tensor contracted with
metric derivatives (Christoffel symbols). We use centered finite differences
for the metric derivatives.
"""
@inline function grmhd_source_terms(
        law::GRMHDEquations{2}, w::SVector{8},
        u_densitized::SVector{8}, md::MetricData2D,
        mesh::StructuredMesh2D, ix::Int, iy::Int
    )
    rho, vx, vy, vz, P, Bx, By, Bz = w
    gamma_eos = law.eos.gamma
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    W = lorentz_factor(vx, vy, vz)
    eps_val = P / ((gamma_eos - 1) * rho)
    h = 1 + eps_val + P / rho

    B_sq = Bx^2 + By^2 + Bz^2
    vdotB = vx * Bx + vy * By + vz * Bz
    b0, b_sq = srmhd_b_quantities(vx, vy, vz, Bx, By, Bz, W)
    rho_h_W2 = rho * h * W^2
    Ptot = P + 0.5 * b_sq

    # Source term formulas from Del Zanna et al. (2007, ECHO code):
    #
    # Momentum (S_{S_j}):
    #   S_{S_j} = sqrt(gamma) * [-(tau+D) partial_j alpha
    #             + S_k partial_j beta^k
    #             + alpha/2 * T^kl partial_j gamma_kl]
    #
    # Energy (S_tau):
    #   S_tau = sqrt(gamma) * [-S_j partial^j alpha + alpha T^kl K_kl]
    #
    # where K_ij is approximated as (partial_i beta_j_low + partial_j beta_i_low)/(2 alpha)
    # (exact for Cartesian-like coordinates, accurate for Kerr-Schild).
    #
    # T^ij = (rho*h*W^2 + b^2) v^i v^j - b^i b^j + P_tot gamma^ij

    # Local metric values
    alp = md.alpha[ix, iy]
    bx_s = md.beta_x[ix, iy]
    by_s = md.beta_y[ix, iy]
    gxx = md.gamma_xx[ix, iy]
    gxy = md.gamma_xy[ix, iy]
    gyy = md.gamma_yy[ix, iy]
    sg = md.sqrtg[ix, iy]

    # Compute centered finite-difference derivatives of metric components
    # Use one-sided differences at boundaries
    # d/dx quantities
    ixm = max(ix - 1, 1)
    ixp = min(ix + 1, nx)
    dx_eff = (ixp - ixm) * dx

    dalpha_dx = (md.alpha[ixp, iy] - md.alpha[ixm, iy]) / dx_eff
    dbetax_dx = (md.beta_x[ixp, iy] - md.beta_x[ixm, iy]) / dx_eff
    dbetay_dx = (md.beta_y[ixp, iy] - md.beta_y[ixm, iy]) / dx_eff
    dgxx_dx = (md.gamma_xx[ixp, iy] - md.gamma_xx[ixm, iy]) / dx_eff
    dgxy_dx = (md.gamma_xy[ixp, iy] - md.gamma_xy[ixm, iy]) / dx_eff
    dgyy_dx = (md.gamma_yy[ixp, iy] - md.gamma_yy[ixm, iy]) / dx_eff

    # d/dy quantities
    iym = max(iy - 1, 1)
    iyp = min(iy + 1, ny)
    dy_eff = (iyp - iym) * dy

    dalpha_dy = (md.alpha[ix, iyp] - md.alpha[ix, iym]) / dy_eff
    dbetax_dy = (md.beta_x[ix, iyp] - md.beta_x[ix, iym]) / dy_eff
    dbetay_dy = (md.beta_y[ix, iyp] - md.beta_y[ix, iym]) / dy_eff
    dgxx_dy = (md.gamma_xx[ix, iyp] - md.gamma_xx[ix, iym]) / dy_eff
    dgxy_dy = (md.gamma_xy[ix, iyp] - md.gamma_xy[ix, iym]) / dy_eff
    dgyy_dy = (md.gamma_yy[ix, iyp] - md.gamma_yy[ix, iym]) / dy_eff

    # Lower the shift: beta_i = gamma_ij beta^j
    beta_x_low = gxx * bx_s + gxy * by_s
    beta_y_low = gxy * bx_s + gyy * by_s

    # 4-metric components
    # g_tt = -(alpha^2 - beta_k beta^k)
    beta_sq = bx_s * beta_x_low + by_s * beta_y_low
    g_tt = -alp^2 + beta_sq
    # g_tx = beta_x_low, g_ty = beta_y_low (lowered shift)

    # Derivatives of g_tt
    # d(g_tt)/dx = -2*alpha*dalpha_dx + d(beta_sq)/dx
    # d(beta_sq)/dx = 2*(beta^k * d(beta_k)/dx)  where beta_k = gamma_kj beta^j
    dbeta_x_low_dx = dgxx_dx * bx_s + gxx * dbetax_dx + dgxy_dx * by_s + gxy * dbetay_dx
    dbeta_y_low_dx = dgxy_dx * bx_s + gxy * dbetax_dx + dgyy_dx * by_s + gyy * dbetay_dx
    dbeta_sq_dx = bx_s * dbeta_x_low_dx + dbetax_dx * beta_x_low + by_s * dbeta_y_low_dx + dbetay_dx * beta_y_low
    dgtt_dx = -2 * alp * dalpha_dx + dbeta_sq_dx

    dbeta_x_low_dy = dgxx_dy * bx_s + gxx * dbetax_dy + dgxy_dy * by_s + gxy * dbetay_dy
    dbeta_y_low_dy = dgxy_dy * bx_s + gxy * dbetax_dy + dgyy_dy * by_s + gyy * dbetay_dy
    dbeta_sq_dy = bx_s * dbeta_x_low_dy + dbetax_dy * beta_x_low + by_s * dbeta_y_low_dy + dbetay_dy * beta_y_low
    dgtt_dy = -2 * alp * dalpha_dy + dbeta_sq_dy

    # Derivatives of g_ti (= beta_i lowered)
    dgtx_dx = dbeta_x_low_dx
    dgtx_dy = dbeta_x_low_dy
    dgty_dx = dbeta_y_low_dx
    dgty_dy = dbeta_y_low_dy

    # Compute spatial stress-energy tensor T^ij (contravariant)
    # b^i = B^i/W + b^0 v^i (contravariant magnetic 4-vector spatial components)
    b_x = Bx / W + b0 * vx
    b_y = By / W + b0 * vy

    # (ρh + b²)W² = ρhW² + B² + b⁰²
    wtot = rho_h_W2 + B_sq + b0^2

    gixx = md.gammaI_xx[ix, iy]
    gixy = md.gammaI_xy[ix, iy]
    giyy = md.gammaI_yy[ix, iy]

    Txx = wtot * vx * vx - b_x * b_x + Ptot * gixx
    Txy = wtot * vx * vy - b_x * b_y + Ptot * gixy
    Tyy = wtot * vy * vy - b_y * b_y + Ptot * giyy

    # Undensitized conserved quantities
    D_und = rho * W
    # τ = ρhW² + B² − P_tot − D
    tau_und = rho_h_W2 + B_sq - Ptot - D_und
    # S_j = (ρhW² + B²)v_j − (v·B)B_j
    Sx_und = (rho_h_W2 + B_sq) * vx - vdotB * Bx
    Sy_und = (rho_h_W2 + B_sq) * vy - vdotB * By

    # ---- Momentum source: S_{S_j} = sqrt(gamma) * [-(tau+D) partial_j alpha
    #                                  + S_k partial_j beta^k
    #                                  + alpha/2 * T^kl partial_j gamma_kl ] ----
    # Note: our densitized variables already include sqrt(gamma), so the source
    # fed into the update should NOT include an extra sqrt(gamma). We return
    # the source for the densitized system, which already has sqrt(gamma) in U.
    # i.e., dU/dt = ... + S, where S = sqrt(gamma) * source_undensitized.

    E_und = tau_und + D_und  # total energy density (undensitized)

    # Momentum source for x-direction (j=x):
    Sx_src = -E_und * dalpha_dx +
        Sx_und * dbetax_dx + Sy_und * dbetay_dx +
        alp * 0.5 * (Txx * dgxx_dx + 2 * Txy * dgxy_dx + Tyy * dgyy_dx)

    # Momentum source for y-direction (j=y):
    Sy_src = -E_und * dalpha_dy +
        Sx_und * dbetax_dy + Sy_und * dbetay_dy +
        alp * 0.5 * (Txx * dgxx_dy + 2 * Txy * dgxy_dy + Tyy * dgyy_dy)

    # ---- Energy source: S_tau = sqrt(gamma) * [-S_j partial^j alpha + alpha T^kl K_kl] ----
    # Approximate: K_ij ~ (partial_i beta_j_low + partial_j beta_i_low) / (2 alpha)
    # T^kl K_kl ~ T^kl (partial_k beta_l_low + partial_l beta_k_low) / (2 alpha)
    # = T^kl partial_k beta_l_low / alpha  (by symmetry)

    # partial_x beta_x_low, partial_y beta_x_low, partial_x beta_y_low, partial_y beta_y_low
    # already computed above as dbeta_x_low_dx, dbeta_x_low_dy, dbeta_y_low_dx, dbeta_y_low_dy

    TKij = Txx * dbeta_x_low_dx + Txy * (dbeta_y_low_dx + dbeta_x_low_dy) + Tyy * dbeta_y_low_dy

    # Raise S_j to get S^j = gamma^jk S_k for the alpha gradient term
    # -S_j partial^j alpha = -(S_x partial_x alpha + S_y partial_y alpha) in Cartesian
    # But partial^j alpha = gamma^jk partial_k alpha
    dalpha_up_x = gixx * dalpha_dx + gixy * dalpha_dy
    dalpha_up_y = gixy * dalpha_dx + giyy * dalpha_dy
    S_dot_grad_alpha = Sx_und * dalpha_up_x + Sy_und * dalpha_up_y

    tau_src = -S_dot_grad_alpha + TKij / alp

    # Sz source (for z-momentum): generally zero in 2D equatorial plane
    # unless there's a z-component of the flow (which can happen in MHD).
    # For our 2D-in-plane formulation, Sz source = 0 since there's no
    # z-dependence in the metric.
    Sz_src = zero(Sx_src)

    # Multiply by sqrt(gamma) for the densitized source
    return SVector(
        zero(Sx_src),
        sg * Sx_src, sg * Sy_src, sg * Sz_src, sg * tau_src,
        zero(Sx_src), zero(Sx_src), zero(Sx_src)
    )
end
