# ============================================================
# 2D GRMHD Solver with Constrained Transport + Source Terms
# ============================================================
#
# Specializes solve_hyperbolic for HyperbolicProblem2D with
# GRMHDEquations{2}. Extends the SRMHD CT solver with:
#
#   1. Precomputed metric data at cell centers and faces
#   2. Valencia flux correction: F_face = alpha * F_riemann - beta * U
#   3. Geometric source terms from the curved spacetime
#   4. Constrained transport for divergence-free B
#
# TWO PATHS, dispatched on the metric type:
#
# MINKOWSKI path (unchanged, bitwise-stable): the Riemann problem is
# solved in the flat frame with the problem's Riemann solver, the flux
# is corrected as alpha*F - beta*U (both trivial for Minkowski), and
# primitives are recovered with srmhd_con2prim. This path is exactly
# the SRMHD solver and remains the validated configuration.
#
# CURVED path (non-Minkowski metrics): the consistent Valencia
# formulation with DENSITIZED conserved state
#   U_tilde = sqrt(gamma) * [D, S_j (covariant), tau, B^j]
# - primitives are recovered per cell with the metric-aware
#   grmhd_con2prim (position-dependent spatial metric),
# - ghost cells are filled in PRIMITIVE space (Transmissive, Reflective,
#   Dirichlet/Inflow, Periodic),
# - face fluxes are the densitized Valencia fluxes
#   F_tilde = sqrt(gamma) [ alpha f - beta U ] combined with an HLL
#   Riemann fan using coordinate-frame wave speeds
#   lambda = alpha*lambda_flat(gamma^nn) - beta^n
#   (the problem's `riemann_solver` field is superseded by HLL on this
#   path — HLLC contact restoration is not formulated here for curved
#   spacetime),
# - geometric sources use the exact stationary-spacetime forms
#   (grmhd_source_terms), consistent with the densitized state,
# - CT evolves the densitized face field B_tilde = sqrt(gamma) B (the
#   GR solenoidal constraint is div(sqrt(gamma) B) = 0); vector-potential
#   initialization already produces B_tilde = curl(A) directly.
#
# Verified invariants for the curved path (see test/grmhd_2d.jl):
# a static polytropic atmosphere in Kerr-Schild Schwarzschild
# (h * sqrt(1 - 2M/r) = const, v^i = beta^i/alpha) is an exact
# stationary solution; the discrete RHS residual on that solution
# converges to zero with resolution, and the state is held for many
# steps with bounded, resolution-decreasing drift.

# ============================================================
# 2D ReflectiveBC for GRMHD
# ============================================================
# Handled by the generic ng-aware ReflectiveBC fallbacks in
# boundary_conditions_2d.jl via normal_velocity_index(::GRMHDEquations).
# (The previous law-specific methods here hard-coded a 2-ghost layout.)

# ============================================================
# CFL Calculation (Metric-Corrected)
# ============================================================

"""
    compute_dt_2d(prob::HyperbolicProblem2D{<:GRMHDEquations{2}}, U, t, md) -> dt

Compute the time step using metric-corrected wave speeds:
  `lambda_coord = alpha * lambda_flat - beta`
"""
function compute_dt_2d(
        prob::HyperbolicProblem2D{<:GRMHDEquations{2}},
        U::AbstractMatrix, t, md::MetricData2D
    )
    law = prob.law
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    cfl = prob.cfl
    dx, dy = mesh.dx, mesh.dy
    # Ghost layers from the padded array itself (the CT path pads 2 layers
    # regardless of the reconstruction's nghost).
    ng = (size(U, 1) - nx) ÷ 2

    curved = _grmhd_is_curved(law)
    max_speed = zero(dx)
    for iy in 1:ny, ix in 1:nx
        alp = md.alpha[ix, iy]
        bx_s = md.beta_x[ix, iy]
        by_s = md.beta_y[ix, iy]

        if curved
            w, result = grmhd_con2prim_cached(
                law, U[ix + ng, iy + ng], md.sqrtg[ix, iy],
                md.gammaI_xx[ix, iy], md.gammaI_xy[ix, iy], md.gammaI_yy[ix, iy],
                md.gamma_xx[ix, iy], md.gamma_xy[ix, iy], md.gamma_yy[ix, iy]
            )
            result.converged || _con2prim_convergence_error(
                "GRMHDEquations (curved CFL)", result, U[ix + ng, iy + ng]
            )
            gm = StaticArrays.SMatrix{2, 2}(
                md.gamma_xx[ix, iy], md.gamma_xy[ix, iy],
                md.gamma_xy[ix, iy], md.gamma_yy[ix, iy]
            )
            gi = StaticArrays.SMatrix{2, 2}(
                md.gammaI_xx[ix, iy], md.gammaI_xy[ix, iy],
                md.gammaI_xy[ix, iy], md.gammaI_yy[ix, iy]
            )
            lm_x, lp_x = _grmhd_coord_wave_speeds(law.eos, w, 1, alp, bx_s, gm, gi)
            lm_y, lp_y = _grmhd_coord_wave_speeds(law.eos, w, 2, alp, by_s, gm, gi)
            lam_x = max(abs(lm_x), abs(lp_x))
            lam_y = max(abs(lm_y), abs(lp_y))
        else
            w = conserved_to_primitive(law, U[ix + ng, iy + ng])
            lam_x = grmhd_max_wave_speed_coord(law, w, 1, alp, bx_s)
            lam_y = grmhd_max_wave_speed_coord(law, w, 2, alp, by_s)
        end
        speed = lam_x / dx + lam_y / dy
        max_speed = max(max_speed, speed)
    end

    if max_speed <= zero(max_speed)
        return zero(dx)
    end

    dt = cfl / max_speed

    if t + dt > prob.final_time
        dt = prob.final_time - t
    end

    return dt
end

# ============================================================
# Curved-spacetime consistent path (non-Minkowski metrics)
# ============================================================

"""
    _grmhd_is_curved(law::GRMHDEquations) -> Bool

True when the law's metric is not Minkowski, selecting the consistent
densitized Valencia path.
"""
_grmhd_is_curved(law::GRMHDEquations) = !(law.metric isa MinkowskiMetric)

"""
    _grmhd_valencia_flux(eos, w, dir, alp, beta, gm, sg) -> SVector{8}

Densitized Valencia flux `F_tilde^n = sqrt(gamma) [alpha f^n - beta^n U]`
in direction `dir` (1 = x, 2 = y) from primitives `w` and the face
metric (`alp` lapse, `beta` shift `SVector{2}`, `gm` spatial metric
`SMatrix{2,2}`, `sg = sqrt(det gamma)`).

Component forms (V^i = alpha v^i - beta^i is the transport velocity):

    F(D)    = D V^n
    F(S_j)  = S_j V^n + alpha P_tot delta^n_j - alpha b_j B^n / W
    F(tau)  = tau V^n + alpha P_tot v^n - alpha b^0 B^n / W
    F(B^j)  = B^j V^n - B^n V^j

with covariant `S_j`, `b_j` (lowered with `gamma_ij`), all multiplied by
`sqrt(gamma)`. For Minkowski data this reduces exactly to
`physical_flux(law, w, dir)`.
"""
@inline function _grmhd_valencia_flux(
        eos, w::SVector{8}, dir::Int,
        alp, beta::SVector{2}, gm::StaticArrays.SMatrix{2, 2}, sg
    )
    rho, vx, vy, vz, P, Bx, By, Bz = w
    gamma_eos = eos.gamma

    v_sq = gm[1, 1] * vx^2 + 2 * gm[1, 2] * vx * vy + gm[2, 2] * vy^2 + vz^2
    v_sq = min(v_sq, 1 - 1.0e-10)
    W = 1 / sqrt(1 - v_sq)

    vdotB = gm[1, 1] * vx * Bx + gm[1, 2] * (vx * By + vy * Bx) + gm[2, 2] * vy * By + vz * Bz
    B_sq = gm[1, 1] * Bx^2 + 2 * gm[1, 2] * Bx * By + gm[2, 2] * By^2 + Bz^2
    b0 = W * vdotB
    b_sq = B_sq / W^2 + vdotB^2

    eps_val = P / ((gamma_eos - 1) * rho)
    h = 1 + eps_val + P / rho
    rho_h_W2 = rho * h * W^2
    Ptot = P + 0.5 * b_sq
    D = rho * W

    # Lowered velocity and magnetic field
    vx_low = gm[1, 1] * vx + gm[1, 2] * vy
    vy_low = gm[1, 2] * vx + gm[2, 2] * vy
    vz_low = vz
    Bx_low = gm[1, 1] * Bx + gm[1, 2] * By
    By_low = gm[1, 2] * Bx + gm[2, 2] * By
    Bz_low = Bz

    # Covariant momentum and energy
    Sx = (rho_h_W2 + B_sq) * vx_low - vdotB * Bx_low
    Sy = (rho_h_W2 + B_sq) * vy_low - vdotB * By_low
    Sz = (rho_h_W2 + B_sq) * vz_low - vdotB * Bz_low
    tau = rho_h_W2 + B_sq - Ptot - D

    # Covariant spatial magnetic 4-vector b_j = B_j/W + b^0 v_j
    bx_low = Bx_low / W + b0 * vx_low
    by_low = By_low / W + b0 * vy_low
    bz_low = Bz_low / W + b0 * vz_low

    # Transport velocities V^i = alpha v^i - beta^i (V^z: beta^z = 0)
    Vx = alp * vx - beta[1]
    Vy = alp * vy - beta[2]
    Vz = alp * vz

    if dir == 1
        vn = vx
        Bn = Bx
        Vn = Vx
        aBnW = alp * Bn / W
        return sg * SVector(
            D * Vn,
            Sx * Vn + alp * Ptot - aBnW * bx_low,
            Sy * Vn - aBnW * by_low,
            Sz * Vn - aBnW * bz_low,
            tau * Vn + alp * Ptot * vn - alp * b0 * Bn / W,
            zero(Vn),
            By * Vn - Bn * Vy,
            Bz * Vn - Bn * Vz
        )
    else
        vn = vy
        Bn = By
        Vn = Vy
        aBnW = alp * Bn / W
        return sg * SVector(
            D * Vn,
            Sx * Vn - aBnW * bx_low,
            Sy * Vn + alp * Ptot - aBnW * by_low,
            Sz * Vn - aBnW * bz_low,
            tau * Vn + alp * Ptot * vn - alp * b0 * Bn / W,
            Bx * Vn - Bn * Vx,
            zero(Vn),
            Bz * Vn - Bn * Vz
        )
    end
end

"""
    _grmhd_coord_wave_speeds(eos, w, dir, alp, betan, gm, gi) -> (lam_minus, lam_plus)

Coordinate-frame fast magnetosonic wave speed bounds in direction `dir`:

    lambda = alpha/(1 - v^2 c^2) * [ v^n (1 - c^2)
             +/- c sqrt((1 - v^2)(gamma^nn (1 - v^2 c^2) - (v^n)^2 (1 - c^2))) ]
             - beta^n

with `v^2 = gamma_ij v^i v^j` and the fast-speed estimate
`c^2 = cs^2 + ca^2 - cs^2 ca^2` (metric-aware `b^2`). Reduces to the flat
`_grmhd_wave_speeds` for Minkowski.
"""
@inline function _grmhd_coord_wave_speeds(
        eos, w::SVector{8}, dir::Int,
        alp, betan, gm::StaticArrays.SMatrix{2, 2}, gi::StaticArrays.SMatrix{2, 2}
    )
    rho, vx, vy, vz, P, Bx, By, Bz = w
    gamma_eos = eos.gamma

    v_sq = gm[1, 1] * vx^2 + 2 * gm[1, 2] * vx * vy + gm[2, 2] * vy^2 + vz^2
    v_sq = min(v_sq, 1 - 1.0e-10)
    W_sq = 1 / (1 - v_sq)

    eps_val = P / ((gamma_eos - 1) * rho)
    h = 1 + eps_val + P / rho
    cs_sq = gamma_eos * P / (rho * h)
    cs_sq = min(cs_sq, 1 - 1.0e-10)

    vdotB = gm[1, 1] * vx * Bx + gm[1, 2] * (vx * By + vy * Bx) + gm[2, 2] * vy * By + vz * Bz
    B_sq = gm[1, 1] * Bx^2 + 2 * gm[1, 2] * Bx * By + gm[2, 2] * By^2 + Bz^2
    b_sq = B_sq / W_sq + vdotB^2

    ca_sq = b_sq / (rho * h + b_sq)
    ca_sq = min(ca_sq, 1 - 1.0e-10)
    c_ms_sq = cs_sq + ca_sq - cs_sq * ca_sq
    c_ms_sq = clamp(c_ms_sq, zero(c_ms_sq), 1 - 1.0e-10)
    c_ms = sqrt(c_ms_sq)

    vn = dir == 1 ? vx : vy
    ginn = dir == 1 ? gi[1, 1] : gi[2, 2]

    denom = 1 - v_sq * c_ms_sq
    discriminant = (1 - v_sq) * (ginn * (1 - v_sq * c_ms_sq) - vn^2 * (1 - c_ms_sq))
    discriminant = max(discriminant, zero(discriminant))
    sqrt_disc = sqrt(discriminant)

    lam_minus = alp * (vn * (1 - c_ms_sq) - c_ms * sqrt_disc) / denom - betan
    lam_plus = alp * (vn * (1 - c_ms_sq) + c_ms * sqrt_disc) / denom - betan
    return lam_minus, lam_plus
end

"""
    _grmhd_hll_flux_curved(law, wL, wR, dir, alp, beta, gm, gi, sg) -> SVector{8}

HLL flux for the densitized Valencia system at a face with the given
metric, from left/right primitive states.
"""
@inline function _grmhd_hll_flux_curved(
        law::GRMHDEquations{2}, wL::SVector{8}, wR::SVector{8}, dir::Int,
        alp, beta::SVector{2}, gm::StaticArrays.SMatrix{2, 2},
        gi::StaticArrays.SMatrix{2, 2}, sg
    )
    betan = dir == 1 ? beta[1] : beta[2]

    UL = _grmhd_prim2con_densitized(law.eos, wL, sg, gm)
    UR = _grmhd_prim2con_densitized(law.eos, wR, sg, gm)
    FL = _grmhd_valencia_flux(law.eos, wL, dir, alp, beta, gm, sg)
    FR = _grmhd_valencia_flux(law.eos, wR, dir, alp, beta, gm, sg)

    lmL, lpL = _grmhd_coord_wave_speeds(law.eos, wL, dir, alp, betan, gm, gi)
    lmR, lpR = _grmhd_coord_wave_speeds(law.eos, wR, dir, alp, betan, gm, gi)

    SL = min(lmL, lmR, zero(lmL))
    SR = max(lpL, lpR, zero(lpL))

    if SL >= zero(SL)
        return FL
    elseif SR <= zero(SR)
        return FR
    end
    return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)
end

"""
    _grmhd_recover_primitives_2d!(W_pad, U, law, mesh, ng)

Recover primitive variables at all interior cells from the densitized
conserved state `U` using the metric-aware con2prim with the metric
evaluated at each cell center. Ghost entries of `W_pad` are filled
separately by `_grmhd_fill_primitive_ghosts_2d!`.
"""
function _grmhd_recover_primitives_2d!(
        W_pad::AbstractMatrix, U::AbstractMatrix,
        law::GRMHDEquations{2}, mesh::StructuredMesh2D, ng::Int
    )
    metric = law.metric
    nx, ny = mesh.nx, mesh.ny
    for iy in 1:ny, ix in 1:nx
        x, y = cell_center(mesh, cell_idx(mesh, ix, iy))
        sg = sqrt_gamma(metric, x, y)
        gm = spatial_metric(metric, x, y)
        gi = inv_spatial_metric(metric, x, y)
        w, result = grmhd_con2prim_cached(
            law, U[ix + ng, iy + ng], sg,
            gi[1, 1], gi[1, 2], gi[2, 2],
            gm[1, 1], gm[1, 2], gm[2, 2]
        )
        result.converged || _con2prim_convergence_error(
            "GRMHDEquations (curved con2prim)", result, U[ix + ng, iy + ng]
        )
        W_pad[ix + ng, iy + ng] = w
    end
    return nothing
end

"""
    _grmhd_fill_primitive_ghosts_2d!(W_pad, prob, ng)

Fill ghost cells of the padded PRIMITIVE array for the curved GRMHD
path. Supported boundary conditions: `TransmissiveBC` (zero-gradient in
primitives), `ReflectiveBC` (normal velocity negated),
`DirichletHyperbolicBC`/`InflowBC` (prescribed primitive state), and
`PeriodicHyperbolicBC`.
"""
function _grmhd_fill_primitive_ghosts_2d!(
        W_pad::AbstractMatrix, prob::HyperbolicProblem2D{<:GRMHDEquations{2}}, ng::Int
    )
    nx, ny = prob.mesh.nx, prob.mesh.ny

    ghost_state(bc::TransmissiveBC, w_int, dir) = w_int
    ghost_state(bc::ReflectiveBC, w_int, dir) =
        Base.setindex(w_int, -w_int[dir + 1], dir + 1)
    ghost_state(bc::DirichletHyperbolicBC, w_int, dir) = bc.state
    ghost_state(bc::InflowBC, w_int, dir) = bc.state
    ghost_state(bc, w_int, dir) = throw(
        ArgumentError(
            "curved-spacetime GRMHD path: unsupported boundary condition " *
                "$(typeof(bc)); use TransmissiveBC, ReflectiveBC, " *
                "DirichletHyperbolicBC, InflowBC, or PeriodicHyperbolicBC."
        )
    )

    # x-direction (interior rows)
    if prob.bc_left isa PeriodicHyperbolicBC && prob.bc_right isa PeriodicHyperbolicBC
        for jj in (ng + 1):(ny + ng), g in 1:ng
            W_pad[ng + 1 - g, jj] = W_pad[nx + ng + 1 - g, jj]
            W_pad[nx + ng + g, jj] = W_pad[ng + g, jj]
        end
    else
        for jj in (ng + 1):(ny + ng), g in 1:ng
            W_pad[ng + 1 - g, jj] = ghost_state(prob.bc_left, W_pad[ng + g, jj], 1)
            W_pad[nx + ng + g, jj] = ghost_state(prob.bc_right, W_pad[nx + ng + 1 - g, jj], 1)
        end
    end

    # y-direction (all padded columns, so corners get consistent data)
    if prob.bc_bottom isa PeriodicHyperbolicBC && prob.bc_top isa PeriodicHyperbolicBC
        for ii in 1:(nx + 2 * ng), g in 1:ng
            W_pad[ii, ng + 1 - g] = W_pad[ii, ny + ng + 1 - g]
            W_pad[ii, ny + ng + g] = W_pad[ii, ng + g]
        end
    else
        for ii in 1:(nx + 2 * ng), g in 1:ng
            W_pad[ii, ng + 1 - g] = ghost_state(prob.bc_bottom, W_pad[ii, ng + g], 2)
            W_pad[ii, ny + ng + g] = ghost_state(prob.bc_top, W_pad[ii, ny + ng + 1 - g], 2)
        end
    end

    return nothing
end

"""
    _grmhd_reconstruct_primitive_pair_x(recon, W_pad, iL, iR, jj) -> (wL, wR)
    _grmhd_reconstruct_primitive_pair_y(recon, W_pad, ii, jL, jR) -> (wL, wR)

Reconstruct left/right primitive face states directly from the padded
primitive array (`NoReconstruction` copies the adjacent cells; 4-point
schemes such as MUSCL/WENO3 go through `reconstruct_interface`).
"""
@inline function _grmhd_reconstruct_primitive_pair_x(recon, W_pad, iL, iR, jj)
    if recon isa NoReconstruction
        return W_pad[iL, jj], W_pad[iR, jj]
    end
    return reconstruct_interface(recon, W_pad[iL - 1, jj], W_pad[iL, jj], W_pad[iR, jj], W_pad[iR + 1, jj])
end

@inline function _grmhd_reconstruct_primitive_pair_y(recon, W_pad, ii, jL, jR)
    if recon isa NoReconstruction
        return W_pad[ii, jL], W_pad[ii, jR]
    end
    return reconstruct_interface(recon, W_pad[ii, jL - 1], W_pad[ii, jL], W_pad[ii, jR], W_pad[ii, jR + 1])
end

"""
    _grmhd_compute_fluxes_curved_2d!(Fx_all, Fy_all, dU, U, W_pad, prob, t)

Curved-path flux computation: recovers nothing itself (expects `W_pad`
current), computes densitized Valencia HLL fluxes at every face with
the metric evaluated analytically at the face center, and accumulates
the flux-difference part of `dU` at interior cells.
"""
function _grmhd_compute_fluxes_curved_2d!(
        Fx_all::AbstractMatrix, Fy_all::AbstractMatrix,
        dU::AbstractMatrix, U::AbstractMatrix, W_pad::AbstractMatrix,
        prob::HyperbolicProblem2D{<:GRMHDEquations{2}}, t
    )
    law = prob.law
    metric = law.metric
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy
    recon = prob.reconstruction
    N = nvariables(law)
    ng = (size(U, 1) - nx) ÷ 2
    FT = eltype(W_pad[ng + 1, ng + 1])

    zero_state = zero(SVector{N, FT})
    for iy in 1:ny, ix in 1:nx
        dU[ix + ng, iy + ng] = zero_state
    end

    # X-direction faces (including one ghost row each side for the EMF)
    for row_idx in 1:(ny + 2)
        jj = row_idx + ng - 1
        yf = mesh.ymin + (row_idx - 1 - FT(0.5)) * dy
        for face_i in 1:(nx + 1)
            iL = face_i + ng - 1
            iR = iL + 1
            xf = mesh.xmin + (face_i - 1) * dx

            wL, wR = _grmhd_reconstruct_primitive_pair_x(recon, W_pad, iL, iR, jj)

            alp = lapse(metric, xf, yf)
            beta = shift(metric, xf, yf)
            gm = spatial_metric(metric, xf, yf)
            gi = inv_spatial_metric(metric, xf, yf)
            sg = sqrt_gamma(metric, xf, yf)

            Fx_all[face_i, row_idx] = _grmhd_hll_flux_curved(law, wL, wR, 1, alp, beta, gm, gi, sg)
        end
    end

    # Y-direction faces
    for col_idx in 1:(nx + 2)
        ii = col_idx + ng - 1
        xf = mesh.xmin + (col_idx - 1 - FT(0.5)) * dx
        for face_j in 1:(ny + 1)
            jL = face_j + ng - 1
            jR = jL + 1
            yf = mesh.ymin + (face_j - 1) * dy

            wL, wR = _grmhd_reconstruct_primitive_pair_y(recon, W_pad, ii, jL, jR)

            alp = lapse(metric, xf, yf)
            beta = shift(metric, xf, yf)
            gm = spatial_metric(metric, xf, yf)
            gi = inv_spatial_metric(metric, xf, yf)
            sg = sqrt_gamma(metric, xf, yf)

            Fy_all[col_idx, face_j] = _grmhd_hll_flux_curved(law, wL, wR, 2, alp, beta, gm, gi, sg)
        end
    end

    # Accumulate dU from stored fluxes
    for iy in 1:ny, ix in 1:nx
        F_right = Fx_all[ix + 1, iy + 1]
        F_left = Fx_all[ix, iy + 1]
        G_top = Fy_all[ix + 1, iy + 1]
        G_bottom = Fy_all[ix + 1, iy]
        dU[ix + ng, iy + ng] = -(F_right - F_left) / dx - (G_top - G_bottom) / dy
    end

    return nothing
end

"""
    _grmhd_add_source_terms_curved!(dU, W_pad, law, md, mesh, nx, ny, ng)

Add geometric source terms on the curved path, using primitives from
the metric-aware recovery (`W_pad`) rather than the flat con2prim.
"""
function _grmhd_add_source_terms_curved!(
        dU::AbstractMatrix, W_pad::AbstractMatrix,
        law::GRMHDEquations{2}, md::MetricData2D,
        mesh::StructuredMesh2D, nx::Int, ny::Int, ng::Int
    )
    for iy in 1:ny, ix in 1:nx
        ii, jj = ix + ng, iy + ng
        w = W_pad[ii, jj]
        S = grmhd_source_terms(law, w, W_pad[ii, jj], md, mesh, ix, iy)
        dU[ii, jj] = dU[ii, jj] + S
    end
    return nothing
end

"""
    _grmhd_stage_rhs!(Fx_all, Fy_all, dU, U, W_pad, prob, t, md, face_data)

One-stop RHS evaluation for a GRMHD RK stage, dispatching on the metric:

- Minkowski: existing flat path (`_grmhd_compute_fluxes_2d!` with the
  problem's Riemann solver + flat sources) — bitwise identical to the
  historical behavior.
- Curved: metric-aware primitive recovery into `W_pad`, primitive-space
  ghost fill, densitized Valencia HLL fluxes, and exact stationary
  geometric sources.
"""
function _grmhd_stage_rhs!(
        Fx_all::AbstractMatrix, Fy_all::AbstractMatrix,
        dU::AbstractMatrix, U::AbstractMatrix,
        W_pad::Union{Nothing, AbstractMatrix},
        prob::HyperbolicProblem2D{<:GRMHDEquations{2}}, t,
        md::MetricData2D, face_data
    )
    law = prob.law
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    ng = (size(U, 1) - nx) ÷ 2
    if _grmhd_is_curved(law)
        _grmhd_recover_primitives_2d!(W_pad, U, law, mesh, ng)
        _grmhd_fill_primitive_ghosts_2d!(W_pad, prob, ng)
        _grmhd_compute_fluxes_curved_2d!(Fx_all, Fy_all, dU, U, W_pad, prob, t)
        _grmhd_add_source_terms_curved!(dU, W_pad, law, md, mesh, nx, ny, ng)
    else
        _grmhd_compute_fluxes_2d!(Fx_all, Fy_all, dU, U, prob, t, md, face_data)
        _grmhd_add_source_terms!(dU, U, law, md, mesh, nx, ny, ng)
    end
    return nothing
end

"""
    _grmhd_initialize_densitized_2d!(U, prob, ng)

Overwrite the interior of the padded state array with the DENSITIZED
Valencia conserved variables computed from the problem's primitive
initial condition and the cell-centered metric (curved path only).
"""
function _grmhd_initialize_densitized_2d!(
        U::AbstractMatrix, prob::HyperbolicProblem2D{<:GRMHDEquations{2}}, ng::Int
    )
    law = prob.law
    metric = law.metric
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    for iy in 1:ny, ix in 1:nx
        x, y = cell_center(mesh, cell_idx(mesh, ix, iy))
        w = prob.initial_condition(x, y)
        sg = sqrt_gamma(metric, x, y)
        gm = spatial_metric(metric, x, y)
        U[ix + ng, iy + ng] = _grmhd_prim2con_densitized(law.eos, w, sg, gm)
    end
    return nothing
end

"""
    _grmhd_densitize_ct_faces!(ct, metric, mesh)

Scale directly-sampled face-centered B by `sqrt(gamma)` at each face so
the CT machinery evolves the densitized field `B_tilde = sqrt(gamma) B`
(the GR solenoidal constraint is `div(B_tilde) = 0`). NOT applied for
vector-potential initialization, whose curl already yields `B_tilde`.
"""
function _grmhd_densitize_ct_faces!(ct, metric, mesh::StructuredMesh2D)
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy
    for j in 1:ny, i in 1:(nx + 1)
        xf = mesh.xmin + (i - 1) * dx
        yf = mesh.ymin + (j - 0.5) * dy
        ct.Bx_face[i, j] *= sqrt_gamma(metric, xf, yf)
    end
    for j in 1:(ny + 1), i in 1:nx
        xf = mesh.xmin + (i - 0.5) * dx
        yf = mesh.ymin + (j - 1) * dy
        ct.By_face[i, j] *= sqrt_gamma(metric, xf, yf)
    end
    return nothing
end

"""
    grmhd_recover_primitive_field(law, U_interior, mesh) -> Matrix{SVector{8}}

Recover the primitive field from an interior conserved-state matrix as
returned by `solve_hyperbolic`. Uses the metric-aware con2prim for
non-Minkowski metrics (where the state is densitized) and the flat
recovery for Minkowski.
"""
function grmhd_recover_primitive_field(
        law::GRMHDEquations{2}, U_interior::AbstractMatrix, mesh::StructuredMesh2D
    )
    nx, ny = size(U_interior)
    if !_grmhd_is_curved(law)
        return [conserved_to_primitive(law, U_interior[ix, iy]) for ix in 1:nx, iy in 1:ny]
    end
    metric = law.metric
    W = similar(U_interior)
    for iy in 1:ny, ix in 1:nx
        x, y = cell_center(mesh, cell_idx(mesh, ix, iy))
        sg = sqrt_gamma(metric, x, y)
        gm = spatial_metric(metric, x, y)
        gi = inv_spatial_metric(metric, x, y)
        w, result = grmhd_con2prim_cached(
            law, U_interior[ix, iy], sg,
            gi[1, 1], gi[1, 2], gi[2, 2],
            gm[1, 1], gm[1, 2], gm[2, 2]
        )
        result.converged || _con2prim_convergence_error(
            "grmhd_recover_primitive_field", result, U_interior[ix, iy]
        )
        W[ix, iy] = w
    end
    return W
end

# ============================================================
# Compute Fluxes with Valencia Correction
# ============================================================

"""
    _grmhd_compute_fluxes_2d!(Fx_all, Fy_all, dU, U, prob, t, md, face_data)

Compute face fluxes with the Valencia metric correction and accumulate dU.

The Valencia flux at each face is:
  F_Valencia = alpha_face * F_Riemann - beta_face * U_face

where U_face is the upwind densitized state and F_Riemann is the flat-space
Riemann flux.
"""
function _grmhd_compute_fluxes_2d!(
        Fx_all::AbstractMatrix, Fy_all::AbstractMatrix,
        dU::AbstractMatrix, U::AbstractMatrix,
        prob::HyperbolicProblem2D{<:GRMHDEquations{2}}, t,
        md::MetricData2D, face_data
    )
    law = prob.law
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy
    solver = prob.riemann_solver
    recon = prob.reconstruction
    N = nvariables(law)
    # Ghost layers from the padded array itself (legacy path pads with the
    # reconstruction's nghost, the SciML cache always pads 2).
    ng = (size(U, 1) - nx) ÷ 2
    FT = eltype(U[ng + 1, ng + 1])

    # Unpack face metric data
    alpha_xf, alpha_yf, betax_xf, betay_xf, betax_yf, betay_yf, sqrtg_xf, sqrtg_yf = face_data

    # Apply BCs to fill ghost cells
    apply_boundary_conditions_2d!(U, prob, ng, t)

    # Zero dU for interior cells
    zero_state = zero(SVector{N, FT})
    for iy in 1:ny, ix in 1:nx
        dU[ix + ng, iy + ng] = zero_state
    end

    # ---- X-direction sweeps (including ghost rows for EMF) ----
    for row_idx in 1:(ny + 2)
        jj = row_idx + ng - 1  # padded j index
        for face_i in 1:(nx + 1)
            iL = face_i + ng - 1
            iR = iL + 1
            wL_face, wR_face = _reconstruct_face_2d(recon, law, U, iL, iR, jj, 1, nx)

            # Flat-space Riemann flux
            F_riemann = solve_riemann(solver, law, wL_face, wR_face, 1)

            # Valencia correction at this face
            # For ghost rows (row_idx=1 or ny+2), use nearest interior face metric
            j_metric = clamp(row_idx - 1, 1, ny)
            alp_f = alpha_xf[face_i, j_metric]
            bx_f = betax_xf[face_i, j_metric]
            sg_f = sqrtg_xf[face_i, j_metric]

            # Average state for the beta*U term (use Roe-like average: arithmetic mean)
            U_avg = 0.5 * (U[iL, jj] + U[iR, jj])

            # Valencia flux: alpha * F_flat - beta^x * U_tilde
            # Note: F_riemann is in undensitized form (from primitive Riemann solver),
            # so we need: F_Valencia = sg * (alpha * F_riemann) - beta^x * U_tilde
            # where U_tilde = sg * U_undensitized = the stored U (which IS densitized in the solver)
            # Actually in our framework, the Riemann solver works with undensitized primitives
            # and returns the undensitized flux. The stored U in the padded array is also
            # undensitized (for compatibility with the existing BC and reconstruction infrastructure).
            # The densitization is handled in the flux correction and source terms.
            Fx_all[face_i, row_idx] = alp_f * F_riemann - bx_f * U_avg
        end
    end

    # ---- Y-direction sweeps (including ghost columns for EMF) ----
    for col_idx in 1:(nx + 2)
        ii = col_idx + ng - 1  # padded i index
        for face_j in 1:(ny + 1)
            jL = face_j + ng - 1
            jR = jL + 1
            wL_face, wR_face = _reconstruct_face_2d_y(recon, law, U, ii, jL, jR, ny)

            F_riemann = solve_riemann(solver, law, wL_face, wR_face, 2)

            i_metric = clamp(col_idx - 1, 1, nx)
            alp_f = alpha_yf[i_metric, face_j]
            by_f = betay_yf[i_metric, face_j]
            sg_f = sqrtg_yf[i_metric, face_j]

            U_avg = 0.5 * (U[ii, jL] + U[ii, jR])

            Fy_all[col_idx, face_j] = alp_f * F_riemann - by_f * U_avg
        end
    end

    # ---- Accumulate dU from stored fluxes ----
    for iy in 1:ny, ix in 1:nx
        F_right = Fx_all[ix + 1, iy + 1]
        F_left = Fx_all[ix, iy + 1]
        G_top = Fy_all[ix + 1, iy + 1]
        G_bottom = Fy_all[ix + 1, iy]
        dU[ix + ng, iy + ng] = -(F_right - F_left) / dx - (G_top - G_bottom) / dy
    end

    return nothing
end

# ============================================================
# Add Geometric Source Terms
# ============================================================

"""
    _grmhd_add_source_terms!(dU, U, law, md, mesh, nx, ny, ng)

Add geometric source terms to dU at all interior cells.
"""
function _grmhd_add_source_terms!(
        dU::AbstractMatrix, U::AbstractMatrix,
        law::GRMHDEquations{2}, md::MetricData2D,
        mesh::StructuredMesh2D, nx::Int, ny::Int, ng::Int
    )
    for iy in 1:ny, ix in 1:nx
        ii, jj = ix + ng, iy + ng
        w = conserved_to_primitive(law, U[ii, jj])
        S = grmhd_source_terms(law, w, U[ii, jj], md, mesh, ix, iy)
        dU[ii, jj] = dU[ii, jj] + S
    end
    return nothing
end

# ============================================================
# 2D GRMHD Solver with CT and Source Terms
# ============================================================

"""
    solve_hyperbolic(prob::HyperbolicProblem2D{<:GRMHDEquations{2}};
                     method=:ssprk3, vector_potential=nothing)
        -> (coords, U_final, t_final, ct)

Solve the 2D GRMHD problem using the Valencia formulation with:
- Constrained transport for divergence-free B
- Metric-corrected fluxes (Valencia flux: alpha*F - beta*U)
- Geometric source terms from spacetime curvature

# Returns
- `coords`: Cell center coordinates `(x, y)` matrix.
- `U_final`: Final conserved variable matrix (nx x ny, undensitized).
- `t_final`: Final time reached.
- `ct`: Final `CTData2D` for inspecting div(B) and face-centered B.
"""
# Internal reference implementation (unexported since v4.0): fixed-step loops
# kept for threaded (`parallel = true`) execution, GPU backend dispatch, and
# the CPU-vs-CUDA parity baseline. The public execution path is
# `sciml_problem(prob)` + `solve`.
function solve_hyperbolic(
        prob::HyperbolicProblem2D{<:GRMHDEquations{2}};
        method::Symbol = :ssprk3,
        vector_potential = nothing,
        callback::Union{Nothing, Function} = nothing,
        backend::AbstractBackend = CPUBackend(),
    )
    _cpu_backend_only("solve_hyperbolic(::HyperbolicProblem2D{<:GRMHDEquations{2}})", backend)
    _validate_ct_reconstruction(prob.reconstruction)
    _grmhd_curved_path_note(prob.law)
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy
    law = prob.law
    N = nvariables(law)

    # Initialize cell-centered solution (padded array). The CT machinery
    # (face_to_cell_B!, EMF loops) assumes a fixed 2-ghost layout.
    # Minkowski path: undensitized flat-form state (historical behavior).
    # Curved path: DENSITIZED Valencia state sqrt(gamma)*[D, S_j, tau, B].
    ng = 2
    curved = _grmhd_is_curved(law)
    U = initialize_2d(prob; nghost = ng)
    FT = eltype(U[ng + 1, ng + 1])
    if curved
        _grmhd_initialize_densitized_2d!(U, prob, ng)
    end
    W_pad = curved ? fill(zero(SVector{N, FT}), size(U)) : nothing

    # Precompute metric data at cell centers and faces
    md = precompute_metric(law.metric, mesh)
    face_data = precompute_metric_at_faces(law.metric, mesh)

    # Initialize CT data (face-centered B). On the curved path the CT field
    # is the densitized B_tilde = sqrt(gamma) B: direct face sampling of the
    # primitive B must be scaled, while curl(A) already yields B_tilde.
    ct = CTData2D(nx, ny, FT)
    if vector_potential !== nothing
        initialize_ct_from_potential!(ct, vector_potential, mesh)
    else
        initialize_ct!(ct, prob, mesh)
        if curved
            _grmhd_densitize_ct_faces!(ct, law.metric, mesh)
        end
    end
    face_to_cell_B!(U, ct, nx, ny)

    # Allocate extended face flux arrays
    zero_flux = zero(SVector{N, FT})
    Fx_all = fill(zero_flux, nx + 1, ny + 2)
    Fy_all = fill(zero_flux, nx + 2, ny + 1)

    # Allocate dU
    dU = similar(U)
    zero_state = zero(SVector{N, FT})
    for j in axes(dU, 2), i in axes(dU, 1)
        dU[i, j] = zero_state
    end

    t = prob.initial_time
    step = 0

    if method == :euler
        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt_2d(prob, U, t, md)
            if dt <= zero(dt)
                break
            end

            # Compute metric-corrected fluxes, dU, and geometric sources
            _grmhd_stage_rhs!(Fx_all, Fy_all, dU, U, W_pad, prob, t, md, face_data)

            # Update all conserved variables
            for iy in 1:ny, ix in 1:nx
                ii, jj = ix + ng, iy + ng
                U[ii, jj] = U[ii, jj] + dt * dU[ii, jj]
            end

            # CT: compute EMF and update face-centered B
            _compute_emf_from_extended!(ct.emf_z, Fx_all, Fy_all, nx, ny)
            ct_update!(ct, dt, dx, dy, nx, ny)
            apply_ct_periodic!(ct, prob, nx, ny)
            face_to_cell_B!(U, ct, nx, ny)

            t += dt
            step += 1
            if callback !== nothing
                callback(U, t, step, dt)
            end
        end

    elseif method == :ssprk3
        # Allocate RK stage arrays
        U1 = similar(U)
        U2 = similar(U)
        for j in axes(U1, 2), i in axes(U1, 1)
            U1[i, j] = zero_state
            U2[i, j] = zero_state
        end

        # CT data for RK stages
        ct0 = CTData2D(nx, ny, FT)
        ct1 = CTData2D(nx, ny, FT)
        ct2 = CTData2D(nx, ny, FT)

        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt_2d(prob, U, t, md)
            if dt <= zero(dt)
                break
            end

            copyto_ct!(ct0, ct)

            # ---- Stage 1: U1 = U + dt * L(U) ----
            _grmhd_stage_rhs!(Fx_all, Fy_all, dU, U, W_pad, prob, t, md, face_data)
            for iy in 1:ny, ix in 1:nx
                ii, jj = ix + ng, iy + ng
                U1[ii, jj] = U[ii, jj] + dt * dU[ii, jj]
            end
            _compute_emf_from_extended!(ct.emf_z, Fx_all, Fy_all, nx, ny)
            copyto_ct!(ct1, ct)
            ct_update!(ct1, dt, dx, dy, nx, ny)
            apply_ct_periodic!(ct1, prob, nx, ny)
            face_to_cell_B!(U1, ct1, nx, ny)

            # ---- Stage 2: U2 = 3/4*U + 1/4*(U1 + dt*L(U1)) ----
            curved || apply_boundary_conditions_2d!(U1, prob, ng, t + dt)
            _grmhd_stage_rhs!(Fx_all, Fy_all, dU, U1, W_pad, prob, t + dt, md, face_data)
            for iy in 1:ny, ix in 1:nx
                ii, jj = ix + ng, iy + ng
                U2[ii, jj] = 0.75 * U[ii, jj] + 0.25 * (U1[ii, jj] + dt * dU[ii, jj])
            end
            _compute_emf_from_extended!(ct1.emf_z, Fx_all, Fy_all, nx, ny)
            ct_weighted_update!(ct2, ct0, ct1, 0.75, 0.25, dt, dx, dy, nx, ny)
            apply_ct_periodic!(ct2, prob, nx, ny)
            face_to_cell_B!(U2, ct2, nx, ny)

            # ---- Stage 3: U = 1/3*U + 2/3*(U2 + dt*L(U2)) ----
            curved || apply_boundary_conditions_2d!(U2, prob, ng, t + 0.5 * dt)
            _grmhd_stage_rhs!(Fx_all, Fy_all, dU, U2, W_pad, prob, t + 0.5 * dt, md, face_data)
            for iy in 1:ny, ix in 1:nx
                ii, jj = ix + ng, iy + ng
                U[ii, jj] = (1.0 / 3.0) * U[ii, jj] + (2.0 / 3.0) * (U2[ii, jj] + dt * dU[ii, jj])
            end
            _compute_emf_from_extended!(ct2.emf_z, Fx_all, Fy_all, nx, ny)
            ct_weighted_update!(ct, ct0, ct2, 1.0 / 3.0, 2.0 / 3.0, dt, dx, dy, nx, ny)
            apply_ct_periodic!(ct, prob, nx, ny)
            face_to_cell_B!(U, ct, nx, ny)

            t += dt
            step += 1
            if callback !== nothing
                callback(U, t, step, dt)
            end
        end
    else
        error("Unknown time integration method: $method. Use :euler or :ssprk3.")
    end

    # Extract interior solution as nx x ny matrix
    U_interior = Matrix{SVector{N, FT}}(undef, nx, ny)
    for iy in 1:ny, ix in 1:nx
        U_interior[ix, iy] = U[ix + ng, iy + ng]
    end

    # Cell center coordinates
    coords = [(cell_center(mesh, cell_idx(mesh, ix, iy))) for ix in 1:nx, iy in 1:ny]

    return coords, U_interior, t, ct
end
