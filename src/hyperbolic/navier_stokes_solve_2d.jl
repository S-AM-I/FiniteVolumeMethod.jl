# ============================================================
# 2D Navier-Stokes Solver
# ============================================================
#
# Overrides compute_dt_2d and hyperbolic_rhs_2d! for problems with
# NavierStokesEquations{2}. The existing solve_hyperbolic time
# loop for HyperbolicProblem2D calls these via dispatch.

"""
    compute_dt_2d(prob::HyperbolicProblem2D{<:NavierStokesEquations{2}}, U, t) -> dt

Compute the time step accounting for both hyperbolic and viscous stability:
- Hyperbolic: `dt_hyp = cfl / (max(|λx|)/dx + max(|λy|)/dy)`
- Viscous: `dt_visc = 0.5 · ρ_min / (μ · (1/dx² + 1/dy²))`
"""
function compute_dt_2d(prob::HyperbolicProblem2D{<:NavierStokesEquations{2}}, U::AbstractMatrix, t)
    law = prob.law
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    cfl = prob.cfl
    dx, dy = mesh.dx, mesh.dy
    μ = law.mu

    max_speed = zero(dx)
    ρ_min = typeof(dx)(Inf)
    for iy in 1:ny, ix in 1:nx
        w = conserved_to_primitive(law, U[ix + 2, iy + 2])
        λx = max_wave_speed(law, w, 1)
        λy = max_wave_speed(law, w, 2)
        speed = λx / dx + λy / dy
        max_speed = max(max_speed, speed)
        ρ_min = min(ρ_min, w[1])
    end

    dt_hyp = 1.0 / max_speed

    dt = dt_hyp
    if μ > 0
        dt_visc = 0.5 * ρ_min / (μ * (1.0 / dx^2 + 1.0 / dy^2))
        dt = min(dt_hyp, dt_visc)
    end

    dt = cfl * dt

    # Don't overshoot final time
    if t + dt > prob.final_time
        dt = prob.final_time - t
    end

    return dt
end

"""
    hyperbolic_rhs_2d!(dU, U, prob::HyperbolicProblem2D{<:NavierStokesEquations{2}}, t)

Compute the 2D RHS including inviscid and viscous fluxes:
  `dU[i,j]/dt = -1/dx * (Fx_{i+1/2,j} - Fx_{i-1/2,j})
                -1/dy * (Fy_{i,j+1/2} - Fy_{i,j-1/2})
                +1/dx * (Fvx_{i+1/2,j} - Fvx_{i-1/2,j})
                +1/dy * (Fvy_{i,j+1/2} - Fvy_{i,j-1/2})`
"""
function hyperbolic_rhs_2d!(
        dU::AbstractMatrix, U::AbstractMatrix,
        prob::HyperbolicProblem2D{<:NavierStokesEquations{2}}, t
    )
    law = prob.law
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy
    solver = prob.riemann_solver
    recon = prob.reconstruction

    # Apply BCs to fill ghost cells
    apply_boundary_conditions_2d!(U, prob, t)

    N = nvariables(law)
    FT = eltype(U[3, 3])

    # Zero out dU for interior cells
    zero_state = zero(SVector{N, FT})
    for iy in 1:ny, ix in 1:nx
        dU[ix + 2, iy + 2] = zero_state
    end

    # ---- Inviscid fluxes (same as Euler) ----

    # X-direction sweeps
    for iy in 1:ny
        jj = iy + 2
        for ix in 0:nx
            iL = ix + 2
            iR = ix + 3
            wL_face, wR_face = _reconstruct_face_2d(recon, law, U, iL, iR, jj, 1, nx)
            F = solve_riemann(solver, law, wL_face, wR_face, 1)
            if ix >= 1
                dU[iL, jj] = dU[iL, jj] - F / dx
            end
            if ix < nx
                dU[iR, jj] = dU[iR, jj] + F / dx
            end
        end
    end

    # Y-direction sweeps
    for ix in 1:nx
        ii = ix + 2
        for iy in 0:ny
            jL = iy + 2
            jR = iy + 3
            wL_face, wR_face = _reconstruct_face_2d_y(recon, law, U, ii, jL, jR, ny)
            F = solve_riemann(solver, law, wL_face, wR_face, 2)
            if iy >= 1
                dU[ii, jL] = dU[ii, jL] - F / dy
            end
            if iy < ny
                dU[ii, jR] = dU[ii, jR] + F / dy
            end
        end
    end

    # ---- Viscous fluxes ----

    # Pre-compute primitive state array for cross-derivative access
    W = Matrix{SVector{N, FT}}(undef, nx + 4, ny + 4)
    for j in 1:(ny + 4), i in 1:(nx + 4)
        W[i, j] = conserved_to_primitive(law, U[i, j])
    end

    # X-direction viscous fluxes: face between (ix, iy) and (ix+1, iy)
    # Padded: face between U[ix+2, iy+2] and U[ix+3, iy+2]
    for iy in 1:ny
        jj = iy + 2
        for ix in 0:nx
            iL = ix + 2
            iR = ix + 3

            wL = W[iL, jj]
            wR = W[iR, jj]

            # Cross-derivatives via 4-cell average
            # ∂vx/∂y ≈ (vx[iL,j+1] + vx[iR,j+1] - vx[iL,j-1] - vx[iR,j-1]) / (4*dy)
            dvx_dy = (
                W[iL, jj + 1][2] + W[iR, jj + 1][2] -
                    W[iL, jj - 1][2] - W[iR, jj - 1][2]
            ) / (4.0 * dy)
            dvy_dy = (
                W[iL, jj + 1][3] + W[iR, jj + 1][3] -
                    W[iL, jj - 1][3] - W[iR, jj - 1][3]
            ) / (4.0 * dy)

            Fv = viscous_flux_x_2d(law, wL, wR, dvx_dy, dvy_dy, dx)

            if ix >= 1
                dU[iL, jj] = dU[iL, jj] + Fv / dx
            end
            if ix < nx
                dU[iR, jj] = dU[iR, jj] - Fv / dx
            end
        end
    end

    # Y-direction viscous fluxes: face between (ix, iy) and (ix, iy+1)
    # Padded: face between U[ix+2, iy+2] and U[ix+2, iy+3]
    for ix in 1:nx
        ii = ix + 2
        for iy in 0:ny
            jL = iy + 2
            jR = iy + 3

            wB = W[ii, jL]
            wT = W[ii, jR]

            # Cross-derivatives via 4-cell average
            # ∂vx/∂x ≈ (vx[i+1,jL] + vx[i+1,jR] - vx[i-1,jL] - vx[i-1,jR]) / (4*dx)
            dvx_dx = (
                W[ii + 1, jL][2] + W[ii + 1, jR][2] -
                    W[ii - 1, jL][2] - W[ii - 1, jR][2]
            ) / (4.0 * dx)
            dvy_dx = (
                W[ii + 1, jL][3] + W[ii + 1, jR][3] -
                    W[ii - 1, jL][3] - W[ii - 1, jR][3]
            ) / (4.0 * dx)

            Fv = viscous_flux_y_2d(law, wB, wT, dvx_dx, dvy_dx, dy)

            if iy >= 1
                dU[ii, jL] = dU[ii, jL] + Fv / dy
            end
            if iy < ny
                dU[ii, jR] = dU[ii, jR] - Fv / dy
            end
        end
    end

    return nothing
end
