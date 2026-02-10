# ============================================================
# Threaded Versions of 2D Hyperbolic Solver Functions
# ============================================================
#
# Multi-threaded implementations for large 2D problems.
# Thread safety:
#   - X-sweep: each row is independent → thread over rows
#   - Y-sweep: each column is independent → thread over columns
#   - CFL: per-row max with final reduction
#   - Implicit solve: each cell is independent → thread over cells

"""
    _hyperbolic_rhs_2d_threaded!(dU, U, prob, t)

Multi-threaded 2D RHS computation. Threads over rows in x-sweeps
and over columns in y-sweeps. Each row/column is independent,
so no race conditions occur.
"""
function _hyperbolic_rhs_2d_threaded!(dU::AbstractMatrix, U::AbstractMatrix, prob::HyperbolicProblem2D, t)
    law = prob.law
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy
    solver = prob.riemann_solver
    recon = prob.reconstruction

    # Apply BCs (serial — fast, touches only boundary cells)
    apply_boundary_conditions_2d!(U, prob, t)

    N = nvariables(law)
    FT = eltype(U[3, 3])
    zero_state = zero(SVector{N, FT})

    # Zero dU for interior cells
    for iy in 1:ny, ix in 1:nx
        @inbounds dU[ix + 2, iy + 2] = zero_state
    end

    # X-direction sweeps: thread over rows (each row is independent)
    Threads.@threads for iy in 1:ny
        jj = iy + 2
        @inbounds for ix in 0:nx
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

    # Y-direction sweeps: thread over columns (each column is independent)
    Threads.@threads for ix in 1:nx
        ii = ix + 2
        @inbounds for iy in 0:ny
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

    return nothing
end

"""
    _compute_dt_2d_threaded(prob, U, t)

Multi-threaded CFL time step computation using per-row max reduction.
Each row computes its own local maximum, avoiding threadid() issues
with Julia's dynamic task scheduler.
"""
function _compute_dt_2d_threaded(prob::HyperbolicProblem2D, U::AbstractMatrix, t)
    law = prob.law
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    cfl = prob.cfl
    dx, dy = mesh.dx, mesh.dy

    row_max = zeros(eltype(dx), ny)

    Threads.@threads for iy in 1:ny
        local_max = zero(dx)
        @inbounds for ix in 1:nx
            w = conserved_to_primitive(law, U[ix + 2, iy + 2])
            λx = max_wave_speed(law, w, 1)
            λy = max_wave_speed(law, w, 2)
            speed = λx / dx + λy / dy
            local_max = max(local_max, speed)
        end
        row_max[iy] = local_max
    end

    max_speed = maximum(row_max)
    dt = cfl / max_speed

    if t + dt > prob.final_time
        dt = prob.final_time - t
    end
    return dt
end

"""
    _implicit_solve_2d_threaded!(U_stage, law, stiff_source, adt, nx, ny, N, tol, maxiter)

Multi-threaded cell-by-cell implicit solve. Each cell is completely
independent, making this embarrassingly parallel.
"""
function _implicit_solve_2d_threaded!(U_stage, law, stiff_source, adt, nx, ny, N, tol, maxiter)
    ncells_total = nx * ny
    Threads.@threads for idx in 1:ncells_total
        ix = mod1(idx, nx)
        iy = div(idx - 1, nx) + 1
        ii, jj = ix + 2, iy + 2

        @inbounds begin
            U_rhs = U_stage[ii, jj]
            U_guess = U_rhs

            for iter in 1:maxiter
                w = conserved_to_primitive(law, U_guess)
                S_val = evaluate_stiff_source(stiff_source, law, w, U_guess)

                residual = U_guess - U_rhs - adt * S_val

                res_norm = _svector_maxabs(residual)
                if res_norm < tol
                    break
                end

                J_source = stiff_source_jacobian(stiff_source, law, w, U_guess)
                J_newton = _identity_smatrix(Val(N), eltype(U_guess)) - adt * J_source

                delta = J_newton \ (-residual)
                U_guess = U_guess + delta
            end

            U_stage[ii, jj] = U_guess
        end
    end
    return nothing
end
