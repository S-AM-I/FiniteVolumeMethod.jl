# ============================================================
# IMEX Solve Loop
# ============================================================
#
# Split time integration for problems with stiff source terms:
#   dU/dt = F_explicit(U) + S_implicit(U)
#
# The explicit part uses the existing hyperbolic_rhs! machinery.
# The implicit stages solve a nonlinear system at each cell:
#   U_i^{(k)} - a_{kk} * dt * S(U_i^{(k)}) = RHS_known
#
# For source terms with a known Jacobian, we use a single Newton
# step (linearised implicit solve):
#   (I - a_{kk} * dt * J) * ΔU = RHS - U_current + a_{kk} * dt * S(U_current)

# ============================================================
# 1D IMEX Solver
# ============================================================

"""
    solve_hyperbolic_imex(prob::HyperbolicProblem, stiff_source::AbstractStiffSource;
                          scheme=IMEX_SSP3_433(), newton_tol=1e-10, newton_maxiter=5)

Solve a 1D hyperbolic problem with stiff source terms using IMEX
Runge-Kutta time integration.

# Arguments
- `prob::HyperbolicProblem`: The hyperbolic problem (provides mesh, law, BCs, etc.).
- `stiff_source::AbstractStiffSource`: The stiff source term treated implicitly.

# Keyword Arguments
- `scheme::AbstractIMEXScheme`: The IMEX RK scheme (default: `IMEX_SSP3_433()`).
- `newton_tol::Real`: Tolerance for implicit Newton solves (default: `1e-10`).
- `newton_maxiter::Int`: Maximum Newton iterations per implicit stage (default: `5`).

# Returns
- `x::Vector`: Cell center coordinates.
- `U_final::Vector{SVector{N}}`: Final conserved variable vectors at cell centers.
- `t_final::Real`: Final time reached.
"""
function solve_hyperbolic_imex(
        prob::HyperbolicProblem, stiff_source::AbstractStiffSource;
        scheme::AbstractIMEXScheme = IMEX_SSP3_433(),
        newton_tol = 1.0e-10,
        newton_maxiter::Int = 5,
        callback::Union{Nothing, Function} = nothing,
    )
    mesh = prob.mesh
    nc = ncells(mesh)
    law = prob.law
    N = nvariables(law)

    U = initialize_1d(prob)
    FT = eltype(U[3])

    tab = imex_tableau(scheme)
    s = tab.s

    # Allocate stage arrays
    # K_ex[k] and K_im[k] are arrays of SVector for each stage
    K_ex = [similar(U) for _ in 1:s]
    K_im = [similar(U) for _ in 1:s]
    U_stage = similar(U)

    zero_state = zero(SVector{N, FT})
    for k in 1:s
        for i in eachindex(U)
            K_ex[k][i] = zero_state
            K_im[k][i] = zero_state
        end
    end
    for i in eachindex(U_stage)
        U_stage[i] = zero_state
    end

    dU_tmp = similar(U)
    for i in eachindex(dU_tmp)
        dU_tmp[i] = zero_state
    end

    t = prob.initial_time
    step = 0

    while t < prob.final_time - eps(typeof(t))
        dt = compute_dt(prob, U, t)
        if dt <= zero(dt)
            break
        end

        # IMEX RK stages
        for k in 1:s
            # Build the argument for stage k:
            #   U_stage = U^n + dt * sum_{j<k} a_ex[k,j] * K_ex[j]
            #                 + dt * sum_{j<k} a_im[k,j] * K_im[j]
            for i in 3:(nc + 2)
                U_stage[i] = U[i]
                for j in 1:(k - 1)
                    U_stage[i] = U_stage[i] + dt * tab.A_ex[k][j] * K_ex[j][i]
                    U_stage[i] = U_stage[i] + dt * tab.A_im[k][j] * K_im[j][i]
                end
            end

            a_kk = tab.A_im[k][k]

            if a_kk == 0.0
                # Explicit stage: no implicit solve needed
                # K_im[k] = S(U_stage)
                _eval_stiff_source_1d!(K_im[k], U_stage, law, stiff_source, nc)

                # K_ex[k] = F(U_stage) (hyperbolic RHS)
                hyperbolic_rhs!(K_ex[k], U_stage, prob, t + tab.c_ex[k] * dt)
            else
                # Implicit stage: solve (I - a_kk * dt * J) * delta = RHS
                # First compute the explicit part: K_ex[k] = F(U_stage)
                hyperbolic_rhs!(K_ex[k], U_stage, prob, t + tab.c_ex[k] * dt)

                # Now solve for U_stage including the implicit part
                # U_stage already contains the known explicit contributions.
                # We need to find U* such that:
                #   U* = U_stage + a_kk * dt * S(U*)
                # Newton iteration: given U_guess, solve
                #   (I - a_kk*dt*J) * (U_new - U_guess) = U_stage + a_kk*dt*S(U_guess) - U_guess
                _implicit_solve_1d!(
                    U_stage, law, stiff_source, a_kk * dt, nc, N,
                    newton_tol, newton_maxiter
                )

                # After the solve, U_stage contains the updated stage value.
                # K_im[k] = S(U_stage)
                _eval_stiff_source_1d!(K_im[k], U_stage, law, stiff_source, nc)
            end
        end

        # Final update: U^{n+1} = U^n + dt * sum_k (b_ex[k]*K_ex[k] + b_im[k]*K_im[k])
        for i in 3:(nc + 2)
            U_new = U[i]
            for k in 1:s
                U_new = U_new + dt * tab.b_ex[k] * K_ex[k][i]
                U_new = U_new + dt * tab.b_im[k] * K_im[k][i]
            end
            U[i] = U_new
        end

        t += dt
        step += 1
        if callback !== nothing
            callback(U, t, step, dt)
        end
    end

    # Extract interior solution
    x = [cell_center(mesh, i) for i in 1:nc]
    U_interior = U[3:(nc + 2)]

    return x, U_interior, t
end

# ============================================================
# Helper: evaluate stiff source at all interior cells (1D)
# ============================================================

function _eval_stiff_source_1d!(S_out, U, law, stiff_source, nc)
    for i in 1:nc
        u = U[i + 2]
        w = conserved_to_primitive(law, u)
        S_out[i + 2] = evaluate_stiff_source(stiff_source, law, w, u)
    end
    return nothing
end

# ============================================================
# Helper: implicit solve at all interior cells (1D)
# ============================================================

"""
    _implicit_solve_1d!(U_stage, law, stiff_source, adt, nc, N, tol, maxiter)

Solve `U* = U_stage + adt * S(U*)` cell-by-cell using Newton iteration.

On entry, `U_stage[i+2]` contains the known RHS for cell i.
On exit, `U_stage[i+2]` contains the solution U*.
"""
function _implicit_solve_1d!(U_stage, law, stiff_source, adt, nc, N, tol, maxiter)
    for i in 1:nc
        idx = i + 2
        U_rhs = U_stage[idx]  # known right-hand side
        U_guess = U_rhs       # initial guess

        for iter in 1:maxiter
            w = conserved_to_primitive(law, U_guess)
            S_val = evaluate_stiff_source(stiff_source, law, w, U_guess)

            # Residual: R = U_guess - U_rhs - adt * S(U_guess) = 0
            residual = U_guess - U_rhs - adt * S_val

            # Check convergence
            res_norm = _svector_maxabs(residual)
            if res_norm < tol
                break
            end

            # Jacobian: J_newton = I - adt * ∂S/∂U
            J_source = stiff_source_jacobian(stiff_source, law, w, U_guess)
            J_newton = _identity_smatrix(Val(N), eltype(U_guess)) - adt * J_source

            # Newton step: delta = J_newton \ (-residual)
            delta = J_newton \ (-residual)
            U_guess = U_guess + delta
        end

        U_stage[idx] = U_guess
    end
    return nothing
end

# ============================================================
# 2D IMEX Solver
# ============================================================

"""
    solve_hyperbolic_imex(prob::HyperbolicProblem2D, stiff_source::AbstractStiffSource;
                          scheme=IMEX_SSP3_433(), newton_tol=1e-10, newton_maxiter=5,
                          parallel=false)

Solve a 2D hyperbolic problem with stiff source terms using IMEX
Runge-Kutta time integration.

# Keyword Arguments
- `parallel::Bool`: Use multi-threaded flux and implicit solve (default: `false`).

# Returns
- `coords`: Cell center coordinates (nx x ny matrix of tuples).
- `U_final::Matrix{SVector{N}}`: Final conserved variable matrix (nx x ny).
- `t_final::Real`: Final time reached.
"""
function solve_hyperbolic_imex(
        prob::HyperbolicProblem2D, stiff_source::AbstractStiffSource;
        scheme::AbstractIMEXScheme = IMEX_SSP3_433(),
        newton_tol = 1.0e-10,
        newton_maxiter::Int = 5,
        parallel::Bool = false,
        callback::Union{Nothing, Function} = nothing,
    )
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    law = prob.law
    N = nvariables(law)

    U = initialize_2d(prob)
    FT = eltype(U[3, 3])

    tab = imex_tableau(scheme)
    s = tab.s

    zero_state = zero(SVector{N, FT})

    # Allocate stage arrays
    K_ex = [similar(U) for _ in 1:s]
    K_im = [similar(U) for _ in 1:s]
    U_stage = similar(U)

    for k in 1:s
        for j in axes(U, 2), i in axes(U, 1)
            K_ex[k][i, j] = zero_state
            K_im[k][i, j] = zero_state
        end
    end
    for j in axes(U_stage, 2), i in axes(U_stage, 1)
        U_stage[i, j] = zero_state
    end

    dU_tmp = similar(U)
    for j in axes(dU_tmp, 2), i in axes(dU_tmp, 1)
        dU_tmp[i, j] = zero_state
    end

    # Select serial or threaded functions
    _rhs! = parallel ? _hyperbolic_rhs_2d_threaded! : hyperbolic_rhs_2d!
    _compute_dt = parallel ? _compute_dt_2d_threaded : compute_dt_2d
    _implicit_solve! = parallel ? _implicit_solve_2d_threaded! : _implicit_solve_2d!

    t = prob.initial_time
    step = 0

    while t < prob.final_time - eps(typeof(t))
        dt = _compute_dt(prob, U, t)
        if dt <= zero(dt)
            break
        end

        for k in 1:s
            # Build stage argument
            for iy in 1:ny, ix in 1:nx
                ii, jj = ix + 2, iy + 2
                U_stage[ii, jj] = U[ii, jj]
                for j in 1:(k - 1)
                    U_stage[ii, jj] = U_stage[ii, jj] + dt * tab.A_ex[k][j] * K_ex[j][ii, jj]
                    U_stage[ii, jj] = U_stage[ii, jj] + dt * tab.A_im[k][j] * K_im[j][ii, jj]
                end
            end

            a_kk = tab.A_im[k][k]

            if a_kk == 0.0
                _eval_stiff_source_2d!(K_im[k], U_stage, law, stiff_source, nx, ny)
                _rhs!(K_ex[k], U_stage, prob, t + tab.c_ex[k] * dt)
            else
                _rhs!(K_ex[k], U_stage, prob, t + tab.c_ex[k] * dt)

                _implicit_solve!(
                    U_stage, law, stiff_source, a_kk * dt, nx, ny, N,
                    newton_tol, newton_maxiter
                )

                _eval_stiff_source_2d!(K_im[k], U_stage, law, stiff_source, nx, ny)
            end
        end

        # Final update
        for iy in 1:ny, ix in 1:nx
            ii, jj = ix + 2, iy + 2
            U_new = U[ii, jj]
            for k in 1:s
                U_new = U_new + dt * tab.b_ex[k] * K_ex[k][ii, jj]
                U_new = U_new + dt * tab.b_im[k] * K_im[k][ii, jj]
            end
            U[ii, jj] = U_new
        end

        t += dt
        step += 1
        if callback !== nothing
            callback(U, t, step, dt)
        end
    end

    # Extract interior solution
    U_interior = Matrix{SVector{N, FT}}(undef, nx, ny)
    for iy in 1:ny, ix in 1:nx
        U_interior[ix, iy] = U[ix + 2, iy + 2]
    end

    coords = [(cell_center(mesh, cell_idx(mesh, ix, iy))) for ix in 1:nx, iy in 1:ny]

    return coords, U_interior, t
end

# ============================================================
# Helper: evaluate stiff source at all interior cells (2D)
# ============================================================

function _eval_stiff_source_2d!(S_out, U, law, stiff_source, nx, ny)
    for iy in 1:ny, ix in 1:nx
        ii, jj = ix + 2, iy + 2
        u = U[ii, jj]
        w = conserved_to_primitive(law, u)
        S_out[ii, jj] = evaluate_stiff_source(stiff_source, law, w, u)
    end
    return nothing
end

# ============================================================
# Helper: implicit solve at all interior cells (2D)
# ============================================================

function _implicit_solve_2d!(U_stage, law, stiff_source, adt, nx, ny, N, tol, maxiter)
    for iy in 1:ny, ix in 1:nx
        ii, jj = ix + 2, iy + 2
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
    return nothing
end

# ============================================================
# Utility functions
# ============================================================

"""
    _svector_maxabs(v::SVector{N}) -> scalar

Compute the maximum absolute value of components of an SVector.
"""
@inline function _svector_maxabs(v::SVector{N}) where {N}
    m = abs(v[1])
    for i in 2:N
        m = max(m, abs(v[i]))
    end
    return m
end

"""
    _identity_smatrix(::Val{N}, ::Type{FT}) -> SMatrix{N,N,FT}

Return an N x N identity SMatrix of the given element type.
"""
@inline function _identity_smatrix(::Val{N}, ::Type{FT}) where {N, FT}
    return one(StaticArrays.SMatrix{N, N, FT})
end
