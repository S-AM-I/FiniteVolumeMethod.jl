# ============================================================
# 2D Hyperbolic Solver on Structured Cartesian Meshes
# ============================================================
#
# Solution stored in a padded matrix U[i, j] of SVector{N}:
#   i = 1:2           -> left ghost columns
#   i = 3:nx+2        -> interior columns
#   i = nx+3:nx+4     -> right ghost columns
#   j = 1:2           -> bottom ghost rows
#   j = 3:ny+2        -> interior rows
#   j = ny+3:ny+4     -> top ghost rows
#
# Interior cell (ix, iy) (1-based) maps to U[ix+2, iy+2].

"""
    initialize_2d(prob::HyperbolicProblem2D) -> Matrix{SVector{N,FT}}

Create the padded 2D solution array from the initial condition.
Returns a matrix of size `(nx+4) × (ny+4)`.
"""
function initialize_2d(prob::HyperbolicProblem2D)
    law = prob.law
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    N = nvariables(law)

    # Determine element type from first cell
    x0, y0 = cell_center(mesh, 1)
    w0 = prob.initial_condition(x0, y0)
    u0 = primitive_to_conserved(law, w0)
    FT = eltype(u0)

    # Allocate padded array
    U = Matrix{SVector{N, FT}}(undef, nx + 4, ny + 4)

    # Fill with zeros first (ghost cells)
    zero_state = zero(SVector{N, FT})
    for j in 1:(ny + 4), i in 1:(nx + 4)
        U[i, j] = zero_state
    end

    # Fill interior cells
    for iy in 1:ny, ix in 1:nx
        x, y = cell_center(mesh, cell_idx(mesh, ix, iy))
        w = prob.initial_condition(x, y)
        U[ix + 2, iy + 2] = primitive_to_conserved(law, w)
    end

    return U
end

"""
    compute_dt_2d(prob::HyperbolicProblem2D, U::AbstractMatrix, t) -> FT

Compute the time step from the 2D CFL condition:
  `Δt = cfl / (max(|λx|)/Δx + max(|λy|)/Δy)`
"""
function compute_dt_2d(prob::HyperbolicProblem2D, U::AbstractMatrix, t)
    law = prob.law
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    cfl = prob.cfl
    dx, dy = mesh.dx, mesh.dy

    max_speed = zero(dx)
    for iy in 1:ny, ix in 1:nx
        w = conserved_to_primitive(law, U[ix + 2, iy + 2])
        λx = max_wave_speed(law, w, 1)
        λy = max_wave_speed(law, w, 2)
        speed = λx / dx + λy / dy
        max_speed = max(max_speed, speed)
    end

    dt = cfl / max_speed

    # Don't overshoot final time
    if t + dt > prob.final_time
        dt = prob.final_time - t
    end

    return dt
end

"""
    hyperbolic_rhs_2d!(dU, U, prob, t)

Compute the 2D RHS of the semi-discrete conservation law using
dimension-by-dimension flux differencing:

  `dU[i,j]/dt = -1/Δx * (Fx_{i+1/2,j} - Fx_{i-1/2,j})
                -1/Δy * (Fy_{i,j+1/2} - Fy_{i,j-1/2})`
"""
function hyperbolic_rhs_2d!(dU::AbstractMatrix, U::AbstractMatrix, prob::HyperbolicProblem2D, t)
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

    # X-direction sweeps: for each row j, compute x-fluxes
    for iy in 1:ny
        jj = iy + 2  # padded j index
        for ix in 0:nx
            # Face between cell ix and ix+1 in x-direction
            # Padded indices: left = ix+2, right = ix+3
            iL = ix + 2
            iR = ix + 3

            wL_face, wR_face = _reconstruct_face_2d(recon, law, U, iL, iR, jj, 1, nx)
            F = solve_riemann(solver, law, wL_face, wR_face, 1)

            # Subtract from right cell, add to left cell
            if ix >= 1
                dU[iL, jj] = dU[iL, jj] - F / dx
            end
            if ix < nx
                dU[iR, jj] = dU[iR, jj] + F / dx
            end
        end
    end

    # Y-direction sweeps: for each column i, compute y-fluxes
    for ix in 1:nx
        ii = ix + 2  # padded i index
        for iy in 0:ny
            # Face between cell iy and iy+1 in y-direction
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

# ============================================================
# 2D Reconstruction helpers
# ============================================================

"""
Reconstruct interface states for an x-direction face at column boundary (iL, iR) in row jj.
"""
@inline function _reconstruct_face_2d(
        recon::CellCenteredMUSCL, law, U::AbstractMatrix,
        iL::Int, iR::Int, jj::Int, dir::Int, nx::Int
    )
    uLL = U[iL - 1, jj]
    uL = U[iL, jj]
    uR = U[iR, jj]
    uRR = U[iR + 1, jj]

    wLL = conserved_to_primitive(law, uLL)
    wL = conserved_to_primitive(law, uL)
    wR = conserved_to_primitive(law, uR)
    wRR = conserved_to_primitive(law, uRR)

    return reconstruct_interface(recon, wLL, wL, wR, wRR)
end

@inline function _reconstruct_face_2d(
        ::NoReconstruction, law, U::AbstractMatrix,
        iL::Int, iR::Int, jj::Int, dir::Int, nx::Int
    )
    wL = conserved_to_primitive(law, U[iL, jj])
    wR = conserved_to_primitive(law, U[iR, jj])
    return wL, wR
end

"""
Reconstruct interface states for a y-direction face at row boundary (jL, jR) in column ii.
"""
@inline function _reconstruct_face_2d_y(
        recon::CellCenteredMUSCL, law, U::AbstractMatrix,
        ii::Int, jL::Int, jR::Int, ny::Int
    )
    uLL = U[ii, jL - 1]
    uL = U[ii, jL]
    uR = U[ii, jR]
    uRR = U[ii, jR + 1]

    wLL = conserved_to_primitive(law, uLL)
    wL = conserved_to_primitive(law, uL)
    wR = conserved_to_primitive(law, uR)
    wRR = conserved_to_primitive(law, uRR)

    return reconstruct_interface(recon, wLL, wL, wR, wRR)
end

@inline function _reconstruct_face_2d_y(
        ::NoReconstruction, law, U::AbstractMatrix,
        ii::Int, jL::Int, jR::Int, ny::Int
    )
    wL = conserved_to_primitive(law, U[ii, jL])
    wR = conserved_to_primitive(law, U[ii, jR])
    return wL, wR
end

# ============================================================
# Time Integration for 2D
# ============================================================

"""
    solve_hyperbolic(prob::HyperbolicProblem2D; method=:ssprk3, parallel=false)
        -> (coords, U_final, t_final)

Solve the 2D hyperbolic problem using explicit time integration.

# Keyword Arguments
- `method::Symbol`: Time integration method (`:euler` or `:ssprk3`).
- `parallel::Bool`: Use multi-threaded flux computation (default: `false`).
  When `true`, threads over rows/columns for flux sweeps and uses
  threaded CFL reduction. Requires `Threads.nthreads() > 1` for speedup.

# Returns
- `coords::Vector{Tuple{Float64,Float64}}`: Cell center coordinates (ix, iy order).
- `U_final::Matrix{SVector{N}}`: Final conserved variable matrix (nx × ny, interior only).
- `t_final::Real`: Final time reached.
"""
function solve_hyperbolic(prob::HyperbolicProblem2D; method::Symbol = :ssprk3, parallel::Bool = false)
    # Select serial or threaded functions
    _rhs! = parallel ? _hyperbolic_rhs_2d_threaded! : hyperbolic_rhs_2d!
    _compute_dt = parallel ? _compute_dt_2d_threaded : compute_dt_2d

    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    N = nvariables(prob.law)

    U = initialize_2d(prob)
    FT = eltype(U[3, 3])

    dU = similar(U)
    zero_state = zero(SVector{N, FT})
    for j in axes(dU, 2), i in axes(dU, 1)
        dU[i, j] = zero_state
    end

    t = prob.initial_time

    if method == :euler
        while t < prob.final_time - eps(typeof(t))
            dt = _compute_dt(prob, U, t)
            if dt <= zero(dt)
                break
            end
            _rhs!(dU, U, prob, t)
            @inbounds for iy in 1:ny, ix in 1:nx
                U[ix + 2, iy + 2] = U[ix + 2, iy + 2] + dt * dU[ix + 2, iy + 2]
            end
            t += dt
        end
    elseif method == :ssprk3
        U1 = similar(U)
        U2 = similar(U)
        for j in axes(U1, 2), i in axes(U1, 1)
            U1[i, j] = zero_state
            U2[i, j] = zero_state
        end

        while t < prob.final_time - eps(typeof(t))
            dt = _compute_dt(prob, U, t)
            if dt <= zero(dt)
                break
            end

            # Stage 1: U1 = U + dt * L(U)
            _rhs!(dU, U, prob, t)
            @inbounds for iy in 1:ny, ix in 1:nx
                ii, jj = ix + 2, iy + 2
                U1[ii, jj] = U[ii, jj] + dt * dU[ii, jj]
            end

            # Stage 2: U2 = 3/4 U + 1/4 (U1 + dt * L(U1))
            apply_boundary_conditions_2d!(U1, prob, t + dt)
            _rhs!(dU, U1, prob, t + dt)
            @inbounds for iy in 1:ny, ix in 1:nx
                ii, jj = ix + 2, iy + 2
                U2[ii, jj] = 0.75 * U[ii, jj] + 0.25 * (U1[ii, jj] + dt * dU[ii, jj])
            end

            # Stage 3: U = 1/3 U + 2/3 (U2 + dt * L(U2))
            apply_boundary_conditions_2d!(U2, prob, t + 0.5 * dt)
            _rhs!(dU, U2, prob, t + 0.5 * dt)
            @inbounds for iy in 1:ny, ix in 1:nx
                ii, jj = ix + 2, iy + 2
                U[ii, jj] = (1.0 / 3.0) * U[ii, jj] + (2.0 / 3.0) * (U2[ii, jj] + dt * dU[ii, jj])
            end

            t += dt
        end
    else
        error("Unknown time integration method: $method. Use :euler or :ssprk3.")
    end

    # Extract interior solution as nx × ny matrix
    U_interior = Matrix{SVector{N, FT}}(undef, nx, ny)
    for iy in 1:ny, ix in 1:nx
        U_interior[ix, iy] = U[ix + 2, iy + 2]
    end

    # Cell center coordinates
    coords = [(cell_center(mesh, cell_idx(mesh, ix, iy))) for ix in 1:nx, iy in 1:ny]

    return coords, U_interior, t
end

"""
    to_primitive(law, U::AbstractMatrix{<:SVector}) -> Matrix{SVector}

Convert a 2D matrix of conserved variable vectors to primitive variable vectors.
"""
function to_primitive(law, U::AbstractMatrix)
    return [conserved_to_primitive(law, U[i, j]) for i in axes(U, 1), j in axes(U, 2)]
end
