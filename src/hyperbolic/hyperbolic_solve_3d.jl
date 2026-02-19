# ============================================================
# 3D Hyperbolic Solver on Structured Cartesian Meshes
# ============================================================
#
# Solution stored in a padded 3D array U[i, j, k] of SVector{N}:
#   i = 1:2           -> left ghost planes   (x-)
#   i = 3:nx+2        -> interior            (x)
#   i = nx+3:nx+4     -> right ghost planes  (x+)
#   j = 1:2           -> bottom ghost planes (y-)
#   j = 3:ny+2        -> interior            (y)
#   j = ny+3:ny+4     -> top ghost planes    (y+)
#   k = 1:2           -> front ghost planes  (z-)
#   k = 3:nz+2        -> interior            (z)
#   k = nz+3:nz+4     -> back ghost planes   (z+)
#
# Interior cell (ix, iy, iz) (1-based) maps to U[ix+2, iy+2, iz+2].

"""
    initialize_3d(prob::HyperbolicProblem3D) -> Array{SVector{N,FT}, 3}

Create the padded 3D solution array from the initial condition.
Returns an array of size `(nx+4) x (ny+4) x (nz+4)`.
"""
function initialize_3d(prob::HyperbolicProblem3D)
    law = prob.law
    mesh = prob.mesh
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    N = nvariables(law)

    # Determine element type from first cell
    x0, y0, z0 = cell_center(mesh, 1)
    w0 = prob.initial_condition(x0, y0, z0)
    u0 = primitive_to_conserved(law, w0)
    FT = eltype(u0)

    # Allocate padded array
    U = Array{SVector{N, FT}, 3}(undef, nx + 4, ny + 4, nz + 4)

    # Fill with zeros first (ghost cells)
    zero_state = zero(SVector{N, FT})
    for k in 1:(nz + 4), j in 1:(ny + 4), i in 1:(nx + 4)
        U[i, j, k] = zero_state
    end

    # Fill interior cells
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        x, y, z = cell_center(mesh, cell_idx_3d(mesh, ix, iy, iz))
        w = prob.initial_condition(x, y, z)
        U[ix + 2, iy + 2, iz + 2] = primitive_to_conserved(law, w)
    end

    return U
end

"""
    compute_dt_3d(prob::HyperbolicProblem3D, U::AbstractArray{T,3}, t) -> FT

Compute the time step from the 3D CFL condition:
  `dt = cfl / (max(|lx|)/dx + max(|ly|)/dy + max(|lz|)/dz)`
"""
function compute_dt_3d(prob::HyperbolicProblem3D, U::AbstractArray{T, 3}, t) where {T}
    law = prob.law
    mesh = prob.mesh
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    cfl = prob.cfl
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz

    max_speed = zero(dx)
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        w = conserved_to_primitive(law, U[ix + 2, iy + 2, iz + 2])
        lx = max_wave_speed(law, w, 1)
        ly = max_wave_speed(law, w, 2)
        lz = max_wave_speed(law, w, 3)
        speed = lx / dx + ly / dy + lz / dz
        max_speed = max(max_speed, speed)
    end

    dt = cfl / max_speed

    # Don't overshoot final time
    if t + dt > prob.final_time
        dt = prob.final_time - t
    end

    return dt
end

# ============================================================
# 3D Reconstruction helpers
# ============================================================

"""
Reconstruct interface states for an x-direction face.
"""
@inline function _reconstruct_face_3d_x(
        recon::CellCenteredMUSCL, law, U::AbstractArray{T, 3},
        iL::Int, iR::Int, jj::Int, kk::Int
    ) where {T}
    uLL = U[iL - 1, jj, kk]
    uL = U[iL, jj, kk]
    uR = U[iR, jj, kk]
    uRR = U[iR + 1, jj, kk]

    wLL = conserved_to_primitive(law, uLL)
    wL = conserved_to_primitive(law, uL)
    wR = conserved_to_primitive(law, uR)
    wRR = conserved_to_primitive(law, uRR)

    return reconstruct_interface(recon, wLL, wL, wR, wRR)
end

@inline function _reconstruct_face_3d_x(
        ::NoReconstruction, law, U::AbstractArray{T, 3},
        iL::Int, iR::Int, jj::Int, kk::Int
    ) where {T}
    wL = conserved_to_primitive(law, U[iL, jj, kk])
    wR = conserved_to_primitive(law, U[iR, jj, kk])
    return wL, wR
end

"""
Reconstruct interface states for a y-direction face.
"""
@inline function _reconstruct_face_3d_y(
        recon::CellCenteredMUSCL, law, U::AbstractArray{T, 3},
        ii::Int, jL::Int, jR::Int, kk::Int
    ) where {T}
    uLL = U[ii, jL - 1, kk]
    uL = U[ii, jL, kk]
    uR = U[ii, jR, kk]
    uRR = U[ii, jR + 1, kk]

    wLL = conserved_to_primitive(law, uLL)
    wL = conserved_to_primitive(law, uL)
    wR = conserved_to_primitive(law, uR)
    wRR = conserved_to_primitive(law, uRR)

    return reconstruct_interface(recon, wLL, wL, wR, wRR)
end

@inline function _reconstruct_face_3d_y(
        ::NoReconstruction, law, U::AbstractArray{T, 3},
        ii::Int, jL::Int, jR::Int, kk::Int
    ) where {T}
    wL = conserved_to_primitive(law, U[ii, jL, kk])
    wR = conserved_to_primitive(law, U[ii, jR, kk])
    return wL, wR
end

"""
Reconstruct interface states for a z-direction face.
"""
@inline function _reconstruct_face_3d_z(
        recon::CellCenteredMUSCL, law, U::AbstractArray{T, 3},
        ii::Int, jj::Int, kL::Int, kR::Int
    ) where {T}
    uLL = U[ii, jj, kL - 1]
    uL = U[ii, jj, kL]
    uR = U[ii, jj, kR]
    uRR = U[ii, jj, kR + 1]

    wLL = conserved_to_primitive(law, uLL)
    wL = conserved_to_primitive(law, uL)
    wR = conserved_to_primitive(law, uR)
    wRR = conserved_to_primitive(law, uRR)

    return reconstruct_interface(recon, wLL, wL, wR, wRR)
end

@inline function _reconstruct_face_3d_z(
        ::NoReconstruction, law, U::AbstractArray{T, 3},
        ii::Int, jj::Int, kL::Int, kR::Int
    ) where {T}
    wL = conserved_to_primitive(law, U[ii, jj, kL])
    wR = conserved_to_primitive(law, U[ii, jj, kR])
    return wL, wR
end

"""
    hyperbolic_rhs_3d!(dU, U, prob, t)

Compute the 3D RHS of the semi-discrete conservation law using
dimension-by-dimension flux differencing:

  `dU[i,j,k]/dt = -1/dx * (Fx_{i+1/2,j,k} - Fx_{i-1/2,j,k})
                  -1/dy * (Fy_{i,j+1/2,k} - Fy_{i,j-1/2,k})
                  -1/dz * (Fz_{i,j,k+1/2} - Fz_{i,j,k-1/2})`
"""
function hyperbolic_rhs_3d!(
        dU::AbstractArray{T, 3}, U::AbstractArray{T, 3},
        prob::HyperbolicProblem3D, t
    ) where {T}
    law = prob.law
    mesh = prob.mesh
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    solver = prob.riemann_solver
    recon = prob.reconstruction

    # Apply BCs to fill ghost cells
    apply_boundary_conditions_3d!(U, prob, t)

    N = nvariables(law)
    FT = eltype(U[3, 3, 3])

    # Zero out dU for interior cells
    zero_state = zero(SVector{N, FT})
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        dU[ix + 2, iy + 2, iz + 2] = zero_state
    end

    # X-direction sweeps: for each (j, k) row, compute x-fluxes
    for iz in 1:nz, iy in 1:ny
        jj = iy + 2
        kk = iz + 2
        for ix in 0:nx
            iL = ix + 2
            iR = ix + 3

            wL_face, wR_face = _reconstruct_face_3d_x(recon, law, U, iL, iR, jj, kk)
            F = solve_riemann(solver, law, wL_face, wR_face, 1)

            if ix >= 1
                dU[iL, jj, kk] = dU[iL, jj, kk] - F / dx
            end
            if ix < nx
                dU[iR, jj, kk] = dU[iR, jj, kk] + F / dx
            end
        end
    end

    # Y-direction sweeps: for each (i, k) column, compute y-fluxes
    for iz in 1:nz, ix in 1:nx
        ii = ix + 2
        kk = iz + 2
        for iy in 0:ny
            jL = iy + 2
            jR = iy + 3

            wL_face, wR_face = _reconstruct_face_3d_y(recon, law, U, ii, jL, jR, kk)
            F = solve_riemann(solver, law, wL_face, wR_face, 2)

            if iy >= 1
                dU[ii, jL, kk] = dU[ii, jL, kk] - F / dy
            end
            if iy < ny
                dU[ii, jR, kk] = dU[ii, jR, kk] + F / dy
            end
        end
    end

    # Z-direction sweeps: for each (i, j) pencil, compute z-fluxes
    for iy in 1:ny, ix in 1:nx
        ii = ix + 2
        jj = iy + 2
        for iz in 0:nz
            kL = iz + 2
            kR = iz + 3

            wL_face, wR_face = _reconstruct_face_3d_z(recon, law, U, ii, jj, kL, kR)
            F = solve_riemann(solver, law, wL_face, wR_face, 3)

            if iz >= 1
                dU[ii, jj, kL] = dU[ii, jj, kL] - F / dz
            end
            if iz < nz
                dU[ii, jj, kR] = dU[ii, jj, kR] + F / dz
            end
        end
    end

    return nothing
end

# ============================================================
# Time Integration for 3D
# ============================================================

"""
    solve_hyperbolic(prob::HyperbolicProblem3D; method=:ssprk3)
        -> (coords, U_final, t_final)

Solve the 3D hyperbolic problem using explicit time integration.

# Returns
- `coords::Array{Tuple{Float64,Float64,Float64},3}`: Cell center coordinates.
- `U_final::Array{SVector{N},3}`: Final conserved variable array (nx x ny x nz, interior only).
- `t_final::Real`: Final time reached.
"""
function solve_hyperbolic(
        prob::HyperbolicProblem3D;
        method::Symbol = :ssprk3,
        callback::Union{Nothing, Function} = nothing,
    )
    mesh = prob.mesh
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    N = nvariables(prob.law)

    U = initialize_3d(prob)
    FT = eltype(U[3, 3, 3])

    dU = similar(U)
    zero_state = zero(SVector{N, FT})
    for k in axes(dU, 3), j in axes(dU, 2), i in axes(dU, 1)
        dU[i, j, k] = zero_state
    end

    t = prob.initial_time
    step = 0

    if method == :euler
        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt_3d(prob, U, t)
            if dt <= zero(dt)
                break
            end
            hyperbolic_rhs_3d!(dU, U, prob, t)
            for iz in 1:nz, iy in 1:ny, ix in 1:nx
                U[ix + 2, iy + 2, iz + 2] = U[ix + 2, iy + 2, iz + 2] + dt * dU[ix + 2, iy + 2, iz + 2]
            end
            t += dt
            step += 1
            if callback !== nothing
                callback(U, t, step, dt)
            end
        end
    elseif method == :ssprk3
        U1 = similar(U)
        U2 = similar(U)
        for k in axes(U1, 3), j in axes(U1, 2), i in axes(U1, 1)
            U1[i, j, k] = zero_state
            U2[i, j, k] = zero_state
        end

        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt_3d(prob, U, t)
            if dt <= zero(dt)
                break
            end

            # Stage 1: U1 = U + dt * L(U)
            hyperbolic_rhs_3d!(dU, U, prob, t)
            for iz in 1:nz, iy in 1:ny, ix in 1:nx
                ii, jj, kk = ix + 2, iy + 2, iz + 2
                U1[ii, jj, kk] = U[ii, jj, kk] + dt * dU[ii, jj, kk]
            end

            # Stage 2: U2 = 3/4 U + 1/4 (U1 + dt * L(U1))
            apply_boundary_conditions_3d!(U1, prob, t + dt)
            hyperbolic_rhs_3d!(dU, U1, prob, t + dt)
            for iz in 1:nz, iy in 1:ny, ix in 1:nx
                ii, jj, kk = ix + 2, iy + 2, iz + 2
                U2[ii, jj, kk] = 0.75 * U[ii, jj, kk] + 0.25 * (U1[ii, jj, kk] + dt * dU[ii, jj, kk])
            end

            # Stage 3: U = 1/3 U + 2/3 (U2 + dt * L(U2))
            apply_boundary_conditions_3d!(U2, prob, t + 0.5 * dt)
            hyperbolic_rhs_3d!(dU, U2, prob, t + 0.5 * dt)
            for iz in 1:nz, iy in 1:ny, ix in 1:nx
                ii, jj, kk = ix + 2, iy + 2, iz + 2
                U[ii, jj, kk] = (1.0 / 3.0) * U[ii, jj, kk] + (2.0 / 3.0) * (U2[ii, jj, kk] + dt * dU[ii, jj, kk])
            end

            t += dt
            step += 1
            if callback !== nothing
                callback(U, t, step, dt)
            end
        end
    else
        error("Unknown time integration method: $method. Use :euler or :ssprk3.")
    end

    # Extract interior solution as nx x ny x nz array
    U_interior = Array{SVector{N, FT}, 3}(undef, nx, ny, nz)
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        U_interior[ix, iy, iz] = U[ix + 2, iy + 2, iz + 2]
    end

    # Cell center coordinates
    coords = [(cell_center(mesh, cell_idx_3d(mesh, ix, iy, iz))) for ix in 1:nx, iy in 1:ny, iz in 1:nz]

    return coords, U_interior, t
end

"""
    to_primitive(law, U::AbstractArray{<:SVector, 3}) -> Array{SVector}

Convert a 3D array of conserved variable vectors to primitive variable vectors.
"""
function to_primitive(law, U::AbstractArray{<:SVector, 3})
    return [conserved_to_primitive(law, U[i, j, k]) for i in axes(U, 1), j in axes(U, 2), k in axes(U, 3)]
end
