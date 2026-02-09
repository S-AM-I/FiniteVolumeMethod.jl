# ============================================================
# 3D MHD Solver with Constrained Transport
# ============================================================
#
# Specializes solve_hyperbolic for HyperbolicProblem3D with
# IdealMHDEquations{3}. Uses constrained transport (CT) to
# maintain div(B) = 0 to machine precision.
#
# Approach:
#   1. All 8 conserved variables updated via flux differencing (3 sweeps)
#   2. Face-centered B updated via CT (edge EMF from face fluxes)
#   3. Cell-centered Bx, By, Bz overwritten from face-centered values
#
# Face flux storage uses extended arrays that include ghost slabs
# so that the edge EMF can be computed uniformly at all edges.

# ============================================================
# CT Periodic Enforcement for 3D
# ============================================================

"""
    apply_ct_periodic_3d!(ct::CTData3D, prob::HyperbolicProblem3D, nx, ny, nz)

Enforce face-centered B periodicity for periodic boundary conditions in 3D.
"""
function apply_ct_periodic_3d!(ct::CTData3D, prob, nx, ny, nz)
    # Periodic in x: Bx_face at face 1 = face nx+1
    if prob.bc_left isa PeriodicHyperbolicBC && prob.bc_right isa PeriodicHyperbolicBC
        for k in 1:nz, j in 1:ny
            avg = 0.5 * (ct.Bx_face[1, j, k] + ct.Bx_face[nx + 1, j, k])
            ct.Bx_face[1, j, k] = avg
            ct.Bx_face[nx + 1, j, k] = avg
        end
    end

    # Periodic in y: By_face at face 1 = face ny+1
    if prob.bc_bottom isa PeriodicHyperbolicBC && prob.bc_top isa PeriodicHyperbolicBC
        for k in 1:nz, i in 1:nx
            avg = 0.5 * (ct.By_face[i, 1, k] + ct.By_face[i, ny + 1, k])
            ct.By_face[i, 1, k] = avg
            ct.By_face[i, ny + 1, k] = avg
        end
    end

    # Periodic in z: Bz_face at face 1 = face nz+1
    if prob.bc_front isa PeriodicHyperbolicBC && prob.bc_back isa PeriodicHyperbolicBC
        for j in 1:ny, i in 1:nx
            avg = 0.5 * (ct.Bz_face[i, j, 1] + ct.Bz_face[i, j, nz + 1])
            ct.Bz_face[i, j, 1] = avg
            ct.Bz_face[i, j, nz + 1] = avg
        end
    end

    return nothing
end

# ============================================================
# Compute fluxes and dU for 3D MHD
# ============================================================

"""
    _mhd_compute_fluxes_3d!(Fx_all, Fy_all, Fz_all, dU, U, prob, t)

Compute face fluxes (including ghost slabs for EMF edges)
and accumulate the cell-centered dU via flux differencing.

Extended flux arrays:
- `Fx_all[face_i, col_j, col_k]`: size `(nx+1) x (ny+2) x (nz+2)`
- `Fy_all[col_i, face_j, col_k]`: size `(nx+2) x (ny+1) x (nz+2)`
- `Fz_all[col_i, col_j, face_k]`: size `(nx+2) x (ny+2) x (nz+1)`
"""
function _mhd_compute_fluxes_3d!(
        Fx_all::AbstractArray{T, 3}, Fy_all::AbstractArray{T, 3},
        Fz_all::AbstractArray{T, 3},
        dU::AbstractArray{T, 3}, U::AbstractArray{T, 3},
        prob::HyperbolicProblem3D, t
    ) where {T}
    law = prob.law
    mesh = prob.mesh
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    solver = prob.riemann_solver
    recon = prob.reconstruction
    N = nvariables(law)
    FT = eltype(U[3, 3, 3])

    # Apply BCs to fill ghost cells
    apply_boundary_conditions_3d!(U, prob, t)

    # Zero dU for interior cells
    zero_state = zero(SVector{N, FT})
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        dU[ix + 2, iy + 2, iz + 2] = zero_state
    end

    # ---- X-direction sweeps (including ghost slabs in j and k) ----
    # col_j = 1:ny+2 maps to padded jj = 2:ny+3
    # col_k = 1:nz+2 maps to padded kk = 2:nz+3
    for col_k in 1:(nz + 2), col_j in 1:(ny + 2)
        jj = col_j + 1
        kk = col_k + 1
        for face_i in 1:(nx + 1)
            iL = face_i + 1
            iR = face_i + 2
            wL_face, wR_face = _reconstruct_face_3d_x(recon, law, U, iL, iR, jj, kk)
            Fx_all[face_i, col_j, col_k] = solve_riemann(solver, law, wL_face, wR_face, 1)
        end
    end

    # ---- Y-direction sweeps (including ghost slabs in i and k) ----
    for col_k in 1:(nz + 2), col_i in 1:(nx + 2)
        ii = col_i + 1
        kk = col_k + 1
        for face_j in 1:(ny + 1)
            jL = face_j + 1
            jR = face_j + 2
            wL_face, wR_face = _reconstruct_face_3d_y(recon, law, U, ii, jL, jR, kk)
            Fy_all[col_i, face_j, col_k] = solve_riemann(solver, law, wL_face, wR_face, 2)
        end
    end

    # ---- Z-direction sweeps (including ghost slabs in i and j) ----
    for col_j in 1:(ny + 2), col_i in 1:(nx + 2)
        ii = col_i + 1
        jj = col_j + 1
        for face_k in 1:(nz + 1)
            kL = face_k + 1
            kR = face_k + 2
            wL_face, wR_face = _reconstruct_face_3d_z(recon, law, U, ii, jj, kL, kR)
            Fz_all[col_i, col_j, face_k] = solve_riemann(solver, law, wL_face, wR_face, 3)
        end
    end

    # ---- Accumulate dU from stored fluxes ----
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        # Interior row/col indices in extended arrays
        cj = iy + 1  # col_j for interior row iy
        ck = iz + 1  # col_k for interior slab iz
        ci = ix + 1  # col_i for interior column ix

        F_xR = Fx_all[ix + 1, cj, ck]
        F_xL = Fx_all[ix, cj, ck]
        F_yT = Fy_all[ci, iy + 1, ck]
        F_yB = Fy_all[ci, iy, ck]
        F_zBa = Fz_all[ci, cj, iz + 1]
        F_zFr = Fz_all[ci, cj, iz]

        dU[ix + 2, iy + 2, iz + 2] = -(F_xR - F_xL) / dx - (F_yT - F_yB) / dy - (F_zBa - F_zFr) / dz
    end

    return nothing
end

# ============================================================
# 3D MHD Solver with Constrained Transport
# ============================================================

"""
    solve_hyperbolic(prob::HyperbolicProblem3D{<:IdealMHDEquations{3}}; method=:ssprk3)
        -> (coords, U_final, t_final, ct)

Solve the 3D MHD problem using constrained transport for div(B) = 0.

# Returns
- `coords`: Cell center coordinates `(x, y, z)` array.
- `U_final`: Final conserved variable array (nx x ny x nz).
- `t_final`: Final time reached.
- `ct`: Final `CTData3D` (for inspecting div(B), face-centered B, etc.).
"""
function solve_hyperbolic(
        prob::HyperbolicProblem3D{<:IdealMHDEquations{3}};
        method::Symbol = :ssprk3,
        vector_potential_x = nothing,
        vector_potential_y = nothing,
        vector_potential_z = nothing
    )
    mesh = prob.mesh
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    law = prob.law
    N = nvariables(law)  # 8

    # Initialize cell-centered solution (padded array)
    U = initialize_3d(prob)
    FT = eltype(U[3, 3, 3])

    # Initialize CT data (face-centered B)
    ct = CTData3D(nx, ny, nz, FT)
    if vector_potential_x !== nothing && vector_potential_y !== nothing && vector_potential_z !== nothing
        initialize_ct_3d_from_potential!(ct, vector_potential_x, vector_potential_y, vector_potential_z, mesh)
    else
        initialize_ct_3d!(ct, prob, mesh)
    end

    # Sync cell-centered B from face values
    face_to_cell_B_3d!(U, ct, nx, ny, nz)

    # Allocate extended face flux arrays
    zero_flux = zero(SVector{N, FT})
    Fx_all = fill(zero_flux, nx + 1, ny + 2, nz + 2)
    Fy_all = fill(zero_flux, nx + 2, ny + 1, nz + 2)
    Fz_all = fill(zero_flux, nx + 2, ny + 2, nz + 1)

    # Allocate dU
    dU = similar(U)
    zero_state = zero(SVector{N, FT})
    for k in axes(dU, 3), j in axes(dU, 2), i in axes(dU, 1)
        dU[i, j, k] = zero_state
    end

    t = prob.initial_time

    if method == :euler
        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt_3d(prob, U, t)
            if dt <= zero(dt)
                break
            end

            # Compute fluxes and dU
            _mhd_compute_fluxes_3d!(Fx_all, Fy_all, Fz_all, dU, U, prob, t)

            # Update all conserved variables via flux differencing
            for iz in 1:nz, iy in 1:ny, ix in 1:nx
                ii, jj, kk = ix + 2, iy + 2, iz + 2
                U[ii, jj, kk] = U[ii, jj, kk] + dt * dU[ii, jj, kk]
            end

            # CT: compute EMF and update face-centered B
            _compute_emf_3d_from_extended!(ct, Fx_all, Fy_all, Fz_all, nx, ny, nz)
            ct_update_3d!(ct, dt, dx, dy, dz, nx, ny, nz)
            apply_ct_periodic_3d!(ct, prob, nx, ny, nz)

            # Sync cell-centered B from face values
            face_to_cell_B_3d!(U, ct, nx, ny, nz)

            t += dt
        end

    elseif method == :ssprk3
        # Allocate RK stage arrays
        U1 = similar(U)
        U2 = similar(U)
        for k in axes(U1, 3), j in axes(U1, 2), i in axes(U1, 1)
            U1[i, j, k] = zero_state
            U2[i, j, k] = zero_state
        end

        # CT data for RK stages
        ct0 = CTData3D(nx, ny, nz, FT)
        ct1 = CTData3D(nx, ny, nz, FT)
        ct2 = CTData3D(nx, ny, nz, FT)

        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt_3d(prob, U, t)
            if dt <= zero(dt)
                break
            end

            # Save initial CT state
            copyto_ct!(ct0, ct)

            # ---- Stage 1: U1 = U + dt * L(U) ----
            _mhd_compute_fluxes_3d!(Fx_all, Fy_all, Fz_all, dU, U, prob, t)
            for iz in 1:nz, iy in 1:ny, ix in 1:nx
                ii, jj, kk = ix + 2, iy + 2, iz + 2
                U1[ii, jj, kk] = U[ii, jj, kk] + dt * dU[ii, jj, kk]
            end
            _compute_emf_3d_from_extended!(ct, Fx_all, Fy_all, Fz_all, nx, ny, nz)
            copyto_ct!(ct1, ct)
            ct_update_3d!(ct1, dt, dx, dy, dz, nx, ny, nz)
            apply_ct_periodic_3d!(ct1, prob, nx, ny, nz)
            face_to_cell_B_3d!(U1, ct1, nx, ny, nz)

            # ---- Stage 2: U2 = 3/4*U + 1/4*(U1 + dt*L(U1)) ----
            apply_boundary_conditions_3d!(U1, prob, t + dt)
            _mhd_compute_fluxes_3d!(Fx_all, Fy_all, Fz_all, dU, U1, prob, t + dt)
            for iz in 1:nz, iy in 1:ny, ix in 1:nx
                ii, jj, kk = ix + 2, iy + 2, iz + 2
                U2[ii, jj, kk] = 0.75 * U[ii, jj, kk] + 0.25 * (U1[ii, jj, kk] + dt * dU[ii, jj, kk])
            end
            _compute_emf_3d_from_extended!(ct1, Fx_all, Fy_all, Fz_all, nx, ny, nz)
            ct_weighted_update_3d!(ct2, ct0, ct1, 0.75, 0.25, dt, dx, dy, dz, nx, ny, nz)
            apply_ct_periodic_3d!(ct2, prob, nx, ny, nz)
            face_to_cell_B_3d!(U2, ct2, nx, ny, nz)

            # ---- Stage 3: U = 1/3*U + 2/3*(U2 + dt*L(U2)) ----
            apply_boundary_conditions_3d!(U2, prob, t + 0.5 * dt)
            _mhd_compute_fluxes_3d!(Fx_all, Fy_all, Fz_all, dU, U2, prob, t + 0.5 * dt)
            for iz in 1:nz, iy in 1:ny, ix in 1:nx
                ii, jj, kk = ix + 2, iy + 2, iz + 2
                U[ii, jj, kk] = (1.0 / 3.0) * U[ii, jj, kk] + (2.0 / 3.0) * (U2[ii, jj, kk] + dt * dU[ii, jj, kk])
            end
            _compute_emf_3d_from_extended!(ct2, Fx_all, Fy_all, Fz_all, nx, ny, nz)
            ct_weighted_update_3d!(ct, ct0, ct2, 1.0 / 3.0, 2.0 / 3.0, dt, dx, dy, dz, nx, ny, nz)
            apply_ct_periodic_3d!(ct, prob, nx, ny, nz)
            face_to_cell_B_3d!(U, ct, nx, ny, nz)

            t += dt
        end
    else
        error("Unknown time integration method: $method. Use :euler or :ssprk3.")
    end

    # Extract interior solution
    U_interior = Array{SVector{N, FT}, 3}(undef, nx, ny, nz)
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        U_interior[ix, iy, iz] = U[ix + 2, iy + 2, iz + 2]
    end

    # Cell center coordinates
    coords = [(cell_center(mesh, cell_idx_3d(mesh, ix, iy, iz))) for ix in 1:nx, iy in 1:ny, iz in 1:nz]

    return coords, U_interior, t, ct
end
