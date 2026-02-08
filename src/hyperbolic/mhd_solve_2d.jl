# ============================================================
# 2D MHD Solver with Constrained Transport
# ============================================================
#
# Specializes solve_hyperbolic for HyperbolicProblem2D with
# IdealMHDEquations{2}. Uses constrained transport (CT) to
# maintain ∇·B = 0 to machine precision.
#
# Approach:
#   1. All 8 conserved variables updated via flux differencing
#   2. Face-centered B updated via CT (corner EMF from face fluxes)
#   3. Cell-centered Bx, By overwritten from face-centered values
#
# Face flux storage uses extended arrays that include ghost rows
# (for x-sweeps) and ghost columns (for y-sweeps) so that the
# corner EMF can be computed uniformly at all corners.

# ============================================================
# 2D ReflectiveBC for MHD
# ============================================================

# Left wall: negate vx (index 2)
function apply_bc_2d_left!(U::AbstractMatrix, ::ReflectiveBC, law::IdealMHDEquations{2}, nx::Int, ny::Int, t)
    for j in 1:(ny + 4)
        w1 = conserved_to_primitive(law, U[3, j])
        w2 = conserved_to_primitive(law, U[4, j])
        U[2, j] = primitive_to_conserved(law, SVector(w1[1], -w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[1, j] = primitive_to_conserved(law, SVector(w2[1], -w2[2], w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

# Right wall: negate vx
function apply_bc_2d_right!(U::AbstractMatrix, ::ReflectiveBC, law::IdealMHDEquations{2}, nx::Int, ny::Int, t)
    for j in 1:(ny + 4)
        w1 = conserved_to_primitive(law, U[nx + 2, j])
        w2 = conserved_to_primitive(law, U[nx + 1, j])
        U[nx + 3, j] = primitive_to_conserved(law, SVector(w1[1], -w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[nx + 4, j] = primitive_to_conserved(law, SVector(w2[1], -w2[2], w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

# Bottom wall: negate vy (index 3)
function apply_bc_2d_bottom!(U::AbstractMatrix, ::ReflectiveBC, law::IdealMHDEquations{2}, nx::Int, ny::Int, t)
    for i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, 3])
        w2 = conserved_to_primitive(law, U[i, 4])
        U[i, 2] = primitive_to_conserved(law, SVector(w1[1], w1[2], -w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[i, 1] = primitive_to_conserved(law, SVector(w2[1], w2[2], -w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

# Top wall: negate vy
function apply_bc_2d_top!(U::AbstractMatrix, ::ReflectiveBC, law::IdealMHDEquations{2}, nx::Int, ny::Int, t)
    for i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, ny + 2])
        w2 = conserved_to_primitive(law, U[i, ny + 1])
        U[i, ny + 3] = primitive_to_conserved(law, SVector(w1[1], w1[2], -w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[i, ny + 4] = primitive_to_conserved(law, SVector(w2[1], w2[2], -w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

# ============================================================
# CT Periodic Enforcement
# ============================================================

"""
    apply_ct_periodic!(ct::CTData2D, prob::HyperbolicProblem2D, nx, ny)

Enforce face-centered B periodicity for periodic boundary conditions.
"""
function apply_ct_periodic!(ct::CTData2D, prob, nx, ny)
    # Periodic in x: Bx_face at face 1 = face nx+1
    if prob.bc_left isa PeriodicHyperbolicBC && prob.bc_right isa PeriodicHyperbolicBC
        for j in 1:ny
            avg = 0.5 * (ct.Bx_face[1, j] + ct.Bx_face[nx + 1, j])
            ct.Bx_face[1, j] = avg
            ct.Bx_face[nx + 1, j] = avg
        end
    end

    # Periodic in y: By_face at face 1 = face ny+1
    if prob.bc_bottom isa PeriodicHyperbolicBC && prob.bc_top isa PeriodicHyperbolicBC
        for i in 1:nx
            avg = 0.5 * (ct.By_face[i, 1] + ct.By_face[i, ny + 1])
            ct.By_face[i, 1] = avg
            ct.By_face[i, ny + 1] = avg
        end
    end

    return nothing
end

# ============================================================
# Compute fluxes and dU for 2D MHD
# ============================================================

"""
    _mhd_compute_fluxes_2d!(Fx_all, Fy_all, dU, U, prob, t)

Compute face fluxes (including ghost rows/columns for EMF corners)
and accumulate the cell-centered dU via flux differencing.

`Fx_all[i, j_store]` stores the x-face flux at face i (1:nx+1),
with j_store=1 for ghost row below, j_store=2:ny+1 for interior rows,
j_store=ny+2 for ghost row above.

`Fy_all[i_store, j]` stores the y-face flux at face j (1:ny+1),
with i_store=1 for ghost column left, i_store=2:nx+1 for interior columns,
i_store=nx+2 for ghost column right.
"""
function _mhd_compute_fluxes_2d!(
        Fx_all::AbstractMatrix, Fy_all::AbstractMatrix,
        dU::AbstractMatrix, U::AbstractMatrix,
        prob::HyperbolicProblem2D, t
    )
    law = prob.law
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy
    solver = prob.riemann_solver
    recon = prob.reconstruction
    N = nvariables(law)
    FT = eltype(U[3, 3])

    # Apply BCs to fill ghost cells
    apply_boundary_conditions_2d!(U, prob, t)

    # Zero dU for interior cells
    zero_state = zero(SVector{N, FT})
    for iy in 1:ny, ix in 1:nx
        dU[ix + 2, iy + 2] = zero_state
    end

    # ---- X-direction sweeps (including ghost rows) ----
    # row_idx = 1:ny+2 maps to padded jj = 2:ny+3
    for row_idx in 1:(ny + 2)
        jj = row_idx + 1  # padded j index
        for face_i in 1:(nx + 1)
            iL = face_i + 1
            iR = face_i + 2
            wL_face, wR_face = _reconstruct_face_2d(recon, law, U, iL, iR, jj, 1, nx)
            Fx_all[face_i, row_idx] = solve_riemann(solver, law, wL_face, wR_face, 1)
        end
    end

    # ---- Y-direction sweeps (including ghost columns) ----
    # col_idx = 1:nx+2 maps to padded ii = 2:nx+3
    for col_idx in 1:(nx + 2)
        ii = col_idx + 1  # padded i index
        for face_j in 1:(ny + 1)
            jL = face_j + 1
            jR = face_j + 2
            wL_face, wR_face = _reconstruct_face_2d_y(recon, law, U, ii, jL, jR, ny)
            Fy_all[col_idx, face_j] = solve_riemann(solver, law, wL_face, wR_face, 2)
        end
    end

    # ---- Accumulate dU from stored fluxes ----
    for iy in 1:ny, ix in 1:nx
        # Fx_all row_idx for interior row iy: row_idx = iy + 1
        F_right = Fx_all[ix + 1, iy + 1]
        F_left = Fx_all[ix, iy + 1]
        # Fy_all col_idx for interior column ix: col_idx = ix + 1
        G_top = Fy_all[ix + 1, iy + 1]
        G_bottom = Fy_all[ix + 1, iy]
        dU[ix + 2, iy + 2] = -(F_right - F_left) / dx - (G_top - G_bottom) / dy
    end

    return nothing
end

# ============================================================
# Compute corner EMF from extended face fluxes
# ============================================================

"""
    _compute_emf_from_extended!(emf_z, Fx_all, Fy_all, nx, ny)

Compute the corner EMF uniformly at all corners from the extended face flux arrays.

Corner (i,j) for i=1:nx+1, j=1:ny+1:
  Ez = 0.25 * (-Fx_all[i,j][7] - Fx_all[i,j+1][7] + Fy_all[i,j][6] + Fy_all[i+1,j][6])
"""
function _compute_emf_from_extended!(emf_z::AbstractMatrix, Fx_all::AbstractMatrix,
        Fy_all::AbstractMatrix, nx::Int, ny::Int)
    for j in 1:(ny + 1), i in 1:(nx + 1)
        # x-face EMF: Ez = -F_By (negative of By induction flux)
        Ez_x_below = -Fx_all[i, j][7]       # x-face in row below corner
        Ez_x_above = -Fx_all[i, j + 1][7]   # x-face in row above corner

        # y-face EMF: Ez = +G_Bx (positive of Bx induction flux)
        Ez_y_left = Fy_all[i, j][6]         # y-face to left of corner
        Ez_y_right = Fy_all[i + 1, j][6]    # y-face to right of corner

        emf_z[i, j] = 0.25 * (Ez_x_below + Ez_x_above + Ez_y_left + Ez_y_right)
    end
    return nothing
end

# ============================================================
# 2D MHD Solver with Constrained Transport
# ============================================================

"""
    solve_hyperbolic(prob::HyperbolicProblem2D{<:IdealMHDEquations{2}}; method=:ssprk3)
        -> (coords, U_final, t_final, ct)

Solve the 2D MHD problem using constrained transport for ∇·B = 0.

# Returns
- `coords`: Cell center coordinates `(x, y)` matrix.
- `U_final`: Final conserved variable matrix (nx × ny).
- `t_final`: Final time reached.
- `ct`: Final `CTData2D` (for inspecting ∇·B, face-centered B, etc.).
"""
function solve_hyperbolic(prob::HyperbolicProblem2D{<:IdealMHDEquations{2}};
        method::Symbol = :ssprk3, vector_potential = nothing)
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy
    law = prob.law
    N = nvariables(law)  # 8

    # Initialize cell-centered solution (padded array)
    U = initialize_2d(prob)
    FT = eltype(U[3, 3])

    # Initialize CT data (face-centered B)
    ct = CTData2D(nx, ny, FT)
    if vector_potential !== nothing
        initialize_ct_from_potential!(ct, vector_potential, mesh)
    else
        initialize_ct!(ct, prob, mesh)
    end

    # Sync cell-centered B from face values
    face_to_cell_B!(U, ct, nx, ny)

    # Allocate extended face flux arrays
    zero_flux = zero(SVector{N, FT})
    Fx_all = fill(zero_flux, nx + 1, ny + 2)   # x-faces: rows include ghost rows
    Fy_all = fill(zero_flux, nx + 2, ny + 1)   # y-faces: cols include ghost cols

    # Allocate dU
    dU = similar(U)
    zero_state = zero(SVector{N, FT})
    for j in axes(dU, 2), i in axes(dU, 1)
        dU[i, j] = zero_state
    end

    t = prob.initial_time

    if method == :euler
        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt_2d(prob, U, t)
            if dt <= zero(dt)
                break
            end

            # Compute fluxes and dU
            _mhd_compute_fluxes_2d!(Fx_all, Fy_all, dU, U, prob, t)

            # Update all conserved variables via flux differencing
            for iy in 1:ny, ix in 1:nx
                ii, jj = ix + 2, iy + 2
                U[ii, jj] = U[ii, jj] + dt * dU[ii, jj]
            end

            # CT: compute EMF and update face-centered B
            _compute_emf_from_extended!(ct.emf_z, Fx_all, Fy_all, nx, ny)
            ct_update!(ct, dt, dx, dy, nx, ny)
            apply_ct_periodic!(ct, prob, nx, ny)

            # Sync cell-centered B from face values (overwrites flux-differenced B)
            face_to_cell_B!(U, ct, nx, ny)

            t += dt
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
        ct0 = CTData2D(nx, ny, FT)   # initial state for each RK step
        ct1 = CTData2D(nx, ny, FT)   # after stage 1
        ct2 = CTData2D(nx, ny, FT)   # after stage 2

        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt_2d(prob, U, t)
            if dt <= zero(dt)
                break
            end

            # Save initial CT state
            copyto_ct!(ct0, ct)

            # ---- Stage 1: U1 = U + dt * L(U) ----
            _mhd_compute_fluxes_2d!(Fx_all, Fy_all, dU, U, prob, t)
            for iy in 1:ny, ix in 1:nx
                ii, jj = ix + 2, iy + 2
                U1[ii, jj] = U[ii, jj] + dt * dU[ii, jj]
            end
            _compute_emf_from_extended!(ct.emf_z, Fx_all, Fy_all, nx, ny)
            copyto_ct!(ct1, ct)  # ct1 gets ct's face B and EMF
            ct_update!(ct1, dt, dx, dy, nx, ny)
            apply_ct_periodic!(ct1, prob, nx, ny)
            face_to_cell_B!(U1, ct1, nx, ny)

            # ---- Stage 2: U2 = 3/4*U + 1/4*(U1 + dt*L(U1)) ----
            apply_boundary_conditions_2d!(U1, prob, t + dt)
            _mhd_compute_fluxes_2d!(Fx_all, Fy_all, dU, U1, prob, t + dt)
            for iy in 1:ny, ix in 1:nx
                ii, jj = ix + 2, iy + 2
                U2[ii, jj] = 0.75 * U[ii, jj] + 0.25 * (U1[ii, jj] + dt * dU[ii, jj])
            end
            _compute_emf_from_extended!(ct1.emf_z, Fx_all, Fy_all, nx, ny)
            ct_weighted_update!(ct2, ct0, ct1, 0.75, 0.25, dt, dx, dy, nx, ny)
            apply_ct_periodic!(ct2, prob, nx, ny)
            face_to_cell_B!(U2, ct2, nx, ny)

            # ---- Stage 3: U = 1/3*U + 2/3*(U2 + dt*L(U2)) ----
            apply_boundary_conditions_2d!(U2, prob, t + 0.5 * dt)
            _mhd_compute_fluxes_2d!(Fx_all, Fy_all, dU, U2, prob, t + 0.5 * dt)
            for iy in 1:ny, ix in 1:nx
                ii, jj = ix + 2, iy + 2
                U[ii, jj] = (1.0 / 3.0) * U[ii, jj] + (2.0 / 3.0) * (U2[ii, jj] + dt * dU[ii, jj])
            end
            _compute_emf_from_extended!(ct2.emf_z, Fx_all, Fy_all, nx, ny)
            ct_weighted_update!(ct, ct0, ct2, 1.0 / 3.0, 2.0 / 3.0, dt, dx, dy, nx, ny)
            apply_ct_periodic!(ct, prob, nx, ny)
            face_to_cell_B!(U, ct, nx, ny)

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

    return coords, U_interior, t, ct
end
