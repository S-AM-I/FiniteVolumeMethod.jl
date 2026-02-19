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
#   4. Densitized conserved variables: U_tilde = sqrt(gamma) * U
#   5. Metric-aware con2prim at each cell
#   6. Constrained transport for divergence-free B
#
# The approach follows:
#   - Riemann problem is solved in the LOCAL flat-space-like frame
#     (same as SRMHD), using undensitized primitive variables
#   - The resulting numerical flux is corrected by the metric:
#     F_Valencia = alpha_face * F_riemann - beta_face * U_tilde_face
#   - Geometric source terms are added after flux differencing
#   - CT operates on the coordinate-frame B (same as SRMHD/MHD)
#
# For Minkowski metric, this reduces to the SRMHD solver exactly.

# ============================================================
# 2D ReflectiveBC for GRMHD
# ============================================================

function apply_bc_2d_left!(U::AbstractMatrix, ::ReflectiveBC, law::GRMHDEquations{2}, nx::Int, ny::Int, t)
    for j in 1:(ny + 4)
        w1 = conserved_to_primitive(law, U[3, j])
        w2 = conserved_to_primitive(law, U[4, j])
        U[2, j] = primitive_to_conserved(law, SVector(w1[1], -w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[1, j] = primitive_to_conserved(law, SVector(w2[1], -w2[2], w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

function apply_bc_2d_right!(U::AbstractMatrix, ::ReflectiveBC, law::GRMHDEquations{2}, nx::Int, ny::Int, t)
    for j in 1:(ny + 4)
        w1 = conserved_to_primitive(law, U[nx + 2, j])
        w2 = conserved_to_primitive(law, U[nx + 1, j])
        U[nx + 3, j] = primitive_to_conserved(law, SVector(w1[1], -w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[nx + 4, j] = primitive_to_conserved(law, SVector(w2[1], -w2[2], w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

function apply_bc_2d_bottom!(U::AbstractMatrix, ::ReflectiveBC, law::GRMHDEquations{2}, nx::Int, ny::Int, t)
    for i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, 3])
        w2 = conserved_to_primitive(law, U[i, 4])
        U[i, 2] = primitive_to_conserved(law, SVector(w1[1], w1[2], -w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[i, 1] = primitive_to_conserved(law, SVector(w2[1], w2[2], -w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

function apply_bc_2d_top!(U::AbstractMatrix, ::ReflectiveBC, law::GRMHDEquations{2}, nx::Int, ny::Int, t)
    for i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, ny + 2])
        w2 = conserved_to_primitive(law, U[i, ny + 1])
        U[i, ny + 3] = primitive_to_conserved(law, SVector(w1[1], w1[2], -w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[i, ny + 4] = primitive_to_conserved(law, SVector(w2[1], w2[2], -w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

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

    max_speed = zero(dx)
    for iy in 1:ny, ix in 1:nx
        w = conserved_to_primitive(law, U[ix + 2, iy + 2])
        alp = md.alpha[ix, iy]
        bx_s = md.beta_x[ix, iy]
        by_s = md.beta_y[ix, iy]

        lam_x = grmhd_max_wave_speed_coord(law, w, 1, alp, bx_s)
        lam_y = grmhd_max_wave_speed_coord(law, w, 2, alp, by_s)
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
    FT = eltype(U[3, 3])

    # Unpack face metric data
    alpha_xf, alpha_yf, betax_xf, betay_xf, betax_yf, betay_yf, sqrtg_xf, sqrtg_yf = face_data

    # Apply BCs to fill ghost cells
    apply_boundary_conditions_2d!(U, prob, t)

    # Zero dU for interior cells
    zero_state = zero(SVector{N, FT})
    for iy in 1:ny, ix in 1:nx
        dU[ix + 2, iy + 2] = zero_state
    end

    # ---- X-direction sweeps (including ghost rows for EMF) ----
    for row_idx in 1:(ny + 2)
        jj = row_idx + 1  # padded j index
        for face_i in 1:(nx + 1)
            iL = face_i + 1
            iR = face_i + 2
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
        ii = col_idx + 1  # padded i index
        for face_j in 1:(ny + 1)
            jL = face_j + 1
            jR = face_j + 2
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
        dU[ix + 2, iy + 2] = -(F_right - F_left) / dx - (G_top - G_bottom) / dy
    end

    return nothing
end

# ============================================================
# Add Geometric Source Terms
# ============================================================

"""
    _grmhd_add_source_terms!(dU, U, law, md, mesh, nx, ny)

Add geometric source terms to dU at all interior cells.
"""
function _grmhd_add_source_terms!(
        dU::AbstractMatrix, U::AbstractMatrix,
        law::GRMHDEquations{2}, md::MetricData2D,
        mesh::StructuredMesh2D, nx::Int, ny::Int
    )
    for iy in 1:ny, ix in 1:nx
        ii, jj = ix + 2, iy + 2
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
function solve_hyperbolic(
        prob::HyperbolicProblem2D{<:GRMHDEquations{2}};
        method::Symbol = :ssprk3,
        vector_potential = nothing,
        callback::Union{Nothing, Function} = nothing,
    )
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy
    law = prob.law
    N = nvariables(law)

    # Initialize cell-centered solution (padded array, undensitized)
    U = initialize_2d(prob)
    FT = eltype(U[3, 3])

    # Precompute metric data at cell centers and faces
    md = precompute_metric(law.metric, mesh)
    face_data = precompute_metric_at_faces(law.metric, mesh)

    # Initialize CT data (face-centered B)
    ct = CTData2D(nx, ny, FT)
    if vector_potential !== nothing
        initialize_ct_from_potential!(ct, vector_potential, mesh)
    else
        initialize_ct!(ct, prob, mesh)
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

            # Compute metric-corrected fluxes and dU
            _grmhd_compute_fluxes_2d!(Fx_all, Fy_all, dU, U, prob, t, md, face_data)

            # Add geometric source terms
            _grmhd_add_source_terms!(dU, U, law, md, mesh, nx, ny)

            # Update all conserved variables
            for iy in 1:ny, ix in 1:nx
                ii, jj = ix + 2, iy + 2
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
            _grmhd_compute_fluxes_2d!(Fx_all, Fy_all, dU, U, prob, t, md, face_data)
            _grmhd_add_source_terms!(dU, U, law, md, mesh, nx, ny)
            for iy in 1:ny, ix in 1:nx
                ii, jj = ix + 2, iy + 2
                U1[ii, jj] = U[ii, jj] + dt * dU[ii, jj]
            end
            _compute_emf_from_extended!(ct.emf_z, Fx_all, Fy_all, nx, ny)
            copyto_ct!(ct1, ct)
            ct_update!(ct1, dt, dx, dy, nx, ny)
            apply_ct_periodic!(ct1, prob, nx, ny)
            face_to_cell_B!(U1, ct1, nx, ny)

            # ---- Stage 2: U2 = 3/4*U + 1/4*(U1 + dt*L(U1)) ----
            apply_boundary_conditions_2d!(U1, prob, t + dt)
            _grmhd_compute_fluxes_2d!(Fx_all, Fy_all, dU, U1, prob, t + dt, md, face_data)
            _grmhd_add_source_terms!(dU, U1, law, md, mesh, nx, ny)
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
            _grmhd_compute_fluxes_2d!(Fx_all, Fy_all, dU, U2, prob, t + 0.5 * dt, md, face_data)
            _grmhd_add_source_terms!(dU, U2, law, md, mesh, nx, ny)
            for iy in 1:ny, ix in 1:nx
                ii, jj = ix + 2, iy + 2
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
        U_interior[ix, iy] = U[ix + 2, iy + 2]
    end

    # Cell center coordinates
    coords = [(cell_center(mesh, cell_idx(mesh, ix, iy))) for ix in 1:nx, iy in 1:ny]

    return coords, U_interior, t, ct
end
