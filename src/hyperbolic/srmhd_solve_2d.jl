# ============================================================
# 2D SRMHD Solver with Constrained Transport
# ============================================================
#
# The 2D SRMHD solver follows the same pattern as the 2D MHD
# solver in mhd_solve_2d.jl: flux differencing for all 8 variables
# plus constrained transport for ∇·B = 0.
#
# CT operates on laboratory-frame B which is the same in both
# Newtonian and SR MHD, so the existing CT infrastructure
# (CTData2D, ct_update!, compute_emf_2d!) is fully reusable.

# ============================================================
# 2D ReflectiveBC for SRMHD
# ============================================================

function apply_bc_2d_left!(U::AbstractMatrix, ::ReflectiveBC, law::SRMHDEquations{2}, nx::Int, ny::Int, t)
    for j in 1:(ny + 4)
        w1 = conserved_to_primitive(law, U[3, j])
        w2 = conserved_to_primitive(law, U[4, j])
        U[2, j] = primitive_to_conserved(law, SVector(w1[1], -w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[1, j] = primitive_to_conserved(law, SVector(w2[1], -w2[2], w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

function apply_bc_2d_right!(U::AbstractMatrix, ::ReflectiveBC, law::SRMHDEquations{2}, nx::Int, ny::Int, t)
    for j in 1:(ny + 4)
        w1 = conserved_to_primitive(law, U[nx + 2, j])
        w2 = conserved_to_primitive(law, U[nx + 1, j])
        U[nx + 3, j] = primitive_to_conserved(law, SVector(w1[1], -w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[nx + 4, j] = primitive_to_conserved(law, SVector(w2[1], -w2[2], w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

function apply_bc_2d_bottom!(U::AbstractMatrix, ::ReflectiveBC, law::SRMHDEquations{2}, nx::Int, ny::Int, t)
    for i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, 3])
        w2 = conserved_to_primitive(law, U[i, 4])
        U[i, 2] = primitive_to_conserved(law, SVector(w1[1], w1[2], -w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[i, 1] = primitive_to_conserved(law, SVector(w2[1], w2[2], -w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

function apply_bc_2d_top!(U::AbstractMatrix, ::ReflectiveBC, law::SRMHDEquations{2}, nx::Int, ny::Int, t)
    for i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, ny + 2])
        w2 = conserved_to_primitive(law, U[i, ny + 1])
        U[i, ny + 3] = primitive_to_conserved(law, SVector(w1[1], w1[2], -w1[3], w1[4], w1[5], w1[6], w1[7], w1[8]))
        U[i, ny + 4] = primitive_to_conserved(law, SVector(w2[1], w2[2], -w2[3], w2[4], w2[5], w2[6], w2[7], w2[8]))
    end
    return nothing
end

# ============================================================
# 2D SRMHD Solver with CT (reuses MHD CT infrastructure)
# ============================================================

"""
    solve_hyperbolic(prob::HyperbolicProblem2D{<:SRMHDEquations{2}}; method=:ssprk3, vector_potential=nothing)

Solve the 2D SRMHD problem using constrained transport for ∇·B = 0.
Same structure as the IdealMHDEquations{2} solver.
"""
function solve_hyperbolic(
        prob::HyperbolicProblem2D{<:SRMHDEquations{2}};
        method::Symbol = :ssprk3,
        vector_potential = nothing,
        callback::Union{Nothing, Function} = nothing,
    )
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy
    law = prob.law
    N = nvariables(law)

    U = initialize_2d(prob)
    FT = eltype(U[3, 3])

    ct = CTData2D(nx, ny, FT)
    if vector_potential !== nothing
        initialize_ct_from_potential!(ct, vector_potential, mesh)
    else
        initialize_ct!(ct, prob, mesh)
    end
    face_to_cell_B!(U, ct, nx, ny)

    zero_flux = zero(SVector{N, FT})
    Fx_all = fill(zero_flux, nx + 1, ny + 2)
    Fy_all = fill(zero_flux, nx + 2, ny + 1)

    dU = similar(U)
    zero_state = zero(SVector{N, FT})
    for j in axes(dU, 2), i in axes(dU, 1)
        dU[i, j] = zero_state
    end

    t = prob.initial_time
    step = 0

    if method == :euler
        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt_2d(prob, U, t)
            if dt <= zero(dt)
                break
            end
            _mhd_compute_fluxes_2d!(Fx_all, Fy_all, dU, U, prob, t)
            for iy in 1:ny, ix in 1:nx
                ii, jj = ix + 2, iy + 2
                U[ii, jj] = U[ii, jj] + dt * dU[ii, jj]
            end
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
        U1 = similar(U)
        U2 = similar(U)
        for j in axes(U1, 2), i in axes(U1, 1)
            U1[i, j] = zero_state
            U2[i, j] = zero_state
        end

        ct0 = CTData2D(nx, ny, FT)
        ct1 = CTData2D(nx, ny, FT)
        ct2 = CTData2D(nx, ny, FT)

        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt_2d(prob, U, t)
            if dt <= zero(dt)
                break
            end
            copyto_ct!(ct0, ct)

            # Stage 1
            _mhd_compute_fluxes_2d!(Fx_all, Fy_all, dU, U, prob, t)
            for iy in 1:ny, ix in 1:nx
                ii, jj = ix + 2, iy + 2
                U1[ii, jj] = U[ii, jj] + dt * dU[ii, jj]
            end
            _compute_emf_from_extended!(ct.emf_z, Fx_all, Fy_all, nx, ny)
            copyto_ct!(ct1, ct)
            ct_update!(ct1, dt, dx, dy, nx, ny)
            apply_ct_periodic!(ct1, prob, nx, ny)
            face_to_cell_B!(U1, ct1, nx, ny)

            # Stage 2
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

            # Stage 3
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
            step += 1
            if callback !== nothing
                callback(U, t, step, dt)
            end
        end
    else
        error("Unknown time integration method: $method. Use :euler or :ssprk3.")
    end

    U_interior = Matrix{SVector{N, FT}}(undef, nx, ny)
    for iy in 1:ny, ix in 1:nx
        U_interior[ix, iy] = U[ix + 2, iy + 2]
    end
    coords = [(cell_center(mesh, cell_idx(mesh, ix, iy))) for ix in 1:nx, iy in 1:ny]

    return coords, U_interior, t, ct
end
