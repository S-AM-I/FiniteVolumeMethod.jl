using StaticArrays: SVector

# ============================================================
# 1D Hyperbolic Solver
# ============================================================

"""
    initialize_1d(prob::HyperbolicProblem) -> Vector{SVector{N,FT}}

Create the padded solution array from the initial condition.
Returns a vector of length `ncells + 4` (2 ghost cells on each side).
Interior cells are at indices `3:ncells+2`.
"""
function initialize_1d(prob::HyperbolicProblem)
    law = prob.law
    mesh = prob.mesh
    nc = ncells(mesh)
    N = nvariables(law)

    # Determine element type
    x0 = cell_center(mesh, 1)
    w0 = prob.initial_condition(x0)
    u0 = primitive_to_conserved(law, w0)
    FT = eltype(u0)

    # Allocate padded array
    U = Vector{SVector{N, FT}}(undef, nc + 4)

    # Fill interior cells
    for i in 1:nc
        x = cell_center(mesh, i)
        w = prob.initial_condition(x)
        U[i + 2] = primitive_to_conserved(law, w)
    end

    return U
end

"""
    compute_dt(prob::HyperbolicProblem, U::AbstractVector, t) -> FT

Compute the time step from the CFL condition:
  `Δt = cfl * Δx / max(|λ|)`
"""
function compute_dt(prob::HyperbolicProblem, U::AbstractVector, t)
    law = prob.law
    mesh = prob.mesh
    nc = ncells(mesh)
    cfl = prob.cfl

    λ_max = zero(mesh.dx)
    for i in 1:nc
        w = conserved_to_primitive(law, U[i + 2])
        λ = max_wave_speed(law, w, 1)
        λ_max = max(λ_max, λ)
    end

    dt = cfl * cell_volume(mesh, 1) / λ_max

    # Don't overshoot final time
    if t + dt > prob.final_time
        dt = prob.final_time - t
    end

    return dt
end

"""
    apply_boundary_conditions!(U, prob, t)

Apply left and right boundary conditions to the padded solution array.
"""
function apply_boundary_conditions!(U::AbstractVector, prob::HyperbolicProblem, t)
    nc = ncells(prob.mesh)
    law = prob.law

    if prob.bc_left isa PeriodicHyperbolicBC && prob.bc_right isa PeriodicHyperbolicBC
        apply_periodic_bcs!(U, law, nc, t)
    else
        apply_bc_left!(U, prob.bc_left, law, nc, t)
        apply_bc_right!(U, prob.bc_right, law, nc, t)
    end
    return nothing
end

"""
    hyperbolic_rhs!(dU, U, prob, t)

Compute the right-hand side of the semi-discrete conservation law:
  `dU[i]/dt = -1/Δx * (F_{i+1/2} - F_{i-1/2})`

This is the 1D version. `U` and `dU` are padded arrays (length `ncells + 4`).
Only interior cells `3:ncells+2` are updated.
"""
function hyperbolic_rhs!(dU::AbstractVector, U::AbstractVector, prob::HyperbolicProblem, t)
    law = prob.law
    mesh = prob.mesh
    nc = ncells(mesh)
    solver = prob.riemann_solver
    recon = prob.reconstruction
    dx = cell_volume(mesh, 1)

    # Apply BCs to fill ghost cells
    apply_boundary_conditions!(U, prob, t)

    # Compute fluxes at all faces (nc + 1 faces for nc cells)
    # Face i is between cell i and cell i+1 (in original 1-based cell numbering)
    # In padded array: face i is between U[i+2] and U[i+3]
    # We need faces 0 through nc, i.e., nc+1 faces total:
    #   Face 0: left boundary face (between ghost and cell 1)
    #   Face i (1 ≤ i ≤ nc-1): internal face
    #   Face nc: right boundary face (between cell nc and ghost)

    # Compute flux at face i (0-based face index, between cell i and cell i+1)
    # In padded array terms: between U[i+2] and U[i+3]
    # For reconstruction we need U[i+1], U[i+2], U[i+3], U[i+4]

    # Update each interior cell
    for i in 1:nc
        # Left face flux (face i-1 in 0-based: between cell i-1 and cell i)
        # padded indices: U[i+1], U[i+2] are the two cells adjacent to this face
        wL_left, wR_left = _reconstruct_face(recon, law, U, i - 1, nc)
        F_left = solve_riemann(solver, law, wL_left, wR_left, 1)

        # Right face flux (face i in 0-based: between cell i and cell i+1)
        wL_right, wR_right = _reconstruct_face(recon, law, U, i, nc)
        F_right = solve_riemann(solver, law, wL_right, wR_right, 1)

        dU[i + 2] = -(F_right - F_left) / dx
    end

    return nothing
end

"""
    _reconstruct_face(recon, law, U, face_idx, ncells) -> (wL, wR)

Reconstruct left and right primitive states at a face.
`face_idx` is 0-based: face 0 is the left boundary face, face ncells is the right boundary face.
"""
@inline function _reconstruct_face(recon, law, U, face_idx, ncells)
    # face_idx (0-based) maps to:
    # Left cell: face_idx + 2 in padded array
    # Right cell: face_idx + 3 in padded array
    # For MUSCL we need: U[face_idx+1], U[face_idx+2], U[face_idx+3], U[face_idx+4]
    return reconstruct_interface_1d(recon, law, U, face_idx, ncells)
end

@inline function reconstruct_interface_1d(recon::CellCenteredMUSCL, law, U::AbstractVector, face_idx::Int, ncells::Int)
    # Padded array: face between U[face_idx+2] and U[face_idx+3]
    iL = face_idx + 2
    iR = face_idx + 3

    uLL = U[iL - 1]
    uL = U[iL]
    uR = U[iR]
    uRR = U[iR + 1]

    wLL = conserved_to_primitive(law, uLL)
    wL = conserved_to_primitive(law, uL)
    wR = conserved_to_primitive(law, uR)
    wRR = conserved_to_primitive(law, uRR)

    wL_face, wR_face = reconstruct_interface(recon, wLL, wL, wR, wRR)
    return wL_face, wR_face
end

@inline function reconstruct_interface_1d(::NoReconstruction, law, U::AbstractVector, face_idx::Int, ncells::Int)
    iL = face_idx + 2
    iR = face_idx + 3
    wL = conserved_to_primitive(law, U[iL])
    wR = conserved_to_primitive(law, U[iR])
    return wL, wR
end

# ============================================================
# Time Integration (explicit forward Euler + SSP-RK3)
# ============================================================

"""
    solve_hyperbolic(prob::HyperbolicProblem; method=:ssprk3) -> (x, U_final, t_final)

Solve the 1D hyperbolic problem using explicit time integration.

# Keyword Arguments
- `method::Symbol`: Time integration method. `:euler` for forward Euler, `:ssprk3` for
  3rd-order strong stability preserving Runge-Kutta (default).

# Returns
- `x::Vector`: Cell center coordinates.
- `U_final::Vector{SVector{N}}`: Final conserved variable vectors at cell centers.
- `t_final::Real`: Final time reached.
"""
function solve_hyperbolic(prob::HyperbolicProblem; method::Symbol = :ssprk3)
    mesh = prob.mesh
    nc = ncells(mesh)
    N = nvariables(prob.law)

    U = initialize_1d(prob)
    FT = eltype(U[3])

    dU = similar(U)
    for i in eachindex(dU)
        dU[i] = zero(eltype(U))
    end

    t = prob.initial_time

    if method == :euler
        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt(prob, U, t)
            if dt <= zero(dt)
                break
            end
            hyperbolic_rhs!(dU, U, prob, t)
            for i in 3:(nc + 2)
                U[i] = U[i] + dt * dU[i]
            end
            t += dt
        end
    elseif method == :ssprk3
        U1 = similar(U)
        U2 = similar(U)
        for i in eachindex(U1)
            U1[i] = zero(eltype(U))
            U2[i] = zero(eltype(U))
        end

        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt(prob, U, t)
            if dt <= zero(dt)
                break
            end

            # Stage 1: U1 = U + dt * L(U)
            hyperbolic_rhs!(dU, U, prob, t)
            for i in 3:(nc + 2)
                U1[i] = U[i] + dt * dU[i]
            end

            # Stage 2: U2 = 3/4 U + 1/4 (U1 + dt * L(U1))
            apply_boundary_conditions!(U1, prob, t + dt)
            hyperbolic_rhs!(dU, U1, prob, t + dt)
            for i in 3:(nc + 2)
                U2[i] = 0.75 * U[i] + 0.25 * (U1[i] + dt * dU[i])
            end

            # Stage 3: U = 1/3 U + 2/3 (U2 + dt * L(U2))
            apply_boundary_conditions!(U2, prob, t + 0.5 * dt)
            hyperbolic_rhs!(dU, U2, prob, t + 0.5 * dt)
            for i in 3:(nc + 2)
                U[i] = (1.0 / 3.0) * U[i] + (2.0 / 3.0) * (U2[i] + dt * dU[i])
            end

            t += dt
        end
    else
        error("Unknown time integration method: $method. Use :euler or :ssprk3.")
    end

    # Extract interior solution
    x = [cell_center(mesh, i) for i in 1:nc]
    U_interior = U[3:(nc + 2)]

    return x, U_interior, t
end

# ============================================================
# Convenience: convert solution to primitives
# ============================================================

"""
    to_primitive(law, U::AbstractVector{<:SVector}) -> Vector{SVector}

Convert an array of conserved variable vectors to primitive variable vectors.
"""
function to_primitive(law, U::AbstractVector)
    return [conserved_to_primitive(law, u) for u in U]
end
