using StaticArrays: SVector

# ============================================================
# 1D Hyperbolic Solver
# ============================================================

"""
    initialize_1d(prob::HyperbolicProblem) -> Vector{SVector{N,FT}}

Create the padded solution array from the initial condition.
Returns a vector of length `ncells + 2*ng` where `ng` is determined by
the reconstruction scheme. Interior cells are at indices `ng+1:ncells+ng`.
"""
function initialize_1d(prob::HyperbolicProblem)
    law = prob.law
    mesh = prob.mesh
    nc = ncells(mesh)
    N = nvariables(law)
    ng = _nghost_for_reconstruction(prob.reconstruction)

    # Determine element type
    x0 = cell_center(mesh, 1)
    w0 = prob.initial_condition(x0)
    u0 = primitive_to_conserved(law, w0)
    FT = eltype(u0)

    # Allocate padded array
    U = Vector{SVector{N, FT}}(undef, nc + 2 * ng)

    # Fill interior cells
    for i in 1:nc
        x = cell_center(mesh, i)
        w = prob.initial_condition(x)
        U[i + ng] = primitive_to_conserved(law, w)
    end

    return U
end

"""
    compute_dt(prob::HyperbolicProblem, U::AbstractVector, t) -> FT

Compute the time step from the CFL condition, per cell:
  `Δt = cfl * min_i(Δx_i / max(|λ_i|))`

Uses the per-cell width `cell_volume(mesh, i)` so nonuniform 1D meshes
are handled correctly (for a uniform mesh this reduces to
`cfl * Δx / max(|λ|)`).
"""
function compute_dt(prob::HyperbolicProblem, U::AbstractVector, t)
    law = prob.law
    mesh = prob.mesh
    nc = ncells(mesh)
    cfl = prob.cfl
    ng = _nghost_for_reconstruction(prob.reconstruction)

    dx1 = cell_volume(mesh, 1)
    dt_min = typemax(typeof(dx1))
    for i in 1:nc
        w = conserved_to_primitive(law, U[i + ng])
        λ = max_wave_speed(law, w, 1)
        dt_min = min(dt_min, cell_volume(mesh, i) / λ)
    end

    dt = cfl * dt_min

    # Don't overshoot final time
    if t + dt > prob.final_time
        dt = prob.final_time - t
    end

    return dt
end

"""
    apply_boundary_conditions!(U, prob, ng, t)

Apply left and right boundary conditions to the padded solution array.
`ng` is the number of ghost cells on each side.
"""
function apply_boundary_conditions!(U::AbstractVector, prob::HyperbolicProblem, ng::Int, t)
    nc = ncells(prob.mesh)
    law = prob.law

    if prob.bc_left isa PeriodicHyperbolicBC && prob.bc_right isa PeriodicHyperbolicBC
        apply_periodic_bcs!(U, law, nc, ng, t)
    else
        apply_bc_left!(U, prob.bc_left, law, nc, ng, t)
        apply_bc_right!(U, prob.bc_right, law, nc, ng, t)
    end
    return nothing
end

"""
    hyperbolic_rhs!(dU, U, prob, t)

Compute the right-hand side of the semi-discrete conservation law:
  `dU[i]/dt = -1/Δx * (F_{i+1/2} - F_{i-1/2})`

This is the 1D version. `U` and `dU` are padded arrays (length `ncells + 2*ng`).
Only interior cells `ng+1:ncells+ng` are updated.
"""
function hyperbolic_rhs!(dU::AbstractVector, U::AbstractVector, prob::HyperbolicProblem, t)
    law = prob.law
    mesh = prob.mesh
    nc = ncells(mesh)
    solver = prob.riemann_solver
    recon = prob.reconstruction

    ng = _nghost_for_reconstruction(recon)

    # Apply BCs to fill ghost cells
    apply_boundary_conditions!(U, prob, ng, t)

    # Compute fluxes at all faces (nc + 1 faces for nc cells)
    # Face i is between cell i and cell i+1 (in original 1-based cell numbering)
    # In padded array: face i is between U[i+ng] and U[i+ng+1]
    # We need faces 0 through nc, i.e., nc+1 faces total:
    #   Face 0: left boundary face (between ghost and cell 1)
    #   Face i (1 <= i <= nc-1): internal face
    #   Face nc: right boundary face (between cell nc and ghost)
    #
    # Each face flux is computed exactly once; the flux at cell i's right
    # face is reused as the left-face flux of cell i+1. Flux differencing
    # uses the per-cell width so nonuniform 1D meshes are handled correctly.

    # Face 0 (left boundary face)
    wL_left, wR_left = _reconstruct_face(recon, law, U, 0, nc)
    F_left = solve_riemann(solver, law, wL_left, wR_left, 1)

    for i in 1:nc
        # Right face flux (face i in 0-based: between cell i and cell i+1)
        wL_right, wR_right = _reconstruct_face(recon, law, U, i, nc)
        F_right = solve_riemann(solver, law, wL_right, wR_right, 1)

        dU[i + ng] = -(F_right - F_left) / cell_volume(mesh, i)

        F_left = F_right
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
    # NoReconstruction declares `nghost = 1`, so the padded array is
    # `nc + 2` long. For face_idx in 0..nc:
    #   left  cell index = face_idx + ng     = face_idx + 1
    #   right cell index = face_idx + ng + 1 = face_idx + 2
    # The MUSCL/WENO overloads above use `+ 2 / + 3` because they run
    # with `nghost = 2`. Using those offsets here overruns U at the
    # right boundary (right_ghost is U[nc+2], not U[nc+3]).
    iL = face_idx + 1
    iR = face_idx + 2
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
function solve_hyperbolic(
        prob::HyperbolicProblem;
        method::Symbol = :ssprk3,
        callback::Union{Nothing, Function} = nothing,
        backend::AbstractBackend = CPUBackend(),
    )
    _v2_api_depwarn(:solve_hyperbolic, "`solve(prob, alg; ...)` or `sciml_problem(prob)`")
    _cpu_backend_only("solve_hyperbolic(::HyperbolicProblem)", backend)
    mesh = prob.mesh
    nc = ncells(mesh)
    N = nvariables(prob.law)
    ng = _nghost_for_reconstruction(prob.reconstruction)

    U = initialize_1d(prob)
    FT = eltype(U[ng + 1])

    dU = similar(U)
    for i in eachindex(dU)
        dU[i] = zero(eltype(U))
    end

    t = prob.initial_time
    step = 0

    if method == :euler
        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt(prob, U, t)
            if dt <= zero(dt)
                break
            end
            hyperbolic_rhs!(dU, U, prob, t)
            for i in (ng + 1):(nc + ng)
                U[i] = U[i] + dt * dU[i]
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
            for i in (ng + 1):(nc + ng)
                U1[i] = U[i] + dt * dU[i]
            end

            # Stage 2: U2 = 3/4 U + 1/4 (U1 + dt * L(U1))
            apply_boundary_conditions!(U1, prob, ng, t + dt)
            hyperbolic_rhs!(dU, U1, prob, t + dt)
            for i in (ng + 1):(nc + ng)
                U2[i] = 0.75 * U[i] + 0.25 * (U1[i] + dt * dU[i])
            end

            # Stage 3: U = 1/3 U + 2/3 (U2 + dt * L(U2))
            apply_boundary_conditions!(U2, prob, ng, t + 0.5 * dt)
            hyperbolic_rhs!(dU, U2, prob, t + 0.5 * dt)
            for i in (ng + 1):(nc + ng)
                U[i] = (1.0 / 3.0) * U[i] + (2.0 / 3.0) * (U2[i] + dt * dU[i])
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

    # Extract interior solution
    x = [cell_center(mesh, i) for i in 1:nc]
    U_interior = U[(ng + 1):(nc + ng)]

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
