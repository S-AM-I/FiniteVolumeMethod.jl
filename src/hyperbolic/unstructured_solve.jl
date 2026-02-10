# ============================================================
# Unstructured Hyperbolic Solver (Cell-Centered FVM)
# ============================================================
#
# Solves hyperbolic conservation laws on unstructured triangular
# meshes using cell-centered FVM with Godunov-type Riemann
# solvers at edge interfaces.
#
# Solution U[i] is an SVector{N} of conserved variables stored
# at the centroid of triangle i.
#
# The edge-based flux loop:
#   1. For each interior edge: rotate states to normal frame,
#      solve 1D Riemann problem, rotate flux back.
#   2. For each boundary edge: apply BC to get ghost state,
#      solve Riemann problem at the interface.
#   3. Accumulate fluxes into dU with sign convention:
#      dU[left] -= F_n * L_edge / A_left
#      dU[right] += F_n * L_edge / A_right

# ============================================================
# State Rotation (physical ↔ edge-normal frame)
# ============================================================

"""
    rotate_to_normal(law, w::SVector, nx, ny)

Rotate primitive state from physical frame to edge-normal frame.
The normal velocity becomes the first velocity component.
"""
@inline function rotate_to_normal(::EulerEquations{2}, w::SVector{4}, nx, ny)
    rho, vx, vy, P = w
    vn = vx * nx + vy * ny
    vt = -vx * ny + vy * nx
    return SVector(rho, vn, vt, P)
end

@inline function rotate_to_normal(::NavierStokesEquations{2}, w::SVector{4}, nx, ny)
    rho, vx, vy, P = w
    vn = vx * nx + vy * ny
    vt = -vx * ny + vy * nx
    return SVector(rho, vn, vt, P)
end

@inline function rotate_to_normal(law::AbstractConservationLaw{2}, w::SVector{8}, nx, ny)
    # MHD-like 8-variable systems: [ρ, vx, vy, vz, P, Bx, By, Bz]
    rho, vx, vy, vz, P, Bx, By, Bz = w
    vn = vx * nx + vy * ny
    vt = -vx * ny + vy * nx
    Bn = Bx * nx + By * ny
    Bt = -Bx * ny + By * nx
    return SVector(rho, vn, vt, vz, P, Bn, Bt, Bz)
end

"""
    rotate_flux_from_normal(law, F::SVector, nx, ny)

Rotate the numerical flux from edge-normal frame back to physical frame.
"""
@inline function rotate_flux_from_normal(::EulerEquations{2}, F::SVector{4}, nx, ny)
    # F = [F_ρ, F_ρvn, F_ρvt, F_E] → [F_ρ, F_ρvx, F_ρvy, F_E]
    return SVector(F[1], F[2] * nx - F[3] * ny, F[2] * ny + F[3] * nx, F[4])
end

@inline function rotate_flux_from_normal(::NavierStokesEquations{2}, F::SVector{4}, nx, ny)
    return SVector(F[1], F[2] * nx - F[3] * ny, F[2] * ny + F[3] * nx, F[4])
end

@inline function rotate_flux_from_normal(law::AbstractConservationLaw{2}, F::SVector{8}, nx, ny)
    return SVector(
        F[1],
        F[2] * nx - F[3] * ny,
        F[2] * ny + F[3] * nx,
        F[4],
        F[5],
        F[6] * nx - F[7] * ny,
        F[6] * ny + F[7] * nx,
        F[8]
    )
end

# ============================================================
# Boundary Ghost State
# ============================================================

"""
    boundary_ghost_state(bc, law, w_interior, nx, ny)

Compute the ghost primitive state for a boundary edge given the
interior state and outward normal.
"""
@inline function boundary_ghost_state(::TransmissiveBC, law, w, nx, ny)
    return w
end

@inline function boundary_ghost_state(bc::DirichletHyperbolicBC, law, w, nx, ny)
    return bc.state
end

@inline function boundary_ghost_state(::ReflectiveBC, law::EulerEquations{2}, w, nx, ny)
    rho, vx, vy, P = w
    # Reflect normal velocity: v_ghost = v - 2*(v·n)*n
    vn = vx * nx + vy * ny
    vx_g = vx - 2 * vn * nx
    vy_g = vy - 2 * vn * ny
    return SVector(rho, vx_g, vy_g, P)
end

@inline function boundary_ghost_state(::ReflectiveBC, law::AbstractConservationLaw{2}, w::SVector{8}, nx, ny)
    rho, vx, vy, vz, P, Bx, By, Bz = w
    vn = vx * nx + vy * ny
    vx_g = vx - 2 * vn * nx
    vy_g = vy - 2 * vn * ny
    Bn = Bx * nx + By * ny
    Bx_g = Bx - 2 * Bn * nx
    By_g = By - 2 * Bn * ny
    return SVector(rho, vx_g, vy_g, vz, P, Bx_g, By_g, Bz)
end

@inline function boundary_ghost_state(bc::InflowBC, law, w, nx, ny)
    return bc.state
end

# ============================================================
# Initialization
# ============================================================

"""
    initialize_unstructured(prob::UnstructuredHyperbolicProblem)

Create the solution vector from the initial condition evaluated
at triangle centroids.
"""
function initialize_unstructured(prob::UnstructuredHyperbolicProblem)
    law = prob.law
    mesh = prob.mesh
    N = nvariables(law)

    x0, y0 = mesh.tri_centroids[1]
    w0 = prob.initial_condition(x0, y0)
    u0 = primitive_to_conserved(law, w0)
    FT = eltype(u0)

    U = Vector{SVector{N, FT}}(undef, mesh.ntri)
    for i in 1:(mesh.ntri)
        x, y = mesh.tri_centroids[i]
        w = prob.initial_condition(x, y)
        U[i] = primitive_to_conserved(law, w)
    end
    return U
end

# ============================================================
# CFL Time Step
# ============================================================

"""
    compute_dt_unstructured(prob, U, t)

Compute the time step from the CFL condition on an unstructured mesh:
  dt = cfl * min_i( area_i / Σ_edges(λ_max * L_edge) )
"""
function compute_dt_unstructured(prob::UnstructuredHyperbolicProblem, U, t)
    mesh = prob.mesh
    law = prob.law
    ntri = mesh.ntri
    FT = eltype(mesh.tri_areas)

    # Accumulate max wave speed * edge length per triangle
    speed_sum = zeros(FT, ntri)

    n_total = mesh.n_interior_edges + mesh.n_boundary_edges
    for e in 1:n_total
        left = mesh.edge_left[e]
        elen = mesh.edge_lengths[e]
        nx = mesh.edge_nx[e]
        ny = mesh.edge_ny[e]

        wL = conserved_to_primitive(law, U[left])
        # Max wave speed in normal direction
        wL_rot = rotate_to_normal(law, wL, nx, ny)
        sL = max_wave_speed(law, wL_rot, 1)

        speed_sum[left] += sL * elen

        right = mesh.edge_right[e]
        if right > 0
            wR = conserved_to_primitive(law, U[right])
            wR_rot = rotate_to_normal(law, wR, nx, ny)
            sR = max_wave_speed(law, wR_rot, 1)
            speed_sum[right] += sR * elen
        end
    end

    dt = typemax(FT)
    for i in 1:ntri
        if speed_sum[i] > zero(FT)
            dt_local = mesh.tri_areas[i] / speed_sum[i]
            dt = min(dt, dt_local)
        end
    end
    dt *= prob.cfl

    # Don't overshoot final time
    if t + dt > prob.final_time
        dt = prob.final_time - t
    end
    return dt
end

# ============================================================
# Right-Hand Side (Flux Computation)
# ============================================================

"""
    unstructured_rhs!(dU, U, prob, t)

Compute the RHS of the semi-discrete conservation law on an
unstructured mesh by looping over edges.
"""
function unstructured_rhs!(dU, U, prob::UnstructuredHyperbolicProblem, t)
    mesh = prob.mesh
    law = prob.law
    solver = prob.riemann_solver
    ntri = mesh.ntri
    N = nvariables(law)
    FT = eltype(U[1])

    # Zero dU
    zero_state = zero(SVector{N, FT})
    for i in 1:ntri
        dU[i] = zero_state
    end

    # Interior edges
    for e in 1:(mesh.n_interior_edges)
        left = mesh.edge_left[e]
        right = mesh.edge_right[e]
        nx = mesh.edge_nx[e]
        ny = mesh.edge_ny[e]
        elen = mesh.edge_lengths[e]

        wL = conserved_to_primitive(law, U[left])
        wR = conserved_to_primitive(law, U[right])

        # Rotate to normal frame
        wL_rot = rotate_to_normal(law, wL, nx, ny)
        wR_rot = rotate_to_normal(law, wR, nx, ny)

        # Solve 1D Riemann problem in normal direction
        F_rot = solve_riemann(solver, law, wL_rot, wR_rot, 1)

        # Rotate flux back to physical frame and scale by edge length
        F_phys = rotate_flux_from_normal(law, F_rot, nx, ny) * elen

        # Accumulate: flux leaves left cell, enters right cell
        dU[left] = dU[left] - F_phys / mesh.tri_areas[left]
        dU[right] = dU[right] + F_phys / mesh.tri_areas[right]
    end

    # Boundary edges
    offset = mesh.n_interior_edges
    for e in 1:(mesh.n_boundary_edges)
        idx = offset + e
        left = mesh.edge_left[idx]
        nx = mesh.edge_nx[idx]
        ny = mesh.edge_ny[idx]
        elen = mesh.edge_lengths[idx]
        seg = mesh.edge_bnd_segment[idx]

        wL = conserved_to_primitive(law, U[left])
        bc = get_bc(prob, seg)
        wR = boundary_ghost_state(bc, law, wL, nx, ny)

        # Rotate to normal frame
        wL_rot = rotate_to_normal(law, wL, nx, ny)
        wR_rot = rotate_to_normal(law, wR, nx, ny)

        # Solve 1D Riemann problem
        F_rot = solve_riemann(solver, law, wL_rot, wR_rot, 1)

        # Rotate flux back and scale
        F_phys = rotate_flux_from_normal(law, F_rot, nx, ny) * elen

        # Outward flux from interior cell
        dU[left] = dU[left] - F_phys / mesh.tri_areas[left]
    end

    return nothing
end

# ============================================================
# Time Integration
# ============================================================

"""
    solve_hyperbolic(prob::UnstructuredHyperbolicProblem; method=:ssprk3)
        -> (centroids, U_final, t_final)

Solve the hyperbolic conservation law on an unstructured mesh.

# Returns
- `centroids::Vector{Tuple{FT,FT}}`: Triangle centroid coordinates.
- `U_final::Vector{SVector{N,FT}}`: Final conserved variable vector per cell.
- `t_final::Real`: Final time reached.
"""
function solve_hyperbolic(prob::UnstructuredHyperbolicProblem; method::Symbol = :ssprk3)
    mesh = prob.mesh
    ntri = mesh.ntri
    N = nvariables(prob.law)

    U = initialize_unstructured(prob)
    FT = eltype(U[1])

    dU = similar(U)
    zero_state = zero(SVector{N, FT})
    for i in 1:ntri
        dU[i] = zero_state
    end

    t = prob.initial_time

    if method == :euler
        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt_unstructured(prob, U, t)
            if dt <= zero(dt)
                break
            end
            unstructured_rhs!(dU, U, prob, t)
            for i in 1:ntri
                U[i] = U[i] + dt * dU[i]
            end
            t += dt
        end
    elseif method == :ssprk3
        U1 = similar(U)
        U2 = similar(U)
        for i in 1:ntri
            U1[i] = zero_state
            U2[i] = zero_state
        end

        while t < prob.final_time - eps(typeof(t))
            dt = compute_dt_unstructured(prob, U, t)
            if dt <= zero(dt)
                break
            end

            # Stage 1: U1 = U + dt * L(U)
            unstructured_rhs!(dU, U, prob, t)
            for i in 1:ntri
                U1[i] = U[i] + dt * dU[i]
            end

            # Stage 2: U2 = 3/4 U + 1/4 (U1 + dt * L(U1))
            unstructured_rhs!(dU, U1, prob, t + dt)
            for i in 1:ntri
                U2[i] = 0.75 * U[i] + 0.25 * (U1[i] + dt * dU[i])
            end

            # Stage 3: U = 1/3 U + 2/3 (U2 + dt * L(U2))
            unstructured_rhs!(dU, U2, prob, t + 0.5 * dt)
            for i in 1:ntri
                U[i] = (1.0 / 3.0) * U[i] + (2.0 / 3.0) * (U2[i] + dt * dU[i])
            end

            t += dt
        end
    else
        error("Unknown time integration method: $method. Use :euler or :ssprk3.")
    end

    return mesh.tri_centroids, U, t
end
