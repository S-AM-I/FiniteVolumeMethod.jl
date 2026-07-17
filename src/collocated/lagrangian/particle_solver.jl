# lagrangian/particle_solver.jl — Lagrangian particle time integration
#
# Advances particles through a collocated velocity field using forward
# Euler integration with drag, gravity, and optional heat transfer.


"""
    _is_in_domain(mesh, point) -> Bool

Check whether `point` lies inside the bounding box of the mesh
(with a small tolerance).
"""
function _is_in_domain(
        mesh::UnstructuredFVMMesh{Dim, T},
        point::SVector{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nc == 0 && return false
    tol = T(1.0e-10)
    for d in 1:Dim
        lo = T(Inf)
        hi = T(-Inf)
        for c in 1:nc
            xd = mesh.cell_centers[d, c]
            xd < lo && (lo = xd)
            xd > hi && (hi = xd)
        end
        # Extend bounding box by half-cell width estimate
        span = hi - lo
        margin = nc > 1 ? span / (nc - 1) : span
        lo_ext = lo - margin / 2 - tol
        hi_ext = hi + margin / 2 + tol
        (point[d] < lo_ext || point[d] > hi_ext) && return false
    end
    return true
end

"""
    advance_particles!(tracker, U, mesh, dt; integration = :euler, kwargs...)

Advance all active particles by one time step `dt`.

# Integration schemes
- `:euler` — Forward Euler (1st order, default)
- `:rk2` — Heun's method (2nd order)
- `:rk4` — Classical Runge-Kutta (4th order)

For each active particle:
1. Interpolate fluid velocity at the particle's cell center
2. Compute drag force and gravity
3. Update velocity and position using the selected scheme
4. Optionally compute heat transfer and update particle temperature
5. Update cell index; deactivate if particle leaves the domain

# Keyword Arguments
- `integration::Symbol` — time integration scheme (default `:euler`)
- `n_substeps::Int` — number of sub-steps per fluid dt (default 1)
- `drag_model`, `heat_model`, `T_field`, `rho_f`, `mu_f`, `k_f`, `Pr`, `gravity`
"""
function advance_particles!(
        tracker::ParticleTracker{Dim, T},
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T;
        integration::Symbol = :euler,
        n_substeps::Int = 1,
        drag_model::AbstractDragModel = SchillerNaumann(),
        heat_model::Union{Nothing, AbstractParticleHeatTransfer} = nothing,
        T_field::Union{Nothing, CollocatedScalarField{T}} = nothing,
        rho_f::T = one(T),
        mu_f::T = T(1.0e-3),
        k_f::T = T(0.026),
        Pr::T = T(0.7),
        gravity::SVector{Dim, T} = zero(SVector{Dim, T}),
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    dt_sub = dt / T(n_substeps)

    for p in tracker.particles
        p.active || continue

        # Ensure valid cell index
        if p.cell_index < 1 || p.cell_index > nc
            p.cell_index = find_nearest_cell(mesh, p.position)
        end
        if p.cell_index < 1 || p.cell_index > nc
            p.active = false
            continue
        end

        # Retrieve particle properties
        d_p = T(p.properties[:diameter])
        rho_p = T(p.properties[:density])
        m_p = T(p.properties[:mass])

        for _ in 1:n_substeps
            p.active || break

            # Fluid velocity at particle cell
            U_f = U.internal[p.cell_index]

            # Acceleration function: given (v, x) → a
            function _accel(v, x)
                F_drag = compute_drag_force(drag_model, U_f, v, d_p, rho_p, rho_f, mu_f)
                return (F_drag + m_p * gravity) / m_p
            end

            if integration === :euler
                a1 = _accel(p.velocity, p.position)
                v_new = p.velocity + dt_sub * a1
                x_new = p.position + dt_sub * v_new

            elseif integration === :rk2
                # Heun's method (RK2)
                a1 = _accel(p.velocity, p.position)
                v_star = p.velocity + dt_sub * a1
                x_star = p.position + dt_sub * v_star
                a2 = _accel(v_star, x_star)
                v_new = p.velocity + dt_sub / 2 * (a1 + a2)
                x_new = p.position + dt_sub / 2 * (p.velocity + v_new)

            elseif integration === :rk4
                # Classical RK4
                a1 = _accel(p.velocity, p.position)
                v1 = p.velocity
                v2 = p.velocity + dt_sub / 2 * a1
                x2 = p.position + dt_sub / 2 * v1
                a2 = _accel(v2, x2)
                v3 = p.velocity + dt_sub / 2 * a2
                x3 = p.position + dt_sub / 2 * v2
                a3 = _accel(v3, x3)
                v4 = p.velocity + dt_sub * a3
                x4 = p.position + dt_sub * v3
                a4 = _accel(v4, x4)
                v_new = p.velocity + dt_sub / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
                x_new = p.position + dt_sub / 6 * (v1 + 2 * v2 + 2 * v3 + v4)

            else
                error("Unknown integration scheme :$integration. Use :euler, :rk2, or :rk4.")
            end

            # Heat transfer (optional)
            if heat_model !== nothing && T_field !== nothing
                T_p = T(p.properties[:temperature])
                Cp_p = T(p.properties[:Cp])
                T_f = T_field.internal[p.cell_index]
                q = compute_particle_heat_transfer(
                    heat_model, T_f, T_p, U_f, p.velocity, d_p, rho_f, mu_f, k_f, Pr,
                )
                dT = q / (m_p * Cp_p) * dt_sub
                p.properties[:temperature] = T_p + dT
            end

            # Update state
            p.velocity = v_new
            p.position = x_new

            # Update cell index
            if _is_in_domain(mesh, x_new)
                p.cell_index = find_nearest_cell(mesh, x_new)
            else
                p.active = false
                p.cell_index = 0
            end
        end
    end

    return nothing
end
