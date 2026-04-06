# lagrangian/particle_solver.jl — Lagrangian particle time integration
#
# Advances particles through a collocated velocity field using forward
# Euler integration with drag, gravity, and optional heat transfer.

"""
    _find_cell_fvm(mesh, point) -> Int

Find the cell whose center is nearest to `point` on an `UnstructuredFVMMesh`
(brute-force nearest-cell-center lookup).  Returns 0 if the mesh has no cells.
"""
function _find_cell_fvm(
        mesh::UnstructuredFVMMesh{Dim, T},
        point::SVector{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nc == 0 && return 0
    best_cell = 1
    best_dist = T(Inf)
    for c in 1:nc
        x_c = cell_center(mesh, c)
        d = norm(point - x_c)
        if d < best_dist
            best_dist = d
            best_cell = c
        end
    end
    return best_cell
end

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
    advance_particles!(tracker, U, mesh, dt;
        drag_model=SchillerNaumann(), heat_model=nothing,
        T_field=nothing, rho_f=1.0, mu_f=1e-3, k_f=0.026, Pr=0.7,
        gravity=zero(SVector{Dim,T}))

Advance all active particles by one time step `dt` using forward Euler.

For each active particle:
1. Interpolate fluid velocity at the particle's cell center
2. Compute drag force and gravity
3. Update velocity: `v_new = v + dt * (F_drag + F_grav) / m_p`
4. Update position: `x_new = x + dt * v_new`
5. Optionally compute heat transfer and update particle temperature
6. Update cell index; deactivate if particle leaves the domain
"""
function advance_particles!(
        tracker::ParticleTracker{Dim, T},
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T;
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

    for p in tracker.particles
        p.active || continue

        # Ensure valid cell index
        if p.cell_index < 1 || p.cell_index > nc
            p.cell_index = _find_cell_fvm(mesh, p.position)
        end
        if p.cell_index < 1 || p.cell_index > nc
            p.active = false
            continue
        end

        # Retrieve particle properties
        d_p = T(p.properties[:diameter])
        rho_p = T(p.properties[:density])
        m_p = T(p.properties[:mass])

        # Fluid velocity at particle cell
        U_f = U.internal[p.cell_index]

        # Drag force
        F_drag = compute_drag_force(drag_model, U_f, p.velocity, d_p, rho_p, rho_f, mu_f)

        # Gravity force
        F_grav = m_p * gravity

        # Forward Euler velocity update
        accel = (F_drag + F_grav) / m_p
        v_new = p.velocity + dt * accel

        # Forward Euler position update
        x_new = p.position + dt * v_new

        # Heat transfer (optional)
        if heat_model !== nothing && T_field !== nothing
            T_p = T(p.properties[:temperature])
            Cp_p = T(p.properties[:Cp])
            T_f = T_field.internal[p.cell_index]
            q = compute_particle_heat_transfer(
                heat_model, T_f, T_p, U_f, p.velocity, d_p, rho_f, mu_f, k_f, Pr,
            )
            dT = q / (m_p * Cp_p) * dt
            p.properties[:temperature] = T_p + dT
        end

        # Update state
        p.velocity = v_new
        p.position = x_new

        # Update cell index
        if _is_in_domain(mesh, x_new)
            p.cell_index = _find_cell_fvm(mesh, x_new)
        else
            p.active = false
            p.cell_index = 0
        end
    end

    return nothing
end
