# lagrangian/two_way_coupling.jl — PSI-cell momentum and energy coupling
#
# Accumulates particle drag forces and heat transfer rates onto the
# Eulerian mesh cells using the PSI-cell (Particle-Source-In-Cell) method.

"""
    set_particle_properties!(p; diameter, density, temperature=300.0, Cp=1000.0)

Initialize physical properties on a `LagrangianParticle`.  Computes and
stores mass as `pi/6 * d^3 * rho_p`.
"""
function set_particle_properties!(
        p::LagrangianParticle;
        diameter, density, temperature = 300.0, Cp = 1000.0,
    )
    p.properties[:diameter] = diameter
    p.properties[:density] = density
    p.properties[:temperature] = temperature
    p.properties[:Cp] = Cp
    p.properties[:mass] = pi / 6 * diameter^3 * density
    return nothing
end

"""
    compute_momentum_source(tracker, drag_model, U, rho_f, mu_f, mesh)

Compute per-cell momentum source from particle drag forces (PSI-cell method).

Returns `Vector{SVector{Dim, T}}` of length `ncells` where each entry is
`S_mom[c] = -(1/V_c) * sum(F_drag_p)` for all active particles in cell `c`.
"""
function compute_momentum_source(
        tracker::ParticleTracker{Dim, T},
        drag_model::AbstractDragModel,
        U::CollocatedVectorField{Dim, T},
        rho_f::T, mu_f::T,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    source = [zero(SVector{Dim, T}) for _ in 1:nc]

    for p in tracker.particles
        p.active || continue
        c = p.cell_index
        (c < 1 || c > nc) && continue

        d = T(p.properties[:diameter])
        rho_p = T(p.properties[:density])
        U_f = U.internal[c]

        F_drag = compute_drag_force(drag_model, U_f, p.velocity, d, rho_p, rho_f, mu_f)
        # Reaction on fluid: -F_drag / V_c
        source[c] = source[c] - F_drag / mesh.cell_volumes[c]
    end

    return source
end

"""
    compute_energy_source(tracker, heat_model, T_field, U, rho_f, mu_f, k_f, Pr, mesh)

Compute per-cell energy source from particle heat transfer (PSI-cell method).

Returns `Vector{T}` of length `ncells` where each entry is
`S_energy[c] = -(1/V_c) * sum(q_p)` for all active particles in cell `c`.
"""
function compute_energy_source(
        tracker::ParticleTracker{Dim, T},
        heat_model::AbstractParticleHeatTransfer,
        T_field::CollocatedScalarField{T},
        U::CollocatedVectorField{Dim, T},
        rho_f::T, mu_f::T,
        k_f::T, Pr::T,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    source = zeros(T, nc)

    for p in tracker.particles
        p.active || continue
        c = p.cell_index
        (c < 1 || c > nc) && continue

        d = T(p.properties[:diameter])
        T_p = T(p.properties[:temperature])
        U_f = U.internal[c]
        T_f = T_field.internal[c]

        q = compute_particle_heat_transfer(
            heat_model, T_f, T_p, U_f, p.velocity, d, rho_f, mu_f, k_f, Pr,
        )
        # Reaction on fluid: -q / V_c
        source[c] -= q / mesh.cell_volumes[c]
    end

    return source
end
