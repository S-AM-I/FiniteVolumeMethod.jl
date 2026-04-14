# dynamic_mesh/mesh_update.jl — Apply displacement and recompute geometry
#
# After computing displacement, this module updates cell centers, face
# centers, cell volumes, and computes the face sweep flux phi_mesh that
# accounts for mesh velocity in ALE transport.

@doc """
    update_mesh!(
        mesh::UnstructuredFVMMesh{Dim, T},
        motion_state::MeshMotionState{Dim, T},
        dt::T,
    ) where {Dim, T}

Apply the displacement field to the mesh geometry and compute the face
sweep flux `phi_mesh`.

The update proceeds as:
1. Store current cell volumes in `motion_state.V_old`.
2. Displace cell centers by `motion_state.displacement`.
3. Recompute face centers:
   - Internal faces: midpoint of displaced owner and neighbour centers.
   - Boundary faces: displaced by owner cell displacement.
4. Approximate new cell volumes using the Jacobian ratio from displaced
   cell centers (volume scaling).
5. Compute `phi_mesh` via [`compute_mesh_flux!`](@ref).

# Arguments
- `mesh` — the FVM mesh (modified in-place)
- `motion_state` — motion state with displacement (phi_mesh updated)
- `dt` — time step size
"""
function update_mesh!(
        mesh::UnstructuredFVMMesh{Dim, T},
        motion_state::MeshMotionState{Dim, T},
        dt::T,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # 1. Store old volumes
    copyto!(motion_state.V_old, mesh.cell_volumes)

    # 2. Displace cell centers
    for c in 1:nc
        for d in 1:Dim
            mesh.cell_centers[d, c] += motion_state.displacement[c][d]
        end
    end

    # 3. Recompute face centers
    for f in 1:nf
        P = mesh.face_cells[1, f]
        N = mesh.face_cells[2, f]
        if N != 0
            # Internal face: midpoint of owner and neighbour
            for d in 1:Dim
                mesh.face_centers[d, f] = (mesh.cell_centers[d, P] + mesh.cell_centers[d, N]) / 2
            end
        else
            # Boundary face: displace by owner displacement
            for d in 1:Dim
                mesh.face_centers[d, f] += motion_state.displacement[P][d]
            end
        end
    end

    # 4. Approximate new cell volumes
    # For small displacements the volume change is approximated by
    # distributing (V_new - V_old) proportionally.  For a Cartesian mesh
    # with uniform displacement, volumes are preserved.  For non-uniform
    # displacement we use a first-order approximation via the divergence
    # of the displacement field.
    _approximate_volumes!(mesh, motion_state)

    # 5. Compute mesh flux
    compute_mesh_flux!(motion_state, mesh, dt)

    return nothing
end

@doc """
    compute_mesh_flux!(
        motion_state::MeshMotionState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T,
    ) where {Dim, T}

Compute the face sweep flux `phi_mesh` from the volume change between
the old and new mesh configurations.

For each cell, the volume change `dV = V_new - V_old` is distributed
equally to all faces of that cell (with sign convention: outward from
owner).  The face flux is `phi_mesh[f] += dV_owner / (n_faces * dt)`.

This is a simplified GCL (Geometric Conservation Law) approximation
suitable for small mesh displacements.

# Arguments
- `motion_state` — motion state (phi_mesh updated in-place)
- `mesh` — the FVM mesh (after geometry update)
- `dt` — time step size
"""
function compute_mesh_flux!(
        motion_state::MeshMotionState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Zero phi_mesh
    fill!(motion_state.phi_mesh, zero(T))

    # Guard against zero dt
    if dt <= zero(T)
        return nothing
    end

    # 2nd-order GCL: compute face sweep flux from face-velocity dotted
    # with face area vector.  The face velocity is interpolated from the
    # cell displacement field: v_face = d_face / dt.
    #
    # phi_mesh[f] = dot(v_face, S_f) = dot(d_face, S_f) / dt
    #
    # This gives a per-face flux that exactly satisfies the GCL when
    # cell volumes are updated via divergence of displacement.
    for f in 1:nf
        P = mesh.face_cells[1, f]
        N = mesh.face_cells[2, f]
        S_f = face_normal_area(mesh, f)

        if N != 0
            # Internal face: interpolate displacement to face
            d_f = (motion_state.displacement[P] + motion_state.displacement[N]) / 2
        else
            # Boundary face: use owner displacement
            d_f = motion_state.displacement[P]
        end

        motion_state.phi_mesh[f] = dot(d_f, S_f) / dt
    end

    return nothing
end

# ── Internal helpers ────────────────────────────────────────────────

"""
Approximate new cell volumes from the divergence of the displacement field.

For each cell, estimates volume change from the net face-area-weighted
displacement flux.  Falls back to preserving old volumes if displacement
is uniform (solid-body motion).
"""
function _approximate_volumes!(
        mesh::UnstructuredFVMMesh{Dim, T},
        motion_state::MeshMotionState{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Compute divergence of displacement: div(d) = sum_f (d_f . S_f) / V
    div_d = zeros(T, nc)

    for f in 1:nf
        P = mesh.face_cells[1, f]
        N = mesh.face_cells[2, f]
        S_f = face_normal_area(mesh, f)

        if N != 0
            # Internal face: interpolate displacement
            d_f = (motion_state.displacement[P] + motion_state.displacement[N]) / 2
            flux = dot(d_f, S_f)
            div_d[P] += flux
            div_d[N] -= flux
        else
            # Boundary face: use owner displacement
            d_f = motion_state.displacement[P]
            flux = dot(d_f, S_f)
            div_d[P] += flux
        end
    end

    # Update volumes: V_new = V_old * (1 + div(d))
    for c in 1:nc
        V_old_c = motion_state.V_old[c]
        if V_old_c > zero(T)
            mesh.cell_volumes[c] = V_old_c * (one(T) + div_d[c] / V_old_c)
        end
    end

    return nothing
end
