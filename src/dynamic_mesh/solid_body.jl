# dynamic_mesh/solid_body.jl — Prescribed solid-body mesh displacement
#
# For SolidBodyMotion, all cells receive the same displacement vector
# at time t.  No PDE solve is required.

@doc """
    compute_displacement!(
        motion_state::MeshMotionState{Dim, T},
        solver::SolidBodyMotion{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        t,
    ) where {Dim, T}

Compute per-cell displacement for [`SolidBodyMotion`](@ref).

Every cell receives the same displacement `solver.displacement_func(t)`.
The result is stored in `motion_state.displacement`.

# Arguments
- `motion_state` — mutable motion state (modified in-place)
- `solver` — solid-body motion solver with `displacement_func`
- `mesh` — the FVM mesh
- `t` — current simulation time
"""
function compute_displacement!(
        motion_state::MeshMotionState{Dim, T},
        solver::SolidBodyMotion{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        t,
    ) where {Dim, T}
    d = solver.displacement_func(t)
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        motion_state.displacement[c] = d
    end
    return nothing
end
