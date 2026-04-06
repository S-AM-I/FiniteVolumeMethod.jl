# turbulence/wall_distance.jl — Cell-to-wall distance computation
#
# Computes the minimum distance from each cell center to the nearest wall
# boundary face. Required by k-ω SST (blending functions F1, F2) and
# Spalart-Allmaras (production and destruction terms).

"""
    compute_wall_distance(
        mesh::UnstructuredFVMMesh{Dim, T},
        wall_patches::Vector{Symbol},
    ) -> Vector{T}

Compute the minimum distance from each cell center to the nearest wall
boundary face center.

Identifies wall faces by matching `mesh.face_tags` against `wall_patches`.
Returns a vector of length `ncells`. Cells far from any wall face get
the distance to the nearest wall face (no capping).

Complexity: O(ncells × n_wall_faces). Computed once at setup.
"""
function compute_wall_distance(
        mesh::UnstructuredFVMMesh{Dim, T},
        wall_patches::Vector{Symbol},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Collect wall face indices
    wall_faces = Int[]
    wall_set = Set(wall_patches)
    for f in 1:nf
        if !is_internal_face(mesh, f)
            tag = _face_tag(mesh, f)
            if tag in wall_set
                push!(wall_faces, f)
            end
        end
    end

    d_wall = fill(T(Inf), nc)

    for f in wall_faces
        x_f = face_center(mesh, f)
        for c in 1:nc
            x_c = cell_center(mesh, c)
            dist = norm(x_c - x_f)
            d_wall[c] = min(d_wall[c], dist)
        end
    end

    # Safety: if no wall faces found, set to 1.0 to avoid division by zero
    if isempty(wall_faces)
        fill!(d_wall, one(T))
    end

    return d_wall
end
