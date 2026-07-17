# Mesh Partitioning and Domain Decomposition - Migrated from Simu.jl SimuGeometry/partitioning.jl

using LinearAlgebra: norm, dot

"""
    PartitionedMesh{M}

Container for a partitioned mesh distributed across logical ranks.
"""
struct PartitionedMesh{M}
    parts::Vector{M} # Local meshes
    global_to_local::Dict{Int, Tuple{Int, Int}} # Global Cell ID -> (Rank, Local Cell ID)
    halo_exchange::Vector{Dict{Int, Vector{Int}}} # Rank -> Neighbor Rank -> List of Shared Nodes/Faces
end

"""
    partition_mesh_rcb(mesh::UnstructuredMesh2D, n_parts::Int)

Partition a 2D unstructured mesh into `n_parts` using Recursive Coordinate Bisection (RCB).
Returns a `PartitionedMesh`.
"""
function partition_mesh_rcb(mesh::UnstructuredMesh2D, n_parts::Int)
    cells_info = [(i, mesh.cells[i].center[1], mesh.cells[i].center[2]) for i in 1:length(mesh.cells)]

    partitions = recursive_bisection(cells_info, n_parts, 1)

    # Construct local meshes
    local_meshes = Vector{UnstructuredMesh2D}(undef, n_parts)
    global_map = Dict{Int, Tuple{Int, Int}}()

    for (rank, cell_indices) in enumerate(partitions)
        local_mesh, local_map = extract_submesh(mesh, cell_indices)
        local_meshes[rank] = local_mesh

        for (loc_idx, glob_idx) in enumerate(cell_indices)
            global_map[glob_idx] = (rank, loc_idx)
        end
    end

    # Compute halo connectivity (simplified)
    halo_exchange = [Dict{Int, Vector{Int}}() for _ in 1:n_parts]

    return PartitionedMesh(local_meshes, global_map, halo_exchange)
end

"""Partition `cells` into `n_parts` groups using recursive coordinate bisection."""
function recursive_bisection(cells, n_parts, depth)
    if n_parts == 1
        return [[c[1] for c in cells]]
    end

    # Determine axis (x or y) based on bounding box aspect ratio
    xs = [c[2] for c in cells]
    ys = [c[3] for c in cells]
    min_x, max_x = minimum(xs), maximum(xs)
    min_y, max_y = minimum(ys), maximum(ys)

    axis = (max_x - min_x) > (max_y - min_y) ? 2 : 3

    # Sort
    sort!(cells, by = x -> x[axis])

    # Split
    mid = div(length(cells), 2)
    left_cells = cells[1:mid]
    right_cells = cells[(mid + 1):end]

    # Recurse
    n_left = div(n_parts, 2)
    n_right = n_parts - n_left

    parts_left = recursive_bisection(left_cells, n_left, depth + 1)
    parts_right = recursive_bisection(right_cells, n_right, depth + 1)

    return vcat(parts_left, parts_right)
end

"""
    extract_submesh(global_mesh, cell_indices)

Create a new UnstructuredMesh2D containing only the specified cells.
"""
function extract_submesh(mesh::UnstructuredMesh2D, cell_indices::Vector{Int})
    # 1. Collect required nodes
    used_nodes = Vector{Node2D}()
    obj_id_map = Dict{UInt, Int}() # ObjectID -> New Local Index

    for c_idx in cell_indices
        cell = mesh.cells[c_idx]
        for n in cell.nodes
            oid = objectid(n)
            if !haskey(obj_id_map, oid)
                push!(used_nodes, n)
                obj_id_map[oid] = length(used_nodes)
            end
        end
    end

    # 2. Create new cells with these nodes
    new_cells = Vector{UnstructuredCell2D}(undef, length(cell_indices))

    for (i, c_idx) in enumerate(cell_indices)
        c_old = mesh.cells[c_idx]
        new_cells[i] = UnstructuredCell2D(c_old.nodes, c_old.center, c_old.volume, Int[])
    end

    # 3. Build internal faces
    u_faces = Vector{UnstructuredFace2D}()
    face_map = Dict{Tuple{UInt, UInt}, Int}() # (oid1, oid2) -> face_idx

    for (c_idx_local, cell) in enumerate(new_cells)
        cell_nodes = cell.nodes
        n = length(cell_nodes)
        for k in 1:n
            n1 = cell_nodes[k]
            n2 = cell_nodes[k == n ? 1 : k + 1]
            oid1, oid2 = objectid(n1), objectid(n2)
            key = oid1 < oid2 ? (oid1, oid2) : (oid2, oid1)

            if haskey(face_map, key)
                f_idx = face_map[key]
                face = u_faces[f_idx]

                # Update neighbor
                new_face = UnstructuredFace2D(face.nodes, face.normal, face.area, face.center, face.owner, c_idx_local)
                u_faces[f_idx] = new_face

                push!(cell.faces, f_idx)
                push!(new_cells[face.owner].faces, f_idx)
            else
                f_idx = length(u_faces) + 1
                face_map[key] = f_idx

                # Geometry
                p1 = [n1.x, n1.y]; p2 = [n2.x, n2.y]
                center = (p1 + p2) / 2
                edge = p2 - p1
                area = norm(edge)
                normal = [edge[2], -edge[1]]
                if norm(normal) > 0
                    normal /= norm(normal)
                end

                # Check orientation
                c_center = cell.center
                d_cf = center - c_center
                if dot(d_cf, normal) < 0
                    normal = -normal
                end

                push!(u_faces, UnstructuredFace2D([n1, n2], normal, area, center, c_idx_local, 0))
                push!(cell.faces, f_idx)
            end
        end
    end

    return UnstructuredMesh2D(used_nodes, new_cells, u_faces), nothing
end
