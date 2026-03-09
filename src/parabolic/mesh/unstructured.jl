# Unstructured Mesh Types and Operations - Migrated from Simu.jl SimuGeometry/unstructured.jl

using LinearAlgebra: norm, cross, dot

# --- Concrete types for Unstructured 2D mesh ---

mutable struct UnstructuredFace2D <: AbstractFace
    nodes::Vector{Node2D}
    normal::Vector{Float64}
    area::Float64
    center::Vector{Float64}
    owner::Int    # Index of owner cell
    neighbor::Int # Index of neighbor cell (0 if boundary)
end

mutable struct UnstructuredCell2D <: AbstractCell
    nodes::Vector{Node2D}
    center::Vector{Float64}
    volume::Float64
    faces::Vector{Int} # Indices of faces
end

mutable struct UnstructuredMesh2D <: AbstractParabolicMesh
    nodes::Vector{Node2D}
    cells::Vector{UnstructuredCell2D}
    faces::Vector{UnstructuredFace2D}
end

"""
    convert_to_unstructured(mesh::Mesh2D)

Converts a structured Mesh2D to an UnstructuredMesh2D.
"""
function convert_to_unstructured(mesh::Mesh2D)
    nx = mesh.nx
    ny = mesh.ny

    # 1. Copy nodes
    nodes = copy(mesh.nodes)

    # 2. Create UnstructuredCells
    u_cells = Vector{UnstructuredCell2D}(undef, nx * ny)

    for k in 1:(nx * ny)
        c = mesh.cells[k]
        u_cells[k] = UnstructuredCell2D(c.nodes, c.center, c.volume, Int[])
    end

    # 3. Create UnstructuredFaces and connectivity
    u_faces = Vector{UnstructuredFace2D}()

    # Vertical faces (x-normal): (nx+1) * ny
    for i in 1:(nx + 1)
        for j in 1:ny
            owner_idx = 0
            neighbor_idx = 0

            if i > 1
                owner_idx = (i - 2) * ny + j
            end

            if i <= nx
                neighbor_idx = (i - 1) * ny + j
            end

            n1_idx = i + (j - 1) * (nx + 1)
            n2_idx = i + j * (nx + 1)

            face_nodes = [nodes[n1_idx], nodes[n2_idx]]
            normal = [1.0, 0.0]

            # Calculate center
            center = [(face_nodes[1].x + face_nodes[2].x) / 2.0, (face_nodes[1].y + face_nodes[2].y) / 2.0]

            area = sqrt((face_nodes[1].x - face_nodes[2].x)^2 + (face_nodes[1].y - face_nodes[2].y)^2)

            # Ensure owner is always a valid cell
            if owner_idx == 0 && neighbor_idx > 0
                owner_idx, neighbor_idx = neighbor_idx, owner_idx
                normal = -normal
            end

            push!(u_faces, UnstructuredFace2D(face_nodes, normal, area, center, owner_idx, neighbor_idx))
            face_idx = length(u_faces)

            # Add to cell connectivity
            if owner_idx > 0
                push!(u_cells[owner_idx].faces, face_idx)
            end
            if neighbor_idx > 0
                push!(u_cells[neighbor_idx].faces, face_idx)
            end
        end
    end

    # Horizontal faces (y-normal): nx * (ny+1)
    for i in 1:nx
        for j in 1:(ny + 1)
            owner_idx = 0
            neighbor_idx = 0

            if j > 1
                owner_idx = (i - 1) * ny + (j - 1)
            end

            if j <= ny
                neighbor_idx = (i - 1) * ny + j
            end

            n1_idx = i + (j - 1) * (nx + 1)
            n2_idx = (i + 1) + (j - 1) * (nx + 1)

            face_nodes = [nodes[n1_idx], nodes[n2_idx]]
            normal = [0.0, 1.0]
            center = [(face_nodes[1].x + face_nodes[2].x) / 2.0, (face_nodes[1].y + face_nodes[2].y) / 2.0]
            area = sqrt((face_nodes[1].x - face_nodes[2].x)^2 + (face_nodes[1].y - face_nodes[2].y)^2)

            # Ensure owner is always a valid cell
            if owner_idx == 0 && neighbor_idx > 0
                owner_idx, neighbor_idx = neighbor_idx, owner_idx
                normal = -normal
            end

            push!(u_faces, UnstructuredFace2D(face_nodes, normal, area, center, owner_idx, neighbor_idx))
            face_idx = length(u_faces)

            if owner_idx > 0
                push!(u_cells[owner_idx].faces, face_idx)
            end
            if neighbor_idx > 0
                push!(u_cells[neighbor_idx].faces, face_idx)
            end
        end
    end

    return UnstructuredMesh2D(nodes, u_cells, u_faces)
end

# --- Concrete types for Unstructured 3D mesh ---

struct UnstructuredFace3D <: AbstractFace
    nodes::Vector{Node3D}
    normal::Vector{Float64}
    area::Float64
    center::Vector{Float64}
    owner::Int    # Index of owner cell
    neighbor::Int # Index of neighbor cell (0 if boundary)
end

struct UnstructuredCell3D <: AbstractCell
    nodes::Vector{Node3D}
    center::Vector{Float64}
    volume::Float64
    faces::Vector{Int} # Indices of faces
    type::CellType
end

struct UnstructuredMesh3D <: AbstractParabolicMesh
    nodes::Vector{Node3D}
    cells::Vector{UnstructuredCell3D}
    faces::Vector{UnstructuredFace3D}
end

"""
    convert_to_unstructured(mesh::Mesh3D)

Converts a structured Mesh3D to an UnstructuredMesh3D.
"""
function convert_to_unstructured(mesh::Mesh3D)
    nx = mesh.nx
    ny = mesh.ny
    nz = mesh.nz

    nodes = copy(mesh.nodes)

    get_cell_idx(i, j, k) = (i - 1) * ny * nz + (j - 1) * nz + k
    get_node_idx(i, j, k) = i + (j - 1) * (nx + 1) + (k - 1) * (nx + 1) * (ny + 1)

    num_cells = nx * ny * nz
    u_cells = Vector{UnstructuredCell3D}(undef, num_cells)

    for idx in 1:num_cells
        c = mesh.cells[idx]
        u_cells[idx] = UnstructuredCell3D(c.nodes, c.center, c.volume, Int[], CT_Hexahedron)
    end

    u_faces = Vector{UnstructuredFace3D}()

    # 1. X-faces (Normal [1,0,0])
    for i in 1:(nx + 1)
        for j in 1:ny
            for k in 1:nz
                owner_idx = 0
                neighbor_idx = 0

                if i > 1
                    owner_idx = get_cell_idx(i - 1, j, k)
                end
                if i <= nx
                    neighbor_idx = get_cell_idx(i, j, k)
                end

                n1 = get_node_idx(i, j, k)
                n2 = get_node_idx(i, j + 1, k)
                n3 = get_node_idx(i, j + 1, k + 1)
                n4 = get_node_idx(i, j, k + 1)

                face_nodes = [nodes[n1], nodes[n2], nodes[n3], nodes[n4]]

                dy = abs(nodes[n2].y - nodes[n1].y)
                dz = abs(nodes[n4].z - nodes[n1].z)
                area = dy * dz
                if area == 0
                    d1 = [nodes[n3].x - nodes[n1].x, nodes[n3].y - nodes[n1].y, nodes[n3].z - nodes[n1].z]
                    d2 = [nodes[n4].x - nodes[n2].x, nodes[n4].y - nodes[n2].y, nodes[n4].z - nodes[n2].z]
                    cp = cross(d1, d2)
                    area = 0.5 * norm(cp)
                end

                center = [
                    (nodes[n1].x + nodes[n3].x) / 2,
                    (nodes[n1].y + nodes[n3].y) / 2,
                    (nodes[n1].z + nodes[n3].z) / 2,
                ]

                normal = [1.0, 0.0, 0.0]

                if owner_idx == 0 && neighbor_idx > 0
                    owner_idx, neighbor_idx = neighbor_idx, owner_idx
                    normal = -normal
                end

                push!(u_faces, UnstructuredFace3D(face_nodes, normal, area, center, owner_idx, neighbor_idx))
                f_idx = length(u_faces)

                if owner_idx > 0
                    push!(u_cells[owner_idx].faces, f_idx)
                end
                if neighbor_idx > 0
                    push!(u_cells[neighbor_idx].faces, f_idx)
                end
            end
        end
    end

    # 2. Y-faces (Normal [0,1,0])
    for i in 1:nx
        for j in 1:(ny + 1)
            for k in 1:nz
                owner_idx = 0
                neighbor_idx = 0

                if j > 1
                    owner_idx = get_cell_idx(i, j - 1, k)
                end
                if j <= ny
                    neighbor_idx = get_cell_idx(i, j, k)
                end

                n1 = get_node_idx(i, j, k)
                n2 = get_node_idx(i + 1, j, k)
                n3 = get_node_idx(i + 1, j, k + 1)
                n4 = get_node_idx(i, j, k + 1)

                face_nodes = [nodes[n1], nodes[n2], nodes[n3], nodes[n4]]

                dx = abs(nodes[n2].x - nodes[n1].x)
                dz = abs(nodes[n4].z - nodes[n1].z)
                area = dx * dz
                if area == 0
                    d1 = [nodes[n3].x - nodes[n1].x, nodes[n3].y - nodes[n1].y, nodes[n3].z - nodes[n1].z]
                    d2 = [nodes[n4].x - nodes[n2].x, nodes[n4].y - nodes[n2].y, nodes[n4].z - nodes[n2].z]
                    area = 0.5 * norm(cross(d1, d2))
                end

                center = [
                    (nodes[n1].x + nodes[n3].x) / 2,
                    (nodes[n1].y + nodes[n3].y) / 2,
                    (nodes[n1].z + nodes[n3].z) / 2,
                ]

                normal = [0.0, 1.0, 0.0]

                if owner_idx == 0 && neighbor_idx > 0
                    owner_idx, neighbor_idx = neighbor_idx, owner_idx
                    normal = -normal
                end

                push!(u_faces, UnstructuredFace3D(face_nodes, normal, area, center, owner_idx, neighbor_idx))
                f_idx = length(u_faces)

                if owner_idx > 0
                    push!(u_cells[owner_idx].faces, f_idx)
                end
                if neighbor_idx > 0
                    push!(u_cells[neighbor_idx].faces, f_idx)
                end
            end
        end
    end

    # 3. Z-faces (Normal [0,0,1])
    for i in 1:nx
        for j in 1:ny
            for k in 1:(nz + 1)
                owner_idx = 0
                neighbor_idx = 0

                if k > 1
                    owner_idx = get_cell_idx(i, j, k - 1)
                end
                if k <= nz
                    neighbor_idx = get_cell_idx(i, j, k)
                end

                n1 = get_node_idx(i, j, k)
                n2 = get_node_idx(i + 1, j, k)
                n3 = get_node_idx(i + 1, j + 1, k)
                n4 = get_node_idx(i, j + 1, k)

                face_nodes = [nodes[n1], nodes[n2], nodes[n3], nodes[n4]]

                dx = abs(nodes[n2].x - nodes[n1].x)
                dy = abs(nodes[n4].y - nodes[n1].y)
                area = dx * dy
                if area == 0
                    d1 = [nodes[n3].x - nodes[n1].x, nodes[n3].y - nodes[n1].y, nodes[n3].z - nodes[n1].z]
                    d2 = [nodes[n4].x - nodes[n2].x, nodes[n4].y - nodes[n2].y, nodes[n4].z - nodes[n2].z]
                    area = 0.5 * norm(cross(d1, d2))
                end

                center = [
                    (nodes[n1].x + nodes[n3].x) / 2,
                    (nodes[n1].y + nodes[n3].y) / 2,
                    (nodes[n1].z + nodes[n3].z) / 2,
                ]

                normal = [0.0, 0.0, 1.0]

                if owner_idx == 0 && neighbor_idx > 0
                    owner_idx, neighbor_idx = neighbor_idx, owner_idx
                    normal = -normal
                end

                push!(u_faces, UnstructuredFace3D(face_nodes, normal, area, center, owner_idx, neighbor_idx))
                f_idx = length(u_faces)

                if owner_idx > 0
                    push!(u_cells[owner_idx].faces, f_idx)
                end
                if neighbor_idx > 0
                    push!(u_cells[neighbor_idx].faces, f_idx)
                end
            end
        end
    end

    return UnstructuredMesh3D(nodes, u_cells, u_faces)
end


"""
    check_mesh_quality(mesh::UnstructuredMesh3D; verbose=true)

Compute quality metrics for a 3D unstructured mesh.
Returns a dictionary with min/max/avg values for volume, non-orthogonality, and aspect ratio.
"""
function check_mesh_quality(mesh::UnstructuredMesh3D; verbose = true)
    min_vol, max_vol = Inf, -Inf
    neg_vol_count = 0
    total_vol = 0.0

    for c in mesh.cells
        vol = c.volume
        min_vol = min(min_vol, vol)
        max_vol = max(max_vol, vol)
        total_vol += vol
        if vol <= 0
            neg_vol_count += 1
        end
    end

    # Orthogonality & Aspect Ratio
    max_non_orth = 0.0
    sum_non_orth = 0.0
    count_faces = 0

    max_aspect = 0.0
    sum_aspect = 0.0

    for f in mesh.faces
        if f.owner > 0 && f.neighbor > 0
            c_own = mesh.cells[f.owner]
            c_nei = mesh.cells[f.neighbor]
            d_PN = c_nei.center - c_own.center
            dist = norm(d_PN)
            if dist > 1.0e-12
                cos_theta = abs(dot(f.normal, d_PN) / (norm(f.normal) * dist))
                theta_deg = rad2deg(acos(clamp(cos_theta, -1.0, 1.0)))
                max_non_orth = max(max_non_orth, theta_deg)
                sum_non_orth += theta_deg
                count_faces += 1
            end
        end
    end

    for c in mesh.cells
        min_e = Inf
        max_e = 0.0

        function check_edge(n1, n2)
            d = sqrt((n1.x - n2.x)^2 + (n1.y - n2.y)^2 + (n1.z - n2.z)^2)
            min_e = min(min_e, d)
            return max_e = max(max_e, d)
        end

        cell_nodes = c.nodes
        if c.type == CT_Tetrahedron
            for i in 1:4, j in (i + 1):4
                check_edge(cell_nodes[i], cell_nodes[j])
            end
        elseif c.type == CT_Hexahedron
            check_edge(cell_nodes[1], cell_nodes[2]); check_edge(cell_nodes[2], cell_nodes[3])
            check_edge(cell_nodes[3], cell_nodes[4]); check_edge(cell_nodes[4], cell_nodes[1])
            check_edge(cell_nodes[5], cell_nodes[6]); check_edge(cell_nodes[6], cell_nodes[7])
            check_edge(cell_nodes[7], cell_nodes[8]); check_edge(cell_nodes[8], cell_nodes[5])
            check_edge(cell_nodes[1], cell_nodes[5]); check_edge(cell_nodes[2], cell_nodes[6])
            check_edge(cell_nodes[3], cell_nodes[7]); check_edge(cell_nodes[4], cell_nodes[8])
        else
            for i in 1:length(cell_nodes), j in (i + 1):length(cell_nodes)
                check_edge(cell_nodes[i], cell_nodes[j])
            end
        end

        if min_e > 1.0e-12
            ar = max_e / min_e
            max_aspect = max(max_aspect, ar)
            sum_aspect += ar
        end
    end

    avg_non_orth = count_faces > 0 ? sum_non_orth / count_faces : 0.0
    avg_aspect = length(mesh.cells) > 0 ? sum_aspect / length(mesh.cells) : 0.0

    metrics = Dict(
        :min_volume => min_vol,
        :max_volume => max_vol,
        :total_volume => total_vol,
        :negative_volumes => neg_vol_count,
        :max_non_orthogonality_deg => max_non_orth,
        :avg_non_orthogonality_deg => avg_non_orth,
        :max_aspect_ratio => max_aspect,
        :avg_aspect_ratio => avg_aspect
    )

    if verbose
        println("Mesh Quality Report:")
        println("  Cells: $(length(mesh.cells))")
        println("  Faces: $(length(mesh.faces))")
        println("  Volume: Range=[$min_vol, $max_vol], Total=$total_vol")
        if neg_vol_count > 0
            println("  WARNING: $neg_vol_count cells have negative or zero volume!")
        end
        println("  Non-Orthogonality (deg): Max=$(round(max_non_orth, digits = 2)), Avg=$(round(avg_non_orth, digits = 2))")
        println("  Aspect Ratio: Max=$(round(max_aspect, digits = 2)), Avg=$(round(avg_aspect, digits = 2))")
    end

    return metrics
end

"""
    refine_uniform(mesh::UnstructuredMesh2D)

Refine the mesh uniformly by splitting each triangle into 4 similar triangles.
Returns a new UnstructuredMesh2D.
"""
function refine_uniform(mesh::UnstructuredMesh2D)
    nodes = copy(mesh.nodes)

    # Map (node_idx1, node_idx2) -> midpoint_node_idx
    edge_map = Dict{Tuple{Int, Int}, Int}()

    # Helper to get or create midpoint
    function get_midpoint(n1_idx, n2_idx)
        k = (min(n1_idx, n2_idx), max(n1_idx, n2_idx))
        if haskey(edge_map, k)
            return edge_map[k]
        end
        n1 = nodes[n1_idx]
        n2 = nodes[n2_idx]
        mid = Node2D((n1.x + n2.x) / 2, (n1.y + n2.y) / 2)
        push!(nodes, mid)
        idx = length(nodes)
        edge_map[k] = idx
        return idx
    end

    # Build node to index map
    node_to_idx = Dict{Node2D, Int}()
    for (i, n) in enumerate(mesh.nodes)
        node_to_idx[n] = i
    end

    new_cells_data = Vector{Vector{Int}}() # Store indices

    for cell in mesh.cells
        if length(cell.nodes) != 3
            error("refine_uniform only supports triangles")
        end

        # Get indices of vertices
        idx1 = node_to_idx[cell.nodes[1]]
        idx2 = node_to_idx[cell.nodes[2]]
        idx3 = node_to_idx[cell.nodes[3]]

        # Get midpoints
        m12 = get_midpoint(idx1, idx2)
        m23 = get_midpoint(idx2, idx3)
        m31 = get_midpoint(idx3, idx1)

        # 4 new triangles (indices)
        push!(new_cells_data, [idx1, m12, m31])
        push!(new_cells_data, [m12, idx2, m23])
        push!(new_cells_data, [m23, idx3, m31])
        push!(new_cells_data, [m12, m23, m31])
    end

    # Create UnstructuredCell2D objects
    u_cells = Vector{UnstructuredCell2D}()

    for indices in new_cells_data
        pts = nodes[indices]
        x1, y1 = pts[1].x, pts[1].y
        x2, y2 = pts[2].x, pts[2].y
        x3, y3 = pts[3].x, pts[3].y
        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        center = [(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3]
        push!(u_cells, UnstructuredCell2D(pts, center, area, Int[]))
    end

    # Build faces
    u_faces = Vector{UnstructuredFace2D}()
    face_map = Dict{Tuple{Int, Int}, Int}() # (n1, n2) -> face_idx

    for (c_idx, indices) in enumerate(new_cells_data)
        # 3 edges
        edges = [(indices[1], indices[2]), (indices[2], indices[3]), (indices[3], indices[1])]

        for (n1, n2) in edges
            key = (min(n1, n2), max(n1, n2))

            if haskey(face_map, key)
                f_idx = face_map[key]
                face = u_faces[f_idx]

                # Update neighbor
                new_face = UnstructuredFace2D(face.nodes, face.normal, face.area, face.center, face.owner, c_idx)
                u_faces[f_idx] = new_face

                push!(u_cells[face.owner].faces, f_idx)
                push!(u_cells[c_idx].faces, f_idx)
            else
                f_idx = length(u_faces) + 1
                face_map[key] = f_idx

                fn1 = nodes[n1]; fn2 = nodes[n2]
                center = [(fn1.x + fn2.x) / 2, (fn1.y + fn2.y) / 2]
                area = sqrt((fn1.x - fn2.x)^2 + (fn1.y - fn2.y)^2)

                dx = fn2.x - fn1.x
                dy = fn2.y - fn1.y
                normal = [dy, -dx]
                nrm = norm(normal)
                if nrm > 0
                    normal /= nrm
                end

                c_center = u_cells[c_idx].center
                d_cf = [center[1] - c_center[1], center[2] - c_center[2]]

                if dot(d_cf, normal) < 0
                    normal = -normal
                end

                push!(u_faces, UnstructuredFace2D([fn1, fn2], normal, area, center, c_idx, 0))
                push!(u_cells[c_idx].faces, f_idx)
            end
        end
    end

    return UnstructuredMesh2D(nodes, u_cells, u_faces)
end
