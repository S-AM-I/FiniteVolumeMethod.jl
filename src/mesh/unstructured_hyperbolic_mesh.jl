# ============================================================
# Unstructured Hyperbolic Mesh (Cell-Centered FVM on Triangles)
# ============================================================
#
# Preprocesses a DelaunayTriangulation into an edge-based data
# structure suitable for Godunov-type hyperbolic solvers.
#
# Each triangle is a finite-volume cell with the solution stored
# at its centroid. Riemann problems are solved at each edge
# interface between adjacent cells.

"""
    UnstructuredHyperbolicMesh{FT}

Edge-based unstructured mesh for cell-centered finite volume
methods on triangulations.

# Fields
- `ntri::Int`: Number of triangles (cells).
- `tri_verts::Vector{NTuple{3,Int}}`: Vertex indices for each triangle.
- `tri_areas::Vector{FT}`: Area of each triangle.
- `tri_centroids::Vector{Tuple{FT,FT}}`: Centroid coordinates.
- `n_interior_edges::Int`: Number of interior edges.
- `n_boundary_edges::Int`: Number of boundary edges.
- `edge_nx::Vector{FT}`: Edge unit normal x-component.
- `edge_ny::Vector{FT}`: Edge unit normal y-component.
- `edge_lengths::Vector{FT}`: Edge lengths.
- `edge_left::Vector{Int}`: Left triangle ID (always valid).
- `edge_right::Vector{Int}`: Right triangle ID (0 for boundary).
- `edge_bnd_segment::Vector{Int}`: Boundary segment ID (0 for interior).
- `vertex_coords::Dict{Int,Tuple{FT,FT}}`: Vertex coordinates.
"""
struct UnstructuredHyperbolicMesh{FT}
    ntri::Int
    tri_verts::Vector{NTuple{3, Int}}
    tri_areas::Vector{FT}
    tri_centroids::Vector{Tuple{FT, FT}}

    n_interior_edges::Int
    n_boundary_edges::Int

    # Edge data: interior edges first (1:n_interior_edges),
    # then boundary (n_interior_edges+1:end)
    edge_nx::Vector{FT}
    edge_ny::Vector{FT}
    edge_lengths::Vector{FT}
    edge_left::Vector{Int}
    edge_right::Vector{Int}
    edge_bnd_segment::Vector{Int}

    vertex_coords::Dict{Int, Tuple{FT, FT}}
end

"""
    nedges(mesh::UnstructuredHyperbolicMesh) -> Int

Total number of edges (interior + boundary).
"""
nedges(mesh::UnstructuredHyperbolicMesh) = mesh.n_interior_edges + mesh.n_boundary_edges

"""
    UnstructuredHyperbolicMesh(tri::Triangulation) -> UnstructuredHyperbolicMesh

Build an edge-based mesh from a DelaunayTriangulation.

Enumerates all triangles, computes areas and centroids, then
extracts unique edges with left/right triangle adjacency and
outward normals.
"""
function UnstructuredHyperbolicMesh(tri)
    # ----------------------------------------------------------
    # Step 1: Enumerate triangles
    # ----------------------------------------------------------
    tri_list = NTuple{3, Int}[]
    for T in each_solid_triangle(tri)
        i, j, k = triangle_vertices(T)
        push!(tri_list, (Int(i), Int(j), Int(k)))
    end
    ntri = length(tri_list)

    # Map from vertex triple → triangle ID (try all 3 rotations)
    verts_to_id = Dict{NTuple{3, Int}, Int}()
    for (id, (i, j, k)) in enumerate(tri_list)
        verts_to_id[(i, j, k)] = id
        verts_to_id[(j, k, i)] = id
        verts_to_id[(k, i, j)] = id
    end

    # ----------------------------------------------------------
    # Step 2: Extract vertex coordinates
    # ----------------------------------------------------------
    vertex_coords = Dict{Int, Tuple{Float64, Float64}}()
    for (_, (i, j, k)) in enumerate(tri_list)
        for v in (i, j, k)
            if !haskey(vertex_coords, v)
                p = get_point(tri, v)
                xy = getxy(p)
                vertex_coords[v] = (Float64(xy[1]), Float64(xy[2]))
            end
        end
    end

    # ----------------------------------------------------------
    # Step 3: Compute triangle areas and centroids
    # ----------------------------------------------------------
    tri_areas = Vector{Float64}(undef, ntri)
    tri_centroids = Vector{Tuple{Float64, Float64}}(undef, ntri)
    for (id, (i, j, k)) in enumerate(tri_list)
        xi, yi = vertex_coords[i]
        xj, yj = vertex_coords[j]
        xk, yk = vertex_coords[k]
        area = abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)) / 2
        tri_areas[id] = area
        tri_centroids[id] = ((xi + xj + xk) / 3, (yi + yj + yk) / 3)
    end

    # ----------------------------------------------------------
    # Step 4: Build boundary edge map for segment lookup
    # ----------------------------------------------------------
    bnd_edge_map = get_boundary_edge_map(tri)
    # Map sorted edge (min,max) → segment index
    bnd_segment_map = Dict{Tuple{Int, Int}, Int}()
    for (edge, (pos, segment_idx)) in bnd_edge_map
        i, j = DelaunayTriangulation.edge_vertices(edge)
        key = min(Int(i), Int(j)), max(Int(i), Int(j))
        bnd_segment_map[key] = Int(segment_idx)
    end

    # ----------------------------------------------------------
    # Step 5: Enumerate edges
    # ----------------------------------------------------------
    processed = Set{Tuple{Int, Int}}()

    # Temporary storage
    int_nx = Float64[]
    int_ny = Float64[]
    int_len = Float64[]
    int_left = Int[]
    int_right = Int[]

    bnd_nx = Float64[]
    bnd_ny = Float64[]
    bnd_len = Float64[]
    bnd_left = Int[]
    bnd_seg = Int[]

    for (tri_id, (i, j, k)) in enumerate(tri_list)
        for (v1, v2) in ((i, j), (j, k), (k, i))
            edge_key = (min(v1, v2), max(v1, v2))
            if edge_key in processed
                continue
            end
            push!(processed, edge_key)

            # Compute edge geometry
            x1, y1 = vertex_coords[v1]
            x2, y2 = vertex_coords[v2]
            dx, dy = x2 - x1, y2 - y1
            elen = sqrt(dx^2 + dy^2)
            # Normal to v1→v2: rotate 90° clockwise → (dy, -dx) / len
            enx, eny = dy / elen, -dx / elen

            # Find triangle on the other side
            other_v = get_adjacent(tri, v2, v1)

            if DelaunayTriangulation.is_ghost_vertex(other_v)
                # Boundary edge — ensure normal points outward
                cx, cy = tri_centroids[tri_id]
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                if (cx - mx) * enx + (cy - my) * eny > 0
                    enx, eny = -enx, -eny
                end
                seg = get(bnd_segment_map, edge_key, 1)
                push!(bnd_nx, enx)
                push!(bnd_ny, eny)
                push!(bnd_len, elen)
                push!(bnd_left, tri_id)
                push!(bnd_seg, seg)
            else
                # Interior edge — find the other triangle's ID
                other_tri_id = _find_tri_id(verts_to_id, v1, v2, Int(other_v))

                # Convention: normal points from left to right
                # "left" is the side where the normal points away from
                cx, cy = tri_centroids[tri_id]
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                dot = (cx - mx) * enx + (cy - my) * eny

                if dot < 0
                    # Normal points away from tri_id → tri_id is left
                    push!(int_nx, enx)
                    push!(int_ny, eny)
                    push!(int_len, elen)
                    push!(int_left, tri_id)
                    push!(int_right, other_tri_id)
                else
                    # Normal points toward tri_id → flip
                    push!(int_nx, -enx)
                    push!(int_ny, -eny)
                    push!(int_len, elen)
                    push!(int_left, other_tri_id)
                    push!(int_right, tri_id)
                end
            end
        end
    end

    n_int = length(int_left)
    n_bnd = length(bnd_left)

    # Concatenate: interior edges first, then boundary
    all_nx = vcat(int_nx, bnd_nx)
    all_ny = vcat(int_ny, bnd_ny)
    all_len = vcat(int_len, bnd_len)
    all_left = vcat(int_left, bnd_left)
    all_right = vcat(int_right, fill(0, n_bnd))
    all_seg = vcat(fill(0, n_int), bnd_seg)

    return UnstructuredHyperbolicMesh{Float64}(
        ntri, tri_list, tri_areas, tri_centroids,
        n_int, n_bnd,
        all_nx, all_ny, all_len, all_left, all_right, all_seg,
        vertex_coords
    )
end

"""
Find the triangle ID for a triangle containing vertices (v1, v2, v3).
"""
function _find_tri_id(verts_to_id::Dict{NTuple{3, Int}, Int}, v1::Int, v2::Int, v3::Int)
    # Try all 3 rotations
    for perm in ((v1, v2, v3), (v2, v3, v1), (v3, v1, v2))
        id = get(verts_to_id, perm, 0)
        if id != 0
            return id
        end
    end
    # Try reversed rotations (opposite orientation)
    for perm in ((v1, v3, v2), (v3, v2, v1), (v2, v1, v3))
        id = get(verts_to_id, perm, 0)
        if id != 0
            return id
        end
    end
    error("Triangle ($v1, $v2, $v3) not found in mesh")
end
