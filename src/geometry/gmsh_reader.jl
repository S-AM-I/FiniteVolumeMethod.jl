# mesh/gmsh_reader.jl — Gmsh .msh v4 file reader
#
# Reads Gmsh mesh files (.msh v4 ASCII) and constructs an
# UnstructuredMesh3D. Supports tetrahedral, hexahedral, prismatic,
# and pyramidal elements. Physical groups become boundary patch tags.

using LinearAlgebra: norm, cross

"""
    read_gmsh(path::AbstractString) -> UnstructuredMesh3D

Read a Gmsh `.msh` v4 file and return an `UnstructuredMesh3D`.

Supports ASCII format only. Element types handled:
- Type 4: Tetrahedron (4 nodes)
- Type 5: Hexahedron (8 nodes)
- Type 6: Prism (6 nodes)
- Type 7: Pyramid (5 nodes)
- Type 2: Triangle (3 nodes, boundary face)
- Type 3: Quadrilateral (4 nodes, boundary face)

Physical groups are used as boundary patch names. Surface elements
(triangles, quads) define boundary faces.

# Arguments
- `path` — path to `.msh` file
"""
function read_gmsh(path::AbstractString)
    nodes = Dict{Int, Node3D}()
    volume_elements = Vector{NamedTuple{(:type, :nodes, :tag), Tuple{Int, Vector{Int}, Int}}}()
    surface_elements = Vector{NamedTuple{(:type, :nodes, :tag), Tuple{Int, Vector{Int}, Int}}}()
    physical_names = Dict{Int, Symbol}()

    open(path) do io
        while !eof(io)
            line = readline(io)
            stripped = strip(line)

            if stripped == "\$PhysicalNames"
                _read_gmsh_physical_names!(io, physical_names)
            elseif stripped == "\$Nodes"
                _read_gmsh_nodes!(io, nodes)
            elseif stripped == "\$Elements"
                _read_gmsh_elements!(io, volume_elements, surface_elements)
            end
        end
    end

    # Build UnstructuredMesh3D from parsed data
    return _build_mesh_from_gmsh(nodes, volume_elements, surface_elements, physical_names)
end

# ── Internal parsers ──────────────────────────────────────────────

function _read_gmsh_physical_names!(io::IO, physical_names::Dict{Int, Symbol})
    n = parse(Int, strip(readline(io)))
    for _ in 1:n
        parts = split(strip(readline(io)))
        # dim, tag, "name"
        tag = parse(Int, parts[2])
        name = Symbol(replace(parts[3], "\"" => ""))
        physical_names[tag] = name
    end
    # Read closing $EndPhysicalNames
    return readline(io)
end

function _read_gmsh_nodes!(io::IO, nodes::Dict{Int, Node3D})
    header = split(strip(readline(io)))
    # v4 format: numEntityBlocks numNodes minNodeTag maxNodeTag
    num_blocks = parse(Int, header[1])
    for _ in 1:num_blocks
        block_header = split(strip(readline(io)))
        # entityDim entityTag parametric numNodesInBlock
        n_block = parse(Int, block_header[4])

        # Read node tags
        node_tags = Vector{Int}(undef, n_block)
        for i in 1:n_block
            node_tags[i] = parse(Int, strip(readline(io)))
        end

        # Read coordinates
        for i in 1:n_block
            coords = split(strip(readline(io)))
            x = parse(Float64, coords[1])
            y = parse(Float64, coords[2])
            z = parse(Float64, coords[3])
            nodes[node_tags[i]] = Node3D(x, y, z)
        end
    end
    return readline(io)  # $EndNodes
end

# Gmsh element type → number of nodes
const _GMSH_ELEM_NODES = Dict(
    1 => 2,   # line
    2 => 3,   # triangle
    3 => 4,   # quad
    4 => 4,   # tetrahedron
    5 => 8,   # hexahedron
    6 => 6,   # prism
    7 => 5,   # pyramid
)

function _read_gmsh_elements!(io::IO, vol_elems, surf_elems)
    header = split(strip(readline(io)))
    num_blocks = parse(Int, header[1])

    for _ in 1:num_blocks
        block_header = split(strip(readline(io)))
        entity_dim = parse(Int, block_header[1])
        entity_tag = parse(Int, block_header[2])
        elem_type = parse(Int, block_header[3])
        n_block = parse(Int, block_header[4])

        n_nodes = get(_GMSH_ELEM_NODES, elem_type, 0)

        for _ in 1:n_block
            parts = split(strip(readline(io)))
            # First value is element tag, rest are node tags
            node_ids = [parse(Int, parts[i + 1]) for i in 1:n_nodes]

            elem = (type = elem_type, nodes = node_ids, tag = entity_tag)

            if entity_dim == 3  # volume element
                push!(vol_elems, elem)
            elseif entity_dim == 2  # surface element (boundary face)
                push!(surf_elems, elem)
            end
            # Skip 1D (lines) and 0D (points)
        end
    end
    return readline(io)  # $EndElements
end

function _build_mesh_from_gmsh(nodes_dict, vol_elems, surf_elems, physical_names)
    # Sort node indices
    sorted_tags = sort(collect(keys(nodes_dict)))
    tag_to_idx = Dict(tag => i for (i, tag) in enumerate(sorted_tags))

    # Build Node3D vector
    nodes_vec = [nodes_dict[tag] for tag in sorted_tags]

    # Build cells from volume elements
    cells = Cell3D[]
    for elem in vol_elems
        cell_nodes = [nodes_vec[tag_to_idx[n]] for n in elem.nodes]
        # Determine cell type
        ct = if elem.type == 4
            CT_Tetrahedron
        elseif elem.type == 5
            CT_Hexahedron
        elseif elem.type == 6
            CT_Prism
        elseif elem.type == 7
            CT_Pyramid
        else
            CT_Polyhedron
        end
        # Compute center
        cx = sum(n.x for n in cell_nodes) / length(cell_nodes)
        cy = sum(n.y for n in cell_nodes) / length(cell_nodes)
        cz = sum(n.z for n in cell_nodes) / length(cell_nodes)
        # Compute volume using appropriate method
        vol = if ct == CT_Tetrahedron
            volume_tet(cell_nodes[1], cell_nodes[2], cell_nodes[3], cell_nodes[4])
        elseif ct == CT_Hexahedron
            volume_hex(cell_nodes)
        else
            # Approximate: decompose into tets from centroid
            _approx_volume(cell_nodes, cx, cy, cz)
        end

        push!(
            cells, Cell3D(
                cell_nodes, Int[], (cx, cy, cz), abs(vol), ct,
            )
        )
    end

    # Build faces from surface elements (boundary faces)
    faces = Face3D[]
    for elem in surf_elems
        face_nodes = [nodes_vec[tag_to_idx[n]] for n in elem.nodes]
        cx = sum(n.x for n in face_nodes) / length(face_nodes)
        cy = sum(n.y for n in face_nodes) / length(face_nodes)
        cz = sum(n.z for n in face_nodes) / length(face_nodes)

        # Compute normal and area
        if length(face_nodes) == 3
            p1 = [face_nodes[1].x, face_nodes[1].y, face_nodes[1].z]
            p2 = [face_nodes[2].x, face_nodes[2].y, face_nodes[2].z]
            p3 = [face_nodes[3].x, face_nodes[3].y, face_nodes[3].z]
            n_vec = cross(p2 - p1, p3 - p1)
            area = norm(n_vec) / 2
            normal = area > 0 ? n_vec / (2 * area) : [0.0, 0.0, 1.0]
        elseif length(face_nodes) == 4
            p1 = [face_nodes[1].x, face_nodes[1].y, face_nodes[1].z]
            p2 = [face_nodes[2].x, face_nodes[2].y, face_nodes[2].z]
            p3 = [face_nodes[3].x, face_nodes[3].y, face_nodes[3].z]
            p4 = [face_nodes[4].x, face_nodes[4].y, face_nodes[4].z]
            n_vec = cross(p3 - p1, p4 - p2)
            area = norm(n_vec) / 2
            normal = area > 0 ? n_vec / norm(n_vec) : [0.0, 0.0, 1.0]
        else
            area = 0.0
            normal = [0.0, 0.0, 1.0]
        end

        # Find owner cell (first cell sharing all face nodes) — simplified
        owner_idx = 0
        for (ci, cell) in enumerate(cells)
            cell_node_set = Set((n.x, n.y, n.z) for n in cell.nodes)
            all_shared = all(
                (fn.x, fn.y, fn.z) in cell_node_set for fn in face_nodes
            )
            if all_shared
                owner_idx = ci
                break
            end
        end

        tag = get(physical_names, elem.tag, Symbol("patch_$(elem.tag)"))

        push!(
            faces, Face3D(
                face_nodes,
                Tuple(normal),
                area,
                (cx, cy, cz),
                owner_idx,  # owner
                0,          # neighbour (boundary)
                true,       # is_boundary
                tag,
            )
        )
    end

    return UnstructuredMesh3D(nodes_vec, cells, faces)
end

"""Approximate volume of a general polyhedron by tet decomposition from centroid."""
function _approx_volume(nodes, cx, cy, cz)
    n = length(nodes)
    n < 4 && return 0.0
    vol = 0.0
    center = Node3D(cx, cy, cz)
    for i in 1:(n - 2)
        vol += abs(volume_tet(center, nodes[1], nodes[i + 1], nodes[i + 2]))
    end
    return vol
end
