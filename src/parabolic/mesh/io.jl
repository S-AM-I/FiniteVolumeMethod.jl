# Mesh I/O - Migrated from Simu.jl SimuGeometry/mesh_io.jl
# Gmsh reader, VTK writer, and related utilities.

using Printf
using LinearAlgebra: norm, cross, dot

# --- Gmsh Reader (v2.2) ---

"""
    read_gmsh(filename::String)

Read a Gmsh (format 2.2) file and return an UnstructuredMesh3D.
"""
function read_gmsh(filename::String)
    nodes = Node3D[]
    cells = UnstructuredCell3D[]

    # Mappings
    node_id_map = Dict{Int, Int}() # Gmsh ID -> Index in nodes vector

    open(filename, "r") do io
        state = :start
        num_nodes = 0
        num_elements = 0

        for line in eachline(io)
            line = strip(line)
            if isempty(line)
                continue
            end

            if startswith(line, "\$Nodes")
                state = :nodes_count
                continue
            elseif startswith(line, "\$EndNodes")
                state = :start
                continue
            elseif startswith(line, "\$Elements")
                state = :elements_count
                continue
            elseif startswith(line, "\$EndElements")
                state = :start
                continue
            end

            if state == :nodes_count
                num_nodes = parse(Int, line)
                state = :reading_nodes
            elseif state == :reading_nodes
                parts = split(line)
                id = parse(Int, parts[1])
                x = parse(Float64, parts[2])
                y = parse(Float64, parts[3])
                z = parse(Float64, parts[4])
                push!(nodes, Node3D(x, y, z))
                node_id_map[id] = length(nodes)
            elseif state == :elements_count
                num_elements = parse(Int, line)
                state = :reading_elements
            elseif state == :reading_elements
                parts = split(line)
                id = parse(Int, parts[1])
                elem_type = parse(Int, parts[2])
                num_tags = parse(Int, parts[3])
                node_start_idx = 4 + num_tags

                cell_nodes_ids = [parse(Int, parts[i]) for i in node_start_idx:length(parts)]
                mapped_nodes = [nodes[node_id_map[nid]] for nid in cell_nodes_ids]

                if elem_type == 4 # Tet
                    center = sum([[n.x, n.y, n.z] for n in mapped_nodes]) / 4.0
                    vol = volume_tet(mapped_nodes)
                    push!(cells, UnstructuredCell3D(mapped_nodes, center, vol, Int[], CT_Tetrahedron))
                elseif elem_type == 5 # Hex
                    center = sum([[n.x, n.y, n.z] for n in mapped_nodes]) / 8.0
                    vol = volume_hex(mapped_nodes)
                    push!(cells, UnstructuredCell3D(mapped_nodes, center, vol, Int[], CT_Hexahedron))
                elseif elem_type == 6 # Prism
                    center = sum([[n.x, n.y, n.z] for n in mapped_nodes]) / 6.0
                    vol = volume_prism(mapped_nodes)
                    push!(cells, UnstructuredCell3D(mapped_nodes, center, vol, Int[], CT_Prism))
                elseif elem_type == 7 # Pyramid
                    center = sum([[n.x, n.y, n.z] for n in mapped_nodes]) / 5.0
                    vol = volume_pyramid(mapped_nodes)
                    push!(cells, UnstructuredCell3D(mapped_nodes, center, vol, Int[], CT_Pyramid))
                end
            end
        end
    end

    # Build Connectivity (Faces)
    faces = build_faces_from_cells(nodes, cells)

    return UnstructuredMesh3D(nodes, cells, faces)
end

"""Compute the volume of a tetrahedron from its four `Node3D` vertices."""
function volume_tet(nodes::Vector{Node3D})
    # V = |(a-d) . ((b-d) x (c-d))| / 6
    a, b, c, d = nodes[1], nodes[2], nodes[3], nodes[4]
    v1 = [a.x - d.x, a.y - d.y, a.z - d.z]
    v2 = [b.x - d.x, b.y - d.y, b.z - d.z]
    v3 = [c.x - d.x, c.y - d.y, c.z - d.z]
    return abs(dot(v1, cross(v2, v3))) / 6.0
end

"""Compute the volume of a hexahedron from its 8 `Node3D` vertices via 5-tet decomposition."""
function volume_hex(nodes::Vector{Node3D})
    length(nodes) == 8 || error("Hexahedron requires exactly 8 nodes, got $(length(nodes))")
    n = nodes
    V = volume_tet([n[1], n[2], n[4], n[5]]) +
        volume_tet([n[2], n[3], n[4], n[7]]) +
        volume_tet([n[2], n[5], n[6], n[7]]) +
        volume_tet([n[4], n[5], n[7], n[8]]) +
        volume_tet([n[2], n[4], n[5], n[7]])
    return V
end

"""
    build_faces_from_cells(nodes, cells)

Reconstruct face connectivity from a list of cells.
"""
function build_faces_from_cells(nodes, cells)
    face_map = Dict{Vector{Int}, Vector{Int}}()

    node_to_idx = Dict{Node3D, Int}()
    for (i, n) in enumerate(nodes)
        node_to_idx[n] = i
    end

    for (c_idx, cell) in enumerate(cells)
        cell_faces = get_cell_faces(cell, node_to_idx)

        for face_nodes_indices in cell_faces
            sorted_indices = sort(face_nodes_indices)

            if !haskey(face_map, sorted_indices)
                face_map[sorted_indices] = [c_idx, 0] # Owner, Neighbor=0
            else
                # Neighbor found
                face_map[sorted_indices][2] = c_idx
            end
        end
    end

    # Convert map to UnstructuredFace3D vector
    u_faces = UnstructuredFace3D[]

    for (indices, owners) in face_map
        owner = owners[1]
        neighbor = owners[2]

        face_nodes = [nodes[i] for i in indices]

        # Compute Normal, Area, Center
        p1 = face_nodes[1]
        p2 = face_nodes[2]
        p3 = face_nodes[3]

        v1 = [p2.x - p1.x, p2.y - p1.y, p2.z - p1.z]
        v2 = [p3.x - p1.x, p3.y - p1.y, p3.z - p1.z]
        cp = cross(v1, v2)
        area = norm(cp) * 0.5 # Triangle area

        if length(face_nodes) == 4
            area *= 2.0 # Assume rectangular quad
        end

        normal = cp / (norm(cp) + 1.0e-16)

        center = sum([[n.x, n.y, n.z] for n in face_nodes]) / length(face_nodes)

        # Orient normal outwards from Owner
        c_own = cells[owner]
        d_CO = [center[1] - c_own.center[1], center[2] - c_own.center[2], center[3] - c_own.center[3]]
        if dot(normal, d_CO) < 0
            normal = -normal
        end

        push!(u_faces, UnstructuredFace3D(face_nodes, normal, area, center, owner, neighbor))

        # Update cell connectivity
        f_idx = length(u_faces)
        push!(cells[owner].faces, f_idx)
        if neighbor > 0
            push!(cells[neighbor].faces, f_idx)
        end
    end

    return u_faces
end

"""Extract the face connectivity for an `UnstructuredCell3D`, returning sorted vertex-index tuples."""
function get_cell_faces(cell::UnstructuredCell3D, node_to_idx)
    ids = [node_to_idx[n] for n in cell.nodes]
    faces = Vector{Vector{Int}}()

    if cell.type == CT_Tetrahedron
        push!(faces, [ids[1], ids[3], ids[2]])
        push!(faces, [ids[1], ids[2], ids[4]])
        push!(faces, [ids[2], ids[3], ids[4]])
        push!(faces, [ids[3], ids[1], ids[4]])

    elseif cell.type == CT_Hexahedron
        push!(faces, [ids[1], ids[4], ids[3], ids[2]]) # Bottom
        push!(faces, [ids[1], ids[2], ids[6], ids[5]]) # Front
        push!(faces, [ids[2], ids[3], ids[7], ids[6]]) # Right
        push!(faces, [ids[3], ids[4], ids[8], ids[7]]) # Back
        push!(faces, [ids[4], ids[1], ids[5], ids[8]]) # Left
        push!(faces, [ids[5], ids[6], ids[7], ids[8]]) # Top

    elseif cell.type == CT_Prism
        push!(faces, [ids[1], ids[3], ids[2]]) # Bottom (Tri)
        push!(faces, [ids[4], ids[5], ids[6]]) # Top (Tri)
        push!(faces, [ids[1], ids[2], ids[5], ids[4]]) # Side 1
        push!(faces, [ids[2], ids[3], ids[6], ids[5]]) # Side 2
        push!(faces, [ids[3], ids[1], ids[4], ids[6]]) # Side 3

    elseif cell.type == CT_Pyramid
        push!(faces, [ids[1], ids[4], ids[3], ids[2]]) # Base (Quad)
        push!(faces, [ids[1], ids[2], ids[5]]) # Front (Tri)
        push!(faces, [ids[2], ids[3], ids[5]]) # Right (Tri)
        push!(faces, [ids[3], ids[4], ids[5]]) # Back (Tri)
        push!(faces, [ids[4], ids[1], ids[5]]) # Left (Tri)
    end
    return faces
end


# --- VTK Writer (Legacy) ---

"""
    write_vtk_unstructured(filename::String, mesh::UnstructuredMesh3D)

Write mesh to Legacy VTK Unstructured Grid format.
"""
function write_vtk_unstructured(filename::String, mesh::UnstructuredMesh3D)
    return open(filename, "w") do io
        println(io, "# vtk DataFile Version 2.0")
        println(io, "Simu Mesh")
        println(io, "ASCII")
        println(io, "DATASET UNSTRUCTURED_GRID")

        # POINTS
        println(io, "POINTS $(length(mesh.nodes)) float")
        for n in mesh.nodes
            @printf(io, "%f %f %f\n", n.x, n.y, n.z)
        end

        # CELLS
        total_size = 0
        for c in mesh.cells
            total_size += 1 + length(c.nodes)
        end
        println(io, "CELLS $(length(mesh.cells)) $total_size")

        # Create map for node indices (0-based for VTK)
        node_to_idx = Dict{Node3D, Int}()
        for (i, n) in enumerate(mesh.nodes)
            node_to_idx[n] = i - 1
        end

        for c in mesh.cells
            print(io, "$(length(c.nodes))")
            for n in c.nodes
                print(io, " $(node_to_idx[n])")
            end
            println(io, "")
        end

        # CELL_TYPES
        println(io, "CELL_TYPES $(length(mesh.cells))")
        for c in mesh.cells
            if c.type == CT_Tetrahedron
                println(io, "10") # VTK_TETRA
            elseif c.type == CT_Hexahedron
                println(io, "12") # VTK_HEXAHEDRON
            elseif c.type == CT_Prism
                println(io, "13") # VTK_WEDGE
            elseif c.type == CT_Pyramid
                println(io, "14") # VTK_PYRAMID
            else
                println(io, "0")
            end
        end
    end
end
