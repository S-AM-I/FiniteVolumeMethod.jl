# Curvilinear Mesh Support - Migrated from Simu.jl SimuGeometry/curvilinear.jl
# Body-fitted (curvilinear) mesh types and geometric helpers.

using LinearAlgebra: norm, cross, dot
using Statistics: mean

# ==============================================================================
# 2D Curvilinear Mesh
# ==============================================================================

"""
    CurvilinearMesh2D <: AbstractParabolicMesh

2D Body-fitted mesh defined by node coordinates `x_nodes` and `y_nodes`.
"""
struct CurvilinearMesh2D <: AbstractParabolicMesh
    # Primary grid data (nodes)
    x_nodes::Matrix{Float64} # (nx+1) x (ny+1)
    y_nodes::Matrix{Float64} # (nx+1) x (ny+1)

    # Dimensions
    nx::Int
    ny::Int

    # Precomputed geometric properties
    cells::Vector{Cell2D}
    faces::Vector{Face2D}
end

"""
    CurvilinearMesh2D(x_nodes::Matrix, y_nodes::Matrix)

Construct a 2D curvilinear mesh from node coordinates.
"""
function CurvilinearMesh2D(x_nodes::Matrix{Float64}, y_nodes::Matrix{Float64})
    nx = size(x_nodes, 1) - 1
    ny = size(x_nodes, 2) - 1

    if size(y_nodes) != (nx + 1, ny + 1)
        error("Dimension mismatch between x_nodes and y_nodes")
    end

    # 1. Compute Cell Properties
    cells = Vector{Cell2D}(undef, nx * ny)

    for j in 1:ny
        for i in 1:nx
            # Nodes for cell (i,j)
            # Counter-clockwise: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
            n1 = Node2D(x_nodes[i, j], y_nodes[i, j])
            n2 = Node2D(x_nodes[i + 1, j], y_nodes[i + 1, j])
            n3 = Node2D(x_nodes[i + 1, j + 1], y_nodes[i + 1, j + 1])
            n4 = Node2D(x_nodes[i, j + 1], y_nodes[i, j + 1])
            cell_nodes = [n1, n2, n3, n4]

            # Center (centroid of vertices)
            cx = (n1.x + n2.x + n3.x + n4.x) / 4.0
            cy = (n1.y + n2.y + n3.y + n4.y) / 4.0

            # Volume (Area) via shoelace formula
            area = 0.5 * abs(
                (n1.x * n2.y - n1.y * n2.x) +
                    (n2.x * n3.y - n2.y * n3.x) +
                    (n3.x * n4.y - n3.y * n4.x) +
                    (n4.x * n1.y - n4.y * n1.x)
            )

            idx = (i - 1) * ny + j
            cells[idx] = Cell2D(cell_nodes, [cx, cy], area)
        end
    end

    faces = Vector{Face2D}()

    return CurvilinearMesh2D(x_nodes, y_nodes, nx, ny, cells, faces)
end


# ==============================================================================
# 3D Curvilinear Mesh
# ==============================================================================

"""
    CurvilinearMesh3D <: AbstractParabolicMesh

3D Body-fitted mesh defined by node coordinates `x_nodes`, `y_nodes`, `z_nodes`.
"""
struct CurvilinearMesh3D <: AbstractParabolicMesh
    x_nodes::Array{Float64, 3} # (nx+1) x (ny+1) x (nz+1)
    y_nodes::Array{Float64, 3}
    z_nodes::Array{Float64, 3}

    nx::Int
    ny::Int
    nz::Int

    cells::Vector{Cell3D}
    faces::Vector{Face3D}
end

"""
    CurvilinearMesh3D(x_nodes::Array{Float64,3}, y_nodes::Array{Float64,3}, z_nodes::Array{Float64,3})

Construct a 3D curvilinear mesh from node coordinates.
"""
function CurvilinearMesh3D(x_nodes::Array{Float64, 3}, y_nodes::Array{Float64, 3}, z_nodes::Array{Float64, 3})
    nx = size(x_nodes, 1) - 1
    ny = size(x_nodes, 2) - 1
    nz = size(x_nodes, 3) - 1

    if size(y_nodes) != (nx + 1, ny + 1, nz + 1) || size(z_nodes) != (nx + 1, ny + 1, nz + 1)
        error("Dimension mismatch between nodes arrays")
    end

    cells = Vector{Cell3D}(undef, nx * ny * nz)

    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                # 8 nodes for hexahedron
                cell_nodes = Node3D[]
                # Bottom face (k)
                push!(cell_nodes, Node3D(x_nodes[i, j, k], y_nodes[i, j, k], z_nodes[i, j, k]))
                push!(cell_nodes, Node3D(x_nodes[i + 1, j, k], y_nodes[i + 1, j, k], z_nodes[i + 1, j, k]))
                push!(cell_nodes, Node3D(x_nodes[i + 1, j + 1, k], y_nodes[i + 1, j + 1, k], z_nodes[i + 1, j + 1, k]))
                push!(cell_nodes, Node3D(x_nodes[i, j + 1, k], y_nodes[i, j + 1, k], z_nodes[i, j + 1, k]))
                # Top face (k+1)
                push!(cell_nodes, Node3D(x_nodes[i, j, k + 1], y_nodes[i, j, k + 1], z_nodes[i, j, k + 1]))
                push!(cell_nodes, Node3D(x_nodes[i + 1, j, k + 1], y_nodes[i + 1, j, k + 1], z_nodes[i + 1, j, k + 1]))
                push!(cell_nodes, Node3D(x_nodes[i + 1, j + 1, k + 1], y_nodes[i + 1, j + 1, k + 1], z_nodes[i + 1, j + 1, k + 1]))
                push!(cell_nodes, Node3D(x_nodes[i, j + 1, k + 1], y_nodes[i, j + 1, k + 1], z_nodes[i, j + 1, k + 1]))

                # Center
                cx = mean([n.x for n in cell_nodes])
                cy = mean([n.y for n in cell_nodes])
                cz = mean([n.z for n in cell_nodes])

                center = [cx, cy, cz]

                # Volume via decomposition into tets
                vol = 0.0

                # Helper for tet volume
                function tet_vol(p1, p2, p3, p4)
                    a = [p1.x, p1.y, p1.z]
                    b = [p2.x, p2.y, p2.z]
                    c = [p3.x, p3.y, p3.z]
                    d = [p4.x, p4.y, p4.z]
                    return abs(dot(a - d, cross(b - d, c - d))) / 6.0
                end

                # Helper for quad face to centroid tets
                function add_face_contribution(n1, n2, n3, n4)
                    v = 0.0
                    v += tet_vol(n1, n2, n3, Node3D(cx, cy, cz))
                    v += tet_vol(n1, n3, n4, Node3D(cx, cy, cz))
                    return v
                end

                vol += add_face_contribution(cell_nodes[1], cell_nodes[2], cell_nodes[3], cell_nodes[4]) # Bottom
                vol += add_face_contribution(cell_nodes[5], cell_nodes[6], cell_nodes[7], cell_nodes[8]) # Top
                vol += add_face_contribution(cell_nodes[1], cell_nodes[2], cell_nodes[6], cell_nodes[5]) # Front
                vol += add_face_contribution(cell_nodes[2], cell_nodes[3], cell_nodes[7], cell_nodes[6]) # Right
                vol += add_face_contribution(cell_nodes[3], cell_nodes[4], cell_nodes[8], cell_nodes[7]) # Back
                vol += add_face_contribution(cell_nodes[4], cell_nodes[1], cell_nodes[5], cell_nodes[8]) # Left

                idx = (i - 1) * ny * nz + (j - 1) * nz + k
                cells[idx] = Cell3D(cell_nodes, center, vol)
            end
        end
    end

    faces = Vector{Face3D}()
    return CurvilinearMesh3D(x_nodes, y_nodes, z_nodes, nx, ny, nz, cells, faces)
end

# ==============================================================================
# Geometric Helpers
# ==============================================================================

"""
    get_cell_center(mesh, i, j)

Get the center coordinates of cell (i,j) in a 2D curvilinear mesh.
"""
function get_cell_center(mesh::CurvilinearMesh2D, i::Int, j::Int)
    idx = (i - 1) * mesh.ny + j
    return mesh.cells[idx].center
end

"""
    get_cell_center(mesh, i, j, k)

Get the center coordinates of cell (i,j,k) in a 3D curvilinear mesh.
"""
function get_cell_center(mesh::CurvilinearMesh3D, i::Int, j::Int, k::Int)
    idx = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
    return mesh.cells[idx].center
end

"""
    get_face_geo(mesh, i, j, side)

Calculate geometry metrics for a face of cell (i,j) in a 2D curvilinear mesh.
Returns `(center, normal, area)`.
`normal` points outward from the cell.
"""
function get_face_geo(mesh::CurvilinearMesh2D, i::Int, j::Int, side::Symbol)
    if side == :bottom # Face 1-2
        n1 = Node2D(mesh.x_nodes[i, j], mesh.y_nodes[i, j])
        n2 = Node2D(mesh.x_nodes[i + 1, j], mesh.y_nodes[i + 1, j])
        center = [(n1.x + n2.x) / 2, (n1.y + n2.y) / 2]
        dx, dy = n2.x - n1.x, n2.y - n1.y
        normal = [dy, -dx]
        area = norm(normal)
        normal ./= (area + 1.0e-100)
        return center, normal, area

    elseif side == :right # Face 2-3
        n2 = Node2D(mesh.x_nodes[i + 1, j], mesh.y_nodes[i + 1, j])
        n3 = Node2D(mesh.x_nodes[i + 1, j + 1], mesh.y_nodes[i + 1, j + 1])
        center = [(n2.x + n3.x) / 2, (n2.y + n3.y) / 2]
        dx, dy = n3.x - n2.x, n3.y - n2.y
        normal = [dy, -dx]
        area = norm(normal)
        normal ./= (area + 1.0e-100)
        return center, normal, area

    elseif side == :top # Face 3-4
        n3 = Node2D(mesh.x_nodes[i + 1, j + 1], mesh.y_nodes[i + 1, j + 1])
        n4 = Node2D(mesh.x_nodes[i, j + 1], mesh.y_nodes[i, j + 1])
        center = [(n3.x + n4.x) / 2, (n3.y + n4.y) / 2]
        dx, dy = n4.x - n3.x, n4.y - n3.y
        normal = [dy, -dx]
        area = norm(normal)
        normal ./= (area + 1.0e-100)
        return center, normal, area

    elseif side == :left # Face 4-1
        n4 = Node2D(mesh.x_nodes[i, j + 1], mesh.y_nodes[i, j + 1])
        n1 = Node2D(mesh.x_nodes[i, j], mesh.y_nodes[i, j])
        center = [(n4.x + n1.x) / 2, (n4.y + n1.y) / 2]
        dx, dy = n1.x - n4.x, n1.y - n4.y
        normal = [dy, -dx]
        area = norm(normal)
        normal ./= (area + 1.0e-100)
        return center, normal, area

    else
        error("Invalid side: $side")
    end
end

"""
    get_face_geo(mesh, i, j, k, side)

Calculate geometry metrics for a face of cell (i,j,k) in a 3D curvilinear mesh.
Returns `(center, normal, area)`.
"""
function get_face_geo(mesh::CurvilinearMesh3D, i::Int, j::Int, k::Int, side::Symbol)
    # Indices helper
    function get_node(ni, nj, nk)
        return [mesh.x_nodes[ni, nj, nk], mesh.y_nodes[ni, nj, nk], mesh.z_nodes[ni, nj, nk]]
    end

    p1, p2, p3, p4 = [], [], [], []

    if side == :bottom
        p1 = get_node(i, j, k)
        p2 = get_node(i + 1, j, k)
        p3 = get_node(i + 1, j + 1, k)
        p4 = get_node(i, j + 1, k)
    elseif side == :top
        p1 = get_node(i, j, k + 1)
        p2 = get_node(i + 1, j, k + 1)
        p3 = get_node(i + 1, j + 1, k + 1)
        p4 = get_node(i, j + 1, k + 1)
    elseif side == :left
        p1 = get_node(i, j, k)
        p2 = get_node(i, j + 1, k)
        p3 = get_node(i, j + 1, k + 1)
        p4 = get_node(i, j, k + 1)
    elseif side == :right
        p1 = get_node(i + 1, j, k)
        p2 = get_node(i + 1, j + 1, k)
        p3 = get_node(i + 1, j + 1, k + 1)
        p4 = get_node(i + 1, j, k + 1)
    elseif side == :front
        p1 = get_node(i, j, k)
        p2 = get_node(i + 1, j, k)
        p3 = get_node(i + 1, j, k + 1)
        p4 = get_node(i, j, k + 1)
    elseif side == :back
        p1 = get_node(i, j + 1, k)
        p2 = get_node(i + 1, j + 1, k)
        p3 = get_node(i + 1, j + 1, k + 1)
        p4 = get_node(i, j + 1, k + 1)
    else
        error("Invalid side: $side")
    end

    # Centroid
    center = (p1 + p2 + p3 + p4) / 4.0

    # Normal and Area using cross product of diagonals
    d1 = p3 - p1
    d2 = p4 - p2
    c = cross(d1, d2)
    area = 0.5 * norm(c)
    normal = c / (norm(c) + 1.0e-100)

    # Orient normal outward
    cell_c = get_cell_center(mesh, i, j, k)
    vec_out = center - cell_c
    if dot(vec_out, normal) < 0
        normal = -normal
    end

    return center, normal, area
end
