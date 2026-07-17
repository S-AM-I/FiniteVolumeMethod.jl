# Structured Mesh Generation - Migrated from Simu.jl SimuGeometry/structured.jl
# All mesh generation functions for structured 1D/2D/3D meshes.

"""
    generate_mesh_1d(nx, L)

Generates a structured 1D mesh with `nx` cells and domain length `L`.
"""
function generate_mesh_1d(nx::Int, L::Float64)
    dx = L / nx
    nodes = [Node1D(i * dx) for i in 0:nx]

    cells = Vector{Cell1D}(undef, nx)
    for i in 1:nx
        cell_nodes = [nodes[i], nodes[i + 1]]
        center = (nodes[i].x + nodes[i + 1].x) / 2.0
        volume = dx
        cells[i] = Cell1D(cell_nodes, center, volume)
    end

    faces = Vector{Face1D}(undef, nx + 1)
    for i in 1:(nx + 1)
        face_nodes = [nodes[i]]
        normal = 1.0
        area = 1.0
        faces[i] = Face1D(face_nodes, normal, area)
    end

    return Mesh1D(nodes, cells, faces)
end

"""
    generate_mesh_1d_nonuniform(nodes::Vector{Float64})

Generates a 1D mesh from user-provided node locations.
"""
function generate_mesh_1d_nonuniform(nodes::Vector{Float64})
    length(nodes) >= 2 || throw(ArgumentError("Need at least 2 nodes"))
    for i in 2:length(nodes)
        nodes[i] > nodes[i - 1] || throw(ArgumentError("Nodes must be monotonically increasing"))
    end

    nx = length(nodes) - 1
    node_objs = [Node1D(x) for x in nodes]

    cells = Vector{Cell1D}(undef, nx)
    for i in 1:nx
        cell_nodes = [node_objs[i], node_objs[i + 1]]
        center = (nodes[i] + nodes[i + 1]) / 2.0
        volume = nodes[i + 1] - nodes[i]
        cells[i] = Cell1D(cell_nodes, center, volume)
    end

    faces = Vector{Face1D}(undef, nx + 1)
    for i in 1:(nx + 1)
        face_nodes = [node_objs[i]]
        normal = 1.0
        area = 1.0
        faces[i] = Face1D(face_nodes, normal, area)
    end

    return Mesh1D(node_objs, cells, faces)
end

"""
    generate_mesh_1d_nonuniform(nx, L, ratio)

Generates a 1D mesh with geometric progression.
"""
function generate_mesh_1d_nonuniform(nx::Int, L::Float64, ratio::Float64)
    ratio > 0 || throw(ArgumentError("ratio must be positive"))
    nx > 0 || throw(ArgumentError("nx must be positive"))

    if ratio ≈ 1.0
        return generate_mesh_1d(nx, L)
    end

    if ratio ≈ 1.0
        dx1 = L / nx
    else
        dx1 = L * (1 - ratio) / (1 - ratio^nx)
    end

    nodes = Vector{Float64}(undef, nx + 1)
    nodes[1] = 0.0
    for i in 2:(nx + 1)
        nodes[i] = nodes[i - 1] + dx1 * ratio^(i - 2)
    end

    scale = L / nodes[end]
    nodes .*= scale

    return generate_mesh_1d_nonuniform(nodes)
end

"""
    generate_mesh_1d_nonuniform(nx, L, stretch_func::Function)

Generates a 1D mesh using a stretching function.
"""
function generate_mesh_1d_nonuniform(nx::Int, L::Float64, stretch_func::Function)
    nx > 0 || throw(ArgumentError("nx must be positive"))
    xi_uniform = [i / nx for i in 0:nx]
    nodes = [stretch_func(xi) * L for xi in xi_uniform]
    nodes[1] = 0.0
    nodes[end] = L
    return generate_mesh_1d_nonuniform(nodes)
end

"""
    generate_mesh_2d(nx, ny, Lx, Ly)

Generates a structured 2D mesh.
"""
function generate_mesh_2d(nx::Int, ny::Int, Lx::Float64, Ly::Float64)
    dx = Lx / nx
    dy = Ly / ny
    nodes = [Node2D(i * dx, j * dy) for i in 0:nx, j in 0:ny]

    cells = Vector{Cell2D}(undef, nx * ny)
    for i in 1:nx
        for j in 1:ny
            cell_nodes = [nodes[i, j], nodes[i + 1, j], nodes[i + 1, j + 1], nodes[i, j + 1]]
            center = [(nodes[i, j].x + nodes[i + 1, j].x) / 2, (nodes[i, j].y + nodes[i, j + 1].y) / 2]
            volume = dx * dy
            cells[(i - 1) * ny + j] = Cell2D(cell_nodes, center, volume)
        end
    end

    v_faces = Vector{Face2D}(undef, (nx + 1) * ny)
    for i in 1:(nx + 1)
        for j in 1:ny
            face_nodes = [nodes[i, j], nodes[i, j + 1]]
            normal = [1.0, 0.0]
            area = dy
            v_faces[(i - 1) * ny + j] = Face2D(face_nodes, normal, area)
        end
    end

    h_faces = Vector{Face2D}(undef, nx * (ny + 1))
    for i in 1:nx
        for j in 1:(ny + 1)
            face_nodes = [nodes[i, j], nodes[i + 1, j]]
            normal = [0.0, 1.0]
            area = dx
            h_faces[(i - 1) * (ny + 1) + j] = Face2D(face_nodes, normal, area)
        end
    end

    faces = [v_faces; h_faces]
    nodes_vec = vec(nodes)
    return Mesh2D(nodes_vec, cells, faces, nx, ny, Lx, Ly)
end

"""
    generate_mesh_2d_nonuniform(nx, ny, Lx, Ly, x_nodes, y_nodes)

Generates a 2D mesh from user-provided node arrays.
"""
function generate_mesh_2d_nonuniform(nx::Int, ny::Int, Lx::Float64, Ly::Float64, x_nodes::Vector{Float64}, y_nodes::Vector{Float64})
    length(x_nodes) == nx + 1 || throw(ArgumentError("x_nodes must have length nx+1"))
    length(y_nodes) == ny + 1 || throw(ArgumentError("y_nodes must have length ny+1"))

    for i in 2:length(x_nodes)
        x_nodes[i] > x_nodes[i - 1] || throw(ArgumentError("x_nodes must be monotonic"))
    end
    for i in 2:length(y_nodes)
        y_nodes[i] > y_nodes[i - 1] || throw(ArgumentError("y_nodes must be monotonic"))
    end

    nodes = [Node2D(x_nodes[i + 1], y_nodes[j + 1]) for i in 0:nx, j in 0:ny]

    cells = Vector{Cell2D}(undef, nx * ny)
    for i in 1:nx
        for j in 1:ny
            cell_nodes = [nodes[i, j], nodes[i + 1, j], nodes[i + 1, j + 1], nodes[i, j + 1]]
            center = [(x_nodes[i] + x_nodes[i + 1]) / 2, (y_nodes[j] + y_nodes[j + 1]) / 2]
            dx = x_nodes[i + 1] - x_nodes[i]
            dy = y_nodes[j + 1] - y_nodes[j]
            volume = dx * dy
            cells[(i - 1) * ny + j] = Cell2D(cell_nodes, center, volume)
        end
    end

    v_faces = Vector{Face2D}(undef, (nx + 1) * ny)
    for i in 1:(nx + 1)
        for j in 1:ny
            face_nodes = [nodes[i, j], nodes[i, j + 1]]
            normal = [1.0, 0.0]
            dy = y_nodes[j + 1] - y_nodes[j]
            area = dy
            v_faces[(i - 1) * ny + j] = Face2D(face_nodes, normal, area)
        end
    end

    h_faces = Vector{Face2D}(undef, nx * (ny + 1))
    for i in 1:nx
        for j in 1:(ny + 1)
            face_nodes = [nodes[i, j], nodes[i + 1, j]]
            normal = [0.0, 1.0]
            dx = x_nodes[i + 1] - x_nodes[i]
            area = dx
            h_faces[(i - 1) * (ny + 1) + j] = Face2D(face_nodes, normal, area)
        end
    end

    faces = [v_faces; h_faces]
    nodes_vec = vec(nodes)
    return Mesh2D(nodes_vec, cells, faces, nx, ny, Lx, Ly)
end

"""
    generate_mesh_2d_nonuniform(nx, ny, Lx, Ly, x_ratio, y_ratio)

Generates a 2D mesh with geometric progression.
"""
function generate_mesh_2d_nonuniform(nx::Int, ny::Int, Lx::Float64, Ly::Float64, x_ratio::Float64, y_ratio::Float64)
    x_nodes = generate_mesh_1d_nonuniform(nx, Lx, x_ratio).nodes
    x_coords = [n.x for n in x_nodes]
    y_nodes = generate_mesh_1d_nonuniform(ny, Ly, y_ratio).nodes
    y_coords = [n.x for n in y_nodes]
    return generate_mesh_2d_nonuniform(nx, ny, Lx, Ly, x_coords, y_coords)
end

"""
    generate_mesh_2d_nonuniform(nx, ny, Lx, Ly, x_stretch, y_stretch)

Generates a 2D mesh using stretching functions.
"""
function generate_mesh_2d_nonuniform(nx::Int, ny::Int, Lx::Float64, Ly::Float64, x_stretch::Function, y_stretch::Function)
    xi_uniform = [i / nx for i in 0:nx]
    x_coords = [x_stretch(xi) * Lx for xi in xi_uniform]
    x_coords[1] = 0.0; x_coords[end] = Lx
    yi_uniform = [j / ny for j in 0:ny]
    y_coords = [y_stretch(yi) * Ly for yi in yi_uniform]
    y_coords[1] = 0.0; y_coords[end] = Ly
    return generate_mesh_2d_nonuniform(nx, ny, Lx, Ly, x_coords, y_coords)
end

"""
    generate_mesh_3d(nx, ny, nz, Lx, Ly, Lz)

Generates a structured 3D mesh.
"""
function generate_mesh_3d(nx::Int, ny::Int, nz::Int, Lx::Float64, Ly::Float64, Lz::Float64)
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    nodes = [Node3D(i * dx, j * dy, k * dz) for i in 0:nx, j in 0:ny, k in 0:nz]

    cells = Vector{Cell3D}(undef, nx * ny * nz)
    for i in 1:nx, j in 1:ny, k in 1:nz
        cell_nodes = [
            nodes[i, j, k], nodes[i + 1, j, k], nodes[i + 1, j + 1, k], nodes[i, j + 1, k],
            nodes[i, j, k + 1], nodes[i + 1, j, k + 1], nodes[i + 1, j + 1, k + 1], nodes[i, j + 1, k + 1],
        ]
        center = [
            (nodes[i, j, k].x + nodes[i + 1, j, k].x) / 2,
            (nodes[i, j, k].y + nodes[i, j + 1, k].y) / 2,
            (nodes[i, j, k].z + nodes[i, j, k + 1].z) / 2,
        ]
        volume = dx * dy * dz
        cells[(i - 1) * ny * nz + (j - 1) * nz + k] = Cell3D(cell_nodes, center, volume)
    end

    x_faces = Vector{Face3D}(undef, (nx + 1) * ny * nz)
    for i in 1:(nx + 1), j in 1:ny, k in 1:nz
        face_nodes = [nodes[i, j, k], nodes[i, j + 1, k], nodes[i, j + 1, k + 1], nodes[i, j, k + 1]]
        normal = [1.0, 0.0, 0.0]; area = dy * dz
        x_faces[(i - 1) * ny * nz + (j - 1) * nz + k] = Face3D(face_nodes, normal, area)
    end

    y_faces = Vector{Face3D}(undef, nx * (ny + 1) * nz)
    for i in 1:nx, j in 1:(ny + 1), k in 1:nz
        face_nodes = [nodes[i, j, k], nodes[i + 1, j, k], nodes[i + 1, j, k + 1], nodes[i, j, k + 1]]
        normal = [0.0, 1.0, 0.0]; area = dx * dz
        y_faces[(i - 1) * (ny + 1) * nz + (j - 1) * nz + k] = Face3D(face_nodes, normal, area)
    end

    z_faces = Vector{Face3D}(undef, nx * ny * (nz + 1))
    for i in 1:nx, j in 1:ny, k in 1:(nz + 1)
        face_nodes = [nodes[i, j, k], nodes[i + 1, j, k], nodes[i + 1, j + 1, k], nodes[i, j, k + 1]]
        normal = [0.0, 0.0, 1.0]; area = dx * dy
        z_faces[(i - 1) * ny * (nz + 1) + (j - 1) * (nz + 1) + k] = Face3D(face_nodes, normal, area)
    end

    faces = [x_faces; y_faces; z_faces]
    return Mesh3D(vec(nodes), cells, faces, nx, ny, nz, Lx, Ly, Lz)
end

"""
    generate_mesh_3d_nonuniform(nx, ny, nz, Lx, Ly, Lz, x_nodes, y_nodes, z_nodes)

Generates a 3D mesh from user-provided node arrays.
"""
function generate_mesh_3d_nonuniform(nx::Int, ny::Int, nz::Int, Lx::Float64, Ly::Float64, Lz::Float64, x_nodes::Vector{Float64}, y_nodes::Vector{Float64}, z_nodes::Vector{Float64})
    length(x_nodes) == nx + 1 && length(y_nodes) == ny + 1 && length(z_nodes) == nz + 1 || throw(ArgumentError("Node array length mismatch"))

    nodes = [Node3D(x_nodes[i + 1], y_nodes[j + 1], z_nodes[k + 1]) for i in 0:nx, j in 0:ny, k in 0:nz]

    cells = Vector{Cell3D}(undef, nx * ny * nz)
    for i in 1:nx, j in 1:ny, k in 1:nz
        cell_nodes = [
            nodes[i, j, k], nodes[i + 1, j, k], nodes[i + 1, j + 1, k], nodes[i, j + 1, k],
            nodes[i, j, k + 1], nodes[i + 1, j, k + 1], nodes[i + 1, j + 1, k + 1], nodes[i, j + 1, k + 1],
        ]
        center = [(x_nodes[i] + x_nodes[i + 1]) / 2, (y_nodes[j] + y_nodes[j + 1]) / 2, (z_nodes[k] + z_nodes[k + 1]) / 2]
        dx, dy, dz = x_nodes[i + 1] - x_nodes[i], y_nodes[j + 1] - y_nodes[j], z_nodes[k + 1] - z_nodes[k]
        cells[(i - 1) * ny * nz + (j - 1) * nz + k] = Cell3D(cell_nodes, center, dx * dy * dz)
    end

    x_faces = Vector{Face3D}(undef, (nx + 1) * ny * nz)
    for i in 1:(nx + 1), j in 1:ny, k in 1:nz
        face_nodes = [nodes[i, j, k], nodes[i, j + 1, k], nodes[i, j + 1, k + 1], nodes[i, j, k + 1]]
        dy, dz = y_nodes[j + 1] - y_nodes[j], z_nodes[k + 1] - z_nodes[k]
        x_faces[(i - 1) * ny * nz + (j - 1) * nz + k] = Face3D(face_nodes, [1.0, 0.0, 0.0], dy * dz)
    end

    y_faces = Vector{Face3D}(undef, nx * (ny + 1) * nz)
    for i in 1:nx, j in 1:(ny + 1), k in 1:nz
        face_nodes = [nodes[i, j, k], nodes[i + 1, j, k], nodes[i + 1, j, k + 1], nodes[i, j, k + 1]]
        dx, dz = x_nodes[i + 1] - x_nodes[i], z_nodes[k + 1] - z_nodes[k]
        y_faces[(i - 1) * (ny + 1) * nz + (j - 1) * nz + k] = Face3D(face_nodes, [0.0, 1.0, 0.0], dx * dz)
    end

    z_faces = Vector{Face3D}(undef, nx * ny * (nz + 1))
    for i in 1:nx, j in 1:ny, k in 1:(nz + 1)
        face_nodes = [nodes[i, j, k], nodes[i + 1, j, k], nodes[i + 1, j + 1, k], nodes[i, j, k + 1]]
        dx, dy = x_nodes[i + 1] - x_nodes[i], y_nodes[j + 1] - y_nodes[j]
        z_faces[(i - 1) * ny * (nz + 1) + (j - 1) * (nz + 1) + k] = Face3D(face_nodes, [0.0, 0.0, 1.0], dx * dy)
    end

    faces = [x_faces; y_faces; z_faces]
    return Mesh3D(vec(nodes), cells, faces, nx, ny, nz, Lx, Ly, Lz)
end

"""
    generate_mesh_3d_nonuniform(nx, ny, nz, Lx, Ly, Lz, x_ratio, y_ratio, z_ratio)

Generates a 3D mesh with geometric progression.
"""
function generate_mesh_3d_nonuniform(nx::Int, ny::Int, nz::Int, Lx::Float64, Ly::Float64, Lz::Float64, x_ratio::Float64, y_ratio::Float64, z_ratio::Float64)
    x_nodes = [n.x for n in generate_mesh_1d_nonuniform(nx, Lx, x_ratio).nodes]
    y_nodes = [n.x for n in generate_mesh_1d_nonuniform(ny, Ly, y_ratio).nodes]
    z_nodes = [n.x for n in generate_mesh_1d_nonuniform(nz, Lz, z_ratio).nodes]
    return generate_mesh_3d_nonuniform(nx, ny, nz, Lx, Ly, Lz, x_nodes, y_nodes, z_nodes)
end

"""
    generate_mesh_3d_nonuniform(nx, ny, nz, Lx, Ly, Lz, x_stretch, y_stretch, z_stretch)

Generates a 3D mesh using stretching functions.
"""
function generate_mesh_3d_nonuniform(nx::Int, ny::Int, nz::Int, Lx::Float64, Ly::Float64, Lz::Float64, x_stretch::Function, y_stretch::Function, z_stretch::Function)
    x_coords = [x_stretch(i / nx) * Lx for i in 0:nx]; x_coords[1] = 0.0; x_coords[end] = Lx
    y_coords = [y_stretch(j / ny) * Ly for j in 0:ny]; y_coords[1] = 0.0; y_coords[end] = Ly
    z_coords = [z_stretch(k / nz) * Lz for k in 0:nz]; z_coords[1] = 0.0; z_coords[end] = Lz
    return generate_mesh_3d_nonuniform(nx, ny, nz, Lx, Ly, Lz, x_coords, y_coords, z_coords)
end
