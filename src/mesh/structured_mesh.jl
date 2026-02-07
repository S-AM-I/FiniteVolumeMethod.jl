"""
    StructuredMesh1D{FT} <: AbstractMesh{1}

A uniform 1D structured mesh on the interval `[xmin, xmax]` with `ncells` cells.

# Fields
- `xmin::FT`: Left boundary coordinate.
- `xmax::FT`: Right boundary coordinate.
- `num_cells::Int`: Number of cells.
- `dx::FT`: Cell width `(xmax - xmin) / num_cells`.
- `cell_centers::Vector{FT}`: Coordinates of cell centers.
"""
struct StructuredMesh1D{FT} <: AbstractMesh{1}
    xmin::FT
    xmax::FT
    num_cells::Int
    dx::FT
    cell_centers::Vector{FT}
end

function StructuredMesh1D(xmin::FT, xmax::FT, num_cells::Int) where {FT}
    dx = (xmax - xmin) / num_cells
    cell_centers = [xmin + (i - FT(0.5)) * dx for i in 1:num_cells]
    return StructuredMesh1D(xmin, xmax, num_cells, dx, cell_centers)
end

ncells(mesh::StructuredMesh1D) = mesh.num_cells
nfaces(mesh::StructuredMesh1D) = mesh.num_cells - 1
cell_center(mesh::StructuredMesh1D, i::Int) = mesh.cell_centers[i]
cell_volume(mesh::StructuredMesh1D, ::Int) = mesh.dx
face_area(::StructuredMesh1D{FT}, ::Int) where {FT} = one(FT)
face_owner(::StructuredMesh1D, f::Int) = f
face_neighbor(::StructuredMesh1D, f::Int) = f + 1

"""
    face_position(mesh::StructuredMesh1D, f::Int)

Return the coordinate of internal face `f` (the interface between cell `f` and cell `f+1`).
"""
face_position(mesh::StructuredMesh1D, f::Int) = mesh.xmin + f * mesh.dx

"""
    StructuredMesh2D{FT} <: AbstractMesh{2}

A uniform 2D structured Cartesian mesh on `[xmin, xmax] x [ymin, ymax]`.

Cells are numbered in column-major order: cell `(i, j)` has global index `(j-1)*nx + i`.
Faces are split into x-faces (vertical, normal in x-direction) and y-faces (horizontal, normal in y-direction).

# Fields
- `xmin::FT`, `xmax::FT`, `ymin::FT`, `ymax::FT`: Domain extents.
- `nx::Int`, `ny::Int`: Number of cells in each direction.
- `dx::FT`, `dy::FT`: Cell sizes.
"""
struct StructuredMesh2D{FT} <: AbstractMesh{2}
    xmin::FT
    xmax::FT
    ymin::FT
    ymax::FT
    nx::Int
    ny::Int
    dx::FT
    dy::FT
end

function StructuredMesh2D(xmin::FT, xmax::FT, ymin::FT, ymax::FT, nx::Int, ny::Int) where {FT}
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    return StructuredMesh2D(xmin, xmax, ymin, ymax, nx, ny, dx, dy)
end

ncells(mesh::StructuredMesh2D) = mesh.nx * mesh.ny

"""
    cell_ij(mesh::StructuredMesh2D, idx::Int) -> (i, j)

Convert global cell index to (i, j) indices (column-major).
"""
@inline function cell_ij(mesh::StructuredMesh2D, idx::Int)
    j, rem = divrem(idx - 1, mesh.nx)
    return rem + 1, j + 1
end

"""
    cell_idx(mesh::StructuredMesh2D, i::Int, j::Int) -> Int

Convert (i, j) indices to global cell index (column-major).
"""
@inline cell_idx(mesh::StructuredMesh2D, i::Int, j::Int) = (j - 1) * mesh.nx + i

function cell_center(mesh::StructuredMesh2D, idx::Int)
    i, j = cell_ij(mesh, idx)
    x = mesh.xmin + (i - 0.5) * mesh.dx
    y = mesh.ymin + (j - 0.5) * mesh.dy
    return (x, y)
end

cell_volume(mesh::StructuredMesh2D, ::Int) = mesh.dx * mesh.dy

# Face numbering for 2D:
# x-faces: f = 1...(nx-1)*ny  (vertical faces, normal in x-direction)
# y-faces: f = (nx-1)*ny + 1 ... (nx-1)*ny + nx*(ny-1)  (horizontal faces, normal in y-direction)
num_x_faces(mesh::StructuredMesh2D) = (mesh.nx - 1) * mesh.ny
num_y_faces(mesh::StructuredMesh2D) = mesh.nx * (mesh.ny - 1)
nfaces(mesh::StructuredMesh2D) = num_x_faces(mesh) + num_y_faces(mesh)

"""
    is_x_face(mesh::StructuredMesh2D, f::Int) -> Bool

Return true if face `f` is an x-face (vertical, normal in x-direction).
"""
@inline is_x_face(mesh::StructuredMesh2D, f::Int) = f <= num_x_faces(mesh)

function face_owner(mesh::StructuredMesh2D, f::Int)
    nxf = num_x_faces(mesh)
    if f <= nxf
        # x-face: face f corresponds to the interface at x_{f_local+1/2}
        j, rem = divrem(f - 1, mesh.nx - 1)
        i = rem + 1
        return cell_idx(mesh, i, j + 1)
    else
        # y-face
        g = f - nxf
        j, rem = divrem(g - 1, mesh.nx)
        i = rem + 1
        return cell_idx(mesh, i, j + 1)
    end
end

function face_neighbor(mesh::StructuredMesh2D, f::Int)
    nxf = num_x_faces(mesh)
    if f <= nxf
        j, rem = divrem(f - 1, mesh.nx - 1)
        i = rem + 1
        return cell_idx(mesh, i + 1, j + 1)
    else
        g = f - nxf
        j, rem = divrem(g - 1, mesh.nx)
        i = rem + 1
        return cell_idx(mesh, i, j + 2)
    end
end

function face_area(mesh::StructuredMesh2D, f::Int)
    if is_x_face(mesh, f)
        return mesh.dy
    else
        return mesh.dx
    end
end

function face_normal(mesh::StructuredMesh2D, f::Int)
    if is_x_face(mesh, f)
        return (1.0, 0.0)
    else
        return (0.0, 1.0)
    end
end
