# ============================================================
# 3D Structured Cartesian Mesh
# ============================================================

"""
    StructuredMesh3D{FT} <: AbstractMesh{3}

A uniform 3D structured Cartesian mesh on `[xmin, xmax] x [ymin, ymax] x [zmin, zmax]`.

Cells are numbered in column-major order:
  cell `(i, j, k)` has global index `(k-1)*nx*ny + (j-1)*nx + i`.

# Fields
- `xmin::FT`, `xmax::FT`, `ymin::FT`, `ymax::FT`, `zmin::FT`, `zmax::FT`: Domain extents.
- `nx::Int`, `ny::Int`, `nz::Int`: Number of cells in each direction.
- `dx::FT`, `dy::FT`, `dz::FT`: Cell sizes.
"""
struct StructuredMesh3D{FT} <: AbstractMesh{3}
    xmin::FT
    xmax::FT
    ymin::FT
    ymax::FT
    zmin::FT
    zmax::FT
    nx::Int
    ny::Int
    nz::Int
    dx::FT
    dy::FT
    dz::FT
end

function StructuredMesh3D(xmin::FT, xmax::FT, ymin::FT, ymax::FT,
        zmin::FT, zmax::FT, nx::Int, ny::Int, nz::Int) where {FT}
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz
    return StructuredMesh3D(xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz, dx, dy, dz)
end

ncells(mesh::StructuredMesh3D) = mesh.nx * mesh.ny * mesh.nz

"""
    cell_ijk(mesh::StructuredMesh3D, idx::Int) -> (i, j, k)

Convert global cell index to (i, j, k) indices (column-major).
"""
@inline function cell_ijk(mesh::StructuredMesh3D, idx::Int)
    nxy = mesh.nx * mesh.ny
    k, rem_k = divrem(idx - 1, nxy)
    j, rem_j = divrem(rem_k, mesh.nx)
    return rem_j + 1, j + 1, k + 1
end

"""
    cell_idx_3d(mesh::StructuredMesh3D, i::Int, j::Int, k::Int) -> Int

Convert (i, j, k) indices to global cell index (column-major).
"""
@inline function cell_idx_3d(mesh::StructuredMesh3D, i::Int, j::Int, k::Int)
    return (k - 1) * mesh.nx * mesh.ny + (j - 1) * mesh.nx + i
end

function cell_center(mesh::StructuredMesh3D, idx::Int)
    i, j, k = cell_ijk(mesh, idx)
    x = mesh.xmin + (i - 0.5) * mesh.dx
    y = mesh.ymin + (j - 0.5) * mesh.dy
    z = mesh.zmin + (k - 0.5) * mesh.dz
    return (x, y, z)
end

cell_volume(mesh::StructuredMesh3D, ::Int) = mesh.dx * mesh.dy * mesh.dz

# Face numbering for 3D:
# x-faces: f = 1...(nx-1)*ny*nz
# y-faces: f = (nx-1)*ny*nz + 1 ... + nx*(ny-1)*nz
# z-faces: f = ... + nx*ny*(nz-1)
num_x_faces(mesh::StructuredMesh3D) = (mesh.nx - 1) * mesh.ny * mesh.nz
num_y_faces(mesh::StructuredMesh3D) = mesh.nx * (mesh.ny - 1) * mesh.nz
num_z_faces(mesh::StructuredMesh3D) = mesh.nx * mesh.ny * (mesh.nz - 1)
nfaces(mesh::StructuredMesh3D) = num_x_faces(mesh) + num_y_faces(mesh) + num_z_faces(mesh)

function face_area(mesh::StructuredMesh3D, f::Int)
    nxf = num_x_faces(mesh)
    nyf = num_y_faces(mesh)
    if f <= nxf
        return mesh.dy * mesh.dz
    elseif f <= nxf + nyf
        return mesh.dx * mesh.dz
    else
        return mesh.dx * mesh.dy
    end
end
