# ============================================================
# Constrained Transport Data Structures for 3D MHD
# ============================================================
#
# For 3D MHD on a structured Cartesian mesh (nx x ny x nz):
#
# Face-centered magnetic field:
#   Bx_face[1:nx+1, 1:ny, 1:nz]  -- x-component at x-faces (yz-planes)
#   By_face[1:nx, 1:ny+1, 1:nz]  -- y-component at y-faces (xz-planes)
#   Bz_face[1:nx, 1:ny, 1:nz+1]  -- z-component at z-faces (xy-planes)
#
# Cell-centered B (averaged from face values):
#   Bx_cell[i,j,k] = 0.5 * (Bx_face[i,j,k] + Bx_face[i+1,j,k])
#   By_cell[i,j,k] = 0.5 * (By_face[i,j,k] + By_face[i,j+1,k])
#   Bz_cell[i,j,k] = 0.5 * (Bz_face[i,j,k] + Bz_face[i,j,k+1])
#
# Edge-centered EMF (electromotive force):
#   emf_x[1:nx, 1:ny+1, 1:nz+1]  -- x-component of E at x-edges
#   emf_y[1:nx+1, 1:ny, 1:nz+1]  -- y-component of E at y-edges
#   emf_z[1:nx+1, 1:ny+1, 1:nz]  -- z-component of E at z-edges
#
# The CT update guarantees div(B) = 0 via Faraday's law:
#   dBx/dt = -dEz/dy + dEy/dz
#   dBy/dt = -dEx/dz + dEz/dx
#   dBz/dt = -dEy/dx + dEx/dy

"""
    CTData3D{FT}

Constrained transport data for 3D MHD on a structured mesh.

Stores face-centered magnetic field components and edge-centered EMFs.

# Fields
- `Bx_face::Array{FT,3}`: x-component of B at x-faces, size `(nx+1) x ny x nz`.
- `By_face::Array{FT,3}`: y-component of B at y-faces, size `nx x (ny+1) x nz`.
- `Bz_face::Array{FT,3}`: z-component of B at z-faces, size `nx x ny x (nz+1)`.
- `emf_x::Array{FT,3}`: x-component of EMF at x-edges, size `nx x (ny+1) x (nz+1)`.
- `emf_y::Array{FT,3}`: y-component of EMF at y-edges, size `(nx+1) x ny x (nz+1)`.
- `emf_z::Array{FT,3}`: z-component of EMF at z-edges, size `(nx+1) x (ny+1) x nz`.
"""
struct CTData3D{FT}
    Bx_face::Array{FT, 3}
    By_face::Array{FT, 3}
    Bz_face::Array{FT, 3}
    emf_x::Array{FT, 3}
    emf_y::Array{FT, 3}
    emf_z::Array{FT, 3}
end

"""
    CTData3D(nx::Int, ny::Int, nz::Int, ::Type{FT}=Float64) -> CTData3D{FT}

Create zero-initialized CT data for a `nx x ny x nz` mesh.
"""
function CTData3D(nx::Int, ny::Int, nz::Int, ::Type{FT} = Float64) where {FT}
    Bx_face = zeros(FT, nx + 1, ny, nz)
    By_face = zeros(FT, nx, ny + 1, nz)
    Bz_face = zeros(FT, nx, ny, nz + 1)
    emf_x = zeros(FT, nx, ny + 1, nz + 1)
    emf_y = zeros(FT, nx + 1, ny, nz + 1)
    emf_z = zeros(FT, nx + 1, ny + 1, nz)
    return CTData3D(Bx_face, By_face, Bz_face, emf_x, emf_y, emf_z)
end

"""
    initialize_ct_3d!(ct::CTData3D, prob::HyperbolicProblem3D, mesh)

Initialize face-centered B fields from the initial condition.
Face values are set by evaluating the initial condition at the face center
and extracting the appropriate B component.
"""
function initialize_ct_3d!(ct::CTData3D, prob, mesh)
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    law = prob.law

    # Initialize Bx at x-faces
    for k in 1:nz, j in 1:ny, i in 1:(nx + 1)
        x_face = mesh.xmin + (i - 1) * dx
        y_face = mesh.ymin + (j - 0.5) * dy
        z_face = mesh.zmin + (k - 0.5) * dz
        w = prob.initial_condition(x_face, y_face, z_face)
        ct.Bx_face[i, j, k] = w[6]
    end

    # Initialize By at y-faces
    for k in 1:nz, j in 1:(ny + 1), i in 1:nx
        x_face = mesh.xmin + (i - 0.5) * dx
        y_face = mesh.ymin + (j - 1) * dy
        z_face = mesh.zmin + (k - 0.5) * dz
        w = prob.initial_condition(x_face, y_face, z_face)
        ct.By_face[i, j, k] = w[7]
    end

    # Initialize Bz at z-faces
    for k in 1:(nz + 1), j in 1:ny, i in 1:nx
        x_face = mesh.xmin + (i - 0.5) * dx
        y_face = mesh.ymin + (j - 0.5) * dy
        z_face = mesh.zmin + (k - 1) * dz
        w = prob.initial_condition(x_face, y_face, z_face)
        ct.Bz_face[i, j, k] = w[8]
    end

    return nothing
end

"""
    initialize_ct_3d_from_potential!(ct::CTData3D, Ax_func, Ay_func, Az_func, mesh)

Initialize face-centered B fields from a vector potential A = (Ax, Ay, Az).
This guarantees div(B) = 0 to machine precision via Stokes' theorem:

    B = curl(A)
    Bx = dAz/dy - dAy/dz
    By = dAx/dz - dAz/dx
    Bz = dAy/dx - dAx/dy

Each face value is the line integral of A around the face edges divided by the face area.
"""
function initialize_ct_3d_from_potential!(ct::CTData3D, Ax_func, Ay_func, Az_func, mesh)
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz

    # Bx at x-faces: face (i,j,k) is a yz-rectangle at x = xmin + (i-1)*dx
    # Bx = integral of A.dl / (dy*dz) around yz face
    # Bx = (Az(y+dy) - Az(y)) / dy - (Ay(z+dz) - Ay(z)) / dz
    for k in 1:nz, j in 1:ny, i in 1:(nx + 1)
        x = mesh.xmin + (i - 1) * dx
        y_lo = mesh.ymin + (j - 1) * dy
        y_hi = mesh.ymin + j * dy
        z_lo = mesh.zmin + (k - 1) * dz
        z_hi = mesh.zmin + k * dz
        y_mid = 0.5 * (y_lo + y_hi)
        z_mid = 0.5 * (z_lo + z_hi)

        # Bx = dAz/dy - dAy/dz (average over face)
        dAz_dy = (Az_func(x, y_hi, z_mid) - Az_func(x, y_lo, z_mid)) / dy
        dAy_dz = (Ay_func(x, y_mid, z_hi) - Ay_func(x, y_mid, z_lo)) / dz
        ct.Bx_face[i, j, k] = dAz_dy - dAy_dz
    end

    # By at y-faces: face (i,j,k) is an xz-rectangle at y = ymin + (j-1)*dy
    for k in 1:nz, j in 1:(ny + 1), i in 1:nx
        x_lo = mesh.xmin + (i - 1) * dx
        x_hi = mesh.xmin + i * dx
        y = mesh.ymin + (j - 1) * dy
        z_lo = mesh.zmin + (k - 1) * dz
        z_hi = mesh.zmin + k * dz
        x_mid = 0.5 * (x_lo + x_hi)
        z_mid = 0.5 * (z_lo + z_hi)

        # By = dAx/dz - dAz/dx
        dAx_dz = (Ax_func(x_mid, y, z_hi) - Ax_func(x_mid, y, z_lo)) / dz
        dAz_dx = (Az_func(x_hi, y, z_mid) - Az_func(x_lo, y, z_mid)) / dx
        ct.By_face[i, j, k] = dAx_dz - dAz_dx
    end

    # Bz at z-faces: face (i,j,k) is an xy-rectangle at z = zmin + (k-1)*dz
    for k in 1:(nz + 1), j in 1:ny, i in 1:nx
        x_lo = mesh.xmin + (i - 1) * dx
        x_hi = mesh.xmin + i * dx
        y_lo = mesh.ymin + (j - 1) * dy
        y_hi = mesh.ymin + j * dy
        z = mesh.zmin + (k - 1) * dz
        x_mid = 0.5 * (x_lo + x_hi)
        y_mid = 0.5 * (y_lo + y_hi)

        # Bz = dAy/dx - dAx/dy
        dAy_dx = (Ay_func(x_hi, y_mid, z) - Ay_func(x_lo, y_mid, z)) / dx
        dAx_dy = (Ax_func(x_mid, y_hi, z) - Ax_func(x_mid, y_lo, z)) / dy
        ct.Bz_face[i, j, k] = dAy_dx - dAx_dy
    end

    return nothing
end

"""
    face_to_cell_B_3d!(U, ct::CTData3D, nx, ny, nz)

Update cell-centered B in the conserved variable array from face-centered values.
Cell-centered B is the arithmetic mean of the two adjacent face values.

Interior cell (ix, iy, iz) maps to `U[ix+2, iy+2, iz+2]` in the padded array.
"""
function face_to_cell_B_3d!(U::AbstractArray{T, 3}, ct::CTData3D, nx::Int, ny::Int, nz::Int) where {T}
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        Bx_cell = 0.5 * (ct.Bx_face[ix, iy, iz] + ct.Bx_face[ix + 1, iy, iz])
        By_cell = 0.5 * (ct.By_face[ix, iy, iz] + ct.By_face[ix, iy + 1, iz])
        Bz_cell = 0.5 * (ct.Bz_face[ix, iy, iz] + ct.Bz_face[ix, iy, iz + 1])
        u = U[ix + 2, iy + 2, iz + 2]
        U[ix + 2, iy + 2, iz + 2] = SVector(u[1], u[2], u[3], u[4], u[5], Bx_cell, By_cell, Bz_cell)
    end
    return nothing
end

"""
    copy_ct(ct::CTData3D) -> CTData3D

Create a deep copy of the 3D CT data.
"""
function copy_ct(ct::CTData3D)
    return CTData3D(
        copy(ct.Bx_face), copy(ct.By_face), copy(ct.Bz_face),
        copy(ct.emf_x), copy(ct.emf_y), copy(ct.emf_z)
    )
end

"""
    copyto_ct!(dst::CTData3D, src::CTData3D)

Copy 3D CT data from `src` to `dst` in-place.
"""
function copyto_ct!(dst::CTData3D, src::CTData3D)
    copyto!(dst.Bx_face, src.Bx_face)
    copyto!(dst.By_face, src.By_face)
    copyto!(dst.Bz_face, src.Bz_face)
    copyto!(dst.emf_x, src.emf_x)
    copyto!(dst.emf_y, src.emf_y)
    copyto!(dst.emf_z, src.emf_z)
    return nothing
end
