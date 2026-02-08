# ============================================================
# Constrained Transport Data Structures
# ============================================================
#
# For 2D MHD on a structured Cartesian mesh (nx × ny):
#
# Face-centered magnetic field:
#   Bx_face[1:nx+1, 1:ny]  — x-component at x-faces (vertical faces)
#   By_face[1:nx, 1:ny+1]  — y-component at y-faces (horizontal faces)
#
# Cell-centered B (averaged from face values):
#   Bx_cell[i,j] = 0.5 * (Bx_face[i,j] + Bx_face[i+1,j])
#   By_cell[i,j] = 0.5 * (By_face[i,j] + By_face[i,j+1])
#
# Corner-centered EMF (electromotive force):
#   emf_z[1:nx+1, 1:ny+1]  — z-component of E at cell corners
#
# The CT update guarantees ∇·B = 0 to machine precision via:
#   Bx_face[i,j] -= dt/dy * (emf_z[i,j+1] - emf_z[i,j])
#   By_face[i,j] += dt/dx * (emf_z[i+1,j] - emf_z[i,j])

"""
    CTData2D{FT}

Constrained transport data for 2D MHD on a structured mesh.

Stores face-centered magnetic field components and corner EMFs.

# Fields
- `Bx_face::Matrix{FT}`: x-component of B at x-faces, size `(nx+1) × ny`.
- `By_face::Matrix{FT}`: y-component of B at y-faces, size `nx × (ny+1)`.
- `emf_z::Matrix{FT}`: z-component of EMF at corners, size `(nx+1) × (ny+1)`.
"""
struct CTData2D{FT}
    Bx_face::Matrix{FT}
    By_face::Matrix{FT}
    emf_z::Matrix{FT}
end

"""
    CTData2D(nx::Int, ny::Int, ::Type{FT}=Float64) -> CTData2D{FT}

Create zero-initialized CT data for a `nx × ny` mesh.
"""
function CTData2D(nx::Int, ny::Int, ::Type{FT} = Float64) where {FT}
    Bx_face = zeros(FT, nx + 1, ny)
    By_face = zeros(FT, nx, ny + 1)
    emf_z = zeros(FT, nx + 1, ny + 1)
    return CTData2D(Bx_face, By_face, emf_z)
end

"""
    initialize_ct!(ct::CTData2D, prob::HyperbolicProblem2D, mesh)

Initialize face-centered B fields from the initial condition.
Face values are set by evaluating the initial condition at the face center
and extracting the appropriate B component.
"""
function initialize_ct!(ct::CTData2D, prob, mesh)
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy
    law = prob.law

    # Initialize Bx at x-faces (located at x_{i+1/2}, y_j)
    for j in 1:ny, i in 1:(nx + 1)
        # Face center: x = xmin + (i-1)*dx, y = ymin + (j-0.5)*dy
        x_face = mesh.xmin + (i - 1) * dx
        y_face = mesh.ymin + (j - 0.5) * dy
        w = prob.initial_condition(x_face, y_face)
        ct.Bx_face[i, j] = w[6]  # Bx from primitive
    end

    # Initialize By at y-faces (located at x_i, y_{j+1/2})
    for j in 1:(ny + 1), i in 1:nx
        x_face = mesh.xmin + (i - 0.5) * dx
        y_face = mesh.ymin + (j - 1) * dy
        w = prob.initial_condition(x_face, y_face)
        ct.By_face[i, j] = w[7]  # By from primitive
    end

    return nothing
end

"""
    face_to_cell_B!(U::AbstractMatrix, ct::CTData2D, nx::Int, ny::Int)

Update cell-centered B in the conserved variable array `U` from face-centered values.
Cell-centered B is the arithmetic mean of the two adjacent face values.

Interior cell (ix, iy) maps to `U[ix+2, iy+2]` in the padded array.
"""
function face_to_cell_B!(U::AbstractMatrix, ct::CTData2D, nx::Int, ny::Int)
    for iy in 1:ny, ix in 1:nx
        Bx_cell = 0.5 * (ct.Bx_face[ix, iy] + ct.Bx_face[ix + 1, iy])
        By_cell = 0.5 * (ct.By_face[ix, iy] + ct.By_face[ix, iy + 1])
        u = U[ix + 2, iy + 2]
        # Update Bx (index 6) and By (index 7), keep everything else
        U[ix + 2, iy + 2] = SVector(u[1], u[2], u[3], u[4], u[5], Bx_cell, By_cell, u[8])
    end
    return nothing
end

"""
    initialize_ct_from_potential!(ct::CTData2D, Az_func, mesh)

Initialize face-centered B fields from a vector potential function `Az(x, y)`.
This guarantees ∇·B = 0 to machine precision via Stokes' theorem:

    Bx_face[i,j] = (Az(x, y_top) - Az(x, y_bottom)) / dy
    By_face[i,j] = -(Az(x_right, y) - Az(x_left, y)) / dx
"""
function initialize_ct_from_potential!(ct::CTData2D, Az_func, mesh)
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    # Bx at x-faces: face (i,j) has corners at
    #   bottom: (xmin + (i-1)*dx, ymin + (j-1)*dy)
    #   top:    (xmin + (i-1)*dx, ymin + j*dy)
    for j in 1:ny, i in 1:(nx + 1)
        x = mesh.xmin + (i - 1) * dx
        y_bottom = mesh.ymin + (j - 1) * dy
        y_top = mesh.ymin + j * dy
        ct.Bx_face[i, j] = (Az_func(x, y_top) - Az_func(x, y_bottom)) / dy
    end

    # By at y-faces: face (i,j) has corners at
    #   left:  (xmin + (i-1)*dx, ymin + (j-1)*dy)
    #   right: (xmin + i*dx, ymin + (j-1)*dy)
    for j in 1:(ny + 1), i in 1:nx
        x_left = mesh.xmin + (i - 1) * dx
        x_right = mesh.xmin + i * dx
        y = mesh.ymin + (j - 1) * dy
        ct.By_face[i, j] = -(Az_func(x_right, y) - Az_func(x_left, y)) / dx
    end

    return nothing
end

"""
    copy_ct(ct::CTData2D) -> CTData2D

Create a deep copy of the CT data.
"""
function copy_ct(ct::CTData2D)
    return CTData2D(copy(ct.Bx_face), copy(ct.By_face), copy(ct.emf_z))
end

"""
    copyto_ct!(dst::CTData2D, src::CTData2D)

Copy CT data from `src` to `dst` in-place.
"""
function copyto_ct!(dst::CTData2D, src::CTData2D)
    copyto!(dst.Bx_face, src.Bx_face)
    copyto!(dst.By_face, src.By_face)
    copyto!(dst.emf_z, src.emf_z)
    return nothing
end
