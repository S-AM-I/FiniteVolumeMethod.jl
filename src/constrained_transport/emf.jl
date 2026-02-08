# ============================================================
# EMF Computation for Constrained Transport
# ============================================================
#
# The EMF (electromotive force) Ez is computed at cell corners from
# the Riemann solver fluxes at cell faces.
#
# From the x-sweep at face (i+1/2, j):
#   Ez_x[i,j] = -F_By = -(By*vx - Bx*vy) at x-face
#   This is the negative of the 7th component of the x-flux (the By induction flux)
#
# From the y-sweep at face (i, j+1/2):
#   Ez_y[i,j] = +G_Bx = (Bx*vy - By*vx) at y-face
#   This is the positive of the 6th component of the y-flux (the Bx induction flux)
#
# The corner EMF is computed by arithmetic averaging of the 4 adjacent face values
# (Balsara & Spicer 1999):
#   emf_z[i,j] = 0.25 * (Ez_x[i,j-1] + Ez_x[i,j] + Ez_y[i-1,j] + Ez_y[i,j])

"""
    compute_emf_2d!(ct::CTData2D, Fx_faces::AbstractMatrix, Fy_faces::AbstractMatrix,
                    nx::Int, ny::Int)

Compute the corner EMF `emf_z` from the x-direction and y-direction numerical fluxes.

# Arguments
- `Fx_faces`: x-fluxes at x-faces, size `(nx+1) × ny`. `Fx_faces[i,j]` is the flux
  at the face between cells (i-1,j) and (i,j).
- `Fy_faces`: y-fluxes at y-faces, size `nx × (ny+1)`. `Fy_faces[i,j]` is the flux
  at the face between cells (i,j-1) and (i,j).

The EMF at x-faces is `Ez_x = -Fx[7]` (negative of the By induction flux).
The EMF at y-faces is `Ez_y = +Fy[6]` (positive of the Bx induction flux).

Corner values are the arithmetic average of the 4 surrounding face EMFs.
"""
function compute_emf_2d!(ct::CTData2D, Fx_faces::AbstractMatrix, Fy_faces::AbstractMatrix,
        nx::Int, ny::Int)
    emf_z = ct.emf_z

    # Interior corners (i=2:nx, j=2:ny) have 4 surrounding faces
    for j in 2:ny, i in 2:nx
        # Ez from x-faces: -Fx_By at face (i, j-1) and (i, j)
        Ez_x_below = -Fx_faces[i, j - 1][7]  # x-face below
        Ez_x_above = -Fx_faces[i, j][7]       # x-face above

        # Ez from y-faces: +Fy_Bx at face (i-1, j) and (i, j)
        Ez_y_left = Fy_faces[i - 1, j][6]   # y-face to left
        Ez_y_right = Fy_faces[i, j][6]       # y-face to right

        emf_z[i, j] = 0.25 * (Ez_x_below + Ez_x_above + Ez_y_left + Ez_y_right)
    end

    # Boundary corners: use available face data
    # Bottom-left corner (1,1)
    _set_boundary_corner_emf!(emf_z, Fx_faces, Fy_faces, nx, ny)

    return nothing
end

"""
Set EMF at boundary corners using available face fluxes.
Boundary corners have fewer than 4 surrounding face values, so we average
over the available ones.
"""
function _set_boundary_corner_emf!(emf_z, Fx_faces, Fy_faces, nx, ny)
    # Bottom row (j=1): corners (i,1) for i=1:nx+1
    for i in 1:(nx + 1)
        Ez_x = -Fx_faces[min(i, size(Fx_faces, 1)), 1][7]
        if i >= 2 && i <= nx
            Ez_y_left = Fy_faces[i - 1, 1][6]
            Ez_y_right = Fy_faces[i, 1][6]
            emf_z[i, 1] = (Ez_x + 0.5 * (Ez_y_left + Ez_y_right)) / 2
        elseif i == 1
            if nx >= 1
                Ez_y = Fy_faces[1, 1][6]
                emf_z[1, 1] = 0.5 * (Ez_x + Ez_y)
            else
                emf_z[1, 1] = Ez_x
            end
        else  # i == nx+1
            Ez_y = Fy_faces[nx, 1][6]
            emf_z[nx + 1, 1] = 0.5 * (Ez_x + Ez_y)
        end
    end

    # Top row (j=ny+1): corners (i,ny+1) for i=1:nx+1
    for i in 1:(nx + 1)
        Ez_x = -Fx_faces[min(i, size(Fx_faces, 1)), ny][7]
        if i >= 2 && i <= nx
            Ez_y_left = Fy_faces[i - 1, ny + 1][6]
            Ez_y_right = Fy_faces[i, ny + 1][6]
            emf_z[i, ny + 1] = (Ez_x + 0.5 * (Ez_y_left + Ez_y_right)) / 2
        elseif i == 1
            if nx >= 1
                Ez_y = Fy_faces[1, ny + 1][6]
                emf_z[1, ny + 1] = 0.5 * (Ez_x + Ez_y)
            else
                emf_z[1, ny + 1] = Ez_x
            end
        else  # i == nx+1
            Ez_y = Fy_faces[nx, ny + 1][6]
            emf_z[nx + 1, ny + 1] = 0.5 * (Ez_x + Ez_y)
        end
    end

    # Left column (i=1): corners (1,j) for j=2:ny (corners 1,1 and 1,ny+1 already done)
    for j in 2:ny
        Ez_x_below = -Fx_faces[1, j - 1][7]
        Ez_x_above = -Fx_faces[1, j][7]
        Ez_y = Fy_faces[1, j][6]
        emf_z[1, j] = (Ez_x_below + Ez_x_above + 2 * Ez_y) / 4
    end

    # Right column (i=nx+1): corners (nx+1,j) for j=2:ny
    for j in 2:ny
        Ez_x_below = -Fx_faces[nx + 1, j - 1][7]
        Ez_x_above = -Fx_faces[nx + 1, j][7]
        Ez_y = Fy_faces[nx, j][6]
        emf_z[nx + 1, j] = (Ez_x_below + Ez_x_above + 2 * Ez_y) / 4
    end

    return nothing
end
