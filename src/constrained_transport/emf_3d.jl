# ============================================================
# 3D EMF Computation for Constrained Transport
# ============================================================
#
# The 3D EMF has 3 edge-centered components:
#
#   Ex at x-edges (ny+1 x nz+1 per x-column):
#     Ex = -(Fy[Bz]) + (Fz[By])
#     Ex[i, j, k] = 0.25 * (-Fy_Bz at adjacent y-faces + Fz_By at adjacent z-faces)
#
#   Ey at y-edges (nx+1 x nz+1 per y-column):
#     Ey = -(Fz[Bx]) + (Fx[Bz])
#     Ey[i, j, k] = 0.25 * (-Fz_Bx at adjacent z-faces + Fx_Bz at adjacent x-faces)
#
#   Ez at z-edges (nx+1 x ny+1 per z-column):
#     Ez = -(Fx[By]) + (Fy[Bx])
#     Ez[i, j, k] = 0.25 * (-Fx_By at adjacent x-faces + Fy_Bx at adjacent y-faces)
#
# This is the Balsara & Spicer (1999) arithmetic averaging of face fluxes
# to edge EMFs, extended to 3D.

"""
    _compute_emf_3d_from_extended!(ct, Fx_all, Fy_all, Fz_all, nx, ny, nz)

Compute the 3 edge-centered EMF components from extended face flux arrays.

The extended flux arrays include ghost slabs for uniform corner computation:
- `Fx_all[face_i, col_j, col_k]`: x-face fluxes, size `(nx+1) x (ny+2) x (nz+2)`
- `Fy_all[col_i, face_j, col_k]`: y-face fluxes, size `(nx+2) x (ny+1) x (nz+2)`
- `Fz_all[col_i, col_j, face_k]`: z-face fluxes, size `(nx+2) x (ny+2) x (nz+1)`

For MHD with 8 variables [rho, rhovx, rhovy, rhovz, E, Bx, By, Bz]:
- Fx induction: Fx[7] = By*vx - Bx*vy (By induction), Fx[8] = Bz*vx - Bx*vz (Bz induction)
- Fy induction: Fy[6] = Bx*vy - By*vx (Bx induction), Fy[8] = Bz*vy - By*vz (Bz induction)
- Fz induction: Fz[6] = Bx*vz - Bz*vx (Bx induction), Fz[7] = By*vz - Bz*vy (By induction)

EMF from face fluxes:
  Ex = -Fy[8] + Fz[7]   (contributions from By and Bz induction)
  Ey = -Fz[6] + Fx[8]   (contributions from Bx and Bz induction)
  Ez = -Fx[7] + Fy[6]   (contributions from Bx and By induction)
"""
function _compute_emf_3d_from_extended!(ct::CTData3D,
        Fx_all::AbstractArray{T, 3}, Fy_all::AbstractArray{T, 3},
        Fz_all::AbstractArray{T, 3}, nx::Int, ny::Int, nz::Int) where {T}

    # Ez at z-edges: size (nx+1) x (ny+1) x nz
    # Edge (i, j, k) is surrounded by:
    #   x-faces: Fx_all[i, j, k] and Fx_all[i, j+1, k]  (y-neighbors)
    #   y-faces: Fy_all[i, j, k] and Fy_all[i+1, j, k]  (x-neighbors)
    for k in 1:nz, j in 1:(ny + 1), i in 1:(nx + 1)
        Ez_fx_lo = -Fx_all[i, j, k][7]      # x-face below in y
        Ez_fx_hi = -Fx_all[i, j + 1, k][7]  # x-face above in y
        Ez_fy_lo = Fy_all[i, j, k][6]       # y-face to left in x
        Ez_fy_hi = Fy_all[i + 1, j, k][6]   # y-face to right in x
        ct.emf_z[i, j, k] = 0.25 * (Ez_fx_lo + Ez_fx_hi + Ez_fy_lo + Ez_fy_hi)
    end

    # Ex at x-edges: size nx x (ny+1) x (nz+1)
    # Edge (i, j, k) is surrounded by:
    #   y-faces: Fy_all[i+1, j, k] and Fy_all[i+1, j, k+1]  (z-neighbors, offset in extended array)
    #   z-faces: Fz_all[i+1, j, k] and Fz_all[i+1, j+1, k]  (y-neighbors, offset in extended array)
    # Using extended arrays where col_i = i+1 for interior column i
    for k in 1:(nz + 1), j in 1:(ny + 1), i in 1:nx
        # Fy contribution: -Fy[8] (Bz induction via y-flux)
        Ex_fy_lo = -Fy_all[i + 1, j, k][8]      # z-neighbor below
        Ex_fy_hi = -Fy_all[i + 1, j, k + 1][8]  # z-neighbor above
        # Fz contribution: +Fz[7] (By induction via z-flux)
        Ex_fz_lo = Fz_all[i + 1, j, k][7]       # y-neighbor below
        Ex_fz_hi = Fz_all[i + 1, j + 1, k][7]   # y-neighbor above
        ct.emf_x[i, j, k] = 0.25 * (Ex_fy_lo + Ex_fy_hi + Ex_fz_lo + Ex_fz_hi)
    end

    # Ey at y-edges: size (nx+1) x ny x (nz+1)
    # Edge (i, j, k) is surrounded by:
    #   z-faces: Fz_all[i, j+1, k] and Fz_all[i+1, j+1, k]  (x-neighbors)
    #   x-faces: Fx_all[i, j+1, k] and Fx_all[i, j+1, k+1]  (z-neighbors)
    for k in 1:(nz + 1), j in 1:ny, i in 1:(nx + 1)
        # Fz contribution: -Fz[6] (Bx induction via z-flux)
        Ey_fz_lo = -Fz_all[i, j + 1, k][6]      # x-neighbor to left
        Ey_fz_hi = -Fz_all[i + 1, j + 1, k][6]  # x-neighbor to right
        # Fx contribution: +Fx[8] (Bz induction via x-flux)
        Ey_fx_lo = Fx_all[i, j + 1, k][8]        # z-neighbor below
        Ey_fx_hi = Fx_all[i, j + 1, k + 1][8]    # z-neighbor above
        ct.emf_y[i, j, k] = 0.25 * (Ey_fz_lo + Ey_fz_hi + Ey_fx_lo + Ey_fx_hi)
    end

    return nothing
end
