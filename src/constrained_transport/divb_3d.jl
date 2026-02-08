# ============================================================
# 3D Divergence of B Diagnostics
# ============================================================

"""
    compute_divB_3d(ct::CTData3D, dx, dy, dz, nx, ny, nz) -> Array{FT, 3}

Compute the cell-centered div(B) from face-centered B values:
  div(B)[i,j,k] = (Bx_face[i+1,j,k] - Bx_face[i,j,k]) / dx
                 + (By_face[i,j+1,k] - By_face[i,j,k]) / dy
                 + (Bz_face[i,j,k+1] - Bz_face[i,j,k]) / dz

Should be zero to machine precision when CT is used.
"""
function compute_divB_3d(ct::CTData3D, dx, dy, dz, nx::Int, ny::Int, nz::Int)
    Bx = ct.Bx_face
    By = ct.By_face
    Bz = ct.Bz_face
    FT = eltype(Bx)

    divB = Array{FT, 3}(undef, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        divB[i, j, k] = (Bx[i + 1, j, k] - Bx[i, j, k]) / dx +
                         (By[i, j + 1, k] - By[i, j, k]) / dy +
                         (Bz[i, j, k + 1] - Bz[i, j, k]) / dz
    end
    return divB
end

"""
    max_divB_3d(ct::CTData3D, dx, dy, dz, nx, ny, nz) -> FT

Compute the maximum |div(B)| over all cells.
"""
function max_divB_3d(ct::CTData3D, dx, dy, dz, nx::Int, ny::Int, nz::Int)
    divB = compute_divB_3d(ct, dx, dy, dz, nx, ny, nz)
    return maximum(abs, divB)
end

"""
    l2_divB_3d(ct::CTData3D, dx, dy, dz, nx, ny, nz) -> FT

Compute the L2 norm of div(B).
"""
function l2_divB_3d(ct::CTData3D, dx, dy, dz, nx::Int, ny::Int, nz::Int)
    divB = compute_divB_3d(ct, dx, dy, dz, nx, ny, nz)
    return sqrt(sum(d^2 for d in divB) / (nx * ny * nz))
end
