# ============================================================
# Divergence of B Diagnostics
# ============================================================

"""
    compute_divB(ct::CTData2D, dx, dy, nx::Int, ny::Int) -> Matrix{FT}

Compute the cell-centered ∇·B from face-centered B values:
  `∇·B[i,j] = (Bx_face[i+1,j] - Bx_face[i,j]) / dx
             + (By_face[i,j+1] - By_face[i,j]) / dy`

Should be zero to machine precision when CT is used.
"""
function compute_divB(ct::CTData2D, dx, dy, nx::Int, ny::Int)
    Bx = ct.Bx_face
    By = ct.By_face
    FT = eltype(Bx)

    divB = Matrix{FT}(undef, nx, ny)
    for j in 1:ny, i in 1:nx
        divB[i, j] = (Bx[i + 1, j] - Bx[i, j]) / dx + (By[i, j + 1] - By[i, j]) / dy
    end
    return divB
end

"""
    max_divB(ct::CTData2D, dx, dy, nx::Int, ny::Int) -> FT

Compute the maximum |∇·B| over all cells.
"""
function max_divB(ct::CTData2D, dx, dy, nx::Int, ny::Int)
    divB = compute_divB(ct, dx, dy, nx, ny)
    return maximum(abs, divB)
end

"""
    l2_divB(ct::CTData2D, dx, dy, nx::Int, ny::Int) -> FT

Compute the L2 norm of ∇·B.
"""
function l2_divB(ct::CTData2D, dx, dy, nx::Int, ny::Int)
    divB = compute_divB(ct, dx, dy, nx, ny)
    return sqrt(sum(d^2 for d in divB) / (nx * ny))
end
