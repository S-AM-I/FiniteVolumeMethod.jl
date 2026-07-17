# ============================================================
# Restriction (Fine -> Coarse)
# ============================================================
#
# When a block is coarsened, the fine solution is averaged back
# to the coarse grid using volume-weighted averaging. This ensures
# conservation of the integrated conserved quantities.
#
# For a refinement ratio of 2:
#   - 2D: 4 fine cells average into 1 coarse cell
#   - 3D: 8 fine cells average into 1 coarse cell

"""
    restrict!(parent::AMRBlock, children::Vector{AMRBlock})

Restrict (average) the fine child solutions onto the coarse parent block.
Uses simple volume-weighted averaging (all fine cells have equal volume
within a coarse cell for uniform refinement).
"""
function restrict!(parent::AMRBlock{N, FT, 2}, children::Vector{<:AMRBlock{N, FT, 2}}) where {N, FT}
    pnx, pny = parent.dims
    pdx, pdy = parent.dx

    # Zero parent data
    zero_state = zero(SVector{N, FT})
    for j in 1:pny, i in 1:pnx
        parent.U[i, j] = zero_state
    end

    for child in children
        cnx, cny = child.dims
        cdx, cdy = child.dx

        for cj in 1:cny, ci in 1:cnx
            # Physical coordinates of fine cell center
            xc = child.origin[1] + (ci - FT(0.5)) * cdx
            yc = child.origin[2] + (cj - FT(0.5)) * cdy

            # Corresponding coarse cell
            pi = clamp(Int(floor((xc - parent.origin[1]) / pdx)) + 1, 1, pnx)
            pj = clamp(Int(floor((yc - parent.origin[2]) / pdy)) + 1, 1, pny)

            # Volume-weighted contribution (each fine cell has volume cdx*cdy)
            parent.U[pi, pj] = parent.U[pi, pj] + child.U[ci, cj]
        end
    end

    # Normalize by number of fine cells per coarse cell (4 for ratio 2 in 2D)
    n_fine_per_coarse = 4
    inv_n = FT(1) / FT(n_fine_per_coarse)
    for j in 1:pny, i in 1:pnx
        parent.U[i, j] = parent.U[i, j] * inv_n
    end

    return nothing
end

function restrict!(parent::AMRBlock{N, FT, 3}, children::Vector{<:AMRBlock{N, FT, 3}}) where {N, FT}
    pnx, pny, pnz = parent.dims
    pdx, pdy, pdz = parent.dx

    # Zero parent data
    zero_state = zero(SVector{N, FT})
    for k in 1:pnz, j in 1:pny, i in 1:pnx
        parent.U[i, j, k] = zero_state
    end

    for child in children
        cnx, cny, cnz = child.dims
        cdx, cdy, cdz = child.dx

        for ck in 1:cnz, cj in 1:cny, ci in 1:cnx
            # Physical coordinates of fine cell center
            xc = child.origin[1] + (ci - FT(0.5)) * cdx
            yc = child.origin[2] + (cj - FT(0.5)) * cdy
            zc = child.origin[3] + (ck - FT(0.5)) * cdz

            # Corresponding coarse cell
            pi = clamp(Int(floor((xc - parent.origin[1]) / pdx)) + 1, 1, pnx)
            pj = clamp(Int(floor((yc - parent.origin[2]) / pdy)) + 1, 1, pny)
            pk = clamp(Int(floor((zc - parent.origin[3]) / pdz)) + 1, 1, pnz)

            parent.U[pi, pj, pk] = parent.U[pi, pj, pk] + child.U[ci, cj, ck]
        end
    end

    # Normalize by number of fine cells per coarse cell (8 for ratio 2 in 3D)
    n_fine_per_coarse = 8
    inv_n = FT(1) / FT(n_fine_per_coarse)
    for k in 1:pnz, j in 1:pny, i in 1:pnx
        parent.U[i, j, k] = parent.U[i, j, k] * inv_n
    end

    return nothing
end

"""
    restrict_B_face!(Bx_coarse, By_coarse, Bx_fine, By_fine, nx_c, ny_c)

Restrict face-centered magnetic field from fine to coarse grid.
Coarse face value = average of the fine faces it covers.

For refinement ratio 2 in 2D:
  Bx_coarse[i,j] = 0.5 * (Bx_fine[2i-1, 2j-1] + Bx_fine[2i-1, 2j])
  By_coarse[i,j] = 0.5 * (By_fine[2i-1, 2j-1] + By_fine[2i, 2j-1])
"""
function restrict_B_face_2d!(
        Bx_coarse::AbstractMatrix, By_coarse::AbstractMatrix,
        Bx_fine::AbstractMatrix, By_fine::AbstractMatrix,
        nx_c::Int, ny_c::Int
    )
    # Restrict Bx at x-faces
    for j in 1:ny_c, i in 1:(nx_c + 1)
        fi = 2 * i - 1
        Bx_coarse[i, j] = 0.5 * (Bx_fine[fi, 2 * j - 1] + Bx_fine[fi, 2 * j])
    end

    # Restrict By at y-faces
    for j in 1:(ny_c + 1), i in 1:nx_c
        fj = 2 * j - 1
        By_coarse[i, j] = 0.5 * (By_fine[2 * i - 1, fj] + By_fine[2 * i, fj])
    end

    return nothing
end

"""
    restrict_B_face_3d!(Bx_c, By_c, Bz_c, Bx_f, By_f, Bz_f, nx_c, ny_c, nz_c)

Restrict face-centered magnetic field from fine to coarse grid in 3D.
Each coarse face value = average of the 4 fine faces it covers.
"""
function restrict_B_face_3d!(
        Bx_coarse::AbstractArray{T, 3}, By_coarse::AbstractArray{T, 3}, Bz_coarse::AbstractArray{T, 3},
        Bx_fine::AbstractArray{T, 3}, By_fine::AbstractArray{T, 3}, Bz_fine::AbstractArray{T, 3},
        nx_c::Int, ny_c::Int, nz_c::Int
    ) where {T}
    # Restrict Bx at x-faces: coarse face (i,j,k) = avg of 4 fine faces
    for k in 1:nz_c, j in 1:ny_c, i in 1:(nx_c + 1)
        fi = 2 * i - 1
        fj1 = 2 * j - 1
        fj2 = 2 * j
        fk1 = 2 * k - 1
        fk2 = 2 * k
        Bx_coarse[i, j, k] = 0.25 * (
            Bx_fine[fi, fj1, fk1] + Bx_fine[fi, fj2, fk1] +
                Bx_fine[fi, fj1, fk2] + Bx_fine[fi, fj2, fk2]
        )
    end

    # Restrict By at y-faces
    for k in 1:nz_c, j in 1:(ny_c + 1), i in 1:nx_c
        fj = 2 * j - 1
        fi1 = 2 * i - 1
        fi2 = 2 * i
        fk1 = 2 * k - 1
        fk2 = 2 * k
        By_coarse[i, j, k] = 0.25 * (
            By_fine[fi1, fj, fk1] + By_fine[fi2, fj, fk1] +
                By_fine[fi1, fj, fk2] + By_fine[fi2, fj, fk2]
        )
    end

    # Restrict Bz at z-faces
    for k in 1:(nz_c + 1), j in 1:ny_c, i in 1:nx_c
        fk = 2 * k - 1
        fi1 = 2 * i - 1
        fi2 = 2 * i
        fj1 = 2 * j - 1
        fj2 = 2 * j
        Bz_coarse[i, j, k] = 0.25 * (
            Bz_fine[fi1, fj1, fk] + Bz_fine[fi2, fj1, fk] +
                Bz_fine[fi1, fj2, fk] + Bz_fine[fi2, fj2, fk]
        )
    end

    return nothing
end
