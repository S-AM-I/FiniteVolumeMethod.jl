# ============================================================
# Prolongation (Coarse -> Fine)
# ============================================================
#
# When a block is refined, the coarse solution must be interpolated
# onto the fine grid. For conserved variables, bilinear (2D) or
# trilinear (3D) interpolation preserves conservation.
#
# For MHD, the magnetic field prolongation must preserve div(B) = 0.
# This is achieved by prolongating face-centered B fields using
# divergence-preserving interpolation (Balsara 2001).

"""
    prolongate!(parent::AMRBlock, children::Vector{AMRBlock}, law)

Prolongate (interpolate) the coarse parent solution onto the fine child blocks.
Uses conservative linear interpolation with slope limiting.
"""
function prolongate!(parent::AMRBlock{N, FT, 2}, children::Vector{<:AMRBlock{N, FT, 2}}, law) where {N, FT}
    pnx, pny = parent.dims
    pdx, pdy = parent.dx

    for child in children
        cnx, cny = child.dims
        cdx, cdy = child.dx

        for cj in 1:cny, ci in 1:cnx
            # Physical coordinates of fine cell center
            xc = child.origin[1] + (ci - FT(0.5)) * cdx
            yc = child.origin[2] + (cj - FT(0.5)) * cdy

            # Corresponding coarse cell indices
            pi = clamp(Int(floor((xc - parent.origin[1]) / pdx)) + 1, 1, pnx)
            pj = clamp(Int(floor((yc - parent.origin[2]) / pdy)) + 1, 1, pny)

            # Coarse cell center
            xp = parent.origin[1] + (pi - FT(0.5)) * pdx
            yp = parent.origin[2] + (pj - FT(0.5)) * pdy

            # Compute limited slopes on coarse grid
            u_c = parent.U[pi, pj]

            # x-slope
            if pi > 1 && pi < pnx
                slope_x = _minmod_sv(parent.U[pi + 1, pj] - u_c, u_c - parent.U[pi - 1, pj])
            else
                slope_x = zero(SVector{N, FT})
            end

            # y-slope
            if pj > 1 && pj < pny
                slope_y = _minmod_sv(parent.U[pi, pj + 1] - u_c, u_c - parent.U[pi, pj - 1])
            else
                slope_y = zero(SVector{N, FT})
            end

            # Linear interpolation
            dx_offset = (xc - xp) / pdx
            dy_offset = (yc - yp) / pdy
            child.U[ci, cj] = u_c + slope_x * dx_offset + slope_y * dy_offset
        end
    end
    return nothing
end

function prolongate!(parent::AMRBlock{N, FT, 3}, children::Vector{<:AMRBlock{N, FT, 3}}, law) where {N, FT}
    pnx, pny, pnz = parent.dims
    pdx, pdy, pdz = parent.dx

    for child in children
        cnx, cny, cnz = child.dims
        cdx, cdy, cdz = child.dx

        for ck in 1:cnz, cj in 1:cny, ci in 1:cnx
            # Physical coordinates of fine cell center
            xc = child.origin[1] + (ci - FT(0.5)) * cdx
            yc = child.origin[2] + (cj - FT(0.5)) * cdy
            zc = child.origin[3] + (ck - FT(0.5)) * cdz

            # Corresponding coarse cell indices
            pi = clamp(Int(floor((xc - parent.origin[1]) / pdx)) + 1, 1, pnx)
            pj = clamp(Int(floor((yc - parent.origin[2]) / pdy)) + 1, 1, pny)
            pk = clamp(Int(floor((zc - parent.origin[3]) / pdz)) + 1, 1, pnz)

            # Coarse cell center
            xp = parent.origin[1] + (pi - FT(0.5)) * pdx
            yp = parent.origin[2] + (pj - FT(0.5)) * pdy
            zp = parent.origin[3] + (pk - FT(0.5)) * pdz

            # Compute limited slopes on coarse grid
            u_c = parent.U[pi, pj, pk]

            # x-slope
            if pi > 1 && pi < pnx
                slope_x = _minmod_sv(parent.U[pi + 1, pj, pk] - u_c, u_c - parent.U[pi - 1, pj, pk])
            else
                slope_x = zero(SVector{N, FT})
            end

            # y-slope
            if pj > 1 && pj < pny
                slope_y = _minmod_sv(parent.U[pi, pj + 1, pk] - u_c, u_c - parent.U[pi, pj - 1, pk])
            else
                slope_y = zero(SVector{N, FT})
            end

            # z-slope
            if pk > 1 && pk < pnz
                slope_z = _minmod_sv(parent.U[pi, pj, pk + 1] - u_c, u_c - parent.U[pi, pj, pk - 1])
            else
                slope_z = zero(SVector{N, FT})
            end

            # Trilinear interpolation
            dx_offset = (xc - xp) / pdx
            dy_offset = (yc - yp) / pdy
            dz_offset = (zc - zp) / pdz
            child.U[ci, cj, ck] = u_c + slope_x * dx_offset + slope_y * dy_offset + slope_z * dz_offset
        end
    end
    return nothing
end

# ============================================================
# Divergence-Preserving Prolongation for B
# ============================================================

"""
    prolongate_B_divergence_preserving!(parent_ct, child_cts, parent, children, mesh_params)

Prolongate face-centered magnetic field from coarse to fine grid while
preserving div(B) = 0 to machine precision.

Uses the Balsara (2001) divergence-preserving interpolation:
1. Prolongate face-normal B components at face centers using area-weighted averaging
2. Adjust tangential components to maintain div(B) = 0 in each fine cell
"""
function prolongate_B_divergence_preserving_2d!(
        Bx_coarse::AbstractMatrix, By_coarse::AbstractMatrix,
        Bx_fine::AbstractMatrix, By_fine::AbstractMatrix,
        nx_c::Int, ny_c::Int
    )
    # Refinement ratio is 2: each coarse cell -> 4 fine cells
    nx_f = 2 * nx_c
    ny_f = 2 * ny_c

    # Step 1: Prolongate Bx at x-faces
    # Coarse x-face (i, j) maps to fine x-faces (2i-1, 2j-1) and (2i-1, 2j)
    for j in 1:ny_c, i in 1:(nx_c + 1)
        fi = 2 * i - 1
        Bx_fine[fi, 2 * j - 1] = Bx_coarse[i, j]
        Bx_fine[fi, 2 * j] = Bx_coarse[i, j]
    end

    # Step 2: Prolongate By at y-faces
    # Coarse y-face (i, j) maps to fine y-faces (2i-1, 2j-1) and (2i, 2j-1)
    for j in 1:(ny_c + 1), i in 1:nx_c
        fj = 2 * j - 1
        By_fine[2 * i - 1, fj] = By_coarse[i, j]
        By_fine[2 * i, fj] = By_coarse[i, j]
    end

    # Step 3: Compute intermediate face values to enforce div(B) = 0 in each fine cell
    # For each coarse cell (i,j), the 4 fine cells must individually satisfy div(B) = 0
    for j in 1:ny_c, i in 1:nx_c
        # Fine cell indices within this coarse cell: (2i-1,2j-1), (2i,2j-1), (2i-1,2j), (2i,2j)
        fi = 2 * i - 1
        fj = 2 * j - 1

        # Internal x-face at fine face index fi+1 = 2i
        # Determined by div(B) = 0 for left sub-cell (fi, fj):
        # (Bx[fi+1,fj] - Bx[fi,fj])/dx + (By[fi,fj+1] - By[fi,fj])/dy = 0
        # Since dx_f = dy_f for uniform refinement:
        Bx_fine[fi + 1, fj] = Bx_fine[fi, fj] - (By_fine[fi, fj + 1] - By_fine[fi, fj])
        Bx_fine[fi + 1, fj + 1] = Bx_fine[fi, fj + 1] - (By_fine[fi, fj + 2] - By_fine[fi, fj + 1])

        # Internal y-face at fine face index fj+1 = 2j
        By_fine[fi, fj + 1] = By_fine[fi, fj] - (Bx_fine[fi + 1, fj] - Bx_fine[fi, fj])
        By_fine[fi + 1, fj + 1] = By_fine[fi + 1, fj] - (Bx_fine[fi + 2, fj] - Bx_fine[fi + 1, fj])
    end

    return nothing
end

# ============================================================
# Component-wise minmod for SVector
# ============================================================

"""
Component-wise minmod limiter for SVectors.
"""
@inline function _minmod_sv(a::SVector{N, FT}, b::SVector{N, FT}) where {N, FT}
    return SVector{N, FT}(ntuple(i -> _minmod_scalar(a[i], b[i]), Val(N)))
end

@inline function _minmod_scalar(a, b)
    if a * b <= 0
        return zero(a)
    elseif abs(a) < abs(b)
        return a
    else
        return b
    end
end
