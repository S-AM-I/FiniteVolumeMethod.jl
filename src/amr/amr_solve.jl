# ============================================================
# AMR Time Stepping with Subcycling
# ============================================================
#
# Berger-Oliger AMR time integration with recursive subcycling:
# - Fine levels take 2x as many steps as the next coarser level
# - After fine steps complete, restriction averages fine -> coarse
#
# The 2D stepping in `solve_amr` uses a synchronized (Jacobi) global
# step: every active block fills its ghost layer from neighboring
# blocks' pre-update data (same-level copy, piecewise-constant
# prolongation from coarser neighbors, conservative 2x2 averaging from
# finer neighbors), all RHS evaluations are computed from that
# consistent snapshot, coarse-side seam fluxes at single-level jumps
# are replaced by the area-averaged fine fluxes (method-of-lines flux
# correction, which makes the semidiscretization conservative at those
# interfaces), and only then are the blocks updated. This is exactly
# the semidiscretization used by `ODEProblem(::AMRProblem)`, so the
# legacy and SciML paths cannot diverge.
#
# REMAINING LIMITATIONS (see `_amr_domain_bc_warn`):
#   - Domain-boundary ghosts are zero-gradient; `prob.boundary_conditions`
#     is not consulted.
#   - Seam fluxes across level jumps >= 2 are NOT flux-corrected (ghost
#     data is still exchanged, but conservation there is approximate).
#   - The 3D block advance still has no ghost exchange: 3D multi-block
#     grids throw (see _amr_ghost_exchange_guard) and 3D single-block
#     grids use the historical zero-gradient path.
#   - The Berger-Colella time-integrated flux registers
#     (apply_flux_correction_2d!/3d!) remain uncalled; conservation at
#     2D coarse-fine seams is achieved by the MOL seam-flux replacement
#     instead.

"""
    AMRProblem{Grid, RS, Rec, BCs, FT}

An AMR problem definition wrapping the grid, solver settings, and boundary conditions.

# Fields
- `grid::Grid`: The AMR grid hierarchy.
- `riemann_solver::RS`: The Riemann solver.
- `reconstruction::Rec`: The reconstruction scheme.
- `boundary_conditions::BCs`: Boundary conditions (applied at domain boundaries).
- `initial_time::FT`, `final_time::FT`: Time span.
- `cfl::FT`: CFL number.
- `regrid_interval::Int`: Number of coarse steps between regrids.
"""
struct AMRProblem{Grid, RS, Rec, BCs, FT}
    grid::Grid
    riemann_solver::RS
    reconstruction::Rec
    boundary_conditions::BCs
    initial_time::FT
    final_time::FT
    cfl::FT
    regrid_interval::Int
end

function Base.show(io::IO, ::MIME"text/plain", prob::AMRProblem)
    nblocks = length(prob.grid.blocks)
    maxlvl = prob.grid.max_level
    t0 = prob.initial_time
    tf = prob.final_time
    law_name = nameof(typeof(prob.grid.law))
    rs_name = nameof(typeof(prob.riemann_solver))
    return print(io, "AMRProblem: $law_name with $rs_name, $nblocks blocks (max level $maxlvl), t ∈ ($t0, $tf)")
end

function AMRProblem(
        grid, riemann_solver, reconstruction, boundary_conditions;
        initial_time = 0.0, final_time, cfl = 0.4, regrid_interval = 4
    )
    return AMRProblem(
        grid, riemann_solver, reconstruction, boundary_conditions,
        initial_time, final_time, cfl, regrid_interval
    )
end

"""
    compute_dt_amr(prob::AMRProblem, level::Int) -> FT

Compute the time step for a given level based on the CFL condition.
The time step is the minimum over all active blocks at this level.
"""
function compute_dt_amr(prob::AMRProblem, level::Int)
    grid = prob.grid
    law = grid.law
    cfl = prob.cfl

    dt_min = typemax(Float64)
    for block in blocks_at_level(grid, level)
        dt_block = _compute_dt_block(block, law, cfl)
        dt_min = min(dt_min, dt_block)
    end

    return dt_min
end

"""
    _compute_dt_block(block, law, cfl) -> FT

Compute CFL time step for a single block.
"""
function _compute_dt_block(block::AMRBlock{N, FT, 2}, law, cfl) where {N, FT}
    nx, ny = block.dims
    dx_val, dy_val = block.dx

    max_speed = zero(FT)
    for j in 1:ny, i in 1:nx
        w = conserved_to_primitive(law, block.U[i, j])
        lx = max_wave_speed(law, w, 1)
        ly = max_wave_speed(law, w, 2)
        speed = lx / dx_val + ly / dy_val
        max_speed = max(max_speed, speed)
    end

    if max_speed > zero(FT)
        return cfl / max_speed
    else
        return typemax(FT)
    end
end

function _compute_dt_block(block::AMRBlock{N, FT, 3}, law, cfl) where {N, FT}
    nx, ny, nz = block.dims
    dx_val, dy_val, dz_val = block.dx

    max_speed = zero(FT)
    for k in 1:nz, j in 1:ny, i in 1:nx
        w = conserved_to_primitive(law, block.U[i, j, k])
        lx = max_wave_speed(law, w, 1)
        ly = max_wave_speed(law, w, 2)
        lz = max_wave_speed(law, w, 3)
        speed = lx / dx_val + ly / dy_val + lz / dz_val
        max_speed = max(max_speed, speed)
    end

    if max_speed > zero(FT)
        return cfl / max_speed
    else
        return typemax(FT)
    end
end

"""
    advance_level!(prob::AMRProblem, level::Int, dt, t)

Advance all blocks at the given level by one time step of size `dt`.
Uses forward Euler for simplicity; SSP-RK3 can be added.

This function handles the recursive subcycling:
- Advance the current level by dt
- If finer levels exist, advance them by dt/2 (twice)
- Restrict fine solution to coarse

!!! warning
    This is the historical per-block path: `_advance_block!` fills each
    block's ghost cells by zero-gradient extrapolation from that block's
    own interior, with NO inter-block ghost exchange. The 2D `solve_amr`
    no longer uses it (see `_amr_global_step_2d!`); it remains for the 3D
    path and for `solve_amr_subcycled(; method = :euler)`.
"""
function advance_level!(prob::AMRProblem, level::Int, dt, t)
    grid = prob.grid
    law = grid.law

    # Advance all blocks at this level
    for block in blocks_at_level(grid, level)
        _advance_block!(block, law, prob.riemann_solver, prob.reconstruction, dt)
    end

    # Subcycle finer levels
    max_lev = max_active_level(grid)
    if level < max_lev
        fine_dt = dt / 2
        # Two fine steps for each coarse step
        advance_level!(prob, level + 1, fine_dt, t)
        advance_level!(prob, level + 1, fine_dt, t + fine_dt)

        # Restrict fine solution to coarse
        _restrict_level!(grid, level)
    end

    return nothing
end

"""
    _advance_block!(block, law, solver, recon, dt)

Advance a single block by one Euler step.

Ghost cells are filled by zero-gradient extrapolation from this block's
own interior — NOT from neighboring blocks and NOT from the problem's
boundary conditions. See `_amr_ghost_exchange_guard`.
"""
function _advance_block!(block::AMRBlock{N, FT, 2}, law, solver, recon, dt) where {N, FT}
    nx, ny = block.dims
    dx_val, dy_val = block.dx

    # Create padded array with ghost cells
    U_pad = Matrix{SVector{N, FT}}(undef, nx + 4, ny + 4)
    zero_state = zero(SVector{N, FT})
    for j in axes(U_pad, 2), i in axes(U_pad, 1)
        U_pad[i, j] = zero_state
    end

    # Copy interior
    for j in 1:ny, i in 1:nx
        U_pad[i + 2, j + 2] = block.U[i, j]
    end

    # Zero-gradient ghost cells
    for j in 1:(ny + 4)
        U_pad[2, j] = U_pad[3, j]
        U_pad[1, j] = U_pad[3, j]
        U_pad[nx + 3, j] = U_pad[nx + 2, j]
        U_pad[nx + 4, j] = U_pad[nx + 2, j]
    end
    for i in 1:(nx + 4)
        U_pad[i, 2] = U_pad[i, 3]
        U_pad[i, 1] = U_pad[i, 3]
        U_pad[i, ny + 3] = U_pad[i, ny + 2]
        U_pad[i, ny + 4] = U_pad[i, ny + 2]
    end

    # Compute RHS via flux differencing
    dU = similar(U_pad)
    for j in axes(dU, 2), i in axes(dU, 1)
        dU[i, j] = zero_state
    end

    # X-sweeps
    for iy in 1:ny
        jj = iy + 2
        for ix in 0:nx
            iL = ix + 2
            iR = ix + 3
            wL = conserved_to_primitive(law, U_pad[iL, jj])
            wR = conserved_to_primitive(law, U_pad[iR, jj])
            F = solve_riemann(solver, law, wL, wR, 1)
            if ix >= 1
                dU[iL, jj] = dU[iL, jj] - F / dx_val
            end
            if ix < nx
                dU[iR, jj] = dU[iR, jj] + F / dx_val
            end
        end
    end

    # Y-sweeps
    for ix in 1:nx
        ii = ix + 2
        for iy in 0:ny
            jL = iy + 2
            jR = iy + 3
            wL = conserved_to_primitive(law, U_pad[ii, jL])
            wR = conserved_to_primitive(law, U_pad[ii, jR])
            F = solve_riemann(solver, law, wL, wR, 2)
            if iy >= 1
                dU[ii, jL] = dU[ii, jL] - F / dy_val
            end
            if iy < ny
                dU[ii, jR] = dU[ii, jR] + F / dy_val
            end
        end
    end

    # Forward Euler update
    for j in 1:ny, i in 1:nx
        block.U[i, j] = block.U[i, j] + dt * dU[i + 2, j + 2]
    end

    return nothing
end

function _advance_block!(block::AMRBlock{N, FT, 3}, law, solver, recon, dt) where {N, FT}
    nx, ny, nz = block.dims
    dx_val, dy_val, dz_val = block.dx

    # Create padded array with ghost cells
    U_pad = Array{SVector{N, FT}, 3}(undef, nx + 4, ny + 4, nz + 4)
    zero_state = zero(SVector{N, FT})
    for k in axes(U_pad, 3), j in axes(U_pad, 2), i in axes(U_pad, 1)
        U_pad[i, j, k] = zero_state
    end

    # Copy interior
    for k in 1:nz, j in 1:ny, i in 1:nx
        U_pad[i + 2, j + 2, k + 2] = block.U[i, j, k]
    end

    # Zero-gradient ghost cells for all 6 faces
    for k in 1:(nz + 4), j in 1:(ny + 4)
        U_pad[2, j, k] = U_pad[3, j, k]
        U_pad[1, j, k] = U_pad[3, j, k]
        U_pad[nx + 3, j, k] = U_pad[nx + 2, j, k]
        U_pad[nx + 4, j, k] = U_pad[nx + 2, j, k]
    end
    for k in 1:(nz + 4), i in 1:(nx + 4)
        U_pad[i, 2, k] = U_pad[i, 3, k]
        U_pad[i, 1, k] = U_pad[i, 3, k]
        U_pad[i, ny + 3, k] = U_pad[i, ny + 2, k]
        U_pad[i, ny + 4, k] = U_pad[i, ny + 2, k]
    end
    for j in 1:(ny + 4), i in 1:(nx + 4)
        U_pad[i, j, 2] = U_pad[i, j, 3]
        U_pad[i, j, 1] = U_pad[i, j, 3]
        U_pad[i, j, nz + 3] = U_pad[i, j, nz + 2]
        U_pad[i, j, nz + 4] = U_pad[i, j, nz + 2]
    end

    # Compute RHS via flux differencing
    dU = similar(U_pad)
    for k in axes(dU, 3), j in axes(dU, 2), i in axes(dU, 1)
        dU[i, j, k] = zero_state
    end

    # X-sweeps
    for iz in 1:nz, iy in 1:ny
        jj = iy + 2
        kk = iz + 2
        for ix in 0:nx
            iL = ix + 2
            iR = ix + 3
            wL = conserved_to_primitive(law, U_pad[iL, jj, kk])
            wR = conserved_to_primitive(law, U_pad[iR, jj, kk])
            F = solve_riemann(solver, law, wL, wR, 1)
            if ix >= 1
                dU[iL, jj, kk] = dU[iL, jj, kk] - F / dx_val
            end
            if ix < nx
                dU[iR, jj, kk] = dU[iR, jj, kk] + F / dx_val
            end
        end
    end

    # Y-sweeps
    for iz in 1:nz, ix in 1:nx
        ii = ix + 2
        kk = iz + 2
        for iy in 0:ny
            jL = iy + 2
            jR = iy + 3
            wL = conserved_to_primitive(law, U_pad[ii, jL, kk])
            wR = conserved_to_primitive(law, U_pad[ii, jR, kk])
            F = solve_riemann(solver, law, wL, wR, 2)
            if iy >= 1
                dU[ii, jL, kk] = dU[ii, jL, kk] - F / dy_val
            end
            if iy < ny
                dU[ii, jR, kk] = dU[ii, jR, kk] + F / dy_val
            end
        end
    end

    # Z-sweeps
    for iy in 1:ny, ix in 1:nx
        ii = ix + 2
        jj = iy + 2
        for iz in 0:nz
            kL = iz + 2
            kR = iz + 3
            wL = conserved_to_primitive(law, U_pad[ii, jj, kL])
            wR = conserved_to_primitive(law, U_pad[ii, jj, kR])
            F = solve_riemann(solver, law, wL, wR, 3)
            if iz >= 1
                dU[ii, jj, kL] = dU[ii, jj, kL] - F / dz_val
            end
            if iz < nz
                dU[ii, jj, kR] = dU[ii, jj, kR] + F / dz_val
            end
        end
    end

    # Forward Euler update
    for k in 1:nz, j in 1:ny, i in 1:nx
        block.U[i, j, k] = block.U[i, j, k] + dt * dU[i + 2, j + 2, k + 2]
    end

    return nothing
end

"""
    _restrict_level!(grid, level)

Restrict solution from level+1 to level for all blocks that have children.
"""
function _restrict_level!(grid::AMRGrid, level::Int)
    for block in values(grid.blocks)
        if block.level == level && !block.active && !isempty(block.child_ids)
            children = [grid.blocks[cid] for cid in block.child_ids if haskey(grid.blocks, cid)]
            if !isempty(children) && all(c -> c.active, children)
                restrict!(block, children)
            end
        end
    end
    return nothing
end

# ============================================================
# Inter-block ghost exchange (2D)
# ============================================================

"""
    _amr_dim(grid::AMRGrid) -> Int

Spatial dimension of the AMR grid.
"""
_amr_dim(grid::AMRGrid) = length(grid.block_size)

"""
    _amr_root(grid::AMRGrid) -> Union{AMRBlock, Nothing}

Return the root block (`parent_id == -1`), or `nothing` if absent.
"""
function _amr_root(grid::AMRGrid)
    for b in values(grid.blocks)
        b.parent_id == -1 && return b
    end
    return nothing
end

"""
    _amr_leaf_at(grid, x, y) -> Union{AMRBlock, Nothing}

Return the active leaf block whose physical extent contains the point
`(x, y)`, walking the block tree from the root. Returns `nothing` when
the point lies outside the domain. Points exactly on internal block
boundaries are assigned to the block on the high side; sampling
positions used by the ghost exchange are cell centers, which never lie
on block boundaries for even block sizes.
"""
function _amr_leaf_at(grid::AMRGrid{N, FT, 2}, x, y) where {N, FT}
    b = _amr_root(grid)
    b === nothing && return nothing
    x0, y0 = b.origin
    x1 = x0 + b.dims[1] * b.dx[1]
    y1 = y0 + b.dims[2] * b.dx[2]
    (x0 <= x < x1 && y0 <= y < y1) || return nothing
    while !b.active
        next = nothing
        for cid in b.child_ids
            haskey(grid.blocks, cid) || continue
            c = grid.blocks[cid]
            cx0, cy0 = c.origin
            cx1 = cx0 + c.dims[1] * c.dx[1]
            cy1 = cy0 + c.dims[2] * c.dx[2]
            if cx0 <= x < cx1 && cy0 <= y < cy1
                next = c
                break
            end
        end
        next === nothing && return nothing
        b = next
    end
    return b
end

"""
    _amr_sample_state(grid, get_cell, x, y, hx, hy, lev, depth = 0)

Sample the leaf-block solution over the footprint of a level-`lev` cell
of size `(hx, hy)` centered at `(x, y)`. `get_cell(bid, i, j)` returns
the interior state of block `bid` at cell `(i, j)`.

- Same-level leaf: exact cell copy.
- Coarser leaf: piecewise-constant prolongation (containing-cell value).
- Finer leaf: conservative average of the four half-size sub-footprints
  (recursively, so arbitrary level jumps are averaged consistently).

Returns `nothing` when the point lies outside the domain.
"""
function _amr_sample_state(grid::AMRGrid{N, FT, 2}, get_cell::F, x, y, hx, hy, lev::Int, depth::Int = 0) where {N, FT, F}
    leaf = _amr_leaf_at(grid, x, y)
    leaf === nothing && return nothing
    if leaf.level <= lev || depth >= 8
        ci = clamp(Int(floor((x - leaf.origin[1]) / leaf.dx[1])) + 1, 1, leaf.dims[1])
        cj = clamp(Int(floor((y - leaf.origin[2]) / leaf.dx[2])) + 1, 1, leaf.dims[2])
        return get_cell(leaf.id, ci, cj)::SVector{N, FT}
    end
    qx = hx / 4
    qy = hy / 4
    acc = zero(SVector{N, FT})
    for (sx, sy) in ((-qx, -qy), (qx, -qy), (-qx, qy), (qx, qy))
        v = _amr_sample_state(grid, get_cell, x + sx, y + sy, hx / 2, hy / 2, lev + 1, depth + 1)
        v === nothing && return nothing
        acc = acc + v
    end
    return acc / 4
end

"""
    _amr_exchange_ghosts_2d!(U_pad, block, grid, get_cell)

Overwrite the side ghost cells of `U_pad` (a `(nx+4, ny+4)` padded array
for `block`) with data sampled from neighboring leaf blocks via
[`_amr_sample_state`](@ref). Ghost cells outside the domain (and the
four corner regions, which the dimension-split flux loops never read)
keep their previous zero-gradient values.
"""
function _amr_exchange_ghosts_2d!(U_pad::AbstractMatrix, block::AMRBlock{N, FT, 2}, grid::AMRGrid, get_cell::F) where {N, FT, F}
    nx, ny = block.dims
    hx, hy = block.dx
    ox, oy = block.origin
    lev = block.level
    for gj in 3:(ny + 2)
        y = oy + ((gj - 2) - FT(0.5)) * hy
        for gi in (1, 2, nx + 3, nx + 4)
            x = ox + ((gi - 2) - FT(0.5)) * hx
            v = _amr_sample_state(grid, get_cell, x, y, hx, hy, lev)
            v === nothing || (U_pad[gi, gj] = v)
        end
    end
    for gi in 3:(nx + 2)
        x = ox + ((gi - 2) - FT(0.5)) * hx
        for gj in (1, 2, ny + 3, ny + 4)
            y = oy + ((gj - 2) - FT(0.5)) * hy
            v = _amr_sample_state(grid, get_cell, x, y, hx, hy, lev)
            v === nothing || (U_pad[gi, gj] = v)
        end
    end
    return nothing
end

"""
    _amr_seam_flux_fix_2d!(dU_pad, U_pad, block, grid, get_pad, law, solver)

Method-of-lines flux correction at coarse-fine seams: for every face
cell of `block` whose across-face neighbor is exactly one level finer,
replace this (coarse) block's face flux contribution in `dU_pad` with
the area-average of the two fine-face fluxes computed from the fine
blocks' padded data (`get_pad(bid)`), making the instantaneous mass,
momentum, and energy transfer identical on both sides of the seam.

Same-level seams need no correction (both sides already compute the
identical Riemann flux from the same pair of states). Level jumps >= 2
are left uncorrected.
"""
function _amr_seam_flux_fix_2d!(
        dU_pad::AbstractMatrix, U_pad::AbstractMatrix, block::AMRBlock{N, FT, 2},
        grid::AMRGrid, get_pad::F, law, solver
    ) where {N, FT, F}
    nx, ny = block.dims
    hx, hy = block.dx
    ox, oy = block.origin
    lev = block.level

    # Each tuple: (face position, padded index of the face's low-side cell,
    # padded index of the interior cell to correct, sign of the flux
    # contribution into that cell). Left/bottom faces feed +F, right/top -F.
    # x-direction faces
    for (xf, ipadL, icell, sgn) in ((ox, 2, 3, +1), (ox + nx * hx, nx + 2, nx + 2, -1))
        xprobe = sgn > 0 ? xf - hx / 4 : xf + hx / 4
        for j in 1:ny
            y = oy + (j - FT(0.5)) * hy
            F_avg = _amr_fine_face_flux_avg(grid, get_pad, law, solver, xprobe, y, hy, lev, 1, sgn)
            F_avg === nothing && continue
            wL = conserved_to_primitive(law, U_pad[ipadL, j + 2])
            wR = conserved_to_primitive(law, U_pad[ipadL + 1, j + 2])
            F_c = solve_riemann(solver, law, wL, wR, 1)
            dU_pad[icell, j + 2] = dU_pad[icell, j + 2] + sgn * (F_avg - F_c) / hx
        end
    end

    # y-direction faces
    for (yf, jpadL, jcell, sgn) in ((oy, 2, 3, +1), (oy + ny * hy, ny + 2, ny + 2, -1))
        yprobe = sgn > 0 ? yf - hy / 4 : yf + hy / 4
        for i in 1:nx
            x = ox + (i - FT(0.5)) * hx
            F_avg = _amr_fine_face_flux_avg(grid, get_pad, law, solver, x, yprobe, hx, lev, 2, sgn)
            F_avg === nothing && continue
            wL = conserved_to_primitive(law, U_pad[i + 2, jpadL])
            wR = conserved_to_primitive(law, U_pad[i + 2, jpadL + 1])
            F_c = solve_riemann(solver, law, wL, wR, 2)
            dU_pad[i + 2, jcell] = dU_pad[i + 2, jcell] + sgn * (F_avg - F_c) / hy
        end
    end

    return nothing
end

"""
    _amr_fine_face_flux_avg(grid, get_pad, law, solver, xp, yp, h_tan, lev, dir, sgn)

Average of the two fine-face fluxes covering one coarse face cell.

`(xp, yp)` is a probe location a quarter coarse-cell inside the
neighboring region on the fine side of the face; `h_tan` is the coarse
cell size tangential to the face; `dir` is the face-normal direction
(1 = x, 2 = y); `sgn > 0` means the fine region lies on the low side of
the coarse block (the fine faces are the fine blocks' high-side
boundary faces) and `sgn < 0` the converse.

Returns `nothing` unless both probe points land in leaves exactly one
level finer whose boundary coincides with the coarse face.
"""
function _amr_fine_face_flux_avg(
        grid::AMRGrid{N, FT, 2}, get_pad::F, law, solver,
        xp, yp, h_tan, lev::Int, dir::Int, sgn::Int
    ) where {N, FT, F}
    acc = zero(SVector{N, FT})
    for s in (-1, +1)
        x = dir == 1 ? xp : xp + s * h_tan / 4
        y = dir == 1 ? yp + s * h_tan / 4 : yp
        leaf = _amr_leaf_at(grid, x, y)
        (leaf === nothing || leaf.level != lev + 1) && return nothing
        fpad = get_pad(leaf.id)
        ci = Int(floor((x - leaf.origin[1]) / leaf.dx[1])) + 1
        cj = Int(floor((y - leaf.origin[2]) / leaf.dx[2])) + 1
        fnx, fny = leaf.dims
        if dir == 1
            # sgn > 0: fine region left of coarse block -> fine right-boundary face
            (sgn > 0 ? ci == fnx : ci == 1) || return nothing
            iface = sgn > 0 ? fnx + 2 : 2
            wL = conserved_to_primitive(law, fpad[iface, cj + 2])
            wR = conserved_to_primitive(law, fpad[iface + 1, cj + 2])
        else
            (sgn > 0 ? cj == fny : cj == 1) || return nothing
            jface = sgn > 0 ? fny + 2 : 2
            wL = conserved_to_primitive(law, fpad[ci + 2, jface])
            wR = conserved_to_primitive(law, fpad[ci + 2, jface + 1])
        end
        acc = acc + solve_riemann(solver, law, wL, wR, dir)
    end
    return acc / 2
end

"""
    _amr_global_step_2d!(prob::AMRProblem, dt)

Advance every active block of a 2D AMR grid by one synchronized forward
Euler step of size `dt`:

1. Build a ghost-padded array per block from pre-update data, with
   zero-gradient fill at the domain boundary and inter-block ghost
   exchange everywhere else.
2. Evaluate the first-order flux-differenced RHS per block
   (`_advance_block_rhs!`).
3. Apply the MOL seam-flux correction at single-level coarse-fine jumps
   (`_amr_seam_flux_fix_2d!`).
4. Update all block interiors.

This is the same semidiscretization as `ODEProblem(::AMRProblem)`.
"""
function _amr_global_step_2d!(prob::AMRProblem, dt)
    grid = prob.grid
    law = grid.law
    solver = prob.riemann_solver
    leaves = active_blocks(grid)
    isempty(leaves) && return nothing
    b1 = first(leaves)
    zero_state = zero(eltype(b1.U))

    pads = Dict{Int, typeof(b1.U)}()
    dUs = Dict{Int, typeof(b1.U)}()
    get_cell = (bid, i, j) -> grid.blocks[bid].U[i, j]

    for b in leaves
        nx, ny = b.dims
        pad = fill(zero_state, nx + 4, ny + 4)
        for j in 1:ny, i in 1:nx
            pad[i + 2, j + 2] = b.U[i, j]
        end
        _fill_amr_ghost_2d!(pad, nx, ny)
        _amr_exchange_ghosts_2d!(pad, b, grid, get_cell)
        pads[b.id] = pad
        dUs[b.id] = fill(zero_state, nx + 4, ny + 4)
    end

    for b in leaves
        _advance_block_rhs!(dUs[b.id], pads[b.id], b, law, solver)
    end

    get_pad = bid -> pads[bid]
    for b in leaves
        _amr_seam_flux_fix_2d!(dUs[b.id], pads[b.id], b, grid, get_pad, law, solver)
    end

    for b in leaves
        nx, ny = b.dims
        dU = dUs[b.id]
        for j in 1:ny, i in 1:nx
            b.U[i, j] = b.U[i, j] + dt * dU[i + 2, j + 2]
        end
    end

    return nothing
end

"""
    _amr_domain_bc_warn(context::AbstractString)

One-time honesty warning for the 2D AMR paths: inter-block ghost
exchange and single-level-jump seam-flux correction are performed, but
domain-boundary ghosts are zero-gradient regardless of
`prob.boundary_conditions`, and seams with level jumps >= 2 are not
flux-corrected.
"""
function _amr_domain_bc_warn(context::AbstractString)
    @warn "$context: domain-boundary ghost cells use zero-gradient extrapolation; " *
        "the problem's boundary_conditions are not applied at the domain boundary. " *
        "Coarse-fine seam fluxes are conservative for single-level jumps only." maxlog = 1
    return nothing
end

"""
    _amr_ghost_exchange_guard(grid, context::AbstractString)

Guard for the 3D AMR block advance, which still fills each block's
ghost cells by zero-gradient extrapolation from that block's own
interior (no inter-block exchange, no flux correction). Multi-block 3D
grids throw an `ArgumentError`; single-block 3D grids emit a one-time
warning. The 2D paths perform real ghost exchange and use
[`_amr_domain_bc_warn`](@ref) instead.
"""
function _amr_ghost_exchange_guard(grid, context::AbstractString)
    n_active = count(b -> b.active, values(grid.blocks))
    if n_active > 1
        throw(
            ArgumentError(
                "$context: the 3D AMR block advance fills ghost cells by zero-gradient " *
                    "extrapolation from each block's own interior — blocks never " *
                    "exchange ghost data and flux correction is never applied. With " *
                    "$n_active active blocks, waves cannot cross block boundaries " *
                    "and the results would be physically wrong. Use a single-block " *
                    "grid (no refinement), a 2D AMR grid (which has ghost exchange), " *
                    "or the structured solvers (HyperbolicProblem2D/HyperbolicProblem3D)."
            )
        )
    end
    @warn "$context: block ghost cells are filled by zero-gradient extrapolation " *
        "from the block's own interior; the problem's boundary conditions are " *
        "not applied at block boundaries and inter-block flux correction is " *
        "never performed. Results are only meaningful for a single block with " *
        "outflow-like boundaries." maxlog = 1
    return nothing
end

"""
    _amr_min_dt(prob::AMRProblem) -> FT

CFL time step minimized over ALL active blocks (finest-limited).
"""
function _amr_min_dt(prob::AMRProblem)
    grid = prob.grid
    law = grid.law
    dt_min = typemax(Float64)
    for block in values(grid.blocks)
        block.active || continue
        dt_min = min(dt_min, _compute_dt_block(block, law, prob.cfl))
    end
    return dt_min
end

"""
    solve_amr(prob::AMRProblem; method=:subcycling) -> (grid, t_final)

Solve an AMR problem with forward-Euler time integration.

For 2D grids, every step is a synchronized global step
([`_amr_global_step_2d!`](@ref)): all active blocks (any mix of levels)
exchange ghost data from pre-update neighbor data, are advanced with a
single finest-CFL-limited `dt` (no subcycling), and coarse-fine seam
fluxes at single-level jumps are flux-corrected so the scheme is
conservative there. This is the same semidiscretization as
`ODEProblem(::AMRProblem)`, so the two paths agree.

For 3D grids, the historical per-block zero-gradient path is used:
multi-block 3D grids throw an `ArgumentError` and single-block 3D
grids emit a one-time warning (see `_amr_ghost_exchange_guard`).

!!! warning
    Domain-boundary ghost cells use zero-gradient extrapolation; the
    problem's `boundary_conditions` field is not applied. Seam fluxes
    across level jumps >= 2 are not flux-corrected.

# Returns
- `grid`: The final AMR grid with solution data.
- `t_final`: The final time reached.
"""
function solve_amr(
        prob::AMRProblem;
        method::Symbol = :subcycling,
        callback::Union{Nothing, Function} = nothing,
    )
    _v2_api_depwarn(:solve_amr, "`solve(prob, alg; ...)` or `sciml_problem(prob)`")
    grid = prob.grid
    is_2d = _amr_dim(grid) == 2
    if is_2d
        _amr_domain_bc_warn("solve_amr")
    else
        _amr_ghost_exchange_guard(grid, "solve_amr")
    end
    t = prob.initial_time
    step = 0

    while t < prob.final_time - eps(typeof(t))
        dt = is_2d ? _amr_min_dt(prob) : compute_dt_amr(prob, 0)

        # Don't overshoot
        if t + dt > prob.final_time
            dt = prob.final_time - t
        end

        if dt <= zero(dt)
            break
        end

        if is_2d
            # Synchronized global step over all levels (no subcycling)
            _amr_global_step_2d!(prob, dt)
        else
            # Historical per-block advance with subcycling (3D)
            advance_level!(prob, 0, dt, t)
        end

        t += dt
        step += 1
        if callback !== nothing
            callback(prob.grid, t, step, dt)
        end

        # Regrid periodically
        if prob.regrid_interval > 0 && mod(step, prob.regrid_interval) == 0
            # Restrict to ensure coarse data is up to date before regridding
            max_lev = max_active_level(grid)
            for lev in (max_lev - 1):-1:0
                _restrict_level!(grid, lev)
            end

            regrid!(grid)

            # Prolongate new fine blocks from coarse data
            for block in values(grid.blocks)
                if !block.active && !isempty(block.child_ids)
                    children = [grid.blocks[cid] for cid in block.child_ids if haskey(grid.blocks, cid)]
                    if !isempty(children) && all(c -> c.active, children)
                        prolongate!(block, children, grid.law)
                    end
                end
            end

            if !is_2d
                # The 3D per-block advance cannot handle multiple blocks
                # (no ghost exchange); abort if regridding split the domain.
                _amr_ghost_exchange_guard(grid, "solve_amr (after regrid)")
            end
        end
    end

    return grid, t
end
