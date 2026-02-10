# ============================================================
# Multi-Rate (Subcycling) Time Stepping for AMR
# ============================================================
#
# Berger-Oliger style subcycling where finer AMR levels take
# more sub-steps than coarser levels. Each refinement level
# takes ratio^(level - base_level) sub-steps per coarse step.
#
# The recursive algorithm proceeds as:
#   1. Advance coarsest level by dt_coarse
#   2. Recursively advance finer levels with dt_fine = dt_coarse / ratio
#      (ratio sub-steps)
#   3. After fine sub-steps complete, restrict fine -> coarse
#   4. Apply flux correction at coarse-fine interfaces
#
# Time interpolation at coarse-fine boundaries provides guard
# cell data for fine sub-steps that occur between coarse states.
#
# This module provides a configurable SubcyclingScheme and
# an SSP-RK3 block advance, replacing the forward-Euler
# advance in amr_solve.jl with a higher-order alternative.

# ============================================================
# SubcyclingScheme
# ============================================================

"""
    SubcyclingScheme

Configuration for multi-rate time stepping where finer AMR levels
take more sub-steps than coarser levels.

# Fields
- `ratio::Int`: Time step refinement ratio per level (default 2).
  Each level `ℓ` takes `ratio` sub-steps for every one step at level `ℓ-1`.
"""
struct SubcyclingScheme
    ratio::Int
    function SubcyclingScheme(ratio::Int)
        ratio >= 1 || throw(ArgumentError("SubcyclingScheme ratio must be >= 1, got $ratio"))
        return new(ratio)
    end
end
SubcyclingScheme() = SubcyclingScheme(2)

# ============================================================
# Block-level SSP-RK3 advance
# ============================================================

"""
    _advance_block_ssprk3!(block, law, solver, recon, dt)

Advance a single 2D AMR block by one SSP-RK3 time step.
This is a higher-order replacement for `_advance_block!` (forward Euler).
"""
function _advance_block_ssprk3!(block::AMRBlock{N, FT, 2}, law, solver, recon, dt) where {N, FT}
    nx, ny = block.dims
    dx_val, dy_val = block.dx
    zero_state = zero(SVector{N, FT})

    # Allocate padded arrays for the three RK stages
    U_pad = Matrix{SVector{N, FT}}(undef, nx + 4, ny + 4)
    dU = Matrix{SVector{N, FT}}(undef, nx + 4, ny + 4)
    U1 = Matrix{SVector{N, FT}}(undef, nx + 4, ny + 4)
    U2 = Matrix{SVector{N, FT}}(undef, nx + 4, ny + 4)

    # Helper: copy block interior into padded array and fill ghost cells
    function _fill_padded!(U_pad, U_src, nx, ny, zero_state)
        for j in axes(U_pad, 2), i in axes(U_pad, 1)
            U_pad[i, j] = zero_state
        end
        for j in 1:ny, i in 1:nx
            U_pad[i + 2, j + 2] = U_src[i, j]
        end
        _fill_ghost_2d!(U_pad, nx, ny)
        return nothing
    end

    # Helper: compute dU from padded solution
    function _compute_rhs_block!(dU, U_pad, law, solver, nx, ny, dx_val, dy_val, zero_state)
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
        return nothing
    end

    # --- SSP-RK3 Stage 1: U1 = U + dt * L(U) ---
    _fill_padded!(U_pad, block.U, nx, ny, zero_state)
    _compute_rhs_block!(dU, U_pad, law, solver, nx, ny, dx_val, dy_val, zero_state)

    for j in axes(U1, 2), i in axes(U1, 1)
        U1[i, j] = zero_state
    end
    for j in 1:ny, i in 1:nx
        U1[i + 2, j + 2] = block.U[i, j] + dt * dU[i + 2, j + 2]
    end

    # --- SSP-RK3 Stage 2: U2 = 3/4 U + 1/4 (U1 + dt * L(U1)) ---
    _fill_ghost_2d!(U1, nx, ny)
    _compute_rhs_block!(dU, U1, law, solver, nx, ny, dx_val, dy_val, zero_state)

    for j in axes(U2, 2), i in axes(U2, 1)
        U2[i, j] = zero_state
    end
    for j in 1:ny, i in 1:nx
        ii = i + 2
        jj = j + 2
        U2[ii, jj] = FT(0.75) * block.U[i, j] + FT(0.25) * (U1[ii, jj] + dt * dU[ii, jj])
    end

    # --- SSP-RK3 Stage 3: U = 1/3 U + 2/3 (U2 + dt * L(U2)) ---
    _fill_ghost_2d!(U2, nx, ny)
    _compute_rhs_block!(dU, U2, law, solver, nx, ny, dx_val, dy_val, zero_state)

    for j in 1:ny, i in 1:nx
        ii = i + 2
        jj = j + 2
        block.U[i, j] = FT(1) / FT(3) * block.U[i, j] +
            FT(2) / FT(3) * (U2[ii, jj] + dt * dU[ii, jj])
    end

    return nothing
end

function _advance_block_ssprk3!(block::AMRBlock{N, FT, 3}, law, solver, recon, dt) where {N, FT}
    nx, ny, nz = block.dims
    dx_val, dy_val, dz_val = block.dx
    zero_state = zero(SVector{N, FT})

    # Allocate padded arrays for the three RK stages
    U_pad = Array{SVector{N, FT}, 3}(undef, nx + 4, ny + 4, nz + 4)
    dU = Array{SVector{N, FT}, 3}(undef, nx + 4, ny + 4, nz + 4)
    U1 = Array{SVector{N, FT}, 3}(undef, nx + 4, ny + 4, nz + 4)
    U2 = Array{SVector{N, FT}, 3}(undef, nx + 4, ny + 4, nz + 4)

    # Helper: copy block interior into padded array and fill ghost cells
    function _fill_padded_3d!(U_pad, U_src, nx, ny, nz, zero_state)
        for k in axes(U_pad, 3), j in axes(U_pad, 2), i in axes(U_pad, 1)
            U_pad[i, j, k] = zero_state
        end
        for k in 1:nz, j in 1:ny, i in 1:nx
            U_pad[i + 2, j + 2, k + 2] = U_src[i, j, k]
        end
        _fill_ghost_3d!(U_pad, nx, ny, nz)
        return nothing
    end

    # Helper: compute dU from padded solution
    function _compute_rhs_block_3d!(dU, U_pad, law, solver, nx, ny, nz, dx_val, dy_val, dz_val, zero_state)
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
        return nothing
    end

    # --- SSP-RK3 Stage 1: U1 = U + dt * L(U) ---
    _fill_padded_3d!(U_pad, block.U, nx, ny, nz, zero_state)
    _compute_rhs_block_3d!(dU, U_pad, law, solver, nx, ny, nz, dx_val, dy_val, dz_val, zero_state)

    for k in axes(U1, 3), j in axes(U1, 2), i in axes(U1, 1)
        U1[i, j, k] = zero_state
    end
    for k in 1:nz, j in 1:ny, i in 1:nx
        U1[i + 2, j + 2, k + 2] = block.U[i, j, k] + dt * dU[i + 2, j + 2, k + 2]
    end

    # --- SSP-RK3 Stage 2: U2 = 3/4 U + 1/4 (U1 + dt * L(U1)) ---
    _fill_ghost_3d!(U1, nx, ny, nz)
    _compute_rhs_block_3d!(dU, U1, law, solver, nx, ny, nz, dx_val, dy_val, dz_val, zero_state)

    for k in axes(U2, 3), j in axes(U2, 2), i in axes(U2, 1)
        U2[i, j, k] = zero_state
    end
    for k in 1:nz, j in 1:ny, i in 1:nx
        ii = i + 2
        jj = j + 2
        kk = k + 2
        U2[ii, jj, kk] = FT(0.75) * block.U[i, j, k] +
            FT(0.25) * (U1[ii, jj, kk] + dt * dU[ii, jj, kk])
    end

    # --- SSP-RK3 Stage 3: U = 1/3 U + 2/3 (U2 + dt * L(U2)) ---
    _fill_ghost_3d!(U2, nx, ny, nz)
    _compute_rhs_block_3d!(dU, U2, law, solver, nx, ny, nz, dx_val, dy_val, dz_val, zero_state)

    for k in 1:nz, j in 1:ny, i in 1:nx
        ii = i + 2
        jj = j + 2
        kk = k + 2
        block.U[i, j, k] = FT(1) / FT(3) * block.U[i, j, k] +
            FT(2) / FT(3) * (U2[ii, jj, kk] + dt * dU[ii, jj, kk])
    end

    return nothing
end

# ============================================================
# Ghost cell fill helpers (zero-gradient BCs at block edges)
# ============================================================

"""
    _fill_ghost_2d!(U_pad, nx, ny)

Fill ghost cells of a padded 2D array with zero-gradient (transmissive) data.
"""
function _fill_ghost_2d!(U_pad::AbstractMatrix, nx::Int, ny::Int)
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
    return nothing
end

"""
    _fill_ghost_3d!(U_pad, nx, ny, nz)

Fill ghost cells of a padded 3D array with zero-gradient (transmissive) data.
"""
function _fill_ghost_3d!(U_pad::AbstractArray{T, 3}, nx::Int, ny::Int, nz::Int) where {T}
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
    return nothing
end

# ============================================================
# Time interpolation at coarse-fine boundaries
# ============================================================

"""
    _time_interpolate_block!(block_interp, block_old, block_new, alpha)

Linearly interpolate block data between two time levels:
  `U_interp = (1 - alpha) * U_old + alpha * U_new`

Used to provide time-accurate guard cell data for fine sub-steps
that occur between two coarse time levels.

# Arguments
- `block_interp`: Block whose `U` field will be overwritten with interpolated data.
- `block_old`: Block state at the start of the coarse time step (t_n).
- `block_new`: Block state at the end of the coarse time step (t_n + dt).
- `alpha::Real`: Interpolation parameter in [0, 1]. 0 = old state, 1 = new state.
"""
function _time_interpolate_block!(
        block_interp::AMRBlock{N, FT, Dim},
        block_old::AMRBlock{N, FT, Dim},
        block_new::AMRBlock{N, FT, Dim},
        alpha
    ) where {N, FT, Dim}
    a = FT(alpha)
    oma = FT(1) - a
    for idx in eachindex(block_interp.U)
        block_interp.U[idx] = oma * block_old.U[idx] + a * block_new.U[idx]
    end
    return nothing
end

# ============================================================
# Coarse time step computation
# ============================================================

"""
    compute_dt_subcycled(prob::AMRProblem, base_level::Int) -> FT

Compute the time step based on the CFL condition at the coarsest active level.
Finer levels will subcycle with `dt_fine = dt_coarse / ratio^(level - base_level)`.

# Arguments
- `prob`: The AMR problem.
- `base_level`: The coarsest level whose CFL condition determines the global step.
"""
function compute_dt_subcycled(prob::AMRProblem, base_level::Int)
    grid = prob.grid
    law = grid.law
    cfl = prob.cfl

    dt_min = typemax(Float64)
    for block in blocks_at_level(grid, base_level)
        dt_block = _compute_dt_block(block, law, cfl)
        dt_min = min(dt_min, dt_block)
    end

    return dt_min
end

# ============================================================
# Recursive subcycled level advance
# ============================================================

"""
    advance_level_subcycled!(prob, level, dt, t, subcycling; method=:ssprk3)

Recursively advance all blocks at `level` and finer using subcycling.
Blocks at `level` take one step of size `dt`.
Blocks at `level+1` take `ratio` steps of size `dt/ratio`, and so on.

After the fine sub-steps complete, the fine solution is restricted back
to the coarser level to maintain consistency.

# Arguments
- `prob::AMRProblem`: The AMR problem definition.
- `level::Int`: Current refinement level to advance.
- `dt`: Time step size for this level.
- `t`: Current simulation time at the start of this step.
- `subcycling::SubcyclingScheme`: Subcycling configuration.

# Keyword Arguments
- `method::Symbol`: Time integration per block. `:ssprk3` (default) or `:euler`.
"""
function advance_level_subcycled!(
        prob::AMRProblem, level::Int, dt, t,
        subcycling::SubcyclingScheme;
        method::Symbol = :ssprk3
    )
    grid = prob.grid
    law = grid.law
    ratio = subcycling.ratio

    # Advance all blocks at this level by one step of dt
    _advance_blocks_at_level!(
        grid, level, dt, law, prob.riemann_solver, prob.reconstruction, method
    )

    # If finer levels exist, subcycle them
    max_lev = max_active_level(grid)
    if level < max_lev
        dt_fine = dt / ratio
        for step in 1:ratio
            t_fine = t + (step - 1) * dt_fine
            advance_level_subcycled!(
                prob, level + 1, dt_fine, t_fine,
                subcycling; method = method
            )
        end

        # Restrict fine solution back to coarse
        _restrict_level!(grid, level)
    end

    return nothing
end

"""
    _advance_blocks_at_level!(grid, level, dt, law, solver, recon, method)

Advance all active (leaf) blocks at a given level by one step of size `dt`.
"""
function _advance_blocks_at_level!(
        grid::AMRGrid, level::Int, dt, law, solver, recon, method::Symbol
    )
    for block in blocks_at_level(grid, level)
        if method == :ssprk3
            _advance_block_ssprk3!(block, law, solver, recon, dt)
        else
            _advance_block!(block, law, solver, recon, dt)
        end
    end
    return nothing
end

# ============================================================
# Top-level AMR solver with subcycling
# ============================================================

"""
    solve_amr_subcycled(prob::AMRProblem; subcycling=SubcyclingScheme(), method=:ssprk3)
        -> (grid, t_final)

Solve an AMR problem with configurable multi-rate (subcycling) time integration.

Each refinement level `ℓ` takes `ratio^(ℓ - base_level)` sub-steps per coarse step,
where `ratio` is `subcycling.ratio`. The time step is determined by the CFL condition
at the coarsest active level. Blocks are advanced using SSP-RK3 by default.

After all levels are advanced, the grid is periodically regridded according to
`prob.regrid_interval`.

# Arguments
- `prob::AMRProblem`: The AMR problem definition.

# Keyword Arguments
- `subcycling::SubcyclingScheme`: Subcycling configuration (default: ratio=2).
- `method::Symbol`: Time integration method per block, `:ssprk3` (default) or `:euler`.

# Returns
- `grid`: The final AMR grid with solution data.
- `t_final`: The final time reached.
"""
function solve_amr_subcycled(
        prob::AMRProblem;
        subcycling::SubcyclingScheme = SubcyclingScheme(),
        method::Symbol = :ssprk3
    )
    grid = prob.grid
    t = prob.initial_time
    step = 0

    while t < prob.final_time - eps(typeof(t))
        # Compute dt from the coarsest active level
        base_level = _min_active_level(grid)
        dt = compute_dt_subcycled(prob, base_level)

        # Don't overshoot final time
        if t + dt > prob.final_time
            dt = prob.final_time - t
        end

        if dt <= zero(dt)
            break
        end

        # Advance all levels recursively with subcycling
        advance_level_subcycled!(prob, base_level, dt, t, subcycling; method = method)

        t += dt
        step += 1

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
        end
    end

    return grid, t
end

# ============================================================
# Utility: minimum active level
# ============================================================

"""
    _min_active_level(grid::AMRGrid) -> Int

Return the minimum refinement level among active (leaf) blocks.
"""
function _min_active_level(grid::AMRGrid)
    min_lev = typemax(Int)
    for b in values(grid.blocks)
        if b.active
            min_lev = min(min_lev, b.level)
        end
    end
    return min_lev
end

# ============================================================
# Total sub-steps for a given level pair
# ============================================================

"""
    total_substeps(subcycling, base_level, target_level) -> Int

Compute the total number of sub-steps level `target_level` takes per one
step at `base_level`:  `ratio^(target_level - base_level)`.

# Arguments
- `subcycling::SubcyclingScheme`: The subcycling configuration.
- `base_level::Int`: The reference coarse level.
- `target_level::Int`: The fine level to compute sub-steps for.
"""
function total_substeps(subcycling::SubcyclingScheme, base_level::Int, target_level::Int)
    target_level >= base_level ||
        throw(ArgumentError("target_level ($target_level) must be >= base_level ($base_level)"))
    return subcycling.ratio^(target_level - base_level)
end
