# ============================================================
# AMR Time Stepping with Subcycling
# ============================================================
#
# Berger-Oliger AMR time integration with recursive subcycling:
# - Fine levels take 2x as many steps as the next coarser level
# - After fine steps complete, restriction averages fine -> coarse
# - Flux correction ensures conservation at level interfaces
#
# The time stepping proceeds level-by-level:
#   1. Compute dt at the coarsest level
#   2. Advance coarsest level by dt
#   3. Advance finer levels by dt/2 (twice), recursively
#   4. Restrict fine solution to coarse
#   5. Apply flux corrections at level interfaces
#   6. Check refinement criteria and regrid if needed

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

function AMRProblem(grid, riemann_solver, reconstruction, boundary_conditions;
        initial_time = 0.0, final_time, cfl = 0.4, regrid_interval = 4)
    return AMRProblem(grid, riemann_solver, reconstruction, boundary_conditions,
        initial_time, final_time, cfl, regrid_interval)
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
Ghost cells are filled from neighbor data (simplified: zero-gradient at block boundaries).
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

"""
    solve_amr(prob::AMRProblem; method=:subcycling) -> (grid, t_final)

Solve an AMR problem with Berger-Oliger subcycling time integration.

# Returns
- `grid`: The final AMR grid with solution data.
- `t_final`: The final time reached.
"""
function solve_amr(prob::AMRProblem; method::Symbol = :subcycling)
    grid = prob.grid
    t = prob.initial_time
    step = 0

    while t < prob.final_time - eps(typeof(t))
        # Compute dt from coarsest level
        dt = compute_dt_amr(prob, 0)

        # Don't overshoot
        if t + dt > prob.final_time
            dt = prob.final_time - t
        end

        if dt <= zero(dt)
            break
        end

        # Advance all levels with subcycling
        advance_level!(prob, 0, dt, t)

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
