# ============================================================
# AMR Refinement Criteria
# ============================================================
#
# Refinement criteria evaluate each block and decide whether
# it should be refined (split into children) or coarsened
# (merged back into parent).

"""
    GradientRefinement{FT} <: AbstractRefinementCriterion

Refinement based on the maximum gradient of a specified variable.
Refines when `max(|grad(var)|) * dx > refine_threshold`.
Coarsens when `max(|grad(var)|) * dx < coarsen_threshold`.

# Fields
- `variable_index::Int`: Index of the variable to monitor (in conserved vector).
- `refine_threshold::FT`: Threshold for refinement.
- `coarsen_threshold::FT`: Threshold for coarsening.
"""
struct GradientRefinement{FT} <: AbstractRefinementCriterion
    variable_index::Int
    refine_threshold::FT
    coarsen_threshold::FT
end

function GradientRefinement(;
        variable_index::Int = 1,
        refine_threshold = 0.1, coarsen_threshold = 0.01
    )
    return GradientRefinement(variable_index, refine_threshold, coarsen_threshold)
end

"""
    CurrentSheetRefinement{FT} <: AbstractRefinementCriterion

Refinement based on current sheet detection for MHD problems.
Refines when `|J| = |curl(B)| * dx > refine_threshold`.
Coarsens when `|J| * dx < coarsen_threshold`.

Designed for 2D/3D MHD where current sheets indicate magnetic reconnection
or sharp field gradients.

# Fields
- `refine_threshold::FT`: Threshold for refinement.
- `coarsen_threshold::FT`: Threshold for coarsening.
"""
struct CurrentSheetRefinement{FT} <: AbstractRefinementCriterion
    refine_threshold::FT
    coarsen_threshold::FT
end

function CurrentSheetRefinement(; refine_threshold = 0.1, coarsen_threshold = 0.01)
    return CurrentSheetRefinement(refine_threshold, coarsen_threshold)
end

"""
    needs_refinement(criterion, block, law) -> Bool

Evaluate whether a block needs refinement.
"""
function needs_refinement end

"""
    needs_coarsening(criterion, block, law) -> Bool

Evaluate whether a block can be coarsened.
"""
function needs_coarsening end

# ============================================================
# GradientRefinement evaluation
# ============================================================

function needs_refinement(crit::GradientRefinement, block::AMRBlock{N, FT, 2}, law) where {N, FT}
    U = block.U
    nx, ny = block.dims
    var_idx = crit.variable_index
    dx_min = min(block.dx...)

    max_grad = zero(FT)
    for j in 1:ny, i in 1:nx
        val = U[i, j][var_idx]
        # Check x-gradient
        if i < nx
            grad_x = abs(U[i + 1, j][var_idx] - val) / block.dx[1]
            max_grad = max(max_grad, grad_x * dx_min)
        end
        # Check y-gradient
        if j < ny
            grad_y = abs(U[i, j + 1][var_idx] - val) / block.dx[2]
            max_grad = max(max_grad, grad_y * dx_min)
        end
    end
    return max_grad > crit.refine_threshold
end

function needs_coarsening(crit::GradientRefinement, block::AMRBlock{N, FT, 2}, law) where {N, FT}
    U = block.U
    nx, ny = block.dims
    var_idx = crit.variable_index
    dx_min = min(block.dx...)

    max_grad = zero(FT)
    for j in 1:ny, i in 1:nx
        val = U[i, j][var_idx]
        if i < nx
            grad_x = abs(U[i + 1, j][var_idx] - val) / block.dx[1]
            max_grad = max(max_grad, grad_x * dx_min)
        end
        if j < ny
            grad_y = abs(U[i, j + 1][var_idx] - val) / block.dx[2]
            max_grad = max(max_grad, grad_y * dx_min)
        end
    end
    return max_grad < crit.coarsen_threshold
end

function needs_refinement(crit::GradientRefinement, block::AMRBlock{N, FT, 3}, law) where {N, FT}
    U = block.U
    nx, ny, nz = block.dims
    var_idx = crit.variable_index
    dx_min = min(block.dx...)

    max_grad = zero(FT)
    for k in 1:nz, j in 1:ny, i in 1:nx
        val = U[i, j, k][var_idx]
        if i < nx
            grad_x = abs(U[i + 1, j, k][var_idx] - val) / block.dx[1]
            max_grad = max(max_grad, grad_x * dx_min)
        end
        if j < ny
            grad_y = abs(U[i, j + 1, k][var_idx] - val) / block.dx[2]
            max_grad = max(max_grad, grad_y * dx_min)
        end
        if k < nz
            grad_z = abs(U[i, j, k + 1][var_idx] - val) / block.dx[3]
            max_grad = max(max_grad, grad_z * dx_min)
        end
    end
    return max_grad > crit.refine_threshold
end

function needs_coarsening(crit::GradientRefinement, block::AMRBlock{N, FT, 3}, law) where {N, FT}
    U = block.U
    nx, ny, nz = block.dims
    var_idx = crit.variable_index
    dx_min = min(block.dx...)

    max_grad = zero(FT)
    for k in 1:nz, j in 1:ny, i in 1:nx
        val = U[i, j, k][var_idx]
        if i < nx
            grad_x = abs(U[i + 1, j, k][var_idx] - val) / block.dx[1]
            max_grad = max(max_grad, grad_x * dx_min)
        end
        if j < ny
            grad_y = abs(U[i, j + 1, k][var_idx] - val) / block.dx[2]
            max_grad = max(max_grad, grad_y * dx_min)
        end
        if k < nz
            grad_z = abs(U[i, j, k + 1][var_idx] - val) / block.dx[3]
            max_grad = max(max_grad, grad_z * dx_min)
        end
    end
    return max_grad < crit.coarsen_threshold
end

# ============================================================
# CurrentSheetRefinement evaluation
# ============================================================

function needs_refinement(crit::CurrentSheetRefinement, block::AMRBlock{N, FT, 2}, law) where {N, FT}
    U = block.U
    nx, ny = block.dims
    dx_val, dy_val = block.dx

    # Jz = dBy/dx - dBx/dy (MHD: Bx=index 6, By=index 7 in conserved)
    max_J = zero(FT)
    for j in 2:(ny - 1), i in 2:(nx - 1)
        w = conserved_to_primitive(law, U[i, j])
        w_xp = conserved_to_primitive(law, U[i + 1, j])
        w_xm = conserved_to_primitive(law, U[i - 1, j])
        w_yp = conserved_to_primitive(law, U[i, j + 1])
        w_ym = conserved_to_primitive(law, U[i, j - 1])

        dBy_dx = (w_xp[7] - w_xm[7]) / (2 * dx_val)
        dBx_dy = (w_yp[6] - w_ym[6]) / (2 * dy_val)
        Jz = abs(dBy_dx - dBx_dy) * min(dx_val, dy_val)
        max_J = max(max_J, Jz)
    end
    return max_J > crit.refine_threshold
end

function needs_coarsening(crit::CurrentSheetRefinement, block::AMRBlock{N, FT, 2}, law) where {N, FT}
    U = block.U
    nx, ny = block.dims
    dx_val, dy_val = block.dx

    max_J = zero(FT)
    for j in 2:(ny - 1), i in 2:(nx - 1)
        w_xp = conserved_to_primitive(law, U[i + 1, j])
        w_xm = conserved_to_primitive(law, U[i - 1, j])
        w_yp = conserved_to_primitive(law, U[i, j + 1])
        w_ym = conserved_to_primitive(law, U[i, j - 1])

        dBy_dx = (w_xp[7] - w_xm[7]) / (2 * dx_val)
        dBx_dy = (w_yp[6] - w_ym[6]) / (2 * dy_val)
        Jz = abs(dBy_dx - dBx_dy) * min(dx_val, dy_val)
        max_J = max(max_J, Jz)
    end
    return max_J < crit.coarsen_threshold
end

function needs_refinement(crit::CurrentSheetRefinement, block::AMRBlock{N, FT, 3}, law) where {N, FT}
    U = block.U
    nx, ny, nz = block.dims
    dx_val, dy_val, dz_val = block.dx
    dx_min = min(dx_val, dy_val, dz_val)

    # |J| = |curl(B)| with components:
    # Jx = dBz/dy - dBy/dz, Jy = dBx/dz - dBz/dx, Jz = dBy/dx - dBx/dy
    # B in primitive: Bx=6, By=7, Bz=8
    max_J = zero(FT)
    for k in 2:(nz - 1), j in 2:(ny - 1), i in 2:(nx - 1)
        w_xp = conserved_to_primitive(law, U[i + 1, j, k])
        w_xm = conserved_to_primitive(law, U[i - 1, j, k])
        w_yp = conserved_to_primitive(law, U[i, j + 1, k])
        w_ym = conserved_to_primitive(law, U[i, j - 1, k])
        w_zp = conserved_to_primitive(law, U[i, j, k + 1])
        w_zm = conserved_to_primitive(law, U[i, j, k - 1])

        dBx_dy = (w_yp[6] - w_ym[6]) / (2 * dy_val)
        dBx_dz = (w_zp[6] - w_zm[6]) / (2 * dz_val)
        dBy_dx = (w_xp[7] - w_xm[7]) / (2 * dx_val)
        dBy_dz = (w_zp[7] - w_zm[7]) / (2 * dz_val)
        dBz_dx = (w_xp[8] - w_xm[8]) / (2 * dx_val)
        dBz_dy = (w_yp[8] - w_ym[8]) / (2 * dy_val)

        Jx = dBz_dy - dBy_dz
        Jy = dBx_dz - dBz_dx
        Jz = dBy_dx - dBx_dy
        J_mag = sqrt(Jx^2 + Jy^2 + Jz^2) * dx_min
        max_J = max(max_J, J_mag)
    end
    return max_J > crit.refine_threshold
end

function needs_coarsening(crit::CurrentSheetRefinement, block::AMRBlock{N, FT, 3}, law) where {N, FT}
    U = block.U
    nx, ny, nz = block.dims
    dx_val, dy_val, dz_val = block.dx
    dx_min = min(dx_val, dy_val, dz_val)

    max_J = zero(FT)
    for k in 2:(nz - 1), j in 2:(ny - 1), i in 2:(nx - 1)
        w_xp = conserved_to_primitive(law, U[i + 1, j, k])
        w_xm = conserved_to_primitive(law, U[i - 1, j, k])
        w_yp = conserved_to_primitive(law, U[i, j + 1, k])
        w_ym = conserved_to_primitive(law, U[i, j - 1, k])
        w_zp = conserved_to_primitive(law, U[i, j, k + 1])
        w_zm = conserved_to_primitive(law, U[i, j, k - 1])

        dBx_dy = (w_yp[6] - w_ym[6]) / (2 * dy_val)
        dBx_dz = (w_zp[6] - w_zm[6]) / (2 * dz_val)
        dBy_dx = (w_xp[7] - w_xm[7]) / (2 * dx_val)
        dBy_dz = (w_zp[7] - w_zm[7]) / (2 * dz_val)
        dBz_dx = (w_xp[8] - w_xm[8]) / (2 * dx_val)
        dBz_dy = (w_yp[8] - w_ym[8]) / (2 * dy_val)

        Jx = dBz_dy - dBy_dz
        Jy = dBx_dz - dBz_dx
        Jz = dBy_dx - dBx_dy
        J_mag = sqrt(Jx^2 + Jy^2 + Jz^2) * dx_min
        max_J = max(max_J, J_mag)
    end
    return max_J < crit.coarsen_threshold
end

# ============================================================
# Refinement / Coarsening Operations
# ============================================================

"""
    refine_block!(grid::AMRGrid, block_id::Int) -> Vector{Int}

Refine a block by splitting it into 2^Dim child blocks.
Returns the IDs of the new child blocks.
The parent block is deactivated.
"""
function refine_block!(grid::AMRGrid{N, FT, Dim}, block_id::Int) where {N, FT, Dim}
    parent = grid.blocks[block_id]

    if !parent.active
        return Int[]
    end
    if parent.level >= grid.max_level
        return Int[]
    end

    n_children = 2^Dim
    child_ids = Int[]
    child_dims = parent.dims
    child_dx = ntuple(d -> parent.dx[d] / 2, Val(Dim))

    # Create children: 2 per dimension
    for ci in 0:(n_children - 1)
        child_id = _new_block_id!(grid)
        push!(child_ids, child_id)

        # Compute offset for this child
        offset = ntuple(d -> ((ci >> (d - 1)) & 1), Val(Dim))
        child_origin = ntuple(d -> parent.origin[d] + offset[d] * parent.dims[d] * child_dx[d], Val(Dim))

        child = AMRBlock(child_id, parent.level + 1, child_origin, child_dims, child_dx, Val(N))
        child.parent_id = block_id

        grid.blocks[child_id] = child
    end

    parent.child_ids = child_ids
    parent.active = false

    return child_ids
end

"""
    coarsen_block!(grid::AMRGrid, block_id::Int) -> Bool

Coarsen a block by removing all its children and reactivating it.
Returns true if coarsening was performed.
All children must be active leaves for coarsening to proceed.
"""
function coarsen_block!(grid::AMRGrid, block_id::Int)
    parent = grid.blocks[block_id]

    if parent.active
        return false
    end
    if isempty(parent.child_ids)
        return false
    end

    # Check all children are active leaves
    for cid in parent.child_ids
        child = grid.blocks[cid]
        if !child.active || !isempty(child.child_ids)
            return false
        end
    end

    # Remove children
    for cid in parent.child_ids
        delete!(grid.blocks, cid)
    end

    parent.child_ids = Int[]
    parent.active = true

    return true
end

"""
    regrid!(grid::AMRGrid)

Evaluate refinement criteria on all active blocks and perform refinement/coarsening.
"""
function regrid!(grid::AMRGrid)
    law = grid.law
    crit = grid.criterion

    # Collect blocks to refine
    to_refine = Int[]
    for block in active_blocks(grid)
        if block.level < grid.max_level && needs_refinement(crit, block, law)
            push!(to_refine, block.id)
        end
    end

    # Refine blocks
    refined_set = Set{Int}()
    for bid in to_refine
        refine_block!(grid, bid)
        push!(refined_set, bid)
    end

    # Collect blocks to coarsen (check parents whose children all want coarsening)
    # Skip parents that were just refined in this regrid! call
    to_coarsen = Int[]
    for block in values(grid.blocks)
        if !block.active && !isempty(block.child_ids) && !(block.id in refined_set)
            all_coarsen = true
            for cid in block.child_ids
                child = grid.blocks[cid]
                if !child.active || !needs_coarsening(crit, child, law)
                    all_coarsen = false
                    break
                end
            end
            if all_coarsen
                push!(to_coarsen, block.id)
            end
        end
    end

    # Coarsen blocks
    for bid in to_coarsen
        coarsen_block!(grid, bid)
    end

    return nothing
end
