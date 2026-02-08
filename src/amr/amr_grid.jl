# ============================================================
# Block-Structured AMR Grid (Berger-Oliger)
# ============================================================
#
# AMR is organized as a tree of blocks, each containing a uniform
# structured mesh. Refinement doubles resolution (ratio 2) in all
# directions. Each block stores its own solution data.
#
# Terminology:
#   - Level 0: coarsest (root) grid
#   - Level L: refined by factor 2^L relative to root
#   - Block: a rectangular patch at a given level with its own mesh
#   - Active: leaf blocks where computation occurs

"""
    AMRBlock{N, FT, Dim}

A single block in the AMR hierarchy.

# Fields
- `id::Int`: Unique block identifier.
- `level::Int`: Refinement level (0 = coarsest).
- `origin::NTuple{Dim, FT}`: Physical coordinates of the block's lower-left corner.
- `dims::NTuple{Dim, Int}`: Number of cells in each direction (nx, ny) or (nx, ny, nz).
- `dx::NTuple{Dim, FT}`: Cell sizes in each direction.
- `U::Array{SVector{N, FT}, Dim}`: Conserved variable data (interior only, no ghosts).
- `parent_id::Int`: ID of the parent block (-1 if root).
- `child_ids::Vector{Int}`: IDs of child blocks (empty if leaf).
- `active::Bool`: Whether this block is a leaf (active for computation).
- `neighbors::Dict{Symbol, Int}`: Neighbor block IDs keyed by face (:left, :right, etc.).
"""
mutable struct AMRBlock{N, FT, Dim}
    id::Int
    level::Int
    origin::NTuple{Dim, FT}
    dims::NTuple{Dim, Int}
    dx::NTuple{Dim, FT}
    U::Array{SVector{N, FT}, Dim}
    parent_id::Int
    child_ids::Vector{Int}
    active::Bool
    neighbors::Dict{Symbol, Int}
end

"""
    AMRBlock(id, level, origin, dims, dx, N, FT, Dim)

Create a zero-initialized AMR block.
"""
function AMRBlock(id::Int, level::Int, origin::NTuple{Dim, FT},
        dims::NTuple{Dim, Int}, dx::NTuple{Dim, FT},
        ::Val{N}) where {N, FT, Dim}
    zero_state = zero(SVector{N, FT})
    U = fill(zero_state, dims...)
    return AMRBlock{N, FT, Dim}(id, level, origin, dims, dx, U, -1, Int[], true, Dict{Symbol, Int}())
end

"""
    is_leaf(block::AMRBlock) -> Bool

Check if the block is a leaf (has no children).
"""
is_leaf(block::AMRBlock) = isempty(block.child_ids)

"""
    AbstractRefinementCriterion

Abstract supertype for refinement criteria that decide when to refine or coarsen.
"""
abstract type AbstractRefinementCriterion end

"""
    AMRGrid{N, FT, Dim, Law, Criterion}

The AMR grid managing the block hierarchy.

# Fields
- `blocks::Dict{Int, AMRBlock{N, FT, Dim}}`: All blocks keyed by ID.
- `max_level::Int`: Maximum allowed refinement level.
- `refinement_ratio::Int`: Refinement factor (always 2).
- `block_size::NTuple{Dim, Int}`: Number of cells per block in each direction.
- `law::Law`: The conservation law.
- `criterion::Criterion`: The refinement criterion.
- `next_id::Base.RefValue{Int}`: Counter for assigning block IDs.
"""
mutable struct AMRGrid{N, FT, Dim, Law, Criterion <: AbstractRefinementCriterion}
    blocks::Dict{Int, AMRBlock{N, FT, Dim}}
    max_level::Int
    refinement_ratio::Int
    block_size::NTuple{Dim, Int}
    law::Law
    criterion::Criterion
    next_id::Base.RefValue{Int}
end

"""
    AMRGrid(law, criterion, block_size, max_level, domain_lo, domain_hi, N_vars, FT)

Create an AMR grid with a single root block covering the entire domain.

# Arguments
- `law`: Conservation law.
- `criterion`: Refinement criterion.
- `block_size`: Number of cells per block in each direction.
- `max_level`: Maximum refinement level.
- `domain_lo`: Lower corner of the domain (tuple).
- `domain_hi`: Upper corner of the domain (tuple).
"""
function AMRGrid(law, criterion::AbstractRefinementCriterion,
        block_size::NTuple{Dim, Int}, max_level::Int,
        domain_lo::NTuple{Dim, FT}, domain_hi::NTuple{Dim, FT},
        ::Val{N}) where {N, FT, Dim}
    # Compute dx for root level
    dx = ntuple(d -> (domain_hi[d] - domain_lo[d]) / block_size[d], Val(Dim))

    # Create root block
    root = AMRBlock(1, 0, domain_lo, block_size, dx, Val(N))

    blocks = Dict{Int, AMRBlock{N, FT, Dim}}(1 => root)
    next_id = Ref(2)

    return AMRGrid{N, FT, Dim, typeof(law), typeof(criterion)}(
        blocks, max_level, 2, block_size, law, criterion, next_id
    )
end

"""
    active_blocks(grid::AMRGrid) -> Vector{AMRBlock}

Return all active (leaf) blocks in the grid, sorted by level.
"""
function active_blocks(grid::AMRGrid)
    result = [b for b in values(grid.blocks) if b.active]
    sort!(result; by = b -> b.level)
    return result
end

"""
    blocks_at_level(grid::AMRGrid, level::Int) -> Vector{AMRBlock}

Return all active blocks at a specific refinement level.
"""
function blocks_at_level(grid::AMRGrid, level::Int)
    return [b for b in values(grid.blocks) if b.active && b.level == level]
end

"""
    max_active_level(grid::AMRGrid) -> Int

Return the maximum refinement level among active blocks.
"""
function max_active_level(grid::AMRGrid)
    max_lev = 0
    for b in values(grid.blocks)
        if b.active
            max_lev = max(max_lev, b.level)
        end
    end
    return max_lev
end

"""
    _new_block_id!(grid::AMRGrid) -> Int

Generate a unique block ID.
"""
function _new_block_id!(grid::AMRGrid)
    id = grid.next_id[]
    grid.next_id[] += 1
    return id
end

"""
    block_cell_center(block::AMRBlock, indices...) -> NTuple{Dim, FT}

Return the physical coordinates of the cell center at the given indices.
"""
function block_cell_center(block::AMRBlock{N, FT, Dim}, indices::Vararg{Int, Dim}) where {N, FT, Dim}
    return ntuple(d -> block.origin[d] + (indices[d] - FT(0.5)) * block.dx[d], Val(Dim))
end
