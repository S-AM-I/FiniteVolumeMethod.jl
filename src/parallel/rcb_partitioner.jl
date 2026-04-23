# parallel/rcb_partitioner.jl — Recursive coordinate bisection
#
# Stage 2b partitioner: deps-free geometric partitioning of an
# `UnstructuredFVMMesh`'s cells across a specified number of ranks. Used
# by the MPI extension to produce a per-cell `cell_to_rank::Vector{Int}`
# mapping without pulling in Metis.jl.
#
# Algorithm: at each level split the cell set along its longest
# bounding-box axis at the median coordinate of the cells' centers.
# Recurse until every cell is in a leaf bucket of size ~ ncells / nranks.
# Cells assigned to rank r get `cell_to_rank[c] = r` (0-based ranks).
#
# Works for arbitrary spatial dimension (Dim in 1, 2, 3) since the cell
# centers are already stored as `Dim × ncells` columns.

"""
    partition_rcb(mesh::UnstructuredFVMMesh{Dim, T}, nranks::Int) -> Vector{Int}

Partition `mesh`'s cells into `nranks` contiguous geometric buckets
using recursive coordinate bisection. Returns a length-`ncells` vector
`cell_to_rank` with entries in `0:nranks-1` (to match MPI's 0-based
rank convention).

Properties:
- Each bucket has size `ncells÷nranks` or `ncells÷nranks + 1`.
- Cells assigned to the same rank are geometrically clustered.
- Deterministic and deps-free; a future `partition_metis` will offer
  better load balance on meshes with poor geometric locality.
"""
function partition_rcb(mesh::UnstructuredFVMMesh{Dim, T}, nranks::Int) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nranks >= 1 || error("nranks must be >= 1, got $nranks")
    nc > 0 || error("mesh has zero cells")

    cell_to_rank = zeros(Int, nc)
    # Work on a permutation of cell indices; recursion fills in cell_to_rank.
    perm = collect(1:nc)
    _rcb_recurse!(cell_to_rank, perm, mesh.cell_centers, 0, nranks)
    return cell_to_rank
end

function _rcb_recurse!(
        cell_to_rank::Vector{Int},
        cells::Vector{Int},
        centers::AbstractMatrix,
        rank_offset::Int,
        nranks::Int,
    )
    if nranks == 1
        for c in cells
            cell_to_rank[c] = rank_offset
        end
        return nothing
    end

    # Split into `left_ranks` / `right_ranks` buckets so that split ratio
    # tracks rank counts (unequal nranks yield proportional splits).
    left_ranks = nranks ÷ 2
    right_ranks = nranks - left_ranks
    n = length(cells)
    left_size = round(Int, n * left_ranks / nranks)
    left_size = clamp(left_size, 1, n - 1)

    # Choose axis: longest bounding-box axis over the current cell subset.
    Dim = size(centers, 1)
    axis = 1
    best_extent = -Inf
    @inbounds for d in 1:Dim
        lo = Inf
        hi = -Inf
        for c in cells
            v = centers[d, c]
            if v < lo
                lo = v
            end
            if v > hi
                hi = v
            end
        end
        extent = hi - lo
        if extent > best_extent
            best_extent = extent
            axis = d
        end
    end

    # Partial sort along the chosen axis so `cells[1:left_size]` has
    # the smaller coordinates. `partialsort!` is O(n) average.
    partialsort!(cells, left_size; by = c -> centers[axis, c])
    left_cells = view(cells, 1:left_size)
    right_cells = view(cells, (left_size + 1):n)

    _rcb_recurse!(cell_to_rank, collect(left_cells), centers, rank_offset, left_ranks)
    _rcb_recurse!(
        cell_to_rank, collect(right_cells), centers, rank_offset + left_ranks, right_ranks,
    )
    return nothing
end
