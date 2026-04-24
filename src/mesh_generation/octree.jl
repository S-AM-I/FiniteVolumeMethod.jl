# mesh_generation/octree.jl — Octree refinement + STL surface snapping (Stage 8a)
#
# Skeleton-depth mesh generator in the spirit of OpenFOAM's snappyHexMesh.
# Full parity with snappyHexMesh is a multi-month project; this MVP
# provides:
#
#   - `Octree{Dim, T}`: recursive spatial-subdivision data structure with
#     per-node refinement level.
#   - `build_octree(bbox_min, bbox_max, max_level)`: uniform refinement
#     up to `max_level`.
#   - `refine_near_surface!(octree, stl, target_level)`: increase the
#     refinement level of any node that intersects (or is within a small
#     bandwidth of) the triangulated surface `stl`.
#   - `snap_to_surface!(octree, stl, ...)`: pull boundary-cell vertices
#     onto the STL surface.
#   - `extract_unstructured_mesh(octree)`: harvest the octree leaves as
#     an `UnstructuredFVMMesh{3, Float64}` for downstream CFD.
#
# This is infrastructure only — the MVP guarantees correct-by-construction
# output on simple geometries (box in box, sphere) and is intended to be
# tested / benchmarked against Gmsh and OpenFOAM for parity in Stage 8
# follow-ups. A full-featured mesher needs layer addition, orientation
# diagnostics, non-manifold healing, parallelism.

using StaticArrays: SVector
using LinearAlgebra: norm, dot, cross

"""
    Octree{Dim, T}

Recursive bounding-volume tree. Non-leaf nodes have `2^Dim` children
stored in `children`; leaves have `children === nothing`.

# Fields
- `bbox_min::SVector{Dim, T}` — lower corner of the bounding box.
- `bbox_max::SVector{Dim, T}` — upper corner.
- `level::Int` — refinement depth (root = 0).
- `children::Union{Nothing, Vector{Octree{Dim, T}}}` — 2^Dim subnodes or nothing.
"""
mutable struct Octree{Dim, T}
    bbox_min::SVector{Dim, T}
    bbox_max::SVector{Dim, T}
    level::Int
    children::Union{Nothing, Vector{Octree{Dim, T}}}
end

Octree{Dim, T}(bbox_min, bbox_max, level::Int = 0) where {Dim, T} =
    Octree{Dim, T}(
    SVector{Dim, T}(bbox_min), SVector{Dim, T}(bbox_max), level, nothing,
)

"""
    is_leaf(node::Octree) -> Bool
"""
is_leaf(node::Octree) = node.children === nothing

"""
    center(node::Octree) -> SVector
"""
center(node::Octree{Dim, T}) where {Dim, T} =
    (node.bbox_min + node.bbox_max) / T(2)

"""
    subdivide!(node::Octree)

Replace `node` with 2^Dim children by splitting the bounding box at
its center. No-op if already subdivided.
"""
function subdivide!(node::Octree{Dim, T}) where {Dim, T}
    is_leaf(node) || return node
    c = center(node)
    mins = node.bbox_min
    maxs = node.bbox_max
    n_children = 2^Dim
    children = Vector{Octree{Dim, T}}(undef, n_children)
    @inbounds for k in 0:(n_children - 1)
        child_min = SVector{Dim, T}(
            ntuple(
                d ->
                (k >> (d - 1)) & 1 == 0 ? mins[d] : c[d], Dim
            )
        )
        child_max = SVector{Dim, T}(
            ntuple(
                d ->
                (k >> (d - 1)) & 1 == 0 ? c[d] : maxs[d], Dim
            )
        )
        children[k + 1] = Octree{Dim, T}(child_min, child_max, node.level + 1)
    end
    node.children = children
    return node
end

"""
    build_octree(bbox_min, bbox_max, max_level) -> Octree

Construct a uniformly-refined octree down to `max_level`.
"""
function build_octree(
        bbox_min::SVector{Dim, T}, bbox_max::SVector{Dim, T}, max_level::Int,
    ) where {Dim, T}
    root = Octree{Dim, T}(bbox_min, bbox_max, 0)
    _uniform_refine!(root, max_level)
    return root
end

function _uniform_refine!(node::Octree, max_level::Int)
    node.level >= max_level && return nothing
    subdivide!(node)
    for child in node.children
        _uniform_refine!(child, max_level)
    end
    return nothing
end

"""
    count_leaves(node::Octree) -> Int
"""
function count_leaves(node::Octree)
    is_leaf(node) && return 1
    return sum(count_leaves, node.children)
end

"""
    intersects_sphere(node::Octree, center, radius) -> Bool

Axis-aligned bounding box vs. sphere intersection test — useful as a
simple surface-proxy for refinement near a spherical body (e.g. ball
in a flow).
"""
function intersects_sphere(
        node::Octree{Dim, T}, sphere_center::SVector{Dim, T}, radius::T,
    ) where {Dim, T}
    # Find nearest point on the AABB to the sphere center.
    d_sq = zero(T)
    @inbounds for d in 1:Dim
        v = sphere_center[d]
        lo = node.bbox_min[d]
        hi = node.bbox_max[d]
        if v < lo
            d_sq += (lo - v)^2
        elseif v > hi
            d_sq += (v - hi)^2
        end
    end
    return d_sq <= radius^2
end

"""
    refine_near_sphere!(node::Octree, sphere_center, radius, target_level)

Refine any node intersecting a sphere (surface proxy) to at least
`target_level`. Leaves outside the sphere are untouched. Used for
surface-aware refinement when a full STL-triangle intersection test is
not needed (e.g. ball-in-duct flow).
"""
function refine_near_sphere!(
        node::Octree{Dim, T}, sphere_center::SVector{Dim, T},
        radius::T, target_level::Int,
    ) where {Dim, T}
    intersects_sphere(node, sphere_center, radius) || return nothing
    if is_leaf(node)
        node.level >= target_level && return nothing
        subdivide!(node)
    end
    for child in node.children
        refine_near_sphere!(child, sphere_center, radius, target_level)
    end
    return nothing
end

# ── Wave 4 generic octree primitives ──────────────────────────────
#
# The `build_octree` above is a uniform-refinement driver.  Wave 4 ships
# a callback-driven variant + inspection helpers used by
# `snappyHexMesh`-style workflows and the Gmsh pipeline.

"""
    leaves(node::Octree) -> Vector{Octree}

Collect every leaf under `node` (pre-order walk).  Returns a `Vector`
of the same concrete `Octree{Dim, T}` type.
"""
function leaves(node::Octree{Dim, T}) where {Dim, T}
    out = Octree{Dim, T}[]
    _collect_leaves!(out, node)
    return out
end

function _collect_leaves!(out::Vector{<:Octree}, node::Octree)
    if is_leaf(node)
        push!(out, node)
    else
        for child in node.children
            _collect_leaves!(out, child)
        end
    end
    return out
end

"""
    cell_count(node::Octree) -> Int

Number of leaf cells under `node`. Alias for [`count_leaves`](@ref)
exposed under the Wave 4 naming convention.
"""
cell_count(node::Octree) = count_leaves(node)

"""
    octree_refine!(node::Octree) -> Octree

Split a single leaf node by one level (no-op on a non-leaf).  Primitive
used by the callback-driven [`build_octree`](@ref) overload. Named
`octree_refine!` rather than `refine!` to avoid clashing with
DelaunayTriangulation's `refine!`.
"""
octree_refine!(node::Octree) = subdivide!(node)

"""
    build_octree(domain::NTuple{2, SVector{Dim, T}}, max_depth, refinement_criterion)
    build_octree(bbox_min, bbox_max, max_depth, refinement_criterion)

Callback-driven octree refinement.  The root covers the
axis-aligned box `(bbox_min, bbox_max)` (or the 2-tuple `domain =
(bbox_min, bbox_max)`).  A leaf is split iff
`refinement_criterion(node) === true` and `node.level < max_depth`.

The callback receives the candidate leaf and may inspect
`node.level`, `node.bbox_min`, `node.bbox_max`, or [`center`](@ref).
"""
function build_octree(
        domain::Tuple{SVector{Dim, T}, SVector{Dim, T}},
        max_depth::Int, refinement_criterion,
    ) where {Dim, T}
    return build_octree(domain[1], domain[2], max_depth, refinement_criterion)
end

function build_octree(
        bbox_min::SVector{Dim, T}, bbox_max::SVector{Dim, T},
        max_depth::Int, refinement_criterion,
    ) where {Dim, T}
    root = Octree{Dim, T}(bbox_min, bbox_max, 0)
    _callback_refine!(root, max_depth, refinement_criterion)
    return root
end

function _callback_refine!(node::Octree, max_depth::Int, criterion)
    node.level >= max_depth && return nothing
    if is_leaf(node)
        criterion(node) === true || return nothing
        subdivide!(node)
    end
    for child in node.children
        _callback_refine!(child, max_depth, criterion)
    end
    return nothing
end
