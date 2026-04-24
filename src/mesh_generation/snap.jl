# mesh_generation/snap.jl — triangle-box intersection + vertex projection
# onto an STL triangle soup (v3.1 surface-snap step).
#
# Two public primitives:
#
#   * `triangle_intersects_aabb(v0, v1, v2, bbox_min, bbox_max)` — SAT test
#     used by `build_castellated_mesh` to decide whether an octree cell
#     sits on the STL surface.
#   * `snap_to_surface!(octree, vertices, faces; max_iters, tol, damping)`
#     — iterative vertex projection onto the nearest STL face for every
#     octree leaf within half-a-cell of the surface.  Fixed-point
#     iteration with under-relaxation.
#
# The point-triangle projection is the standard barycentric / edge-clamp
# algorithm (Eberly, "Distance Between a Point and a Triangle", 1999).

using StaticArrays: SVector
using LinearAlgebra: dot, norm

# ── Triangle / axis-aligned-box intersection (Akenine-Möller SAT) ─────

"""
    triangle_intersects_aabb(v0, v1, v2, bbox_min, bbox_max) -> Bool

Separating-axis test between the triangle `(v0, v1, v2)` and the
axis-aligned box `(bbox_min, bbox_max)`.  Based on Akenine-Möller's
"Fast 3D Triangle-Box Overlap Testing" (2001).  Returns `true` if the
triangle and box overlap (touching counts).
"""
function triangle_intersects_aabb(
        v0::SVector{3, T}, v1::SVector{3, T}, v2::SVector{3, T},
        bbox_min::SVector{3, T}, bbox_max::SVector{3, T},
    ) where {T <: AbstractFloat}
    # Translate triangle into box-centred frame.
    c = (bbox_min + bbox_max) / T(2)
    h = (bbox_max - bbox_min) / T(2)
    p0 = v0 - c
    p1 = v1 - c
    p2 = v2 - c

    # 1. Bounding-box check along coordinate axes.
    for d in 1:3
        mn = min(p0[d], p1[d], p2[d])
        mx = max(p0[d], p1[d], p2[d])
        if mn > h[d] || mx < -h[d]
            return false
        end
    end

    # 2. Triangle-plane vs. box.
    e0 = p1 - p0
    e1 = p2 - p1
    normal = SVector{3, T}(
        e0[2] * e1[3] - e0[3] * e1[2],
        e0[3] * e1[1] - e0[1] * e1[3],
        e0[1] * e1[2] - e0[2] * e1[1],
    )
    if !_plane_box_overlap(normal, p0, h)
        return false
    end

    # 3. Nine edge-cross-axis tests.
    e2 = p0 - p2
    for (e, v_a, v_b) in ((e0, p0, p2), (e1, p0, p1), (e2, p1, p2))
        if !_axis_test_x(e, v_a, v_b, h) ||
                !_axis_test_y(e, v_a, v_b, h) ||
                !_axis_test_z(e, v_a, v_b, h)
            return false
        end
    end
    return true
end

function _plane_box_overlap(normal::SVector{3, T}, point::SVector{3, T}, h::SVector{3, T}) where {T}
    vmin = SVector{3, T}(
        normal[1] > 0 ? -h[1] - point[1] : h[1] - point[1],
        normal[2] > 0 ? -h[2] - point[2] : h[2] - point[2],
        normal[3] > 0 ? -h[3] - point[3] : h[3] - point[3],
    )
    vmax = SVector{3, T}(
        normal[1] > 0 ? h[1] - point[1] : -h[1] - point[1],
        normal[2] > 0 ? h[2] - point[2] : -h[2] - point[2],
        normal[3] > 0 ? h[3] - point[3] : -h[3] - point[3],
    )
    if dot(normal, vmin) > 0
        return false
    end
    if dot(normal, vmax) >= 0
        return true
    end
    return false
end

function _axis_test_x(e::SVector{3, T}, a::SVector{3, T}, b::SVector{3, T}, h::SVector{3, T}) where {T}
    p_a = e[3] * a[2] - e[2] * a[3]
    p_b = e[3] * b[2] - e[2] * b[3]
    mn, mx = min(p_a, p_b), max(p_a, p_b)
    rad = abs(e[3]) * h[2] + abs(e[2]) * h[3]
    return !(mn > rad || mx < -rad)
end

function _axis_test_y(e::SVector{3, T}, a::SVector{3, T}, b::SVector{3, T}, h::SVector{3, T}) where {T}
    p_a = -e[3] * a[1] + e[1] * a[3]
    p_b = -e[3] * b[1] + e[1] * b[3]
    mn, mx = min(p_a, p_b), max(p_a, p_b)
    rad = abs(e[3]) * h[1] + abs(e[1]) * h[3]
    return !(mn > rad || mx < -rad)
end

function _axis_test_z(e::SVector{3, T}, a::SVector{3, T}, b::SVector{3, T}, h::SVector{3, T}) where {T}
    p_a = e[2] * a[1] - e[1] * a[2]
    p_b = e[2] * b[1] - e[1] * b[2]
    mn, mx = min(p_a, p_b), max(p_a, p_b)
    rad = abs(e[2]) * h[1] + abs(e[1]) * h[2]
    return !(mn > rad || mx < -rad)
end

# ── Point-to-triangle projection (barycentric + edge clamp) ───────────

"""
    closest_point_on_triangle(p, v0, v1, v2) -> SVector{3}

Euclidean-closest point on triangle `(v0, v1, v2)` to `p`.  Falls back
to edge / vertex projection when the barycentric foot lies outside the
triangle.
"""
function closest_point_on_triangle(
        p::SVector{3, T}, v0::SVector{3, T}, v1::SVector{3, T}, v2::SVector{3, T},
    ) where {T <: AbstractFloat}
    edge0 = v1 - v0
    edge1 = v2 - v0
    v = v0 - p
    a = dot(edge0, edge0)
    b = dot(edge0, edge1)
    c = dot(edge1, edge1)
    d = dot(edge0, v)
    e = dot(edge1, v)
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    if s + t <= det
        if s < 0
            if t < 0
                # region 4
                if d < 0
                    t = zero(T)
                    s = clamp(-d / a, zero(T), one(T))
                else
                    s = zero(T)
                    t = clamp(-e / c, zero(T), one(T))
                end
            else
                # region 3
                s = zero(T)
                t = clamp(-e / c, zero(T), one(T))
            end
        elseif t < 0
            # region 5
            t = zero(T)
            s = clamp(-d / a, zero(T), one(T))
        else
            # region 0 (interior)
            inv_det = one(T) / det
            s *= inv_det
            t *= inv_det
        end
    else
        if s < 0
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0
                numer = tmp1 - tmp0
                denom = a - 2 * b + c
                s = clamp(numer / denom, zero(T), one(T))
                t = one(T) - s
            else
                s = zero(T)
                t = clamp(-e / c, zero(T), one(T))
            end
        elseif t < 0
            # region 6
            tmp0 = b + e
            tmp1 = a + d
            if tmp1 > tmp0
                numer = tmp1 - tmp0
                denom = a - 2 * b + c
                t = clamp(numer / denom, zero(T), one(T))
                s = one(T) - t
            else
                t = zero(T)
                s = clamp(-d / a, zero(T), one(T))
            end
        else
            # region 1
            numer = c + e - b - d
            if numer <= 0
                s = zero(T)
            else
                denom = a - 2 * b + c
                s = clamp(numer / denom, zero(T), one(T))
            end
            t = one(T) - s
        end
    end

    return v0 + s * edge0 + t * edge1
end

"""
    nearest_point_on_stl(p, vertices, faces) -> (closest::SVector{3}, dist::Float64, face_idx::Int)

Brute-force nearest-point query against the STL triangle soup.  `O(n_faces)`
per call — sufficient for the snap iteration on a small triangle soup,
but a BVH is the obvious v3.2 follow-up.
"""
function nearest_point_on_stl(
        p::SVector{3, T},
        vertices::AbstractVector{<:SVector{3}},
        faces::AbstractVector{<:NTuple{3, <:Integer}},
    ) where {T <: AbstractFloat}
    best_d2 = typemax(T)
    best_pt = p
    best_idx = 0
    @inbounds for (idx, face) in enumerate(faces)
        v0 = SVector{3, T}(vertices[face[1]])
        v1 = SVector{3, T}(vertices[face[2]])
        v2 = SVector{3, T}(vertices[face[3]])
        q = closest_point_on_triangle(p, v0, v1, v2)
        dx = p - q
        d2 = dot(dx, dx)
        if d2 < best_d2
            best_d2 = d2
            best_pt = q
            best_idx = idx
        end
    end
    return best_pt, sqrt(best_d2), best_idx
end

# ── Octree cell-size helper ───────────────────────────────────────────

"""
    cell_diagonal(node::Octree) -> Float64

Length of the space-diagonal of an octree cell.  Used as the reference
length for "close to surface" tests and snap damping.
"""
function cell_diagonal(node::Octree{Dim, T}) where {Dim, T}
    d = node.bbox_max - node.bbox_min
    return sqrt(sum(d .* d))
end

# ── Octree corner enumeration ─────────────────────────────────────────

"""
    octree_corners(node::Octree{3, T}) -> NTuple{8, SVector{3, T}}

Return the 8 corners of the axis-aligned box defined by `node` in
lexicographic order.  Used by `snap_to_surface!` and the castellated
intersection pass.
"""
function octree_corners(node::Octree{3, T}) where {T}
    mn = node.bbox_min
    mx = node.bbox_max
    return (
        SVector{3, T}(mn[1], mn[2], mn[3]),
        SVector{3, T}(mx[1], mn[2], mn[3]),
        SVector{3, T}(mn[1], mx[2], mn[3]),
        SVector{3, T}(mx[1], mx[2], mn[3]),
        SVector{3, T}(mn[1], mn[2], mx[3]),
        SVector{3, T}(mx[1], mn[2], mx[3]),
        SVector{3, T}(mn[1], mx[2], mx[3]),
        SVector{3, T}(mx[1], mx[2], mx[3]),
    )
end

# ── Surface-snap iteration ────────────────────────────────────────────

# Registry keyed on an `Octree` instance so `snap_to_surface!` can keep
# the snapped corner table stable across repeated calls — the octree
# `bbox_min`/`bbox_max` are AABB-only and cannot faithfully represent a
# body-fitted 8-corner cell.  Storing the snapped vertex table outside
# the octree preserves idempotency without extending `Octree` itself.
const _SNAP_REGISTRY = IdDict{Any, Any}()

"""
    snapped_vertex_table(octree::Octree) -> Union{Nothing, Vector{SVector{3, T}}}

Return the deduplicated snapped corner table produced by the most
recent [`snap_to_surface!`](@ref) call on `octree`, or `nothing` if
the octree has not been snapped yet.
"""
function snapped_vertex_table(octree::Octree)
    entry = get(_SNAP_REGISTRY, octree, nothing)
    entry === nothing && return nothing
    return entry.vtable
end

struct _SnapState{T}
    vtable::Vector{SVector{3, T}}
    leaf_corner_idx::Vector{NTuple{8, Int}}
    threshold::Vector{T}
end

"""
    snap_to_surface!(
        octree::Octree{3, T},
        stl_vertices::AbstractVector{<:SVector{3}},
        stl_faces::AbstractVector{<:NTuple{3, <:Integer}};
        max_iters::Int = 20,
        tol::Real = 1.0e-8,
        damping::Real = 0.5,
    ) -> (moved_total::Float64, iters::Int)

Iteratively project the corners of every octree leaf that sits close
to the STL surface (within ½ × cell diagonal) onto their nearest STL
triangle.  Each vertex move is damped by `damping` (under-relaxation)
to stabilise the fixed point.  The loop terminates when the worst
single-iteration move falls below `tol` (measured in absolute length)
or `max_iters` is hit.

On the first call, a deduplicated corner table is built from the
current octree leaves.  The snapped table is cached against the
`octree` object (see [`snapped_vertex_table`](@ref)), so repeated
calls reuse the already-snapped positions — a second invocation is
a near-identity (fixed-point idempotency) even though the octree's
`bbox_min`/`bbox_max` fields remain AABB-only.

Returns the total Euclidean distance vertices moved across all
iterations and the number of iterations executed.
"""
function snap_to_surface!(
        octree::Octree{3, T},
        stl_vertices::AbstractVector{<:SVector{3}},
        stl_faces::AbstractVector{<:NTuple{3, <:Integer}};
        max_iters::Int = 20,
        tol::Real = 1.0e-8,
        damping::Real = 0.5,
    ) where {T <: AbstractFloat}
    state = _snap_state!(octree)

    tol_f = T(tol)
    damping_f = T(damping)
    moved_total = zero(T)
    iters = 0

    for it in 1:max_iters
        iters = it
        worst = zero(T)
        @inbounds for j in eachindex(state.vtable)
            p = state.vtable[j]
            q, dist, _ = nearest_point_on_stl(p, stl_vertices, stl_faces)
            # Dead-band at `tol`: vertices already essentially on the
            # surface do not move.  Combined with the cached vertex
            # table this guarantees a proper fixed point.
            if dist <= state.threshold[j] && dist > tol_f
                new_p = p + damping_f * (q - p)
                step = norm(new_p - p)
                moved_total += step
                if step > worst
                    worst = step
                end
                state.vtable[j] = new_p
            end
        end
        if worst < tol_f
            break
        end
    end

    _write_back_aabb!(octree, state)

    return Float64(moved_total), iters
end

function _snap_state!(octree::Octree{3, T}) where {T <: AbstractFloat}
    cached = get(_SNAP_REGISTRY, octree, nothing)
    cached === nothing || return cached::_SnapState{T}

    leaf_nodes = leaves(octree)
    vtable = SVector{3, T}[]
    vindex = Dict{NTuple{3, T}, Int}()
    leaf_corner_idx = Vector{NTuple{8, Int}}(undef, length(leaf_nodes))

    for (li, leaf) in enumerate(leaf_nodes)
        corners = octree_corners(leaf)
        idx = ntuple(8) do k
            c = corners[k]
            key = (c[1], c[2], c[3])
            j = get(vindex, key, 0)
            if j == 0
                push!(vtable, c)
                j = length(vtable)
                vindex[key] = j
            end
            j
        end
        leaf_corner_idx[li] = idx
    end

    threshold = fill(T(0), length(vtable))
    for (li, leaf) in enumerate(leaf_nodes)
        diag = T(cell_diagonal(leaf))
        half = T(0.5) * diag
        for j in leaf_corner_idx[li]
            if half > threshold[j]
                threshold[j] = half
            end
        end
    end

    state = _SnapState{T}(vtable, leaf_corner_idx, threshold)
    _SNAP_REGISTRY[octree] = state
    return state
end

function _write_back_aabb!(octree::Octree{3, T}, state::_SnapState{T}) where {T <: AbstractFloat}
    leaf_nodes = leaves(octree)
    length(leaf_nodes) == length(state.leaf_corner_idx) || return octree
    for (li, leaf) in enumerate(leaf_nodes)
        idx = state.leaf_corner_idx[li]
        c = state.vtable[idx[1]]
        mn = c
        mx = c
        @inbounds for k in 2:8
            c = state.vtable[idx[k]]
            mn = SVector{3, T}(min(mn[1], c[1]), min(mn[2], c[2]), min(mn[3], c[3]))
            mx = SVector{3, T}(max(mx[1], c[1]), max(mx[2], c[2]), max(mx[3], c[3]))
        end
        leaf.bbox_min = mn
        leaf.bbox_max = mx
    end
    return octree
end
