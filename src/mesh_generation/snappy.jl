# mesh_generation/snappy.jl — snappyHexMesh-style native mesher (v3.1).
#
# Minimal but functional port of OpenFOAM's `snappyHexMesh` pipeline.
# v3.1 covers the first two stages:
#
#   1. **Castellated**: uniform octree to `base_level`, then per-cell
#      refinement to `surface_level` for every leaf that overlaps an
#      STL triangle (SAT intersection).
#   2. **Surface snap**: iteratively project every octree corner that
#      sits close to the surface onto the nearest STL face.
#
# Layer addition remains deferred to v3.2 — `build_snappy_mesh` emits an
# `@info` to that effect.
#
# The public surface is:
#
#   * `SnappyMesher{T}` — parameter bundle (constructor with defaults).
#   * `SnappySnapshot{T}` — opaque result (snapped octree + bookkeeping).
#   * `build_snappy_mesh(mesher)` — full pipeline driver.
#   * `build_castellated_mesh(mesher)` — exposed for V&V.
#   * `read_stl_ascii` / `write_stl_ascii` — via `stl_reader.jl`.
#   * `snap_to_surface!`, `triangle_intersects_aabb`,
#     `closest_point_on_triangle`, `nearest_point_on_stl` — via `snap.jl`.

using StaticArrays: SVector

include("stl_reader.jl")
include("snap.jl")

"""
    SnappyMesher{T}

Parameter bundle for a snappyHexMesh-style run.  All fields are optional
with sensible defaults so the mesher can be driven from a single-line
constructor during experimentation.

# Fields
- `stl_path::String` — path to a closed triangulated surface (ASCII STL).
- `bbox_min::SVector{3, T}`, `bbox_max::SVector{3, T}` — background-mesh bounding box.
- `base_level::Int` — uniform refinement level used to seed the octree.
- `surface_level::Int` — target refinement level at cells that intersect the STL.
- `n_layers::Int` — number of boundary layers to add (deferred to v3.2).
- `layer_thickness::T` — first-layer thickness (deferred to v3.2).
- `expansion_ratio::T` — layer growth ratio (deferred to v3.2).
- `snap_iterations::Int` — surface-snap iterations (maps to `max_iters`).
"""
struct SnappyMesher{T}
    stl_path::String
    bbox_min::SVector{3, T}
    bbox_max::SVector{3, T}
    base_level::Int
    surface_level::Int
    n_layers::Int
    layer_thickness::T
    expansion_ratio::T
    snap_iterations::Int
end

function SnappyMesher(;
        stl_path::AbstractString = "",
        bbox_min = SVector(0.0, 0.0, 0.0),
        bbox_max = SVector(1.0, 1.0, 1.0),
        base_level::Int = 1,
        surface_level::Int = 2,
        n_layers::Int = 0,
        layer_thickness::Real = 0.0,
        expansion_ratio::Real = 1.2,
        snap_iterations::Int = 20,
    )
    T = promote_type(
        eltype(bbox_min), eltype(bbox_max),
        typeof(float(layer_thickness)), typeof(float(expansion_ratio)),
    )
    return SnappyMesher{T}(
        String(stl_path),
        SVector{3, T}(bbox_min),
        SVector{3, T}(bbox_max),
        base_level,
        surface_level,
        n_layers,
        T(layer_thickness),
        T(expansion_ratio),
        snap_iterations,
    )
end

"""
    SnappySnapshot{T}

Opaque return type of [`build_snappy_mesh`](@ref).  Wraps the snapped
octree together with bookkeeping telling the caller which pipeline
stages were executed.

# Fields
- `octree::Octree{3, T}` — castellated + snapped octree.
- `snap_applied::Bool` — `true` if surface snap ran and moved vertices.
- `layers_added::Int` — always `0` (deferred to v3.2).
- `n_cells::Int` — convenience leaf count.
- `snap_moved::Float64` — total vertex displacement accumulated during snap.
- `snap_iters::Int` — snap iterations executed (≤ `snap_iterations`).
"""
struct SnappySnapshot{T}
    octree::Octree{3, T}
    snap_applied::Bool
    layers_added::Int
    n_cells::Int
    snap_moved::Float64
    snap_iters::Int
end

# Back-compat no-STL convenience constructor kept as-is.
SnappySnapshot{T}(octree::Octree{3, T}, snap_applied::Bool, layers::Int, n_cells::Int) where {T} =
    SnappySnapshot{T}(octree, snap_applied, layers, n_cells, 0.0, 0)

"""
    _refine_cell_on_surface!(node, stl_vertices, stl_faces, target_level)

Recursively subdivide `node` until every leaf that still touches an STL
triangle has `level ≥ target_level`.  Uses the Akenine-Möller SAT test
(`triangle_intersects_aabb`) for the per-triangle intersection check.
"""
function _refine_cell_on_surface!(
        node::Octree{3, T},
        stl_vertices::AbstractVector{<:SVector{3}},
        stl_faces::AbstractVector{<:NTuple{3, <:Integer}},
        target_level::Int,
    ) where {T <: AbstractFloat}
    # Any triangle overlapping this cell?
    hit = false
    mn = node.bbox_min
    mx = node.bbox_max
    @inbounds for face in stl_faces
        v0 = SVector{3, T}(stl_vertices[face[1]])
        v1 = SVector{3, T}(stl_vertices[face[2]])
        v2 = SVector{3, T}(stl_vertices[face[3]])
        if triangle_intersects_aabb(v0, v1, v2, mn, mx)
            hit = true
            break
        end
    end
    hit || return nothing
    if is_leaf(node)
        node.level >= target_level && return nothing
        subdivide!(node)
    end
    for child in node.children
        _refine_cell_on_surface!(child, stl_vertices, stl_faces, target_level)
    end
    return nothing
end

"""
    build_castellated_mesh(mesher::SnappyMesher) -> Octree{3, Float64}

Build the castellated (octree-refined) background mesh for `mesher`.
Stages:

  1. Uniform refinement to `base_level`.
  2. STL-driven refinement: every cell that overlaps an STL triangle is
     recursively split until it reaches `surface_level`.

If `mesher.stl_path` is empty or the file is missing, only stage 1 runs
and the function returns the uniform octree.  The caller is responsible
for any additional proximity bands beyond the two stages above.
"""
function build_castellated_mesh(mesher::SnappyMesher{T}) where {T}
    octree = build_octree(mesher.bbox_min, mesher.bbox_max, mesher.base_level)
    if mesher.surface_level > mesher.base_level && !isempty(mesher.stl_path) &&
            isfile(mesher.stl_path)
        stl_vertices, stl_faces, _ = read_stl_ascii(mesher.stl_path)
        _refine_cell_on_surface!(octree, stl_vertices, stl_faces, mesher.surface_level)
    end
    return octree
end

"""
    build_snappy_mesh(mesher::SnappyMesher) -> SnappySnapshot

Run the v3.1 snappyHexMesh-style pipeline:

  1. Castellated mesh (uniform octree + STL-driven refinement).
  2. Surface snap (project octree vertices onto the STL surface).
  3. Layer addition — **deferred to v3.2** (emits an `@info`).

Returns a [`SnappySnapshot`](@ref) wrapping the snapped octree and some
bookkeeping.  If no STL is supplied the pipeline still runs but the
snap step is a no-op.
"""
function build_snappy_mesh(mesher::SnappyMesher{T}) where {T}
    octree = build_castellated_mesh(mesher)

    snap_applied = false
    snap_moved = 0.0
    snap_iters = 0
    if !isempty(mesher.stl_path) && isfile(mesher.stl_path)
        stl_vertices, stl_faces, _ = read_stl_ascii(mesher.stl_path)
        if mesher.snap_iterations > 0
            moved, iters = snap_to_surface!(
                octree, stl_vertices, stl_faces;
                max_iters = mesher.snap_iterations,
            )
            snap_moved = moved
            snap_iters = iters
            snap_applied = moved > 0
        end
    end

    if mesher.n_layers > 0
        @info "snappyHexMesh: layer addition deferred to v3.2" n_layers = mesher.n_layers layer_thickness = mesher.layer_thickness
    else
        @info "snappyHexMesh: castellated + snap complete (layer addition deferred to v3.2)" n_cells = cell_count(octree) snap_moved = snap_moved snap_iters = snap_iters
    end

    return SnappySnapshot{T}(octree, snap_applied, 0, cell_count(octree), snap_moved, snap_iters)
end
