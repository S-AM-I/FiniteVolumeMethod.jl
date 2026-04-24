# mesh_generation/snappy.jl — snappyHexMesh-style stub (Wave 4, time-boxed)
#
# Native Julia port of OpenFOAM's `snappyHexMesh` pipeline
# (octree → snap → layer-add) is a multi-month project.  This file ships
# a one-day time-boxed stub that:
#
#   * defines the user-facing `SnappyMesher{T}` container so downstream
#     call sites can be written today;
#   * offers `build_snappy_mesh(mesher)` which attempts the native
#     pipeline.  Because snapping and layer addition are not yet
#     implemented, the function emits a deferral `@warn` and falls back
#     to a pure octree mesh (no surface snap, no boundary layers);
#   * recommends callers use `FVMGmshExt` via
#     [`run_gmsh_pipeline`](@ref) when they need a production-quality
#     body-fitted mesh.

using StaticArrays: SVector

"""
    SnappyMesher{T}

Parameter bundle for a snappyHexMesh-style run.  All fields are
optional with sensible defaults so the stub can be driven from a
single-line constructor during experimentation.

# Fields
- `stl_path::String` — path to the closed triangulated surface (STL).
- `bbox_min::SVector{3, T}`, `bbox_max::SVector{3, T}` — background-mesh bounding box.
- `base_level::Int` — uniform refinement level used to seed the octree.
- `surface_level::Int` — target refinement level on the STL surface.
- `n_layers::Int` — number of boundary layers to add (currently ignored).
- `layer_thickness::T` — first-layer thickness (currently ignored).
- `expansion_ratio::T` — layer growth ratio (currently ignored).
- `snap_iterations::Int` — snap-to-surface iterations (currently ignored).
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
        snap_iterations::Int = 0,
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

Opaque return type for the [`build_snappy_mesh`](@ref) stub.  Wraps the
raw octree that the fallback path emits together with bookkeeping
telling the caller what is (and is not) yet implemented.

# Fields
- `octree::Octree{3, T}` — background octree produced by uniform + surface refinement.
- `snap_applied::Bool` — always `false` in the stub.
- `layers_added::Int` — always `0` in the stub.
- `n_cells::Int` — convenience cell count (== `cell_count(octree)`).
"""
struct SnappySnapshot{T}
    octree::Octree{3, T}
    snap_applied::Bool
    layers_added::Int
    n_cells::Int
end

"""
    build_snappy_mesh(mesher::SnappyMesher) -> SnappySnapshot

Attempt the snappyHexMesh-style pipeline.  This is an **experimental
stub**: snap-to-surface and layer addition are deferred, and the
function emits a `@warn` redirecting callers to the Gmsh pipeline via
the `FVMGmshExt` extension.

The fallback produces a uniformly refined octree at `mesher.base_level`.
If an STL path is supplied the octree is additionally refined inside
the axis-aligned bounding box of the surface (a bounding-box proxy;
true STL-triangle intersection tests are TODO).
"""
function build_snappy_mesh(mesher::SnappyMesher{T}) where {T}
    @warn (
        "snappyHexMesh native is experimental; prefer Gmsh pipeline via " *
            "FVMGmshExt (see `run_gmsh_pipeline`). Falling back to octree-only " *
            "background mesh without surface snap or boundary layers."
    ) stl_path = mesher.stl_path n_layers = mesher.n_layers

    # Seed a uniformly refined background octree at `base_level`.
    octree = build_octree(mesher.bbox_min, mesher.bbox_max, mesher.base_level)

    # Axis-aligned-box proxy refinement near the (unknown) surface: if
    # the user supplied a real STL path we cannot read it without an
    # STL loader, so we treat the interior of the bounding box as the
    # proxy surface region and refine toward `surface_level`.
    if mesher.surface_level > mesher.base_level
        surf_center = (mesher.bbox_min + mesher.bbox_max) / T(2)
        bbox_diag = mesher.bbox_max - mesher.bbox_min
        radius = T(0.25) * sqrt(sum(bbox_diag .* bbox_diag))
        refine_near_sphere!(octree, surf_center, radius, mesher.surface_level)
    end

    return SnappySnapshot{T}(octree, false, 0, cell_count(octree))
end
