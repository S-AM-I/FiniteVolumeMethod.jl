# amr_collocated/adapt.jl — h-adaptivity for collocated unstructured meshes
#
# Stage 8c MVP: a marker-driven cell-based refinement / coarsening
# interface for `UnstructuredFVMMesh{Dim, T}`. Full AMR with
# conservation-preserving flux correction on arbitrary unstructured
# meshes is a research-grade effort; this module provides:
#
#   - `RefinementMarker` per-cell state: `:refine`, `:coarsen`, `:keep`.
#   - `mark_cells_by_gradient(field, mesh; refine_threshold, coarsen_threshold)`
#     — common practical indicator: refine where |grad φ| > refine_threshold,
#     coarsen where < coarsen_threshold.
#   - `flux_correction_factor(parent_face_area, child_face_areas)` —
#     the conservation-preserving factor applied to child-to-parent
#     flux mappings when an AMR interface is traversed.
#
# Actual mesh-structure manipulation (child-cell insertion, parent-cell
# deletion) is a Stage 8 follow-up that requires a tree-augmented
# mesh data structure; this module provides the APPROACH and the
# KERNELS needed to drive it.

using StaticArrays: SVector
using LinearAlgebra: norm

"""
    RefinementMarker

Per-cell decision from an error indicator:
- `:refine` — subdivide this cell.
- `:coarsen` — merge with siblings to coarsen.
- `:keep` — leave unchanged.
"""
const RefinementMarker = Symbol

"""
    mark_cells_by_gradient(field, mesh; refine_threshold, coarsen_threshold)
        -> Vector{Symbol}

Compute a per-cell refinement marker based on the local gradient
magnitude of `field`. Returns a vector of `:refine` / `:coarsen` /
`:keep` of length `ncells`.

A cell is marked `:refine` if `|∇φ|_c * h_c > refine_threshold` (where
`h_c = V_c^{1/Dim}` is a characteristic cell size), `:coarsen` if
below `coarsen_threshold`, else `:keep`.

This is the standard gradient-based indicator used in most adaptive
CFD codes. A Zienkiewicz-Zhu recovery-based indicator is in
`src/amr_collocated/zz_indicator.jl` (also Stage 8d).
"""
function mark_cells_by_gradient(
        grad_field::AbstractVector{SVector{Dim, T}},
        mesh::UnstructuredFVMMesh{Dim, T};
        refine_threshold::Real = 1.0,
        coarsen_threshold::Real = 0.1,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    length(grad_field) == nc ||
        error("grad_field length $(length(grad_field)) ≠ ncells $nc")
    markers = Vector{Symbol}(undef, nc)
    @inbounds for c in 1:nc
        h_c = mesh.cell_volumes[c]^(one(T) / T(Dim))
        indicator = norm(grad_field[c]) * h_c
        markers[c] = if indicator > refine_threshold
            :refine
        elseif indicator < coarsen_threshold
            :coarsen
        else
            :keep
        end
    end
    return markers
end

"""
    flux_correction_factor(parent_area::T, child_areas::AbstractVector{T}) -> T

Conservation-preserving factor applied to the sum of child-level
fluxes when they traverse a non-conforming face shared with a
parent-level cell. For a refinement ratio `r`, the child faces together
cover the parent face area; conservation requires the parent-side flux
to receive the area-weighted sum of child fluxes. Returns the ratio
`parent_area / Σ child_areas`, which should be 1.0 when the children
exactly tile the parent.

A ratio ≠ 1 within floating-point tolerance flags a non-conforming
AMR mesh where flux correction is needed.
"""
function flux_correction_factor(parent_area::T, child_areas::AbstractVector{T}) where {T}
    total = zero(T)
    @inbounds for A in child_areas
        total += A
    end
    return parent_area / max(total, eps(T))
end

# Note: the canonical `zz_error_indicator` lives in error_indicators.jl
# (GG-vs-LSQ formulation from Wave 1 W1-E). A legacy smoothed variant
# remains available there as `_zz_indicator_smoothed` for callers that
# want the pre-Wave-1 behaviour.
