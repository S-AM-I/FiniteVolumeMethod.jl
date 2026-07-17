# amr_collocated/error_indicators.jl — Cell-based error indicators for
# driving collocated h-adaptivity.
#
# Two classical families:
#
#   1. Residual-based indicator — r_c = (A·u - b)_c / V_c for an
#      assembled `CollocatedEquation`. Measures local algebraic
#      residual; spikes on cells that the current discretisation
#      cannot resolve.
#
#   2. Zienkiewicz-Zhu (ZZ) recovery-based indicator — compares the
#      Green-Gauss gradient (first-order accurate on polyhedra) to
#      the least-squares gradient (superconvergent on linear fields).
#      Their discrepancy in each cell is an a-posteriori error
#      estimate for smooth fields, and spikes on cells near
#      discontinuities.
#
# Both return `Vector{T}` of per-cell non-negative indicator values.

using LinearAlgebra: norm
using SparseArrays: SparseMatrixCSC

# ── Residual indicator ──────────────────────────────────────────────

"""
    residual_error_indicator(eq::CollocatedEquation{T}, u::AbstractVector{T},
                             mesh::UnstructuredFVMMesh{Dim, T}) -> Vector{T}

Per-cell algebraic residual indicator:

```
η_c = |(A·u - (b + source))_c| / V_c
```

Properties:
- Exact algebraic match ⇒ `η ≡ 0`.
- Scales linearly with perturbation magnitude in `u`.
- Non-negative everywhere.

The volume normalisation converts the cell-integrated residual
(units of φ-flux × volume) into a cell-density residual (units of
φ × 1/time or the equation's natural density) so values are
directly comparable across a non-uniform mesh.
"""
function residual_error_indicator(
        eq::CollocatedEquation{T},
        u::AbstractVector{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    length(u) == nc ||
        error("u length $(length(u)) ≠ ncells $nc")
    length(eq.b) == nc ||
        error("eq.b length $(length(eq.b)) ≠ ncells $nc")

    rhs = eq.b .+ eq.source
    Au = eq.A * u
    indicator = Vector{T}(undef, nc)
    @inbounds for c in 1:nc
        V_c = mesh.cell_volumes[c]
        V_safe = V_c > zero(T) ? V_c : eps(T)
        indicator[c] = abs(Au[c] - rhs[c]) / V_safe
    end
    return indicator
end

# ── Zienkiewicz-Zhu indicator ──────────────────────────────────────

"""
    zz_error_indicator(phi::CollocatedScalarField{T},
                       mesh::UnstructuredFVMMesh{Dim, T}) -> Vector{T}

Zienkiewicz-Zhu (1987) recovery-based error indicator.

Compares two independently reconstructed gradients at every cell:

- `∇_GG φ` — Green-Gauss gradient (first-order on polyhedra; one-sided
  near discontinuities).
- `∇_LSQ φ` — weighted least-squares gradient (exact for linear
  fields; smoother in practice).

The per-cell indicator is the Euclidean norm of their difference
scaled by the characteristic cell size `h_c = V_c^{1/Dim}`:

```
η_c = h_c · ‖∇_GG φ_c - ∇_LSQ φ_c‖₂
```

The `h_c` factor converts the gradient discrepancy (units of φ/length)
into an L²-like error density (units of φ) that is directly comparable
between coarse and fine cells. For a linear field both gradients are
exact and `η ≡ 0`; for a strong jump the two reconstructions diverge
and the indicator peaks at interface cells.

This is the minimal ZZ form specified by the v3 fast-path plan
(Wave 4 Agent B); a full SPR (superconvergent patch recovery) would
fit a higher-order polynomial on a patch and is a follow-up.

The allocating implementation in `src/amr_collocated/adapt.jl` that
uses volume-weighted face-neighbour averaging is superseded by this
more faithful GG-vs-LSQ comparison; the older implementation is kept
as `_zz_indicator_smoothed` for backwards compatibility.
"""
function zz_error_indicator(
        phi::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)

    grad_gg = gradient(phi, mesh)
    grad_lsq = least_squares_gradient(phi, mesh)

    indicator = Vector{T}(undef, nc)
    @inbounds for c in 1:nc
        h_c = mesh.cell_volumes[c]^(one(T) / T(Dim))
        indicator[c] = h_c * norm(grad_gg[c] - grad_lsq[c])
    end
    return indicator
end

"""
    _zz_indicator_smoothed(phi, mesh) -> Vector{T}

Legacy volume-weighted face-neighbour-average ZZ variant. Kept so
callers that want the original behaviour of `zz_error_indicator`
(prior to the GG-vs-LSQ reformulation) can opt in explicitly.
"""
function _zz_indicator_smoothed(
        phi::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    grad_local = gradient(phi, mesh)

    weights = zeros(T, nc)
    accum = fill(zero(SVector{Dim, T}), nc)

    for c in 1:nc
        accum[c] = grad_local[c] * mesh.cell_volumes[c]
        weights[c] = mesh.cell_volumes[c]
    end

    nf = size(mesh.face_cells, 2)
    @inbounds for f in 1:nf
        P = mesh.face_cells[1, f]
        N = mesh.face_cells[2, f]
        N == 0 && continue
        accum[P] += grad_local[N] * mesh.cell_volumes[N]
        weights[P] += mesh.cell_volumes[N]
        accum[N] += grad_local[P] * mesh.cell_volumes[P]
        weights[N] += mesh.cell_volumes[P]
    end

    indicator = Vector{T}(undef, nc)
    @inbounds for c in 1:nc
        grad_rec = accum[c] / max(weights[c], eps(T))
        indicator[c] = norm(grad_rec - grad_local[c])
    end
    return indicator
end

# ── Marker utilities ───────────────────────────────────────────────

"""
    mark_for_refinement(indicator_values::AbstractVector{T}, threshold::Real) -> Vector{Int}

Return the 1-based cell indices whose indicator exceeds `threshold`.

Used as the handoff between error indicators and `RefinementPlan`:

```julia
eta = zz_error_indicator(phi, mesh)
to_refine = mark_for_refinement(eta, 0.1 * maximum(eta))
plan = RefinementPlan(to_refine, mesh)
```
"""
function mark_for_refinement(
        indicator_values::AbstractVector{T}, threshold::Real,
    ) where {T}
    th = T(threshold)
    result = Int[]
    @inbounds for c in eachindex(indicator_values)
        indicator_values[c] > th && push!(result, c)
    end
    return result
end

"""
    mark_for_coarsening(indicator_values::AbstractVector{T}, threshold::Real) -> Vector{Int}

Return the 1-based cell indices whose indicator falls below `threshold`.
Intended as input to a `CoarseningPlan`.
"""
function mark_for_coarsening(
        indicator_values::AbstractVector{T}, threshold::Real,
    ) where {T}
    th = T(threshold)
    result = Int[]
    @inbounds for c in eachindex(indicator_values)
        indicator_values[c] < th && push!(result, c)
    end
    return result
end
