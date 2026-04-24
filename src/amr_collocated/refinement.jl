# amr_collocated/refinement.jl — Cell-based h-adaptivity primitives.
#
# v3.0 fast-path ships the flux-correction / conservative-interpolation
# helpers; actual mesh mutation (edge splits + face re-assembly on an
# `UnstructuredFVMMesh`) lives with Stage 8 and is a v3.1 follow-up.

"""
    RefinementPlan{T}

Plan data-structure: which cells to refine + the per-cell volume budget
for the 2^Dim children (uniform by default). Consumed by
[`apply_refinement!`](@ref).
"""
struct RefinementPlan{T}
    cells::Vector{Int}
    children_per_cell::Int
end

RefinementPlan(cells::Vector{Int}, dim::Int) = RefinementPlan{Float64}(cells, 2^dim)

# Note: `mark_for_refinement` lives in error_indicators.jl.

"""
    apply_refinement!(field::Vector{T}, volumes::Vector{T}, plan::RefinementPlan;
                      copy_value = true) -> (field_new, volumes_new)

Conservative refinement: each parent cell is split into
`plan.children_per_cell` children of equal volume
`V_parent / children_per_cell`. The child field value equals the parent
field value (piecewise-constant prolongation) so
`Σ_children φ_child · V_child = φ_parent · V_parent` is preserved to
machine precision.
"""
function apply_refinement!(
        field::AbstractVector{T}, volumes::AbstractVector{T}, plan::RefinementPlan;
        copy_value::Bool = true,
    ) where {T}
    nc_old = length(field)
    length(volumes) == nc_old || error("field / volumes length mismatch")
    to_refine = Set(plan.cells)
    nc_new = 0
    for c in 1:nc_old
        nc_new += c in to_refine ? plan.children_per_cell : 1
    end
    field_new = Vector{T}(undef, nc_new)
    volumes_new = Vector{T}(undef, nc_new)
    k = 1
    for c in 1:nc_old
        if c in to_refine
            V_child = volumes[c] / plan.children_per_cell
            for _ in 1:plan.children_per_cell
                field_new[k] = copy_value ? field[c] : zero(T)
                volumes_new[k] = V_child
                k += 1
            end
        else
            field_new[k] = field[c]
            volumes_new[k] = volumes[c]
            k += 1
        end
    end
    return (field_new, volumes_new)
end
