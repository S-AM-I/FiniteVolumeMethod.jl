# amr_collocated/coarsening.jl — Sibling-merge primitives for h-coarsening.

"""
    CoarseningPlan{T}

Plan data-structure: each inner vector is a group of sibling indices to
merge into one parent cell. Volume-weighted mean preserves
`Σ φ·V` exactly.
"""
struct CoarseningPlan{T}
    sibling_groups::Vector{Vector{Int}}
end

CoarseningPlan(groups::Vector{Vector{Int}}) = CoarseningPlan{Float64}(groups)

"""
    apply_coarsening!(field::Vector{T}, volumes::Vector{T}, plan::CoarseningPlan)
        -> (field_new, volumes_new)

Volume-weighted merge of sibling groups. Cells not named in any group
are preserved. Indices within `plan.sibling_groups` must be distinct
and cover each old cell at most once.
"""
function apply_coarsening!(
        field::AbstractVector{T}, volumes::AbstractVector{T}, plan::CoarseningPlan,
    ) where {T}
    nc_old = length(field)
    in_group = fill(false, nc_old)
    for group in plan.sibling_groups
        for c in group
            (c < 1 || c > nc_old) && error("CoarseningPlan: index $c out of range")
            in_group[c] && error("CoarseningPlan: cell $c appears in more than one group")
            in_group[c] = true
        end
    end
    field_new = T[]
    volumes_new = T[]
    for c in 1:nc_old
        in_group[c] && continue
        push!(field_new, field[c])
        push!(volumes_new, volumes[c])
    end
    for group in plan.sibling_groups
        V_parent = zero(T)
        phi_V = zero(T)
        for c in group
            V_parent += volumes[c]
            phi_V += field[c] * volumes[c]
        end
        push!(field_new, phi_V / V_parent)
        push!(volumes_new, V_parent)
    end
    return (field_new, volumes_new)
end
