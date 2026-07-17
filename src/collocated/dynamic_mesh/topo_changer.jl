# dynamic_mesh/topo_changer.jl — Topological mesh-change plans
#
# The `TopoChanger` carries a list of cell splits (one parent → N children)
# and cell merges (N parents → one child). `apply_topo_change!` applies a
# plan to a reduced-state representation — cell volumes + tracked fields —
# in a way that preserves total volume and total ∫φ·V for every listed
# field. Full mesh-topology rebuild (face-list regeneration) is out of
# scope for this V&V slice; the goal here is the conservation invariants
# that any future full implementation must respect.
#
# Key invariants for a single split (V_parent → Σ V_children):
#     Σ V_children  = V_parent                                   (volume)
#     Σ φ_child · V_child = φ_parent · V_parent   for every tracked field
#
# For a merge (Σ V_parents → V_child):
#     V_child              = Σ V_parents
#     φ_child · V_child    = Σ φ_parent · V_parent   (conservative mean)

using StaticArrays: SVector

@doc """
    CellSplit{T}

Plan for a single cell split. The parent cell (index `parent`) is replaced
by `children` new cells with volume fractions `volume_fractions` summing
to 1. Children share the parent's field values (conservative: V-weighted
average is trivially preserved).

# Fields
- `parent::Int` — index of the parent cell in the pre-split mesh
- `children::Vector{Int}` — indices of the child cells in the post-split mesh
- `volume_fractions::Vector{T}` — volume fractions of each child
  (must sum to 1)
"""
struct CellSplit{T}
    parent::Int
    children::Vector{Int}
    volume_fractions::Vector{T}
    function CellSplit{T}(
            parent::Int,
            children::Vector{Int},
            volume_fractions::Vector{T},
        ) where {T}
        length(children) == length(volume_fractions) ||
            error("children and volume_fractions must have matching length")
        length(children) >= 2 ||
            error("a split must produce at least 2 children, got $(length(children))")
        total = sum(volume_fractions)
        isapprox(total, one(T); atol = 100 * eps(T)) ||
            error("volume_fractions must sum to 1 within 100·eps, got $total")
        for frac in volume_fractions
            frac > zero(T) || error("volume_fractions must be positive, got $frac")
        end
        return new{T}(parent, children, volume_fractions)
    end
end

CellSplit(parent::Int, children::Vector{Int}, volume_fractions::Vector{T}) where {T} =
    CellSplit{T}(parent, children, volume_fractions)

@doc """
    CellMerge{T}

Plan for a single cell merge. The `parents` cells are merged into one
`child` cell whose volume is `Σ V_parent` and whose tracked field values
are V-weighted averages.
"""
struct CellMerge{T}
    parents::Vector{Int}
    child::Int
    function CellMerge{T}(parents::Vector{Int}, child::Int) where {T}
        length(parents) >= 2 ||
            error("a merge must consume at least 2 parents, got $(length(parents))")
        return new{T}(parents, child)
    end
end

CellMerge(parents::Vector{Int}, child::Int) = CellMerge{Float64}(parents, child)

@doc """
    TopoChanger{T}

Plan describing the topology changes to apply in one `apply_topo_change!`
invocation.

# Fields
- `splits::Vector{CellSplit{T}}`
- `merges::Vector{CellMerge{T}}`
"""
struct TopoChanger{T}
    splits::Vector{CellSplit{T}}
    merges::Vector{CellMerge{T}}
end

TopoChanger{T}() where {T} = TopoChanger{T}(CellSplit{T}[], CellMerge{T}[])

@doc """
    apply_topo_change!(volumes, fields, plan)

Apply a `TopoChanger` plan to a volume vector and a list of tracked scalar
fields (each `Vector{T}`). Splits are applied first, then merges. For each
operation:

- A split replaces `volumes[parent]` with volumes for each child (fractions
  of the parent volume). Tracked field values are copied verbatim — this
  preserves `φ · V` because each child inherits the parent field value and
  the sum of child volumes is the parent volume.
- A merge replaces the `parents` entries with the merged `child` entry.
  The merged volume is `Σ V_parent` and the merged field is
  `Σ φ_parent · V_parent / Σ V_parent`.

Returns `(new_volumes, new_fields)` where both are freshly allocated.
The original inputs are not mutated; this pure-functional signature makes
unit-level conservation checks straightforward.
"""
function apply_topo_change!(
        volumes::AbstractVector{T},
        fields::AbstractVector{<:AbstractVector{T}},
        plan::TopoChanger{T},
    ) where {T}
    nc = length(volumes)
    for fld in fields
        length(fld) == nc ||
            error("each tracked field must have length nc=$nc, got $(length(fld))")
    end

    # Apply splits first
    V_after_split = copy(volumes)
    fields_after_split = [copy(f) for f in fields]

    for split in plan.splits
        V_parent = V_after_split[split.parent]
        n_children = length(split.children)

        # Parent field values (captured before we overwrite)
        parent_fields = [f[split.parent] for f in fields_after_split]

        # Resize each vector: remove parent, append n_children
        # We instead rebuild compactly: start from current length and
        # append child volumes/fields, mark parent index for removal.
        # For simplicity we take the compacting path.
        mark_remove = falses(length(V_after_split))
        mark_remove[split.parent] = true

        new_V = Vector{T}(undef, length(V_after_split) - 1 + n_children)
        new_fields = [similar(f, length(V_after_split) - 1 + n_children) for f in fields_after_split]

        # Copy surviving cells
        idx = 0
        for c in 1:length(V_after_split)
            if !mark_remove[c]
                idx += 1
                new_V[idx] = V_after_split[c]
                for (k, f) in enumerate(fields_after_split)
                    new_fields[k][idx] = f[c]
                end
            end
        end
        # Append children
        for k in 1:n_children
            idx += 1
            new_V[idx] = V_parent * split.volume_fractions[k]
            for (j, _) in enumerate(fields_after_split)
                new_fields[j][idx] = parent_fields[j]
            end
        end

        V_after_split = new_V
        fields_after_split = new_fields
    end

    # Apply merges
    V_after_merge = copy(V_after_split)
    fields_after_merge = [copy(f) for f in fields_after_split]

    for merge_plan in plan.merges
        V_total = zero(T)
        phi_V_totals = zeros(T, length(fields_after_merge))
        for p in merge_plan.parents
            V_total += V_after_merge[p]
            for (k, f) in enumerate(fields_after_merge)
                phi_V_totals[k] += f[p] * V_after_merge[p]
            end
        end

        mark_remove = falses(length(V_after_merge))
        for p in merge_plan.parents
            mark_remove[p] = true
        end

        new_V = Vector{T}(undef, length(V_after_merge) - length(merge_plan.parents) + 1)
        new_fields = [similar(f, length(new_V)) for f in fields_after_merge]

        idx = 0
        for c in 1:length(V_after_merge)
            if !mark_remove[c]
                idx += 1
                new_V[idx] = V_after_merge[c]
                for (k, f) in enumerate(fields_after_merge)
                    new_fields[k][idx] = f[c]
                end
            end
        end
        # Append merged cell
        idx += 1
        new_V[idx] = V_total
        for (k, _) in enumerate(fields_after_merge)
            new_fields[k][idx] = V_total > zero(T) ? phi_V_totals[k] / V_total : zero(T)
        end

        V_after_merge = new_V
        fields_after_merge = new_fields
    end

    return V_after_merge, fields_after_merge
end

@doc """
    total_volume(volumes) -> T

Sum of cell volumes — a convenience helper used by conservation checks.
"""
total_volume(volumes::AbstractVector{T}) where {T} = sum(volumes)

@doc """
    total_phi_V(phi, volumes) -> T

Return the total integrated quantity `Σ φ_c · V_c`. Used by
`apply_topo_change!` conservation checks.
"""
function total_phi_V(phi::AbstractVector{T}, volumes::AbstractVector{T}) where {T}
    length(phi) == length(volumes) ||
        error("phi length $(length(phi)) ≠ volumes length $(length(volumes))")
    s = zero(T)
    @inbounds for c in 1:length(phi)
        s += phi[c] * volumes[c]
    end
    return s
end
