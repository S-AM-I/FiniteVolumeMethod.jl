# dynamic_mesh/overset.jl — Chimera / overset grid assembly and transfer
#
# Overset (chimera) grids let us couple an independently-generated component
# mesh to a background mesh. Each *receiver* cell on the overset mesh fetches
# its value from a set of *donor* cells on the background (or another
# component) mesh using barycentric-style linear interpolation weights.
#
# This file provides a minimal but exact implementation of the transfer
# step — sufficient for the V&V invariants (constant field reproduced
# exactly, linear field reproduced exactly, weights summing to 1).
# Donor search (hole cutting, donor identification, fringing) is left
# to a separate facility; the assembly here consumes a pre-built
# `OversetAssembly` describing which donors feed each receiver.

using StaticArrays: SVector

@doc """
    DonorStencil{T}

Per-receiver donor stencil.

# Fields
- `donors::Vector{Int}` — donor cell indices (on the donor mesh)
- `weights::Vector{T}`  — interpolation weights (must sum to 1)
"""
struct DonorStencil{T}
    donors::Vector{Int}
    weights::Vector{T}
    function DonorStencil{T}(donors::Vector{Int}, weights::Vector{T}) where {T}
        length(donors) == length(weights) ||
            error("donors and weights must have matching length")
        length(donors) >= 1 ||
            error("a donor stencil must have at least 1 donor")
        return new{T}(donors, weights)
    end
end

DonorStencil(donors::Vector{Int}, weights::Vector{T}) where {T} =
    DonorStencil{T}(donors, weights)

@doc """
    OversetAssembly{T}

Mapping from receiver cells (on the overset mesh) to donor stencils
(on the background mesh).

# Fields
- `receivers::Vector{Int}` — indices of receiver cells on the overset mesh
- `stencils::Vector{DonorStencil{T}}` — one stencil per receiver
- `receiver_mask::Vector{Bool}` — per-cell flag on the overset mesh marking
  whether a cell is a receiver (fringe) or an active interior cell
"""
struct OversetAssembly{T}
    receivers::Vector{Int}
    stencils::Vector{DonorStencil{T}}
    receiver_mask::Vector{Bool}
    function OversetAssembly{T}(
            receivers::Vector{Int},
            stencils::Vector{DonorStencil{T}},
            receiver_mask::AbstractVector{Bool},
        ) where {T}
        length(receivers) == length(stencils) ||
            error("receivers and stencils must have matching length")
        for s in stencils
            total = sum(s.weights)
            isapprox(total, one(T); atol = 1.0e-10) ||
                error("donor stencil weights must sum to 1, got $total")
        end
        return new{T}(receivers, stencils, Vector{Bool}(receiver_mask))
    end
end

OversetAssembly(
    receivers::Vector{Int},
    stencils::Vector{DonorStencil{T}},
    receiver_mask::AbstractVector{Bool},
) where {T} = OversetAssembly{T}(receivers, stencils, receiver_mask)

@doc """
    interpolate_overset!(phi_receiver, phi_donor, assembly) -> phi_receiver

Transfer values from donor cells to receiver cells using the stencils in
`assembly`. Only the entries of `phi_receiver` listed in
`assembly.receivers` are overwritten; other entries are left untouched.

```
φ_receiver[r] = Σ_j w_j · φ_donor[ donors[j] ]
```

Arguments:
- `phi_receiver::AbstractVector{T}` — mutated in place
- `phi_donor::AbstractVector{T}`    — donor-mesh field (read-only)
- `assembly::OversetAssembly{T}`
"""
function interpolate_overset!(
        phi_receiver::AbstractVector{T},
        phi_donor::AbstractVector{T},
        assembly::OversetAssembly{T},
    ) where {T}
    for (k, r) in enumerate(assembly.receivers)
        stencil = assembly.stencils[k]
        acc = zero(T)
        @inbounds for (j, d) in enumerate(stencil.donors)
            acc += stencil.weights[j] * phi_donor[d]
        end
        phi_receiver[r] = acc
    end
    return phi_receiver
end

@doc """
    build_nearest_neighbour_assembly(overset_centers, donor_centers,
                                     receiver_idxs; receiver_mask = nothing)
        -> OversetAssembly{T}

Construct an `OversetAssembly` where each receiver uses its single nearest
donor cell (weight = 1). The result is exact for constant fields and
first-order accurate for smooth fields.

# Arguments
- `overset_centers::AbstractMatrix{T}` — `Dim × nc_overset` cell centres
- `donor_centers::AbstractMatrix{T}`   — `Dim × nc_donor`   cell centres
- `receiver_idxs::Vector{Int}`         — indices of receiver cells

# Keyword Arguments
- `receiver_mask::Vector{Bool}` — optional per-cell mask on the overset
  mesh (default: a fresh `falses(nc_overset)` with `receiver_idxs` marked)
"""
function build_nearest_neighbour_assembly(
        overset_centers::AbstractMatrix{T},
        donor_centers::AbstractMatrix{T},
        receiver_idxs::Vector{Int};
        receiver_mask::Union{Nothing, Vector{Bool}} = nothing,
    ) where {T}
    Dim = size(overset_centers, 1)
    size(donor_centers, 1) == Dim ||
        error("overset and donor cell_centers dimensional mismatch")

    nc_overset = size(overset_centers, 2)
    nc_donor = size(donor_centers, 2)

    mask = receiver_mask === nothing ? falses(nc_overset) : copy(receiver_mask)
    for r in receiver_idxs
        mask[r] = true
    end

    stencils = Vector{DonorStencil{T}}(undef, length(receiver_idxs))
    for (k, r) in enumerate(receiver_idxs)
        # nearest donor
        best_d = 1
        best_d2 = typemax(T)
        for d in 1:nc_donor
            dist2 = zero(T)
            for dim in 1:Dim
                diff = overset_centers[dim, r] - donor_centers[dim, d]
                dist2 += diff * diff
            end
            if dist2 < best_d2
                best_d2 = dist2
                best_d = d
            end
        end
        stencils[k] = DonorStencil{T}([best_d], [one(T)])
    end

    return OversetAssembly{T}(receiver_idxs, stencils, mask)
end

@doc """
    build_linear_donor_assembly(overset_centers, donor_centers, receiver_idxs,
                                donor_triplets; receiver_mask = nothing)

Construct an `OversetAssembly` using explicit 3-donor barycentric-linear
stencils. For a 2D triangle with vertices `x_a, x_b, x_c` and receiver
position `x_r`, the barycentric weights `(w_a, w_b, w_c)` satisfy
`w_a + w_b + w_c = 1` and `w_a x_a + w_b x_b + w_c x_c = x_r`, making the
interpolation exact for any linear field.

# Arguments
- `donor_triplets::Vector{NTuple{3, Int}}` — one donor-triplet per receiver
"""
function build_linear_donor_assembly(
        overset_centers::AbstractMatrix{T},
        donor_centers::AbstractMatrix{T},
        receiver_idxs::Vector{Int},
        donor_triplets::Vector{NTuple{3, Int}};
        receiver_mask::Union{Nothing, Vector{Bool}} = nothing,
    ) where {T}
    size(overset_centers, 1) == 2 == size(donor_centers, 1) ||
        error("build_linear_donor_assembly currently supports 2D only")
    length(donor_triplets) == length(receiver_idxs) ||
        error("donor_triplets length must match receiver_idxs")

    nc_overset = size(overset_centers, 2)
    mask = receiver_mask === nothing ? falses(nc_overset) : copy(receiver_mask)
    for r in receiver_idxs
        mask[r] = true
    end

    stencils = Vector{DonorStencil{T}}(undef, length(receiver_idxs))
    for (k, r) in enumerate(receiver_idxs)
        a, b, c = donor_triplets[k]
        xa = SVector(donor_centers[1, a], donor_centers[2, a])
        xb = SVector(donor_centers[1, b], donor_centers[2, b])
        xc = SVector(donor_centers[1, c], donor_centers[2, c])
        xr = SVector(overset_centers[1, r], overset_centers[2, r])
        # Barycentric in 2D: solve
        #   [xb-xa  xc-xa] · [λb; λc] = xr - xa ;  λa = 1 - λb - λc
        M11 = xb[1] - xa[1]
        M12 = xc[1] - xa[1]
        M21 = xb[2] - xa[2]
        M22 = xc[2] - xa[2]
        det = M11 * M22 - M12 * M21
        abs(det) > 1.0e-14 || error(
            "degenerate donor triangle for receiver $r (det=$det)",
        )
        rhs1 = xr[1] - xa[1]
        rhs2 = xr[2] - xa[2]
        λb = (M22 * rhs1 - M12 * rhs2) / det
        λc = (-M21 * rhs1 + M11 * rhs2) / det
        λa = one(T) - λb - λc
        stencils[k] = DonorStencil{T}([a, b, c], [λa, λb, λc])
    end

    return OversetAssembly{T}(receiver_idxs, stencils, mask)
end

@doc """
    is_receiver(assembly::OversetAssembly, cell::Int) -> Bool

Return whether `cell` (in the overset mesh) is a fringe / receiver cell.
"""
function is_receiver(assembly::OversetAssembly, cell::Int)
    (1 <= cell <= length(assembly.receiver_mask)) ||
        error("cell index $cell out of range [1, $(length(assembly.receiver_mask))]")
    return assembly.receiver_mask[cell]
end
