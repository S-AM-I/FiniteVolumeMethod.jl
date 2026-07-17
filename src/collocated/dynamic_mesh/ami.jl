# dynamic_mesh/ami.jl — Arbitrary Mesh Interface (AMI) sliding coupling
#
# An AMI couples two non-conformal, non-matching face patches (e.g. the
# rotating and stationary sides of a sliding interface) by distributing
# quantities across pairs of overlapping face segments.  For each
# `AMIFacePair`, the overlap area (the portion of the donor face that
# covers the receiver face, in the projected sense) weights the transfer.
#
# The invariants this module guarantees:
# - If every donor face has the same scalar value φ, the receiver faces
#   all get that same φ (uniform donor ⇒ uniform receiver).
# - Σ_donor (φ_d · A_donor) = Σ_receiver (φ_r · A_receiver) where the
#   receiver-side sum is over pairs and A_receiver is the overlap area.
# - Zero overlap area ⇒ zero transfer.
# - Full overlap of a single donor onto a single receiver reproduces the
#   1:1 transfer limit exactly.

using StaticArrays: SVector

@doc """
    AMIFacePair{T}

A single donor-receiver face pair with a pre-computed overlap area.

# Fields
- `donor::Int`         — index of the donor-side face
- `receiver::Int`      — index of the receiver-side face
- `overlap_area::T`    — overlap (projected) area shared by the two faces
"""
struct AMIFacePair{T}
    donor::Int
    receiver::Int
    overlap_area::T
    function AMIFacePair{T}(donor::Int, receiver::Int, overlap_area::T) where {T}
        overlap_area >= zero(T) ||
            error("overlap_area must be ≥ 0, got $overlap_area")
        return new{T}(donor, receiver, overlap_area)
    end
end

AMIFacePair(donor::Int, receiver::Int, overlap_area::T) where {T} =
    AMIFacePair{T}(donor, receiver, overlap_area)

@doc """
    AMIInterface{T}

Describes a sliding / non-conformal mesh interface between two face sets.

# Fields
- `pairs::Vector{AMIFacePair{T}}` — the donor-receiver pairs with overlap area
- `donor_areas::Vector{T}`        — total area of each donor face
- `receiver_areas::Vector{T}`     — total area of each receiver face

Internally the module indexes donors and receivers by integer IDs
consistent across `pairs`, `donor_areas`, and `receiver_areas`.
"""
struct AMIInterface{T}
    pairs::Vector{AMIFacePair{T}}
    donor_areas::Vector{T}
    receiver_areas::Vector{T}
    function AMIInterface{T}(
            pairs::Vector{AMIFacePair{T}},
            donor_areas::Vector{T},
            receiver_areas::Vector{T},
        ) where {T}
        for p in pairs
            (1 <= p.donor <= length(donor_areas)) ||
                error("pair donor index $(p.donor) out of range")
            (1 <= p.receiver <= length(receiver_areas)) ||
                error("pair receiver index $(p.receiver) out of range")
        end
        for a in donor_areas
            a >= zero(T) ||
                error("donor_areas must be ≥ 0, got $a")
        end
        for a in receiver_areas
            a >= zero(T) ||
                error("receiver_areas must be ≥ 0, got $a")
        end
        return new{T}(pairs, donor_areas, receiver_areas)
    end
end

AMIInterface(
    pairs::Vector{AMIFacePair{T}},
    donor_areas::Vector{T},
    receiver_areas::Vector{T},
) where {T} = AMIInterface{T}(pairs, donor_areas, receiver_areas)

@doc """
    project_ami_flux!(phi_receiver, phi_donor, ami) -> phi_receiver

Project donor-side face flux values onto the receiver side, weighted by
overlap area. The result on each receiver face is the overlap-area-
weighted average of donor-face values:

```
φ_receiver[r] =
    (Σ_pairs(r) φ_donor[d(pair)] · overlap_area(pair))
        / Σ_pairs(r) overlap_area(pair)
```

A receiver with no overlap gets zero (no flux transferred). The
quantity `Σ_pairs φ_donor[d] · overlap_area` is equal to
`Σ_pairs φ_receiver[r] · overlap_area`, giving exact integrated-flux
conservation across the interface.

Mutates and returns `phi_receiver`.
"""
function project_ami_flux!(
        phi_receiver::AbstractVector{T},
        phi_donor::AbstractVector{T},
        ami::AMIInterface{T},
    ) where {T}
    length(phi_receiver) == length(ami.receiver_areas) ||
        error("phi_receiver length must match receiver_areas")
    length(phi_donor) == length(ami.donor_areas) ||
        error("phi_donor length must match donor_areas")

    # Zero receivers that are reachable from any pair (leave untouched
    # receivers that have no pair at all: they're not part of the AMI)
    touched = falses(length(phi_receiver))
    overlap_sum = zeros(T, length(phi_receiver))
    weighted_sum = zeros(T, length(phi_receiver))

    @inbounds for p in ami.pairs
        touched[p.receiver] = true
        overlap_sum[p.receiver] += p.overlap_area
        weighted_sum[p.receiver] += phi_donor[p.donor] * p.overlap_area
    end

    @inbounds for r in 1:length(phi_receiver)
        if touched[r]
            if overlap_sum[r] > zero(T)
                phi_receiver[r] = weighted_sum[r] / overlap_sum[r]
            else
                phi_receiver[r] = zero(T)
            end
        end
    end

    return phi_receiver
end

@doc """
    ami_flux_integral(phi, areas) -> T

Return `Σ φ_f · A_f` over a face list. Convenience helper for the
conservation invariant `Σ donor = Σ receiver` over the interface.
"""
function ami_flux_integral(phi::AbstractVector{T}, areas::AbstractVector{T}) where {T}
    length(phi) == length(areas) ||
        error("phi and areas length mismatch")
    s = zero(T)
    @inbounds for f in 1:length(phi)
        s += phi[f] * areas[f]
    end
    return s
end

@doc """
    ami_flux_integral_over_overlaps(phi, ami; side = :donor) -> T

Return the interface-wide integrated flux `Σ_pairs φ_f · overlap_area`
using either the donor or the receiver side of each pair (via `side`).
When donor and receiver flux fields are related through `project_ami_flux!`,
the two sums are equal to within rounding.
"""
function ami_flux_integral_over_overlaps(
        phi::AbstractVector{T},
        ami::AMIInterface{T};
        side::Symbol = :donor,
    ) where {T}
    s = zero(T)
    @inbounds for p in ami.pairs
        idx = side === :donor ? p.donor : p.receiver
        s += phi[idx] * p.overlap_area
    end
    return s
end

@doc """
    build_matching_ami(n_donors::Int) -> AMIInterface{Float64}

Construct a trivial 1:1 matching AMI with `n_donors` face pairs, each
with unit donor area, unit receiver area, and unit overlap. Useful for
exercising the transfer code in unit tests.
"""
function build_matching_ami(n_donors::Int)
    T = Float64
    pairs = AMIFacePair{T}[AMIFacePair{T}(f, f, one(T)) for f in 1:n_donors]
    areas = fill(one(T), n_donors)
    return AMIInterface{T}(pairs, areas, copy(areas))
end
