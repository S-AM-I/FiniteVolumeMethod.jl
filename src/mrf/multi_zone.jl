# mrf/multi_zone.jl — Multi-zone MRF dispatch (Wave 3)
#
# Rotor-stator and multi-stage rotating-machinery problems need several
# `MRFZone`s in a single mesh. Each cell must belong to at most one MRF
# zone (stationary cells belong to none). `build_multi_mrf_from_zones`
# enforces the disjoint-cell invariant up front; `add_multi_mrf_source!`
# then loops per zone and accumulates into the momentum source.

using StaticArrays: SVector

"""
    build_multi_mrf_from_zones(zones::Vector{MRFZone{T}}) -> MultiMRF{T}

Construct a `MultiMRF` from a list of zones. Raises `ArgumentError` if
any cell index appears in more than one zone.
"""
function build_multi_mrf_from_zones(zones::Vector{MRFZone{T}}) where {T}
    seen = Set{Int}()
    for zone in zones
        for c in zone.cells
            if c in seen
                throw(
                    ArgumentError(
                        "MRF zones must be disjoint: cell $c appears in multiple zones",
                    ),
                )
            end
            push!(seen, c)
        end
    end
    return MultiMRF{T}(zones)
end

"""
    add_multi_mrf_source!(source_U, U, mesh, multi::MultiMRF{T}, rho) -> source_U

Accumulate the MRF momentum source density from every zone in `multi`
into `source_U`. Because zones are guaranteed disjoint by
`build_multi_mrf_from_zones`, the result is unambiguous — each cell is
touched by at most one zone.

`rho` may be either a scalar (same density in every zone) or a
`Vector{T}` with one entry per zone.
"""
function add_multi_mrf_source!(
        source_U::AbstractVector{SVector{3, T}},
        U::AbstractVector{SVector{3, T}},
        mesh,
        multi::MultiMRF{T},
        rho::T,
    ) where {T}
    for zone in multi.zones
        add_mrf_source!(source_U, U, mesh, zone, rho)
    end
    return source_U
end

function add_multi_mrf_source!(
        source_U::AbstractVector{SVector{3, T}},
        U::AbstractVector{SVector{3, T}},
        mesh,
        multi::MultiMRF{T},
        rho::AbstractVector{T},
    ) where {T}
    length(rho) == length(multi.zones) || throw(
        ArgumentError(
            "rho vector length $(length(rho)) does not match number of zones $(length(multi.zones))",
        ),
    )
    for (k, zone) in enumerate(multi.zones)
        add_mrf_source!(source_U, U, mesh, zone, rho[k])
    end
    return source_U
end
