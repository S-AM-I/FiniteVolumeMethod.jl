# fsi/interface.jl — Fluid-structure interface mapping (Wave 3 Agent C)
#
# Builds a pair of face-index lists on the fluid and solid sides of a
# shared FSI interface and provides the two transfer operators that the
# partitioned Dirichlet-Neumann loop needs:
#
#   - `interpolate_displacement_to_fluid(u_solid, interface)`: push the
#     structure's interface displacement onto the fluid-side face
#     displacement array (the fluid sees this as a Dirichlet mesh-motion
#     BC).
#   - `interpolate_traction_to_structure(t_fluid, interface)`: push the
#     fluid's interface traction onto the solid-side face traction array
#     (the solid sees this as a Neumann stress BC).
#
# Only matching meshes are supported: the interpolation is a 1:1 copy
# keyed by the paired face indices (`FSIInterface` carries no weight
# vectors). Non-matching-mesh transfer (e.g. nearest-face weighted
# gather) is not implemented.

using StaticArrays: SVector

"""
    build_matched_interface(fluid_faces, solid_faces; Dim=2, T=Float64)

Construct an [`FSIInterface`](@ref) for a pair of matching face lists
(equal length, index-aligned). This is the common case when the fluid
and solid share the coupled boundary mesh.

Returns an `FSIInterface{Dim, T}` with empty displacement and traction
exchange buffers ready for the first partitioned iterate.
"""
function build_matched_interface(
        fluid_faces::AbstractVector{<:Integer},
        solid_faces::AbstractVector{<:Integer};
        Dim::Int = 2, T::Type = Float64,
    )
    length(fluid_faces) == length(solid_faces) ||
        error("build_matched_interface: fluid/solid face lists must be equal length")
    ff = Vector{Int}(fluid_faces)
    sf = Vector{Int}(solid_faces)
    return FSIInterface{Dim, T}(ff, sf)
end

"""
    interpolate_displacement_to_fluid!(interface, u_solid)

Push the structure-side displacement `u_solid` (one `SVector{Dim,T}` per
solid interface face, indexed 1..length(solid_face_indices)) onto the
fluid-side exchange buffer `interface.displacement`.

Returns `interface.displacement` for convenience.
"""
function interpolate_displacement_to_fluid!(
        interface::FSIInterface{Dim, T},
        u_solid::AbstractVector{SVector{Dim, T}},
    ) where {Dim, T}
    n = length(interface.solid_face_indices)
    length(u_solid) == n ||
        error("interpolate_displacement_to_fluid!: length(u_solid)=$(length(u_solid)) ≠ $(n)")
    @inbounds for i in 1:n
        interface.displacement[i] = u_solid[i]
    end
    return interface.displacement
end

"""
    interpolate_displacement_to_fluid(u_solid, interface)

Non-mutating flavor: returns a freshly allocated vector of fluid-side
interface displacements. The interface buffer is left untouched.
"""
function interpolate_displacement_to_fluid(
        u_solid::AbstractVector{SVector{Dim, T}},
        interface::FSIInterface{Dim, T},
    ) where {Dim, T}
    n = length(interface.solid_face_indices)
    length(u_solid) == n ||
        error("interpolate_displacement_to_fluid: length(u_solid)=$(length(u_solid)) ≠ $(n)")
    out = Vector{SVector{Dim, T}}(undef, n)
    @inbounds for i in 1:n
        out[i] = u_solid[i]
    end
    return out
end

"""
    interpolate_traction_to_structure!(interface, t_fluid)

Push the fluid-side traction `t_fluid` (one `SVector{Dim,T}` per fluid
interface face) onto the solid-side exchange buffer
`interface.traction`. Returns `interface.traction`.
"""
function interpolate_traction_to_structure!(
        interface::FSIInterface{Dim, T},
        t_fluid::AbstractVector{SVector{Dim, T}},
    ) where {Dim, T}
    n = length(interface.fluid_face_indices)
    length(t_fluid) == n ||
        error("interpolate_traction_to_structure!: length(t_fluid)=$(length(t_fluid)) ≠ $(n)")
    @inbounds for i in 1:n
        interface.traction[i] = t_fluid[i]
    end
    return interface.traction
end

"""
    interpolate_traction_to_structure(t_fluid, interface)

Non-mutating flavor: returns a freshly allocated vector of solid-side
interface tractions.
"""
function interpolate_traction_to_structure(
        t_fluid::AbstractVector{SVector{Dim, T}},
        interface::FSIInterface{Dim, T},
    ) where {Dim, T}
    n = length(interface.fluid_face_indices)
    length(t_fluid) == n ||
        error("interpolate_traction_to_structure: length(t_fluid)=$(length(t_fluid)) ≠ $(n)")
    out = Vector{SVector{Dim, T}}(undef, n)
    @inbounds for i in 1:n
        out[i] = t_fluid[i]
    end
    return out
end

"""
    flatten_interface_displacement(interface)

Flatten the interface-displacement `SVector` array to a plain
`Vector{T}` of length `Dim * n`. Used by the Aitken residual norm and
the convergence check inside the partitioned loop.
"""
function flatten_interface_displacement(
        d::AbstractVector{SVector{Dim, T}},
    ) where {Dim, T}
    n = length(d)
    out = Vector{T}(undef, Dim * n)
    @inbounds for i in 1:n
        for k in 1:Dim
            out[(i - 1) * Dim + k] = d[i][k]
        end
    end
    return out
end

"""
    unflatten_interface_displacement(v, Dim)

Inverse of [`flatten_interface_displacement`](@ref).
"""
function unflatten_interface_displacement(
        v::AbstractVector{T}, ::Val{Dim},
    ) where {T, Dim}
    n, r = divrem(length(v), Dim)
    r == 0 || error("unflatten_interface_displacement: length(v) not divisible by Dim=$(Dim)")
    out = Vector{SVector{Dim, T}}(undef, n)
    @inbounds for i in 1:n
        out[i] = SVector{Dim, T}(ntuple(k -> v[(i - 1) * Dim + k], Dim))
    end
    return out
end
