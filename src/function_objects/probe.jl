# function_objects/probe.jl — Runtime probe / sampling / force primitives.

using StaticArrays: SVector

"""
    Probe{Dim, T}

Point probe with a user-supplied callback invoked each solver step.
"""
struct Probe{Dim, T}
    position::SVector{Dim, T}
    on_step::Function
end

"""
    Force{Dim, T}

Face-integrated force/moment probe. `face_indices` list faces over
which the solver integrates `∫(pressure + τ_wall) dS`; the callback is
invoked with the integrated 3-vector per step.
"""
struct Force{Dim, T}
    face_indices::Vector{Int}
    reference_direction::SVector{Dim, T}
    on_step::Function
end

"""
    SamplingPlane{T}

Axis-aligned plane sampler; `axis ∈ {1, 2, 3}` selects X/Y/Z;
`value` is the coordinate of the plane.
"""
struct SamplingPlane{T}
    axis::Int
    value::T
    on_step::Function
end

"""
    trigger_probe(probe::Probe, state, t)

Wrapper that invokes the stored callback with the current solver state.
Kept as a thin indirection so probes remain callable through a uniform
interface once the solver loop owns the trigger point.
"""
trigger_probe(probe::Probe, state, t) = probe.on_step(probe.position, state, t)
trigger_probe(force::Force, state, t) = force.on_step(force.face_indices, state, t)
trigger_probe(plane::SamplingPlane, state, t) = plane.on_step(plane.axis, plane.value, state, t)
