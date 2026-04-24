# kernels/ka_stubs.jl — CPU-only implementations of the KA-dispatched
# kernels. The real KA variants live in `ext/FVMKAExt.jl` and override
# these when `KernelAbstractions.jl` is loaded.

"""
    interpolate_face_ka!(out::AbstractVector, field::AbstractVector, weights::AbstractVector;
                        backend = CPUBackend())

Face-weighted interpolation: `out[f] = weights[f]·field[P] + (1-weights[f])·field[N]`.
For the CPU backend this is a plain loop; the KA extension overrides
to launch a `@kernel` on the provided backend.
"""
function interpolate_face_ka!(
        out::AbstractVector{T}, field_P::AbstractVector{T}, field_N::AbstractVector{T},
        weights::AbstractVector{T}; backend::KernelBackend = CPUBackend(),
    ) where {T}
    nf = length(out)
    @inbounds for f in 1:nf
        out[f] = weights[f] * field_P[f] + (one(T) - weights[f]) * field_N[f]
    end
    return out
end

"""
    elementwise_sum_ka!(out, a, b; backend)

Element-wise sum `out[i] = a[i] + b[i]`. Exists as a minimal proxy for
the full kernel zoo so the backend dispatch surface can be exercised
in V&V without dragging in the full gradient/laplacian stencils.
"""
function elementwise_sum_ka!(
        out::AbstractVector{T}, a::AbstractVector{T}, b::AbstractVector{T};
        backend::KernelBackend = CPUBackend(),
    ) where {T}
    @inbounds for i in eachindex(out)
        out[i] = a[i] + b[i]
    end
    return out
end
