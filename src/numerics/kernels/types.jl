# kernels/types.jl — Backend-agnostic kernel dispatch traits.

"""
    KernelBackend

Supertype for KA-capable backend wrappers. `CPUBackend` (defined in
`core/backends.jl`) is the default CPU path and is *not* a subtype of
this — `core/backends.jl` was loaded first and owns the `CPUBackend`
type. `KABackend` wraps a `KernelAbstractions.Backend` and is
activated when the `FVMKAExt` extension is loaded.

`KernelBackend <: AbstractBackend` so KA-flavored backends still
satisfy the older solver-side `::AbstractBackend` keyword.
"""
abstract type KernelBackend <: AbstractBackend end

"""
    KABackend{B}

Wrapper around a `KernelAbstractions.Backend` instance (e.g.
`KernelAbstractions.CPU()`, `CUDABackend()`, `ROCBackend()`). Populated
by the `FVMKAExt` extension.
"""
struct KABackend{B} <: KernelBackend
    backend::B
end

"""
    kernel_backend(x) -> KernelBackend

Default backend picker. Returns [`CPUBackend`](@ref) unless overridden
by an extension or a user-side method.
"""
kernel_backend(::Any) = CPUBackend()

"""
    per_term_ad(term_fn::Function, input::AbstractVector, direction::AbstractVector;
                epsilon = 1e-6)

Forward-mode finite-difference surrogate for per-term AD, used as a
v3.0 placeholder until the `FVMEnzymeExt` extension lands. Evaluates
the directional derivative `(f(x + ε·d) − f(x − ε·d)) / (2·ε)`.
"""
function per_term_ad(
        term_fn::Function, input::AbstractVector{T}, direction::AbstractVector{T};
        epsilon::Real = 1.0e-6,
    ) where {T}
    plus = copy(input); plus .+= T(epsilon) .* direction
    minus = copy(input); minus .-= T(epsilon) .* direction
    return (term_fn(plus) - term_fn(minus)) / (2 * T(epsilon))
end
