"""Abstract supertype for execution backends (CPU, CUDA, etc.)."""
abstract type AbstractBackend end

"""CPU execution backend (default). All solvers support this backend."""
struct CPUBackend <: AbstractBackend end

"""CUDA GPU execution backend. Specify `device` to select a GPU (defaults to `nothing` for automatic selection)."""
struct CUDASolverBackend <: AbstractBackend
    device::Union{Nothing, Int}
end

CUDASolverBackend(; device = nothing) = CUDASolverBackend(device)

"""Transfer array `x` to the given execution backend."""
to_backend(x, ::CPUBackend) = x
to_backend(x, backend::AbstractBackend) = _unsupported_backend("to_backend", backend)
"""Transfer array `x` from device back to host memory."""
to_host(x) = x

"""Return `true` if `prob` supports the given execution backend."""
supports_backend(::Any, ::CPUBackend) = true
supports_backend(::Any, ::AbstractBackend) = false

"""Return a human-readable summary string for the execution backend."""
backend_summary(backend::AbstractBackend) = string(typeof(backend))
backend_summary(::CPUBackend) = "CPU backend"

function _cpu_backend_only(name::AbstractString, backend::AbstractBackend)
    backend isa CPUBackend && return nothing
    error("$name currently supports only CPUBackend(). Received $(backend_summary(backend)).")
end

function _unsupported_backend(name::AbstractString, backend::AbstractBackend)
    error("$name does not support $(backend_summary(backend)) in this build.")
end
