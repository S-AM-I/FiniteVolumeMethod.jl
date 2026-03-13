abstract type AbstractBackend end

struct CPUBackend <: AbstractBackend end

struct CUDASolverBackend <: AbstractBackend
    device::Union{Nothing, Int}
end

CUDASolverBackend(; device = nothing) = CUDASolverBackend(device)

to_backend(x, ::CPUBackend) = x
to_backend(x, backend::AbstractBackend) = _unsupported_backend("to_backend", backend)
to_host(x) = x

supports_backend(::Any, ::CPUBackend) = true
supports_backend(::Any, ::AbstractBackend) = false

backend_summary(backend::AbstractBackend) = string(typeof(backend))
backend_summary(::CPUBackend) = "CPU backend"

function _cpu_backend_only(name::AbstractString, backend::AbstractBackend)
    backend isa CPUBackend && return nothing
    error("$name currently supports only CPUBackend(). Received $(backend_summary(backend)).")
end

function _unsupported_backend(name::AbstractString, backend::AbstractBackend)
    error("$name does not support $(backend_summary(backend)) in this build.")
end
