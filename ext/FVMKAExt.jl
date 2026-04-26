module FVMKAExt

using FiniteVolumeMethod
using KernelAbstractions: KernelAbstractions

# Extension override: KABackend wraps a KernelAbstractions backend
# handle. When `KernelAbstractions.jl` is loaded, users can construct
# `FiniteVolumeMethod.KABackend(KernelAbstractions.CPU())` or a GPU
# backend and pass it to the `*_ka!` kernels. For v3.0 we ship the
# interface here; the actual @kernel launches are stubbed and fall
# back to the serial CPU path until per-operator KA kernels are
# implemented in v3.1.
#
# Note: the main-module `kernel_backend(::Any) = CPUBackend()` is the
# serial fallback. The extension provides a `KernelAbstractions.Backend`
# dispatch so users can pass a KA backend handle directly and get the
# wrapped `KABackend` back. Overriding `::Any` here would be a method
# overwrite (illegal during precompile).

function FiniteVolumeMethod.kernel_backend(backend::KernelAbstractions.Backend)
    return FiniteVolumeMethod.KABackend(backend)
end

end
