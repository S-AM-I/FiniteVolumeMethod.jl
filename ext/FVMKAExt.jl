module FVMKAExt

using FiniteVolumeMethod
using KernelAbstractions

# Extension override: KABackend wraps a KernelAbstractions backend
# handle. When `KernelAbstractions.jl` is loaded, users can construct
# `FiniteVolumeMethod.KABackend(KernelAbstractions.CPU())` or a GPU
# backend and pass it to the `*_ka!` kernels. For v3.0 we ship the
# interface here; the actual @kernel launches are stubbed and fall
# back to the serial CPU path until per-operator KA kernels are
# implemented in v3.1.

function FiniteVolumeMethod.kernel_backend(::Any)
    return FiniteVolumeMethod.KABackend(KernelAbstractions.CPU())
end

end
