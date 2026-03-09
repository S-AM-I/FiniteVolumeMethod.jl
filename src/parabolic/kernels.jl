# kernels.jl - GPU/CPU kernels using KernelAbstractions
# Migrated from Simu.jl SimuFVM/kernels.jl
# Note: KernelAbstractions is an optional dependency; this file is a placeholder
# that will work if KernelAbstractions is available in the environment.

# Conditional loading: only define kernel functions if KernelAbstractions is loaded
# For now, provide CPU-only fallback implementations

"""
    compute_fluxes_cpu!(F, phi, gamma, dx)

CPU fallback for computing diffusion fluxes.
F[i] is flux at face i+1/2 (between cell i and i+1).
"""
function compute_fluxes_cpu!(F, phi, gamma, dx)
    n = length(phi)
    for i in 1:(n - 1)
        grad = (phi[i + 1] - phi[i]) / dx[i]
        F[i] = -gamma[i] * grad
    end
    return
end
