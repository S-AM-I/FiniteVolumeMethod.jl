# multiphase/mixture.jl — Mixture property computation from volume fraction
#
# Blends density and viscosity linearly by alpha for two-phase VOF.

"""
    update_mixture_properties!(vof_state, props)

Update mixture density and viscosity from current volume fraction:
- `ρ[c] = α[c]·ρ₁ + (1 - α[c])·ρ₂`
- `μ[c] = α[c]·μ₁ + (1 - α[c])·μ₂`
"""
function update_mixture_properties!(
        vof_state::VOFState{T},
        props::TwoPhaseProperties{T},
    ) where {T}
    nc = length(vof_state.rho)
    for c in 1:nc
        a = vof_state.alpha.internal[c]
        vof_state.rho[c] = a * props.rho1 + (one(T) - a) * props.rho2
        vof_state.mu[c] = a * props.mu1 + (one(T) - a) * props.mu2
    end
    return nothing
end
