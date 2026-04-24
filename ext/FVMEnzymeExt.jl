module FVMEnzymeExt

using FiniteVolumeMethod
using Enzyme

function FiniteVolumeMethod.autodiff_forward_step(
        step_fn::Function, state::AbstractVector, dstate::AbstractVector,
    )
    @warn "Enzyme full-solver AD deferred to v3.1 — per-term AD only" maxlog = 1
    return error("Enzyme full-solver AD not implemented in v3.0")
end

end
