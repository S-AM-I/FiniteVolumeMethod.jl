# kernels/enzyme_stub.jl — Stub for the Enzyme full-solver AD hook.

"""
    autodiff_forward_step(step_fn, state, dstate)

Stub dispatched by the `FVMEnzymeExt` weak-dep extension once
`Enzyme.jl` is loaded. In the pure-Julia path it warns + errors; for
v3.0 per-term AD via [`per_term_ad`](@ref) is the supported surface.
"""
function autodiff_forward_step(step_fn::Function, state, dstate)
    @warn "Enzyme full-solver AD deferred to v3.1 — per-term AD only" maxlog = 1
    return error("Enzyme.jl required for full-solver AD; use `per_term_ad` in v3.0")
end
