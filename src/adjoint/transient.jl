# adjoint/transient.jl — Transient PIMPLE adjoint (v3.1 research stub).
#
# Transient adjoint for nonlinear pressure-velocity-coupled schemes
# requires full-trajectory checkpointing (Griewank, Walther 2000) and a
# time-reverse sweep over the inner correctors. Ship as a marker that
# warns + errors so callers fail loudly rather than silently returning
# zero gradients.

"""
    solve_transient_adjoint(args...; kwargs...)

Stub for the transient PIMPLE discrete adjoint. Deferred to v3.1 — the
underlying research problem (reverse-mode differentiation through a
nonlinear inner-corrector loop with checkpointing) is out of scope for
v3.0. Use finite-difference gradients
([`verify_adjoint_gradient`](@ref)) for design studies in the meantime.
"""
function solve_transient_adjoint(args...; kwargs...)
    @warn "Transient PIMPLE adjoint deferred to v3.1 — requires checkpointing" maxlog = 1
    return error(
        "Transient adjoint not implemented; use finite-difference (verify_adjoint_gradient) for design studies in v3.0",
    )
end
