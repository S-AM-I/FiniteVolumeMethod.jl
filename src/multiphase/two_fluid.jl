# multiphase/two_fluid.jl — Eulerian two-fluid model types and rates
#
# Separate momentum + continuity equations per phase (liquid, gas)
# coupled via drag closure (Ishii-Zuber / Gibilaro). The coupled
# momentum block is intended to be assembled into the existing
# `BlockCollocatedEquation{T, 2}` (one block per phase); this file
# owns only the types, invariants, and primitive rate computations.
#
# `TwoFluidSolver` is marked **experimental** — production-hardened
# SIMPLE-like two-fluid solver is deferred to v3.1. The coupled
# pressure-velocity-volume-fraction block system is a research-grade
# placeholder at this stage.

"""
    TwoFluidProperties{T}

Physical properties for an Eulerian two-fluid (liquid / gas) flow.

# Fields
- `rho_l::T` — liquid-phase density.
- `rho_g::T` — gas-phase density.
- `mu_l::T`  — liquid-phase dynamic viscosity.
- `mu_g::T`  — gas-phase dynamic viscosity.
- `sigma::T` — interfacial tension.
- `d_b::T`   — characteristic bubble diameter (dispersed phase).
- `C_D::T`   — drag-coefficient scaling multiplier (defaults to 1).
"""
struct TwoFluidProperties{T}
    rho_l::T
    rho_g::T
    mu_l::T
    mu_g::T
    sigma::T
    d_b::T
    C_D::T
end

"""
    TwoFluidProperties(; rho_l, rho_g, mu_l, mu_g, sigma, d_b, C_D = 1)

Keyword constructor with typical water/air-at-20°C defaults. Validates
that phase properties are positive and that the liquid is denser than
the gas (which is what identifies the liquid phase — swap if needed).
"""
function TwoFluidProperties(;
        rho_l::Real = 1000.0,
        rho_g::Real = 1.225,
        mu_l::Real = 1.0e-3,
        mu_g::Real = 1.8e-5,
        sigma::Real = 0.072,
        d_b::Real = 1.0e-3,
        C_D::Real = 1.0,
    )
    T = promote_type(
        typeof(rho_l), typeof(rho_g),
        typeof(mu_l), typeof(mu_g),
        typeof(sigma), typeof(d_b), typeof(C_D),
    )
    rho_l > 0 || throw(ArgumentError("rho_l must be positive"))
    rho_g > 0 || throw(ArgumentError("rho_g must be positive"))
    mu_l > 0 || throw(ArgumentError("mu_l must be positive"))
    mu_g > 0 || throw(ArgumentError("mu_g must be positive"))
    d_b > 0 || throw(ArgumentError("d_b (bubble diameter) must be positive"))
    return TwoFluidProperties{T}(
        T(rho_l), T(rho_g), T(mu_l), T(mu_g),
        T(sigma), T(d_b), T(C_D),
    )
end

"""
    density_ratio(props)

Return `ρ_l / ρ_g`. Typical water/air value is ~816.
"""
density_ratio(p::TwoFluidProperties) = p.rho_l / p.rho_g

"""
    TwoFluidState{Dim, T}

Mutable Eulerian two-fluid state — per-phase velocities and volume
fractions plus a single shared pressure (mixture-model pressure
coupling). Expected to be wired into `BlockCollocatedEquation{T, 2}`
for the coupled momentum system.

# Fields
- `U_l::CollocatedVectorField{Dim, T}` — liquid velocity.
- `U_g::CollocatedVectorField{Dim, T}` — gas velocity.
- `alpha_l::CollocatedScalarField{T}`  — liquid volume fraction.
- `alpha_g::CollocatedScalarField{T}`  — gas volume fraction.
- `p_shared::CollocatedScalarField{T}` — shared pressure field.
"""
mutable struct TwoFluidState{Dim, T}
    U_l::CollocatedVectorField{Dim, T}
    U_g::CollocatedVectorField{Dim, T}
    alpha_l::CollocatedScalarField{T}
    alpha_g::CollocatedScalarField{T}
    p_shared::CollocatedScalarField{T}
end

"""
    TwoFluidState(mesh; alpha_g_init = 0.0)

Construct a two-fluid state on `mesh` with uniform initial gas
volume fraction. The liquid volume fraction is set to `1 - α_g` on
construction so the `α_l + α_g = 1` invariant holds exactly
(subject to floating-point).
"""
function TwoFluidState(
        mesh::UnstructuredFVMMesh{Dim, T};
        alpha_g_init::Real = 0.0,
    ) where {Dim, T}
    zero(T) <= alpha_g_init <= one(T) || throw(
        ArgumentError("alpha_g_init must be in [0, 1]"),
    )
    U_l = CollocatedVectorField(:U_l, mesh; value = zero(SVector{Dim, T}))
    U_g = CollocatedVectorField(:U_g, mesh; value = zero(SVector{Dim, T}))
    alpha_g = CollocatedScalarField(:alpha_g, mesh; value = T(alpha_g_init))
    alpha_l = CollocatedScalarField(:alpha_l, mesh; value = T(one(T) - T(alpha_g_init)))
    p_shared = CollocatedScalarField(:p_shared, mesh; value = zero(T))
    return TwoFluidState{Dim, T}(U_l, U_g, alpha_l, alpha_g, p_shared)
end

"""
    enforce_volume_fraction_sum!(state; atol = 1e-10)

Assert (and restore to floating-point tolerance) the closure relation

```
α_l + α_g = 1  in every cell.
```

Walks both internal and boundary storage and resets `α_l` from
`α_g` so the invariant holds exactly on return. Returns the maximum
deviation observed before correction (useful for regression tests).
"""
function enforce_volume_fraction_sum!(state::TwoFluidState{Dim, T}) where {Dim, T}
    max_dev = zero(T)
    @inbounds for c in eachindex(state.alpha_g.internal)
        s = state.alpha_l.internal[c] + state.alpha_g.internal[c]
        max_dev = max(max_dev, abs(s - one(T)))
        state.alpha_l.internal[c] = one(T) - state.alpha_g.internal[c]
    end
    @inbounds for i in eachindex(state.alpha_g.boundary)
        s = state.alpha_l.boundary[i] + state.alpha_g.boundary[i]
        max_dev = max(max_dev, abs(s - one(T)))
        state.alpha_l.boundary[i] = one(T) - state.alpha_g.boundary[i]
    end
    return max_dev
end

"""
    TwoFluidSolver

Marker type for the Eulerian two-fluid coupled solver.

**Experimental** — types, invariants, and drag closures are
primitive-test covered here; the full coupled SIMPLE-like momentum /
pressure / volume-fraction loop is deferred to v3.1. Users selecting
this solver receive a one-shot warning on entry.
"""
struct TwoFluidSolver end

"""
    warn_experimental!(::TwoFluidSolver)

Emit the one-shot experimental warning used at solver entry.
"""
function warn_experimental!(::TwoFluidSolver)
    @warn "Eulerian two-fluid: experimental; production-hardening deferred to v3.1" maxlog = 1
    return nothing
end

"""
    interphase_drag(props, U_l, U_g, alpha_g; closure = IshiiZuberDrag())

Convenience wrapper that feeds `TwoFluidProperties` into
[`drag_force_density`](@ref). Uses `U_rel = U_g − U_l`, the
continuous-phase liquid as the carrier, and the bubble diameter +
liquid viscosity from `props`. The returned vector has the same
shape as `U_l`/`U_g`.
"""
function interphase_drag(
        props::TwoFluidProperties{T}, U_l, U_g, alpha_g::T;
        closure::AbstractDragClosure = IshiiZuberDrag(),
    ) where {T}
    U_rel = U_g - U_l
    F = drag_force_density(closure, props.rho_l, U_rel, alpha_g, props.d_b, props.mu_l)
    return props.C_D * F
end
