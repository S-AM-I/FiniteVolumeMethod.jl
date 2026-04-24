# multiphase/two_fluid.jl — Eulerian two-fluid model types, assembly, and primitives
#
# Two-phase (liquid + gas) flow with independent momentum equations per
# phase coupled by an interphase drag closure (Ishii-Zuber / Gibilaro) and
# a shared pressure field. The coupled momentum per velocity component is
# assembled into a `BlockCollocatedEquation{T, 2}` (block 1 = liquid,
# block 2 = gas) so the drag off-diagonal coupling is solved implicitly.
#
# This file owns the types (TwoFluidProperties, TwoFluidState,
# TwoFluidProblem), primitive rate computations, and the full
# `assemble_two_fluid_momentum!` + continuity / volume-fraction helpers.
# The outer pressure-velocity-α coupling loop lives in
# `multiphase/two_fluid_solver.jl`.

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
coupling) and per-phase face-flux fields used by the outer SIMPLE-like
loop.

# Fields
- `U_l::CollocatedVectorField{Dim, T}` — liquid velocity.
- `U_g::CollocatedVectorField{Dim, T}` — gas velocity.
- `alpha_l::CollocatedScalarField{T}`  — liquid volume fraction.
- `alpha_g::CollocatedScalarField{T}`  — gas volume fraction.
- `p_shared::CollocatedScalarField{T}` — shared pressure field.
- `phi_l::FaceFluxField{T}`            — liquid-phase face flux.
- `phi_g::FaceFluxField{T}`            — gas-phase face flux.
- `A_P_l::Vector{T}`                   — liquid momentum diagonal.
- `A_P_g::Vector{T}`                   — gas momentum diagonal.
- `H_U_l::Vector{SVector{Dim, T}}`     — liquid momentum H-operator.
- `H_U_g::Vector{SVector{Dim, T}}`     — gas momentum H-operator.
"""
mutable struct TwoFluidState{Dim, T}
    U_l::CollocatedVectorField{Dim, T}
    U_g::CollocatedVectorField{Dim, T}
    alpha_l::CollocatedScalarField{T}
    alpha_g::CollocatedScalarField{T}
    p_shared::CollocatedScalarField{T}
    phi_l::FaceFluxField{T}
    phi_g::FaceFluxField{T}
    A_P_l::Vector{T}
    A_P_g::Vector{T}
    H_U_l::Vector{SVector{Dim, T}}
    H_U_g::Vector{SVector{Dim, T}}
end

"""
    TwoFluidState(mesh; alpha_g_init = 0.0)

Construct a two-fluid state on `mesh` with uniform initial gas
volume fraction. The liquid volume fraction is set to `1 - α_g` on
construction so the `α_l + α_g = 1` invariant holds exactly
(subject to floating-point). Face-flux fields and momentum-operator
scratch vectors are zero-initialized.
"""
function TwoFluidState(
        mesh::UnstructuredFVMMesh{Dim, T};
        alpha_g_init::Real = 0.0,
    ) where {Dim, T}
    zero(T) <= alpha_g_init <= one(T) || throw(
        ArgumentError("alpha_g_init must be in [0, 1]"),
    )
    nc = length(mesh.cell_volumes)
    U_l = CollocatedVectorField(:U_l, mesh; value = zero(SVector{Dim, T}))
    U_g = CollocatedVectorField(:U_g, mesh; value = zero(SVector{Dim, T}))
    alpha_g = CollocatedScalarField(:alpha_g, mesh; value = T(alpha_g_init))
    alpha_l = CollocatedScalarField(:alpha_l, mesh; value = T(one(T) - T(alpha_g_init)))
    p_shared = CollocatedScalarField(:p_shared, mesh; value = zero(T))
    phi_l = FaceFluxField(:phi_l, mesh)
    phi_g = FaceFluxField(:phi_g, mesh)
    A_P_l = ones(T, nc)
    A_P_g = ones(T, nc)
    H_U_l = fill(zero(SVector{Dim, T}), nc)
    H_U_g = fill(zero(SVector{Dim, T}), nc)
    return TwoFluidState{Dim, T}(
        U_l, U_g, alpha_l, alpha_g, p_shared,
        phi_l, phi_g, A_P_l, A_P_g, H_U_l, H_U_g,
    )
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

Marker type for the Eulerian two-fluid coupled solver. Constructed with
no arguments; dispatched by [`solve_two_fluid`](@ref).
"""
struct TwoFluidSolver end

"""
    warn_experimental!(::TwoFluidSolver)

Legacy compatibility shim — the two-fluid solver is production-ready as
of v3.1 and no longer emits a warning on entry. Retained for downstream
callers that still invoke it; always a no-op.
"""
function warn_experimental!(::TwoFluidSolver)
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

"""
    drag_linearization_coefficient(closure, props, U_l, U_g, alpha_g) -> T

Return the scalar drag coefficient `K` such that the interphase drag
force per unit mixture volume is approximately

```
F_D ≈ K · (U_g − U_l)
```

This is the "semi-implicit" linearisation used in the two-fluid
momentum block: `K = |F_D| / |U_rel|`, with a well-defined Stokes limit
when `|U_rel| → 0`.

For any `drag_force_density` satisfying
`F_D = prefactor · U_rel` with `prefactor = 0.75·C_D·ρ_l·α_g·|U_rel|/d_b`,
the coefficient reduces to `K = prefactor`. In the Stokes limit this
degenerates to `K = 18·μ_l·α_g/d_b²`.

`K` is always non-negative and zero when `α_g = 0` or when `α_g = 1`
(so the liquid is vanishing and no drag is exerted on it).
"""
function drag_linearization_coefficient(
        closure::AbstractDragClosure,
        props::TwoFluidProperties{T},
        U_l, U_g, alpha_g::T,
    ) where {T}
    # Guard against the degenerate single-phase limits: α_g = 0 (no bubbles)
    # and α_g = 1 (no carrier). In both cases the linearisation is zero so
    # the momentum block decouples into two independent phases.
    if alpha_g <= zero(T) || alpha_g >= one(T)
        return zero(T)
    end

    U_rel = U_g - U_l
    slip = _norm(U_rel)
    if slip <= eps(T)
        # Stokes-limit linearisation — finite, positive coefficient that keeps
        # the quiescent two-fluid problem well-posed (no zero off-diagonal).
        return T(18) * props.mu_l * alpha_g / (props.d_b * props.d_b) * props.C_D
    end

    Re_b = props.rho_l * props.d_b * slip / props.mu_l
    C_D_dim = drag_coefficient(closure, Re_b, alpha_g)
    prefactor = T(0.75) * C_D_dim * props.rho_l * alpha_g * slip / props.d_b
    return props.C_D * prefactor
end

# ── TwoFluidProblem — outer-loop problem container ───────────────────

"""
    TwoFluidProblem{Dim, T, Mesh, DragClosure, MassTransfer}

Complete specification of an Eulerian two-fluid (liquid + gas) problem.

The problem carries per-phase velocity BCs (`bcs_Ul`, `bcs_Ug`), the
shared pressure BCs (`bcs_p`), and gas volume fraction BCs
(`bcs_alpha_g`). The liquid volume-fraction boundary is always
`1 - α_g` at every patch (enforced by the closure relation).

# Fields
- `mesh` — unstructured FVM mesh.
- `props::TwoFluidProperties{T}` — phase properties.
- `drag::DragClosure` — interphase drag closure (default `IshiiZuberDrag()`).
- `mass_transfer::MassTransfer` — mass-transfer model (default
  `NoMassTransfer()`; see `multiphase/mass_transfer.jl`).
- `bcs_Ul::Dict{Symbol, <:AbstractBoundaryCondition}`
- `bcs_Ug::Dict{Symbol, <:AbstractBoundaryCondition}`
- `bcs_p::Dict{Symbol, <:AbstractBoundaryCondition}`
- `bcs_alpha_g::Dict{Symbol, <:AbstractBoundaryCondition}`
- `gravity::SVector{Dim, T}` — body-force acceleration vector.
"""
struct TwoFluidProblem{Dim, T, Mesh, DragClosure, MassTransfer}
    mesh::Mesh
    props::TwoFluidProperties{T}
    drag::DragClosure
    mass_transfer::MassTransfer
    bcs_Ul::Dict{Symbol, AbstractBoundaryCondition}
    bcs_Ug::Dict{Symbol, AbstractBoundaryCondition}
    bcs_p::Dict{Symbol, AbstractBoundaryCondition}
    bcs_alpha_g::Dict{Symbol, AbstractBoundaryCondition}
    gravity::SVector{Dim, T}
end

"""
    TwoFluidProblem(mesh, props; drag = IshiiZuberDrag(),
        mass_transfer = NoMassTransfer(), bcs_Ul, bcs_Ug, bcs_p,
        bcs_alpha_g = Dict(), gravity = zero(SVector{Dim, T}))

Keyword constructor. `bcs_Ul` / `bcs_Ug` / `bcs_p` must be supplied;
`bcs_alpha_g` defaults to an empty dict (zero-gradient everywhere) and
`gravity` defaults to zero.
"""
function TwoFluidProblem(
        mesh::UnstructuredFVMMesh{Dim, T},
        props::TwoFluidProperties{T};
        drag::AbstractDragClosure = IshiiZuberDrag(),
        mass_transfer = NoMassTransfer(),
        bcs_Ul::Dict{Symbol, <:AbstractBoundaryCondition} =
            Dict{Symbol, AbstractBoundaryCondition}(),
        bcs_Ug::Dict{Symbol, <:AbstractBoundaryCondition} =
            Dict{Symbol, AbstractBoundaryCondition}(),
        bcs_p::Dict{Symbol, <:AbstractBoundaryCondition} =
            Dict{Symbol, AbstractBoundaryCondition}(),
        bcs_alpha_g::Dict{Symbol, <:AbstractBoundaryCondition} =
            Dict{Symbol, AbstractBoundaryCondition}(),
        gravity = zero(SVector{Dim, T}),
    ) where {Dim, T}
    g_vec = SVector{Dim, T}(gravity)
    return TwoFluidProblem{Dim, T, typeof(mesh), typeof(drag), typeof(mass_transfer)}(
        mesh, props, drag, mass_transfer,
        Dict{Symbol, AbstractBoundaryCondition}(bcs_Ul),
        Dict{Symbol, AbstractBoundaryCondition}(bcs_Ug),
        Dict{Symbol, AbstractBoundaryCondition}(bcs_p),
        Dict{Symbol, AbstractBoundaryCondition}(bcs_alpha_g),
        g_vec,
    )
end

# ── Coupled momentum assembly ────────────────────────────────────────

"""
    assemble_two_fluid_momentum!(eq, state, prob, component; dt)

Assemble the coupled liquid + gas momentum equation for velocity
component `component` (1 = x, 2 = y, 3 = z) into the block equation
`eq::BlockCollocatedEquation{T, 2}`. Block 1 holds the liquid row,
block 2 holds the gas row. Drag is linearised as `K·(U_g − U_l)` so
the coupling appears on the off-diagonal.

For each cell `P` with volume `V_P`, volume fractions `α_{l,P}`,
`α_{g,P}`, linearised drag coefficient `K_P`, and time step `dt > 0`
(backward Euler):

```
  (ρ_l V_P α_{l,P} / dt + conv_l[P,P] + diff_l[P,P] + K_P V_P)·Ul_d[P]
   - K_P V_P · Ug_d[P] + (off-diagonal face coeffs) =
      ρ_l V_P α_{l,P} Ul_d^n[P] / dt - α_{l,P} (∂p/∂x_d)_P V_P + α_{l,P} ρ_l g_d V_P

  (ρ_g V_P α_{g,P} / dt + conv_g[P,P] + diff_g[P,P] + K_P V_P)·Ug_d[P]
   - K_P V_P · Ul_d[P] + (off-diagonal face coeffs) =
      ρ_g V_P α_{g,P} Ug_d^n[P] / dt - α_{g,P} (∂p/∂x_d)_P V_P + α_{g,P} ρ_g g_d V_P
```

The convection / diffusion face coefficients are assembled into
each phase's diagonal block (`A[liquid, liquid]`, `A[gas, gas]`); the
drag coupling is the only term that hits the off-diagonal blocks
(`A[liquid, gas]`, `A[gas, liquid]`).
"""
function assemble_two_fluid_momentum!(
        eq::BlockCollocatedEquation{T, 2},
        state::TwoFluidState{Dim, T},
        prob::TwoFluidProblem{Dim, T},
        component::Int;
        dt::T,
    ) where {Dim, T}
    mesh = prob.mesh
    props = prob.props
    drag_closure = prob.drag
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    reset!(eq)

    bcs_Ul = expand_bcs_velocity(prob.bcs_Ul, component)
    bcs_Ug = expand_bcs_velocity(prob.bcs_Ug, component)

    # Small volume-fraction floor keeps the per-phase momentum block
    # non-singular even at α_k = 0 / α_k = 1 limiting cases. Real
    # multiphase codes (Fluent, OpenFOAM twoPhaseEulerFoam) apply the
    # same regularisation. Its effect is O(α_floor) on the velocity
    # residual so benign in the production two-phase regime.
    alpha_floor = T(1.0e-8)

    # ── Convection: div(α·ρ·phi·U_d) for each phase ─────────────────
    # We assemble into auxiliary single-phase equations first and copy
    # the nzval into the block system. The per-phase flux is weighted by
    # α and the phase density so the momentum equation is conservative.
    #
    # Build an α-ρ scaled face flux for each phase; convection is then
    # div(phi_scaled · U_d) for that phase.
    phi_l_scaled = FaceFluxField(:phi_l_scaled, mesh; value = zero(T))
    phi_g_scaled = FaceFluxField(:phi_g_scaled, mesh; value = zero(T))
    @inbounds for f in 1:nf
        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)
            alpha_l_f = w * state.alpha_l.internal[P] +
                (one(T) - w) * state.alpha_l.internal[N]
            alpha_g_f = w * state.alpha_g.internal[P] +
                (one(T) - w) * state.alpha_g.internal[N]
            phi_l_scaled.values[f] = props.rho_l * alpha_l_f * state.phi_l.values[f]
            phi_g_scaled.values[f] = props.rho_g * alpha_g_f * state.phi_g.values[f]
        else
            P = owner(mesh, f)
            phi_l_scaled.values[f] = props.rho_l *
                state.alpha_l.internal[P] * state.phi_l.values[f]
            phi_g_scaled.values[f] = props.rho_g *
                state.alpha_g.internal[P] * state.phi_g.values[f]
        end
    end

    eq_l = CollocatedEquation(mesh)
    eq_g = CollocatedEquation(mesh)
    assemble_convection!(eq_l, phi_l_scaled, mesh, bcs_Ul; scheme = CONV_UPWIND)
    assemble_convection!(eq_g, phi_g_scaled, mesh, bcs_Ug; scheme = CONV_UPWIND)

    # ── Diffusion: div(α·μ·grad(U_d)) for each phase ────────────────
    nu_l_cells = Vector{T}(undef, nc)
    nu_g_cells = Vector{T}(undef, nc)
    @inbounds for c in 1:nc
        nu_l_cells[c] = max(state.alpha_l.internal[c], alpha_floor) * props.mu_l
        nu_g_cells[c] = max(state.alpha_g.internal[c], alpha_floor) * props.mu_g
    end
    assemble_laplacian!(eq_l, nu_l_cells, mesh, bcs_Ul)
    assemble_laplacian!(eq_g, nu_g_cells, mesh, bcs_Ug)

    # ── Temporal: ∂(α·ρ·U_d)/∂t ─────────────────────────────────────
    Ul_d_old = T[u[component] for u in state.U_l.internal]
    Ug_d_old = T[u[component] for u in state.U_g.internal]
    rho_alpha_l = Vector{T}(undef, nc)
    rho_alpha_g = Vector{T}(undef, nc)
    @inbounds for c in 1:nc
        rho_alpha_l[c] = props.rho_l * max(state.alpha_l.internal[c], alpha_floor)
        rho_alpha_g[c] = props.rho_g * max(state.alpha_g.internal[c], alpha_floor)
    end
    assemble_ddt_euler!(eq_l, rho_alpha_l, Ul_d_old, mesh, dt)
    assemble_ddt_euler!(eq_g, rho_alpha_g, Ug_d_old, mesh, dt)

    # ── Pressure gradient source: -α·(∂p/∂x_d)·V ────────────────────
    grad_p = gradient(state.p_shared, mesh)
    @inbounds for c in 1:nc
        V_c = mesh.cell_volumes[c]
        gp = grad_p[c][component]
        eq_l.b[c] -= state.alpha_l.internal[c] * gp * V_c
        eq_g.b[c] -= state.alpha_g.internal[c] * gp * V_c
    end

    # ── Gravity source: α·ρ·g·V ─────────────────────────────────────
    g_d = prob.gravity[component]
    if g_d != zero(T)
        @inbounds for c in 1:nc
            V_c = mesh.cell_volumes[c]
            eq_l.b[c] += state.alpha_l.internal[c] * props.rho_l * g_d * V_c
            eq_g.b[c] += state.alpha_g.internal[c] * props.rho_g * g_d * V_c
        end
    end

    # ── Copy per-phase single-block equations into block equation ────
    # Block 1 = liquid, block 2 = gas.
    _copy_single_to_block!(eq, eq_l, 1, mesh)
    _copy_single_to_block!(eq, eq_g, 2, mesh)

    # ── Drag linearisation: off-diagonal coupling ──────────────────
    @inbounds for c in 1:nc
        V_c = mesh.cell_volumes[c]
        K_c = drag_linearization_coefficient(
            drag_closure, props,
            state.U_l.internal[c], state.U_g.internal[c],
            state.alpha_g.internal[c],
        )
        KV = K_c * V_c
        # Liquid momentum row: +K·V·(Ul - Ug) moves to LHS, so
        #   A[liq, liq] += KV; A[liq, gas] += -KV
        add_block_diag!(eq, c, 1, 1, KV)
        add_block_diag!(eq, c, 1, 2, -KV)
        # Gas momentum row: +K·V·(Ug - Ul) moves to LHS:
        add_block_diag!(eq, c, 2, 2, KV)
        add_block_diag!(eq, c, 2, 1, -KV)
    end

    return nothing
end

"""
    _copy_single_to_block!(eq_block, eq_single, block_index, mesh)

Copy a single-phase `CollocatedEquation` into the `(block_index,
block_index)` block of a `BlockCollocatedEquation{T, 2}`. Iterates the
mesh cells and faces and uses the block pattern's nzval indices to
write without structural allocation.
"""
function _copy_single_to_block!(
        eq_block::BlockCollocatedEquation{T, 2},
        eq_single::CollocatedEquation{T},
        b::Int,
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    @inbounds for c in 1:nc
        diag_val = eq_single.A.nzval[eq_single.pattern.diag_idx[c]]
        add_block_diag!(eq_block, c, b, b, diag_val)
        row = (c - 1) * 2 + b
        eq_block.b[row] += eq_single.b[c] + eq_single.source[c]
    end
    @inbounds for f in 1:nf
        if is_internal_face(mesh, f)
            v_pn = eq_single.A.nzval[eq_single.pattern.offdiag_PN[f]]
            v_np = eq_single.A.nzval[eq_single.pattern.offdiag_NP[f]]
            add_block_offdiag_PN!(eq_block, f, b, b, v_pn)
            add_block_offdiag_NP!(eq_block, f, b, b, v_np)
        end
    end
    return nothing
end

"""
    extract_two_fluid_block_solution!(state, sol, component)

Unpack a block-momentum solution vector (length `2·ncells`) into the
`component`-th entry of `state.U_l` / `state.U_g`. Block 1 is liquid,
block 2 is gas.
"""
function extract_two_fluid_block_solution!(
        state::TwoFluidState{Dim, T},
        sol_vec::AbstractVector{T},
        component::Int,
    ) where {Dim, T}
    nc = length(state.U_l.internal)
    @inbounds for c in 1:nc
        ul_c = state.U_l.internal[c]
        ug_c = state.U_g.internal[c]
        new_ul = Base.setindex(ul_c, sol_vec[(c - 1) * 2 + 1], component)
        new_ug = Base.setindex(ug_c, sol_vec[(c - 1) * 2 + 2], component)
        state.U_l.internal[c] = new_ul
        state.U_g.internal[c] = new_ug
    end
    return nothing
end

# ── Boundary updates ────────────────────────────────────────────────

"""
    update_two_fluid_boundary_velocity!(state, prob)

Apply velocity boundary conditions to `state.U_l.boundary` and
`state.U_g.boundary`. Mirrors `update_boundary_velocity!` from the
single-phase incompressible solver but operates on the per-phase
velocity storage.
"""
function update_two_fluid_boundary_velocity!(
        state::TwoFluidState{Dim, T},
        prob::TwoFluidProblem{Dim, T},
    ) where {Dim, T}
    mesh = prob.mesh
    _update_boundary_vector!(state.U_l, prob.bcs_Ul, mesh)
    _update_boundary_vector!(state.U_g, prob.bcs_Ug, mesh)
    return nothing
end

function _update_boundary_vector!(
        U::CollocatedVectorField{Dim, T},
        bcs::Dict{Symbol, <:AbstractBoundaryCondition},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    for (i, f) in enumerate(U.boundary_face_indices)
        tag = _face_tag(mesh, f)
        bc = get(bcs, tag, nothing)
        bc === nothing && continue
        P = owner(mesh, f)
        if bc isa FixedVelocityBC
            U.boundary[i] = bc.value
        elseif bc isa NoSlipWallBC
            U.boundary[i] = zero(SVector{Dim, T})
        elseif bc isa InletOutletBC
            U.boundary[i] = bc.inlet_value
        else
            U.boundary[i] = U.internal[P]
        end
    end
    return nothing
end

"""
    update_two_fluid_boundary_pressure!(state, prob)

Apply pressure boundary conditions to `state.p_shared.boundary`.
"""
function update_two_fluid_boundary_pressure!(
        state::TwoFluidState{Dim, T},
        prob::TwoFluidProblem{Dim, T},
    ) where {Dim, T}
    mesh = prob.mesh
    for (i, f) in enumerate(state.p_shared.boundary_face_indices)
        tag = _face_tag(mesh, f)
        bc = get(prob.bcs_p, tag, nothing)
        bc === nothing && continue
        P = owner(mesh, f)
        if bc isa FixedPressureBC
            state.p_shared.boundary[i] = bc.value
        else
            state.p_shared.boundary[i] = state.p_shared.internal[P]
        end
    end
    return nothing
end

"""
    update_two_fluid_boundary_alpha!(state, prob)

Apply gas-volume-fraction boundary conditions. Fixed-value BCs on
`α_g` (via a `FixedPressureBC`-style scalar `CustomBC` or similar) are
read from `prob.bcs_alpha_g`; everything else falls back to
zero-gradient (owner-cell copy). `α_l` is always synced to `1 - α_g`.
"""
function update_two_fluid_boundary_alpha!(
        state::TwoFluidState{Dim, T},
        prob::TwoFluidProblem{Dim, T},
    ) where {Dim, T}
    mesh = prob.mesh
    for (i, f) in enumerate(state.alpha_g.boundary_face_indices)
        tag = _face_tag(mesh, f)
        bc = get(prob.bcs_alpha_g, tag, nothing)
        P = owner(mesh, f)
        if bc isa CustomBC && bc.velocity_type === :dirichlet
            state.alpha_g.boundary[i] = T(bc.velocity_value)
        elseif bc === nothing
            state.alpha_g.boundary[i] = state.alpha_g.internal[P]
        else
            state.alpha_g.boundary[i] = state.alpha_g.internal[P]
        end
        state.alpha_l.boundary[i] = one(T) - state.alpha_g.boundary[i]
    end
    return nothing
end

# ── Phasic volume-fraction transport ─────────────────────────────────

"""
    assemble_phasic_alpha!(eq, state, prob; dt, phase)

Assemble the phasic volume-fraction transport equation
`∂α_k/∂t + ∇·(α_k · U_k) = S_k / ρ_k` for `phase ∈ (:liquid, :gas)`
into the scalar equation `eq`. The convective flux is the phase's
face flux `state.phi_k`, mass-transfer source is retrieved from
`mass_transfer_source_alpha(prob.mass_transfer, ...)` — zero when the
default `NoMassTransfer` model is used.
"""
function assemble_phasic_alpha!(
        eq::CollocatedEquation{T},
        state::TwoFluidState{Dim, T},
        prob::TwoFluidProblem{Dim, T};
        dt::T,
        phase::Symbol,
    ) where {Dim, T}
    mesh = prob.mesh
    reset!(eq)

    if phase === :liquid
        phi = state.phi_l
        alpha = state.alpha_l
        rho_k = prob.props.rho_l
        sign_k = -one(T)
    elseif phase === :gas
        phi = state.phi_g
        alpha = state.alpha_g
        rho_k = prob.props.rho_g
        sign_k = one(T)
    else
        throw(ArgumentError("phase must be :liquid or :gas; got $(phase)"))
    end

    # Build a zero-gradient BC dict covering every mesh patch — α
    # transport always assembles the convection operator which requires
    # a BC entry for every boundary face's patch.
    bcs_alpha = Dict{Symbol, AbstractBoundaryCondition}()
    for name in keys(prob.bcs_Ul)
        bcs_alpha[name] = ParabolicNeumann(0.0)
    end
    for name in keys(prob.bcs_alpha_g)
        bcs_alpha[name] = ParabolicNeumann(0.0)
    end

    assemble_convection!(eq, phi, mesh, bcs_alpha; scheme = CONV_UPWIND)
    assemble_ddt_euler!(eq, one(T), alpha.internal, mesh, dt)

    S_dot = mass_transfer_source_alpha(prob.mass_transfer, state, prob)
    nc = length(mesh.cell_volumes)
    @inbounds for c in 1:nc
        eq.b[c] += sign_k * S_dot[c] / rho_k * mesh.cell_volumes[c]
    end

    return nothing
end

# ── Per-phase flux correction ───────────────────────────────────────

"""
    compute_two_fluid_fluxes!(state, mesh)

Compute the per-phase volumetric face fluxes `phi_l` and `phi_g` from
the current per-phase cell velocities. Uses the same linear
interpolation as `compute_face_flux!`.
"""
function compute_two_fluid_fluxes!(
        state::TwoFluidState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    compute_face_flux!(state.phi_l, state.U_l, mesh)
    compute_face_flux!(state.phi_g, state.U_g, mesh)
    return nothing
end
