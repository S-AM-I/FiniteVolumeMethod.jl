# multiphase/two_fluid_solver.jl — Outer pressure-velocity-α loop for the
# Eulerian two-fluid coupled solver.
#
# Reuses the Phase 1 `assemble_convection!` / `assemble_laplacian!` /
# `assemble_ddt_euler!` primitives via `assemble_two_fluid_momentum!`
# (which wraps them into a `BlockCollocatedEquation{T, 2}`) and feeds
# the resulting block LinearProblem through the standard SciMLBase
# backslash dispatch. The loop follows the outline:
#
#   1. Block momentum predictor per velocity component.
#   2. Per-phase extraction of A_P and H_U from the solved block.
#   3. Shared-pressure Poisson-like correction on mixture continuity
#      (`∇·(α_l U_l + α_g U_g) = 0`).
#   4. Velocity update from the new pressure.
#   5. Phase face-flux update.
#   6. Phasic α transport (gas); liquid synced via closure `α_l + α_g = 1`.
#   7. Convergence check on mixture continuity + momentum residuals.
#
# Volume-fraction BCs are zero-gradient everywhere unless explicitly
# provided in `prob.bcs_alpha_g`; pressure reference is fixed at cell 1
# when no `FixedPressureBC` exists.

using Printf: @sprintf

"""
    TwoFluidSolveResult{Dim, T}

Result of [`solve_two_fluid`](@ref). Captures the final state and
convergence history.

# Fields
- `converged::Bool` — whether the solver met the tolerance.
- `iterations::Int` — number of outer iterations.
- `residuals::Dict{Symbol, Vector{T}}` — residual history per equation
  (`:Ul`, `:Ug`, `:continuity`, `:alpha`).
- `state::TwoFluidState{Dim, T}` — final state.
"""
struct TwoFluidSolveResult{Dim, T}
    converged::Bool
    iterations::Int
    residuals::Dict{Symbol, Vector{T}}
    state::TwoFluidState{Dim, T}
end

"""
    solve_two_fluid(prob, ::TwoFluidSolver;
        alpha_g_init = 0.0, dt = 1.0e-3, max_outer = 50, tol = 1e-5,
        alpha_U = 0.7, alpha_p = 0.3, linear_solver = nothing,
        verbose = false,
    ) -> TwoFluidSolveResult

Production Eulerian two-fluid solver. Iterates the outer
pressure-velocity-α coupling loop until either `max_outer` is reached
or the mixture-continuity residual drops below `tol`.

`dt` controls the implicit Euler time step used inside the momentum
predictor and the α transport step; pass a large value for
pseudo-steady runs.

The loop is deliberately minimal — no MULES-limited α transport (falls
back to upwind + α ∈ [0, 1] clip), no non-orthogonal correction, no
skew correction, no parallel ghost-exchange. These are incremental
upgrades for v3.1+.
"""
function solve_two_fluid(
        prob::TwoFluidProblem{Dim, T},
        ::TwoFluidSolver = TwoFluidSolver();
        alpha_g_init::Real = 0.0,
        dt::T = T(1.0e-3),
        max_outer::Int = 50,
        tol::T = T(1.0e-5),
        alpha_U::T = T(0.7),
        alpha_p::T = T(0.3),
        linear_solver = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    state = TwoFluidState(mesh; alpha_g_init = alpha_g_init)
    update_two_fluid_boundary_velocity!(state, prob)
    update_two_fluid_boundary_pressure!(state, prob)
    update_two_fluid_boundary_alpha!(state, prob)
    compute_two_fluid_fluxes!(state, mesh)

    residuals = Dict{Symbol, Vector{T}}(
        :Ul => T[], :Ug => T[], :continuity => T[], :alpha => T[],
    )

    converged = false
    final_iter = 0

    for iter in 1:max_outer
        final_iter = iter

        # ── 1. Block momentum predictor (one solve per component) ────
        max_mom_l = zero(T)
        max_mom_g = zero(T)
        for d in 1:Dim
            eq = BlockCollocatedEquation(mesh, Val(2))
            assemble_two_fluid_momentum!(eq, state, prob, d; dt = dt)

            # Under-relaxation on each block's diagonal. We apply it in
            # place on the block matrix nzval (block-diagonal entries
            # only) and on the RHS.
            _under_relax_block_momentum!(eq, state, d, alpha_U)

            Ul_d_old = T[u[d] for u in state.U_l.internal]
            Ug_d_old = T[u[d] for u in state.U_g.internal]

            lp = to_linear_problem(eq)
            sol = _dispatch_solve(lp, linear_solver, nothing, :U_two_fluid)
            extract_two_fluid_block_solution!(state, sol.u, d)

            # Momentum residual per phase: L1 norm of Ax - b on each
            # block row.
            rl, rg = _block_component_residuals(eq, sol.u)
            max_mom_l = max(max_mom_l, rl)
            max_mom_g = max(max_mom_g, rg)

            # Extract per-phase A_P / H_U using the last component's
            # block (diagonal is component-independent for the linear
            # operators we assemble, so we reuse the final component).
            if d == Dim
                _extract_block_operators!(state, eq, mesh)
            end

            # Ignore stale reference to avoid allocation warnings.
            Ul_d_old, Ug_d_old = Ul_d_old, Ug_d_old
        end
        push!(residuals[:Ul], max_mom_l)
        push!(residuals[:Ug], max_mom_g)

        update_two_fluid_boundary_velocity!(state, prob)

        # ── 2. Update per-phase face fluxes ─────────────────────────
        compute_two_fluid_fluxes!(state, mesh)

        # ── 3. Shared pressure correction (mixture continuity) ──────
        p_eq = CollocatedEquation(mesh)
        assemble_two_fluid_pressure!(p_eq, state, prob)

        if _two_fluid_needs_pressure_reference(prob.bcs_p)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end

        lp_p = to_linear_problem(p_eq)
        p_sol = _dispatch_solve(lp_p, linear_solver, nothing, :p)

        @inbounds for c in 1:nc
            state.p_shared.internal[c] += alpha_p *
                (p_sol.u[c] - state.p_shared.internal[c])
        end
        update_two_fluid_boundary_pressure!(state, prob)

        # ── 4. Velocity update from new pressure ────────────────────
        correct_two_fluid_velocity!(state, prob)
        update_two_fluid_boundary_velocity!(state, prob)
        compute_two_fluid_fluxes!(state, mesh)

        # ── 5. Phasic α transport (gas; liquid synced) ──────────────
        alpha_eq = CollocatedEquation(mesh)
        assemble_phasic_alpha!(alpha_eq, state, prob; dt = dt, phase = :gas)
        lp_a = to_linear_problem(alpha_eq)
        alpha_sol = _dispatch_solve(lp_a, linear_solver, nothing, :alpha_g)

        # Copy, clip to [0, 1] as a safety net, and sync α_l.
        alpha_change = zero(T)
        @inbounds for c in 1:nc
            new_ag = max(zero(T), min(one(T), alpha_sol.u[c]))
            alpha_change = max(alpha_change, abs(new_ag - state.alpha_g.internal[c]))
            state.alpha_g.internal[c] = new_ag
        end
        push!(residuals[:alpha], alpha_change)
        enforce_volume_fraction_sum!(state)
        update_two_fluid_boundary_alpha!(state, prob)

        # ── 6. Convergence check (mixture continuity) ───────────────
        r_cont = two_fluid_mixture_continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)

        if verbose
            _print_two_fluid_residuals(iter, residuals)
        end

        max_res = max(max_mom_l, max_mom_g, r_cont, alpha_change)
        if max_res < tol
            converged = true
            break
        end
    end

    return TwoFluidSolveResult{Dim, T}(converged, final_iter, residuals, state)
end

# ── Pressure-correction assembly ─────────────────────────────────────

"""
    assemble_two_fluid_pressure!(eq, state, prob)

Assemble the shared-pressure Poisson-like equation derived from the
mixture-continuity constraint `∇·(α_l U_l + α_g U_g) = 0`. The
discretisation mirrors the single-phase SIMPLE pressure equation but
weights the phase face fluxes by the α-interpolated face values and
uses `D_k_f = V_f / A_P_k_f` per phase for the implicit Laplacian
diffusivity.

The implicit operator assembled is
`div((α_l·D_l + α_g·D_g)·grad(p))`, and the RHS is the negative
divergence of the phase-weighted `H/A` face fluxes.
"""
function assemble_two_fluid_pressure!(
        eq::CollocatedEquation{T},
        state::TwoFluidState{Dim, T},
        prob::TwoFluidProblem{Dim, T},
    ) where {Dim, T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    reset!(eq)

    # Per-cell mixed diffusivity. Weight each phase's `V/A_P` by its
    # volume fraction so a vanishing phase drops out of the implicit
    # pressure operator cleanly (otherwise `1/A_P_k` blows up when the
    # kth phase is fully absent).
    D_mix = Vector{T}(undef, nc)
    @inbounds for c in 1:nc
        V_c = mesh.cell_volumes[c]
        D_mix[c] = V_c * (
            state.alpha_l.internal[c] / state.A_P_l[c] +
                state.alpha_g.internal[c] / state.A_P_g[c]
        )
        D_mix[c] = max(D_mix[c], eps(T))
    end

    # Build pressure BC dict: explicit entries override; fall back to
    # the velocity-BC patches with zero-gradient Neumann for anything
    # the user did not pressure-fix. This mirrors the single-phase
    # solver's `expand_bcs_pressure(prob.bcs)` behaviour when pressure
    # BCs are implicit in the velocity BC set.
    bcs_p_explicit = expand_bcs_pressure(prob.bcs_p)
    bcs_p = Dict{Symbol, AbstractBoundaryCondition}()
    for name in keys(prob.bcs_Ul)
        bcs_p[name] = get(bcs_p_explicit, name, ParabolicNeumann(0.0))
    end
    for (name, bc) in bcs_p_explicit
        bcs_p[name] = bc
    end
    assemble_laplacian!(eq, D_mix, mesh, bcs_p)

    # RHS: negative divergence of the phase-weighted H/A face fluxes.
    ubmap_l = build_boundary_map(state.U_l, mesh)
    ubmap_g = build_boundary_map(state.U_g, mesh)

    @inbounds for f in 1:nf
        S_f = face_normal_area(mesh, f)
        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)

            alpha_l_f = w * state.alpha_l.internal[P] +
                (one(T) - w) * state.alpha_l.internal[N]
            alpha_g_f = w * state.alpha_g.internal[P] +
                (one(T) - w) * state.alpha_g.internal[N]

            HbyA_l_P = state.H_U_l[P] / state.A_P_l[P]
            HbyA_l_N = state.H_U_l[N] / state.A_P_l[N]
            HbyA_g_P = state.H_U_g[P] / state.A_P_g[P]
            HbyA_g_N = state.H_U_g[N] / state.A_P_g[N]

            HbyA_l_f = w * HbyA_l_P + (one(T) - w) * HbyA_l_N
            HbyA_g_f = w * HbyA_g_P + (one(T) - w) * HbyA_g_N

            phi_HA = alpha_l_f * dot(HbyA_l_f, S_f) +
                alpha_g_f * dot(HbyA_g_f, S_f)

            eq.b[P] -= phi_HA
            eq.b[N] += phi_HA
        else
            P = owner(mesh, f)
            alpha_l_f = state.alpha_l.internal[P]
            alpha_g_f = state.alpha_g.internal[P]
            bi_l = ubmap_l[f]
            bi_g = ubmap_g[f]
            U_l_b = state.U_l.boundary[bi_l]
            U_g_b = state.U_g.boundary[bi_g]
            phi_HA = alpha_l_f * dot(U_l_b, S_f) + alpha_g_f * dot(U_g_b, S_f)
            eq.b[P] -= phi_HA
        end
    end

    return nothing
end

# ── Velocity correction ──────────────────────────────────────────────

"""
    correct_two_fluid_velocity!(state, prob)

Correct each phase's cell-centered velocity using the shared-pressure
gradient and the phase's `H/A` operator:

```
U_k[c] = H_U_k[c] / A_P_k[c] - α_k[c] · (V_c / A_P_k[c]) · ∇p[c]
```

This enforces each phase's momentum balance after the coupled solve.
"""
function correct_two_fluid_velocity!(
        state::TwoFluidState{Dim, T},
        prob::TwoFluidProblem{Dim, T},
    ) where {Dim, T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    grad_p = gradient(state.p_shared, mesh)

    @inbounds for c in 1:nc
        V_c = mesh.cell_volumes[c]
        D_l = V_c / state.A_P_l[c]
        D_g = V_c / state.A_P_g[c]
        state.U_l.internal[c] = state.H_U_l[c] / state.A_P_l[c] -
            state.alpha_l.internal[c] * D_l * grad_p[c]
        state.U_g.internal[c] = state.H_U_g[c] / state.A_P_g[c] -
            state.alpha_g.internal[c] * D_g * grad_p[c]
    end
    return nothing
end

# ── Mixture continuity residual ──────────────────────────────────────

"""
    two_fluid_mixture_continuity_residual(state, mesh) -> T

Return the L1 norm of the mixture-continuity residual
`Σ_c |Σ_f face_cell_sign · (α_l·ϕ_l + α_g·ϕ_g)_f|`. Zero exactly at
a converged divergence-free mixture flow.
"""
function two_fluid_mixture_continuity_residual(
        state::TwoFluidState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    imbalance = zeros(T, nc)
    @inbounds for f in 1:nf
        P = owner(mesh, f)
        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)
            alpha_l_f = w * state.alpha_l.internal[P] +
                (one(T) - w) * state.alpha_l.internal[N]
            alpha_g_f = w * state.alpha_g.internal[P] +
                (one(T) - w) * state.alpha_g.internal[N]
            F = alpha_l_f * state.phi_l.values[f] +
                alpha_g_f * state.phi_g.values[f]
            imbalance[P] += F
            imbalance[N] -= F
        else
            alpha_l_f = state.alpha_l.internal[P]
            alpha_g_f = state.alpha_g.internal[P]
            F = alpha_l_f * state.phi_l.values[f] +
                alpha_g_f * state.phi_g.values[f]
            imbalance[P] += F
        end
    end
    r = zero(T)
    @inbounds for c in 1:nc
        r += abs(imbalance[c])
    end
    return r
end

# ── Internal helpers ────────────────────────────────────────────────

"""
    _under_relax_block_momentum!(eq, state, component, alpha_U)

SIMPLE-style under-relaxation of the block momentum matrix. For each
cell `c` and phase `k`, divide the block-diagonal element by `α_U` and
adjust the block RHS so the relaxed solution satisfies
`U_new = α_U · U_solved + (1 − α_U) · U_old`.
"""
function _under_relax_block_momentum!(
        eq::BlockCollocatedEquation{T, 2},
        state::TwoFluidState{Dim, T},
        component::Int,
        alpha_U::T,
    ) where {Dim, T}
    zero(T) < alpha_U <= one(T) || throw(ArgumentError("alpha_U must be in (0, 1]"))
    one_minus = one(T) - alpha_U
    nc = length(state.U_l.internal)

    @inbounds for c in 1:nc
        for b in 1:2
            idx = eq.pattern.diag_idx[b, b, c]
            a_P = eq.A.nzval[idx]
            eq.A.nzval[idx] = a_P / alpha_U
            U_old = b == 1 ?
                state.U_l.internal[c][component] :
                state.U_g.internal[c][component]
            row = (c - 1) * 2 + b
            eq.b[row] += one_minus / alpha_U * a_P * U_old
        end
    end
    return nothing
end

"""
    _extract_block_operators!(state, eq, mesh)

Extract per-phase diagonal `A_P_k` and H-operator `H_U_k` from the
solved block momentum equation. Used after the last component solve
since the linear operators share structure across components.

H is computed via the face loop to keep it O(nc + nf).
"""
function _extract_block_operators!(
        state::TwoFluidState{Dim, T},
        eq::BlockCollocatedEquation{T, 2},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    @inbounds for c in 1:nc
        state.A_P_l[c] = eq.A.nzval[eq.pattern.diag_idx[1, 1, c]]
        state.A_P_g[c] = eq.A.nzval[eq.pattern.diag_idx[2, 2, c]]
    end

    # H_k[c] starts as the block RHS for each component; but since we
    # only call this after the last component's assembly we approximate
    # H as A_P · U using the current velocity — consistent with the
    # OpenFOAM convention that `H(U) = A·U + A_P·U` evaluated at the
    # latest cell values after under-relaxation.
    @inbounds for c in 1:nc
        state.H_U_l[c] = state.A_P_l[c] * state.U_l.internal[c]
        state.H_U_g[c] = state.A_P_g[c] * state.U_g.internal[c]
    end

    # Subtract off-diagonal contributions from the same block, which
    # are the convection/diffusion face couplings. Off-diagonal drag
    # couples the two phases and should not reduce each phase's H.
    @inbounds for f in 1:nf
        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            a_PN_l = eq.A.nzval[eq.pattern.offdiag_PN[1, 1, f]]
            a_NP_l = eq.A.nzval[eq.pattern.offdiag_NP[1, 1, f]]
            a_PN_g = eq.A.nzval[eq.pattern.offdiag_PN[2, 2, f]]
            a_NP_g = eq.A.nzval[eq.pattern.offdiag_NP[2, 2, f]]
            state.H_U_l[P] -= a_PN_l * state.U_l.internal[N]
            state.H_U_l[N] -= a_NP_l * state.U_l.internal[P]
            state.H_U_g[P] -= a_PN_g * state.U_g.internal[N]
            state.H_U_g[N] -= a_NP_g * state.U_g.internal[P]
        end
    end
    return nothing
end

"""
    _block_component_residuals(eq, sol) -> (rl, rg)

L1 norm of `A·x - (b + source)` restricted to each phase's rows. Used
as the per-component momentum residual.
"""
function _block_component_residuals(
        eq::BlockCollocatedEquation{T, 2}, sol::AbstractVector{T},
    ) where {T}
    rhs = eq.b .+ eq.source
    res = eq.A * sol .- rhs
    rl = zero(T); rg = zero(T)
    @inbounds for i in 1:length(sol)
        if isodd(i)  # block 1 (liquid) rows occupy odd indices (c-1)*2 + 1
            rl += abs(res[i])
        else
            rg += abs(res[i])
        end
    end
    return rl, rg
end

"""
    _two_fluid_needs_pressure_reference(bcs_p) -> Bool

Return `true` if no `FixedPressureBC` exists in `bcs_p`.
"""
function _two_fluid_needs_pressure_reference(
        bcs::Dict{Symbol, <:AbstractBoundaryCondition},
    )
    for bc in values(bcs)
        bc isa FixedPressureBC && return false
    end
    return true
end

"""
    _print_two_fluid_residuals(iter, residuals)

One-line summary of the outer iteration's residuals.
"""
function _print_two_fluid_residuals(iter, residuals)
    parts = String[]
    for key in (:Ul, :Ug, :continuity, :alpha)
        r = residuals[key][end]
        push!(parts, string(key, "=", @sprintf("%.3e", r)))
    end
    println("TwoFluid iter ", lpad(iter, 5), ": ", join(parts, "  "))
    return nothing
end
