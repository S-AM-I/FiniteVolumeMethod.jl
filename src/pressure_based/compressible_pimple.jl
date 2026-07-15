# pressure_based/compressible_pimple.jl — Weakly-compressible PIMPLE
#
# Transient counterpart of `CompressibleSIMPLE`. Structure mirrors the
# incompressible `_pimple_step!`: outer SIMPLE-like passes with inner
# PISO correctors. Each step advances the solution by a fixed dt and
# updates ρ and (optionally) T so the EOS coupling can relax.
#
# HONESTY NOTE (what this solver actually does): the pressure-velocity
# loop enforces INCOMPRESSIBLE continuity (div(U) = 0); density is
# updated from the EOS only AFTER each outer pass and never enters the
# continuity constraint (face densities `rho_f` are computed but the
# mass flux is not density-weighted).  This is a low-Mach,
# weakly-compressible approximation: valid when density variations are
# small and slow.  It is NOT a conservative rhoPimpleFoam analogue —
# mass is NOT conserved for genuinely compressible flows.  A @warn at
# solver entry states this.

using Printf: @sprintf

# ── CompressiblePIMPLE algorithm ────────────────────────────────────

@doc """
    CompressiblePIMPLE{T} <: AbstractPVCoupling

Transient weakly-compressible pressure-based coupling. Combines outer
SIMPLE-style under-relaxation with inner PISO correctors and an
explicit EOS density post-update.  The inner loop enforces
INCOMPRESSIBLE continuity (`div(U) = 0`); density never enters the
mass balance, so mass is not conserved for genuinely compressible
flows.  Valid only for low-Mach, weakly-compressible use; at higher
Mach the density-based stack in `src/hyperbolic/` is appropriate.

# Fields
- `n_outer::Int`       — number of outer iterations per time step
- `n_correctors::Int`  — number of inner pressure correctors
- `alpha_U::T`
- `alpha_p::T`
- `alpha_rho::T`
- `tolerance::T`
"""
struct CompressiblePIMPLE{T} <: AbstractPVCoupling
    n_outer::Int
    n_correctors::Int
    alpha_U::T
    alpha_p::T
    alpha_rho::T
    tolerance::T
end

@doc """
    CompressiblePIMPLE(; n_outer = 2, n_correctors = 1,
                         alpha_U = 0.7, alpha_p = 0.3, alpha_rho = 0.7,
                         tolerance = 1e-6)

Construct a [`CompressiblePIMPLE`](@ref) algorithm.
"""
function CompressiblePIMPLE(;
        n_outer::Int = 2,
        n_correctors::Int = 1,
        alpha_U::T = 0.7,
        alpha_p::T = 0.3,
        alpha_rho::T = 0.7,
        tolerance::T = 1.0e-6,
    ) where {T}
    return CompressiblePIMPLE{T}(n_outer, n_correctors, alpha_U, alpha_p, alpha_rho, tolerance)
end

# ── Compressible PIMPLE step ────────────────────────────────────────

"""
    _compressible_pimple_step!(cstate, prob, dt; kwargs...)

Advance the compressible state by one time step of size `dt` using
the [`CompressiblePIMPLE`](@ref) algorithm.
"""
function _compressible_pimple_step!(
        cstate::CompressibleState{Dim, T},
        prob::CompressibleProblem{Dim, T, Mesh, BC, CompressiblePIMPLE{T}, Model},
        dt::T;
        linear_solver = nothing,
        solver_config = nothing,
        cyclic_pairs::Vector{Vector{Tuple{Int, Int}}} = Vector{Vector{Tuple{Int, Int}}}(),
    ) where {Dim, T, Mesh, BC, Model}
    algo = prob.algorithm
    mesh = prob.mesh
    n_outer = algo.n_outer
    n_correctors = algo.n_correctors
    alpha_U = algo.alpha_U
    alpha_p = algo.alpha_p
    alpha_rho = algo.alpha_rho

    state = cstate.base
    nc = length(mesh.cell_volumes)
    rho_prev = copy(cstate.rho)

    # Old-time snapshot for the ddt term (shared by all outer iterations)
    _snapshot_old_time!(state)

    for outer in 1:n_outer
        is_final = (outer == n_outer)

        # Property updates
        update_viscosity!(cstate.mu_cells, prob.thermo, cstate.T_cells)
        mu_mean = sum(cstate.mu_cells) / nc
        rho_mean = sum(cstate.rho) / nc
        compute_face_densities!(
            cstate.rho_f, prob.thermo,
            mesh, state.p.internal, cstate.T_cells
        )

        shim = _incompressible_shim(prob, rho_mean, mu_mean)

        # Momentum assemble + solve
        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(
                eq, state, shim, d; dt = dt,
                nu_eff = cstate.mu_cells ./ cstate.rho
            )
            apply_cyclic_to_equation!(
                eq, _make_scalar_field(_extract_component(state.U, d), state),
                mesh, cyclic_pairs,
            )
            push!(eqs, eq)
        end
        for d in 1:Dim
            if !is_final
                U_old_d = _extract_component(state.U, d)
                under_relax_momentum!(eqs[d], U_old_d, alpha_U)
            end
            lp = to_linear_problem(eqs[d])
            sol = _dispatch_solve(
                lp, linear_solver, solver_config,
                d == 1 ? :Ux : (d == 2 ? :Uy : :Uz),
            )
            _set_component!(state.U, d, sol.u)
        end
        update_boundary_velocity!(state, prob.bcs, mesh)
        update_boundary_cyclic!(state, mesh, cyclic_pairs)

        # Extract A_P/H(U) from the (relaxed) solved equations
        extract_momentum_operators!(state, eqs, mesh)

        # PISO inner corrector loop
        for k in 1:n_correctors
            needs_ref = _needs_pressure_reference(prob.bcs)
            p_mean_target = needs_ref ? sum(state.p.internal) / nc : zero(T)

            p_eq = CollocatedEquation(mesh)
            assemble_pressure!(p_eq, state, shim)
            apply_cyclic_to_equation!(p_eq, state.p, mesh, cyclic_pairs)
            if needs_ref
                fix_pressure_reference!(p_eq, 1, state.p.internal[1])
            end
            lp_p = to_linear_problem(p_eq)
            p_sol = _dispatch_solve(lp_p, linear_solver, solver_config, :p)

            if !is_final
                @inbounds for c in 1:nc
                    state.p.internal[c] += alpha_p * (p_sol.u[c] - state.p.internal[c])
                end
            else
                @inbounds for c in 1:nc
                    state.p.internal[c] = p_sol.u[c]
                end
            end

            if needs_ref
                p_mean_now = sum(state.p.internal) / nc
                shift = p_mean_target - p_mean_now
                @inbounds for c in 1:nc
                    state.p.internal[c] += shift
                end
            end
            update_boundary_pressure!(state, prob.bcs, mesh)

            correct_velocity!(state, mesh)
            update_boundary_velocity!(state, prob.bcs, mesh)
            update_boundary_cyclic!(state, mesh, cyclic_pairs)
            correct_fluxes!(state, mesh)
        end

        # Density update (EOS coupling) with under-relaxation
        copyto!(rho_prev, cstate.rho)
        update_density!(cstate.rho, prob.thermo, state.p.internal, cstate.T_cells)
        @inbounds for c in 1:nc
            cstate.rho[c] = rho_prev[c] + alpha_rho * (cstate.rho[c] - rho_prev[c])
        end
    end

    return nothing
end

# ── Transient driver ────────────────────────────────────────────────

@doc """
    solve_compressible(prob, tspan, dt; kwargs...) -> NamedTuple

Transient compressible PIMPLE solver. Advances the state from
`tspan[1]` to `tspan[2]` with fixed time step `dt`, calling
`_compressible_pimple_step!` each time.

# Returns
A named tuple with fields:
- `converged::Bool`
- `iterations::Int`         — number of time steps taken
- `residuals::Dict`         — residual history
- `state::CompressibleState`
"""
function solve_compressible(
        prob::CompressibleProblem{Dim, T, Mesh, BC, CompressiblePIMPLE{T}, Model},
        tspan::Tuple{T, T},
        dt::T;
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        p0::Real = 1.01325e5,
    ) where {Dim, T, Mesh, BC, Model}
    @warn "CompressiblePIMPLE enforces incompressible continuity (div(U)=0) " *
        "with an EOS density post-update. Mass is NOT conserved for genuinely " *
        "compressible cases — use this solver only for low-Mach, " *
        "weakly-compressible flows." maxlog = 1
    mesh = prob.mesh
    t_start, t_end = tspan

    cstate = CompressibleState(mesh, prob.thermo; p0 = p0, T0 = prob.T_ref)
    state = cstate.base
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)
    cyclic_pairs = collect_cyclic_pairs(prob.bcs, mesh)

    residuals = Dict{Symbol, Vector{T}}(:continuity => T[], :density => T[])

    t = t_start
    n_steps = 0
    while t < t_end - eps(T) * abs(t_end)
        dt_actual = min(dt, t_end - t)
        _compressible_pimple_step!(
            cstate, prob, dt_actual;
            linear_solver = linear_solver,
            solver_config = solver_config,
            cyclic_pairs = cyclic_pairs,
        )
        t += dt_actual
        n_steps += 1

        r_cont = continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)
        push!(residuals[:density], sum(cstate.rho) / length(cstate.rho))

        if verbose
            println(
                "cPIMPLE step ", lpad(n_steps, 6),
                "  t=", @sprintf("%.4e", t),
                "  cont=", @sprintf("%.3e", r_cont)
            )
        end
    end

    return (
        converged = true, iterations = n_steps,
        residuals = residuals, state = cstate,
    )
end
