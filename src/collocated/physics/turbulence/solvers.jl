# turbulence/solvers.jl — Turbulent SIMPLE/PISO/PIMPLE solver wrappers
#
# These functions extend the Phase 1 incompressible solvers with a
# turbulence step after velocity correction.

using Printf: @sprintf

"""
    solve_simple_turbulent(
        prob::IncompressibleProblem{Dim, T},
        turb_model::AbstractRANSModel;
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        linear_solver = nothing,
        verbose = false,
    ) -> Tuple{SolveResult{Dim, T}, RANSTurbulenceState{T}}

Solve steady incompressible flow with RANS turbulence using SIMPLE.

Same algorithm as `solve_simple` but with turbulence equations solved
after each velocity correction and `nu_eff = nu + nu_t` used in momentum.
Accepts the same optional `porous_zones` / `mrf_zones` keyword arguments
as [`solve_simple`](@ref) (Darcy-Forchheimer sinks use the MOLECULAR
viscosity `prob.nu`, not `nu_eff`).
"""
function solve_simple_turbulent(
        prob::IncompressibleProblem{Dim, T},
        turb_model;
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = nothing,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = nothing,
    ) where {Dim, T}
    algo = prob.algorithm::SIMPLE{T}
    mesh = prob.mesh

    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    turb_state = _init_turb_state(turb_model, mesh)
    _update_turbulence!(turb_state, turb_model, state, prob, mesh, turb_bcs)

    # Cyclic (periodic) support — mirror the laminar loop.  Previously the
    # turbulent loops silently dropped cyclic coupling, so a periodic RANS
    # channel decoupled across the boundary with no error.
    cyclic_pairs = collect_cyclic_pairs(prob.bcs, mesh)
    eqs, p_eq = _make_incompressible_workspace(prob, cyclic_pairs)

    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    converged = false
    final_iter = 0

    for iter in 1:algo.max_iterations
        final_iter = iter
        nu_eff = compute_nu_eff(prob.nu, turb_state.nu_t)

        # ── Momentum ────────────────────────────────────────────
        for d in 1:Dim
            reset!(eqs[d])
            assemble_momentum!(
                eqs[d], state, prob, d; nu_eff = nu_eff,
                porous_zones = porous_zones, mrf_zones = mrf_zones,
            )
            apply_cyclic_to_equation!(
                eqs[d], _make_scalar_field(_extract_component(state.U, d), state),
                mesh, cyclic_pairs,
            )
        end

        for d in 1:Dim
            U_old_d = _extract_component(state.U, d)
            under_relax_momentum!(eqs[d], U_old_d, algo.alpha_U)
            sol = _dispatch_solve(
                to_linear_problem(eqs[d]), linear_solver, solver_config,
                d == 1 ? :Ux : (d == 2 ? :Uy : :Uz),
            )
            _set_component!(state.U, d, sol.u)
        end
        update_boundary_velocity!(state, prob.bcs, mesh)
        update_boundary_cyclic!(state, mesh, cyclic_pairs)

        # Extract A_P/H(U) from the relaxed, solved equations
        extract_momentum_operators!(state, eqs, mesh; porous_zones = porous_zones)

        # ── Pressure ────────────────────────────────────────────
        reset!(p_eq)
        assemble_pressure!(p_eq, state, prob; mrf_zones = mrf_zones)
        apply_cyclic_to_equation!(p_eq, state.p, mesh, cyclic_pairs)
        if _needs_pressure_reference(prob.bcs)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end
        p_sol = _dispatch_solve(to_linear_problem(p_eq), linear_solver, solver_config, :p)

        nc = length(mesh.cell_volumes)
        for c in 1:nc
            state.p.internal[c] += algo.alpha_p * (p_sol.u[c] - state.p.internal[c])
        end
        update_boundary_pressure!(state, prob.bcs, mesh)

        correct_velocity!(state, mesh; porous_zones = porous_zones)
        update_boundary_velocity!(state, prob.bcs, mesh)
        update_boundary_cyclic!(state, mesh, cyclic_pairs)
        correct_fluxes!(state, mesh; porous_zones = porous_zones)
        if mrf_zones !== nothing
            mrf_make_relative!(state.phi.values, mesh, mrf_zones)
        end

        # ── Turbulence ──────────────────────────────────────────
        _update_turbulence!(
            turb_state, turb_model, state, prob, mesh, turb_bcs;
            linear_solver = linear_solver,
        )

        # ── Convergence ─────────────────────────────────────────
        max_res = zero(T)
        for d in 1:Dim
            u_d = _extract_component(state.U, d)
            r = momentum_residual(eqs[d], u_d)
            push!(residuals[component_labels[d]], r)
            max_res = max(max_res, r)
        end
        r_cont = continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)
        max_res = max(max_res, r_cont)

        if verbose
            _print_simple_residuals(iter, residuals, component_labels)
        end

        # No convergence exit on the first outer iteration (see
        # solve_simple / solve_simple_thermal — degenerate startup
        # residuals before the coupled fields feed back into momentum).
        if iter > 1 && max_res < algo.tolerance
            converged = true
            break
        end
    end

    result = SolveResult{Dim, T}(converged, final_iter, residuals, state)
    return (result, turb_state)
end

"""
    solve_incompressible_turbulent(
        prob, turb_model, tspan, dt; turb_bcs, kwargs...,
    ) -> Tuple{SolveResult, RANSTurbulenceState}

Solve transient incompressible flow with RANS turbulence using PISO or PIMPLE.
Accepts the same optional `porous_zones` / `mrf_zones` keyword arguments
as [`solve_incompressible`](@ref).
"""
function solve_incompressible_turbulent(
        prob::IncompressibleProblem{Dim, T},
        turb_model,
        tspan::Tuple{T, T},
        dt::T;
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        save_every::Int = 1,
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = nothing,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = nothing,
    ) where {Dim, T}
    mesh = prob.mesh

    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    turb_state = _init_turb_state(turb_model, mesh)
    _update_turbulence!(turb_state, turb_model, state, prob, mesh, turb_bcs)

    # Cyclic (periodic) support + equation workspace (allocated once)
    cyclic_pairs = collect_cyclic_pairs(prob.bcs, mesh)
    ws = _make_incompressible_workspace(prob, cyclic_pairs)

    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    t_start, t_end = tspan
    t = t_start
    n_steps = 0

    while t < t_end - eps(T) * abs(t_end)
        dt_actual = min(dt, t_end - t)
        nu_eff = compute_nu_eff(prob.nu, turb_state.nu_t)

        # Time step with nu_eff — use existing step functions but with modified momentum
        # For simplicity, inline the PISO/PIMPLE step with nu_eff
        if prob.algorithm isa PISO
            _turbulent_piso_step!(
                state, prob, dt_actual, prob.algorithm.n_correctors,
                nu_eff; linear_solver = linear_solver, solver_config = solver_config,
                cyclic_pairs = cyclic_pairs, t = t + dt_actual, ws = ws,
                porous_zones = porous_zones, mrf_zones = mrf_zones,
            )
        elseif prob.algorithm isa PIMPLE
            _turbulent_pimple_step!(
                state, prob, dt_actual, nu_eff;
                linear_solver = linear_solver, solver_config = solver_config,
                cyclic_pairs = cyclic_pairs, t = t + dt_actual, ws = ws,
                porous_zones = porous_zones, mrf_zones = mrf_zones,
            )
        end

        # Turbulence update
        _update_turbulence!(
            turb_state, turb_model, state, prob, mesh, turb_bcs;
            dt = dt_actual, linear_solver = linear_solver,
        )

        t += dt_actual
        n_steps += 1

        r_cont = continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)

        if verbose && n_steps % max(1, round(Int, (t_end - t_start) / dt / 20)) == 0
            println(
                "Step ", lpad(n_steps, 6), "  t=", @sprintf("%.4e", t),
                "  cont=", @sprintf("%.3e", r_cont)
            )
        end
    end

    result = SolveResult{Dim, T}(true, n_steps, residuals, state)
    return (result, turb_state)
end

# ── Turbulence state initialization ────────────────────────────────

"""Initialize appropriate turbulence state based on model type."""
_init_turb_state(model, mesh) = RANSTurbulenceState(model, mesh)

# ── Turbulent PISO step (with nu_eff) ───────────────────────────────

function _turbulent_piso_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T, n_correctors::Int,
        nu_eff::Vector{T};
        linear_solver = nothing,
        solver_config = nothing,
        cyclic_pairs::Vector{Vector{Tuple{Int, Int}}} = Vector{Vector{Tuple{Int, Int}}}(),
        t::T = zero(T),
        ws = nothing,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = nothing,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = nothing,
    ) where {Dim, T}
    mesh = prob.mesh

    # Old-time snapshot for the ddt term (once per time step)
    _snapshot_old_time!(state)

    eqs, p_eq = ws === nothing ? _make_incompressible_workspace(prob, cyclic_pairs) : ws

    for d in 1:Dim
        reset!(eqs[d])
        assemble_momentum!(
            eqs[d], state, prob, d; dt = dt, nu_eff = nu_eff, t = t,
            porous_zones = porous_zones, mrf_zones = mrf_zones,
        )
        apply_cyclic_to_equation!(
            eqs[d], _make_scalar_field(_extract_component(state.U, d), state),
            mesh, cyclic_pairs,
        )
    end

    for d in 1:Dim
        sol = _dispatch_solve(
            to_linear_problem(eqs[d]), linear_solver, solver_config,
            d == 1 ? :Ux : (d == 2 ? :Uy : :Uz),
        )
        _set_component!(state.U, d, sol.u)
    end
    update_boundary_velocity!(state, prob.bcs, mesh; t = t)
    update_boundary_cyclic!(state, mesh, cyclic_pairs)

    extract_momentum_operators!(state, eqs, mesh; porous_zones = porous_zones)

    for k in 1:n_correctors
        reset!(p_eq)
        assemble_pressure!(p_eq, state, prob; mrf_zones = mrf_zones)
        apply_cyclic_to_equation!(p_eq, state.p, mesh, cyclic_pairs)
        if _needs_pressure_reference(prob.bcs)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end
        p_sol = _dispatch_solve(to_linear_problem(p_eq), linear_solver, solver_config, :p)

        nc = length(mesh.cell_volumes)
        for c in 1:nc
            state.p.internal[c] = p_sol.u[c]
        end
        update_boundary_pressure!(state, prob.bcs, mesh)
        correct_velocity!(state, mesh; porous_zones = porous_zones)
        update_boundary_velocity!(state, prob.bcs, mesh; t = t)
        update_boundary_cyclic!(state, mesh, cyclic_pairs)
        correct_fluxes!(state, mesh; porous_zones = porous_zones)
        if mrf_zones !== nothing
            mrf_make_relative!(state.phi.values, mesh, mrf_zones)
        end

        if k < n_correctors
            for d in 1:Dim
                reset!(eqs[d])
                assemble_momentum!(
                    eqs[d], state, prob, d; dt = dt, nu_eff = nu_eff, t = t,
                    porous_zones = porous_zones, mrf_zones = mrf_zones,
                )
                apply_cyclic_to_equation!(
                    eqs[d], _make_scalar_field(_extract_component(state.U, d), state),
                    mesh, cyclic_pairs,
                )
            end
            extract_momentum_operators!(state, eqs, mesh; porous_zones = porous_zones)
        end
    end

    return nothing
end

# ── Turbulent PIMPLE step (with nu_eff) ─────────────────────────────

function _turbulent_pimple_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T, nu_eff::Vector{T};
        linear_solver = nothing,
        solver_config = nothing,
        cyclic_pairs::Vector{Vector{Tuple{Int, Int}}} = Vector{Vector{Tuple{Int, Int}}}(),
        t::T = zero(T),
        ws = nothing,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = nothing,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = nothing,
    ) where {Dim, T}
    algo = prob.algorithm::PIMPLE{T}
    mesh = prob.mesh

    # Old-time snapshot for the ddt term (once per time step, shared by
    # all outer iterations)
    _snapshot_old_time!(state)

    eqs, p_eq = ws === nothing ? _make_incompressible_workspace(prob, cyclic_pairs) : ws

    for outer in 1:algo.n_outer
        is_final = (outer == algo.n_outer)

        for d in 1:Dim
            reset!(eqs[d])
            assemble_momentum!(
                eqs[d], state, prob, d; dt = dt, nu_eff = nu_eff, t = t,
                porous_zones = porous_zones, mrf_zones = mrf_zones,
            )
            apply_cyclic_to_equation!(
                eqs[d], _make_scalar_field(_extract_component(state.U, d), state),
                mesh, cyclic_pairs,
            )
        end

        for d in 1:Dim
            if !is_final
                U_old_d = _extract_component(state.U, d)
                under_relax_momentum!(eqs[d], U_old_d, algo.alpha_U)
            end
            sol = _dispatch_solve(
                to_linear_problem(eqs[d]), linear_solver, solver_config,
                d == 1 ? :Ux : (d == 2 ? :Uy : :Uz),
            )
            _set_component!(state.U, d, sol.u)
        end
        update_boundary_velocity!(state, prob.bcs, mesh; t = t)
        update_boundary_cyclic!(state, mesh, cyclic_pairs)

        extract_momentum_operators!(state, eqs, mesh; porous_zones = porous_zones)

        nc = length(mesh.cell_volumes)
        for k in 1:algo.n_correctors
            reset!(p_eq)
            assemble_pressure!(p_eq, state, prob; mrf_zones = mrf_zones)
            apply_cyclic_to_equation!(p_eq, state.p, mesh, cyclic_pairs)
            if _needs_pressure_reference(prob.bcs)
                fix_pressure_reference!(p_eq, 1, zero(T))
            end
            p_sol = _dispatch_solve(to_linear_problem(p_eq), linear_solver, solver_config, :p)

            if !is_final
                for c in 1:nc
                    state.p.internal[c] += algo.alpha_p * (p_sol.u[c] - state.p.internal[c])
                end
            else
                for c in 1:nc
                    state.p.internal[c] = p_sol.u[c]
                end
            end
            update_boundary_pressure!(state, prob.bcs, mesh)
            correct_velocity!(state, mesh; porous_zones = porous_zones)
            update_boundary_velocity!(state, prob.bcs, mesh; t = t)
            update_boundary_cyclic!(state, mesh, cyclic_pairs)
            correct_fluxes!(state, mesh; porous_zones = porous_zones)
            if mrf_zones !== nothing
                mrf_make_relative!(state.phi.values, mesh, mrf_zones)
            end
        end
    end

    return nothing
end
