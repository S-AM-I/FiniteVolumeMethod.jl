# turbulence/solvers.jl — Turbulent SIMPLE/PISO/PIMPLE solver wrappers
#
# These functions extend the Phase 1 incompressible solvers with a
# turbulence step after velocity correction.

using Printf: @sprintf

"""
    solve_simple_turbulent(
        prob::AnyIncompressibleProblem{Dim, T},
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
        prob::AnyIncompressibleProblem{Dim, T},
        turb_model;
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = prob.model.porous_zones,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = prob.model.mrf_zones,
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

        # Momentum/pressure via the shared SIMPLE core, with the turbulent
        # effective viscosity as the only coupling. The residuals it records
        # depend on the frozen momentum equations and the corrected velocity,
        # neither of which the turbulence update below touches, so computing
        # them here (before that update) matches the previous ordering.
        max_res = _simple_outer_step!(
            state, prob, eqs, p_eq, cyclic_pairs, residuals, component_labels;
            nu_eff = nu_eff,
            linear_solver = linear_solver, solver_config = solver_config,
            porous_zones = porous_zones, mrf_zones = mrf_zones,
        )

        # ── Turbulence transport ────────────────────────────────
        _update_turbulence!(
            turb_state, turb_model, state, prob, mesh, turb_bcs;
            linear_solver = linear_solver,
        )

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
        prob::AnyIncompressibleProblem{Dim, T},
        turb_model,
        tspan::Tuple{T, T},
        dt::T;
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = prob.model.porous_zones,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = prob.model.mrf_zones,
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

        # Flow step with the turbulent effective viscosity fed into the shared
        # momentum core.
        if prob.algorithm isa PISO
            _piso_step!(
                state, prob, dt_actual, prob.algorithm.n_correctors;
                nu_eff = nu_eff, cyclic_pairs = cyclic_pairs, t = t + dt_actual, ws = ws,
                linear_solver = linear_solver, solver_config = solver_config,
                porous_zones = porous_zones, mrf_zones = mrf_zones,
            )
        elseif prob.algorithm isa PIMPLE
            _pimple_step!(
                state, prob, dt_actual;
                nu_eff = nu_eff, cyclic_pairs = cyclic_pairs, t = t + dt_actual, ws = ws,
                linear_solver = linear_solver, solver_config = solver_config,
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

    # A transient run "converged" iff it completed with finite residuals
    # (converged used to be hardcoded true, masking NaN/Inf blow-ups).
    r_hist = residuals[:continuity]
    converged = isempty(r_hist) || isfinite(r_hist[end])

    result = SolveResult{Dim, T}(converged, n_steps, residuals, state)
    return (result, turb_state)
end

# ── Turbulence state initialization ────────────────────────────────

"""Initialize appropriate turbulence state based on model type."""
_init_turb_state(model, mesh) = RANSTurbulenceState(model, mesh)

