# thermal/solvers.jl — Thermal SIMPLE/PISO/PIMPLE solver wrappers
#
# Extends the incompressible solvers with energy equation and optional
# buoyancy coupling. Follows the Phase 2a turbulence wrapper pattern.

using Printf: @sprintf

"""
    solve_simple_thermal(
        prob::IncompressibleProblem{Dim, T},
        thermal_props::FluidThermalProperties{Dim, T};
        bcs_T,
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        T_init = thermal_props.T_ref,
        linear_solver = nothing,
        verbose = false,
    ) -> Tuple{SolveResult{Dim, T}, ThermalState{T}}

Solve steady incompressible flow with energy equation using SIMPLE.

Each iteration:
1. Update effective conductivity and thermal diffusivity
2. Compute buoyancy force (if beta > 0)
3. Assemble + solve momentum with `nu_eff` and `body_force`
4. Pressure solve + correction
5. Solve turbulence (if turbulence model provided)
6. Assemble + solve energy equation
7. Check convergence

Accepts the same optional `porous_zones` / `mrf_zones` keyword arguments
as [`solve_simple`](@ref).
"""
function solve_simple_thermal(
        prob::IncompressibleProblem{Dim, T},
        thermal_props::FluidThermalProperties{Dim, T};
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition},
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        T_init::Real = thermal_props.T_ref,
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = nothing,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = nothing,
    ) where {Dim, T}
    algo = prob.algorithm::SIMPLE{T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    # Initialize flow state
    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    # Initialize thermal state
    thermal_state = ThermalState(mesh; T_init = T(T_init), k_init = thermal_props.k)

    # Initialize turbulence (optional)
    turb_state = nothing
    if turb_model !== nothing
        turb_state = RANSTurbulenceState(turb_model, mesh)
        turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
    end

    # Residual tracking
    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    converged = false
    final_iter = 0

    for iter in 1:algo.max_iterations
        final_iter = iter

        # ── Effective properties ────────────────────────────────
        nu_t_vec = turb_state === nothing ? nothing : turb_state.nu_t
        update_k_eff!(thermal_state, thermal_props, nu_t_vec, prob.density)
        nu_eff = turb_state === nothing ? prob.nu : compute_nu_eff(prob.nu, turb_state.nu_t)
        alpha_eff = compute_alpha_eff(thermal_state.k_eff, prob.density, thermal_props.Cp)

        # ── Buoyancy ────────────────────────────────────────────
        body_force = compute_buoyancy_source(thermal_state.T_field, thermal_props, prob.density)

        # ── Momentum ────────────────────────────────────────────
        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(
                eq, state, prob, d;
                nu_eff = nu_eff, body_force = body_force,
                porous_zones = porous_zones, mrf_zones = mrf_zones,
            )
            push!(eqs, eq)
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

        # Extract A_P/H(U) from the relaxed, solved equations
        extract_momentum_operators!(state, eqs, mesh; porous_zones = porous_zones)

        # ── Pressure ────────────────────────────────────────────
        p_eq = CollocatedEquation(mesh)
        assemble_pressure!(p_eq, state, prob; mrf_zones = mrf_zones)
        if _needs_pressure_reference(prob.bcs)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end
        p_sol = _dispatch_solve(to_linear_problem(p_eq), linear_solver, solver_config, :p)

        for c in 1:nc
            state.p.internal[c] += algo.alpha_p * (p_sol.u[c] - state.p.internal[c])
        end
        update_boundary_pressure!(state, prob.bcs, mesh)

        correct_velocity!(state, mesh; porous_zones = porous_zones)
        update_boundary_velocity!(state, prob.bcs, mesh)
        correct_fluxes!(state, mesh; porous_zones = porous_zones)
        if mrf_zones !== nothing
            mrf_make_relative!(state.phi.values, mesh, mrf_zones)
        end

        # ── Turbulence (optional) ───────────────────────────────
        if turb_model !== nothing
            solve_turbulence!(
                turb_state, turb_model, state.U, state.phi, prob.nu, mesh, turb_bcs;
                linear_solver = linear_solver,
            )
            turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
            _apply_realizability!(turb_state, turb_model, state.U, mesh)
        end

        # ── Energy equation ─────────────────────────────────────
        if thermal_props.use_enthalpy
            _advance_enthalpy_step!(
                thermal_state, thermal_props, state.phi, alpha_eff, mesh, bcs_T;
                dt = nothing, linear_solver = linear_solver,
                solver_config = solver_config,
            )
        else
            T_eq = CollocatedEquation(mesh)
            assemble_energy!(T_eq, thermal_state.T_field, state.phi, alpha_eff, mesh, bcs_T)
            T_sol = _dispatch_solve(to_linear_problem(T_eq), linear_solver, solver_config, :T)
            for c in 1:nc
                thermal_state.T_field.internal[c] = T_sol.u[c]
            end
        end

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

        # No convergence exit on the first outer iteration — the
        # buoyancy-driven startup iterate (U = 0, uniform T) has exactly
        # zero momentum residuals before the energy solve feeds back.
        if iter > 1 && max_res < algo.tolerance
            converged = true
            break
        end
    end

    result = SolveResult{Dim, T}(converged, final_iter, residuals, state)
    return (result, thermal_state)
end

"""
    solve_incompressible_thermal(
        prob, thermal_props, tspan, dt;
        bcs_T, turb_model, turb_bcs, T_init, linear_solver, verbose,
    ) -> Tuple{SolveResult, ThermalState}

Solve transient incompressible flow with energy equation using PISO or PIMPLE.
Accepts the same optional `porous_zones` / `mrf_zones` keyword arguments
as [`solve_incompressible`](@ref).
"""
function solve_incompressible_thermal(
        prob::IncompressibleProblem{Dim, T},
        thermal_props::FluidThermalProperties{Dim, T},
        tspan::Tuple{T, T},
        dt::T;
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition},
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        T_init::Real = thermal_props.T_ref,
        save_every::Int = 1,
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = nothing,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = nothing,
    ) where {Dim, T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    thermal_state = ThermalState(mesh; T_init = T(T_init), k_init = thermal_props.k)

    turb_state = nothing
    if turb_model !== nothing
        turb_state = RANSTurbulenceState(turb_model, mesh)
        turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
    end

    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    t_start, t_end = tspan
    t = t_start
    n_steps = 0

    while t < t_end - eps(T) * abs(t_end)
        dt_actual = min(dt, t_end - t)

        # Effective properties
        nu_t_vec = turb_state === nothing ? nothing : turb_state.nu_t
        update_k_eff!(thermal_state, thermal_props, nu_t_vec, prob.density)
        nu_eff = turb_state === nothing ? prob.nu : compute_nu_eff(prob.nu, turb_state.nu_t)
        alpha_eff = compute_alpha_eff(thermal_state.k_eff, prob.density, thermal_props.Cp)
        body_force = compute_buoyancy_source(thermal_state.T_field, thermal_props, prob.density)

        # Flow step with thermal coupling
        if prob.algorithm isa PISO
            _thermal_piso_step!(
                state, prob, dt_actual, prob.algorithm.n_correctors,
                nu_eff, body_force;
                linear_solver = linear_solver, solver_config = solver_config,
                porous_zones = porous_zones, mrf_zones = mrf_zones,
            )
        elseif prob.algorithm isa PIMPLE
            _thermal_pimple_step!(
                state, prob, dt_actual, nu_eff, body_force;
                linear_solver = linear_solver, solver_config = solver_config,
                porous_zones = porous_zones, mrf_zones = mrf_zones,
            )
        end

        # Turbulence update
        if turb_model !== nothing
            solve_turbulence!(
                turb_state, turb_model, state.U, state.phi,
                prob.nu, mesh, turb_bcs; dt = dt_actual, linear_solver = linear_solver
            )
            turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
            _apply_realizability!(turb_state, turb_model, state.U, mesh)
        end

        # Energy equation
        if thermal_props.use_enthalpy
            _advance_enthalpy_step!(
                thermal_state, thermal_props, state.phi, alpha_eff, mesh, bcs_T;
                dt = dt_actual, linear_solver = linear_solver,
                solver_config = solver_config,
            )
        else
            T_eq = CollocatedEquation(mesh)
            assemble_energy!(
                T_eq, thermal_state.T_field, state.phi, alpha_eff, mesh, bcs_T;
                dt = dt_actual
            )
            T_sol = _dispatch_solve(to_linear_problem(T_eq), linear_solver, solver_config, :T)
            for c in 1:nc
                thermal_state.T_field.internal[c] = T_sol.u[c]
            end
        end

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
    return (result, thermal_state)
end

# ── Thermal PISO step ────────────────────────────────────────────────

function _thermal_piso_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T, n_correctors::Int,
        nu_eff::Union{T, Vector{T}},
        body_force::Union{Nothing, Vector{SVector{Dim, T}}};
        linear_solver = nothing,
        solver_config = nothing,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = nothing,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = nothing,
    ) where {Dim, T}
    mesh = prob.mesh

    # Old-time snapshot for the ddt term (once per time step)
    _snapshot_old_time!(state)

    eqs = CollocatedEquation{T}[]
    for d in 1:Dim
        eq = CollocatedEquation(mesh)
        assemble_momentum!(
            eq, state, prob, d;
            dt = dt, nu_eff = nu_eff, body_force = body_force,
            porous_zones = porous_zones, mrf_zones = mrf_zones,
        )
        push!(eqs, eq)
    end

    for d in 1:Dim
        sol = _dispatch_solve(
            to_linear_problem(eqs[d]), linear_solver, solver_config,
            d == 1 ? :Ux : (d == 2 ? :Uy : :Uz),
        )
        _set_component!(state.U, d, sol.u)
    end
    update_boundary_velocity!(state, prob.bcs, mesh)

    # Extract A_P/H(U) from the solved equations
    extract_momentum_operators!(state, eqs, mesh; porous_zones = porous_zones)

    for k in 1:n_correctors
        p_eq = CollocatedEquation(mesh)
        assemble_pressure!(p_eq, state, prob; mrf_zones = mrf_zones)
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
        update_boundary_velocity!(state, prob.bcs, mesh)
        correct_fluxes!(state, mesh; porous_zones = porous_zones)
        if mrf_zones !== nothing
            mrf_make_relative!(state.phi.values, mesh, mrf_zones)
        end

        if k < n_correctors
            eqs_k = CollocatedEquation{T}[]
            for d in 1:Dim
                eq = CollocatedEquation(mesh)
                assemble_momentum!(
                    eq, state, prob, d;
                    dt = dt, nu_eff = nu_eff, body_force = body_force,
                    porous_zones = porous_zones, mrf_zones = mrf_zones,
                )
                push!(eqs_k, eq)
            end
            extract_momentum_operators!(state, eqs_k, mesh; porous_zones = porous_zones)
        end
    end

    return nothing
end

# ── Thermal PIMPLE step ──────────────────────────────────────────────

function _thermal_pimple_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T,
        nu_eff::Union{T, Vector{T}},
        body_force::Union{Nothing, Vector{SVector{Dim, T}}};
        linear_solver = nothing,
        solver_config = nothing,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = nothing,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = nothing,
    ) where {Dim, T}
    algo = prob.algorithm::PIMPLE{T}
    mesh = prob.mesh

    # Old-time snapshot for the ddt term (shared by all outer iterations)
    _snapshot_old_time!(state)

    for outer in 1:algo.n_outer
        is_final = (outer == algo.n_outer)

        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(
                eq, state, prob, d;
                dt = dt, nu_eff = nu_eff, body_force = body_force,
                porous_zones = porous_zones, mrf_zones = mrf_zones,
            )
            push!(eqs, eq)
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
        update_boundary_velocity!(state, prob.bcs, mesh)

        # Extract A_P/H(U) from the (relaxed) solved equations
        extract_momentum_operators!(state, eqs, mesh; porous_zones = porous_zones)

        nc = length(mesh.cell_volumes)
        for k in 1:algo.n_correctors
            p_eq = CollocatedEquation(mesh)
            assemble_pressure!(p_eq, state, prob; mrf_zones = mrf_zones)
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
            update_boundary_velocity!(state, prob.bcs, mesh)
            correct_fluxes!(state, mesh; porous_zones = porous_zones)
            if mrf_zones !== nothing
                mrf_make_relative!(state.phi.values, mesh, mrf_zones)
            end
        end
    end

    return nothing
end

# ── Enthalpy advance helper ──────────────────────────────────────────
#
# Internal bridge between the temperature-state (`ThermalState`) and the
# enthalpy-form energy equation. The enthalpy field is a transient
# working variable; after each solve we convert back to T so the rest
# of the solver (buoyancy, `k_eff`, diagnostics) sees consistent data.

function _advance_enthalpy_step!(
        thermal_state::ThermalState{T},
        thermal_props::FluidThermalProperties{Dim, T},
        phi::FaceFluxField{T},
        alpha_eff::Vector{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
        linear_solver = nothing,
        solver_config = nothing,
    ) where {Dim, T}
    Cp = thermal_props.Cp
    T_ref = thermal_props.T_ref

    # T → h (internal + boundary).
    h_field = enthalpy_field_from_temperature(thermal_state.T_field, T_ref, Cp)
    bcs_h = enthalpy_bcs_from_temperature(bcs_T, T_ref, Cp)

    # Enthalpy diffusivity equals alpha_eff for constant Cp.
    alpha_h = alpha_eff

    solve_enthalpy_equation(
        h_field, phi, alpha_h, mesh, bcs_h;
        dt = dt, linear_solver = linear_solver, solver_config = solver_config,
    )

    # h → T.
    temperature_from_enthalpy!(thermal_state.T_field, h_field, T_ref, Cp)
    return nothing
end
