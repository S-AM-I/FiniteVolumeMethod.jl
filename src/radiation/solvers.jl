# radiation/solvers.jl — Combined thermal + radiation solver wrapper
#
# Extends the Phase 3 thermal solver with a P1 radiation step after
# the energy equation. Radiation source is lagged one iteration.

using Printf: @sprintf

"""
    solve_simple_thermal_radiation(
        prob, thermal_props, rad_model;
        bcs_T, bcs_G,
        turb_model, turb_bcs, T_init,
        linear_solver, verbose,
    ) -> Tuple{SolveResult, ThermalState, RadiationState}

Solve steady incompressible flow with energy equation and P1 radiation.

Each SIMPLE iteration:
1. Update effective properties (k_eff, nu_eff, buoyancy)
2. Momentum + pressure + correction
3. Turbulence (optional)
4. Solve energy equation with radiation source in RHS
5. Solve P1 radiation equation for G
6. Update radiation source
7. Check convergence
"""
function solve_simple_thermal_radiation(
        prob::IncompressibleProblem{Dim, T},
        thermal_props::FluidThermalProperties{Dim, T},
        rad_model::P1Model{T};
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition},
        bcs_G::Dict{Symbol, <:AbstractBoundaryCondition},
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        T_init::Real = thermal_props.T_ref,
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    algo = prob.algorithm::SIMPLE{T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    # Initialize states
    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    thermal_state = ThermalState(mesh; T_init = T(T_init), k_init = thermal_props.k)
    rad_state = RadiationState(mesh; G_init = T(4) * T(STEFAN_BOLTZMANN) * T(T_init)^4)

    # Turbulence (optional)
    turb_state = nothing
    if turb_model !== nothing
        turb_state = RANSTurbulenceState(turb_model, mesh)
        turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
    end

    # Radiation source (initialized to zero, updated after first G solve)
    S_rad = zeros(T, nc)

    # Residuals
    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    converged = false
    final_iter = 0

    for iter in 1:algo.max_iterations
        final_iter = iter

        # -- Effective properties ------------------------------------------
        nu_t_vec = turb_state === nothing ? nothing : turb_state.nu_t
        update_k_eff!(thermal_state, thermal_props, nu_t_vec, prob.density)
        nu_eff = turb_state === nothing ? prob.nu : compute_nu_eff(prob.nu, turb_state.nu_t)
        alpha_eff = compute_alpha_eff(thermal_state.k_eff, prob.density, thermal_props.Cp)

        # Buoyancy
        body_force = compute_buoyancy_source(thermal_state.T_field, thermal_props, prob.density)

        # -- Momentum ------------------------------------------------------
        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(
                eq, state, prob, d;
                nu_eff = nu_eff, body_force = body_force
            )
            push!(eqs, eq)
        end

        extract_momentum_operators!(state, eqs, mesh)

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

        # -- Pressure ------------------------------------------------------
        p_eq = CollocatedEquation(mesh)
        assemble_pressure!(p_eq, state, prob)
        if _needs_pressure_reference(prob.bcs)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end
        p_sol = _dispatch_solve(to_linear_problem(p_eq), linear_solver, solver_config, :p)

        for c in 1:nc
            state.p.internal[c] += algo.alpha_p * (p_sol.u[c] - state.p.internal[c])
        end
        update_boundary_pressure!(state, prob.bcs, mesh)

        correct_velocity!(state, mesh)
        update_boundary_velocity!(state, prob.bcs, mesh)
        correct_fluxes!(state, mesh)

        # -- Turbulence (optional) -----------------------------------------
        if turb_model !== nothing
            _update_turbulence!(
                turb_state, turb_model, state, prob, mesh, turb_bcs;
                linear_solver = linear_solver,
            )
        end

        # -- Energy equation + radiation source ----------------------------
        T_eq = CollocatedEquation(mesh)
        assemble_energy!(T_eq, thermal_state.T_field, state.phi, alpha_eff, mesh, bcs_T)

        # Add radiation source to energy RHS (scaled by 1/(rho*Cp))
        rho_Cp = prob.density * thermal_props.Cp
        for c in 1:nc
            T_eq.b[c] += S_rad[c] * mesh.cell_volumes[c] / rho_Cp
        end

        T_sol = _dispatch_solve(to_linear_problem(T_eq), linear_solver, solver_config, :T)
        for c in 1:nc
            thermal_state.T_field.internal[c] = T_sol.u[c]
        end

        # -- P1 radiation --------------------------------------------------
        rad_state.G = solve_p1_radiation(
            rad_model, thermal_state.T_field, mesh, bcs_G;
            linear_solver = linear_solver, solver_config = solver_config,
        )
        S_rad = compute_radiation_source(rad_model, rad_state.G, thermal_state.T_field)

        # -- Convergence ---------------------------------------------------
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

        if max_res < algo.tolerance
            converged = true
            break
        end
    end

    result = SolveResult{Dim, T}(converged, final_iter, residuals, state)
    return (result, thermal_state, rad_state)
end
