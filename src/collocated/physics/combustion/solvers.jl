# combustion/solvers.jl — Combined reacting flow solver
#
# Wraps the SIMPLE loop with turbulence, EDM reaction rates,
# species transport, and energy equation with heat release.

using Printf: @sprintf

"""
    solve_simple_reacting(
        prob, thermal_props, combustion_props, edm;
        bcs_T, bcs_species,
        turb_model, turb_bcs,
        Y_init, T_init,
        linear_solver, verbose,
    ) -> Tuple{SolveResult, ThermalState, SpeciesState}

Solve steady incompressible reacting flow with SIMPLE.

Each iteration:
1. Momentum + pressure solve (with `nu_eff` from turbulence)
2. Turbulence solve (provides k, ε for EDM)
3. Compute EDM reaction rates
4. Solve species transport with reaction sources
5. Compute heat release from reaction rates
6. Solve energy equation with heat release in RHS
7. Check convergence

# Arguments
- `prob::IncompressibleProblem` — flow problem
- `thermal_props::FluidThermalProperties` — thermal properties
- `combustion_props::CombustionProperties` — thermochemical properties
- `edm::EddyDissipationModel` — EDM constants
- `bcs_T` — temperature boundary conditions
- `bcs_species` — species BCs: `Dict{Symbol, Dict{Symbol, BC}}`
- `turb_model` — RANS turbulence model (or `nothing`)
- `turb_bcs` — turbulence BCs
- `Y_init` — initial mass fractions: `Dict{Symbol, T}`
- `T_init` — initial temperature
- `linear_solver` — linear solver algorithm
- `verbose` — print residuals each iteration
"""
function solve_simple_reacting(
        prob::IncompressibleProblem{Dim, T},
        thermal_props::FluidThermalProperties{Dim, T},
        combustion_props::CombustionProperties{NS, T},
        edm::EddyDissipationModel{T};
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition},
        bcs_species::Dict{Symbol, <:Any},
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        Y_init::Dict{Symbol, <:Real} = Dict{Symbol, T}(),
        T_init::Real = thermal_props.T_ref,
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
    ) where {Dim, T, NS}
    algo = prob.algorithm::SIMPLE{T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    # Initialize flow state
    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    # Initialize thermal state
    thermal_state = ThermalState(mesh; T_init = T(T_init), k_init = thermal_props.k)

    # Initialize species state
    species_state = SpeciesState(
        mesh, combustion_props; (
            name => T(get(Y_init, name, zero(T)))
                for name in combustion_props.species_names
        )...
    )

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

        # Extract A_P/H(U) from the RELAXED, solved momentum equations —
        # standard SIMPLE ordering, so that D = V/A_P in the pressure
        # equation (and in the Rhie-Chow face flux, which shares the same
        # face D_f) is consistent with the velocity the momentum solve
        # actually produced.
        extract_momentum_operators!(state, eqs, mesh)

        # ── Pressure ────────────────────────────────────────────
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

        # ── Turbulence (optional) ───────────────────────────────
        if turb_model !== nothing
            _update_turbulence!(
                turb_state, turb_model, state, prob, mesh, turb_bcs;
                linear_solver = linear_solver,
            )
        end

        # ── EDM reaction rates ──────────────────────────────────
        k_field = if turb_state !== nothing && haskey(turb_state.fields, :k)
            turb_state.fields[:k].internal
        else
            nothing
        end
        eps_field = if turb_state !== nothing && haskey(turb_state.fields, :epsilon)
            turb_state.fields[:epsilon].internal
        else
            nothing
        end

        reaction_rates = compute_edm_reaction_rates(
            edm, species_state, combustion_props,
            k_field, eps_field, prob.density, mesh,
        )

        # ── Species transport ───────────────────────────────────
        solve_species!(
            species_state, state.phi, combustion_props, reaction_rates,
            nu_t_vec, prob.density, mesh, bcs_species;
            dt = nothing, linear_solver = linear_solver, solver_config = solver_config,
        )

        # ── Energy equation with heat release ───────────────────
        S_h = compute_heat_release(reaction_rates, combustion_props)

        T_eq = CollocatedEquation(mesh)
        assemble_energy!(T_eq, thermal_state.T_field, state.phi, alpha_eff, mesh, bcs_T)

        # Add heat release to energy RHS: S_h / (ρ·Cp) × V_c
        rho_Cp = prob.density * thermal_props.Cp
        for c in 1:nc
            T_eq.b[c] += S_h[c] * mesh.cell_volumes[c] / rho_Cp
        end

        T_sol = _dispatch_solve(to_linear_problem(T_eq), linear_solver, solver_config, :T)
        for c in 1:nc
            thermal_state.T_field.internal[c] = T_sol.u[c]
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

        # Never declare convergence on the FIRST outer iteration: the startup
        # iterate's residuals are degenerate when the initial fields solve the
        # momentum equations trivially (U = 0, uniform T ⇒ zero body force),
        # and temperature/species have not yet fed back into momentum.
        if iter > 1 && max_res < algo.tolerance
            converged = true
            break
        end
    end

    result = SolveResult{Dim, T}(converged, final_iter, residuals, state)
    return (result, thermal_state, species_state)
end

"""
    solve_simple_reacting(prob, thermal_props, combustion_props, mechanism::MultiStepMechanism; kwargs...) -> (SolveResult, ThermalState, SpeciesState)

Multi-step finite-rate Arrhenius variant of [`solve_simple_reacting`](@ref).

The reaction-rate source at each SIMPLE iteration is computed from
`mechanism::MultiStepMechanism` via [`compute_multi_step_rates`](@ref)
(temperature-dependent, chemistry-limited). All other kwargs match the
EDM dispatch. Pass `lewis` / `alpha_thermal` to enable variable Lewis
species transport.
"""
function solve_simple_reacting(
        prob::IncompressibleProblem{Dim, T},
        thermal_props::FluidThermalProperties{Dim, T},
        combustion_props::CombustionProperties{NS, T},
        mechanism::MultiStepMechanism{NR, NS, T};
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition},
        bcs_species::Dict{Symbol, <:Any},
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        Y_init::Dict{Symbol, <:Real} = Dict{Symbol, T}(),
        T_init::Real = thermal_props.T_ref,
        linear_solver = nothing,
        solver_config = nothing,
        lewis::Union{Nothing, VariableLewis{NS, T}} = nothing,
        verbose::Bool = false,
    ) where {Dim, T, NR, NS}
    algo = prob.algorithm::SIMPLE{T}
    mesh = prob.mesh
    nc = length(mesh.cell_volumes)

    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    thermal_state = ThermalState(mesh; T_init = T(T_init), k_init = thermal_props.k)

    species_state = SpeciesState(
        mesh, combustion_props; (
            name => T(get(Y_init, name, zero(T)))
                for name in combustion_props.species_names
        )...
    )

    turb_state = nothing
    if turb_model !== nothing
        turb_state = RANSTurbulenceState(turb_model, mesh)
        turbulent_viscosity!(turb_state.nu_t, turb_model, turb_state, mesh)
    end

    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    converged = false
    final_iter = 0

    for iter in 1:algo.max_iterations
        final_iter = iter

        nu_t_vec = turb_state === nothing ? nothing : turb_state.nu_t
        update_k_eff!(thermal_state, thermal_props, nu_t_vec, prob.density)
        nu_eff = turb_state === nothing ? prob.nu : compute_nu_eff(prob.nu, turb_state.nu_t)
        alpha_eff = compute_alpha_eff(thermal_state.k_eff, prob.density, thermal_props.Cp)

        body_force = compute_buoyancy_source(thermal_state.T_field, thermal_props, prob.density)

        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(
                eq, state, prob, d;
                nu_eff = nu_eff, body_force = body_force,
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

        # Extract A_P/H(U) from the RELAXED, solved momentum equations —
        # standard SIMPLE ordering, so that D = V/A_P in the pressure
        # equation (and in the Rhie-Chow face flux, which shares the same
        # face D_f) is consistent with the velocity the momentum solve
        # actually produced.
        extract_momentum_operators!(state, eqs, mesh)

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

        if turb_model !== nothing
            _update_turbulence!(
                turb_state, turb_model, state, prob, mesh, turb_bcs;
                linear_solver = linear_solver,
            )
        end

        # Multi-step reaction rates (temperature-dependent).
        reaction_rates = compute_multi_step_rates(
            mechanism, species_state, thermal_state.T_field, prob.density, mesh,
        )

        # Species transport (honours `lewis` if supplied).
        alpha_species_arg = lewis === nothing ? nothing : alpha_eff
        solve_species!(
            species_state, state.phi, combustion_props, reaction_rates,
            nu_t_vec, prob.density, mesh, bcs_species;
            dt = nothing, linear_solver = linear_solver, solver_config = solver_config,
            lewis = lewis, alpha_thermal = alpha_species_arg,
        )

        S_h = compute_heat_release(reaction_rates, combustion_props)

        T_eq = CollocatedEquation(mesh)
        assemble_energy!(T_eq, thermal_state.T_field, state.phi, alpha_eff, mesh, bcs_T)

        rho_Cp = prob.density * thermal_props.Cp
        for c in 1:nc
            T_eq.b[c] += S_h[c] * mesh.cell_volumes[c] / rho_Cp
        end

        T_sol = _dispatch_solve(to_linear_problem(T_eq), linear_solver, solver_config, :T)
        for c in 1:nc
            thermal_state.T_field.internal[c] = T_sol.u[c]
        end

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

        # Never declare convergence on the FIRST outer iteration: the startup
        # iterate's residuals are degenerate when the initial fields solve the
        # momentum equations trivially (U = 0, uniform T ⇒ zero body force),
        # and temperature/species have not yet fed back into momentum.
        if iter > 1 && max_res < algo.tolerance
            converged = true
            break
        end
    end

    result = SolveResult{Dim, T}(converged, final_iter, residuals, state)
    return (result, thermal_state, species_state)
end
