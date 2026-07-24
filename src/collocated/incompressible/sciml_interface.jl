# incompressible/sciml_interface.jl — SciML CommonSolve dispatch
#
# Enables the standard `solve(prob, alg; kwargs...)` pattern for
# IncompressibleProblem, returning IncompressibleSolution.
#
# The physics to solve comes from `prob.model` (an `IncompressibleModel`), not
# from keyword arguments: this façade only carries numerics (linear solver,
# convection scheme, time-stepping controls).  Component dependencies are
# validated when the model is constructed, so the dispatch below just reads
# traits and unpacks.

"""
    solve(prob::IncompressibleProblem, alg::SIMPLE; kwargs...)

Solve a steady-state incompressible problem using SIMPLE.
Returns an [`IncompressibleSolution`](@ref) with symbolic field access.

The physics solved is whatever `prob.model` carries — see
[`IncompressibleModel`](@ref).  Pass `model` here to override it for this
solve; the returned solution records the problem actually solved.

# Keyword arguments
- `model` — override `prob.model` for this solve
- `linear_solver`, `solver_config` — linear-solve selection and per-field configuration
- `scheme::ConvectionScheme`, `blend` — momentum convection scheme (plain path;
  see [`solve_simple`](@ref))
- `verbose` — print residuals each iteration
"""
function CommonSolve.solve(
        prob::IncompressibleProblem{Dim, T},
        alg::SIMPLE;
        model = nothing,
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        scheme::ConvectionScheme = CONV_UPWIND,
        blend::T = T(0.5),
    ) where {Dim, T}
    actual_prob = _with_solve_overrides(prob, alg, model)
    physics = actual_prob.model

    if has_combustion(physics)
        combustion = physics.combustion
        thermal = physics.thermal
        result, _thermal_state, _species_state = _solve_reacting(
            actual_prob, thermal, combustion;
            linear_solver = linear_solver, solver_config = solver_config,
            verbose = verbose,
        )
        return IncompressibleSolution(result, actual_prob)
    elseif has_radiation(physics)
        thermal = physics.thermal
        radiation = physics.radiation
        result, _thermal_state, _rad_state = solve_simple_thermal_radiation(
            actual_prob, thermal.properties, radiation.model;
            bcs_T = thermal.bcs, bcs_G = radiation.bcs,
            turb_model = turbulence_model(physics),
            turb_bcs = turbulence_bcs(physics),
            T_init = thermal.T_init,
            linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
        )
        return IncompressibleSolution(result, actual_prob)
    elseif has_thermal(physics)
        thermal = physics.thermal
        result, _thermal_state = solve_simple_thermal(
            actual_prob, thermal.properties;
            bcs_T = thermal.bcs,
            turb_model = turbulence_model(physics),
            turb_bcs = turbulence_bcs(physics),
            T_init = thermal.T_init,
            linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
            porous_zones = physics.porous_zones, mrf_zones = physics.mrf_zones,
        )
        return IncompressibleSolution(result, actual_prob)
    elseif has_turbulence(physics)
        result, _turb_state = solve_simple_turbulent(
            actual_prob, turbulence_model(physics);
            turb_bcs = turbulence_bcs(physics),
            linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
            porous_zones = physics.porous_zones, mrf_zones = physics.mrf_zones,
        )
        return IncompressibleSolution(result, actual_prob)
    else
        result = solve_simple(
            actual_prob;
            linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
            porous_zones = physics.porous_zones, mrf_zones = physics.mrf_zones,
            scheme = scheme, blend = blend,
        )
        return IncompressibleSolution(result, actual_prob)
    end
end

"""
    solve(prob::IncompressibleProblem, alg::Union{PISO, PIMPLE}; tspan, dt, kwargs...)

Solve a transient incompressible problem using PISO or PIMPLE.
Returns an [`IncompressibleSolution`](@ref) with symbolic field access.

The physics solved is whatever `prob.model` carries — see
[`IncompressibleModel`](@ref).  Radiation and combustion have no transient
solve path and are rejected here.

# Keyword arguments
- `tspan`, `dt` — time interval and fixed step
- `model` — override `prob.model` for this solve
- `save_every` — snapshot interval (plain path only; the physics paths reject
  it rather than accept and ignore it)
- `U0`, `p0` — initial velocity and pressure (plain path only)
- `linear_solver`, `solver_config`, `verbose`
"""
function CommonSolve.solve(
        prob::IncompressibleProblem{Dim, T},
        alg::Union{PISO, PIMPLE};
        tspan::Tuple{T, T},
        dt::T,
        model = nothing,
        # `nothing` distinguishes "not requested" from an explicit `1`: only the
        # plain incompressible path can produce snapshots, and the physics paths
        # reject the kwarg rather than accept and ignore it.
        save_every::Union{Nothing, Int} = nothing,
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        U0::Union{Nothing, Vector{SVector{Dim, T}}} = nothing,
        p0::Union{Nothing, Vector{T}} = nothing,
    ) where {Dim, T}
    actual_prob = _with_solve_overrides(prob, alg, model)
    physics = actual_prob.model

    _reject_steady_only_physics(physics)

    if has_thermal(physics)
        _reject_save_every(save_every, "thermal", "the temperature field")
        thermal = physics.thermal
        result, _thermal_state = solve_incompressible_thermal(
            actual_prob, thermal.properties, tspan, dt;
            bcs_T = thermal.bcs,
            turb_model = turbulence_model(physics),
            turb_bcs = turbulence_bcs(physics),
            T_init = thermal.T_init,
            linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
            porous_zones = physics.porous_zones, mrf_zones = physics.mrf_zones,
        )
        return IncompressibleSolution(result, actual_prob)
    elseif has_turbulence(physics)
        _reject_save_every(save_every, "turbulent", "the turbulence state")
        result, _turb_state = solve_incompressible_turbulent(
            actual_prob, turbulence_model(physics), tspan, dt;
            turb_bcs = turbulence_bcs(physics),
            linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
            porous_zones = physics.porous_zones, mrf_zones = physics.mrf_zones,
        )
        return IncompressibleSolution(result, actual_prob)
    else
        result = solve_incompressible(
            actual_prob, tspan, dt;
            save_every = something(save_every, 1), linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
            U0 = U0, p0 = p0,
            porous_zones = physics.porous_zones, mrf_zones = physics.mrf_zones,
        )
        return IncompressibleSolution(result, actual_prob)
    end
end

"""
    solve(prob::IncompressibleProblem; kwargs...)

Solve using the algorithm stored in `prob.algorithm`.
"""
function CommonSolve.solve(
        prob::IncompressibleProblem{Dim, T};
        kwargs...,
    ) where {Dim, T}
    return CommonSolve.solve(prob, prob.algorithm; kwargs...)
end

# ── Façade helpers ──────────────────────────────────────────────────

# Fold the solve-time algorithm and model overrides into the problem, so the
# solution carries the problem that was actually solved rather than the one
# originally constructed.
function _with_solve_overrides(prob, alg, model)
    out = alg === prob.algorithm ? prob : remake(prob; algorithm = alg)
    model === nothing && return out
    return remake(out; model = model)
end

# The reacting solvers take their rate closure positionally, and only the
# multi-step (finite-rate Arrhenius) method accepts a variable-Lewis closure.
function _solve_reacting(
        prob, thermal, combustion;
        linear_solver, solver_config, verbose,
    )
    shared = (
        bcs_T = thermal.bcs, bcs_species = combustion.bcs,
        turb_model = turbulence_model(prob.model),
        turb_bcs = turbulence_bcs(prob.model),
        Y_init = combustion.Y_init, T_init = thermal.T_init,
        linear_solver = linear_solver, solver_config = solver_config,
        verbose = verbose,
    )
    combustion.lewis === nothing && return solve_simple_reacting(
        prob, thermal.properties, combustion.properties, combustion.reaction;
        shared...,
    )
    combustion.reaction isa MultiStepMechanism || throw(
        ArgumentError(
            "CombustionComponent `lewis` requires a MultiStepMechanism reaction " *
                "closure: variable-Lewis species transport is only implemented for " *
                "the finite-rate Arrhenius path, not for $(typeof(combustion.reaction))."
        )
    )
    return solve_simple_reacting(
        prob, thermal.properties, combustion.properties, combustion.reaction;
        shared..., lewis = combustion.lewis,
    )
end

function _reject_steady_only_physics(physics)
    has_radiation(physics) && throw(
        ArgumentError(
            "radiation has no transient solve path: RadiationComponent is supported " *
                "with SIMPLE only. Solve the steady problem, or drop the component."
        )
    )
    has_combustion(physics) && throw(
        ArgumentError(
            "combustion has no transient solve path: CombustionComponent is supported " *
                "with SIMPLE only. Solve the steady problem, or drop the component."
        )
    )
    return nothing
end

function _reject_save_every(save_every, path, omitted)
    save_every === nothing && return nothing
    throw(
        ArgumentError(
            "save_every is not supported on the $path path: SolveResult.snapshots holds " *
                "IncompressibleState only, so a snapshot would silently omit $omitted. " *
                "Drop the kwarg, or use the plain incompressible path."
        )
    )
end
