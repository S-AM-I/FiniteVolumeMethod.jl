# incompressible/sciml_interface.jl — SciML CommonSolve dispatch
#
# Enables standard `solve(prob, alg; kwargs...)` pattern for
# IncompressibleProblem, returning IncompressibleSolution.
#
# A single `solve` method per algorithm type accepts optional physics
# kwargs (turbulence, thermal, radiation, combustion).  When all are
# `nothing` the plain incompressible solver runs.

"""
    solve(prob::IncompressibleProblem, alg::SIMPLE; kwargs...)

Solve a steady-state incompressible problem using SIMPLE.
Returns an [`IncompressibleSolution`](@ref) with symbolic field access.

# Optional physics kwargs
- `turb_model` — RANS turbulence model (e.g. `KEpsilonModel()`)
- `turb_bcs` — turbulence boundary conditions
- `thermal_props::FluidThermalProperties` — enables energy equation
- `bcs_T` — temperature boundary conditions (required when `thermal_props` given)
- `T_init` — initial temperature (defaults to `thermal_props.T_ref`)
- `rad_model::P1Model` — radiation model (requires `thermal_props` and `bcs_G`)
- `bcs_G` — incident radiation boundary conditions
- `combustion_props::CombustionProperties` — enables reacting flow
- `edm::EddyDissipationModel` — EDM reaction model (required with `combustion_props`)
- `bcs_species` — species boundary conditions
- `Y_init` — initial mass fractions `Dict{Symbol, T}`
- `porous_zones::Vector{PorousZone}` — Darcy-Forchheimer porous zones
  (plain, turbulent, and thermal paths; see [`assemble_momentum!`](@ref))
- `mrf_zones::Vector{MRFZone}` — rotating reference-frame zones
  (plain, turbulent, and thermal paths; see [`assemble_momentum!`](@ref))
- `scheme::ConvectionScheme`, `blend` — momentum convection scheme for
  the plain path (default first-order `CONV_UPWIND`; see
  [`solve_simple`](@ref))
"""
function CommonSolve.solve(
        prob::IncompressibleProblem{Dim, T},
        alg::SIMPLE;
        # Base kwargs
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        # Zone kwargs
        porous_zones = nothing,
        mrf_zones = nothing,
        # Convection scheme kwargs (plain path; see solve_simple)
        scheme::ConvectionScheme = CONV_UPWIND,
        blend::T = T(0.5),
        # Turbulence kwargs
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        # Thermal kwargs
        thermal_props = nothing,
        bcs_T = nothing,
        T_init = nothing,
        # Radiation kwargs
        rad_model = nothing,
        bcs_G = nothing,
        # Combustion kwargs
        combustion_props = nothing,
        edm = nothing,
        bcs_species = nothing,
        Y_init = Dict{Symbol, Float64}(),
    ) where {Dim, T}
    actual_prob = alg === prob.algorithm ? prob : remake(prob; algorithm = alg)

    if (porous_zones !== nothing || mrf_zones !== nothing) &&
            (combustion_props !== nothing || rad_model !== nothing)
        throw(
            ArgumentError(
                "porous_zones/mrf_zones are supported on the plain, " *
                    "turbulent, and thermal solve paths only (not with " *
                    "combustion_props or rad_model)"
            )
        )
    end

    if combustion_props !== nothing
        # ── Reacting flow ──────────────────────────────────────
        thermal_props === nothing && throw(
            ArgumentError(
                "combustion_props requires thermal_props"
            )
        )
        bcs_T === nothing && throw(
            ArgumentError(
                "combustion_props requires bcs_T"
            )
        )
        edm === nothing && throw(
            ArgumentError(
                "combustion_props requires edm"
            )
        )
        bcs_species === nothing && throw(
            ArgumentError(
                "combustion_props requires bcs_species"
            )
        )
        actual_T_init = T_init === nothing ? thermal_props.T_ref : T_init
        result, _thermal_state, _species_state = solve_simple_reacting(
            actual_prob, thermal_props, combustion_props, edm;
            bcs_T = bcs_T, bcs_species = bcs_species,
            turb_model = turb_model, turb_bcs = turb_bcs,
            Y_init = Y_init, T_init = actual_T_init,
            linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
        )
        return IncompressibleSolution(result, actual_prob)
    elseif rad_model !== nothing
        # ── Thermal + radiation ────────────────────────────────
        thermal_props === nothing && throw(
            ArgumentError(
                "rad_model requires thermal_props"
            )
        )
        bcs_T === nothing && throw(
            ArgumentError(
                "rad_model requires bcs_T"
            )
        )
        bcs_G === nothing && throw(
            ArgumentError(
                "rad_model requires bcs_G"
            )
        )
        actual_T_init = T_init === nothing ? thermal_props.T_ref : T_init
        result, _thermal_state, _rad_state = solve_simple_thermal_radiation(
            actual_prob, thermal_props, rad_model;
            bcs_T = bcs_T, bcs_G = bcs_G,
            turb_model = turb_model, turb_bcs = turb_bcs,
            T_init = actual_T_init,
            linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
        )
        return IncompressibleSolution(result, actual_prob)
    elseif thermal_props !== nothing
        # ── Thermal (+ optional turbulence) ────────────────────
        bcs_T === nothing && throw(
            ArgumentError(
                "thermal_props requires bcs_T"
            )
        )
        actual_T_init = T_init === nothing ? thermal_props.T_ref : T_init
        result, _thermal_state = solve_simple_thermal(
            actual_prob, thermal_props;
            bcs_T = bcs_T,
            turb_model = turb_model, turb_bcs = turb_bcs,
            T_init = actual_T_init,
            linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
            porous_zones = porous_zones, mrf_zones = mrf_zones,
        )
        return IncompressibleSolution(result, actual_prob)
    elseif turb_model !== nothing
        # ── Turbulence only ────────────────────────────────────
        result, _turb_state = solve_simple_turbulent(
            actual_prob, turb_model;
            turb_bcs = turb_bcs,
            linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
            porous_zones = porous_zones, mrf_zones = mrf_zones,
        )
        return IncompressibleSolution(result, actual_prob)
    else
        # ── Plain incompressible ───────────────────────────────
        result = solve_simple(
            actual_prob;
            linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
            porous_zones = porous_zones, mrf_zones = mrf_zones,
            scheme = scheme, blend = blend,
        )
        return IncompressibleSolution(result, actual_prob)
    end
end

"""
    solve(prob::IncompressibleProblem, alg::Union{PISO, PIMPLE}; tspan, dt, kwargs...)

Solve a transient incompressible problem using PISO or PIMPLE.
Returns an [`IncompressibleSolution`](@ref) with symbolic field access.

# Optional physics kwargs
- `turb_model` — RANS turbulence model
- `turb_bcs` — turbulence boundary conditions
- `thermal_props::FluidThermalProperties` — enables energy equation
- `bcs_T` — temperature boundary conditions (required when `thermal_props` given)
- `T_init` — initial temperature (defaults to `thermal_props.T_ref`)
- `porous_zones::Vector{PorousZone}` — Darcy-Forchheimer porous zones
  (plain, turbulent, and thermal paths; see [`assemble_momentum!`](@ref))
- `mrf_zones::Vector{MRFZone}` — rotating reference-frame zones
  (plain, turbulent, and thermal paths; see [`assemble_momentum!`](@ref))
"""
function CommonSolve.solve(
        prob::IncompressibleProblem{Dim, T},
        alg::Union{PISO, PIMPLE};
        tspan::Tuple{T, T},
        dt::T,
        # `nothing` distinguishes "not requested" from an explicit `1`: only the
        # plain incompressible path can produce snapshots, and the physics paths
        # reject the kwarg rather than accept and ignore it.
        save_every::Union{Nothing, Int} = nothing,
        # Base kwargs
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        # Zone kwargs
        porous_zones = nothing,
        mrf_zones = nothing,
        # Initial conditions (plain incompressible path only)
        U0::Union{Nothing, Vector{SVector{Dim, T}}} = nothing,
        p0::Union{Nothing, Vector{T}} = nothing,
        # Turbulence kwargs
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        # Thermal kwargs
        thermal_props = nothing,
        bcs_T = nothing,
        T_init = nothing,
    ) where {Dim, T}
    actual_prob = alg === prob.algorithm ? prob : remake(prob; algorithm = alg)

    if thermal_props !== nothing
        # ── Thermal (+ optional turbulence) ────────────────────
        bcs_T === nothing && throw(
            ArgumentError(
                "thermal_props requires bcs_T"
            )
        )
        save_every === nothing || throw(
            ArgumentError(
                "save_every is not supported on the thermal path: SolveResult.snapshots " *
                    "holds IncompressibleState only, so a snapshot would silently omit the " *
                    "temperature field. Drop the kwarg, or use the plain incompressible path."
            )
        )
        actual_T_init = T_init === nothing ? thermal_props.T_ref : T_init
        result, _thermal_state = solve_incompressible_thermal(
            actual_prob, thermal_props, tspan, dt;
            bcs_T = bcs_T,
            turb_model = turb_model, turb_bcs = turb_bcs,
            T_init = actual_T_init,
            linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
            porous_zones = porous_zones, mrf_zones = mrf_zones,
        )
        return IncompressibleSolution(result, actual_prob)
    elseif turb_model !== nothing
        # ── Turbulence only ────────────────────────────────────
        save_every === nothing || throw(
            ArgumentError(
                "save_every is not supported on the turbulent path: SolveResult.snapshots " *
                    "holds IncompressibleState only, so a snapshot would silently omit the " *
                    "turbulence state. Drop the kwarg, or use the plain incompressible path."
            )
        )
        result, _turb_state = solve_incompressible_turbulent(
            actual_prob, turb_model, tspan, dt;
            turb_bcs = turb_bcs,
            linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
            porous_zones = porous_zones, mrf_zones = mrf_zones,
        )
        return IncompressibleSolution(result, actual_prob)
    else
        # ── Plain incompressible ───────────────────────────────
        result = solve_incompressible(
            actual_prob, tspan, dt;
            save_every = something(save_every, 1), linear_solver = linear_solver,
            solver_config = solver_config, verbose = verbose,
            U0 = U0, p0 = p0,
            porous_zones = porous_zones, mrf_zones = mrf_zones,
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
