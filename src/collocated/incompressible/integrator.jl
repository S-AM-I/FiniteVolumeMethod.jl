# incompressible/integrator.jl — CommonSolve integrator interface
#
# `init` / `step!` / `solve!` for the collocated incompressible solvers, so a
# caller can drive the coupling one iteration (SIMPLE) or one time step
# (PISO/PIMPLE) at a time and inspect or modify the state in between.
#
# The integrator does not re-implement any discretisation: it holds the same
# state, workspace and cyclic pairing the batch solvers build, and each `step!`
# calls the shared Stage-5e cores (`_simple_outer_step!`, `_piso_step!`,
# `_pimple_step!`). The stepping is therefore identical to `solve` by
# construction, not by parallel maintenance.

@doc """
    IncompressibleIntegrator

Stateful driver for an incompressible solve, returned by
[`init`](@ref CommonSolve.init).

Advance it with `step!(integrator)` and run it to completion with
`solve!(integrator)`. `integrator.u` is the flat solution vector held by
[`IncompressibleState`](@ref) — velocity block followed by pressure block — so
mutating it between steps feeds straight back into the next assembly.

# Fields
- `prob` — the problem being solved
- `alg` — the pressure-velocity coupling
- `state::IncompressibleState` — full solver state (fields are views into `u`)
- `uprev::Vector` — copy of `u` from before the most recent step
- `t`, `dt`, `tfinal` — time bookkeeping (transient couplings; zero for SIMPLE)
- `iter` — completed steps (SIMPLE outer iterations, or time steps)
- `converged` — tolerance met (SIMPLE) / run finished with finite residuals
- `residuals::Dict{Symbol, Vector}` — residual history per equation

# Scope
Plain incompressible flow only. Problems carrying an
[`IncompressibleModel`](@ref) with turbulence, thermal, radiation or combustion
components are rejected by `init`: those couplings interleave scalar transports
between the momentum and pressure stages, and are driven by `solve`.
"""
mutable struct IncompressibleIntegrator{P, A, S, WS, CP, T, LS, SC, L}
    prob::P
    alg::A
    state::S
    ws::WS
    cyclic_pairs::CP
    uprev::Vector{T}
    residuals::Dict{Symbol, Vector{T}}
    component_labels::L
    t::T
    dt::T
    tfinal::T
    iter::Int
    converged::Bool
    linear_solver::LS
    solver_config::SC
    scheme::ConvectionScheme
    blend::T
    verbose::Bool
end

# `integrator.u` is the canonical SciML name for the state vector; it lives on
# the state so that the field views stay attached to it.
function Base.getproperty(integrator::IncompressibleIntegrator, sym::Symbol)
    sym === :u && return getfield(integrator, :state).u
    return getfield(integrator, sym)
end

function Base.propertynames(::IncompressibleIntegrator)
    return (
        :u, :uprev, :prob, :alg, :state, :t, :dt, :tfinal, :iter, :converged,
        :residuals,
    )
end

function _reject_integrator_physics(prob)
    is_plain_flow(prob.model) && return nothing
    return throw(
        ArgumentError(
            "the integrator supports plain incompressible flow only: this problem's " *
                "IncompressibleModel carries physics components whose scalar transports " *
                "are interleaved with the momentum/pressure stages. Use `solve(prob, alg)`."
        )
    )
end

function _init_common(prob::AnyIncompressibleProblem{Dim, T}, t_start::T) where {Dim, T}
    mesh = prob.mesh
    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh; t = t_start)
    update_boundary_pressure!(state, prob.bcs, mesh)
    cyclic_pairs = collect_cyclic_pairs(prob.bcs, mesh)
    ws = _make_incompressible_workspace(prob, cyclic_pairs)
    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )
    return state, cyclic_pairs, ws, component_labels, residuals
end

@doc """
    init(prob::SteadyIncompressibleProblem, alg::SIMPLE = prob.algorithm; kwargs...)

Build an [`IncompressibleIntegrator`](@ref) that advances one SIMPLE outer
iteration per `step!`.
"""
function CommonSolve.init(
        prob::SteadyIncompressibleProblem{Dim, T},
        alg::SIMPLE = prob.algorithm;
        linear_solver = nothing,
        solver_config = nothing,
        scheme::ConvectionScheme = CONV_UPWIND,
        blend::T = T(0.5),
        verbose::Bool = false,
    ) where {Dim, T}
    _reject_integrator_physics(prob)
    actual_prob = alg === prob.algorithm ? prob : remake(prob; algorithm = alg)
    state, cyclic_pairs, ws, labels, residuals = _init_common(actual_prob, zero(T))
    return IncompressibleIntegrator(
        actual_prob, alg, state, ws, cyclic_pairs, copy(state.u), residuals, labels,
        zero(T), zero(T), zero(T), 0, false,
        linear_solver, solver_config, scheme, blend, verbose,
    )
end

@doc """
    init(prob::IncompressibleProblem, alg = prob.algorithm; tspan, dt, kwargs...)

Build an [`IncompressibleIntegrator`](@ref) that advances one time step of
size `dt` per `step!`, using PISO or PIMPLE.
"""
function CommonSolve.init(
        prob::IncompressibleProblem{Dim, T},
        alg::Union{PISO, PIMPLE} = prob.algorithm;
        tspan::Tuple{T, T},
        dt::T,
        linear_solver = nothing,
        solver_config = nothing,
        scheme::ConvectionScheme = CONV_UPWIND,
        blend::T = T(0.5),
        U0::Union{Nothing, Vector{SVector{Dim, T}}} = nothing,
        p0::Union{Nothing, Vector{T}} = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    _reject_integrator_physics(prob)
    actual_prob = alg === prob.algorithm ? prob : remake(prob; algorithm = alg)
    mesh = actual_prob.mesh
    nc = length(mesh.cell_volumes)
    t_start, t_end = tspan
    state, cyclic_pairs, ws, labels, residuals = _init_common(actual_prob, t_start)

    if U0 !== nothing
        length(U0) == nc || throw(ArgumentError("U0 must have length ncells = $nc"))
        copyto!(state.U.internal, U0)
        update_boundary_velocity!(state, actual_prob.bcs, mesh; t = t_start)
        # Consistent initial face fluxes from the initial velocity.
        compute_face_flux!(state.phi, state.U, mesh)
        mrf = actual_prob.model.mrf_zones
        mrf === nothing || mrf_make_relative!(state.phi.values, mesh, mrf)
    end
    if p0 !== nothing
        length(p0) == nc || throw(ArgumentError("p0 must have length ncells = $nc"))
        copyto!(state.p.internal, p0)
        update_boundary_pressure!(state, actual_prob.bcs, mesh)
    end

    return IncompressibleIntegrator(
        actual_prob, alg, state, ws, cyclic_pairs, copy(state.u), residuals, labels,
        t_start, dt, t_end, 0, false,
        linear_solver, solver_config, scheme, blend, verbose,
    )
end

@doc """
    step!(integrator::IncompressibleIntegrator)

Advance one SIMPLE outer iteration, or one time step for PISO/PIMPLE.
Returns the integrator.
"""
function CommonSolve.step!(
        integrator::IncompressibleIntegrator{P, <:SIMPLE},
    ) where {P}
    copyto!(integrator.uprev, integrator.state.u)
    prob = integrator.prob
    algo = prob.algorithm
    eqs, p_eq = integrator.ws
    max_res = _simple_outer_step!(
        integrator.state, prob, eqs, p_eq, integrator.cyclic_pairs,
        integrator.residuals, integrator.component_labels;
        scheme = integrator.scheme, blend = integrator.blend,
        linear_solver = integrator.linear_solver,
        solver_config = integrator.solver_config,
    )
    integrator.iter += 1
    if integrator.verbose
        _print_simple_residuals(
            integrator.iter, integrator.residuals, integrator.component_labels,
        )
    end
    # Never declare convergence on the first outer iteration — the startup
    # iterate's residuals are degenerate (see `solve_simple`).
    integrator.converged = integrator.iter > 1 && max_res < algo.tolerance
    return integrator
end

function CommonSolve.step!(
        integrator::IncompressibleIntegrator{P, <:Union{PISO, PIMPLE}},
    ) where {P}
    copyto!(integrator.uprev, integrator.state.u)
    prob = integrator.prob
    mesh = prob.mesh
    dt_actual = min(integrator.dt, integrator.tfinal - integrator.t)
    step_fn! = _select_step_function(prob.algorithm, integrator.cyclic_pairs)
    # Boundary conditions of the implicit step are evaluated at the NEW time.
    step_fn!(
        integrator.state, prob, dt_actual;
        linear_solver = integrator.linear_solver,
        solver_config = integrator.solver_config,
        t = integrator.t + dt_actual, ws = integrator.ws,
    )
    integrator.t += dt_actual
    integrator.iter += 1
    r_cont = continuity_residual(integrator.state, mesh)
    push!(integrator.residuals[:continuity], r_cont)
    integrator.converged = isfinite(r_cont)
    if integrator.verbose
        _print_transient_progress(integrator.iter, integrator.t, r_cont)
    end
    return integrator
end

@doc """
    solve!(integrator::IncompressibleIntegrator) -> IncompressibleSolution

Run the integrator to completion — to `max_iterations` or the convergence
tolerance for SIMPLE, to the end of `tspan` for PISO/PIMPLE — and wrap the
final state in an [`IncompressibleSolution`](@ref).
"""
function CommonSolve.solve!(
        integrator::IncompressibleIntegrator{P, <:SIMPLE},
    ) where {P}
    max_iter = integrator.prob.algorithm.max_iterations
    while integrator.iter < max_iter && !integrator.converged
        CommonSolve.step!(integrator)
    end
    return _integrator_solution(integrator)
end

function CommonSolve.solve!(
        integrator::IncompressibleIntegrator{P, <:Union{PISO, PIMPLE}},
    ) where {P}
    while integrator.t < integrator.tfinal - eps(integrator.tfinal) * abs(integrator.tfinal)
        CommonSolve.step!(integrator)
    end
    # A transient run "converged" iff it completed with finite residuals.
    r_hist = integrator.residuals[:continuity]
    integrator.converged = isempty(r_hist) || isfinite(r_hist[end])
    return _integrator_solution(integrator)
end

function _integrator_solution(
        integrator::IncompressibleIntegrator{P, A, S, WS, CP, T},
    ) where {P, A, S, WS, CP, T}
    Dim = _problem_dim(integrator.prob)
    result = SolveResult{Dim, T}(
        integrator.converged, integrator.iter, integrator.residuals, integrator.state,
    )
    return IncompressibleSolution(result, integrator.prob)
end

_problem_dim(::AnyIncompressibleProblem{Dim}) where {Dim} = Dim
