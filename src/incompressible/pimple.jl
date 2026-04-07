# incompressible/pimple.jl — PIMPLE step + unified transient solver
#
# Implements the merged PISO-SIMPLE (PIMPLE) algorithm as a single-step
# function and provides `solve_incompressible`, the unified transient
# entry point that dispatches to PISO or PIMPLE per time step.

using Printf: @sprintf

# ── PIMPLE step ────────────────────────────────────────────────────

@doc """
    _pimple_step!(state, prob, dt; linear_solver = nothing)

Advance the incompressible solver state by one time step `dt` using the
[`PIMPLE`](@ref) algorithm.

PIMPLE combines outer SIMPLE-like iterations with inner PISO pressure
correctors:
- **Outer loop** (`n_outer` passes): Each pass assembles the momentum
  equations with under-relaxation (except the final pass), solves for
  velocity, then runs `n_correctors` pressure correction steps.
- **Inner loop** (`n_correctors` per outer): Each correction assembles
  the pressure Poisson equation, solves it, and applies velocity/flux
  corrections.  Pressure is under-relaxed during non-final outer
  iterations and directly assigned in the final one.

# Arguments
- `state::IncompressibleState{Dim, T}` — solver state (modified in-place)
- `prob::IncompressibleProblem{Dim, T}` — problem definition with [`PIMPLE`](@ref) algorithm
- `dt::T` — time step size

# Keyword Arguments
- `linear_solver` — solver algorithm for `LinearProblem` (default: `nothing`)
"""
function _pimple_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T;
        linear_solver = nothing,
        solver_config = nothing,
    ) where {Dim, T}
    algo = prob.algorithm::PIMPLE{T}
    mesh = prob.mesh
    n_outer = algo.n_outer
    n_correctors = algo.n_correctors
    alpha_U = algo.alpha_U
    alpha_p = algo.alpha_p

    for outer in 1:n_outer
        is_final = (outer == n_outer)

        # ── 1. Assemble momentum ────────────────────────────────────
        eqs = CollocatedEquation{T}[]
        for d in 1:Dim
            eq = CollocatedEquation(mesh)
            assemble_momentum!(eq, state, prob, d; dt = dt)
            push!(eqs, eq)
        end

        # ── 2. Extract operators ────────────────────────────────────
        extract_momentum_operators!(state, eqs, mesh)

        # ── 3. Under-relax (except final outer iteration) + solve ───
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

        # ── 4. PISO inner corrector loop ────────────────────────────
        nc = length(mesh.cell_volumes)
        for k in 1:n_correctors
            # 4a. Pressure solve
            p_eq = CollocatedEquation(mesh)
            assemble_pressure!(p_eq, state, prob)
            if _needs_pressure_reference(prob.bcs)
                fix_pressure_reference!(p_eq, 1, zero(T))
            end
            lp_p = to_linear_problem(p_eq)
            p_sol = _dispatch_solve(lp_p, linear_solver, solver_config, :p)

            # 4b. Under-relax pressure if not final outer, else direct
            if !is_final
                for c in 1:nc
                    state.p.internal[c] += alpha_p * (p_sol.u[c] - state.p.internal[c])
                end
            else
                for c in 1:nc
                    state.p.internal[c] = p_sol.u[c]
                end
            end

            # 4c. Update boundary pressure
            update_boundary_pressure!(state, prob.bcs, mesh)

            # 4d. Correct velocity + fluxes
            correct_velocity!(state, mesh)
            update_boundary_velocity!(state, prob.bcs, mesh)
            correct_fluxes!(state, mesh)
        end
    end

    return nothing
end

# ── State copying ──────────────────────────────────────────────────

@doc """
    _copy_state(state::IncompressibleState{Dim, T}, mesh) -> IncompressibleState{Dim, T}

Create an independent deep copy of the solver state.

All field vectors are copied so that mutating the returned state does not
affect the original.
"""
function _copy_state(
        state::IncompressibleState{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    U = CollocatedVectorField{Dim, T}(
        state.U.name,
        copy(state.U.internal),
        copy(state.U.boundary),
        copy(state.U.boundary_face_indices),
    )
    p = CollocatedScalarField{T}(
        state.p.name,
        copy(state.p.internal),
        copy(state.p.boundary),
        copy(state.p.boundary_face_indices),
    )
    phi = FaceFluxField{T}(state.phi.name, copy(state.phi.values))
    A_P = copy(state.A_P)
    H_U = copy(state.H_U)
    return IncompressibleState{Dim, T}(U, p, phi, A_P, H_U)
end

# ── Unified transient solver ───────────────────────────────────────

@doc """
    solve_incompressible(prob, tspan, dt; kwargs...) -> SolveResult{Dim, T}

Solve a transient incompressible Navier-Stokes problem over the time
interval `tspan = (t_start, t_end)` with fixed time step `dt`.

Dispatches to [`_piso_step!`](@ref) for [`PISO`](@ref) algorithms or
[`_pimple_step!`](@ref) for [`PIMPLE`](@ref) algorithms.

State snapshots are stored every `save_every` time steps.  The returned
[`SolveResult`](@ref) contains the final state and residual histories.

# Arguments
- `prob::IncompressibleProblem{Dim, T}` — problem with PISO or PIMPLE algorithm
- `tspan::Tuple{T, T}` — `(t_start, t_end)`
- `dt::T` — time step size

# Keyword Arguments
- `save_every::Int` — save a state snapshot every N steps (default: `1`)
- `linear_solver` — solver algorithm for `LinearProblem` (default: `nothing`)
- `verbose::Bool` — print progress each time step (default: `false`)

# Returns
A [`SolveResult`](@ref) with:
- `converged = true` (transient solvers always complete the time span)
- `iterations` = number of time steps taken
- `residuals[:continuity]` = continuity residual at each saved step
- `state` = final solver state
"""
function solve_incompressible(
        prob::IncompressibleProblem{Dim, T},
        tspan::Tuple{T, T},
        dt::T;
        save_every::Int = 1,
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    mesh = prob.mesh
    algo = prob.algorithm
    t_start, t_end = tspan

    # Initialize state
    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    # Determine step function
    step_fn! = _select_step_function(algo)

    # Residual + snapshot tracking
    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )
    snapshots = IncompressibleState{Dim, T}[]

    # Time-stepping loop
    t = t_start
    n_steps = 0
    while t < t_end - eps(T) * abs(t_end)
        dt_actual = min(dt, t_end - t)
        step_fn!(state, prob, dt_actual; linear_solver = linear_solver, solver_config = solver_config)
        t += dt_actual
        n_steps += 1

        # Record residuals
        r_cont = continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)

        # Save snapshot
        if mod(n_steps, save_every) == 0
            push!(snapshots, _copy_state(state, mesh))
        end

        if verbose
            _print_transient_progress(n_steps, t, r_cont)
        end
    end

    return SolveResult{Dim, T}(true, n_steps, residuals, state)
end

# ── Step function dispatch ─────────────────────────────────────────

"""
    _select_step_function(algo) -> Function

Return the appropriate single-step function for the given algorithm type.
"""
function _select_step_function(algo::PISO)
    n_correctors = algo.n_correctors
    return (state, prob, dt; linear_solver = nothing, solver_config = nothing) ->
    _piso_step!(
        state, prob, dt, n_correctors;
        linear_solver = linear_solver, solver_config = solver_config,
    )
end

function _select_step_function(algo::PIMPLE)
    return (state, prob, dt; linear_solver = nothing, solver_config = nothing) ->
    _pimple_step!(
        state, prob, dt;
        linear_solver = linear_solver, solver_config = solver_config,
    )
end

"""
    _print_transient_progress(step, t, r_cont)

Print a one-line summary of the current time step.
"""
function _print_transient_progress(step, t, r_cont)
    println(
        "Step ", lpad(step, 6),
        "  t=", @sprintf("%.4e", t),
        "  cont=", @sprintf("%.3e", r_cont),
    )
    return nothing
end
