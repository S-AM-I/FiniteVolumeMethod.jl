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
- `porous_zones` — optional Darcy-Forchheimer zones (see [`assemble_momentum!`](@ref))
- `mrf_zones` — optional MRF zones (see [`assemble_momentum!`](@ref))
"""
function _pimple_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T;
        linear_solver = nothing,
        solver_config = nothing,
        cyclic_pairs::Vector{Vector{Tuple{Int, Int}}} = Vector{Vector{Tuple{Int, Int}}}(),
        t::T = zero(T),
        ws = nothing,
        nu_eff::Union{T, Vector{T}} = prob.nu,
        body_force::Union{Nothing, Vector{SVector{Dim, T}}} = nothing,
        scheme::ConvectionScheme = CONV_UPWIND,
        blend::T = T(0.5),
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = prob.model.porous_zones,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = prob.model.mrf_zones,
    ) where {Dim, T}
    algo = prob.algorithm::PIMPLE{T}
    mesh = prob.mesh
    n_outer = algo.n_outer
    n_correctors = algo.n_correctors
    alpha_U = algo.alpha_U
    alpha_p = algo.alpha_p

    # Snapshot the old-time velocity ONCE per time step: all outer
    # iterations discretize (U^{n+1} - U^n)/dt against this snapshot.
    # (Previously the ddt was assembled against the previous OUTER
    # ITERATE, so outer iterations 2+ degenerated toward the steady
    # equations.)
    _snapshot_old_time!(state)

    eqs, p_eq = ws === nothing ? _make_incompressible_workspace(prob, cyclic_pairs) : ws

    for outer in 1:n_outer
        is_final = (outer == n_outer)

        # ── 1. Assemble momentum ────────────────────────────────────
        for d in 1:Dim
            reset!(eqs[d])
            assemble_momentum!(
                eqs[d], state, prob, d; dt = dt, t = t,
                nu_eff = nu_eff, body_force = body_force,
                scheme = scheme, blend = blend,
                porous_zones = porous_zones, mrf_zones = mrf_zones,
            )
            apply_cyclic_to_equation!(
                eqs[d], _make_scalar_field(_extract_component(state.U, d), state),
                mesh, cyclic_pairs,
            )
        end

        # ── 2. Under-relax (except final outer iteration) + solve ───
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
        update_boundary_velocity!(state, prob.bcs, mesh; t = t)
        update_boundary_cyclic!(state, mesh, cyclic_pairs)

        # ── 3. Extract operators from the (relaxed) solved equations ─
        extract_momentum_operators!(state, eqs, mesh; porous_zones = porous_zones)

        # ── 4. PISO inner corrector loop ────────────────────────────
        nc = length(mesh.cell_volumes)
        for k in 1:n_correctors
            # 4a. Pressure solve
            reset!(p_eq)
            assemble_pressure!(p_eq, state, prob; mrf_zones = mrf_zones)
            apply_cyclic_to_equation!(p_eq, state.p, mesh, cyclic_pairs)
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
    # Deep-copy the flat backing vector and rebuild the U/p views into the copy,
    # so the snapshot is an independent, identically-typed flat-backed state.
    nc = length(mesh.cell_volumes)
    u = copy(state.u)
    U_internal = reinterpret(SVector{Dim, T}, view(u, 1:(nc * Dim)))
    p_internal = view(u, (nc * Dim + 1):(nc * Dim + nc))
    U = CollocatedVectorField{Dim, T}(
        state.U.name,
        U_internal,
        copy(state.U.boundary),
        copy(state.U.boundary_face_indices),
    )
    p = CollocatedScalarField{T}(
        state.p.name,
        p_internal,
        copy(state.p.boundary),
        copy(state.p.boundary_face_indices),
    )
    phi = FaceFluxField{T}(state.phi.name, copy(state.phi.values))
    return IncompressibleState{Dim, T, typeof(U), typeof(p), typeof(phi)}(
        u, U, p, phi, copy(state.A_P), copy(state.H_U), copy(state.U_old),
    )
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
- `cfl_max::Union{Nothing, T}` — if set, adaptively adjust `dt` each step
  to keep the maximum face Courant number below this limit (default: `nothing`,
  i.e. fixed time step)
- `U0::Union{Nothing, Vector{SVector{Dim, T}}}` — initial cell velocities
  (default `nothing` = zero field)
- `p0::Union{Nothing, Vector{T}}` — initial cell pressures
  (default `nothing` = zero field)
- `porous_zones` — optional Darcy-Forchheimer zones (see [`assemble_momentum!`](@ref))
- `mrf_zones` — optional MRF zones (see [`assemble_momentum!`](@ref))

# Returns
A [`SolveResult`](@ref) with:
- `converged` — `true` iff the run completed with finite final residuals
- `iterations` = number of time steps taken
- `residuals[:continuity]` = continuity residual at each saved step
- `state` = final solver state
- `snapshots` = state snapshots saved every `save_every` steps
"""
function solve_incompressible(
        prob::IncompressibleProblem{Dim, T},
        tspan::Tuple{T, T},
        dt::T;
        save_every::Int = 1,
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
        cfl_max::Union{Nothing, T} = nothing,
        U0::Union{Nothing, Vector{SVector{Dim, T}}} = nothing,
        p0::Union{Nothing, Vector{T}} = nothing,
        porous_zones::Union{Nothing, Vector{PorousZone{T}}} = prob.model.porous_zones,
        mrf_zones::Union{Nothing, Vector{MRFZone{T}}} = prob.model.mrf_zones,
    ) where {Dim, T}
    mesh = prob.mesh
    algo = prob.algorithm
    t_start, t_end = tspan
    nc = length(mesh.cell_volumes)

    # Initialize state (with optional initial conditions)
    state = IncompressibleState(mesh)
    if U0 !== nothing
        length(U0) == nc || throw(ArgumentError("U0 must have length ncells = $nc"))
        copyto!(state.U.internal, U0)
    end
    if p0 !== nothing
        length(p0) == nc || throw(ArgumentError("p0 must have length ncells = $nc"))
        copyto!(state.p.internal, p0)
    end
    update_boundary_velocity!(state, prob.bcs, mesh; t = t_start)
    update_boundary_pressure!(state, prob.bcs, mesh)
    if U0 !== nothing
        # Consistent initial face fluxes from the initial velocity
        compute_face_flux!(state.phi, state.U, mesh)
        if mrf_zones !== nothing
            mrf_make_relative!(state.phi.values, mesh, mrf_zones)
        end
    end

    # Pre-compute cyclic face pairs (empty vector if no CyclicBC)
    cyclic_pairs = collect_cyclic_pairs(prob.bcs, mesh)

    # Equation workspace: allocate once per solve, reuse in every step
    ws = _make_incompressible_workspace(prob, cyclic_pairs)

    # Determine step function
    step_fn! = _select_step_function(algo, cyclic_pairs)

    # Residual + snapshot tracking
    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )
    snapshots = IncompressibleState{Dim, T}[]

    # Time-stepping loop
    t = t_start
    n_steps = 0
    dt_current = dt
    while t < t_end - eps(T) * abs(t_end)
        dt_actual = min(dt_current, t_end - t)
        # BCs of the implicit step are evaluated at the NEW time level
        step_fn!(
            state, prob, dt_actual;
            linear_solver = linear_solver, solver_config = solver_config,
            t = t + dt_actual, ws = ws,
            porous_zones = porous_zones, mrf_zones = mrf_zones,
        )
        t += dt_actual
        n_steps += 1

        # Adaptive CFL: adjust dt for next step based on current Courant number
        if cfl_max !== nothing && n_steps > 1
            co = compute_max_courant(state, mesh, dt_actual)
            if co > eps(T)
                dt_current = min(dt, cfl_max / co * dt_actual)
            end
        end

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

    # A transient run "converged" iff it completed with finite residuals
    # (converged used to be hardcoded true, masking NaN/Inf blow-ups).
    r_hist = residuals[:continuity]
    converged = isempty(r_hist) || isfinite(r_hist[end])

    return SolveResult{Dim, T}(converged, n_steps, residuals, state, snapshots)
end

# ── Step function dispatch ─────────────────────────────────────────

"""
    _select_step_function(algo) -> Function

Return the appropriate single-step function for the given algorithm type.
"""
function _select_step_function(
        algo::PISO,
        cyclic_pairs::Vector{Vector{Tuple{Int, Int}}} = Vector{Vector{Tuple{Int, Int}}}(),
    )
    n_correctors = algo.n_correctors
    return (
        state, prob, dt;
        linear_solver = nothing, solver_config = nothing,
        t = zero(dt), ws = nothing,
        porous_zones = nothing, mrf_zones = nothing,
    ) ->
    _piso_step!(
        state, prob, dt, n_correctors;
        linear_solver = linear_solver, solver_config = solver_config,
        cyclic_pairs = cyclic_pairs, t = t, ws = ws,
        porous_zones = porous_zones, mrf_zones = mrf_zones,
    )
end

function _select_step_function(
        algo::PIMPLE,
        cyclic_pairs::Vector{Vector{Tuple{Int, Int}}} = Vector{Vector{Tuple{Int, Int}}}(),
    )
    return (
        state, prob, dt;
        linear_solver = nothing, solver_config = nothing,
        t = zero(dt), ws = nothing,
        porous_zones = nothing, mrf_zones = nothing,
    ) ->
    _pimple_step!(
        state, prob, dt;
        linear_solver = linear_solver, solver_config = solver_config,
        cyclic_pairs = cyclic_pairs, t = t, ws = ws,
        porous_zones = porous_zones, mrf_zones = mrf_zones,
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
