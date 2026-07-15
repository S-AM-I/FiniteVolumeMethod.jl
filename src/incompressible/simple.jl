# incompressible/simple.jl — SIMPLE steady-state solver loop
#
# Implements the Semi-Implicit Method for Pressure-Linked Equations
# (SIMPLE) algorithm for steady-state incompressible Navier-Stokes.
# Each outer iteration assembles momentum equations with under-relaxation,
# solves the pressure Poisson equation, and applies velocity/flux corrections.

using Printf: @sprintf

# ── SIMPLE solver ──────────────────────────────────────────────────

@doc """
    solve_simple(prob::IncompressibleProblem{Dim, T}; kwargs...) -> SolveResult{Dim, T}

Solve a steady-state incompressible Navier-Stokes problem using the
[`SIMPLE`](@ref) pressure-velocity coupling algorithm.

The algorithm iterates:
1. Assemble and solve momentum equations (one per spatial dimension)
   with under-relaxation.
2. Assemble and solve the pressure Poisson equation.
3. Under-relax the pressure field.
4. Correct cell velocities and face fluxes from the new pressure.
5. Check convergence via momentum and continuity residuals.

# Arguments
- `prob` — [`IncompressibleProblem`](@ref) with a [`SIMPLE`](@ref) algorithm

# Keyword Arguments
- `linear_solver` — solver algorithm for `LinearProblem` (default: `nothing` → backslash)
- `verbose::Bool` — print residuals each iteration (default: `false`)

# Returns
A [`SolveResult`](@ref) containing convergence status, iteration count,
residual history, and the final [`IncompressibleState`](@ref).
"""
function solve_simple(
        prob::IncompressibleProblem{Dim, T};
        linear_solver = nothing,
        solver_config = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    algo = prob.algorithm::SIMPLE{T}
    mesh = prob.mesh
    alpha_U = algo.alpha_U
    alpha_p = algo.alpha_p
    max_iter = algo.max_iterations
    tol = algo.tolerance

    # Initialize state and apply boundary conditions
    state = IncompressibleState(mesh)
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_pressure!(state, prob.bcs, mesh)

    # Pre-compute cyclic face pairs (empty vector if no CyclicBC)
    cyclic_pairs = collect_cyclic_pairs(prob.bcs, mesh)
    cell_pairs = _cyclic_cell_pairs(mesh, cyclic_pairs)

    # Residual history
    component_labels = _velocity_labels(Val(Dim))
    residuals = Dict{Symbol, Vector{T}}(
        label => T[] for label in [component_labels..., :continuity]
    )

    converged = false
    final_iter = 0

    # Allocate equations once; reset! + reassemble each iteration.
    eqs = [CollocatedEquation(mesh; extra_cell_pairs = cell_pairs) for _ in 1:Dim]
    p_eq = CollocatedEquation(mesh; extra_cell_pairs = cell_pairs)

    for iter in 1:max_iter
        final_iter = iter

        # ── 1. Assemble momentum equations ──────────────────────────
        for d in 1:Dim
            reset!(eqs[d])
            assemble_momentum!(eqs[d], state, prob, d)
            # Apply cyclic coupling to momentum
            apply_cyclic_to_equation!(
                eqs[d], _make_scalar_field(_extract_component(state.U, d), state),
                mesh, cyclic_pairs,
            )
        end

        # ── 2. Under-relax + solve momentum ─────────────────────────
        for d in 1:Dim
            U_old_d = _extract_component(state.U, d)
            under_relax_momentum!(eqs[d], U_old_d, alpha_U)
            lp = to_linear_problem(eqs[d])
            sol = _dispatch_solve(
                lp, linear_solver, solver_config,
                d == 1 ? :Ux : (d == 2 ? :Uy : :Uz),
            )
            _set_component!(state.U, d, sol.u)
        end

        # ── 3. Update boundary velocity ─────────────────────────────
        update_boundary_velocity!(state, prob.bcs, mesh)

        # ── 4. Extract operators (A_P, H_U) from the RELAXED, solved
        # momentum equations — standard SIMPLE ordering, so that
        # D = V/A_P in the pressure equation is consistent with the
        # velocity actually produced by the momentum solve.
        extract_momentum_operators!(state, eqs, mesh)

        # ── 5. Assemble + solve pressure ────────────────────────────
        reset!(p_eq)
        assemble_pressure!(p_eq, state, prob)
        apply_cyclic_to_equation!(p_eq, state.p, mesh, cyclic_pairs)
        if _needs_pressure_reference(prob.bcs)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end
        lp_p = to_linear_problem(p_eq)
        p_sol = _dispatch_solve(lp_p, linear_solver, solver_config, :p)

        # ── 6. Under-relax pressure ─────────────────────────────────
        nc = length(mesh.cell_volumes)
        for c in 1:nc
            state.p.internal[c] += alpha_p * (p_sol.u[c] - state.p.internal[c])
        end

        # ── 7. Update boundary pressure ─────────────────────────────
        update_boundary_pressure!(state, prob.bcs, mesh)

        # ── 8. Correct velocity + fluxes ────────────────────────────
        correct_velocity!(state, mesh)
        update_boundary_velocity!(state, prob.bcs, mesh)
        update_boundary_cyclic!(state, mesh, cyclic_pairs)
        correct_fluxes!(state, mesh)

        # ── 9. Compute residuals + check convergence ────────────────
        max_residual = zero(T)
        for d in 1:Dim
            u_d = _extract_component(state.U, d)
            r = momentum_residual(eqs[d], u_d)
            push!(residuals[component_labels[d]], r)
            max_residual = max(max_residual, r)
        end
        r_cont = continuity_residual(state, mesh)
        push!(residuals[:continuity], r_cont)
        max_residual = max(max_residual, r_cont)

        if verbose
            _print_simple_residuals(iter, residuals, component_labels)
        end

        if max_residual < tol
            converged = true
            break
        end
    end

    return SolveResult{Dim, T}(converged, final_iter, residuals, state)
end

# ── Internal helpers ───────────────────────────────────────────────

"""
    _velocity_labels(::Val{Dim}) -> NTuple{Dim, Symbol}

Return canonical residual label symbols for each velocity component.
"""
_velocity_labels(::Val{2}) = (:Ux, :Uy)
_velocity_labels(::Val{3}) = (:Ux, :Uy, :Uz)

"""
    _solve_linear(lp, linear_solver)

Solve a `LinearProblem`, dispatching to the given solver or the default.
"""
function _solve_linear(lp, linear_solver)
    if linear_solver === nothing
        return solve(lp)
    else
        return solve(lp, linear_solver)
    end
end

"""
    _print_simple_residuals(iter, residuals, labels)

Print a one-line summary of the current iteration's residuals.
"""
function _print_simple_residuals(iter, residuals, labels)
    parts = String[]
    for label in labels
        r = residuals[label][end]
        push!(parts, string(label, "=", @sprintf("%.3e", r)))
    end
    r_cont = residuals[:continuity][end]
    push!(parts, string("cont=", @sprintf("%.3e", r_cont)))
    println("SIMPLE iter ", lpad(iter, 5), ": ", join(parts, "  "))
    return nothing
end
