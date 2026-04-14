# incompressible/piso.jl — PISO transient step function
#
# Implements the Pressure Implicit with Splitting of Operators (PISO)
# algorithm as a single-time-step advancement function.  PISO is a
# non-iterative transient scheme with one momentum predictor followed
# by multiple pressure corrector steps.

# ── PISO step ──────────────────────────────────────────────────────

@doc """
    _piso_step!(state, prob, dt, n_correctors; linear_solver = nothing)

Advance the incompressible solver state by one time step `dt` using the
[`PISO`](@ref) algorithm.

The algorithm proceeds as:
1. **Momentum predictor**: Assemble the transient momentum equations
   (with `dt`) and solve for an intermediate velocity — no under-relaxation.
2. **Pressure corrector loop** (`n_correctors` passes):
   a. Assemble and solve the pressure Poisson equation.
   b. Assign the new pressure to `state.p`.
   c. Correct cell velocities, update boundary conditions, and
      recompute Rhie-Chow face fluxes.
   d. If more corrector steps remain, re-extract the momentum operators
      from a fresh momentum assembly so that the next pressure equation
      uses updated `A_P` and `H(U)`.

# Arguments
- `state::IncompressibleState{Dim, T}` — solver state (modified in-place)
- `prob::IncompressibleProblem{Dim, T}` — problem definition
- `dt::T` — time step size
- `n_correctors::Int` — number of pressure correction steps (typically 2)

# Keyword Arguments
- `linear_solver` — solver algorithm for `LinearProblem` (default: `nothing`)
"""
function _piso_step!(
        state::IncompressibleState{Dim, T},
        prob::IncompressibleProblem{Dim, T},
        dt::T,
        n_correctors::Int;
        linear_solver = nothing,
        solver_config = nothing,
        cyclic_pairs::Vector{Vector{Tuple{Int, Int}}} = Vector{Vector{Tuple{Int, Int}}}(),
    ) where {Dim, T}
    mesh = prob.mesh

    # ── 1. Momentum predictor (no under-relaxation) ─────────────────
    eqs = CollocatedEquation{T}[]
    for d in 1:Dim
        eq = CollocatedEquation(mesh)
        assemble_momentum!(eq, state, prob, d; dt = dt)
        # Apply cyclic coupling to momentum
        apply_cyclic_to_equation!(
            eq, _make_scalar_field(_extract_component(state.U, d), state),
            mesh, cyclic_pairs,
        )
        push!(eqs, eq)
    end

    extract_momentum_operators!(state, eqs, mesh)

    for d in 1:Dim
        lp = to_linear_problem(eqs[d])
        sol = _dispatch_solve(
            lp, linear_solver, solver_config,
            d == 1 ? :Ux : (d == 2 ? :Uy : :Uz),
        )
        _set_component!(state.U, d, sol.u)
    end
    update_boundary_velocity!(state, prob.bcs, mesh)
    update_boundary_cyclic!(state, mesh, cyclic_pairs)

    # ── 2. Pressure corrector loop ──────────────────────────────────
    for k in 1:n_correctors
        # 2a. Assemble + solve pressure
        p_eq = CollocatedEquation(mesh)
        assemble_pressure!(p_eq, state, prob)
        apply_cyclic_to_equation!(p_eq, state.p, mesh, cyclic_pairs)
        if _needs_pressure_reference(prob.bcs)
            fix_pressure_reference!(p_eq, 1, zero(T))
        end
        lp_p = to_linear_problem(p_eq)
        p_sol = _dispatch_solve(lp_p, linear_solver, solver_config, :p)

        # 2b. Direct assign (no under-relaxation in PISO)
        nc = length(mesh.cell_volumes)
        for c in 1:nc
            state.p.internal[c] = p_sol.u[c]
        end

        # 2c. Update boundary pressure
        update_boundary_pressure!(state, prob.bcs, mesh)

        # 2d. Correct velocity + fluxes
        correct_velocity!(state, mesh)
        update_boundary_velocity!(state, prob.bcs, mesh)
        update_boundary_cyclic!(state, mesh, cyclic_pairs)
        correct_fluxes!(state, mesh)

        # 2e. Re-assemble momentum + extract operators for next corrector
        if k < n_correctors
            eqs_k = CollocatedEquation{T}[]
            for d in 1:Dim
                eq = CollocatedEquation(mesh)
                assemble_momentum!(eq, state, prob, d; dt = dt)
                apply_cyclic_to_equation!(
                    eq, _make_scalar_field(_extract_component(state.U, d), state),
                    mesh, cyclic_pairs,
                )
                push!(eqs_k, eq)
            end
            extract_momentum_operators!(state, eqs_k, mesh)
        end
    end

    return nothing
end
