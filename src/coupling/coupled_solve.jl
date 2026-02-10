# ============================================================
# Coupled Solver via Operator Splitting
# ============================================================
#
# Time integration loop for coupled multi-physics problems.
# At each time step:
#   1. Compute dt = min over all operator CFL constraints.
#   2. Apply the splitting scheme to advance all operators.

"""
    solve_coupled(prob::HyperbolicProblem, source::AbstractStiffSource; kwargs...)

Solve a 1D hyperbolic problem with stiff source terms using operator splitting.

# Keyword Arguments
- `splitting::AbstractSplittingScheme`: Splitting method (default: `StrangSplitting()`).
- `newton_tol::Real`: Tolerance for implicit Newton solves (default: `1e-10`).
- `newton_maxiter::Int`: Maximum Newton iterations (default: `10`).

# Returns
- `x::Vector`: Cell center coordinates.
- `U_final::Vector{SVector{N}}`: Final conserved variables at cell centers.
- `t_final::Real`: Final time reached.
"""
function solve_coupled(
        prob::HyperbolicProblem, source::AbstractStiffSource;
        splitting::AbstractSplittingScheme = StrangSplitting(),
        newton_tol = 1.0e-10, newton_maxiter = 10
    )
    hyp_op = HyperbolicOperator(prob)
    src_op = SourceOperator(prob.law, source; newton_tol, newton_maxiter)
    coupled = CoupledProblem(
        (hyp_op, src_op), splitting,
        prob.initial_time, prob.final_time
    )
    return _solve_coupled_1d(coupled)
end

"""
    solve_coupled(prob::HyperbolicProblem2D, source::AbstractStiffSource; kwargs...)

Solve a 2D hyperbolic problem with stiff source terms using operator splitting.

# Returns
- `coords`: Cell center coordinates.
- `U_final::Matrix{SVector{N}}`: Final conserved variables.
- `t_final::Real`: Final time reached.
"""
function solve_coupled(
        prob::HyperbolicProblem2D, source::AbstractStiffSource;
        splitting::AbstractSplittingScheme = StrangSplitting(),
        newton_tol = 1.0e-10, newton_maxiter = 10
    )
    hyp_op = HyperbolicOperator(prob)
    src_op = SourceOperator(prob.law, source; newton_tol, newton_maxiter)
    coupled = CoupledProblem(
        (hyp_op, src_op), splitting,
        prob.initial_time, prob.final_time
    )
    return _solve_coupled_2d(coupled)
end

"""
    solve_coupled(coupled::CoupledProblem)

Solve a general coupled problem using operator splitting.
Dispatches to 1D or 2D solver based on the HyperbolicOperator type.
"""
function solve_coupled(coupled::CoupledProblem)
    for op in coupled.operators
        if op isa HyperbolicOperator{<:HyperbolicProblem}
            return _solve_coupled_1d(coupled)
        elseif op isa HyperbolicOperator{<:HyperbolicProblem2D}
            return _solve_coupled_2d(coupled)
        end
    end
    error("CoupledProblem must contain at least one HyperbolicOperator")
end

# ============================================================
# 1D Solver
# ============================================================

function _solve_coupled_1d(coupled::CoupledProblem)
    hyp_op = _find_hyperbolic_op(coupled.operators, HyperbolicProblem)
    prob = hyp_op.problem
    mesh = prob.mesh
    nc = ncells(mesh)
    N = nvariables(prob.law)

    # Initialize state from the HyperbolicProblem's initial condition
    U = initialize_1d(prob)
    FT = eltype(U[3])

    # Pre-allocate workspace
    zero_state = zero(SVector{N, FT})
    dU = similar(U)
    U1 = similar(U)
    U2 = similar(U)
    for i in eachindex(dU)
        dU[i] = zero_state
        U1[i] = zero_state
        U2[i] = zero_state
    end
    workspace = (; dU_vec = dU, U1_vec = U1, U2_vec = U2, nc = nc)

    t = coupled.initial_time

    while t < coupled.final_time - eps(typeof(t))
        # Compute dt as minimum over all operators
        dt = typemax(FT)
        for op in coupled.operators
            dt_op = compute_operator_dt(op, U, t)
            dt = min(dt, dt_op)
        end

        # Don't overshoot final time
        if t + dt > coupled.final_time
            dt = coupled.final_time - t
        end
        if dt <= zero(dt)
            break
        end

        # Apply splitting scheme
        _apply_splitting!(U, coupled.operators, coupled.splitting, dt, t, workspace)

        t += dt
    end

    # Extract interior solution
    x = [cell_center(mesh, i) for i in 1:nc]
    U_interior = U[3:(nc + 2)]

    return x, U_interior, t
end

# ============================================================
# 2D Solver
# ============================================================

function _solve_coupled_2d(coupled::CoupledProblem)
    hyp_op = _find_hyperbolic_op(coupled.operators, HyperbolicProblem2D)
    prob = hyp_op.problem
    mesh = prob.mesh
    nx, ny = mesh.nx, mesh.ny
    N = nvariables(prob.law)

    # Initialize state
    U = initialize_2d(prob)
    FT = eltype(U[3, 3])

    # Pre-allocate workspace
    zero_state = zero(SVector{N, FT})
    dU = similar(U)
    U1 = similar(U)
    U2 = similar(U)
    for j in axes(dU, 2), i in axes(dU, 1)
        dU[i, j] = zero_state
        U1[i, j] = zero_state
        U2[i, j] = zero_state
    end
    workspace = (; dU_mat = dU, U1_mat = U1, U2_mat = U2, nx = nx, ny = ny)

    t = coupled.initial_time

    while t < coupled.final_time - eps(typeof(t))
        dt = typemax(FT)
        for op in coupled.operators
            dt_op = compute_operator_dt(op, U, t)
            dt = min(dt, dt_op)
        end

        if t + dt > coupled.final_time
            dt = coupled.final_time - t
        end
        if dt <= zero(dt)
            break
        end

        _apply_splitting!(U, coupled.operators, coupled.splitting, dt, t, workspace)

        t += dt
    end

    # Extract interior solution
    U_interior = Matrix{SVector{N, FT}}(undef, nx, ny)
    for iy in 1:ny, ix in 1:nx
        U_interior[ix, iy] = U[ix + 2, iy + 2]
    end
    coords = [(cell_center(mesh, cell_idx(mesh, ix, iy))) for ix in 1:nx, iy in 1:ny]

    return coords, U_interior, t
end

# ============================================================
# Splitting Step Implementations
# ============================================================

"""
    _apply_splitting!(U, operators, ::LieTrotterSplitting, dt, t, workspace)

Apply first-order Lie-Trotter splitting: each operator gets a full time step.
"""
function _apply_splitting!(U, operators::Tuple, ::LieTrotterSplitting, dt, t, workspace)
    for op in operators
        advance!(U, op, dt, t, workspace)
    end
    return nothing
end

"""
    _apply_splitting!(U, operators, ::StrangSplitting, dt, t, workspace)

Apply second-order Strang splitting: symmetric forward-backward sweep.
For two operators: L₁(dt/2) ∘ L₂(dt) ∘ L₁(dt/2).
"""
function _apply_splitting!(U, operators::Tuple, ::StrangSplitting, dt, t, workspace)
    n = length(operators)
    if n == 1
        advance!(U, operators[1], dt, t, workspace)
        return nothing
    end

    # Forward sweep: operators 1 to n-1 with dt/2
    for i in 1:(n - 1)
        advance!(U, operators[i], dt / 2, t, workspace)
    end

    # Middle: operator n with full dt
    advance!(U, operators[n], dt, t, workspace)

    # Backward sweep: operators n-1 to 1 with dt/2
    for i in (n - 1):-1:1
        advance!(U, operators[i], dt / 2, t, workspace)
    end

    return nothing
end

# ============================================================
# Helpers
# ============================================================

"""
    _find_hyperbolic_op(operators, ProbType) -> HyperbolicOperator

Find the first HyperbolicOperator whose problem matches `ProbType`.
"""
function _find_hyperbolic_op(operators::Tuple, ::Type{P}) where {P}
    for op in operators
        if op isa HyperbolicOperator && op.problem isa P
            return op
        end
    end
    error("No HyperbolicOperator with problem type $P found in operators")
end
