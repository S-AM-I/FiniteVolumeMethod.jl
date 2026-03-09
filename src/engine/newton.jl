# newton.jl - Newton-Raphson and related nonlinear solvers
# Migrated from Simu.jl SimuEngine/newton.jl
# LinearAlgebra is already imported by the parent module.

"""
    newton_raphson(f, J, x0; tol=1e-8, max_iter=20)

Solve f(x) = 0 using Newton-Raphson method.
x_{n+1} = x_n - J(x_n)^{-1} * f(x_n)

# Arguments
- `f`: Function `f(x)` returning the residual vector.
- `J`: Function `J(x)` returning the Jacobian matrix.
- `x0`: Initial guess.
- `tol`: Convergence tolerance (L2 norm of residual).
- `max_iter`: Maximum number of iterations.

# Returns
- `x`: Solution vector.
- `converged`: Boolean indicating success.
- `iters`: Number of iterations performed.
"""
function newton_raphson(f, J, x0; tol = 1.0e-8, max_iter = 20)
    x = copy(x0)
    for i in 1:max_iter
        res = f(x)
        err = norm(res)

        if err < tol
            return x, true, i
        end

        # J(x) * dx = -res
        jac = J(x)
        dx = jac \ (-res)

        x .+= dx
    end

    return x, false, max_iter
end

# --- JFNK ---

"""
    JacobianOperator{F, T}

Matrix-free Jacobian-vector product operator for Jacobian-Free Newton-Krylov (JFNK).
Computes `J*v approx (F(x + eps*v) - F(x)) / eps` via finite differences.
"""
struct JacobianOperator{F, T}
    f::F
    x::Vector{T}
    Fx::Vector{T}
    epsilon::T
end

function Base.:*(J::JacobianOperator, v::Vector)
    x_perturb = J.x + J.epsilon * v
    F_perturb = J.f(x_perturb)
    return (F_perturb - J.Fx) / J.epsilon
end

function Base.size(J::JacobianOperator, dim::Integer)
    return length(J.x)
end
function Base.size(J::JacobianOperator)
    n = length(J.x)
    return (n, n)
end
function Base.eltype(J::JacobianOperator)
    return eltype(J.x)
end

"""
    newton_krylov(f, x0, linear_solver; tol=1e-8, max_iter=20, epsilon=1e-6)

Solve f(x) = 0 using Jacobian-Free Newton-Krylov method.

# Arguments
- `f`: Function `f(x)` returning the residual vector.
- `x0`: Initial guess.
- `linear_solver`: Function `(A, b) -> x` that solves Ax=b using a Krylov method.
- `tol`: Convergence tolerance.
- `max_iter`: Maximum Newton iterations.
- `epsilon`: Finite-difference step for Jacobian-vector products.

# Returns
- `x`: Solution vector.
- `converged`: Boolean indicating success.
- `iters`: Number of iterations performed.
"""
function newton_krylov(f, x0, linear_solver; tol = 1.0e-8, max_iter = 20, epsilon = 1.0e-6)
    x = copy(x0)
    for i in 1:max_iter
        res = f(x)
        err = norm(res)

        if err < tol
            return x, true, i
        end

        # J*dx = -res via matrix-free operator
        J_op = JacobianOperator(f, x, res, epsilon)

        dx = linear_solver(J_op, -res)

        x .+= dx
    end

    return x, false, max_iter
end

# --- Anderson Acceleration ---

"""
    anderson_acceleration(g, x0; m=5, tol=1e-8, max_iter=100)

Find fixed point x = g(x) using Anderson Acceleration.

# Arguments
- `g`: Fixed-point iteration function.
- `x0`: Initial guess.
- `m`: History depth (mixing window size).
- `tol`: Convergence tolerance.
- `max_iter`: Maximum iterations.

# Returns
- `x`: Solution vector.
- `converged`: Boolean indicating success.
- `iters`: Number of iterations performed.
"""
function anderson_acceleration(g, x0; m = 5, tol = 1.0e-8, max_iter = 100)
    x = copy(x0)
    n = length(x)

    # History of iterates and g-evaluations
    X_hist = Vector{Vector{Float64}}()
    G_hist = Vector{Vector{Float64}}()

    push!(X_hist, x)
    gx = g(x)
    push!(G_hist, gx)

    res = gx - x
    if norm(res) < tol
        return x, true, 0
    end

    x = gx # First step is just Picard

    for k in 1:max_iter
        push!(X_hist, x)
        gx = g(x)
        push!(G_hist, gx)

        # Maintain history size m+1 (m differences)
        if length(X_hist) > m + 1
            popfirst!(X_hist)
            popfirst!(G_hist)
        end

        res = gx - x
        if norm(res) < tol
            return x, true, k
        end

        mk = length(X_hist) - 1
        if mk == 0
            x = gx
            continue
        end

        # Anderson mixing: minimize || f_k - Df * gamma ||
        f_k = G_hist[end] - X_hist[end]

        # Matrix of differences of residuals
        Df = zeros(n, mk)
        for j in 1:mk
            f_prev = G_hist[end - j] - X_hist[end - j]
            Df[:, j] = f_k - f_prev
        end

        # Least squares: Df * gamma = f_k
        gammas = Df \ f_k

        # Correction term
        correction = zeros(n)
        for j in 1:mk
            g_diff = G_hist[end] - G_hist[end - j]
            correction += gammas[j] * g_diff
        end

        x = gx - correction
    end

    return x, false, max_iter
end
