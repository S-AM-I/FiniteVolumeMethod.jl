# solvers.jl - Steady-state, transient, and adaptive solvers
# Migrated from Simu.jl SimuEngine/solvers.jl
# LinearAlgebra is already imported by the parent module.

"""
    solve_steady_state(A, b; method=:direct, maxiter=1000, tol=1e-10, verbose=false)

Solve the steady-state problem Ax = b.
method can be :direct (default), :gmres, or :bicgstab.
"""
function solve_steady_state(A, b; method = :direct, maxiter = 1000, tol = 1.0e-10, verbose = false)
    return try
        if method == :direct
            # Direct solver
            x = A \ b
            if !all(isfinite.(x))
                error("Solution contains non-finite values. Matrix may be singular or ill-conditioned.")
            end
            return x
        elseif method == :gmres
            return solve_steady_state_gmres(A, b; maxiter = maxiter, tol = tol, verbose = verbose)
        elseif method == :bicgstab
            return solve_steady_state_bicgstab(A, b; maxiter = maxiter, tol = tol, verbose = verbose)
        else
            error("Unknown method: $method. Use :direct, :gmres, or :bicgstab.")
        end
    catch e
        if isa(e, SingularException)
            error("Matrix is singular. Check boundary conditions and source terms.")
        elseif isa(e, PosDefException)
            error("Matrix is not positive definite. Check problem setup.")
        else
            rethrow(e)
        end
    end
end

"""
    solve_steady_state_gmres(A, b; maxiter=1000, tol=1e-10, verbose=false)

Solve Ax = b using GMRES iterative method.
"""
function solve_steady_state_gmres(A, b; maxiter = 1000, tol = 1.0e-10, verbose = false)
    x = zeros(size(b))
    r = b - A * x
    r_norm = norm(r)
    initial_norm = r_norm

    if initial_norm < tol
        return x
    end

    for iter in 1:maxiter
        # Simplified GMRES via iterative refinement
        dx = A \ r
        x = x + dx
        r = b - A * x
        r_norm = norm(r)

        if verbose && iter % 100 == 0
            println("GMRES iteration $iter: residual = $r_norm")
        end

        if r_norm < tol * initial_norm
            if verbose
                println("GMRES converged in $iter iterations")
            end
            return x
        end
    end

    if verbose
        println("GMRES did not converge after $maxiter iterations. Final residual: $r_norm")
    end
    return x
end

"""
    solve_steady_state_bicgstab(A, b; maxiter=1000, tol=1e-10, verbose=false)

Solve Ax = b using BiCGStab iterative method.
"""
function solve_steady_state_bicgstab(A, b; maxiter = 1000, tol = 1.0e-10, verbose = false)
    x = zeros(size(b))
    r = b - A * x
    r0 = copy(r)
    rho = 1.0
    alpha = 1.0
    omega = 1.0
    v = zeros(size(b))
    p = zeros(size(b))

    initial_norm = norm(r)
    if initial_norm < tol
        return x
    end

    for iter in 1:maxiter
        rho_prev = rho
        rho = dot(r0, r)

        if abs(rho) < 1.0e-15
            if verbose
                println("BiCGStab breakdown at iteration $iter")
            end
            break
        end

        beta = (rho / rho_prev) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = A * p

        denom = dot(r0, v)
        if abs(denom) < 1.0e-20
            if verbose
                println("BiCGStab breakdown: dot(r0, v) too small")
            end
            break
        end

        alpha = rho / denom
        s = r - alpha * v
        t = A * s

        tt = dot(t, t)
        if tt < 1.0e-20
            # t is zero, s is zero?
            x = x + alpha * p
            if verbose
                println("BiCGStab converged at half-step")
            end
            return x
        end

        omega = dot(t, s) / tt
        x = x + alpha * p + omega * s
        r = s - omega * t

        r_norm = norm(r)
        if verbose && iter % 100 == 0
            println("BiCGStab iteration $iter: residual = $r_norm")
        end

        if r_norm < tol * initial_norm
            if verbose
                println("BiCGStab converged in $iter iterations")
            end
            return x
        end
    end

    if verbose
        println("BiCGStab did not converge after $maxiter iterations.")
    end
    return x
end

# --- Transient Solvers ---

"""
    solve_transient(x0, A, M, b, dt, t_final)

Solve the transient problem M*dx/dt + Ax = b using explicit Euler.
"""
function solve_transient(x0, A, M, b, dt, t_final)
    x = copy(x0)
    t = 0.0

    while t < t_final
        x = x + dt * (M \ (b - A * x))
        t += dt
    end

    return x
end

"""
    solve_transient_rk2(x0, A, M, b, dt, t_final)

Solve the transient problem M*dx/dt + Ax = b using second-order Runge-Kutta.
"""
function solve_transient_rk2(x0, A, M, b, dt, t_final)
    x = copy(x0)
    t = 0.0

    while t < t_final
        k1 = dt * (M \ (b - A * x))
        k2 = dt * (M \ (b - A * (x + 0.5 * k1)))
        x = x + k2
        t += dt
    end

    return x
end

"""
    solve_transient_crank_nicolson(x0, A, M, b, dt, t_final)

Solve the transient problem M*dx/dt + Ax = b using Crank-Nicolson.
"""
function solve_transient_crank_nicolson(x0, A, M, b, dt, t_final)
    x = copy(x0)
    t = 0.0

    LHS = M + dt / 2 * A

    while t < t_final
        RHS = (M - dt / 2 * A) * x + dt * b
        x = LHS \ RHS
        t += dt
    end

    return x
end

# --- Numerical Jacobian ---

"""
    compute_numerical_jacobian!(J, f!, x, t; epsilon=1e-8)

Compute Jacobian J via finite differences.
`f!` has signature `f!(fx, x, t)`.
"""
function compute_numerical_jacobian!(J, f!, x, t; epsilon = 1.0e-8)
    n = length(x)
    fx = zeros(n)
    f!(fx, x, t)

    x_perturb = copy(x)
    f_perturb = zeros(n)

    for i in 1:n
        original_val = x[i]
        h = max(epsilon * abs(original_val), epsilon)
        x_perturb[i] += h

        f!(f_perturb, x_perturb, t)

        @. J[:, i] = (f_perturb - fx) / h

        x_perturb[i] = original_val
    end
    return
end

# --- Adaptive Solver ---

"""
    solve_adaptive(f!, u0, t_span; stepper, tol, controller, update_jacobian!, post_step_callback)

Adaptive time-stepping solver for nonlinear systems M du/dt = f(u,t).
Currently assumes M=I.
"""
function solve_adaptive(
        f!, u0, t_span;
        stepper = ImplicitEuler(),
        tol = 1.0e-4,
        controller = nothing,
        update_jacobian! = nothing,
        post_step_callback = nothing
    )

    t_start, t_end = t_span

    if controller === nothing
        controller = TimeController(t_start, 1.0e-4, t_end)
    end

    u = copy(u0)
    history = []
    push!(history, (t_start, copy(u)))

    n = length(u0)
    J = zeros(n, n)

    while controller.t < controller.t_end
        dt = controller.dt

        if controller.t + dt > controller.t_end
            dt = controller.t_end - controller.t
        end

        try
            # Step 1: Full step
            u_full = step_nonlinear(stepper, f!, u, dt, controller.t, J, update_jacobian!)

            # Step 2: Half steps
            u_half = step_nonlinear(stepper, f!, u, dt / 2, controller.t, J, update_jacobian!)
            u_half = step_nonlinear(stepper, f!, u_half, dt / 2, controller.t + dt / 2, J, update_jacobian!)

            # Error Estimate (Richardson)
            p = 1
            if stepper isa CrankNicolson
                p = 2
            end

            err_norm = norm(u_full - u_half)
            err = err_norm / (2^p - 1)

            scaled_err = err / tol

            dt_next = propose_step(controller, scaled_err)

            if scaled_err <= 1.0
                # Accept
                u = u_half
                accept_step!(controller, dt)

                if post_step_callback !== nothing
                    post_step_callback(u, controller.t, dt)
                end

                controller.dt = dt_next
                push!(history, (controller.t, copy(u)))
            else
                # Reject
                controller.dt = max(dt_next, controller.dt_min)
            end

        catch e
            println("Step failed at t=$(controller.t): ", e)
            controller.dt *= 0.5
            if controller.dt < controller.dt_min
                error("Time step too small")
            end
        end
    end

    return history
end

"""
    step_nonlinear(stepper::ImplicitEuler, f!, x_old, dt, t_old, J, update_jacobian!)

Take one implicit Euler step for a nonlinear system using Newton-Raphson.
"""
function step_nonlinear(stepper::ImplicitEuler, f!, x_old, dt, t_old, J, update_jacobian!)
    t_new = t_old + dt

    function residual(x_new)
        fx = zeros(length(x_new))
        f!(fx, x_new, t_new)
        return x_new - x_old - dt * fx
    end

    function jacobian(x_new)
        if update_jacobian! !== nothing
            update_jacobian!(J, x_new, t_new)
        else
            compute_numerical_jacobian!(J, f!, x_new, t_new)
        end
        return I - dt * J
    end

    x_guess = x_old
    x_new, converged, _ = newton_raphson(residual, jacobian, x_guess)

    if !converged
        error("Newton convergence failed")
    end

    return x_new
end

"""
    step_nonlinear(stepper::CrankNicolson, f!, x_old, dt, t_old, J, update_jacobian!)

Take one Crank-Nicolson step for a nonlinear system using Newton-Raphson.
"""
function step_nonlinear(stepper::CrankNicolson, f!, x_old, dt, t_old, J, update_jacobian!)
    t_new = t_old + dt

    fx_old = zeros(length(x_old))
    f!(fx_old, x_old, t_old)

    function residual(x_new)
        fx_new = zeros(length(x_new))
        f!(fx_new, x_new, t_new)
        return x_new - x_old - 0.5 * dt * (fx_new + fx_old)
    end

    function jacobian(x_new)
        if update_jacobian! !== nothing
            update_jacobian!(J, x_new, t_new)
        else
            compute_numerical_jacobian!(J, f!, x_new, t_new)
        end
        return I - 0.5 * dt * J
    end

    x_guess = x_old + dt * fx_old
    x_new, converged, _ = newton_raphson(residual, jacobian, x_guess)

    if !converged
        error("Newton convergence failed")
    end

    return x_new
end
