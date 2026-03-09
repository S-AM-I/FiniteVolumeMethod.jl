using Test
using FiniteVolumeMethod
using LinearAlgebra
using SparseArrays

@testset "Engine" begin
    @testset "TimeController" begin
        tc = TimeController(0.0, 0.01, 1.0)
        @test tc.t ≈ 0.0
        @test tc.dt ≈ 0.01
        @test tc.t_end ≈ 1.0

        # Accept a step
        accept_step!(tc, 0.01)
        @test tc.t ≈ 0.01
        @test length(tc.history.t) == 1
        @test tc.history.accepted[1] == true

        # Propose step with adaptivity
        dt_new = propose_step(tc, 0.5) # error < 1 => step should grow
        @test dt_new > 0.0
        @test dt_new >= tc.dt_min
    end

    @testset "TimeGrid" begin
        tg = TimeGrid([0.0, 0.5, 1.0])
        @test length(tg.times) == 3
        @test tg.times[2] ≈ 0.5
    end

    @testset "TimeStepHistory" begin
        h = TimeStepHistory(Float64[])
        @test isempty(h.t)
        @test isempty(h.dt)
        @test isempty(h.accepted)
    end

    @testset "Simulation struct" begin
        sim = Simulation(:prob, :alg, :sol, :cb, :diag)
        @test sim.problem == :prob
        @test sim.algorithm == :alg
    end

    @testset "EventType enum" begin
        @test ROOT_EVENT isa EventType
        @test TIME_EVENT isa EventType
    end

    @testset "Newton-Raphson" begin
        # Solve x^2 - 4 = 0
        f(x) = [x[1]^2 - 4.0]
        J(x) = reshape([2.0 * x[1]], 1, 1)
        x0 = [1.0]
        x_sol, converged, iters = newton_raphson(f, J, x0)
        @test converged
        @test abs(x_sol[1]) ≈ 2.0 atol = 1.0e-8

        # Solve a 2D system: x^2 + y^2 = 5, x + y = 3
        f2(x) = [x[1]^2 + x[2]^2 - 5.0, x[1] + x[2] - 3.0]
        J2(x) = [2.0 * x[1] 2.0 * x[2]; 1.0 1.0]
        x0_2 = [2.5, 0.5]
        x_sol2, converged2, _ = newton_raphson(f2, J2, x0_2)
        @test converged2
        @test x_sol2[1] + x_sol2[2] ≈ 3.0 atol = 1.0e-8
    end

    @testset "Newton-Krylov" begin
        # Solve x^2 - 4 = 0 using JFNK
        f(x) = [x[1]^2 - 4.0]
        # Simple Richardson-iteration linear solver that uses J*v product
        function linsolve(J_op, b)
            # For a 1x1 system, approximate: dx = b / (J*e1)
            e1 = [1.0]
            Jv = J_op * e1
            return b ./ Jv
        end
        x0 = [3.0]
        x_sol, converged, _ = newton_krylov(f, x0, linsolve)
        @test converged
        @test abs(x_sol[1]) ≈ 2.0 atol = 1.0e-6
    end

    @testset "Anderson Acceleration" begin
        # Find fixed point of g(x) = sqrt(2 + x) near x ≈ 2
        g(x) = [sqrt(2.0 + x[1])]
        x0 = [1.0]
        x_sol, converged, _ = anderson_acceleration(g, x0; m = 3, tol = 1.0e-10)
        @test converged
        # Fixed point: x = sqrt(2+x) => x^2 = 2+x => x^2 - x - 2 = 0 => x = 2
        @test abs(x_sol[1] - 2.0) < 1.0e-8
    end

    @testset "Time Steppers" begin
        # Simple 1D decay: du/dt = -u, M=I, A=I, b=0
        # Exact: u(dt) = u0 * exp(-dt)
        n = 3
        A = Matrix(1.0I, n, n)
        M = Matrix(1.0I, n, n)
        b = zeros(n)
        x0 = ones(n)
        dt = 0.01

        # Forward Euler
        x_fe = step!(ForwardEuler(), x0, A, M, b, dt)
        @test all(x_fe .≈ 0.99)  # 1 - 0.01 * 1 = 0.99

        # RK2
        x_rk2 = step!(RK2(), x0, A, M, b, dt)
        @test all(isapprox.(x_rk2, 1.0 - dt + 0.5 * dt^2; atol = 1.0e-12))

        # Implicit Euler
        x_ie = step!(ImplicitEuler(), x0, A, M, b, dt)
        # (I + dt*I) x_new = I*x0 => x_new = x0/(1+dt)
        @test all(isapprox.(x_ie, 1.0 / (1.0 + dt); atol = 1.0e-12))

        # Crank-Nicolson
        x_cn = step!(CrankNicolson(), x0, A, M, b, dt)
        # (I + 0.5dt*I) x_new = (I - 0.5dt*I) x0 => x_new = (1-0.5dt)/(1+0.5dt)
        @test all(isapprox.(x_cn, (1.0 - 0.5 * dt) / (1.0 + 0.5 * dt); atol = 1.0e-12))

        # Rosenbrock23 (currently delegates to ImplicitEuler)
        x_ros = step!(Rosenbrock23(), x0, A, M, b, dt)
        @test all(isapprox.(x_ros, 1.0 / (1.0 + dt); atol = 1.0e-12))
    end

    @testset "Steady-State Solvers" begin
        # Solve Ax = b
        A = [2.0 1.0; 1.0 3.0]
        b = [5.0, 7.0]
        x_direct = solve_steady_state(A, b)
        @test norm(A * x_direct - b) < 1.0e-10

        x_gmres = solve_steady_state(A, b; method = :gmres)
        @test norm(A * x_gmres - b) < 1.0e-8

        x_bicg = solve_steady_state(A, b; method = :bicgstab)
        @test norm(A * x_bicg - b) < 1.0e-8
    end

    @testset "Transient Solvers" begin
        # du/dt = -u => u(T) = u0 * exp(-T)
        n = 2
        A = Matrix(1.0I, n, n)
        M = Matrix(1.0I, n, n)
        b = zeros(n)
        x0 = ones(n)
        dt = 0.001
        t_final = 0.1

        x_euler = solve_transient(x0, A, M, b, dt, t_final)
        x_exact = exp(-t_final)
        @test all(abs.(x_euler .- x_exact) .< 0.01)

        x_rk2 = solve_transient_rk2(x0, A, M, b, dt, t_final)
        @test all(abs.(x_rk2 .- x_exact) .< 0.001)

        x_cn = solve_transient_crank_nicolson(x0, A, M, b, dt, t_final)
        @test all(abs.(x_cn .- x_exact) .< 0.001)
    end

    @testset "Graph Coloring" begin
        # Simple path graph: 1-2-3
        adj = [Int[2], Int[1, 3], Int[2]]
        colors = color_graph_greedy(adj)
        @test length(colors) == 3
        # Adjacent nodes must have different colors
        @test colors[1] != colors[2]
        @test colors[2] != colors[3]
        # Maximum 2 colors needed for a path
        @test maximum(colors) <= 2
    end

    @testset "Numerical Jacobian" begin
        # f(x, t) = [x1^2, x2^3]
        # J = [2x1 0; 0 3x2^2]
        function f!(fx, x, t)
            fx[1] = x[1]^2
            fx[2] = x[2]^3
        end

        x = [2.0, 3.0]
        J = zeros(2, 2)
        compute_numerical_jacobian!(J, f!, x, 0.0)

        @test J[1, 1] ≈ 4.0 atol = 1.0e-5   # 2*x1
        @test J[1, 2] ≈ 0.0 atol = 1.0e-5
        @test J[2, 1] ≈ 0.0 atol = 1.0e-5
        @test J[2, 2] ≈ 27.0 atol = 1.0e-4  # 3*x2^2
    end

    @testset "Adjoint Sensitivity" begin
        # Simple test: J^T * lambda = dJdu
        JT = [1.0 0.0; 0.0 2.0]
        dJdu = [3.0, 4.0]
        lambda = compute_adjoint(JT, dJdu)
        @test lambda ≈ [3.0, 2.0]

        # Sensitivity: dJ/dp = dJdp - lambda^T * dRdp
        dRdp = [1.0 0.0; 0.0 1.0]
        dJdp = [0.0, 0.0]
        sens = compute_sensitivity(lambda, dRdp, dJdp)
        @test sens ≈ [-3.0, -2.0]
    end

    @testset "InverseProblem struct" begin
        cost(p) = sum(p .^ 2)
        grad!(G, p) = (G .= 2.0 .* p)
        prob = InverseProblem(cost, grad!, [1.0, 2.0], nothing)
        @test prob.initial_params ≈ [1.0, 2.0]
        @test prob.cost_func([1.0]) == 1.0

        # calibrate_model should throw a helpful error
        @test_throws ErrorException calibrate_model(prob)
    end

    @testset "ControllerWithEvents" begin
        tc = TimeController(0.0, 0.01, 1.0; adaptivity = false)
        events = Event[]
        cwe = ControllerWithEvents(tc, events)
        @test cwe.time === tc
        @test isempty(cwe.events)
    end
end
