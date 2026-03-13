# ============================================================
# CUDA Backend Tests for 2D Hyperbolic Solver
# ============================================================
#
# Gated on CUDA.functional() — skips gracefully on non-GPU systems.

using FiniteVolumeMethod
using StaticArrays
using Test

has_cuda = try
    using CUDA
    CUDA.functional()
catch
    false
end

if !has_cuda
    @info "CUDA not available — skipping CUDA tests"
    @testset "CUDA 2D Hyperbolic (skipped)" begin
        @test_broken false
    end
else

    @info "CUDA available — running GPU tests on $(CUDA.name(CUDA.device()))"

    function make_euler_2d(; nx = 32, ny = 32, cfl = 0.4, final_time = 0.001)
        eos = IdealGasEOS(1.4)
        law = EulerEquations{2}(eos)
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)
        bc = TransmissiveBC()
        ic(x, y) = x < 0.5 ? SVector(1.0, 0.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.0, 0.1)
        return HyperbolicProblem2D(
            law, mesh, HLLCSolver(), NoReconstruction(),
            bc, bc, bc, bc,
            ic; cfl = cfl, final_time = final_time,
        )
    end

    @testset "CUDA 2D Hyperbolic" begin

        @testset "backend_summary reports ready" begin
            s = backend_summary(CUDASolverBackend())
            @test occursin("CUDA", s)
            @test occursin("ready", s)
        end

        @testset "supports_backend" begin
            prob = make_euler_2d()
            @test supports_backend(prob, CUDASolverBackend())
        end

        @testset "GPU smoke test (ssprk3)" begin
            prob = make_euler_2d()
            coords, U, t = solve_hyperbolic(prob; backend = CUDASolverBackend())
            @test size(U) == (32, 32)
            @test t ≈ 0.001 atol = 1.0e-12
            @test U[1, 1] isa SVector
        end

        @testset "GPU smoke test (euler)" begin
            prob = make_euler_2d()
            coords, U, t = solve_hyperbolic(prob; method = :euler, backend = CUDASolverBackend())
            @test size(U) == (32, 32)
            @test t ≈ 0.001 atol = 1.0e-12
        end

        @testset "CPU vs GPU: Euler method" begin
            prob = make_euler_2d()
            _, U_cpu, t_cpu = solve_hyperbolic(prob; method = :euler, backend = CPUBackend())
            _, U_gpu, t_gpu = solve_hyperbolic(prob; method = :euler, backend = CUDASolverBackend())

            @test t_cpu ≈ t_gpu atol = 1.0e-14
            for j in axes(U_cpu, 2), i in axes(U_cpu, 1)
                @test U_cpu[i, j] ≈ U_gpu[i, j] atol = 1.0e-10
            end
        end

        @testset "CPU vs GPU: SSP-RK3 method" begin
            prob = make_euler_2d()
            _, U_cpu, t_cpu = solve_hyperbolic(prob; method = :ssprk3, backend = CPUBackend())
            _, U_gpu, t_gpu = solve_hyperbolic(prob; method = :ssprk3, backend = CUDASolverBackend())

            @test t_cpu ≈ t_gpu atol = 1.0e-14
            for j in axes(U_cpu, 2), i in axes(U_cpu, 1)
                @test U_cpu[i, j] ≈ U_gpu[i, j] atol = 1.0e-10
            end
        end

        @testset "Riemann solvers: LaxFriedrichs" begin
            eos = IdealGasEOS(1.4)
            law = EulerEquations{2}(eos)
            mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 16, 16)
            bc = TransmissiveBC()
            ic(x, y) = x < 0.5 ? SVector(1.0, 0.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.0, 0.1)
            prob = HyperbolicProblem2D(
                law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
                bc, bc, bc, bc,
                ic; cfl = 0.3, final_time = 0.001,
            )
            _, U_cpu, t_cpu = solve_hyperbolic(prob; method = :euler, backend = CPUBackend())
            _, U_gpu, t_gpu = solve_hyperbolic(prob; method = :euler, backend = CUDASolverBackend())
            @test t_cpu ≈ t_gpu atol = 1.0e-14
            for j in axes(U_cpu, 2), i in axes(U_cpu, 1)
                @test U_cpu[i, j] ≈ U_gpu[i, j] atol = 1.0e-10
            end
        end

        @testset "Riemann solvers: HLL" begin
            eos = IdealGasEOS(1.4)
            law = EulerEquations{2}(eos)
            mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 16, 16)
            bc = TransmissiveBC()
            ic(x, y) = x < 0.5 ? SVector(1.0, 0.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.0, 0.1)
            prob = HyperbolicProblem2D(
                law, mesh, HLLSolver(), NoReconstruction(),
                bc, bc, bc, bc,
                ic; cfl = 0.3, final_time = 0.001,
            )
            _, U_cpu, t_cpu = solve_hyperbolic(prob; method = :euler, backend = CPUBackend())
            _, U_gpu, t_gpu = solve_hyperbolic(prob; method = :euler, backend = CUDASolverBackend())
            @test t_cpu ≈ t_gpu atol = 1.0e-14
            for j in axes(U_cpu, 2), i in axes(U_cpu, 1)
                @test U_cpu[i, j] ≈ U_gpu[i, j] atol = 1.0e-10
            end
        end

        @testset "Boundary conditions: Reflective" begin
            eos = IdealGasEOS(1.4)
            law = EulerEquations{2}(eos)
            mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 16, 16)
            ic(x, y) = SVector(1.0, 0.1, 0.0, 2.5)
            prob = HyperbolicProblem2D(
                law, mesh, HLLCSolver(), NoReconstruction(),
                ReflectiveBC(), ReflectiveBC(), ReflectiveBC(), ReflectiveBC(),
                ic; cfl = 0.3, final_time = 0.001,
            )
            _, U_cpu, t_cpu = solve_hyperbolic(prob; method = :euler, backend = CPUBackend())
            _, U_gpu, t_gpu = solve_hyperbolic(prob; method = :euler, backend = CUDASolverBackend())
            @test t_cpu ≈ t_gpu atol = 1.0e-14
            for j in axes(U_cpu, 2), i in axes(U_cpu, 1)
                @test U_cpu[i, j] ≈ U_gpu[i, j] atol = 1.0e-10
            end
        end

        @testset "Boundary conditions: Periodic" begin
            eos = IdealGasEOS(1.4)
            law = EulerEquations{2}(eos)
            mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 16, 16)
            ic(x, y) = SVector(1.0 + 0.2 * sin(2π * x), 0.1, 0.0, 2.5)
            prob = HyperbolicProblem2D(
                law, mesh, HLLCSolver(), NoReconstruction(),
                PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
                PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
                ic; cfl = 0.3, final_time = 0.001,
            )
            _, U_cpu, t_cpu = solve_hyperbolic(prob; method = :euler, backend = CPUBackend())
            _, U_gpu, t_gpu = solve_hyperbolic(prob; method = :euler, backend = CUDASolverBackend())
            @test t_cpu ≈ t_gpu atol = 1.0e-14
            for j in axes(U_cpu, 2), i in axes(U_cpu, 1)
                @test U_cpu[i, j] ≈ U_gpu[i, j] atol = 1.0e-10
            end
        end

        @testset "MUSCL reconstruction on GPU" begin
            prob_muscl = let
                eos = IdealGasEOS(1.4)
                law = EulerEquations{2}(eos)
                mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 32, 32)
                bc = TransmissiveBC()
                ic(x, y) = x < 0.5 ? SVector(1.0, 0.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.0, 0.1)
                HyperbolicProblem2D(
                    law, mesh, HLLCSolver(), CellCenteredMUSCL(),
                    bc, bc, bc, bc,
                    ic; cfl = 0.3, final_time = 0.001,
                )
            end
            _, U_cpu, t_cpu = solve_hyperbolic(prob_muscl; method = :euler, backend = CPUBackend())
            _, U_gpu, t_gpu = solve_hyperbolic(prob_muscl; method = :euler, backend = CUDASolverBackend())
            @test t_cpu ≈ t_gpu atol = 1.0e-14
            for j in axes(U_cpu, 2), i in axes(U_cpu, 1)
                @test U_cpu[i, j] ≈ U_gpu[i, j] atol = 1.0e-10
            end
        end

        @testset "initialize_2d with CUDA backend" begin
            prob = make_euler_2d()
            U_cpu = initialize_2d(prob; backend = CPUBackend())
            U_gpu = initialize_2d(prob; backend = CUDASolverBackend())
            @test U_gpu isa CUDA.CuArray
            @test Array(U_gpu) == U_cpu
        end
    end

end # has_cuda
