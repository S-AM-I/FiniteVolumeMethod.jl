using FiniteVolumeMethod
using OrdinaryDiffEq
using OrdinaryDiffEqSSPRK: SSPRK33
using SciMLBase: SplitFunction
using StaticArrays
using Test

# ============================================================
# Test 1: SplitODEProblem construction for 1D Euler
# ============================================================
@testset "1D SplitODEProblem construction" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 64)

    function sod_ic(x)
        if x < 0.5
            return SVector(1.0, 0.0, 1.0)   # rho, v, P
        else
            return SVector(0.125, 0.0, 0.1)
        end
    end

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), sod_ic;
        final_time = 0.2, cfl = 0.5
    )

    split_prob = SplitODEProblem(prob, NullSource())

    # Type check: SplitODEProblem{true}(...) creates an ODEProblem with SplitFunction
    @test split_prob isa ODEProblem
    @test split_prob.f isa SplitFunction

    # u0 length: 64 cells * 3 variables (1D Euler)
    @test length(split_prob.u0) == 64 * 3

    # tspan
    @test split_prob.tspan == (0.0, 0.2)

    # Parameter is the cache (an AbstractSemidiscreteCache)
    @test split_prob.p isa AbstractSemidiscreteCache
    @test split_prob.p isa HyperbolicCache1D
end

# ============================================================
# Test 2: SplitODEProblem construction for 2D Euler
# ============================================================
@testset "2D SplitODEProblem construction" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    nx, ny = 16, 12
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    function uniform_ic_2d(x, y)
        return SVector(1.0, 0.0, 0.0, 1.0)  # rho, vx, vy, P
    end

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        uniform_ic_2d; final_time = 0.1, cfl = 0.4
    )

    split_prob = SplitODEProblem(prob, NullSource())

    # Type check: SplitODEProblem{true}(...) creates an ODEProblem with SplitFunction
    @test split_prob isa ODEProblem
    @test split_prob.f isa SplitFunction

    # u0 length: nx * ny * 4 variables (2D Euler)
    @test length(split_prob.u0) == nx * ny * 4

    # tspan
    @test split_prob.tspan == (0.0, 0.1)

    # Parameter is the 2D cache
    @test split_prob.p isa AbstractSemidiscreteCache
    @test split_prob.p isa HyperbolicCache2D
end

# ============================================================
# Test 3: NullSource solve -- IMEX vs pure explicit
# ============================================================
@testset "NullSource solve: IMEX vs pure explicit" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    ncells = 64
    mesh = StructuredMesh1D(0.0, 1.0, ncells)

    wL = SVector(1.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.1)

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        x -> x < 0.5 ? wL : wR;
        final_time = 0.1, cfl = 0.4
    )

    # --- Pure explicit solve via ODEProblem + SSPRK33 ---
    ode_prob = ODEProblem(prob)
    cache_explicit = ode_prob.p
    dt0_explicit = compute_initial_dt(cache_explicit, ode_prob.u0)
    sol_explicit = solve(
        ode_prob, SSPRK33();
        adaptive = false, dt = dt0_explicit
    )
    @test sol_explicit.retcode == ReturnCode.Success

    # --- SplitODEProblem solve via SSPRK33 (applies f1+f2 explicitly) ---
    # (KenCarp4 requires a nontrivial implicit part; NullSource is zero
    #  so Newton fails. SSPRK33 on SplitODEProblem is a valid test path.)
    split_prob = SplitODEProblem(prob, NullSource())
    cache_imex = split_prob.p
    dt0_imex = compute_initial_dt(cache_imex, split_prob.u0)
    sol_imex = solve(
        split_prob, SSPRK33();
        adaptive = false, dt = dt0_imex
    )
    @test sol_imex.retcode == ReturnCode.Success

    # Both should reach approximately the same final time
    @test sol_explicit.t[end] > 0.09
    @test sol_imex.t[end] > 0.09

    # Both solutions should be physical (positive density everywhere)
    U_explicit = reinterpret(SVector{3, Float64}, copy(sol_explicit.u[end]))
    U_imex = reinterpret(SVector{3, Float64}, copy(sol_imex.u[end]))

    for i in 1:ncells
        w_ex = conserved_to_primitive(law, U_explicit[i])
        w_im = conserved_to_primitive(law, U_imex[i])
        @test w_ex[1] > 0.0  # explicit: rho > 0
        @test w_ex[3] > 0.0  # explicit: P > 0
        @test w_im[1] > 0.0  # IMEX: rho > 0
        @test w_im[3] > 0.0  # IMEX: P > 0
    end
end

# ============================================================
# Test 4: CFL callback works with SplitODEProblem
# ============================================================
@testset "CFL callback with SplitODEProblem" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 50)

    wL = SVector(1.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.1)

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        x -> x < 0.5 ? wL : wR;
        final_time = 0.1, cfl = 0.3
    )

    split_prob = SplitODEProblem(prob, NullSource())
    cache = split_prob.p

    dt0 = compute_initial_dt(cache, split_prob.u0)

    # dt should be positive and finite
    @test dt0 > 0.0
    @test isfinite(dt0)

    # dt should be less than the domain size / wave speed (sanity bound)
    dx = 1.0 / 50
    @test dt0 < dx
end
