using FiniteVolumeMethod
using OrdinaryDiffEq
using OrdinaryDiffEqSSPRK: SSPRK33
using OrdinaryDiffEqSDIRK: KenCarp47
using ADTypes: AutoFiniteDiff
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

    # --- SplitODEProblem solve via SSPRK33 ---
    # SSPRK33 is not an IMEX method: it integrates the SplitFunction as a
    # single combined RHS (f1 + f2, both explicit). With NullSource this
    # cannot distinguish a dropped f2 from a zero f2 — the genuine
    # source-delivery gate is the dropped-f2 regression testset below,
    # which uses a Newton-based IMEX algorithm and a nonzero source.
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

# ============================================================
# Test 5: the stiff source genuinely fires on the split path
# ============================================================
#
# Regression for the dropped-f2 defect: `solve` used to rebuild the
# SplitODEProblem into a plain ODEProblem inside get_concrete_problem
# and silently integrate only the hyperbolic part f1. A counting
# source proves f2 is evaluated at all, and the relaxation physics
# proves its contribution actually reaches the state.

struct SDCountingSource{S} <: FiniteVolumeMethod.AbstractStiffSource
    inner::S
    calls::Base.RefValue{Int}
end
function FiniteVolumeMethod.evaluate_stiff_source(src::SDCountingSource, law, w, u)
    src.calls[] += 1
    return evaluate_stiff_source(src.inner, law, w, u)
end
function FiniteVolumeMethod.stiff_source_jacobian(src::SDCountingSource, law, w, u)
    return stiff_source_jacobian(src.inner, law, w, u)
end

@testset "Stiff source fires on the split path (dropped-f2 regression)" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    ncells = 32
    mesh = StructuredMesh1D(0.0, 1.0, ncells)

    # Uniform static state: the hyperbolic RHS is exactly zero, so the
    # state evolves by the cooling source alone. With T = P/rho
    # (mu_mol = 1, rho = 1, v = 0) and Lambda(T) = lambda * (T - P_target):
    #   dP/dt = -(gamma - 1) * lambda * (P - P_target)
    # so P(t) = P_target + (P0 - P_target) * exp(-(gamma - 1) * lambda * t).
    lambda = 50.0
    P_target = 1.0
    P0 = 3.0
    source = SDCountingSource(
        CoolingSource(T -> lambda * (T - P_target); mu_mol = 1.0), Ref(0)
    )

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), x -> SVector(1.0, 0.0, P0);
        final_time = 0.05, cfl = 0.4
    )
    split_prob = SplitODEProblem(prob, source)
    dt0 = compute_initial_dt(split_prob.p, split_prob.u0)
    sol = solve(
        split_prob, KenCarp47(autodiff = AutoFiniteDiff());
        adaptive = false, dt = dt0
    )
    @test sol.retcode == ReturnCode.Success

    # The source must actually have been evaluated (a dropped f2 never
    # calls it)
    @test source.calls[] > 0

    # And its contribution must reach the state: compare against the
    # exact relaxation ODE. A dropped f2 leaves P at P0 = 3.0, which is
    # ~73% away from the exact value — far outside the 5% gate.
    P_exact = P_target + (P0 - P_target) * exp(-(eos.gamma - 1) * lambda * sol.t[end])
    u_final = reinterpret(SVector{3, Float64}, copy(sol.u[end]))
    for i in 1:ncells
        w = conserved_to_primitive(law, u_final[i])
        @test abs(w[3] - P_exact) / P_exact < 0.05
    end
end
