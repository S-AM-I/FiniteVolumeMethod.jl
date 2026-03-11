# sciml_audit.jl — Verification tests for the SciML overlap audit
# Validates:
# 1. Limiter unification (ParabolicLimiters → schemes/limiters.jl)
# 2. k-epsilon interop (ParabolicKEpsilon ↔ StandardKEpsilon)
# 3. SciML bridge (parabolic_to_odefunction, parabolic_to_linearproblem)
# 4. Engine removal (no engine symbols exported)
# 5. VTK backward compatibility

using FiniteVolumeMethod
using Test
using LinearAlgebra
using SparseArrays
using SciMLBase

@testset verbose = true "SciML Audit Verification" begin
    # ------------------------------------------------------------------
    # 1. Limiter unification
    # ------------------------------------------------------------------
    @testset "Limiter unification — ParabolicLimiters delegates to canonical" begin
        # Core limiter functions should be identical (same function object)
        for val_pair in [(1.0, 2.0), (-1.0, 2.0), (0.5, 0.5), (3.0, -1.0)]
            a, b = val_pair
            @test ParabolicLimiters.minmod(a, b) === minmod(a, b)
            @test ParabolicLimiters.superbee(a, b) === superbee(a, b)
            @test ParabolicLimiters.van_leer(a, b) === van_leer(a, b)
        end

        for r in [0.0, 0.5, 1.0, 2.0, -1.0]
            @test ParabolicLimiters.ospre(r) === ospre(r)
        end

        # ParabolicLimiters-specific helpers still work
        phi = [1.0, 2.0, 4.0, 7.0, 11.0]
        slope = ParabolicLimiters.limit_slope_1d(phi, 3, :left, :minmod)
        @test isfinite(slope)

        ratio = ParabolicLimiters.compute_slope_ratio_1d(phi, 3, :left)
        @test isfinite(ratio)

        # Symbol-based dispatch works
        @test ParabolicLimiters.apply_limiter(:minmod, 0.5) == minmod(0.5, 1.0)
        @test ParabolicLimiters.apply_limiter(:superbee, 0.5) == superbee(0.5, 1.0)

        # Strategy selection works
        @test ParabolicLimiters.select_limiter_strategy(:conservative) == :minmod
        @test ParabolicLimiters.select_limiter_strategy(:accuracy) == :van_leer
    end

    # ------------------------------------------------------------------
    # 2. k-epsilon interop
    # ------------------------------------------------------------------
    @testset "k-epsilon interop — StandardKEpsilon ↔ ParabolicKEpsilon" begin
        # Default construction
        ske = StandardKEpsilon()
        pke = ParabolicKEpsilon()

        @test ske.C_mu == pke.C_mu == 0.09
        @test ske.sigma_k == pke.sigma_k == 1.0
        @test ske.sigma_epsilon == pke.sigma_epsilon == 1.3
        @test ske.C1_epsilon == pke.C1_epsilon == 1.44
        @test ske.C2_epsilon == pke.C2_epsilon == 1.92

        # Conversion: StandardKEpsilon → ParabolicKEpsilon
        pke2 = ParabolicKEpsilon(ske)
        @test pke2.C_mu == ske.C_mu
        @test pke2.sigma_k == ske.sigma_k
        @test pke2.sigma_epsilon == ske.sigma_epsilon

        # Conversion: ParabolicKEpsilon → StandardKEpsilon
        ske2 = StandardKEpsilon(pke)
        @test ske2.C_mu == pke.C_mu
        @test ske2.kappa == 0.41  # default kappa added

        # Custom coefficients round-trip
        custom_ske = StandardKEpsilon(; C_mu = 0.1, C1_epsilon = 1.5)
        custom_pke = ParabolicKEpsilon(custom_ske)
        @test custom_pke.C_mu == 0.1
        @test custom_pke.C1_epsilon == 1.5

        # ParabolicKEpsilon keyword constructor
        pke_kw = ParabolicKEpsilon(; C_mu = 0.1, sigma_k = 0.8)
        @test pke_kw.C_mu == 0.1
        @test pke_kw.sigma_k == 0.8
    end

    # ------------------------------------------------------------------
    # 3. SciML bridge
    # ------------------------------------------------------------------
    @testset "SciML bridge — parabolic_to_odefunction" begin
        # Create a simple 3x3 system: du/dt = b - A*u
        A = sparse([2.0 -1.0 0.0; -1.0 2.0 -1.0; 0.0 -1.0 2.0])
        b = [1.0, 0.0, 1.0]
        u0 = zeros(3)

        # Without mass matrix
        f = parabolic_to_odefunction(A, b)
        @test f isa SciMLBase.ODEFunction
        du = similar(u0)
        f(du, u0, nothing, 0.0)
        @test du ≈ b  # at u=0, du = b - A*0 = b

        # With mass matrix
        M = sparse(2.0 * I(3))
        f2 = parabolic_to_odefunction(A, M, b)
        @test f2 isa SciMLBase.ODEFunction
        du2 = similar(u0)
        f2(du2, u0, nothing, 0.0)
        @test du2 ≈ b / 2  # M^{-1} * b = b/2
    end

    @testset "SciML bridge — parabolic_to_linearproblem" begin
        A = sparse([2.0 -1.0; -1.0 2.0])
        b = [1.0, 2.0]
        prob = parabolic_to_linearproblem(A, b)
        @test prob isa SciMLBase.LinearProblem
        @test prob.A === A
        @test prob.b === b
    end

    # ------------------------------------------------------------------
    # 4. Engine removal verification
    # ------------------------------------------------------------------
    @testset "Engine symbols removed from exports" begin
        # These types/functions should no longer be exported
        @test !isdefined(FiniteVolumeMethod, :ForwardEuler)
        @test !isdefined(FiniteVolumeMethod, :RK2)
        @test !isdefined(FiniteVolumeMethod, :Rosenbrock23)
        @test !isdefined(FiniteVolumeMethod, :CrankNicolson)
        @test !isdefined(FiniteVolumeMethod, :Simulation)
        @test !isdefined(FiniteVolumeMethod, :TimeGrid)
        @test !isdefined(FiniteVolumeMethod, :TimeController)
        # Note: :Event is Base.Event (threading primitive), not engine's Event — skip
        @test !isdefined(FiniteVolumeMethod, :newton_raphson)
        @test !isdefined(FiniteVolumeMethod, :anderson_acceleration)
        @test !isdefined(FiniteVolumeMethod, :solve_steady_state)
        @test !isdefined(FiniteVolumeMethod, :solve_steady_state_gmres)
        @test !isdefined(FiniteVolumeMethod, :solve_transient)
        @test !isdefined(FiniteVolumeMethod, :color_graph_greedy)
        @test !isdefined(FiniteVolumeMethod, :compute_adjoint)
        @test !isdefined(FiniteVolumeMethod, :InverseProblem)
        @test !isdefined(FiniteVolumeMethod, :event_to_callback)
        @test !isdefined(FiniteVolumeMethod, :events_to_callbackset)
    end

    # ------------------------------------------------------------------
    # 5. VTK output still works
    # ------------------------------------------------------------------
    @testset "VTK output backward compatibility" begin
        tmpfile = tempname() * ".vtk"
        xcoords = [0.0, 1.0, 2.0, 3.0]
        scalars = [1.0, 2.0, 3.0, 4.0]
        path = write_line_vtk(tmpfile, xcoords, scalars; label = "temperature")
        @test isfile(path)
        content = read(path, String)
        @test contains(content, "vtk DataFile Version 3.0")
        @test contains(content, "SCALARS temperature float")
        rm(path)

        # 3D stub returns nothing
        @test write_structured_vtk_3d() === nothing
    end
end
