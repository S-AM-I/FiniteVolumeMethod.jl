using FiniteVolumeMethod
using Test
using StaticArrays

# ============================================================
# SRMHDEquations Type Tests
# ============================================================
@testset "SRMHDEquations Type" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)

    @testset "1D SRMHD" begin
        law = SRMHDEquations{1}(eos)
        @test nvariables(law) == 8
        @test law.eos === eos
        @test law.con2prim_tol == 1.0e-12
        @test law.con2prim_maxiter == 50
    end

    @testset "2D SRMHD" begin
        law = SRMHDEquations{2}(eos)
        @test nvariables(law) == 8
    end

    @testset "Custom con2prim params" begin
        law = SRMHDEquations{1}(eos; con2prim_tol = 1.0e-10, con2prim_maxiter = 100)
        @test law.con2prim_tol == 1.0e-10
        @test law.con2prim_maxiter == 100
    end
end

# ============================================================
# Con2Prim Roundtrip Tests
# ============================================================
@testset "Con2Prim Roundtrip" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{1}(eos)

    @testset "Static states (v=0)" begin
        states = [
            SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0),
            SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0),
            SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        ]
        for w in states
            u = primitive_to_conserved(law, w)
            w2 = conserved_to_primitive(law, u)
            @test w2 ≈ w atol = 1.0e-10
        end
    end

    @testset "Mildly relativistic (v ~ 0.1-0.5)" begin
        states = [
            SVector(1.0, 0.1, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0),
            SVector(1.0, 0.3, 0.2, 0.0, 2.0, 0.5, 0.8, 0.3),
            SVector(0.5, -0.4, 0.1, 0.2, 3.0, 1.0, -0.5, 0.5),
            SVector(1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0),
        ]
        for w in states
            u = primitive_to_conserved(law, w)
            w2 = conserved_to_primitive(law, u)
            @test w2 ≈ w atol = 1.0e-9
        end
    end

    @testset "Moderately relativistic (v ~ 0.5-0.9)" begin
        states = [
            SVector(1.0, 0.7, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0),
            SVector(1.0, 0.5, 0.5, 0.0, 2.0, 0.5, 0.8, 0.0),
            SVector(0.5, 0.0, 0.0, 0.8, 3.0, 1.0, 0.0, 0.5),
        ]
        for w in states
            u = primitive_to_conserved(law, w)
            w2 = conserved_to_primitive(law, u)
            @test w2 ≈ w atol = 1.0e-8
        end
    end

    @testset "Ultra-relativistic (W=10)" begin
        # v = sqrt(1 - 1/W²) ≈ 0.995 for W=10
        v = sqrt(1 - 1 / 100.0)
        w = SVector(1.0, v, 0.0, 0.0, 10.0, 0.5, 0.0, 0.0)
        u = primitive_to_conserved(law, w)
        w2 = conserved_to_primitive(law, u)
        @test w2[1] ≈ w[1] rtol = 1.0e-6   # density
        @test w2[2] ≈ w[2] rtol = 1.0e-6   # velocity
        @test w2[5] ≈ w[5] rtol = 1.0e-6   # pressure
    end

    @testset "B=0 (SR hydro limit)" begin
        w = SVector(1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        u = primitive_to_conserved(law, w)
        w2 = conserved_to_primitive(law, u)
        @test w2 ≈ w atol = 1.0e-10
    end

    @testset "Random valid states" begin
        for _ in 1:100
            ρ = 0.1 + 9.9 * rand()
            v_mag = 0.9 * rand()
            θ = 2π * rand()
            φ = π * rand()
            vx = v_mag * sin(φ) * cos(θ)
            vy = v_mag * sin(φ) * sin(θ)
            vz = v_mag * cos(φ)
            P = 0.01 + 9.99 * rand()
            Bx = 2 * rand() - 1
            By = 2 * rand() - 1
            Bz = 2 * rand() - 1
            w = SVector(ρ, vx, vy, vz, P, Bx, By, Bz)
            u = primitive_to_conserved(law, w)
            w2 = conserved_to_primitive(law, u)
            @test w2[1] ≈ w[1] rtol = 1.0e-8
            @test w2[2] ≈ w[2] atol = 1.0e-8
            @test w2[3] ≈ w[3] atol = 1.0e-8
            @test w2[4] ≈ w[4] atol = 1.0e-8
            @test w2[5] ≈ w[5] rtol = 1.0e-8
        end
    end
end

# ============================================================
# Con2Prim Convergence
# ============================================================
@testset "Con2Prim Convergence" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)

    w = SVector(1.0, 0.5, 0.3, 0.0, 2.0, 0.5, 0.8, 0.3)
    u = FiniteVolumeMethod.primitive_to_conserved(SRMHDEquations{1}(eos), w)

    w_rec, result = FiniteVolumeMethod.srmhd_con2prim(eos, u, 1.0e-12, 50)
    @test result.converged == true
    @test result.iterations < 50
    @test result.residual < 1.0e-10
end

# ============================================================
# Physical Flux Tests
# ============================================================
@testset "Physical Flux" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{1}(eos)

    @testset "B=0 reduces to SR Euler flux" begin
        w = SVector(1.0, 0.3, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0)
        f = physical_flux(law, w, 1)
        # With B=0, the SRMHD flux should simplify to SR Euler:
        γ_eos = eos.gamma
        ρ, vx, vy, vz, P = w[1], w[2], w[3], w[4], w[5]
        W = 1 / sqrt(1 - vx^2)
        ε = P / ((γ_eos - 1) * ρ)
        h = 1 + ε + P / ρ
        D = ρ * W
        Sx = ρ * h * W^2 * vx
        tau = ρ * h * W^2 - P - D

        @test f[1] ≈ D * vx atol = 1.0e-14           # D*vx
        @test f[5] ≈ Sx - D * vx atol = 1.0e-12       # τ flux = Sx - D*vx
    end

    @testset "v≪1 approaches Newtonian MHD flux" begin
        v_small = 1.0e-4
        w = SVector(1.0, v_small, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
        f_sr = physical_flux(law, w, 1)

        # Newtonian MHD flux
        law_mhd = IdealMHDEquations{1}(eos)
        f_mhd = physical_flux(law_mhd, w, 1)

        # At low v, SRMHD flux ≈ Newtonian MHD flux (to leading order in v)
        # D flux ≈ ρv (since W ≈ 1)
        @test f_sr[1] ≈ f_mhd[1] rtol = 1.0e-3
        # Bx flux = 0 in both
        @test f_sr[6] ≈ 0.0 atol = 1.0e-14
        # Induction fluxes should be similar
        @test f_sr[7] ≈ f_mhd[7] rtol = 1.0e-3
        @test f_sr[8] ≈ f_mhd[8] atol = 1.0e-3
    end

    @testset "Flux consistency (F·n = 0 for Bn)" begin
        # The normal B flux should be zero (Bx flux for dir=1, By flux for dir=2)
        w = SVector(1.0, 0.5, 0.3, 0.0, 2.0, 0.5, 0.8, 0.3)
        f1 = physical_flux(law, w, 1)
        @test f1[6] ≈ 0.0 atol = 1.0e-14  # Bx flux = 0 for x-direction

        law2d = SRMHDEquations{2}(eos)
        f2 = physical_flux(law2d, w, 2)
        @test f2[7] ≈ 0.0 atol = 1.0e-14  # By flux = 0 for y-direction
    end
end

# ============================================================
# Wave Speed Tests
# ============================================================
@testset "Wave Speeds" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{1}(eos)

    @testset "v=0 -> classical magnetosonic" begin
        # Use a genuinely non-relativistic state (P << ρ, B² << ρ)
        # so the classical fast magnetosonic speed is a good approximation
        ρ = 1.0
        P = 1.0e-4
        Bx = 0.01
        w = SVector(ρ, 0.0, 0.0, 0.0, P, Bx, 0.0, 0.0)
        λm, λp = wave_speeds(law, w, 1)
        @test λp > 0
        @test λm < 0
        @test λp ≈ -λm atol = 1.0e-14  # symmetric when v=0

        # In the non-relativistic limit (h ≈ 1), should match classical fast speed
        γ = eos.gamma
        cs_sq = γ * P / ρ
        va_sq = Bx^2 / ρ
        cf_class = sqrt(cs_sq + va_sq)
        @test λp ≈ cf_class rtol = 0.01
    end

    @testset "B=0 -> SR hydro sound speed" begin
        w = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        λm, λp = wave_speeds(law, w, 1)
        γ = eos.gamma
        ε = 1.0 / ((γ - 1) * 1.0)
        h = 1 + ε + 1.0
        cs = sqrt(γ * 1.0 / (1.0 * h))
        @test λp ≈ cs atol = 1.0e-12
        @test λm ≈ -cs atol = 1.0e-12
    end

    @testset "Wave speeds bounded by speed of light" begin
        # Even for very high B and v, wave speeds < 1
        w = SVector(0.01, 0.9, 0.0, 0.0, 100.0, 10.0, 10.0, 0.0)
        λm, λp = wave_speeds(law, w, 1)
        @test abs(λm) < 1.0
        @test abs(λp) < 1.0
    end

    @testset "max_wave_speed consistency" begin
        w = SVector(1.0, 0.3, 0.2, 0.0, 2.0, 0.5, 0.8, 0.3)
        λmax = max_wave_speed(law, w, 1)
        λm, λp = wave_speeds(law, w, 1)
        @test λmax ≈ max(abs(λm), abs(λp))
    end
end

# ============================================================
# Balsara Shock Tube Tests
# ============================================================
@testset "Balsara Shock Tubes" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)

    @testset "Balsara 1 (Generic Alfvén test)" begin
        law = SRMHDEquations{1}(eos)
        mesh = StructuredMesh1D(0.0, 1.0, 400)

        wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
        wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)

        ic(x) = x < 0.5 ? wL : wR

        prob = HyperbolicProblem(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(), ic;
            final_time = 0.4, cfl = 0.4
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        # Basic sanity: solution should not have NaN or Inf
        @test all(isfinite.(w[1]) for w in W)
        @test all(isfinite.(w[5]) for w in W)

        # Density bounds
        @test all(w[1] > 0 for w in W)
        # Pressure positive
        @test all(w[5] > 0 for w in W)

        # Check that solution evolved (not stuck at IC)
        @test !all(w[1] ≈ wL[1] for w in W)
    end

    @testset "Balsara 2 (Mildly relativistic)" begin
        law = SRMHDEquations{1}(eos)
        mesh = StructuredMesh1D(0.0, 1.0, 400)

        wL = SVector(1.0, 0.0, 0.0, 0.0, 30.0, 5.0, 6.0, 6.0)
        wR = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.7, 0.7)

        ic(x) = x < 0.5 ? wL : wR

        prob = HyperbolicProblem(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(), ic;
            final_time = 0.4, cfl = 0.3
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test all(isfinite.(w[1]) for w in W)
        @test all(w[1] > 0 for w in W)
        @test all(w[5] > 0 for w in W)
    end

    @testset "Balsara 3 (Strongly magnetized)" begin
        law = SRMHDEquations{1}(eos)
        mesh = StructuredMesh1D(0.0, 1.0, 400)

        wL = SVector(1.0, 0.0, 0.0, 0.0, 1000.0, 10.0, 7.0, 7.0)
        wR = SVector(1.0, 0.0, 0.0, 0.0, 0.1, 10.0, 0.7, 0.7)

        ic(x) = x < 0.5 ? wL : wR

        prob = HyperbolicProblem(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(), ic;
            final_time = 0.4, cfl = 0.2
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test all(isfinite.(w[1]) for w in W)
        @test all(w[1] > 0 for w in W)
        @test all(w[5] > 0 for w in W)
    end
end

# ============================================================
# Conservation Tests
# ============================================================
@testset "Conservation with Periodic BCs" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 100)

    # Smooth initial condition
    ic(x) = SVector(
        1.0 + 0.1 * sin(2π * x),
        0.1 * cos(2π * x),
        0.0, 0.0,
        1.0 + 0.1 * sin(2π * x),
        1.0, 0.5 * sin(2π * x), 0.0
    )

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), ic;
        final_time = 0.1, cfl = 0.4
    )

    x, U, t = solve_hyperbolic(prob)
    dx = mesh.dx

    # Total conserved quantities
    D_total = sum(u[1] for u in U) * dx
    Sx_total = sum(u[2] for u in U) * dx
    tau_total = sum(u[5] for u in U) * dx

    # Compute initial totals
    U0 = [FiniteVolumeMethod.primitive_to_conserved(law, ic(xi)) for xi in x]
    D_total_0 = sum(u[1] for u in U0) * dx
    Sx_total_0 = sum(u[2] for u in U0) * dx
    tau_total_0 = sum(u[5] for u in U0) * dx

    @test D_total ≈ D_total_0 rtol = 1.0e-10
    @test Sx_total ≈ Sx_total_0 atol = 1.0e-12
    @test tau_total ≈ tau_total_0 rtol = 1.0e-10
end

# ============================================================
# Relativistic Alfvén Wave
# ============================================================
@testset "Relativistic Alfvén Wave Advection" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{1}(eos)

    # A small-amplitude circularly polarized Alfvén wave propagating in x
    ρ0 = 1.0
    P0 = 1.0
    Bx0 = 1.0
    A = 0.01  # small amplitude
    γ_eos = eos.gamma

    ic(x) = SVector(
        ρ0, 0.0, A * sin(2π * x), A * cos(2π * x),
        P0, Bx0, A * sin(2π * x), A * cos(2π * x)
    )

    for N in [100, 200]
        mesh = StructuredMesh1D(0.0, 1.0, N)
        prob = HyperbolicProblem(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), ic;
            final_time = 0.1, cfl = 0.4
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        # Check density stays approximately constant (Alfvén wave is incompressible to leading order)
        ρ_vals = [w[1] for w in W]
        @test maximum(abs, ρ_vals .- ρ0) < 0.1  # small perturbation
    end
end

# ============================================================
# CFL Test
# ============================================================
@testset "CFL Stability" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 50)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)
    ic(x) = x < 0.5 ? wL : wR

    prob = HyperbolicProblem(
        law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), ic;
        final_time = 0.1, cfl = 0.8
    )

    x, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    @test t ≈ 0.1 atol = 1.0e-10
    @test all(isfinite.(w[1]) for w in W)
    @test all(w[1] > 0 for w in W)
end

# ============================================================
# All Limiters
# ============================================================
@testset "All Limiters" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 100)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)
    ic(x) = x < 0.5 ? wL : wR

    limiters = [
        MinmodLimiter(), SuperbeeLimiter(), VanLeerLimiter(),
        KorenLimiter(), OspreLimiter(),
    ]

    for lim in limiters
        prob = HyperbolicProblem(
            law, mesh, HLLSolver(), CellCenteredMUSCL(lim),
            TransmissiveBC(), TransmissiveBC(), ic;
            final_time = 0.1, cfl = 0.3
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test all(isfinite.(w[1]) for w in W)
        @test all(w[1] > 0 for w in W)
    end
end

# ============================================================
# Newtonian Limit
# ============================================================
@testset "Newtonian Limit" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law_sr = SRMHDEquations{1}(eos)
    law_mhd = IdealMHDEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 200)

    # Low-velocity Riemann problem
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0)
    ic(x) = x < 0.5 ? wL : wR

    prob_sr = HyperbolicProblem(
        law_sr, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), ic;
        final_time = 0.1, cfl = 0.3
    )

    prob_mhd = HyperbolicProblem(
        law_mhd, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), ic;
        final_time = 0.1, cfl = 0.3
    )

    x_sr, U_sr, t_sr = solve_hyperbolic(prob_sr)
    x_mhd, U_mhd, t_mhd = solve_hyperbolic(prob_mhd)

    W_sr = to_primitive(law_sr, U_sr)
    W_mhd = to_primitive(law_mhd, U_mhd)

    # In the Newtonian limit (low v), SRMHD density profile should match MHD
    ρ_sr = [w[1] for w in W_sr]
    ρ_mhd = [w[1] for w in W_mhd]

    # L2 error should be small (not exact due to different conserved variables)
    l2_err = sqrt(sum((ρ_sr .- ρ_mhd) .^ 2) / length(ρ_sr))
    @test l2_err < 0.15  # generous tolerance for different formulations
end
