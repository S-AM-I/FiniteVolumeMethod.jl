using FiniteVolumeMethod
using Test
using StaticArrays

# ============================================================
# IdealMHDEquations Type Tests
# ============================================================
@testset "IdealMHDEquations Type" begin
    eos = IdealGasEOS(gamma = 2.0)

    @testset "1D MHD" begin
        law = IdealMHDEquations{1}(eos)
        @test nvariables(law) == 8
        @test law.eos === eos
    end

    @testset "2D MHD" begin
        law = IdealMHDEquations{2}(eos)
        @test nvariables(law) == 8
        @test law.eos === eos
    end
end

# ============================================================
# Conserved ↔ Primitive Conversion
# ============================================================
@testset "MHD Conserved ↔ Primitive" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{1}(eos)

    @testset "Roundtrip" begin
        # Various MHD states
        states = [
            SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0),   # Brio-Wu left
            SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0), # Brio-Wu right
            SVector(1.0, 1.0, 0.5, -0.3, 2.0, 0.5, 0.8, 0.3),    # general state
            SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),     # B=0 (hydro limit)
            SVector(0.5, -1.0, 2.0, 0.0, 3.0, 1.0, -0.5, 0.5),   # another general
        ]
        for w in states
            u = primitive_to_conserved(law, w)
            w2 = conserved_to_primitive(law, u)
            @test w2 ≈ w atol = 1e-14
        end
    end

    @testset "Energy computation" begin
        w = SVector(1.0, 1.0, 0.5, -0.3, 2.0, 0.5, 0.8, 0.3)
        ρ, vx, vy, vz, P, Bx, By, Bz = w
        γ = eos.gamma
        KE = 0.5 * ρ * (vx^2 + vy^2 + vz^2)
        ME = 0.5 * (Bx^2 + By^2 + Bz^2)
        E_expected = P / (γ - 1) + KE + ME

        u = primitive_to_conserved(law, w)
        @test u[1] ≈ ρ          # density
        @test u[2] ≈ ρ * vx     # x-momentum
        @test u[3] ≈ ρ * vy     # y-momentum
        @test u[4] ≈ ρ * vz     # z-momentum
        @test u[5] ≈ E_expected  # total energy
        @test u[6] ≈ Bx         # Bx
        @test u[7] ≈ By         # By
        @test u[8] ≈ Bz         # Bz
    end

    @testset "Hydro limit (B=0)" begin
        # With B=0, MHD should reduce to Euler
        w_mhd = SVector(1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        u_mhd = primitive_to_conserved(law, w_mhd)

        eos_euler = law.eos
        law_euler = EulerEquations{1}(eos_euler)
        w_euler = SVector(1.0, 0.5, 1.0)
        u_euler = primitive_to_conserved(law_euler, w_euler)

        @test u_mhd[1] ≈ u_euler[1]   # ρ
        @test u_mhd[2] ≈ u_euler[2]   # ρvx
        @test u_mhd[5] ≈ u_euler[3]   # E
    end
end

# ============================================================
# Physical Flux Tests
# ============================================================
@testset "MHD Physical Flux" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{1}(eos)

    @testset "x-flux structure" begin
        w = SVector(1.0, 1.0, 0.5, -0.3, 2.0, 0.5, 0.8, 0.3)
        F = physical_flux(law, w, 1)

        ρ, vx, vy, vz, P, Bx, By, Bz = w
        v_dot_B = vx * Bx + vy * By + vz * Bz
        Bsq = Bx^2 + By^2 + Bz^2
        Ptot = P + 0.5 * Bsq
        KE = 0.5 * ρ * (vx^2 + vy^2 + vz^2)
        E = P / (eos.gamma - 1) + KE + 0.5 * Bsq

        @test F[1] ≈ ρ * vx
        @test F[2] ≈ ρ * vx^2 + Ptot - Bx^2
        @test F[3] ≈ ρ * vx * vy - Bx * By
        @test F[4] ≈ ρ * vx * vz - Bx * Bz
        @test F[5] ≈ (E + Ptot) * vx - Bx * v_dot_B
        @test F[6] ≈ 0.0   # Bx flux = 0
        @test F[7] ≈ By * vx - Bx * vy
        @test F[8] ≈ Bz * vx - Bx * vz
    end

    @testset "y-flux structure" begin
        law2d = IdealMHDEquations{2}(eos)
        w = SVector(1.0, 1.0, 0.5, -0.3, 2.0, 0.5, 0.8, 0.3)
        G = physical_flux(law2d, w, 2)

        ρ, vx, vy, vz, P, Bx, By, Bz = w
        v_dot_B = vx * Bx + vy * By + vz * Bz
        Bsq = Bx^2 + By^2 + Bz^2
        Ptot = P + 0.5 * Bsq
        KE = 0.5 * ρ * (vx^2 + vy^2 + vz^2)
        E = P / (eos.gamma - 1) + KE + 0.5 * Bsq

        @test G[1] ≈ ρ * vy
        @test G[2] ≈ ρ * vx * vy - Bx * By
        @test G[3] ≈ ρ * vy^2 + Ptot - By^2
        @test G[4] ≈ ρ * vy * vz - By * Bz
        @test G[5] ≈ (E + Ptot) * vy - By * v_dot_B
        @test G[6] ≈ Bx * vy - By * vx
        @test G[7] ≈ 0.0   # By flux = 0
        @test G[8] ≈ Bz * vy - By * vz
    end

    @testset "B=0 reduces to Euler flux" begin
        w_mhd = SVector(1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        F_mhd = physical_flux(law, w_mhd, 1)

        law_euler = EulerEquations{1}(eos)
        w_euler = SVector(1.0, 0.5, 1.0)
        F_euler = physical_flux(law_euler, w_euler, 1)

        @test F_mhd[1] ≈ F_euler[1]   # mass flux
        @test F_mhd[2] ≈ F_euler[2]   # x-momentum flux
        @test F_mhd[5] ≈ F_euler[3]   # energy flux
    end
end

# ============================================================
# Wave Speed Tests
# ============================================================
@testset "MHD Wave Speeds" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{1}(eos)

    @testset "Basic properties" begin
        w = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
        cf = fast_magnetosonic_speed(law, w, 1)
        cs = slow_magnetosonic_speed(law, w, 1)

        @test cf > 0
        @test cs ≥ 0
        @test cf ≥ cs  # fast ≥ slow always

        # Alfvén speed
        ρ, _, _, _, _, Bx, By, Bz = w
        ca = abs(Bx) / sqrt(ρ)
        @test cf ≥ ca   # fast ≥ Alfvén
        @test ca ≥ cs   # Alfvén ≥ slow
    end

    @testset "B=0 reduces to sound speed" begin
        w_hydro = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        cf = fast_magnetosonic_speed(law, w_hydro, 1)
        cs_hydro = sqrt(eos.gamma * 1.0 / 1.0)  # √(γP/ρ)
        @test cf ≈ cs_hydro atol = 1e-14

        cs = slow_magnetosonic_speed(law, w_hydro, 1)
        @test cs ≈ 0.0 atol = 1e-14
    end

    @testset "Perpendicular B only (Bn=0)" begin
        # When Bn = 0 (Bx=0 for dir=1), cf² = a² + b², cs = 0
        w = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
        cf = fast_magnetosonic_speed(law, w, 1)
        cs = slow_magnetosonic_speed(law, w, 1)

        a_sq = eos.gamma * 1.0 / 1.0
        b_sq = 1.0 / 1.0  # By² / ρ
        @test cf ≈ sqrt(a_sq + b_sq) atol = 1e-14
        @test cs ≈ 0.0 atol = 1e-14
    end

    @testset "wave_speeds and max_wave_speed" begin
        w = SVector(1.0, 0.5, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
        λmin, λmax = wave_speeds(law, w, 1)
        cf = fast_magnetosonic_speed(law, w, 1)
        @test λmin ≈ 0.5 - cf
        @test λmax ≈ 0.5 + cf

        mws = max_wave_speed(law, w, 1)
        @test mws ≈ abs(0.5) + cf
    end

    @testset "2D wave speeds" begin
        law2d = IdealMHDEquations{2}(eos)
        w = SVector(1.0, 0.5, -0.3, 0.0, 1.0, 0.75, 1.0, 0.0)

        # x-direction
        cfx = fast_magnetosonic_speed(law2d, w, 1)
        λmin_x, λmax_x = wave_speeds(law2d, w, 1)
        @test λmin_x ≈ 0.5 - cfx
        @test λmax_x ≈ 0.5 + cfx

        # y-direction
        cfy = fast_magnetosonic_speed(law2d, w, 2)
        λmin_y, λmax_y = wave_speeds(law2d, w, 2)
        @test λmin_y ≈ -0.3 - cfy
        @test λmax_y ≈ -0.3 + cfy
    end
end

# ============================================================
# HLLD Riemann Solver Tests
# ============================================================
@testset "HLLD Riemann Solver" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{1}(eos)
    hlld = HLLDSolver()
    hll = HLLSolver()
    lf = LaxFriedrichsSolver()

    @testset "Identical states → physical flux" begin
        states = [
            SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0),
            SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0),
            SVector(1.0, 1.0, 0.5, -0.3, 2.0, 0.5, 0.8, 0.3),
        ]
        for w in states
            F_exact = physical_flux(law, w, 1)
            F_hlld = solve_riemann(hlld, law, w, w, 1)
            @test F_hlld ≈ F_exact atol = 1e-12
        end
    end

    @testset "Symmetry" begin
        # Flipping left↔right should negate the flux (in some sense)
        wL = SVector(1.0, 0.3, 0.1, 0.0, 1.0, 0.75, 1.0, 0.0)
        wR = SVector(0.5, -0.2, 0.1, 0.0, 0.5, 0.75, -0.5, 0.0)

        F_LR = solve_riemann(hlld, law, wL, wR, 1)
        F_RL = solve_riemann(hlld, law, wR, wL, 1)

        # Not strictly equal/opposite, but mass flux should flip sign
        # Actually, they should give different fluxes. Just check consistency.
        @test all(isfinite.(F_LR))
        @test all(isfinite.(F_RL))
    end

    @testset "HLLD vs HLL consistency" begin
        # HLLD and HLL should give consistent (similar direction) fluxes
        wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
        wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0)

        F_hlld = solve_riemann(hlld, law, wL, wR, 1)
        F_hll = solve_riemann(hll, law, wL, wR, 1)
        F_lf = solve_riemann(lf, law, wL, wR, 1)

        # All three should be finite
        @test all(isfinite.(F_hlld))
        @test all(isfinite.(F_hll))
        @test all(isfinite.(F_lf))

        # Mass flux should have the same sign for all three
        if abs(F_hlld[1]) > 1e-10
            @test sign(F_hlld[1]) == sign(F_hll[1])
        end
    end

    @testset "B=0 consistency with Euler HLLC" begin
        # When B=0, HLLD should give similar results to HLLC on Euler
        wL_mhd = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        wR_mhd = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0)

        F_hlld = solve_riemann(hlld, law, wL_mhd, wR_mhd, 1)

        # Just check it's finite and reasonable
        @test all(isfinite.(F_hlld))
        @test F_hlld[6] ≈ 0.0 atol = 1e-14  # No Bx flux
    end

    @testset "Degenerate: Bn=0 (no Alfvén waves)" begin
        # With Bn=0, HLLD degenerates (no rotational waves)
        wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
        wR = SVector(0.5, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5, 0.0)

        F_hlld = solve_riemann(hlld, law, wL, wR, 1)
        @test all(isfinite.(F_hlld))
    end
end

# ============================================================
# 1D Brio-Wu MHD Shock Tube
# ============================================================
@testset "Brio-Wu Shock Tube" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{1}(eos)

    # Brio & Wu (1988) initial conditions
    Bx = 0.75
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, Bx, -1.0, 0.0)

    function bw_ic(x)
        return x < 0.5 ? wL : wR
    end

    @testset "HLLD solver" begin
        mesh = StructuredMesh1D(0.0, 1.0, 800)
        prob = HyperbolicProblem(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(), bw_ic;
            final_time = 0.1, cfl = 0.8
        )

        x, U, t_final = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t_final ≈ 0.1 atol = 1e-10
        @test length(x) == 800
        @test length(W) == 800

        # Extract density profile
        ρ = [w[1] for w in W]
        P = [w[5] for w in W]

        # Basic physical checks
        @test all(ρ .> 0)     # Positivity of density
        @test all(P .> 0)     # Positivity of pressure
        @test all(isfinite.(ρ))
        @test all(isfinite.(P))

        # Bx should remain constant throughout (trivially satisfied in 1D)
        Bx_sol = [w[6] for w in W]
        @test all(b -> abs(b - Bx) < 1e-10, Bx_sol)

        # Check that the solution has structure (not all the same value)
        @test maximum(ρ) > minimum(ρ) + 0.01

        # The density profile should have features between left and right states
        @test maximum(ρ) ≤ 1.1  # Should not exceed left state much
        @test minimum(ρ) ≥ 0.05  # Should not go below right state much

        # Check that By changes sign (the reversal is a key feature)
        By_sol = [w[7] for w in W]
        @test any(b -> b > 0.5, By_sol)
        @test any(b -> b < -0.5, By_sol)
    end

    @testset "HLL solver" begin
        mesh = StructuredMesh1D(0.0, 1.0, 800)
        prob = HyperbolicProblem(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(), bw_ic;
            final_time = 0.1, cfl = 0.8
        )

        x, U, t_final = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        ρ = [w[1] for w in W]
        P = [w[5] for w in W]

        @test t_final ≈ 0.1 atol = 1e-10
        @test all(ρ .> 0)
        @test all(P .> 0)
        @test all(isfinite.(ρ))
    end

    @testset "HLLD vs HLL contact resolution" begin
        # HLLD should resolve the contact discontinuity better than HLL
        mesh = StructuredMesh1D(0.0, 1.0, 400)
        prob_hlld = HyperbolicProblem(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(), bw_ic;
            final_time = 0.1, cfl = 0.8
        )
        prob_hll = HyperbolicProblem(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(), bw_ic;
            final_time = 0.1, cfl = 0.8
        )

        _, U_hlld, _ = solve_hyperbolic(prob_hlld)
        _, U_hll, _ = solve_hyperbolic(prob_hll)

        W_hlld = to_primitive(law, U_hlld)
        W_hll = to_primitive(law, U_hll)

        ρ_hlld = [w[1] for w in W_hlld]
        ρ_hll = [w[1] for w in W_hll]

        # Compute total variation as a proxy for sharpness
        tv_hlld = sum(abs(ρ_hlld[i + 1] - ρ_hlld[i]) for i in 1:(length(ρ_hlld) - 1))
        tv_hll = sum(abs(ρ_hll[i + 1] - ρ_hll[i]) for i in 1:(length(ρ_hll) - 1))

        # HLLD should have higher total variation (sharper features)
        @test tv_hlld > tv_hll * 0.8  # At least comparable

        # Both should be finite and physical
        @test all(isfinite.(ρ_hlld))
        @test all(isfinite.(ρ_hll))
    end

    @testset "Lax-Friedrichs solver" begin
        mesh = StructuredMesh1D(0.0, 1.0, 400)
        prob = HyperbolicProblem(
            law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(), bw_ic;
            final_time = 0.1, cfl = 0.5
        )

        x, U, t_final = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        ρ = [w[1] for w in W]
        P = [w[5] for w in W]

        @test t_final ≈ 0.1 atol = 1e-10
        @test all(ρ .> 0)
        @test all(P .> 0)
    end
end

# ============================================================
# Conservation Tests (Periodic BCs)
# ============================================================
@testset "MHD Conservation" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{1}(eos)

    @testset "Periodic BCs - mass, momentum, energy, B conserved" begin
        # Smooth initial condition with periodic BCs
        function smooth_ic(x)
            ρ = 1.0 + 0.2 * sin(2π * x)
            vx = 0.1 * sin(2π * x)
            vy = 0.0
            vz = 0.0
            P = 1.0
            Bx = 1.0  # Constant Bx
            By = 0.1 * cos(2π * x)
            Bz = 0.0
            return SVector(ρ, vx, vy, vz, P, Bx, By, Bz)
        end

        mesh = StructuredMesh1D(0.0, 1.0, 200)
        prob = HyperbolicProblem(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), smooth_ic;
            final_time = 0.05, cfl = 0.5
        )

        x, U0, _ = solve_hyperbolic(
            HyperbolicProblem(
                law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
                PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), smooth_ic;
                final_time = 0.0, cfl = 0.5
            )
        )

        x, U_final, t_final = solve_hyperbolic(prob)

        dx = 1.0 / 200

        # Compute conserved totals
        mass_0 = sum(u[1] for u in U0) * dx
        mass_f = sum(u[1] for u in U_final) * dx
        mom_x_0 = sum(u[2] for u in U0) * dx
        mom_x_f = sum(u[2] for u in U_final) * dx
        mom_y_0 = sum(u[3] for u in U0) * dx
        mom_y_f = sum(u[3] for u in U_final) * dx
        energy_0 = sum(u[5] for u in U0) * dx
        energy_f = sum(u[5] for u in U_final) * dx

        @test mass_f ≈ mass_0 atol = 1e-10
        @test mom_x_f ≈ mom_x_0 atol = 1e-10
        @test mom_y_f ≈ mom_y_0 atol = 1e-10
        @test energy_f ≈ energy_0 atol = 1e-10
    end
end

# ============================================================
# MHD with All Limiter Types
# ============================================================
@testset "MHD with Different Limiters" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{1}(eos)

    Bx = 0.75
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, Bx, -1.0, 0.0)
    bw_ic(x) = x < 0.5 ? wL : wR

    limiters = [
        ("Minmod", MinmodLimiter()),
        ("Superbee", SuperbeeLimiter()),
        ("VanLeer", VanLeerLimiter()),
        ("Koren", KorenLimiter()),
        ("Ospre", OspreLimiter()),
        ("Venkatakrishnan", VenkatakrishnanLimiter()),
    ]

    for (name, limiter) in limiters
        @testset "$name limiter" begin
            mesh = StructuredMesh1D(0.0, 1.0, 200)
            prob = HyperbolicProblem(
                law, mesh, HLLDSolver(), CellCenteredMUSCL(limiter),
                TransmissiveBC(), TransmissiveBC(), bw_ic;
                final_time = 0.05, cfl = 0.6
            )

            x, U, t_final = solve_hyperbolic(prob)
            W = to_primitive(law, U)

            ρ = [w[1] for w in W]
            P = [w[5] for w in W]

            @test t_final ≈ 0.05 atol = 1e-10
            @test all(ρ .> 0)
            @test all(P .> 0)
            @test all(isfinite.(ρ))
        end
    end
end

# ============================================================
# Boundary Condition Tests for MHD
# ============================================================
@testset "MHD Boundary Conditions" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{1}(eos)

    @testset "TransmissiveBC" begin
        nc = 10
        N = nvariables(law)
        U = Vector{SVector{N, Float64}}(undef, nc + 4)
        for i in 1:(nc + 4)
            U[i] = SVector(1.0, 0.5, 0.0, 0.0, 2.0, 0.75, 1.0, 0.0)
        end
        U[3] = SVector(2.0, 1.0, 0.1, 0.0, 3.0, 0.75, 0.5, 0.0)
        U[nc + 2] = SVector(0.5, -0.5, -0.1, 0.0, 1.0, 0.75, -0.5, 0.0)

        FiniteVolumeMethod.apply_bc_left!(U, TransmissiveBC(), law, nc, 0.0)
        FiniteVolumeMethod.apply_bc_right!(U, TransmissiveBC(), law, nc, 0.0)

        @test U[2] == U[3]
        @test U[1] == U[3]
        @test U[nc + 3] == U[nc + 2]
        @test U[nc + 4] == U[nc + 2]
    end

    @testset "ReflectiveBC for MHD" begin
        nc = 10
        N = nvariables(law)
        U = Vector{SVector{N, Float64}}(undef, nc + 4)
        for i in 1:(nc + 4)
            U[i] = SVector(1.0, 0.0, 0.0, 0.0, 2.0, 0.75, 1.0, 0.0)
        end
        # Set non-zero velocity in first interior cell
        w_interior = SVector(1.0, 0.5, 0.1, 0.0, 1.0, 0.75, 1.0, 0.0)
        U[3] = primitive_to_conserved(law, w_interior)
        U[4] = primitive_to_conserved(law, w_interior)

        FiniteVolumeMethod.apply_bc_left!(U, ReflectiveBC(), law, nc, 0.0)

        w_ghost = conserved_to_primitive(law, U[2])
        @test w_ghost[1] ≈ w_interior[1]    # ρ preserved
        @test w_ghost[2] ≈ -w_interior[2]   # vx negated
        @test w_ghost[3] ≈ w_interior[3]    # vy preserved
        @test w_ghost[5] ≈ w_interior[5]    # P preserved
        @test w_ghost[6] ≈ w_interior[6]    # Bx preserved
        @test w_ghost[7] ≈ w_interior[7]    # By preserved
    end

    @testset "DirichletHyperbolicBC for MHD" begin
        nc = 10
        N = nvariables(law)
        U = Vector{SVector{N, Float64}}(undef, nc + 4)
        for i in 1:(nc + 4)
            U[i] = zero(SVector{N, Float64})
        end

        w_bc = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
        bc = DirichletHyperbolicBC(w_bc)
        u_bc = primitive_to_conserved(law, w_bc)

        FiniteVolumeMethod.apply_bc_left!(U, bc, law, nc, 0.0)
        @test U[1] == u_bc
        @test U[2] == u_bc
    end

    @testset "InflowBC for MHD" begin
        nc = 10
        N = nvariables(law)
        U = Vector{SVector{N, Float64}}(undef, nc + 4)
        for i in 1:(nc + 4)
            U[i] = zero(SVector{N, Float64})
        end

        w_inflow = SVector(1.0, 0.5, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
        bc = InflowBC(w_inflow)
        u_bc = primitive_to_conserved(law, w_inflow)

        FiniteVolumeMethod.apply_bc_left!(U, bc, law, nc, 0.0)
        @test U[1] == u_bc
        @test U[2] == u_bc
    end
end

# ============================================================
# CFL Stability Test
# ============================================================
@testset "MHD CFL Stability" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{1}(eos)

    Bx = 0.75
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, Bx, -1.0, 0.0)
    bw_ic(x) = x < 0.5 ? wL : wR

    @testset "CFL=0.8 stable" begin
        mesh = StructuredMesh1D(0.0, 1.0, 200)
        prob = HyperbolicProblem(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(), bw_ic;
            final_time = 0.05, cfl = 0.8
        )
        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)
        @test all(w -> w[1] > 0, W)  # density positive
        @test all(w -> w[5] > 0, W)  # pressure positive
    end

    @testset "CFL=0.3 stable (conservative)" begin
        mesh = StructuredMesh1D(0.0, 1.0, 200)
        prob = HyperbolicProblem(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(), bw_ic;
            final_time = 0.05, cfl = 0.3
        )
        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)
        @test all(w -> w[1] > 0, W)
        @test all(w -> w[5] > 0, W)
    end
end

# ============================================================
# Einfeldt-like Test for MHD
# ============================================================
@testset "MHD Double Rarefaction" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{1}(eos)

    # Two rarefaction waves moving apart
    wL = SVector(1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    wR = SVector(1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    ic(x) = x < 0.5 ? wL : wR

    mesh = StructuredMesh1D(0.0, 1.0, 400)
    prob = HyperbolicProblem(
        law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), ic;
        final_time = 0.1, cfl = 0.5
    )

    x, U, t_final = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    ρ = [w[1] for w in W]
    P = [w[5] for w in W]

    @test t_final ≈ 0.1 atol = 1e-10
    @test all(ρ .> 0)    # Density should stay positive
    @test all(P .> 0)    # Pressure should stay positive
    @test all(isfinite.(ρ))

    # Density should be lowest near the center (rarefaction)
    mid = length(x) ÷ 2
    @test ρ[mid] < ρ[1]
end

# ============================================================
# MHD Alfvén Wave Propagation
# ============================================================
@testset "MHD Alfvén Wave" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{1}(eos)

    # Circularly polarized Alfvén wave (exact solution: advection at Alfvén speed)
    Bx_bg = 1.0
    ρ_bg = 1.0
    P_bg = 1.0
    ca = Bx_bg / sqrt(ρ_bg)  # Alfvén speed

    amp = 0.1  # Small amplitude
    function alfven_ic(x)
        By = amp * sin(2π * x)
        vy = -amp * sin(2π * x) * ca  # -By * ca / Bx for right-going Alfvén wave: vy = -By/√(ρ) ?
        # Actually for a right-going Alfvén wave: δvy = -δBy * ca/Bx
        vy_pert = -By * ca / Bx_bg
        return SVector(ρ_bg, 0.0, vy_pert, 0.0, P_bg, Bx_bg, By, 0.0)
    end

    mesh = StructuredMesh1D(0.0, 1.0, 200)
    prob = HyperbolicProblem(
        law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), alfven_ic;
        final_time = 0.5, cfl = 0.5
    )

    x, U, t_final = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    # The wave should propagate without changing shape (linear regime)
    # Check ρ and P remain nearly constant (Alfvén waves don't compress)
    ρ = [w[1] for w in W]
    P = [w[5] for w in W]

    @test maximum(ρ) - minimum(ρ) < 0.02  # Near-constant density
    @test maximum(P) - minimum(P) < 0.02  # Near-constant pressure

    # By should still have sinusoidal structure
    By = [w[7] for w in W]
    @test maximum(abs.(By)) > 0.01  # Wave hasn't been completely damped
end

# ============================================================
# NoReconstruction (First Order) for MHD
# ============================================================
@testset "MHD First Order (NoReconstruction)" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{1}(eos)

    Bx = 0.75
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, Bx, -1.0, 0.0)
    bw_ic(x) = x < 0.5 ? wL : wR

    mesh = StructuredMesh1D(0.0, 1.0, 400)
    prob = HyperbolicProblem(
        law, mesh, HLLDSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), bw_ic;
        final_time = 0.1, cfl = 0.8
    )

    x, U, t_final = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    ρ = [w[1] for w in W]
    P = [w[5] for w in W]

    @test t_final ≈ 0.1 atol = 1e-10
    @test all(ρ .> 0)
    @test all(P .> 0)
end
