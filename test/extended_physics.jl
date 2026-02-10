using FiniteVolumeMethod
using Test
using StaticArrays

# ============================================================
# 1. ShallowWaterEquations
# ============================================================
@testset "ShallowWaterEquations" begin

    @testset "Construction and nvariables" begin
        @testset "1D" begin
            law = ShallowWaterEquations{1}()
            @test law.g == 9.81
            @test nvariables(law) == 2

            law2 = ShallowWaterEquations{1}(g = 1.0)
            @test law2.g == 1.0
        end

        @testset "2D" begin
            law = ShallowWaterEquations{2}()
            @test law.g == 9.81
            @test nvariables(law) == 3

            law2 = ShallowWaterEquations{2}(g = 10.0)
            @test law2.g == 10.0
        end
    end

    @testset "Conserved-Primitive round-trip" begin
        @testset "1D" begin
            law = ShallowWaterEquations{1}(g = 9.81)
            primitives = [
                SVector(1.0, 0.0),
                SVector(2.0, 1.5),
                SVector(0.5, -3.0),
                SVector(10.0, 0.1),
            ]
            for w in primitives
                u = primitive_to_conserved(law, w)
                w2 = conserved_to_primitive(law, u)
                @test w2 ≈ w atol = 1.0e-14
            end
        end

        @testset "2D" begin
            law = ShallowWaterEquations{2}(g = 9.81)
            primitives = [
                SVector(1.0, 0.0, 0.0),
                SVector(2.0, 1.5, -0.5),
                SVector(0.5, -3.0, 2.0),
                SVector(10.0, 0.1, 0.3),
            ]
            for w in primitives
                u = primitive_to_conserved(law, w)
                w2 = conserved_to_primitive(law, u)
                @test w2 ≈ w atol = 1.0e-14
            end
        end
    end

    @testset "Physical flux correctness" begin
        @testset "1D" begin
            law = ShallowWaterEquations{1}(g = 9.81)
            # w = [h, u]
            w = SVector(2.0, 3.0)
            f = physical_flux(law, w, 1)
            # F = [h*u, h*u^2 + 0.5*g*h^2]
            @test f[1] ≈ 2.0 * 3.0
            @test f[2] ≈ 2.0 * 9.0 + 0.5 * 9.81 * 4.0
        end

        @testset "2D x-direction" begin
            law = ShallowWaterEquations{2}(g = 9.81)
            w = SVector(2.0, 3.0, 1.0)
            f = physical_flux(law, w, 1)
            # Fx = [h*vx, h*vx^2 + 0.5*g*h^2, h*vx*vy]
            @test f[1] ≈ 2.0 * 3.0
            @test f[2] ≈ 2.0 * 9.0 + 0.5 * 9.81 * 4.0
            @test f[3] ≈ 2.0 * 3.0 * 1.0
        end

        @testset "2D y-direction" begin
            law = ShallowWaterEquations{2}(g = 9.81)
            w = SVector(2.0, 3.0, 1.0)
            f = physical_flux(law, w, 2)
            # Fy = [h*vy, h*vx*vy, h*vy^2 + 0.5*g*h^2]
            @test f[1] ≈ 2.0 * 1.0
            @test f[2] ≈ 2.0 * 3.0 * 1.0
            @test f[3] ≈ 2.0 * 1.0 + 0.5 * 9.81 * 4.0
        end
    end

    @testset "Wave speeds and max_wave_speed" begin
        @testset "1D" begin
            law = ShallowWaterEquations{1}(g = 9.81)
            w = SVector(4.0, 1.0)  # h=4, u=1
            c = sqrt(9.81 * 4.0)
            lmin, lmax = wave_speeds(law, w, 1)
            @test lmin ≈ 1.0 - c
            @test lmax ≈ 1.0 + c
            @test max_wave_speed(law, w, 1) ≈ abs(1.0) + c
        end

        @testset "2D" begin
            law = ShallowWaterEquations{2}(g = 9.81)
            w = SVector(4.0, 1.0, -2.0)  # h=4, vx=1, vy=-2
            c = sqrt(9.81 * 4.0)
            lmin_x, lmax_x = wave_speeds(law, w, 1)
            @test lmin_x ≈ 1.0 - c
            @test lmax_x ≈ 1.0 + c

            lmin_y, lmax_y = wave_speeds(law, w, 2)
            @test lmin_y ≈ -2.0 - c
            @test lmax_y ≈ -2.0 + c

            @test max_wave_speed(law, w, 1) ≈ abs(1.0) + c
            @test max_wave_speed(law, w, 2) ≈ abs(-2.0) + c
        end
    end

    @testset "HLLC solver" begin
        @testset "1D consistency: F(u, u) = F(u)" begin
            law = ShallowWaterEquations{1}(g = 9.81)
            solver = HLLCSolver()
            states = [
                SVector(1.0, 0.0),
                SVector(2.0, 1.5),
                SVector(0.5, -3.0),
            ]
            for w in states
                f_hllc = solve_riemann(solver, law, w, w, 1)
                f_exact = physical_flux(law, w, 1)
                @test f_hllc ≈ f_exact atol = 1.0e-12
            end
        end

        @testset "1D symmetry" begin
            law = ShallowWaterEquations{1}(g = 9.81)
            solver = HLLCSolver()
            wL = SVector(1.0, 1.0)
            wR = SVector(0.5, -0.5)
            # Reverse: negate velocity for mirror
            wL_m = SVector(wR[1], -wR[2])
            wR_m = SVector(wL[1], -wL[2])

            f1 = solve_riemann(solver, law, wL, wR, 1)
            f2 = solve_riemann(solver, law, wL_m, wR_m, 1)

            # Mass flux should have opposite sign, momentum flux should be same
            @test f1[1] ≈ -f2[1] atol = 1.0e-12
            @test f1[2] ≈ f2[2] atol = 1.0e-12
        end

        @testset "2D consistency: F(u, u) = F(u)" begin
            law = ShallowWaterEquations{2}(g = 9.81)
            solver = HLLCSolver()
            states = [
                SVector(1.0, 0.0, 0.0),
                SVector(2.0, 1.5, -0.5),
                SVector(0.5, -3.0, 2.0),
            ]
            for w in states
                for dir in [1, 2]
                    f_hllc = solve_riemann(solver, law, w, w, dir)
                    f_exact = physical_flux(law, w, dir)
                    @test f_hllc ≈ f_exact atol = 1.0e-12
                end
            end
        end
    end

    @testset "1D dam break" begin
        law = ShallowWaterEquations{1}(g = 1.0)
        mesh = StructuredMesh1D(0.0, 1.0, 200)

        wL = SVector(2.0, 0.0)   # [h, u] - deep water
        wR = SVector(1.0, 0.0)   # shallow water

        prob = HyperbolicProblem(
            law, mesh, HLLCSolver(), CellCenteredMUSCL(),
            TransmissiveBC(), TransmissiveBC(),
            x -> x < 0.5 ? wL : wR;
            final_time = 0.15, cfl = 0.4
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t ≈ 0.15 atol = 1.0e-10

        # Water depth should be positive everywhere
        for i in eachindex(W)
            @test W[i][1] > 0.0
        end

        # Dam break structure: left rarefaction + right shock
        # At t=0.15, the original discontinuity at x=0.5 should have produced
        # a wave structure where leftmost cells are still at h=2
        # and rightmost cells still at h=1
        @test W[1][1] ≈ 2.0 atol = 0.01
        @test W[end][1] ≈ 1.0 atol = 0.01

        # Intermediate region should exist with depth between 1 and 2
        mid = length(W) ÷ 2
        @test 1.0 < W[mid][1] < 2.0
    end

    @testset "Bottom topography" begin
        @testset "BottomTopography construction" begin
            topo = BottomTopography(x -> 0.1 * sin(x))
            @test topo.b(0.0) ≈ 0.0
            @test topo.b(pi / 2) ≈ 0.1
        end

        @testset "Lake-at-rest: source balances flux gradient" begin
            # For lake at rest: h + b = const, u = 0
            # The topography source should balance the pressure gradient
            law = ShallowWaterEquations{1}(g = 9.81)

            # Water surface at elevation H=1.0
            H = 1.0
            b_L = 0.2
            b_R = 0.3
            dx = 0.1
            h = H - 0.5 * (b_L + b_R)  # water depth at cell center

            src = topography_source_1d(law, h, b_L, b_R, dx)

            # Mass source should be zero
            @test src[1] ≈ 0.0 atol = 1.0e-15

            # Momentum source: -g*h*(b_R - b_L)/dx
            expected_mom = -9.81 * h * (b_R - b_L) / dx
            @test src[2] ≈ expected_mom atol = 1.0e-12
        end

        @testset "Zero topography gives zero source" begin
            law = ShallowWaterEquations{1}(g = 9.81)
            src = topography_source_1d(law, 1.0, 0.5, 0.5, 0.1)
            @test src[1] ≈ 0.0 atol = 1.0e-15
            @test src[2] ≈ 0.0 atol = 1.0e-15
        end
    end
end

# ============================================================
# 2. SRHydroEquations
# ============================================================
@testset "SRHydroEquations" begin

    @testset "Construction and nvariables" begin
        eos = IdealGasEOS(5.0 / 3.0)

        @testset "1D" begin
            law = SRHydroEquations{1}(eos)
            @test nvariables(law) == 3
            @test law.eos === eos
            @test law.con2prim_tol == 1.0e-12
            @test law.con2prim_maxiter == 50
        end

        @testset "2D" begin
            law = SRHydroEquations{2}(eos)
            @test nvariables(law) == 4
        end

        @testset "Custom con2prim params" begin
            law = SRHydroEquations{1}(eos; con2prim_tol = 1.0e-10, con2prim_maxiter = 100)
            @test law.con2prim_tol == 1.0e-10
            @test law.con2prim_maxiter == 100
        end
    end

    @testset "Conserved-Primitive round-trip 1D" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = SRHydroEquations{1}(eos)

        @testset "Static states (v=0)" begin
            states = [
                SVector(1.0, 0.0, 1.0),
                SVector(0.125, 0.0, 0.1),
                SVector(10.0, 0.0, 5.0),
            ]
            for w in states
                u = primitive_to_conserved(law, w)
                w2 = conserved_to_primitive(law, u)
                @test w2 ≈ w atol = 1.0e-10
            end
        end

        @testset "Mildly relativistic (v ~ 0.1-0.5)" begin
            states = [
                SVector(1.0, 0.1, 1.0),
                SVector(1.0, 0.3, 2.0),
                SVector(0.5, -0.4, 3.0),
                SVector(1.0, 0.5, 1.0),
            ]
            for w in states
                u = primitive_to_conserved(law, w)
                w2 = conserved_to_primitive(law, u)
                @test w2 ≈ w atol = 1.0e-9
            end
        end

        @testset "Moderately relativistic (v ~ 0.7-0.9)" begin
            states = [
                SVector(1.0, 0.7, 1.0),
                SVector(1.0, -0.8, 2.0),
                SVector(0.5, 0.9, 3.0),
            ]
            for w in states
                u = primitive_to_conserved(law, w)
                w2 = conserved_to_primitive(law, u)
                @test w2 ≈ w atol = 1.0e-8
            end
        end
    end

    @testset "Conserved-Primitive round-trip 2D" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = SRHydroEquations{2}(eos)

        states = [
            SVector(1.0, 0.0, 0.0, 1.0),
            SVector(1.0, 0.3, 0.2, 2.0),
            SVector(0.5, -0.4, 0.1, 3.0),
            SVector(1.0, 0.5, -0.3, 1.0),
        ]
        for w in states
            u = primitive_to_conserved(law, w)
            w2 = conserved_to_primitive(law, u)
            @test w2 ≈ w atol = 1.0e-9
        end
    end

    @testset "Con2prim convergence" begin
        eos = IdealGasEOS(5.0 / 3.0)

        @testset "1D" begin
            w = SVector(1.0, 0.5, 1.0)
            u = primitive_to_conserved(SRHydroEquations{1}(eos), w)
            w_rec, result = srhydro_con2prim(eos, u, 1.0e-12, 50)
            @test result.converged == true
            @test result.iterations < 50
            @test result.residual < 1.0e-10
        end

        @testset "2D" begin
            w = SVector(1.0, 0.3, 0.2, 2.0)
            u = primitive_to_conserved(SRHydroEquations{2}(eos), w)
            w_rec, result = srhydro_con2prim(eos, u, 1.0e-12, 50)
            @test result.converged == true
            @test result.iterations < 50
            @test result.residual < 1.0e-10
        end
    end

    @testset "Physical flux correctness 1D" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = SRHydroEquations{1}(eos)

        # Static state: v=0
        w_static = SVector(1.0, 0.0, 1.0)
        f = physical_flux(law, w_static, 1)
        # F = [D*vx, Sx*vx + P, (tau+P)*vx] => for v=0, F=[0, P, 0]
        @test f[1] ≈ 0.0 atol = 1.0e-15
        @test f[2] ≈ 1.0 atol = 1.0e-14   # P = 1.0
        @test f[3] ≈ 0.0 atol = 1.0e-15
    end

    @testset "Physical flux correctness 2D" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = SRHydroEquations{2}(eos)

        # Static state: all v=0
        w_static = SVector(1.0, 0.0, 0.0, 1.0)
        fx = physical_flux(law, w_static, 1)
        fy = physical_flux(law, w_static, 2)
        # For v=0: Fx = [0, P, 0, 0], Fy = [0, 0, P, 0]
        @test fx[1] ≈ 0.0 atol = 1.0e-15
        @test fx[2] ≈ 1.0 atol = 1.0e-14
        @test fx[3] ≈ 0.0 atol = 1.0e-15
        @test fx[4] ≈ 0.0 atol = 1.0e-15

        @test fy[1] ≈ 0.0 atol = 1.0e-15
        @test fy[2] ≈ 0.0 atol = 1.0e-15
        @test fy[3] ≈ 1.0 atol = 1.0e-14
        @test fy[4] ≈ 0.0 atol = 1.0e-15
    end

    @testset "Relativistic wave speeds" begin
        eos = IdealGasEOS(5.0 / 3.0)

        @testset "1D" begin
            law = SRHydroEquations{1}(eos)
            w = SVector(1.0, 0.0, 1.0)
            lmin, lmax = wave_speeds(law, w, 1)
            # For v=0: lambda = +/- cs where cs = sqrt(gamma*P/(rho*h))
            @test lmin < 0
            @test lmax > 0
            @test lmin ≈ -lmax atol = 1.0e-14  # symmetric for v=0

            # Wave speeds should be subluminal
            @test abs(lmin) < 1.0
            @test abs(lmax) < 1.0

            smax = max_wave_speed(law, w, 1)
            @test smax ≈ lmax atol = 1.0e-14
        end

        @testset "2D" begin
            law = SRHydroEquations{2}(eos)
            w = SVector(1.0, 0.5, 0.0, 1.0)
            lmin, lmax = wave_speeds(law, w, 1)
            # With v > 0, right-going wave should be faster
            @test lmax > abs(lmin)
            # Both subluminal
            @test abs(lmin) < 1.0
            @test abs(lmax) < 1.0
        end

        @testset "Ultra-relativistic speeds are subluminal" begin
            law = SRHydroEquations{1}(eos)
            w = SVector(1.0, 0.99, 10.0)  # v close to c
            lmin, lmax = wave_speeds(law, w, 1)
            @test abs(lmin) < 1.0
            @test abs(lmax) < 1.0
        end
    end

    @testset "1D relativistic shock tube (v=0.5c)" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = SRHydroEquations{1}(eos)
        mesh = StructuredMesh1D(0.0, 1.0, 200)

        # Mildly relativistic Sod-like problem
        wL = SVector(1.0, 0.0, 1.0)     # [rho, vx, P]
        wR = SVector(0.125, 0.0, 0.1)

        prob = HyperbolicProblem(
            law, mesh, HLLSolver(), NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(),
            x -> x < 0.5 ? wL : wR;
            final_time = 0.2, cfl = 0.3
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t ≈ 0.2 atol = 1.0e-10

        # All densities and pressures positive
        for i in eachindex(W)
            @test W[i][1] > 0.0  # rho > 0
            @test W[i][3] > 0.0  # P > 0
        end

        # Shock propagation: left boundary should be near original state
        @test W[1][1] ≈ 1.0 atol = 0.01
        @test W[end][1] ≈ 0.125 atol = 0.01

        # All velocities subluminal
        for i in eachindex(W)
            @test abs(W[i][2]) < 1.0
        end
    end

    @testset "Non-relativistic limit approximates Euler" begin
        # In the non-relativistic limit (v << c), SR hydro should
        # match Euler equations
        eos = IdealGasEOS(1.4)
        law_sr = SRHydroEquations{1}(eos)
        law_euler = EulerEquations{1}(eos)

        # Very non-relativistic state
        rho, vx, P = 1.0, 0.001, 1.0
        w_sr = SVector(rho, vx, P)
        w_euler = SVector(rho, vx, P)

        f_sr = physical_flux(law_sr, w_sr, 1)
        f_euler = physical_flux(law_euler, w_euler, 1)

        # Mass flux: D*vx ≈ rho*vx for small v (W ≈ 1)
        @test f_sr[1] ≈ f_euler[1] atol = 1.0e-4
        # Momentum flux: approximately same
        @test f_sr[2] ≈ f_euler[2] atol = 1.0e-4
        # Energy flux: approximately same
        @test f_sr[3] ≈ f_euler[3] atol = 1.0e-2
    end
end

# ============================================================
# 3. ResistiveMHDEquations
# ============================================================
@testset "ResistiveMHDEquations" begin

    @testset "Construction and nvariables" begin
        eos = IdealGasEOS(5.0 / 3.0)

        @testset "1D" begin
            law = ResistiveMHDEquations{1}(eos; eta = 0.01)
            @test nvariables(law) == 8
            @test law.eos === eos
            @test law.eta == 0.01
        end

        @testset "2D" begin
            law = ResistiveMHDEquations{2}(eos; eta = 0.05)
            @test nvariables(law) == 8
            @test law.eta == 0.05
        end

        @testset "Default eta=0" begin
            law = ResistiveMHDEquations{1}(eos)
            @test law.eta == 0.0
        end
    end

    @testset "Delegation matches IdealMHDEquations" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law_rmhd = ResistiveMHDEquations{1}(eos; eta = 0.01)
        law_mhd = IdealMHDEquations{1}(eos)

        w = SVector(1.0, 0.3, 0.0, 0.0, 1.0, 0.5, 0.8, 0.0)

        @test conserved_to_primitive(law_rmhd, primitive_to_conserved(law_rmhd, w)) ≈
            conserved_to_primitive(law_mhd, primitive_to_conserved(law_mhd, w)) atol = 1.0e-14

        @test physical_flux(law_rmhd, w, 1) ≈ physical_flux(law_mhd, w, 1) atol = 1.0e-14

        @test max_wave_speed(law_rmhd, w, 1) ≈ max_wave_speed(law_mhd, w, 1) atol = 1.0e-14

        lmin_r, lmax_r = wave_speeds(law_rmhd, w, 1)
        lmin_m, lmax_m = wave_speeds(law_mhd, w, 1)
        @test lmin_r ≈ lmin_m atol = 1.0e-14
        @test lmax_r ≈ lmax_m atol = 1.0e-14
    end

    @testset "Resistive flux: diffusion direction" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = ResistiveMHDEquations{1}(eos; eta = 0.1)

        # B-field difference: By increases from left to right
        uL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0)
        uR = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.5, 0.0)
        dx = 0.1

        f = resistive_flux_x(law, uL, uR, dx)

        # No mass, momentum corrections
        @test f[1] ≈ 0.0 atol = 1.0e-15
        @test f[2] ≈ 0.0 atol = 1.0e-15
        @test f[3] ≈ 0.0 atol = 1.0e-15
        @test f[4] ≈ 0.0 atol = 1.0e-15
        @test f[6] ≈ 0.0 atol = 1.0e-15  # Bx not diffused in x

        # By diffusion: -eta * dBy/dx < 0 (diffuses By toward mean)
        dBy = uR[7] - uL[7]
        @test f[7] ≈ -law.eta * dBy / dx atol = 1.0e-14
    end

    @testset "Ohmic heating" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = ResistiveMHDEquations{1}(eos; eta = 0.05)

        J_sq = 4.0
        @test ohmic_heating(law, J_sq) ≈ 0.05 * 4.0 atol = 1.0e-15
        @test ohmic_heating(law, 0.0) ≈ 0.0 atol = 1.0e-15
    end

    @testset "Resistive dt" begin
        eos = IdealGasEOS(5.0 / 3.0)

        @testset "1D" begin
            law = ResistiveMHDEquations{1}(eos; eta = 0.1)
            dx = 0.01
            dt = resistive_dt(law, dx)
            @test dt ≈ 0.5 * dx^2 / 0.1 atol = 1.0e-15
        end

        @testset "2D" begin
            law = ResistiveMHDEquations{2}(eos; eta = 0.1)
            dx, dy = 0.01, 0.02
            dt = resistive_dt(law, dx, dy)
            @test dt ≈ 0.5 / (0.1 * (1 / dx^2 + 1 / dy^2)) atol = 1.0e-15
        end
    end

    @testset "HLLD forwarding" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law_rmhd = ResistiveMHDEquations{1}(eos; eta = 0.01)
        law_mhd = IdealMHDEquations{1}(eos)
        solver = HLLDSolver()

        wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
        wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0)

        f_rmhd = solve_riemann(solver, law_rmhd, wL, wR, 1)
        f_mhd = solve_riemann(solver, law_mhd, wL, wR, 1)

        @test f_rmhd ≈ f_mhd atol = 1.0e-14
    end

    @testset "eta=0 gives zero resistive flux" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = ResistiveMHDEquations{1}(eos; eta = 0.0)

        uL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.3)
        uR = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.5, -0.2)
        dx = 0.1

        f = resistive_flux_x(law, uL, uR, dx)
        for k in 1:8
            @test f[k] ≈ 0.0 atol = 1.0e-15
        end
    end

    @testset "Resistive flux y-direction" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = ResistiveMHDEquations{2}(eos; eta = 0.1)

        uB = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0)
        uT = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 1.5, 0.0, 0.0)
        dy = 0.1

        f = resistive_flux_y(law, uB, uT, dy)

        # Bx diffusion in y-direction
        dBx = uT[6] - uB[6]
        @test f[6] ≈ -law.eta * dBx / dy atol = 1.0e-14
        # No By correction
        @test f[7] ≈ 0.0 atol = 1.0e-15
    end
end

# ============================================================
# 4. HallMHDEquations
# ============================================================
@testset "HallMHDEquations" begin

    @testset "Construction and nvariables" begin
        eos = IdealGasEOS(5.0 / 3.0)

        @testset "1D" begin
            law = HallMHDEquations{1}(eos; di = 0.5, eta = 0.01)
            @test nvariables(law) == 8
            @test law.eos === eos
            @test law.di == 0.5
            @test law.eta == 0.01
        end

        @testset "2D" begin
            law = HallMHDEquations{2}(eos)
            @test nvariables(law) == 8
            @test law.di == 1.0
            @test law.eta == 0.0
        end
    end

    @testset "Delegation matches IdealMHDEquations" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law_hall = HallMHDEquations{1}(eos; di = 1.0)
        law_mhd = IdealMHDEquations{1}(eos)

        w = SVector(1.0, 0.3, 0.0, 0.0, 1.0, 0.5, 0.8, 0.0)

        @test conserved_to_primitive(law_hall, primitive_to_conserved(law_hall, w)) ≈
            conserved_to_primitive(law_mhd, primitive_to_conserved(law_mhd, w)) atol = 1.0e-14

        @test physical_flux(law_hall, w, 1) ≈ physical_flux(law_mhd, w, 1) atol = 1.0e-14
        @test max_wave_speed(law_hall, w, 1) ≈ max_wave_speed(law_mhd, w, 1) atol = 1.0e-14

        lmin_h, lmax_h = wave_speeds(law_hall, w, 1)
        lmin_m, lmax_m = wave_speeds(law_mhd, w, 1)
        @test lmin_h ≈ lmin_m atol = 1.0e-14
        @test lmax_h ≈ lmax_m atol = 1.0e-14
    end

    @testset "Whistler speed formula" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = HallMHDEquations{1}(eos; di = 2.0)

        rho = 4.0
        B_mag = 3.0
        dx = 0.1

        cw = whistler_speed(law, rho, B_mag, dx)
        # c_w = di * |B| / (sqrt(rho) * dx)
        @test cw ≈ 2.0 * 3.0 / (sqrt(4.0) * 0.1) atol = 1.0e-14
        @test cw ≈ 30.0 atol = 1.0e-14
    end

    @testset "Hall flux: non-zero only in B and energy" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = HallMHDEquations{1}(eos; di = 1.0, eta = 0.0)

        uL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.3, 0.2)
        uR = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.8, -0.1)
        dx = 0.1

        f = hall_flux_x(law, uL, uR, dx)

        # Mass (f[1]) and momentum (f[2], f[3], f[4]) should be zero
        @test f[1] ≈ 0.0 atol = 1.0e-15
        @test f[2] ≈ 0.0 atol = 1.0e-15
        @test f[3] ≈ 0.0 atol = 1.0e-15
        @test f[4] ≈ 0.0 atol = 1.0e-15

        # Bx should be zero (no correction to normal B component in 1D)
        @test f[6] ≈ 0.0 atol = 1.0e-15

        # By (f[7]) and Bz (f[8]) should be non-zero
        # because dBy/dx != 0 and dBz/dx != 0
        @test abs(f[7]) > 1.0e-10
        @test abs(f[8]) > 1.0e-10

        # Energy (f[5]) should be non-zero (Poynting flux correction)
        @test abs(f[5]) > 1.0e-10
    end

    @testset "Hall dt: inversely proportional to di and B" begin
        eos = IdealGasEOS(5.0 / 3.0)

        # Create two laws with different di
        law1 = HallMHDEquations{1}(eos; di = 1.0)
        law2 = HallMHDEquations{1}(eos; di = 2.0)

        # Uniform state with non-zero B
        w = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
        u = primitive_to_conserved(law1, w)

        nc = 10
        dx = 0.1
        U = Vector{SVector{8, Float64}}(undef, nc + 4)
        for i in 1:(nc + 4)
            U[i] = u
        end

        dt1 = hall_dt(law1, U, dx, nc)
        dt2 = hall_dt(law2, U, dx, nc)

        # Larger di => smaller dt (faster whistler waves)
        @test dt2 < dt1
        # Specifically dt should scale as 1/di
        @test dt1 / dt2 ≈ 2.0 atol = 1.0e-10
    end

    @testset "di=0 gives zero Hall flux" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = HallMHDEquations{1}(eos; di = 0.0, eta = 0.0)

        uL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.3, 0.2)
        uR = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.8, -0.1)
        dx = 0.1

        f = hall_flux_x(law, uL, uR, dx)
        for k in 1:8
            @test f[k] ≈ 0.0 atol = 1.0e-15
        end
    end

    @testset "HLLD forwarding" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law_hall = HallMHDEquations{1}(eos; di = 1.0)
        law_mhd = IdealMHDEquations{1}(eos)
        solver = HLLDSolver()

        wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
        wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0)

        f_hall = solve_riemann(solver, law_hall, wL, wR, 1)
        f_mhd = solve_riemann(solver, law_mhd, wL, wR, 1)

        @test f_hall ≈ f_mhd atol = 1.0e-14
    end

    @testset "Hall flux y-direction" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = HallMHDEquations{2}(eos; di = 1.0, eta = 0.0)

        uB = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.3, 0.5, 0.2)
        uT = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.8, 0.5, -0.1)
        dy = 0.1

        f = hall_flux_y(law, uB, uT, dy)

        # Mass and momentum are zero
        @test f[1] ≈ 0.0 atol = 1.0e-15
        @test f[2] ≈ 0.0 atol = 1.0e-15
        @test f[3] ≈ 0.0 atol = 1.0e-15
        @test f[4] ≈ 0.0 atol = 1.0e-15

        # By should not be corrected (normal B component in y-sweep)
        @test f[7] ≈ 0.0 atol = 1.0e-15

        # Bx (f[6]) and Bz (f[8]) should be non-zero
        @test abs(f[6]) > 1.0e-10
        @test abs(f[8]) > 1.0e-10
    end

    @testset "Hall dt 2D" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = HallMHDEquations{2}(eos; di = 1.0)

        w = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
        u = primitive_to_conserved(law, w)

        nx, ny = 5, 5
        dx, dy = 0.1, 0.1
        U = Array{SVector{8, Float64}}(undef, nx + 4, ny + 4)
        for j in 1:(ny + 4), i in 1:(nx + 4)
            U[i, j] = u
        end

        dt = hall_dt(law, U, dx, dy, nx, ny)
        @test dt > 0.0
        @test isfinite(dt)
    end

    @testset "Hall dt with zero B gives Inf" begin
        eos = IdealGasEOS(5.0 / 3.0)
        law = HallMHDEquations{1}(eos; di = 1.0)

        w = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        u = primitive_to_conserved(law, w)

        nc = 10
        dx = 0.1
        U = Vector{SVector{8, Float64}}(undef, nc + 4)
        for i in 1:(nc + 4)
            U[i] = u
        end

        dt = hall_dt(law, U, dx, nc)
        @test isinf(dt)
    end
end

# ============================================================
# 5. TwoFluidEquations
# ============================================================
@testset "TwoFluidEquations" begin

    @testset "Construction and nvariables" begin
        eos_i = IdealGasEOS(5.0 / 3.0)
        eos_e = IdealGasEOS(5.0 / 3.0)

        @testset "1D" begin
            law = TwoFluidEquations{1}(eos_i, eos_e)
            @test nvariables(law) == 6
            @test law.eos_ion === eos_i
            @test law.eos_electron === eos_e
            @test law.mass_ratio == 1836.0
            @test law.charge_ratio == -1.0
        end

        @testset "2D" begin
            law = TwoFluidEquations{2}(eos_i, eos_e)
            @test nvariables(law) == 8
        end

        @testset "Custom parameters" begin
            law = TwoFluidEquations{1}(eos_i, eos_e; mass_ratio = 100.0, charge_ratio = -2.0)
            @test law.mass_ratio == 100.0
            @test law.charge_ratio == -2.0
        end
    end

    @testset "Conserved-Primitive round-trip" begin
        eos_i = IdealGasEOS(5.0 / 3.0)
        eos_e = IdealGasEOS(5.0 / 3.0)

        @testset "1D" begin
            law = TwoFluidEquations{1}(eos_i, eos_e)
            primitives = [
                SVector(1.0, 0.0, 1.0, 0.5, 0.0, 0.5),       # static
                SVector(1.0, 0.5, 2.0, 0.1, -0.3, 0.1),       # moving
                SVector(2.0, -1.0, 3.0, 0.01, 1.0, 0.01),     # different densities
            ]
            for w in primitives
                u = primitive_to_conserved(law, w)
                w2 = conserved_to_primitive(law, u)
                @test w2 ≈ w atol = 1.0e-12
            end
        end

        @testset "2D" begin
            law = TwoFluidEquations{2}(eos_i, eos_e)
            primitives = [
                SVector(1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5),
                SVector(1.0, 0.5, -0.3, 2.0, 0.1, -0.3, 0.2, 0.1),
            ]
            for w in primitives
                u = primitive_to_conserved(law, w)
                w2 = conserved_to_primitive(law, u)
                @test w2 ≈ w atol = 1.0e-12
            end
        end
    end

    @testset "Species accessors extract correct sub-vectors" begin
        eos_i = IdealGasEOS(5.0 / 3.0)
        eos_e = IdealGasEOS(5.0 / 3.0)

        @testset "1D primitive" begin
            law = TwoFluidEquations{1}(eos_i, eos_e)
            w = SVector(1.0, 0.5, 2.0, 0.1, -0.3, 0.1)

            wi = ion_primitive(law, w)
            @test wi == SVector(1.0, 0.5, 2.0)

            we = electron_primitive(law, w)
            @test we == SVector(0.1, -0.3, 0.1)
        end

        @testset "1D conserved" begin
            law = TwoFluidEquations{1}(eos_i, eos_e)
            w = SVector(1.0, 0.5, 2.0, 0.1, -0.3, 0.1)
            u = primitive_to_conserved(law, w)

            ui = ion_conserved(law, u)
            @test ui == SVector(u[1], u[2], u[3])

            ue = electron_conserved(law, u)
            @test ue == SVector(u[4], u[5], u[6])
        end

        @testset "2D primitive" begin
            law = TwoFluidEquations{2}(eos_i, eos_e)
            w = SVector(1.0, 0.5, -0.3, 2.0, 0.1, -0.3, 0.2, 0.1)

            wi = ion_primitive(law, w)
            @test wi == SVector(1.0, 0.5, -0.3, 2.0)

            we = electron_primitive(law, w)
            @test we == SVector(0.1, -0.3, 0.2, 0.1)
        end

        @testset "2D conserved" begin
            law = TwoFluidEquations{2}(eos_i, eos_e)
            w = SVector(1.0, 0.5, -0.3, 2.0, 0.1, -0.3, 0.2, 0.1)
            u = primitive_to_conserved(law, w)

            ui = ion_conserved(law, u)
            @test ui == SVector(u[1], u[2], u[3], u[4])

            ue = electron_conserved(law, u)
            @test ue == SVector(u[5], u[6], u[7], u[8])
        end
    end

    @testset "Physical flux: stacked Euler fluxes" begin
        eos_i = IdealGasEOS(5.0 / 3.0)
        eos_e = IdealGasEOS(5.0 / 3.0)

        @testset "1D: each species independent" begin
            law = TwoFluidEquations{1}(eos_i, eos_e)
            law_euler = EulerEquations{1}(eos_i)

            w = SVector(1.0, 0.5, 2.0, 0.1, -0.3, 0.1)
            f = physical_flux(law, w, 1)

            # Ion flux should match Euler for ion primitive
            wi = SVector(1.0, 0.5, 2.0)
            fi = physical_flux(law_euler, wi, 1)
            @test f[1] ≈ fi[1] atol = 1.0e-14
            @test f[2] ≈ fi[2] atol = 1.0e-14
            @test f[3] ≈ fi[3] atol = 1.0e-14

            # Electron flux should match Euler for electron primitive
            we = SVector(0.1, -0.3, 0.1)
            fe = physical_flux(law_euler, we, 1)
            @test f[4] ≈ fe[1] atol = 1.0e-14
            @test f[5] ≈ fe[2] atol = 1.0e-14
            @test f[6] ≈ fe[3] atol = 1.0e-14
        end

        @testset "2D: each species independent" begin
            law = TwoFluidEquations{2}(eos_i, eos_e)
            law_euler = EulerEquations{2}(eos_i)

            w = SVector(1.0, 0.5, -0.3, 2.0, 0.1, -0.3, 0.2, 0.1)

            for dir in [1, 2]
                f = physical_flux(law, w, dir)

                wi = SVector(1.0, 0.5, -0.3, 2.0)
                fi = physical_flux(law_euler, wi, dir)
                @test f[1] ≈ fi[1] atol = 1.0e-14
                @test f[2] ≈ fi[2] atol = 1.0e-14
                @test f[3] ≈ fi[3] atol = 1.0e-14
                @test f[4] ≈ fi[4] atol = 1.0e-14

                we = SVector(0.1, -0.3, 0.2, 0.1)
                fe = physical_flux(law_euler, we, dir)
                @test f[5] ≈ fe[1] atol = 1.0e-14
                @test f[6] ≈ fe[2] atol = 1.0e-14
                @test f[7] ≈ fe[3] atol = 1.0e-14
                @test f[8] ≈ fe[4] atol = 1.0e-14
            end
        end
    end

    @testset "Wave speeds: envelope of both species" begin
        eos_i = IdealGasEOS(5.0 / 3.0)
        eos_e = IdealGasEOS(5.0 / 3.0)

        @testset "1D" begin
            law = TwoFluidEquations{1}(eos_i, eos_e)
            # Ion: rho=1, v=0.5, P=2  => c_i = sqrt(5/3 * 2/1)
            # Electron: rho=0.1, v=-0.3, P=0.1 => c_e = sqrt(5/3 * 0.1/0.1) = sqrt(5/3)
            w = SVector(1.0, 0.5, 2.0, 0.1, -0.3, 0.1)

            c_i = sound_speed(eos_i, 1.0, 2.0)
            c_e = sound_speed(eos_e, 0.1, 0.1)

            lmin, lmax = wave_speeds(law, w, 1)
            @test lmin ≈ min(0.5 - c_i, -0.3 - c_e) atol = 1.0e-14
            @test lmax ≈ max(0.5 + c_i, -0.3 + c_e) atol = 1.0e-14

            smax = max_wave_speed(law, w, 1)
            @test smax ≈ max(abs(0.5) + c_i, abs(-0.3) + c_e) atol = 1.0e-14
        end

        @testset "2D" begin
            law = TwoFluidEquations{2}(eos_i, eos_e)
            w = SVector(1.0, 0.5, -0.3, 2.0, 0.1, -0.3, 0.2, 0.1)

            c_i = sound_speed(eos_i, 1.0, 2.0)
            c_e = sound_speed(eos_e, 0.1, 0.1)

            # x-direction
            lmin_x, lmax_x = wave_speeds(law, w, 1)
            @test lmin_x ≈ min(0.5 - c_i, -0.3 - c_e) atol = 1.0e-14
            @test lmax_x ≈ max(0.5 + c_i, -0.3 + c_e) atol = 1.0e-14

            # y-direction
            lmin_y, lmax_y = wave_speeds(law, w, 2)
            @test lmin_y ≈ min(-0.3 - c_i, 0.2 - c_e) atol = 1.0e-14
            @test lmax_y ≈ max(-0.3 + c_i, 0.2 + c_e) atol = 1.0e-14
        end
    end

    @testset "Lorentz source" begin
        eos_i = IdealGasEOS(5.0 / 3.0)
        eos_e = IdealGasEOS(5.0 / 3.0)

        @testset "1D: zero when E=0 and B=0" begin
            law = TwoFluidEquations{1}(eos_i, eos_e)
            w = SVector(1.0, 0.5, 2.0, 0.1, -0.3, 0.1)
            src = lorentz_source_1d(law, w, 0.0, 0.0, 0.0, 0.0)
            for k in 1:6
                @test src[k] ≈ 0.0 atol = 1.0e-15
            end
        end

        @testset "2D: zero when E=0 and B=0" begin
            law = TwoFluidEquations{2}(eos_i, eos_e)
            w = SVector(1.0, 0.5, -0.3, 2.0, 0.1, -0.3, 0.2, 0.1)
            src = lorentz_source_2d(law, w, 0.0, 0.0, 0.0, 0.0, 0.0)
            for k in 1:8
                @test src[k] ≈ 0.0 atol = 1.0e-15
            end
        end

        @testset "1D: momentum conservation (ion + electron cancel)" begin
            # For equal density, opposite charge-to-mass ratio:
            # qm_i = 1.0, qm_e = charge_ratio * mass_ratio
            # Forces cancel when rho_i * qm_i + rho_e * qm_e = 0
            # i.e., rho_i + rho_e * charge_ratio * mass_ratio = 0
            # Choose rho_e = rho_i / (|charge_ratio| * mass_ratio) for cancellation.
            law = TwoFluidEquations{1}(eos_i, eos_e; mass_ratio = 1.0, charge_ratio = -1.0)
            # qm_i = 1, qm_e = -1*1 = -1
            # Force_x_ion  = rho_i * qm_i * Ex = rho_i * Ex
            # Force_x_elec = rho_e * qm_e * Ex = -rho_e * Ex
            # Sum = (rho_i - rho_e) * Ex, cancels when rho_i = rho_e

            # Equal density, same velocity => perfect cancellation
            w = SVector(1.0, 0.5, 2.0, 1.0, 0.5, 2.0)
            Ex = 3.0
            src = lorentz_source_1d(law, w, 1.0, 0.0, 0.0, Ex)

            # Ion momentum + electron momentum forces should cancel
            @test src[2] + src[5] ≈ 0.0 atol = 1.0e-14

            # Energy sources should also cancel (same v)
            @test src[3] + src[6] ≈ 0.0 atol = 1.0e-14
        end

        @testset "2D: momentum conservation" begin
            law = TwoFluidEquations{2}(eos_i, eos_e; mass_ratio = 1.0, charge_ratio = -1.0)

            # Equal density, same velocity
            w = SVector(1.0, 0.5, -0.3, 2.0, 1.0, 0.5, -0.3, 2.0)
            src = lorentz_source_2d(law, w, 1.0, 0.5, 0.3, 2.0, 1.0)

            # F_ix + F_ex should cancel
            @test src[2] + src[6] ≈ 0.0 atol = 1.0e-14  # x-momentum
            @test src[3] + src[7] ≈ 0.0 atol = 1.0e-14  # y-momentum

            # Work should also cancel
            @test src[4] + src[8] ≈ 0.0 atol = 1.0e-14
        end

        @testset "1D: mass source is always zero" begin
            law = TwoFluidEquations{1}(eos_i, eos_e)
            w = SVector(1.0, 0.5, 2.0, 0.1, -0.3, 0.1)
            src = lorentz_source_1d(law, w, 1.0, 2.0, 3.0, 5.0)
            @test src[1] ≈ 0.0 atol = 1.0e-15
            @test src[4] ≈ 0.0 atol = 1.0e-15
        end

        @testset "2D: mass source is always zero" begin
            law = TwoFluidEquations{2}(eos_i, eos_e)
            w = SVector(1.0, 0.5, -0.3, 2.0, 0.1, -0.3, 0.2, 0.1)
            src = lorentz_source_2d(law, w, 1.0, 2.0, 3.0, 5.0, 4.0)
            @test src[1] ≈ 0.0 atol = 1.0e-15
            @test src[5] ≈ 0.0 atol = 1.0e-15
        end

        @testset "1D: non-zero Ex produces force" begin
            law = TwoFluidEquations{1}(eos_i, eos_e)
            w = SVector(1.0, 0.0, 1.0, 1.0, 0.0, 1.0)
            Ex = 1.0
            src = lorentz_source_1d(law, w, 0.0, 0.0, 0.0, Ex)
            # qm_i = 1, F_ix = rho_i * 1 * Ex = 1.0
            @test src[2] ≈ 1.0 atol = 1.0e-14
            # qm_e = charge_ratio * mass_ratio = -1836
            qm_e = law.charge_ratio * law.mass_ratio
            @test src[5] ≈ 1.0 * qm_e * Ex atol = 1.0e-10
        end
    end

    @testset "1D two-fluid Riemann problem" begin
        eos_i = IdealGasEOS(1.4)
        eos_e = IdealGasEOS(1.4)
        law = TwoFluidEquations{1}(eos_i, eos_e)
        mesh = StructuredMesh1D(0.0, 1.0, 100)

        # Both species have same Sod-like IC (no coupling in the hyperbolic flux)
        wL = SVector(1.0, 0.0, 1.0, 1.0, 0.0, 1.0)
        wR = SVector(0.125, 0.0, 0.1, 0.125, 0.0, 0.1)

        prob = HyperbolicProblem(
            law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(),
            x -> x < 0.5 ? wL : wR;
            final_time = 0.2, cfl = 0.4
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t ≈ 0.2 atol = 1.0e-10

        # All densities and pressures positive
        for i in eachindex(W)
            @test W[i][1] > 0.0   # rho_i > 0
            @test W[i][3] > 0.0   # P_i > 0
            @test W[i][4] > 0.0   # rho_e > 0
            @test W[i][6] > 0.0   # P_e > 0
        end

        # Since both species have identical ICs and no coupling,
        # the ion and electron solutions should be identical
        for i in eachindex(W)
            @test W[i][1] ≈ W[i][4] atol = 1.0e-12  # rho_i = rho_e
            @test W[i][2] ≈ W[i][5] atol = 1.0e-12  # v_i = v_e
            @test W[i][3] ≈ W[i][6] atol = 1.0e-12  # P_i = P_e
        end

        # Shock structure: boundary values should be near initial states
        @test W[1][1] ≈ 1.0 atol = 0.01
        @test W[end][1] ≈ 0.125 atol = 0.01
    end
end
