using FiniteVolumeMethod
using StaticArrays
using Test

# Reuse the exact Sod solution from Phase 1 tests
function sod_exact(x, t; x0 = 0.5, γ = 1.4)
    ρL, vL, PL = 1.0, 0.0, 1.0
    ρR, vR, PR = 0.125, 0.0, 0.1
    cL = sqrt(γ * PL / ρL)
    cR = sqrt(γ * PR / ρR)

    P_star = 0.30313017805064707
    v_star = 0.92745262004895057
    ρ_star_L = 0.42631942817849544
    ρ_star_R = 0.26557371170530708
    c_star_L = sqrt(γ * P_star / ρ_star_L)

    x_head = x0 - cL * t
    x_tail = x0 + (v_star - c_star_L) * t
    x_contact = x0 + v_star * t
    S_shock = vR + cR * sqrt((γ + 1) / (2γ) * P_star / PR + (γ - 1) / (2γ))
    x_shock = x0 + S_shock * t

    ξ = (x - x0) / t

    if x <= x_head
        return ρL, vL, PL
    elseif x <= x_tail
        gm1 = γ - 1
        gp1 = γ + 1
        v = 2 / gp1 * (cL + ξ)
        c = cL - gm1 / 2 * v
        ρ = ρL * (c / cL)^(2 / gm1)
        P = PL * (c / cL)^(2γ / gm1)
        return ρ, v, P
    elseif x <= x_contact
        return ρ_star_L, v_star, P_star
    elseif x <= x_shock
        return ρ_star_R, v_star, P_star
    else
        return ρR, vR, PR
    end
end

# ============================================================
# HLLC Riemann Solver Tests
# ============================================================

@testset "HLLC Riemann Solver" begin
    eos = IdealGasEOS(1.4)

    @testset "1D HLLC" begin
        law = EulerEquations{1}(eos)

        @testset "Identical states give exact flux" begin
            w = SVector(1.0, 0.5, 1.0)
            f_exact = physical_flux(law, w, 1)
            f_hllc = solve_riemann(HLLCSolver(), law, w, w, 1)
            @test f_hllc ≈ f_exact atol = 1e-13
        end

        @testset "Sod shock tube" begin
            wL = SVector(1.0, 0.0, 1.0)
            wR = SVector(0.125, 0.0, 0.1)
            mesh = StructuredMesh1D(0.0, 1.0, 200)

            prob = HyperbolicProblem(
                law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
                DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
                x -> x < 0.5 ? wL : wR; final_time = 0.2, cfl = 0.5
            )

            x, U, t = solve_hyperbolic(prob)
            W = to_primitive(law, U)

            @test t ≈ 0.2 atol = 1e-10

            dx = 1.0 / 200
            ρ_err = sum(abs(W[i][1] - sod_exact(x[i], 0.2)[1]) * dx for i in 1:200)
            @test ρ_err < 0.04
        end

        @testset "HLLC resolves contact better than HLL" begin
            wL = SVector(1.0, 0.0, 1.0)
            wR = SVector(0.125, 0.0, 0.1)
            N = 200
            mesh = StructuredMesh1D(0.0, 1.0, N)

            function run_sod_1d(solver)
                prob = HyperbolicProblem(
                    law, mesh, solver, CellCenteredMUSCL(MinmodLimiter()),
                    DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
                    x -> x < 0.5 ? wL : wR; final_time = 0.2, cfl = 0.5
                )
                x, U, t = solve_hyperbolic(prob)
                W = to_primitive(law, U)
                dx = 1.0 / N
                return sum(abs(W[i][1] - sod_exact(x[i], 0.2)[1]) * dx for i in 1:N)
            end

            err_hll = run_sod_1d(HLLSolver())
            err_hllc = run_sod_1d(HLLCSolver())
            # HLLC should have less density error than HLL (better contact resolution)
            @test err_hllc < err_hll
        end

        @testset "Einfeldt 1-2-3 with HLLC" begin
            wL = SVector(1.0, -2.0, 0.4)
            wR = SVector(1.0, 2.0, 0.4)
            mesh = StructuredMesh1D(0.0, 1.0, 200)

            prob = HyperbolicProblem(
                law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
                DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
                x -> x < 0.5 ? wL : wR; final_time = 0.15, cfl = 0.4
            )

            x, U, t = solve_hyperbolic(prob)
            W = to_primitive(law, U)

            # Positivity: ρ > 0 and P > 0
            for i in 1:200
                @test W[i][1] > 0
                @test W[i][3] > 0
            end
        end
    end

    @testset "2D HLLC" begin
        law = EulerEquations{2}(eos)

        @testset "Identical states give exact flux" begin
            w = SVector(1.0, 0.5, 0.3, 1.0)
            for dir in (1, 2)
                f_exact = physical_flux(law, w, dir)
                f_hllc = solve_riemann(HLLCSolver(), law, w, w, dir)
                @test f_hllc ≈ f_exact atol = 1e-13
            end
        end

        @testset "Consistency with 1D" begin
            # A 2D state with vy=0 should give same mass flux as 1D
            law1d = EulerEquations{1}(eos)
            wL_1d = SVector(1.0, 0.5, 2.0)
            wR_1d = SVector(0.5, -0.3, 1.0)
            wL_2d = SVector(1.0, 0.5, 0.0, 2.0)
            wR_2d = SVector(0.5, -0.3, 0.0, 1.0)

            f1d = solve_riemann(HLLCSolver(), law1d, wL_1d, wR_1d, 1)
            f2d = solve_riemann(HLLCSolver(), law, wL_2d, wR_2d, 1)

            # Mass flux should match
            @test f2d[1] ≈ f1d[1] atol = 1e-12
            # x-momentum flux should match
            @test f2d[2] ≈ f1d[2] atol = 1e-12
        end
    end
end

# ============================================================
# 2D Structured Solver Tests
# ============================================================

@testset "2D Structured Solver" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    @testset "2D Sod (x-direction, recover 1D)" begin
        # Run 1D problem on a 2D mesh (variation only in x)
        wL = SVector(1.0, 0.0, 0.0, 1.0)
        wR = SVector(0.125, 0.0, 0.0, 0.1)
        N = 200
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, N, 4)

        prob = HyperbolicProblem2D(
            law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
            DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
            TransmissiveBC(), TransmissiveBC(),
            (x, y) -> x < 0.5 ? wL : wR;
            final_time = 0.2, cfl = 0.4
        )

        coords, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t ≈ 0.2 atol = 1e-10

        # Check that solution is uniform in y-direction
        for ix in 1:N
            for iy in 2:4
                @test W[ix, iy][1] ≈ W[ix, 1][1] atol = 1e-12
                @test W[ix, iy][2] ≈ W[ix, 1][2] atol = 1e-12
                @test W[ix, iy][4] ≈ W[ix, 1][4] atol = 1e-12
            end
        end

        # Check accuracy against 1D exact solution (use middle row)
        jmid = 2
        dx = 1.0 / N
        ρ_err = sum(abs(W[i, jmid][1] - sod_exact(coords[i, jmid][1], 0.2)[1]) * dx for i in 1:N)
        @test ρ_err < 0.04
    end

    @testset "2D Sod (y-direction, recover 1D)" begin
        # Same test but with the discontinuity in y-direction
        wB = SVector(1.0, 0.0, 0.0, 1.0)    # bottom state
        wT = SVector(0.125, 0.0, 0.0, 0.1)   # top state
        N = 200
        mesh = StructuredMesh2D(0.0, 0.1, 0.0, 1.0, 4, N)

        prob = HyperbolicProblem2D(
            law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(),
            DirichletHyperbolicBC(wB), DirichletHyperbolicBC(wT),
            (x, y) -> y < 0.5 ? wB : wT;
            final_time = 0.2, cfl = 0.4
        )

        coords, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t ≈ 0.2 atol = 1e-10

        # Check uniform in x
        for iy in 1:N
            for ix in 2:4
                @test W[ix, iy][1] ≈ W[1, iy][1] atol = 1e-12
            end
        end

        # Check accuracy against 1D exact solution along y
        imid = 2
        dy = 1.0 / N
        ρ_err = sum(abs(W[imid, j][1] - sod_exact(coords[imid, j][2], 0.2)[1]) * dy for j in 1:N)
        @test ρ_err < 0.04
    end

    @testset "2D Conservation (periodic)" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 50, 50)
        dx, dy = mesh.dx, mesh.dy

        # Smooth initial condition with variation in both x and y
        ic = (x, y) -> SVector(
            1.0 + 0.2 * sin(2π * x) * cos(2π * y),
            0.3,
            0.2,
            1.0
        )

        prob = HyperbolicProblem2D(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            ic; final_time = 0.05, cfl = 0.3
        )

        # Compute initial totals
        U0 = FiniteVolumeMethod.initialize_2d(prob)
        FiniteVolumeMethod.apply_boundary_conditions_2d!(U0, prob, 0.0)
        mass0 = sum(U0[ix + 2, iy + 2][1] for ix in 1:50, iy in 1:50) * dx * dy
        mom_x0 = sum(U0[ix + 2, iy + 2][2] for ix in 1:50, iy in 1:50) * dx * dy
        mom_y0 = sum(U0[ix + 2, iy + 2][3] for ix in 1:50, iy in 1:50) * dx * dy
        energy0 = sum(U0[ix + 2, iy + 2][4] for ix in 1:50, iy in 1:50) * dx * dy

        coords, U_final, t = solve_hyperbolic(prob)
        mass_f = sum(U_final[ix, iy][1] for ix in 1:50, iy in 1:50) * dx * dy
        mom_xf = sum(U_final[ix, iy][2] for ix in 1:50, iy in 1:50) * dx * dy
        mom_yf = sum(U_final[ix, iy][3] for ix in 1:50, iy in 1:50) * dx * dy
        energy_f = sum(U_final[ix, iy][4] for ix in 1:50, iy in 1:50) * dx * dy

        @test mass_f ≈ mass0 rtol = 1e-12
        @test mom_xf ≈ mom_x0 rtol = 1e-12
        @test mom_yf ≈ mom_y0 rtol = 1e-12
        @test energy_f ≈ energy0 rtol = 1e-12
    end

    @testset "2D Sedov blast (qualitative)" begin
        # Sedov blast wave: high-energy point source in uniform medium
        # Check cylindrical symmetry (approximately)
        N = 40
        mesh = StructuredMesh2D(-1.0, 1.0, -1.0, 1.0, N, N)

        # Background state + small region of high pressure at center
        P_bg = 1e-5
        P_blast = 1.0
        r_blast = 3.0 * mesh.dx  # few cells

        ic = function (x, y)
            r = sqrt(x^2 + y^2)
            P = r < r_blast ? P_blast : P_bg
            return SVector(1.0, 0.0, 0.0, P)
        end

        prob = HyperbolicProblem2D(
            law, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(),
            TransmissiveBC(), TransmissiveBC(),
            ic; final_time = 0.1, cfl = 0.3
        )

        coords, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        # All values should be physical
        for iy in 1:N, ix in 1:N
            @test W[ix, iy][1] > 0  # ρ > 0
            @test W[ix, iy][4] > 0  # P > 0
        end

        # Check approximate cylindrical symmetry:
        # Points at equal radius should have similar density
        # Compare (N/4, N/2) and (N/2, N/4) — both at roughly same radius from center
        ix_a, iy_a = div(N, 4), div(N, 2)
        ix_b, iy_b = div(N, 2), div(N, 4)
        xa, ya = coords[ix_a, iy_a]
        xb, yb = coords[ix_b, iy_b]
        ra = sqrt(xa^2 + ya^2)
        rb = sqrt(xb^2 + yb^2)

        if abs(ra - rb) / max(ra, rb) < 0.1
            # If radii are similar, densities should be similar
            ρ_a = W[ix_a, iy_a][1]
            ρ_b = W[ix_b, iy_b][1]
            @test abs(ρ_a - ρ_b) / max(ρ_a, ρ_b) < 0.3
        else
            @test true  # radii too different, skip
        end
    end

    @testset "2D Reflective BC" begin
        # A uniform flow hitting a reflective wall should reflect
        N = 20
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, N, N)

        # Uniform state
        w0 = SVector(1.0, 0.0, 0.0, 1.0)

        prob = HyperbolicProblem2D(
            law, mesh, HLLSolver(), NoReconstruction(),
            ReflectiveBC(), ReflectiveBC(),
            ReflectiveBC(), ReflectiveBC(),
            (x, y) -> w0;
            final_time = 0.01, cfl = 0.3
        )

        coords, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        # Uniform state with zero velocity + reflective walls should stay uniform
        for iy in 1:N, ix in 1:N
            @test W[ix, iy][1] ≈ 1.0 atol = 1e-10
            @test W[ix, iy][2] ≈ 0.0 atol = 1e-10
            @test W[ix, iy][3] ≈ 0.0 atol = 1e-10
            @test W[ix, iy][4] ≈ 1.0 atol = 1e-10
        end
    end

    @testset "2D CFL time step" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 10, 10)
        w0 = SVector(1.0, 0.0, 0.0, 1.0)

        prob = HyperbolicProblem2D(
            law, mesh, HLLSolver(), NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(),
            TransmissiveBC(), TransmissiveBC(),
            (x, y) -> w0;
            final_time = 1.0, cfl = 0.5
        )

        U = FiniteVolumeMethod.initialize_2d(prob)
        FiniteVolumeMethod.apply_boundary_conditions_2d!(U, prob, 0.0)
        dt = compute_dt_2d(prob, U, 0.0)

        # dt > 0
        @test dt > 0

        # CFL: dt ≤ cfl / (c/dx + c/dy) where c = sound speed
        c = sound_speed(eos, 1.0, 1.0)
        dt_max = 0.5 / (c / mesh.dx + c / mesh.dy)
        @test dt <= dt_max + 1e-14
    end

    @testset "2D All solvers produce valid results" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, 50, 4)
        wL = SVector(1.0, 0.0, 0.0, 1.0)
        wR = SVector(0.125, 0.0, 0.0, 0.1)

        for solver in [LaxFriedrichsSolver(), HLLSolver(), HLLCSolver()]
            @testset "$(nameof(typeof(solver)))" begin
                prob = HyperbolicProblem2D(
                    law, mesh, solver, CellCenteredMUSCL(MinmodLimiter()),
                    DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
                    TransmissiveBC(), TransmissiveBC(),
                    (x, y) -> x < 0.5 ? wL : wR;
                    final_time = 0.2, cfl = 0.3
                )

                coords, U, t = solve_hyperbolic(prob)
                W = to_primitive(law, U)

                @test t ≈ 0.2 atol = 1e-10
                for iy in 1:4, ix in 1:50
                    @test W[ix, iy][1] > 0  # ρ > 0
                    @test W[ix, iy][4] > 0  # P > 0
                end
            end
        end
    end

    @testset "HLLC vs HLL: better contact resolution in 2D" begin
        N = 100
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, N, 4)
        wL = SVector(1.0, 0.0, 0.0, 1.0)
        wR = SVector(0.125, 0.0, 0.0, 0.1)

        function run_2d_sod(solver)
            prob = HyperbolicProblem2D(
                law, mesh, solver, CellCenteredMUSCL(MinmodLimiter()),
                DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
                TransmissiveBC(), TransmissiveBC(),
                (x, y) -> x < 0.5 ? wL : wR;
                final_time = 0.2, cfl = 0.3
            )
            coords, U, t = solve_hyperbolic(prob)
            W = to_primitive(law, U)
            dx = 1.0 / N
            jmid = 2
            return sum(abs(W[i, jmid][1] - sod_exact(coords[i, jmid][1], 0.2)[1]) * dx for i in 1:N)
        end

        err_hll = run_2d_sod(HLLSolver())
        err_hllc = run_2d_sod(HLLCSolver())
        @test err_hllc < err_hll
    end
end
