using FiniteVolumeMethod
using Test
using StaticArrays

# ============================================================
# 2D MHD Solver Basic Tests
# ============================================================
@testset "2D MHD Solver Basics" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)

    @testset "Uniform state stays uniform" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 20, 20)
        w_const = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        ic(x, y) = w_const

        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
            ic; final_time = 0.05, cfl = 0.3
        )

        coords, U_final, t_final, ct = solve_hyperbolic(prob)
        W = to_primitive(law, U_final)

        u_ref = primitive_to_conserved(law, w_const)
        for iy in 1:20, ix in 1:20
            @test U_final[ix, iy] ≈ u_ref atol = 1e-12
        end
        @test max_divB(ct, mesh.dx, mesh.dy, 20, 20) < 1e-13
    end

    @testset "Return value structure" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 10, 10)
        ic(x, y) = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0)

        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
            ic; final_time = 0.01, cfl = 0.3
        )

        result = solve_hyperbolic(prob)
        @test length(result) == 4  # (coords, U, t, ct)
        coords, U, t, ct = result
        @test size(U) == (10, 10)
        @test size(coords) == (10, 10)
        @test t ≈ 0.01 atol = 1e-10
        @test ct isa CTData2D
    end
end

# ============================================================
# ∇·B = 0 Constraint Tests
# ============================================================
@testset "∇·B = 0 (Constrained Transport)" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)

    @testset "Uniform field: ∇·B = 0" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 30, 30)
        ic(x, y) = SVector(1.0, 0.5, 0.3, 0.0, 1.0, 1.0, 0.5, 0.0)

        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
            ic; final_time = 0.05, cfl = 0.4
        )

        _, _, _, ct = solve_hyperbolic(prob)
        @test max_divB(ct, mesh.dx, mesh.dy, 30, 30) < 1e-12
    end

    @testset "Non-uniform field: ∇·B = 0 maintained" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 30, 30)
        function nonuniform_ic(x, y)
            ρ = 1.0 + 0.2 * sin(2π * x) * cos(2π * y)
            vx = 0.1 * sin(2π * y)
            vy = -0.1 * sin(2π * x)
            vz = 0.0
            P = 1.0
            # Divergence-free B: Bx = ∂A/∂y, By = -∂A/∂x for A = cos(2πx)*cos(2πy)/(2π)
            Bx = -cos(2π * x) * sin(2π * y)
            By = sin(2π * x) * cos(2π * y)
            Bz = 0.0
            return SVector(ρ, vx, vy, vz, P, Bx, By, Bz)
        end

        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            nonuniform_ic; final_time = 0.05, cfl = 0.4
        )

        _, _, _, ct = solve_hyperbolic(prob)
        divB_max = max_divB(ct, mesh.dx, mesh.dy, 30, 30)
        @test divB_max < 1e-12
    end

    @testset "Euler method: ∇·B = 0" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 20, 20)
        ic(x, y) = SVector(1.0, 0.3, -0.2, 0.0, 1.0, 0.5, 0.8, 0.0)

        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
            ic; final_time = 0.02, cfl = 0.3
        )

        _, _, _, ct = solve_hyperbolic(prob; method = :euler)
        @test max_divB(ct, mesh.dx, mesh.dy, 20, 20) < 1e-12
    end
end

# ============================================================
# Orszag-Tang Vortex
# ============================================================
@testset "Orszag-Tang Vortex" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)
    γ = 5.0 / 3.0

    function ot_ic(x, y)
        ρ = γ^2
        P = γ
        vx = -sin(2π * y)
        vy = sin(2π * x)
        vz = 0.0
        Bx = -sin(2π * y) / sqrt(4π)
        By = sin(4π * x) / sqrt(4π)
        Bz = 0.0
        return SVector(ρ, vx, vy, vz, P, Bx, By, Bz)
    end

    @testset "HLLD solver" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 50, 50)
        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            ot_ic; final_time = 0.1, cfl = 0.4
        )

        coords, U_final, t_final, ct = solve_hyperbolic(prob)
        W = to_primitive(law, U_final)

        ρ = [w[1] for w in W]
        P = [w[5] for w in W]

        @test t_final ≈ 0.1 atol = 1e-10
        @test all(ρ .> 0)
        @test all(P .> 0)
        @test all(isfinite.(ρ))
        @test max_divB(ct, mesh.dx, mesh.dy, 50, 50) < 1e-12

        # Orszag-Tang should develop structure (not remain uniform)
        @test maximum(ρ) > minimum(ρ) + 0.1
    end

    @testset "HLL solver" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 40, 40)
        prob = HyperbolicProblem2D(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            ot_ic; final_time = 0.1, cfl = 0.4
        )

        coords, U_final, t_final, ct = solve_hyperbolic(prob)
        W = to_primitive(law, U_final)
        ρ = [w[1] for w in W]
        P = [w[5] for w in W]

        @test t_final ≈ 0.1 atol = 1e-10
        @test all(ρ .> 0)
        @test all(P .> 0)
        @test max_divB(ct, mesh.dx, mesh.dy, 40, 40) < 1e-12
    end

    @testset "HLLD vs HLL comparison" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 40, 40)

        prob_hlld = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            ot_ic; final_time = 0.05, cfl = 0.4
        )
        prob_hll = HyperbolicProblem2D(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            ot_ic; final_time = 0.05, cfl = 0.4
        )

        _, U_hlld, _, _ = solve_hyperbolic(prob_hlld)
        _, U_hll, _, _ = solve_hyperbolic(prob_hll)

        # Both should be valid
        W_hlld = to_primitive(law, U_hlld)
        W_hll = to_primitive(law, U_hll)
        @test all(w -> w[1] > 0, W_hlld)
        @test all(w -> w[1] > 0, W_hll)

        # HLLD should give a different (generally sharper) solution
        ρ_hlld = [w[1] for w in W_hlld]
        ρ_hll = [w[1] for w in W_hll]
        @test maximum(abs.(ρ_hlld .- ρ_hll)) > 1e-6
    end
end

# ============================================================
# Magnetic Field Loop Advection
# ============================================================
@testset "Field Loop Advection" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)

    # A magnetic field loop advected by a uniform flow.
    # The B-field is a divergence-free loop defined by a vector potential.
    vx_bg = 1.0
    vy_bg = 0.5
    ρ_bg = 1.0
    P_bg = 1.0
    R0 = 0.3  # loop radius
    A0 = 1e-3  # amplitude (weak field → linear regime)

    function loop_ic(x, y)
        # Center at (0.5, 0.5)
        r = sqrt((x - 0.5)^2 + (y - 0.5)^2)
        if r < R0
            # Vector potential Az = A0 * (R0 - r)
            # Bx = ∂Az/∂y = A0 * (-1) * (y - 0.5) / r
            # By = -∂Az/∂x = A0 * (x - 0.5) / r
            Bx = -A0 * (y - 0.5) / r
            By = A0 * (x - 0.5) / r
        else
            Bx = 0.0
            By = 0.0
        end
        return SVector(ρ_bg, vx_bg, vy_bg, 0.0, P_bg, Bx, By, 0.0)
    end

    # Vector potential for divergence-free initialization
    function Az_loop(x, y)
        r = sqrt((x - 0.5)^2 + (y - 0.5)^2)
        return r < R0 ? A0 * (R0 - r) : 0.0
    end

    @testset "Advection preserves ∇·B = 0" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 50, 50)
        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            loop_ic; final_time = 0.1, cfl = 0.4
        )

        _, _, t_final, ct = solve_hyperbolic(prob; vector_potential = Az_loop)
        @test t_final ≈ 0.1 atol = 1e-10
        @test max_divB(ct, mesh.dx, mesh.dy, 50, 50) < 1e-12
    end

    @testset "Density and pressure remain positive" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 40, 40)
        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            loop_ic; final_time = 0.1, cfl = 0.4
        )

        _, U_final, _, _ = solve_hyperbolic(prob)
        W = to_primitive(law, U_final)
        ρ = [w[1] for w in W]
        P = [w[5] for w in W]

        @test all(ρ .> 0)
        @test all(P .> 0)
    end

    @testset "Weak field preserves density/pressure uniformity" begin
        # In the weak-field limit, the field loop shouldn't disturb ρ or P much
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 40, 40)
        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            loop_ic; final_time = 0.1, cfl = 0.4
        )

        _, U_final, _, _ = solve_hyperbolic(prob)
        W = to_primitive(law, U_final)
        ρ = [w[1] for w in W]
        P = [w[5] for w in W]

        @test maximum(ρ) - minimum(ρ) < 0.01  # Near-uniform density
        @test maximum(P) - minimum(P) < 0.01  # Near-uniform pressure
    end
end

# ============================================================
# 2D MHD Conservation Tests
# ============================================================
@testset "2D MHD Conservation" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)
    γ = 5.0 / 3.0

    @testset "Periodic BCs conserve mass and momentum" begin
        function smooth_ic(x, y)
            ρ = 1.0 + 0.1 * sin(2π * x) * cos(2π * y)
            vx = 0.1 * sin(2π * y)
            vy = -0.1 * sin(2π * x)
            vz = 0.0
            P = 1.0
            Bx = 0.5
            By = 0.3
            Bz = 0.0
            return SVector(ρ, vx, vy, vz, P, Bx, By, Bz)
        end

        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 30, 30)
        dx, dy = mesh.dx, mesh.dy

        # Initial state
        prob0 = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            smooth_ic; final_time = 0.0, cfl = 0.4
        )
        _, U0, _, _ = solve_hyperbolic(prob0)

        # Final state
        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            smooth_ic; final_time = 0.05, cfl = 0.4
        )
        _, U_final, _, _ = solve_hyperbolic(prob)

        dA = dx * dy
        mass_0 = sum(u[1] for u in U0) * dA
        mass_f = sum(u[1] for u in U_final) * dA
        momx_0 = sum(u[2] for u in U0) * dA
        momx_f = sum(u[2] for u in U_final) * dA
        momy_0 = sum(u[3] for u in U0) * dA
        momy_f = sum(u[3] for u in U_final) * dA
        energy_0 = sum(u[5] for u in U0) * dA
        energy_f = sum(u[5] for u in U_final) * dA

        @test mass_f ≈ mass_0 atol = 1e-10
        @test momx_f ≈ momx_0 atol = 1e-10
        @test momy_f ≈ momy_0 atol = 1e-10
        @test energy_f ≈ energy_0 atol = 1e-8  # Energy may have slightly larger error
    end
end

# ============================================================
# 2D MHD Boundary Condition Tests
# ============================================================
@testset "2D MHD Boundary Conditions" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)

    @testset "TransmissiveBC" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 20, 20)
        ic(x, y) = SVector(1.0, 0.5, 0.3, 0.0, 1.0, 0.5, 0.8, 0.0)

        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
            ic; final_time = 0.02, cfl = 0.3
        )

        _, U, t, ct = solve_hyperbolic(prob)
        W = to_primitive(law, U)
        @test all(w -> w[1] > 0, W)
        @test all(w -> w[5] > 0, W)
        @test max_divB(ct, mesh.dx, mesh.dy, 20, 20) < 1e-12
    end

    @testset "ReflectiveBC" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 20, 20)
        function reflect_ic(x, y)
            ρ = 1.0 + 0.5 * exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.01)
            P = 1.0 + 0.5 * exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.01)
            return SVector(ρ, 0.0, 0.0, 0.0, P, 0.5, 0.5, 0.0)
        end

        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), NoReconstruction(),
            ReflectiveBC(), ReflectiveBC(), ReflectiveBC(), ReflectiveBC(),
            reflect_ic; final_time = 0.02, cfl = 0.3
        )

        _, U, t, ct = solve_hyperbolic(prob)
        W = to_primitive(law, U)
        @test all(w -> w[1] > 0, W)
        @test all(w -> w[5] > 0, W)
        @test max_divB(ct, mesh.dx, mesh.dy, 20, 20) < 1e-12
    end

    @testset "PeriodicBC" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 20, 20)
        function periodic_ic(x, y)
            ρ = 1.0 + 0.1 * sin(2π * x) * sin(2π * y)
            return SVector(ρ, 0.1, 0.1, 0.0, 1.0, 0.5, 0.3, 0.0)
        end

        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            periodic_ic; final_time = 0.03, cfl = 0.4
        )

        _, U, t, ct = solve_hyperbolic(prob)
        W = to_primitive(law, U)
        @test all(w -> w[1] > 0, W)
        @test all(w -> w[5] > 0, W)
        @test max_divB(ct, mesh.dx, mesh.dy, 20, 20) < 1e-12
    end
end

# ============================================================
# 2D MHD CFL Stability
# ============================================================
@testset "2D MHD CFL Stability" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)
    γ = 5.0 / 3.0

    function ot_ic(x, y)
        ρ = γ^2
        P = γ
        vx = -sin(2π * y)
        vy = sin(2π * x)
        vz = 0.0
        Bx = -sin(2π * y) / sqrt(4π)
        By = sin(4π * x) / sqrt(4π)
        Bz = 0.0
        return SVector(ρ, vx, vy, vz, P, Bx, By, Bz)
    end

    cfls = [0.2, 0.4]
    for cfl in cfls
        @testset "CFL = $cfl" begin
            mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 30, 30)
            prob = HyperbolicProblem2D(
                law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
                PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
                PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
                ot_ic; final_time = 0.05, cfl = cfl
            )

            _, U, t, ct = solve_hyperbolic(prob)
            W = to_primitive(law, U)
            @test all(w -> w[1] > 0, W)
            @test all(w -> w[5] > 0, W)
            @test max_divB(ct, mesh.dx, mesh.dy, 30, 30) < 1e-12
        end
    end
end

# ============================================================
# 2D MHD with All Riemann Solvers
# ============================================================
@testset "2D MHD All Solvers" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)

    ic(x, y) = SVector(1.0, 0.3, -0.2, 0.0, 1.0, 0.5, 0.8, 0.0)

    solvers = [
        ("HLLD", HLLDSolver()),
        ("HLL", HLLSolver()),
        ("Lax-Friedrichs", LaxFriedrichsSolver()),
    ]

    for (name, solver) in solvers
        @testset "$name" begin
            mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 20, 20)
            prob = HyperbolicProblem2D(
                law, mesh, solver, NoReconstruction(),
                TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
                ic; final_time = 0.02, cfl = 0.3
            )

            _, U, t, ct = solve_hyperbolic(prob)
            W = to_primitive(law, U)
            @test all(w -> w[1] > 0, W)
            @test all(w -> w[5] > 0, W)
            @test all(w -> all(isfinite, w), W)
            @test max_divB(ct, mesh.dx, mesh.dy, 20, 20) < 1e-12
        end
    end
end

# ============================================================
# 2D Sod Shock Tube (1D problem on 2D mesh)
# ============================================================
@testset "2D MHD Sod (1D on 2D)" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{2}(eos)

    @testset "x-direction" begin
        Bx = 0.75
        wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx, 1.0, 0.0)
        wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, Bx, -1.0, 0.0)
        ic(x, y) = x < 0.5 ? wL : wR

        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, 200, 4)
        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
            ic; final_time = 0.1, cfl = 0.4
        )

        _, U, t, ct = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        ρ = [W[ix, 1][1] for ix in 1:200]
        @test all(ρ .> 0)

        # Solution should be uniform in y
        for ix in 1:200
            for iy in 2:4
                @test W[ix, iy][1] ≈ W[ix, 1][1] atol = 1e-10
            end
        end

        # Should have structure in x
        @test maximum(ρ) > minimum(ρ) + 0.01
    end
end

# ============================================================
# 2D MHD Limiters
# ============================================================
@testset "2D MHD with Different Limiters" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)

    ic(x, y) = SVector(
        1.0 + 0.5 * exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.01),
        0.0, 0.0, 0.0,
        1.0, 0.5, 0.3, 0.0
    )

    limiters = [
        ("Minmod", MinmodLimiter()),
        ("VanLeer", VanLeerLimiter()),
        ("Superbee", SuperbeeLimiter()),
    ]

    for (name, limiter) in limiters
        @testset "$name" begin
            mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 20, 20)
            prob = HyperbolicProblem2D(
                law, mesh, HLLDSolver(), CellCenteredMUSCL(limiter),
                TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
                ic; final_time = 0.02, cfl = 0.3
            )

            _, U, t, ct = solve_hyperbolic(prob)
            W = to_primitive(law, U)
            @test all(w -> w[1] > 0, W)
            @test all(w -> w[5] > 0, W)
            @test max_divB(ct, mesh.dx, mesh.dy, 20, 20) < 1e-12
        end
    end
end

# ============================================================
# MHD Rotor Problem
# ============================================================
@testset "MHD Rotor" begin
    eos = IdealGasEOS(gamma = 1.4)
    law = IdealMHDEquations{2}(eos)

    # MHD rotor: dense spinning disk in a magnetized ambient medium
    r0 = 0.1   # rotor radius
    r1 = 0.115 # taper radius
    v0 = 2.0   # angular velocity at r0

    function rotor_ic(x, y)
        r = sqrt((x - 0.5)^2 + (y - 0.5)^2)
        if r <= r0
            ρ = 10.0
            vx = -v0 * (y - 0.5) / r0
            vy = v0 * (x - 0.5) / r0
        elseif r <= r1
            f = (r1 - r) / (r1 - r0)
            ρ = 1.0 + 9.0 * f
            vx = -f * v0 * (y - 0.5) / r
            vy = f * v0 * (x - 0.5) / r
        else
            ρ = 1.0
            vx = 0.0
            vy = 0.0
        end
        P = 1.0
        Bx = 5.0 / sqrt(4π)
        return SVector(ρ, vx, vy, 0.0, P, Bx, 0.0, 0.0)
    end

    @testset "Short-time stability" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 40, 40)
        prob = HyperbolicProblem2D(
            law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
            rotor_ic; final_time = 0.02, cfl = 0.3
        )

        _, U, t, ct = solve_hyperbolic(prob)
        W = to_primitive(law, U)
        ρ = [w[1] for w in W]
        P = [w[5] for w in W]

        @test t ≈ 0.02 atol = 1e-10
        @test all(ρ .> 0)
        @test all(P .> 0)
        @test max_divB(ct, mesh.dx, mesh.dy, 40, 40) < 1e-12
    end
end

# ============================================================
# CT Data Structure Tests
# ============================================================
@testset "CTData2D" begin
    @testset "Construction" begin
        ct = CTData2D(10, 8)
        @test size(ct.Bx_face) == (11, 8)
        @test size(ct.By_face) == (10, 9)
        @test size(ct.emf_z) == (11, 9)
    end

    @testset "copy_ct and copyto_ct!" begin
        ct1 = CTData2D(5, 5)
        ct1.Bx_face .= 1.0
        ct1.By_face .= 2.0

        ct2 = copy_ct(ct1)
        @test ct2.Bx_face == ct1.Bx_face
        @test ct2.By_face == ct1.By_face

        ct3 = CTData2D(5, 5)
        copyto_ct!(ct3, ct1)
        @test ct3.Bx_face == ct1.Bx_face
        @test ct3.By_face == ct1.By_face
    end

    @testset "divB diagnostics" begin
        ct = CTData2D(10, 10)
        # Uniform Bx, zero By → ∇·B = 0
        ct.Bx_face .= 1.0
        ct.By_face .= 0.0
        @test max_divB(ct, 0.1, 0.1, 10, 10) ≈ 0.0 atol = 1e-15
        @test l2_divB(ct, 0.1, 0.1, 10, 10) ≈ 0.0 atol = 1e-15

        # Non-zero divergence
        ct.Bx_face[5, 5] = 2.0
        divB_val = max_divB(ct, 0.1, 0.1, 10, 10)
        @test divB_val > 0
    end
end
