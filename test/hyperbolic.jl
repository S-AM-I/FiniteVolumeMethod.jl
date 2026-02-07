using FiniteVolumeMethod
using StaticArrays
using Test

# ============================================================
# Exact Riemann solver for Sod shock tube (used for verification)
# ============================================================

"""
Exact solution for the Sod shock tube at time `t`.
Left state: (ρ=1, v=0, P=1), Right state: (ρ=0.125, v=0, P=0.1), γ=1.4.
Returns (ρ, v, P) at position x (diaphragm at x=0.5, domain [0,1]).
"""
function sod_exact(x, t; x0 = 0.5, γ = 1.4)
    # Left and right states
    ρL, vL, PL = 1.0, 0.0, 1.0
    ρR, vR, PR = 0.125, 0.0, 0.1

    # Sound speeds
    cL = sqrt(γ * PL / ρL)
    cR = sqrt(γ * PR / ρR)

    # Exact solution parameters (pre-computed for standard Sod)
    # These are the solution of the nonlinear algebraic system for the star region
    P_star = 0.30313017805064707
    v_star = 0.92745262004895057
    ρ_star_L = 0.42631942817849544
    ρ_star_R = 0.26557371170530708

    c_star_L = sqrt(γ * P_star / ρ_star_L)

    # Wave speeds
    # Left rarefaction: head at x0 - cL*t, tail at x0 + (v_star - c_star_L)*t
    x_head = x0 - cL * t
    x_tail = x0 + (v_star - c_star_L) * t

    # Contact discontinuity at x0 + v_star*t
    x_contact = x0 + v_star * t

    # Right shock speed: from Rankine-Hugoniot
    S_shock = v_star + cR * sqrt((γ + 1) / (2γ) * (P_star / PR - 1) + 1)
    # More precise: use the exact formula
    # S = vR + cR * sqrt((γ+1)/(2γ) * P_star/PR + (γ-1)/(2γ))
    S_shock = vR + cR * sqrt((γ + 1) / (2γ) * P_star / PR + (γ - 1) / (2γ))
    x_shock = x0 + S_shock * t

    ξ = (x - x0) / t  # self-similar variable

    if x <= x_head
        # Undisturbed left state
        return ρL, vL, PL
    elseif x <= x_tail
        # Inside rarefaction fan
        gm1 = γ - 1
        gp1 = γ + 1
        v = 2 / gp1 * (cL + ξ)
        c = cL - gm1 / 2 * v
        ρ = ρL * (c / cL)^(2 / gm1)
        P = PL * (c / cL)^(2γ / gm1)
        return ρ, v, P
    elseif x <= x_contact
        # Star region (left of contact)
        return ρ_star_L, v_star, P_star
    elseif x <= x_shock
        # Star region (right of contact)
        return ρ_star_R, v_star, P_star
    else
        # Undisturbed right state
        return ρR, vR, PR
    end
end

# ============================================================
# Tests
# ============================================================

@testset "Mesh" begin
    @testset "StructuredMesh1D" begin
        mesh = StructuredMesh1D(0.0, 1.0, 10)
        @test ncells(mesh) == 10
        @test nfaces(mesh) == 9
        @test cell_volume(mesh, 1) ≈ 0.1
        @test cell_center(mesh, 1) ≈ 0.05
        @test cell_center(mesh, 10) ≈ 0.95
        @test ndims_mesh(mesh) == 1
        @test face_area(mesh, 1) == 1.0
        @test face_owner(mesh, 1) == 1
        @test face_neighbor(mesh, 1) == 2
    end

    @testset "StructuredMesh2D" begin
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 4, 3)
        @test ncells(mesh) == 12
        @test ndims_mesh(mesh) == 2
        @test cell_volume(mesh, 1) ≈ 0.25 * (1.0 / 3.0)
        cx, cy = cell_center(mesh, 1)
        @test cx ≈ 0.125
        @test cy ≈ 1.0 / 6.0
    end
end

@testset "EOS" begin
    eos = IdealGasEOS(1.4)
    @test eos.gamma == 1.4

    # P = (γ-1)ρε → ε = P/((γ-1)ρ)
    ρ, P = 1.0, 1.0
    ε = internal_energy(eos, ρ, P)
    @test ε ≈ P / (0.4 * ρ)
    @test pressure(eos, ρ, ε) ≈ P
    @test sound_speed(eos, ρ, P) ≈ sqrt(1.4)

    # Total energy
    E = total_energy(eos, 1.0, 0.0, 1.0)
    @test E ≈ 1.0 / 0.4  # P/(γ-1) + 0
end

@testset "Conservation Law Interface" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)

    @test nvariables(law) == 3

    # Test primitive ↔ conserved roundtrip
    w = SVector(1.0, 0.5, 1.0)  # ρ, v, P
    u = primitive_to_conserved(law, w)
    w2 = conserved_to_primitive(law, u)
    @test w2 ≈ w atol = 1e-14

    # Test flux computation
    f = physical_flux(law, w, 1)
    @test f[1] ≈ 1.0 * 0.5  # ρv
    @test f[2] ≈ 1.0 * 0.25 + 1.0  # ρv² + P

    # Test wave speeds
    λ = max_wave_speed(law, w, 1)
    c = sound_speed(eos, 1.0, 1.0)
    @test λ ≈ abs(0.5) + c

    λ_min, λ_max = wave_speeds(law, w, 1)
    @test λ_min ≈ 0.5 - c
    @test λ_max ≈ 0.5 + c
end

@testset "Riemann Solvers" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)

    wL = SVector(1.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.1)

    @testset "Lax-Friedrichs" begin
        flux = solve_riemann(LaxFriedrichsSolver(), law, wL, wR, 1)
        @test length(flux) == 3
        @test isfinite(flux[1])
        @test isfinite(flux[2])
        @test isfinite(flux[3])
    end

    @testset "HLL" begin
        flux = solve_riemann(HLLSolver(), law, wL, wR, 1)
        @test length(flux) == 3
        @test isfinite(flux[1])
        @test isfinite(flux[2])
        @test isfinite(flux[3])
    end

    @testset "Identical states give exact flux" begin
        w = SVector(1.0, 0.5, 1.0)
        f_exact = physical_flux(law, w, 1)

        f_lf = solve_riemann(LaxFriedrichsSolver(), law, w, w, 1)
        @test f_lf ≈ f_exact atol = 1e-14

        f_hll = solve_riemann(HLLSolver(), law, w, w, 1)
        @test f_hll ≈ f_exact atol = 1e-14
    end
end

@testset "Sod Shock Tube" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)

    wL = SVector(1.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.1)

    @testset "HLL + MUSCL (N=$N)" for N in [100, 200]
        mesh = StructuredMesh1D(0.0, 1.0, N)
        bc_left = DirichletHyperbolicBC(wL)
        bc_right = DirichletHyperbolicBC(wR)

        prob = HyperbolicProblem(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            bc_left, bc_right,
            x -> x < 0.5 ? wL : wR;
            final_time = 0.2, cfl = 0.5
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t ≈ 0.2 atol = 1e-10

        # Compare against exact solution at sample points
        # Use L1 error norm — should decrease with resolution
        ρ_err = 0.0
        v_err = 0.0
        P_err = 0.0
        dx = 1.0 / N
        for i in 1:N
            ρ_ex, v_ex, P_ex = sod_exact(x[i], 0.2)
            ρ_err += abs(W[i][1] - ρ_ex) * dx
            v_err += abs(W[i][2] - v_ex) * dx
            P_err += abs(W[i][3] - P_ex) * dx
        end

        # L1 errors should be reasonable (not exact due to numerical diffusion)
        @test ρ_err < 0.05  # Density L1 error
        @test v_err < 0.05  # Velocity L1 error
        @test P_err < 0.03  # Pressure L1 error
    end

    @testset "Convergence: finer mesh has smaller error" begin
        mesh_coarse = StructuredMesh1D(0.0, 1.0, 100)
        mesh_fine = StructuredMesh1D(0.0, 1.0, 400)

        function run_sod(mesh, N)
            bc_left = DirichletHyperbolicBC(wL)
            bc_right = DirichletHyperbolicBC(wR)
            prob = HyperbolicProblem(
                law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
                bc_left, bc_right,
                x -> x < 0.5 ? wL : wR;
                final_time = 0.2, cfl = 0.5
            )
            x, U, t = solve_hyperbolic(prob)
            W = to_primitive(law, U)
            dx = 1.0 / N
            err = sum(abs(W[i][1] - sod_exact(x[i], 0.2)[1]) * dx for i in 1:N)
            return err
        end

        err_coarse = run_sod(mesh_coarse, 100)
        err_fine = run_sod(mesh_fine, 400)
        @test err_fine < err_coarse
    end
end

@testset "Einfeldt 1-2-3 (Two-Rarefaction)" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)

    # Two rarefaction waves moving apart (Einfeldt's test)
    wL = SVector(1.0, -2.0, 0.4)
    wR = SVector(1.0, 2.0, 0.4)

    mesh = StructuredMesh1D(0.0, 1.0, 200)
    bc_left = DirichletHyperbolicBC(wL)
    bc_right = DirichletHyperbolicBC(wR)

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        bc_left, bc_right,
        x -> x < 0.5 ? wL : wR;
        final_time = 0.15, cfl = 0.4
    )

    x, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    # Physical checks: density and pressure should remain positive
    for i in 1:length(W)
        @test W[i][1] > 0  # ρ > 0
        @test W[i][3] > 0  # P > 0
    end

    # The solution should be symmetric about x=0.5
    N = length(W)
    for i in 1:div(N, 4)
        @test W[i][1] ≈ W[N + 1 - i][1] atol = 1e-10
        @test W[i][2] ≈ -W[N + 1 - i][2] atol = 1e-10
        @test W[i][3] ≈ W[N + 1 - i][3] atol = 1e-10
    end
end

@testset "Conservation" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)

    # Use periodic BCs to ensure perfect conservation
    mesh = StructuredMesh1D(0.0, 1.0, 100)
    dx = mesh.dx

    # Smooth initial condition (no discontinuities)
    ic = x -> SVector(1.0 + 0.2 * sin(2π * x), 0.5, 1.0)

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), ic;
        final_time = 0.1, cfl = 0.3
    )

    # Initialize and compute initial conserved quantities
    U0 = FiniteVolumeMethod.initialize_1d(prob)
    FiniteVolumeMethod.apply_boundary_conditions!(U0, prob, 0.0)

    mass0 = sum(U0[i][1] for i in 3:102) * dx
    momentum0 = sum(U0[i][2] for i in 3:102) * dx
    energy0 = sum(U0[i][3] for i in 3:102) * dx

    x, U_final, t = solve_hyperbolic(prob)

    mass_final = sum(U_final[i][1] for i in 1:100) * dx
    momentum_final = sum(U_final[i][2] for i in 1:100) * dx
    energy_final = sum(U_final[i][3] for i in 1:100) * dx

    # Conservation should hold to near machine precision for periodic BCs
    @test mass_final ≈ mass0 rtol = 1e-12
    @test momentum_final ≈ momentum0 rtol = 1e-12
    @test energy_final ≈ energy0 rtol = 1e-12
end

@testset "CFL Stability" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 50)

    wL = SVector(1.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.1)

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        x -> x < 0.5 ? wL : wR;
        final_time = 0.2, cfl = 0.5
    )

    U = FiniteVolumeMethod.initialize_1d(prob)
    dt = compute_dt(prob, U, 0.0)

    # dt should be positive and bounded by CFL * dx / max_wave_speed
    @test dt > 0
    c_max = sound_speed(eos, 1.0, 1.0)  # max sound speed in left state
    @test dt <= 0.5 * mesh.dx / c_max + 1e-14

    # dt should not exceed final_time - t
    prob2 = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        x -> x < 0.5 ? wL : wR;
        final_time = 0.001, cfl = 0.5
    )
    U2 = FiniteVolumeMethod.initialize_1d(prob2)
    dt2 = compute_dt(prob2, U2, 0.0)
    @test dt2 ≈ 0.001
end

@testset "All Limiters with MUSCL" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)

    wL = SVector(1.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.1)
    mesh = StructuredMesh1D(0.0, 1.0, 100)

    limiters = [
        MinmodLimiter(),
        SuperbeeLimiter(),
        VanLeerLimiter(),
        VenkatakrishnanLimiter(),
        KorenLimiter(),
        OspreLimiter(),
    ]

    for lim in limiters
        @testset "$(nameof(typeof(lim)))" begin
            prob = HyperbolicProblem(
                law, mesh, HLLSolver(), CellCenteredMUSCL(lim),
                DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
                x -> x < 0.5 ? wL : wR;
                final_time = 0.2, cfl = 0.4
            )

            x, U, t = solve_hyperbolic(prob)
            W = to_primitive(law, U)

            # Solution should be physically valid
            for i in 1:100
                @test W[i][1] > 0  # ρ > 0
                @test W[i][3] > 0  # P > 0
            end

            # Should reach final time
            @test t ≈ 0.2 atol = 1e-10

            # L1 error should be reasonable
            dx = 1.0 / 100
            ρ_err = sum(abs(W[i][1] - sod_exact(x[i], 0.2)[1]) * dx for i in 1:100)
            @test ρ_err < 0.10  # All limiters should give reasonable results
        end
    end
end

@testset "First Order (NoReconstruction)" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)

    wL = SVector(1.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.1)
    mesh = StructuredMesh1D(0.0, 1.0, 100)

    prob = HyperbolicProblem(
        law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        x -> x < 0.5 ? wL : wR;
        final_time = 0.2, cfl = 0.5
    )

    x, U, t = solve_hyperbolic(prob; method = :euler)
    W = to_primitive(law, U)

    # Should complete and give valid results (more diffusive than MUSCL)
    @test t ≈ 0.2 atol = 1e-10
    for i in 1:100
        @test W[i][1] > 0
        @test W[i][3] > 0
    end
end

@testset "Boundary Conditions" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 50)
    N = nvariables(law)

    @testset "TransmissiveBC" begin
        w = SVector(1.0, 0.5, 1.0)
        U = Vector{SVector{N, Float64}}(undef, 54)
        for i in 3:52
            U[i] = primitive_to_conserved(law, w)
        end
        FiniteVolumeMethod.apply_bc_left!(U, TransmissiveBC(), law, 50, 0.0)
        FiniteVolumeMethod.apply_bc_right!(U, TransmissiveBC(), law, 50, 0.0)
        @test U[1] == U[3]
        @test U[2] == U[3]
        @test U[53] == U[52]
        @test U[54] == U[52]
    end

    @testset "ReflectiveBC" begin
        w = SVector(1.0, 0.5, 1.0)
        U = Vector{SVector{N, Float64}}(undef, 54)
        for i in 3:52
            U[i] = primitive_to_conserved(law, w)
        end
        FiniteVolumeMethod.apply_bc_left!(U, ReflectiveBC(), law, 50, 0.0)
        w_ghost = conserved_to_primitive(law, U[2])
        @test w_ghost[1] ≈ 1.0     # ρ preserved
        @test w_ghost[2] ≈ -0.5    # v negated
        @test w_ghost[3] ≈ 1.0     # P preserved
    end

    @testset "PeriodicHyperbolicBC" begin
        U = Vector{SVector{N, Float64}}(undef, 54)
        for i in 3:52
            w = SVector(1.0 + 0.1 * (i - 2), 0.0, 1.0)
            U[i] = primitive_to_conserved(law, w)
        end
        FiniteVolumeMethod.apply_periodic_bcs!(U, law, 50, 0.0)
        @test U[2] == U[52]  # left ghost 1 = last interior
        @test U[1] == U[51]  # left ghost 2 = second-to-last interior
        @test U[53] == U[3]  # right ghost 1 = first interior
        @test U[54] == U[4]  # right ghost 2 = second interior
    end
end

@testset "2D Euler Types" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    @test nvariables(law) == 4

    w = SVector(1.0, 0.5, 0.3, 1.0)  # ρ, vx, vy, P
    u = primitive_to_conserved(law, w)
    w2 = conserved_to_primitive(law, u)
    @test w2 ≈ w atol = 1e-14

    # X-flux
    fx = physical_flux(law, w, 1)
    @test fx[1] ≈ 1.0 * 0.5  # ρvx

    # Y-flux
    fy = physical_flux(law, w, 2)
    @test fy[1] ≈ 1.0 * 0.3  # ρvy

    # Wave speeds
    λ_min, λ_max = wave_speeds(law, w, 1)
    c = sound_speed(eos, 1.0, 1.0)
    @test λ_min ≈ 0.5 - c
    @test λ_max ≈ 0.5 + c
end
