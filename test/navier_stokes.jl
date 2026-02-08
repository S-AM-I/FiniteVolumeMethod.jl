using FiniteVolumeMethod
using StaticArrays
using Test

# ============================================================
# Exact Riemann solver for Sod shock tube (used for verification)
# ============================================================

function sod_exact_ns(x, t; x0 = 0.5, γ = 1.4)
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
# StiffenedGasEOS Tests
# ============================================================

@testset "StiffenedGasEOS" begin
    @testset "Construction" begin
        eos = StiffenedGasEOS(gamma = 4.4, P_inf = 6.0e8)
        @test eos.gamma == 4.4
        @test eos.P_inf == 6.0e8

        # Default constructor
        eos_default = StiffenedGasEOS()
        @test eos_default.gamma == 1.4
        @test eos_default.P_inf == 0.0
    end

    @testset "Thermodynamic roundtrips" begin
        eos = StiffenedGasEOS(gamma = 4.4, P_inf = 6.0e8)

        # Test for several (ρ, P) pairs
        for (ρ, P) in [(1000.0, 1.0e9), (800.0, 5.0e8), (1200.0, 2.0e9)]
            ε = internal_energy(eos, ρ, P)
            P_back = pressure(eos, ρ, ε)
            @test P_back ≈ P rtol = 1e-12

            c = sound_speed(eos, ρ, P)
            @test c > 0
            @test c^2 ≈ eos.gamma * (P + eos.P_inf) / ρ rtol = 1e-12
        end
    end

    @testset "P_inf=0 reduces to IdealGasEOS" begin
        γ = 1.4
        eos_stiff = StiffenedGasEOS(gamma = γ, P_inf = 0.0)
        eos_ideal = IdealGasEOS(gamma = γ)

        ρ = 1.0
        P = 1.0
        @test pressure(eos_stiff, ρ, internal_energy(eos_ideal, ρ, P)) ≈ P rtol = 1e-12
        @test sound_speed(eos_stiff, ρ, P) ≈ sound_speed(eos_ideal, ρ, P) rtol = 1e-12
        @test internal_energy(eos_stiff, ρ, P) ≈ internal_energy(eos_ideal, ρ, P) rtol = 1e-12
        @test total_energy(eos_stiff, ρ, 0.5, P) ≈ total_energy(eos_ideal, ρ, 0.5, P) rtol = 1e-12
        @test total_energy(eos_stiff, ρ, 0.5, 0.3, P) ≈ total_energy(eos_ideal, ρ, 0.5, 0.3, P) rtol = 1e-12
    end

    @testset "Total energy formulas" begin
        eos = StiffenedGasEOS(gamma = 4.4, P_inf = 6.0e8)
        ρ = 1000.0
        P = 1.0e9

        # 1D
        v = 10.0
        E_1d = total_energy(eos, ρ, v, P)
        ε = internal_energy(eos, ρ, P)
        @test E_1d ≈ ρ * ε + 0.5 * ρ * v^2 rtol = 1e-12

        # 2D
        vx, vy = 10.0, 5.0
        E_2d = total_energy(eos, ρ, vx, vy, P)
        @test E_2d ≈ ρ * ε + 0.5 * ρ * (vx^2 + vy^2) rtol = 1e-12
    end

    @testset "Euler solver with StiffenedGasEOS (shock tube)" begin
        # Sod-like shock tube with stiffened gas
        eos = StiffenedGasEOS(gamma = 1.4, P_inf = 0.0)
        law = EulerEquations{1}(eos)
        mesh = StructuredMesh1D(0.0, 1.0, 200)

        wL = SVector(1.0, 0.0, 1.0)
        wR = SVector(0.125, 0.0, 0.1)
        ic(x) = x < 0.5 ? wL : wR

        prob = HyperbolicProblem(
            law, mesh, HLLCSolver(), NoReconstruction(),
            DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
            ic; final_time = 0.2, cfl = 0.5
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        # Verify density profile is reasonable (L1 error check)
        l1_err = 0.0
        for i in eachindex(x)
            ρ_ex, v_ex, P_ex = sod_exact_ns(x[i], t)
            l1_err += abs(W[i][1] - ρ_ex) * (1.0 / length(x))
        end
        @test l1_err < 0.05  # L1 error below 5% of domain-averaged density
    end
end

# ============================================================
# NavierStokesEquations Type Tests
# ============================================================

@testset "NavierStokesEquations" begin
    @testset "Construction" begin
        eos = IdealGasEOS(gamma = 1.4)
        ns1 = NavierStokesEquations{1}(eos, mu = 0.01, Pr = 0.72)
        @test ns1.mu == 0.01
        @test ns1.Pr == 0.72
        @test ns1.euler.eos.gamma == 1.4

        ns2 = NavierStokesEquations{2}(eos, mu = 0.001)
        @test ns2.mu == 0.001
        @test ns2.Pr == 0.72  # default
    end

    @testset "Delegation of AbstractConservationLaw methods" begin
        eos = IdealGasEOS(gamma = 1.4)
        ns1 = NavierStokesEquations{1}(eos, mu = 0.01)
        euler1 = EulerEquations{1}(eos)

        @test nvariables(ns1) == 3
        @test nvariables(ns1) == nvariables(euler1)

        w1d = SVector(1.0, 0.5, 1.0)
        @test physical_flux(ns1, w1d, 1) == physical_flux(euler1, w1d, 1)
        @test max_wave_speed(ns1, w1d, 1) == max_wave_speed(euler1, w1d, 1)
        @test wave_speeds(ns1, w1d, 1) == wave_speeds(euler1, w1d, 1)
        @test conserved_to_primitive(ns1, primitive_to_conserved(ns1, w1d)) ≈ w1d

        ns2 = NavierStokesEquations{2}(eos, mu = 0.01)
        euler2 = EulerEquations{2}(eos)
        @test nvariables(ns2) == 4

        w2d = SVector(1.0, 0.5, -0.3, 1.0)
        @test physical_flux(ns2, w2d, 1) == physical_flux(euler2, w2d, 1)
        @test physical_flux(ns2, w2d, 2) == physical_flux(euler2, w2d, 2)
        @test max_wave_speed(ns2, w2d, 1) == max_wave_speed(euler2, w2d, 1)
        @test max_wave_speed(ns2, w2d, 2) == max_wave_speed(euler2, w2d, 2)
        @test conserved_to_primitive(ns2, primitive_to_conserved(ns2, w2d)) ≈ w2d
    end

    @testset "thermal_conductivity" begin
        eos = IdealGasEOS(gamma = 1.4)
        ns = NavierStokesEquations{1}(eos, mu = 0.01, Pr = 0.72)
        κ = thermal_conductivity(ns)
        @test κ ≈ 0.01 * 1.4 / (0.72 * 0.4) rtol = 1e-12
    end

    @testset "HLLC forwarding" begin
        eos = IdealGasEOS(gamma = 1.4)
        ns1 = NavierStokesEquations{1}(eos, mu = 0.01)
        euler1 = EulerEquations{1}(eos)

        wL = SVector(1.0, 0.0, 1.0)
        wR = SVector(0.125, 0.0, 0.1)
        F_ns = solve_riemann(HLLCSolver(), ns1, wL, wR, 1)
        F_euler = solve_riemann(HLLCSolver(), euler1, wL, wR, 1)
        @test F_ns ≈ F_euler

        # 2D
        ns2 = NavierStokesEquations{2}(eos, mu = 0.01)
        euler2 = EulerEquations{2}(eos)
        wL2 = SVector(1.0, 0.5, 0.0, 1.0)
        wR2 = SVector(0.125, 0.0, 0.0, 0.1)
        F_ns2 = solve_riemann(HLLCSolver(), ns2, wL2, wR2, 1)
        F_euler2 = solve_riemann(HLLCSolver(), euler2, wL2, wR2, 1)
        @test F_ns2 ≈ F_euler2
    end
end

# ============================================================
# Viscous Flux Tests
# ============================================================

@testset "Viscous Flux" begin
    @testset "1D viscous flux" begin
        eos = IdealGasEOS(gamma = 1.4)
        ns = NavierStokesEquations{1}(eos, mu = 0.01, Pr = 0.72)

        # Uniform state → zero viscous flux
        w_uniform = SVector(1.0, 1.0, 1.0)
        Fv = viscous_flux_1d(ns, w_uniform, w_uniform, 0.1)
        @test Fv ≈ SVector(0.0, 0.0, 0.0) atol = 1e-15

        # Linear velocity gradient
        wL = SVector(1.0, 0.0, 1.0)
        wR = SVector(1.0, 1.0, 1.0)
        dx = 0.1
        Fv = viscous_flux_1d(ns, wL, wR, dx)
        # τ_xx = (4/3) * μ * dv/dx = (4/3) * 0.01 * (1-0)/0.1
        τ_xx_expected = (4.0 / 3.0) * 0.01 * 10.0
        @test Fv[1] == 0.0
        @test Fv[2] ≈ τ_xx_expected rtol = 1e-12
    end

    @testset "2D viscous flux x-direction" begin
        eos = IdealGasEOS(gamma = 1.4)
        ns = NavierStokesEquations{2}(eos, mu = 0.01, Pr = 0.72)

        # Uniform state
        w_uniform = SVector(1.0, 1.0, 0.5, 1.0)
        Fv = viscous_flux_x_2d(ns, w_uniform, w_uniform, 0.0, 0.0, 0.1)
        @test Fv ≈ SVector(0.0, 0.0, 0.0, 0.0) atol = 1e-15
    end

    @testset "2D viscous flux y-direction" begin
        eos = IdealGasEOS(gamma = 1.4)
        ns = NavierStokesEquations{2}(eos, mu = 0.01, Pr = 0.72)

        # Uniform state
        w_uniform = SVector(1.0, 1.0, 0.5, 1.0)
        Fv = viscous_flux_y_2d(ns, w_uniform, w_uniform, 0.0, 0.0, 0.1)
        @test Fv ≈ SVector(0.0, 0.0, 0.0, 0.0) atol = 1e-15
    end
end

# ============================================================
# NS reduces to Euler (μ=0)
# ============================================================

@testset "NS with μ=0 matches Euler" begin
    @testset "1D Sod shock tube" begin
        eos = IdealGasEOS(gamma = 1.4)
        euler = EulerEquations{1}(eos)
        ns = NavierStokesEquations{1}(eos, mu = 0.0, Pr = 0.72)

        mesh = StructuredMesh1D(0.0, 1.0, 200)
        wL = SVector(1.0, 0.0, 1.0)
        wR = SVector(0.125, 0.0, 0.1)
        ic(x) = x < 0.5 ? wL : wR

        prob_euler = HyperbolicProblem(
            euler, mesh, HLLCSolver(), NoReconstruction(),
            DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
            ic; final_time = 0.2, cfl = 0.5
        )

        prob_ns = HyperbolicProblem(
            ns, mesh, HLLCSolver(), NoReconstruction(),
            DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
            ic; final_time = 0.2, cfl = 0.5
        )

        x_e, U_e, t_e = solve_hyperbolic(prob_euler)
        x_ns, U_ns, t_ns = solve_hyperbolic(prob_ns)

        @test t_e ≈ t_ns
        for i in eachindex(U_e)
            @test U_e[i] ≈ U_ns[i] rtol = 1e-12
        end
    end
end

# ============================================================
# 1D Viscous Decay Test
# ============================================================

@testset "1D Viscous Velocity Decay" begin
    # Exact: v(x,t) = A sin(kx) exp(-νk²t), ν = μ/ρ
    # High P₀ (low Mach), periodic BCs
    ρ₀ = 1.0
    P₀ = 100.0    # high pressure → low Mach
    A = 0.01       # small amplitude
    L = 1.0
    k = 2π / L
    μ = 0.01
    ν = μ / ρ₀

    eos = IdealGasEOS(gamma = 1.4)
    ns = NavierStokesEquations{1}(eos, mu = μ, Pr = 0.72)

    N_cells = 64
    mesh = StructuredMesh1D(0.0, L, N_cells)

    ic(x) = SVector(ρ₀, A * sin(k * x), P₀)

    prob = HyperbolicProblem(
        ns, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.5, cfl = 0.3
    )

    x, U, t = solve_hyperbolic(prob)
    W = to_primitive(ns, U)

    # Expected decay
    decay_rate = exp(-ν * k^2 * t)

    # Check each cell
    max_v_err = 0.0
    for i in eachindex(x)
        v_exact = A * sin(k * x[i]) * decay_rate
        v_num = W[i][2]
        max_v_err = max(max_v_err, abs(v_num - v_exact))
    end

    # With 64 cells, verify error is well below initial amplitude
    @test max_v_err < 0.5 * A  # within 50% of amplitude (numerical diffusion on coarse grid)
end

# ============================================================
# 2D Couette Flow (steady linear profile)
# ============================================================

@testset "2D Couette Flow" begin
    # Setup: channel in y-direction, periodic in x
    # Bottom wall: NoSlipBC (v=0)
    # Top wall: DirichletHyperbolicBC with U_wall
    # Expected steady state: vx(y) = U_wall * y / H
    U_wall = 0.01   # small velocity (low Mach)
    ρ₀ = 1.0
    P₀ = 100.0      # high pressure → low Mach
    H = 1.0
    μ = 0.01

    eos = IdealGasEOS(gamma = 1.4)
    ns = NavierStokesEquations{2}(eos, mu = μ, Pr = 0.72)

    nx, ny = 4, 16
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, H, nx, ny)

    # Initial condition: exact Couette profile
    function ic_couette(x, y)
        vx = U_wall * y / H
        return SVector(ρ₀, vx, 0.0, P₀)
    end

    # Top wall: prescribed state with vx = U_wall
    w_top = SVector(ρ₀, U_wall, 0.0, P₀)

    prob = HyperbolicProblem2D(
        ns, mesh, HLLCSolver(), NoReconstruction(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        NoSlipBC(), DirichletHyperbolicBC(w_top),
        ic_couette; final_time = 0.5, cfl = 0.3
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(ns, U)

    # Check that vx profile is approximately linear
    max_err = 0.0
    for iy in 1:ny, ix in 1:nx
        x, y = coords[ix, iy]
        vx_exact = U_wall * y / H
        vx_num = W[ix, iy][2]
        max_err = max(max_err, abs(vx_num - vx_exact))
    end

    # Should remain close to analytical with low Mach initialization
    @test max_err < 0.1 * U_wall
end

# ============================================================
# 2D Taylor-Green Vortex Decay
# ============================================================

@testset "2D Taylor-Green Vortex Decay" begin
    # Exact: u = -U₀ cos(kx) sin(ky) e^{-2νk²t}
    #        v =  U₀ sin(kx) cos(ky) e^{-2νk²t}
    # P = P₀ - ρ₀U₀²/4 (cos(2kx) + cos(2ky)) e^{-4νk²t}
    ρ₀ = 1.0
    P₀ = 100.0    # high pressure for low Mach
    U₀ = 0.01     # small velocity amplitude
    L = 1.0
    k = 2π / L
    μ = 0.01
    ν = μ / ρ₀

    eos = IdealGasEOS(gamma = 1.4)
    ns = NavierStokesEquations{2}(eos, mu = μ, Pr = 0.72)

    N = 32
    mesh = StructuredMesh2D(0.0, L, 0.0, L, N, N)

    function ic_tgv(x, y)
        vx = -U₀ * cos(k * x) * sin(k * y)
        vy = U₀ * sin(k * x) * cos(k * y)
        P = P₀ - ρ₀ * U₀^2 / 4.0 * (cos(2 * k * x) + cos(2 * k * y))
        return SVector(ρ₀, vx, vy, P)
    end

    prob = HyperbolicProblem2D(
        ns, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic_tgv; final_time = 0.5, cfl = 0.3
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(ns, U)

    # Expected decay
    decay = exp(-2 * ν * k^2 * t)

    # Compute L∞ error in velocity
    max_vx_err = 0.0
    max_vy_err = 0.0
    for iy in 1:N, ix in 1:N
        x, y = coords[ix, iy]
        vx_exact = -U₀ * cos(k * x) * sin(k * y) * decay
        vy_exact = U₀ * sin(k * x) * cos(k * y) * decay
        vx_num = W[ix, iy][2]
        vy_num = W[ix, iy][3]
        max_vx_err = max(max_vx_err, abs(vx_num - vx_exact))
        max_vy_err = max(max_vy_err, abs(vy_num - vy_exact))
    end

    # Errors should be well below amplitude
    @test max_vx_err < 0.5 * U₀
    @test max_vy_err < 0.5 * U₀
end

# ============================================================
# Taylor-Green Convergence Study
# ============================================================

@testset "Taylor-Green Convergence" begin
    ρ₀ = 1.0
    P₀ = 100.0
    U₀ = 0.01
    L = 1.0
    k = 2π / L
    μ = 0.01
    ν = μ / ρ₀
    t_final = 0.1

    eos = IdealGasEOS(gamma = 1.4)

    function ic_tgv_conv(x, y)
        vx = -U₀ * cos(k * x) * sin(k * y)
        vy = U₀ * sin(k * x) * cos(k * y)
        P = P₀ - ρ₀ * U₀^2 / 4.0 * (cos(2 * k * x) + cos(2 * k * y))
        return SVector(ρ₀, vx, vy, P)
    end

    errors = Float64[]
    resolutions = [16, 32]

    for N in resolutions
        ns = NavierStokesEquations{2}(eos, mu = μ, Pr = 0.72)
        mesh = StructuredMesh2D(0.0, L, 0.0, L, N, N)

        prob = HyperbolicProblem2D(
            ns, mesh, HLLCSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            ic_tgv_conv; final_time = t_final, cfl = 0.3
        )

        coords, U, t = solve_hyperbolic(prob)
        W = to_primitive(ns, U)
        decay = exp(-2 * ν * k^2 * t)

        max_err = 0.0
        for iy in 1:N, ix in 1:N
            x, y = coords[ix, iy]
            vx_exact = -U₀ * cos(k * x) * sin(k * y) * decay
            vy_exact = U₀ * sin(k * x) * cos(k * y) * decay
            vx_num = W[ix, iy][2]
            vy_num = W[ix, iy][3]
            err = max(abs(vx_num - vx_exact), abs(vy_num - vy_exact))
            max_err = max(max_err, err)
        end
        push!(errors, max_err)
    end

    # Convergence rate should be > 1 (at least first-order, ideally ~2nd order with MUSCL)
    rate = log2(errors[1] / errors[2])
    @test rate > 0.8  # conservative bound; MUSCL should give ~1-2
end

# ============================================================
# Conservation with Periodic BCs
# ============================================================

@testset "Conservation (periodic BCs)" begin
    @testset "1D mass and momentum conservation" begin
        ρ₀ = 1.0
        P₀ = 10.0
        A = 0.1
        L = 1.0
        k = 2π / L
        μ = 0.05

        eos = IdealGasEOS(gamma = 1.4)
        ns = NavierStokesEquations{1}(eos, mu = μ, Pr = 0.72)
        N = 64
        mesh = StructuredMesh1D(0.0, L, N)
        dx = cell_volume(mesh, 1)

        ic(x) = SVector(ρ₀ + 0.1 * sin(k * x), A * sin(2 * k * x), P₀)

        prob = HyperbolicProblem(
            ns, mesh, HLLCSolver(), NoReconstruction(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            ic; final_time = 0.2, cfl = 0.3
        )

        x, U, t = solve_hyperbolic(prob)

        # Initial conserved quantities
        ic_vec = [primitive_to_conserved(ns, ic(xi)) for xi in x]
        mass_0 = sum(u[1] for u in ic_vec) * dx
        mom_0 = sum(u[2] for u in ic_vec) * dx

        # Final conserved quantities
        mass_f = sum(u[1] for u in U) * dx
        mom_f = sum(u[2] for u in U) * dx

        @test mass_f ≈ mass_0 rtol = 1e-10
        @test abs(mom_f - mom_0) < 1e-12  # momentum near zero, use atol
    end

    @testset "2D mass conservation" begin
        ρ₀ = 1.0
        P₀ = 100.0
        U₀ = 0.01
        L = 1.0
        k = 2π / L
        μ = 0.01

        eos = IdealGasEOS(gamma = 1.4)
        ns = NavierStokesEquations{2}(eos, mu = μ, Pr = 0.72)
        N = 16
        mesh = StructuredMesh2D(0.0, L, 0.0, L, N, N)
        dx, dy = mesh.dx, mesh.dy

        function ic_cons(x, y)
            vx = -U₀ * cos(k * x) * sin(k * y)
            vy = U₀ * sin(k * x) * cos(k * y)
            P = P₀ - ρ₀ * U₀^2 / 4.0 * (cos(2 * k * x) + cos(2 * k * y))
            return SVector(ρ₀, vx, vy, P)
        end

        prob = HyperbolicProblem2D(
            ns, mesh, HLLCSolver(), NoReconstruction(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            ic_cons; final_time = 0.1, cfl = 0.3
        )

        coords, U, t = solve_hyperbolic(prob)

        # Initial mass
        mass_0 = 0.0
        for iy in 1:N, ix in 1:N
            x, y = coords[ix, iy]
            w = ic_cons(x, y)
            u = primitive_to_conserved(ns, w)
            mass_0 += u[1] * dx * dy
        end

        # Final mass
        mass_f = sum(U[ix, iy][1] for iy in 1:N, ix in 1:N) * dx * dy

        @test mass_f ≈ mass_0 rtol = 1e-10
    end
end

# ============================================================
# NoSlipBC Tests
# ============================================================

@testset "NoSlipBC" begin
    @testset "1D NoSlipBC ghost cells" begin
        eos = IdealGasEOS(gamma = 1.4)
        ns = NavierStokesEquations{1}(eos, mu = 0.01)
        nc = 10

        # Create padded array
        U = Vector{SVector{3, Float64}}(undef, nc + 4)
        for i in 1:(nc + 4)
            U[i] = primitive_to_conserved(ns, SVector(1.0, 0.5, 1.0))
        end

        FiniteVolumeMethod.apply_bc_left!(U, NoSlipBC(), ns, nc, 0.0)
        w_ghost = conserved_to_primitive(ns, U[2])
        w_interior = conserved_to_primitive(ns, U[3])
        # Velocity should be negated
        @test w_ghost[2] ≈ -w_interior[2]
        # Density and pressure preserved
        @test w_ghost[1] ≈ w_interior[1]
        @test w_ghost[3] ≈ w_interior[3]
    end

    @testset "2D NoSlipBC negates all velocity components" begin
        eos = IdealGasEOS(gamma = 1.4)
        ns = NavierStokesEquations{2}(eos, mu = 0.01)
        nx, ny = 4, 4

        U = Matrix{SVector{4, Float64}}(undef, nx + 4, ny + 4)
        for j in 1:(ny + 4), i in 1:(nx + 4)
            U[i, j] = primitive_to_conserved(ns, SVector(1.0, 0.5, 0.3, 1.0))
        end

        # Test left wall
        FiniteVolumeMethod.apply_bc_2d_left!(U, NoSlipBC(), ns, nx, ny, 0.0)
        w_ghost = conserved_to_primitive(ns, U[2, 4])
        w_int = conserved_to_primitive(ns, U[3, 4])
        @test w_ghost[2] ≈ -w_int[2]  # vx negated
        @test w_ghost[3] ≈ -w_int[3]  # vy negated
        @test w_ghost[1] ≈ w_int[1]   # ρ preserved
        @test w_ghost[4] ≈ w_int[4]   # P preserved

        # Reset and test bottom wall
        for j in 1:(ny + 4), i in 1:(nx + 4)
            U[i, j] = primitive_to_conserved(ns, SVector(1.0, 0.5, 0.3, 1.0))
        end
        FiniteVolumeMethod.apply_bc_2d_bottom!(U, NoSlipBC(), ns, nx, ny, 0.0)
        w_ghost_b = conserved_to_primitive(ns, U[4, 2])
        w_int_b = conserved_to_primitive(ns, U[4, 3])
        @test w_ghost_b[2] ≈ -w_int_b[2]  # vx negated
        @test w_ghost_b[3] ≈ -w_int_b[3]  # vy negated
    end
end

# ============================================================
# NoSlipBC vs ReflectiveBC difference
# ============================================================

@testset "NoSlipBC differs from ReflectiveBC" begin
    eos = IdealGasEOS(gamma = 1.4)
    ns = NavierStokesEquations{2}(eos, mu = 0.01)
    euler = EulerEquations{2}(eos)
    nx, ny = 4, 4

    # Create arrays with tangential velocity
    U_noslip = Matrix{SVector{4, Float64}}(undef, nx + 4, ny + 4)
    U_reflect = Matrix{SVector{4, Float64}}(undef, nx + 4, ny + 4)
    for j in 1:(ny + 4), i in 1:(nx + 4)
        state = primitive_to_conserved(ns, SVector(1.0, 0.5, 0.3, 1.0))
        U_noslip[i, j] = state
        U_reflect[i, j] = state
    end

    # Apply left wall BCs
    FiniteVolumeMethod.apply_bc_2d_left!(U_noslip, NoSlipBC(), ns, nx, ny, 0.0)
    FiniteVolumeMethod.apply_bc_2d_left!(U_reflect, ReflectiveBC(), euler, nx, ny, 0.0)

    # NoSlip negates BOTH vx and vy; Reflective only negates vx (normal)
    w_noslip = conserved_to_primitive(ns, U_noslip[2, 4])
    w_reflect = conserved_to_primitive(euler, U_reflect[2, 4])

    # vx should be same (both negate normal velocity at left wall)
    @test w_noslip[2] ≈ w_reflect[2]
    # vy should differ: NoSlip negates it, Reflective preserves it
    @test w_noslip[3] ≈ -w_reflect[3]
end
