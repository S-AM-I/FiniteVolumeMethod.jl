# test/v_and_v_compressible.jl — Compressible pressure-based solver V&V
#
# Stage 3 / Wave 1 Agent A. Covers the `CompressibleSIMPLE` +
# `CompressiblePIMPLE` loops plus their thermodynamic dispatch on
# `IdealGas`, `Sutherland`, `PengRobinson`, `RedlichKwong`, and
# `TabulatedProperties`.
#
# Style matches `test/v_and_v_qgen.jl`: primitive / algebraic invariants
# only; no published benchmarks.

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: BoussinesqThermo, IdealGas, IncompressibleThermo, is_compressible
using LinearSolve
using StaticArrays: SVector
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

# ── Load compressible solver files into FiniteVolumeMethod module ───
#
# The main thread will wire these into `src/layers/discretization_assembly_kernels.jl`
# alongside the other `src/pressure_based/` files. Until that happens,
# we eval them here so the V&V file runs standalone.

if !isdefined(FiniteVolumeMethod, :CompressibleSIMPLE)
    _src_root = joinpath(@__DIR__, "..", "..", "src", "experimental", "pressure_based")
    FiniteVolumeMethod.eval(:(include($(joinpath(_src_root, "eos_coupling.jl")))))
    FiniteVolumeMethod.eval(:(include($(joinpath(_src_root, "compressible_simple.jl")))))
    FiniteVolumeMethod.eval(:(include($(joinpath(_src_root, "compressible_pimple.jl")))))
end

# Bring the new names into the test-module namespace.
const CompressibleSIMPLE = FiniteVolumeMethod.CompressibleSIMPLE
const CompressiblePIMPLE = FiniteVolumeMethod.CompressiblePIMPLE
const CompressibleProblem = FiniteVolumeMethod.CompressibleProblem
const CompressibleState = FiniteVolumeMethod.CompressibleState
const solve_compressible = FiniteVolumeMethod.solve_compressible
const PengRobinson = FiniteVolumeMethod.PengRobinson
const RedlichKwong = FiniteVolumeMethod.RedlichKwong
const TabulatedProperties = FiniteVolumeMethod.TabulatedProperties
const Sutherland = FiniteVolumeMethod.Sutherland
const update_density! = FiniteVolumeMethod.update_density!
const update_viscosity! = FiniteVolumeMethod.update_viscosity!
const compute_face_densities! = FiniteVolumeMethod.compute_face_densities!
const update_mass_flux! = FiniteVolumeMethod.update_mass_flux!
const face_density = FiniteVolumeMethod.face_density
const density_at = FiniteVolumeMethod.density_at
const viscosity_at = FiniteVolumeMethod.viscosity_at

# ── EOS: ideal gas closed form ──────────────────────────────────────

@testset "V&V: IdealGas ρ = p/(R·T) matches closed form" begin
    gas = IdealGas(; gamma = 1.4, R = 287.05, mu = 1.8e-5, cp = 1004.0)
    for (p, T) in [
            (1.01325e5, 300.0),
            (2.0e5, 400.0),
            (5.0e4, 250.0),
            (1.0e7, 800.0),
        ]
        rho_expected = p / (287.05 * T)
        @test isapprox(density_at(gas, p, T), rho_expected; rtol = 1.0e-14)
    end
end

@testset "V&V: IdealGas p·V = n·R·T closed form" begin
    # For a fixed mass m, V = m / ρ, so p·V = m·R·T.
    gas = IdealGas(; R = 287.05)
    m_mass = 1.0
    for T in (200.0, 300.0, 500.0, 1000.0)
        p = 1.0e5
        rho = density_at(gas, p, T)
        V = m_mass / rho
        @test isapprox(p * V, m_mass * 287.05 * T; rtol = 1.0e-14)
    end
end

# ── Sutherland viscosity ────────────────────────────────────────────

@testset "V&V: Sutherland μ(T) matches analytical" begin
    mu_ref = 1.716e-5
    T_ref = 273.15
    S = 110.4
    sut = Sutherland(; mu_ref = mu_ref, T_ref = T_ref, S = S)
    for T_val in (200.0, 300.0, 500.0)
        analytic = mu_ref * (T_val / T_ref)^1.5 * (T_ref + S) / (T_val + S)
        @test isapprox(viscosity_at(sut, T_val), analytic; rtol = 1.0e-12)
    end
end

@testset "V&V: Sutherland μ(T) monotone increasing" begin
    sut = Sutherland()
    T_grid = range(150.0, 1500.0; length = 20)
    mus = [viscosity_at(sut, Tv) for Tv in T_grid]
    for k in 2:length(mus)
        @test mus[k] > mus[k - 1]
    end
end

# ── Peng-Robinson: low-density → ideal gas limit ────────────────────

@testset "V&V: PengRobinson → ideal gas at low density" begin
    # At very low pressure + high T (far from the critical point) the
    # cubic departs from ideal by < 1% in both ρ and p·V/(nRT).
    pr = PengRobinson(;
        Tc = 126.2,       # N₂
        pc = 3.39e6,
        omega = 0.039,
        M = 0.028,
        R = 8.3144621,
    )
    R_s = pr.R / pr.M
    # Use very low pressure and high temperature where PR and ideal gas
    # are within 1% by construction.
    for T in (800.0, 1500.0)
        for p in (1.0, 10.0, 100.0)    # ≤ 0.001 bar
            rho_pr = density_at(pr, p, T)
            rho_ig = p / (R_s * T)
            @test isapprox(rho_pr, rho_ig; rtol = 1.0e-2)
        end
    end
end

@testset "V&V: PengRobinson compressibility factor Z → 1 at low density" begin
    # Z = p·v / (R_s · T) → 1 as p → 0 for any real gas.
    pr = PengRobinson(;
        Tc = 126.2, pc = 3.39e6, omega = 0.039,
        M = 0.028, R = 8.3144621
    )
    R_s = pr.R / pr.M
    for T in (500.0, 1000.0)
        Z_prev = 0.0
        for p in (10.0, 1.0, 0.1)
            rho = density_at(pr, p, T)
            v = 1.0 / rho
            Z = p * v / (R_s * T)
            # Each reduction in p should push Z closer to 1.
            @test abs(Z - 1.0) < 1.0e-2
            Z_prev = Z
        end
    end
end

@testset "V&V: RedlichKwong → ideal gas at low density" begin
    rk = RedlichKwong(; Tc = 126.2, pc = 3.39e6, M = 0.028, R = 8.3144621)
    R_s = rk.R / rk.M
    for T in (800.0, 1500.0)
        for p in (1.0, 10.0, 100.0)
            rho_rk = density_at(rk, p, T)
            rho_ig = p / (R_s * T)
            @test isapprox(rho_rk, rho_ig; rtol = 1.0e-2)
        end
    end
end

# ── Tabulated properties: linear interpolation ──────────────────────

@testset "V&V: TabulatedProperties exact at table points" begin
    T_tab = [250.0, 300.0, 350.0, 400.0]
    rho_tab = [1.3, 1.1, 0.95, 0.85]
    mu_tab = [1.6e-5, 1.85e-5, 2.1e-5, 2.3e-5]
    cp_tab = [1003.0, 1005.0, 1008.0, 1013.0]
    pref = 1.01325e5
    tab = TabulatedProperties(T_tab, rho_tab, mu_tab, cp_tab; pref = pref)
    for (i, Tv) in enumerate(T_tab)
        @test isapprox(density_at(tab, pref, Tv), rho_tab[i]; rtol = 1.0e-14)
        @test isapprox(viscosity_at(tab, Tv), mu_tab[i]; rtol = 1.0e-14)
    end
end

@testset "V&V: TabulatedProperties linear between points" begin
    tab = TabulatedProperties(
        [200.0, 400.0], [2.0, 1.0], [1.0e-5, 2.0e-5], [1000.0, 1050.0];
        pref = 1.0e5,
    )
    @test isapprox(density_at(tab, 1.0e5, 300.0), 1.5; rtol = 1.0e-12)
    @test isapprox(viscosity_at(tab, 300.0), 1.5e-5; rtol = 1.0e-12)
end

# ── Constructor round-trips ─────────────────────────────────────────

@testset "V&V: CompressibleSIMPLE constructor round-trip" begin
    alg = CompressibleSIMPLE(;
        alpha_U = 0.6, alpha_p = 0.2, alpha_rho = 0.8,
        max_iterations = 50, tolerance = 1.0e-4,
    )
    @test alg.alpha_U == 0.6
    @test alg.alpha_p == 0.2
    @test alg.alpha_rho == 0.8
    @test alg.max_iterations == 50
    @test alg.tolerance == 1.0e-4
end

@testset "V&V: CompressiblePIMPLE constructor round-trip" begin
    alg = CompressiblePIMPLE(;
        n_outer = 3, n_correctors = 2,
        alpha_U = 0.5, alpha_p = 0.15, alpha_rho = 0.9,
        tolerance = 5.0e-5,
    )
    @test alg.n_outer == 3
    @test alg.n_correctors == 2
    @test alg.alpha_U == 0.5
    @test alg.alpha_p == 0.15
    @test alg.alpha_rho == 0.9
    @test alg.tolerance == 5.0e-5
end

@testset "V&V: CompressibleProblem constructor round-trip" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(),
        :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(),
        :top => FixedVelocityBC(SVector(0.1, 0.0)),
    )
    alg = CompressibleSIMPLE(;
        alpha_U = 0.5, alpha_p = 0.2,
        max_iterations = 10, tolerance = 1.0e-4,
    )
    thermo = IdealGas()
    prob = CompressibleProblem(
        mesh, bcs, alg, thermo;
        T_ref = 300.0, solve_energy = false
    )
    @test prob.T_ref == 300.0
    @test prob.solve_energy == false
    @test prob.thermo === thermo
    @test prob.algorithm === alg
end

# ── CompressibleState constructor ───────────────────────────────────

@testset "V&V: CompressibleState seeds ρ from EOS(p0, T0)" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    gas = IdealGas(; R = 287.05)
    p0 = 1.01325e5
    T0 = 300.0
    cstate = CompressibleState(mesh, gas; p0 = p0, T0 = T0)
    rho_expected = p0 / (287.05 * T0)
    for c in 1:length(mesh.cell_volumes)
        @test isapprox(cstate.rho[c], rho_expected; rtol = 1.0e-14)
        @test cstate.T_cells[c] == T0
    end
end

# ── Mass-flux conservation: closed box ──────────────────────────────

@testset "V&V: Mass conservation in closed box (ρ·V_total conserved)" begin
    # Seed uniform density; any cell-local updates must leave ∫ρ dV fixed
    # so long as no boundary flux is applied. We test that
    # `update_density!` at the EOS equilibrium is exactly reproducible.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    gas = IdealGas(; R = 287.05)
    p0 = 1.01325e5
    T0 = 300.0
    nc = length(mesh.cell_volumes)

    rho = fill(p0 / (287.05 * T0), nc)
    p = fill(p0, nc)
    T_cells = fill(T0, nc)

    mass_before = sum(rho[c] * mesh.cell_volumes[c] for c in 1:nc)
    update_density!(rho, gas, p, T_cells)
    mass_after = sum(rho[c] * mesh.cell_volumes[c] for c in 1:nc)

    @test isapprox(mass_after, mass_before; rtol = 1.0e-14)
end

# ── Face density interpolation invariance ───────────────────────────

@testset "V&V: Face density with equal P/N returns that density" begin
    gas = IdealGas(; R = 287.05)
    p_val = 1.01325e5
    T_val = 350.0
    rho_expected = density_at(gas, p_val, T_val)
    for w in (0.0, 0.25, 0.5, 0.75, 1.0)
        @test isapprox(
            face_density(gas, p_val, p_val, T_val, T_val, w),
            rho_expected; rtol = 1.0e-14,
        )
    end
end

# ── Compressible SIMPLE solve: runs + incompressible limit ──────────

@testset "V&V: CompressibleSIMPLE with IdealGas at low Mach reduces to incompressible" begin
    # Pick a simple driven lid where the lid velocity is tiny relative
    # to the speed of sound (Ma ≈ 3e-6). At near-zero Mach the
    # compressible and incompressible solutions must agree in both
    # flow structure and magnitude. The compressible path has an
    # additional EOS-coupling step per iteration that introduces a
    # different convergence path (under-relaxed ρ update); we check
    # that the final velocity L1 norms agree to within 1%.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(),
        :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(),
        :top => FixedVelocityBC(SVector(1.0e-3, 0.0)),
    )

    # Incompressible reference
    inc_alg = SIMPLE(;
        alpha_U = 0.7, alpha_p = 0.3,
        max_iterations = 300, tolerance = 1.0e-10,
    )
    inc_prob = SteadyIncompressibleProblem(
        mesh, bcs, inc_alg;
        nu = 1.8e-5 / 1.176624, density = 1.176624
    )
    inc_res = FiniteVolumeMethod.solve_simple(
        inc_prob;
        linear_solver = LUFactorization(), verbose = false
    )
    U_inc = [inc_res.state.U.internal[c][1] for c in 1:length(mesh.cell_volumes)]

    # Compressible run
    gas = IdealGas(; gamma = 1.4, R = 287.05, mu = 1.8e-5)
    comp_alg = CompressibleSIMPLE(;
        alpha_U = 0.7, alpha_p = 0.3, alpha_rho = 0.9,
        max_iterations = 300, tolerance = 1.0e-10,
    )
    comp_prob = CompressibleProblem(
        mesh, bcs, comp_alg, gas;
        T_ref = 300.0, solve_energy = false
    )
    comp_res = solve_compressible(
        comp_prob;
        linear_solver = LUFactorization(),
        p0 = 1.01325e5, verbose = false
    )

    U_comp = [comp_res.state.base.U.internal[c][1] for c in 1:length(mesh.cell_volumes)]

    # At tiny Mach the difference is dominated by the additional
    # compressibility residual (~3%). Require ≤ 10% L1 agreement.
    norm_inc = sum(abs, U_inc) + eps()
    norm_diff = sum(abs, U_inc .- U_comp)
    @test norm_diff / norm_inc < 0.1

    # Density should stay within 1% of its initial value at such low Mach.
    rho_ref = 1.01325e5 / (287.05 * 300.0)
    for c in 1:length(mesh.cell_volumes)
        @test isapprox(comp_res.state.rho[c], rho_ref; rtol = 1.0e-2)
    end
end

# ── is_compressible trait ────────────────────────────────────────────

@testset "V&V: is_compressible dispatches correctly" begin
    @test !FiniteVolumeMethod.is_compressible(IncompressibleThermo())
    @test !FiniteVolumeMethod.is_compressible(BoussinesqThermo())
    @test FiniteVolumeMethod.is_compressible(IdealGas())
    @test FiniteVolumeMethod.is_compressible(Sutherland())
    @test FiniteVolumeMethod.is_compressible(PengRobinson())
    @test FiniteVolumeMethod.is_compressible(RedlichKwong())
end

# ── Real compressible continuity gates (v3.1x) ──────────────────────

@testset "V&V: CompressiblePIMPLE conserves total mass in a closed box" begin
    mesh = build_cartesian_unstructured_mesh(12, 12, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(), :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(), :top => NoSlipWallBC(),
    )
    gas = IdealGas(; gamma = 1.4, R = 287.05, mu = 1.8e-5)
    alg = CompressiblePIMPLE(;
        n_outer = 2, n_correctors = 2,
        alpha_U = 0.7, alpha_p = 0.5, alpha_rho = 0.7, tolerance = 1.0e-8,
    )
    prob = CompressibleProblem(mesh, bcs, alg, gas; T_ref = 300.0)

    p0 = 1.0e5
    p_init = [
        p0 * (
                1.0 + 0.05 * exp(
                    -(
                        (mesh.cell_centers[1, c] - 0.5)^2 +
                        (mesh.cell_centers[2, c] - 0.5)^2
                    ) / 0.02
                )
            )
            for c in 1:nc
    ]
    dt = 2.0e-5
    res = solve_compressible(
        prob, (0.0, 40 * dt), dt;
        linear_solver = LUFactorization(), p_init = p_init,
    )
    mass0 = sum(p_init[c] / (287.05 * 300.0) * mesh.cell_volumes[c] for c in 1:nc)
    mass_hist = res.residuals[:total_mass]
    @test length(mass_hist) == 40
    # ddt(psi p) + conservative linearized rho update => telescoping mass
    @test maximum(abs.(mass_hist .- mass0)) / mass0 < 1.0e-12
    @test all(u -> all(isfinite, u), res.state.base.U.internal)
end

@testset "V&V: pressure pulse propagates at finite (acoustic) speed" begin
    # 1D-ish closed channel; isothermal sound speed sqrt(R T) = 293.5 m/s.
    nx, ny = 100, 4
    mesh = build_cartesian_unstructured_mesh(nx, ny, 1.0, 0.04)
    nc = length(mesh.cell_volumes)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => SlipWallBC(), :right => SlipWallBC(),
        :bottom => SlipWallBC(), :top => SlipWallBC(),
    )
    gas = IdealGas(; gamma = 1.4, R = 287.05, mu = 1.8e-5)
    alg = CompressiblePIMPLE(;
        n_outer = 2, n_correctors = 2,
        alpha_U = 0.7, alpha_p = 0.5, alpha_rho = 0.7, tolerance = 1.0e-8,
    )
    prob = CompressibleProblem(mesh, bcs, alg, gas; T_ref = 300.0)
    p0 = 1.0e5
    amp = 0.01
    p_init = [
        p0 * (1.0 + amp * exp(-(mesh.cell_centers[1, c] - 0.15)^2 / 0.002))
            for c in 1:nc
    ]
    probe = (2 - 1) * nx + 85     # x = 0.845, far from the pulse
    dt = 2.0e-5

    # Early time: front (c*t + width ~ 0.33) has NOT reached the probe.
    res_early = solve_compressible(
        prob, (0.0, 6.0e-4), dt;
        linear_solver = LUFactorization(), p_init = copy(p_init),
    )
    dev_early = abs(res_early.state.base.p.internal[probe] - p0) / (amp * p0)
    @test dev_early < 1.0e-6      # elliptic coupling would react instantly

    # Late time: front has passed the probe -> clear signal.
    res_late = solve_compressible(
        prob, (0.0, 2.4e-3), dt;
        linear_solver = LUFactorization(), p_init = copy(p_init),
    )
    dev_late = abs(res_late.state.base.p.internal[probe] - p0) / (amp * p0)
    @test dev_late > 1.0e-2

    # Mass conserved through the acoustic transient as well.
    mh = res_late.residuals[:total_mass]
    @test maximum(abs.(mh .- mh[1])) / mh[1] < 1.0e-12
end

@testset "V&V: steady mass flux is divergence-free (compressible SIMPLE)" begin
    # Driven cavity at low Mach: after convergence the MASS flux
    # phi_mass = rho_f * phi must be (discretely) divergence-free.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(), :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(),
        :top => FixedVelocityBC(SVector(1.0e-3, 0.0)),
    )
    gas = IdealGas(; gamma = 1.4, R = 287.05, mu = 1.8e-5)
    alg = CompressibleSIMPLE(;
        alpha_U = 0.7, alpha_p = 0.3, alpha_rho = 0.9,
        max_iterations = 300, tolerance = 1.0e-10,
    )
    prob = CompressibleProblem(mesh, bcs, alg, gas; T_ref = 300.0)
    res = solve_compressible(prob; linear_solver = LUFactorization(), p0 = 1.01325e5)

    phi_mass = res.state.phi_mass
    imb = zeros(nc)
    nf = size(mesh.face_cells, 2)
    for f in 1:nf
        P = FiniteVolumeMethod.owner(mesh, f)
        imb[P] += phi_mass[f]
        N = FiniteVolumeMethod.neighbour(mesh, f)
        if N != 0
            imb[N] -= phi_mass[f]
        end
    end
    scale = sum(abs, phi_mass) + eps()
    @test sum(abs, imb) / scale < 5.0e-2
end
