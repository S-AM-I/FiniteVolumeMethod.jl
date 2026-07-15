using FiniteVolumeMethod
using OrdinaryDiffEq
using StaticArrays
using Test

# ============================================================
# Test 1: Brio-Wu Shock Tube via ODEProblem + SSPRK33
# ============================================================
@testset "Brio-Wu shock tube (ODEProblem)" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{2}(eos)

    Bx_val = 0.75
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx_val, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, Bx_val, -1.0, 0.0)

    function briowu_ic(x, y)
        return x < 0.5 ? wL : wR
    end

    nx, ny = 200, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    prob = HyperbolicProblem2D(
        law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        briowu_ic; final_time = 0.1, cfl = 0.4
    )

    # Build augmented-state ODEProblem
    ode_prob = ODEProblem(prob)
    @test ode_prob isa ODEProblem
    @test ode_prob.p isa MHDCTCache2D

    # Augmented state: cell vars + face Bx + face By
    expected_len = nx * ny * 8 + (nx + 1) * ny + nx * (ny + 1)
    @test length(ode_prob.u0) == expected_len

    # Solve with SSPRK33 and MHD stage limiter
    cache = ode_prob.p
    dt0 = compute_initial_dt(cache, ode_prob.u0)
    @test dt0 > 0

    limiter = mhd_stage_limiter(cache)
    sol = solve(
        ode_prob, SSPRK33(; stage_limiter! = limiter);
        adaptive = false, dt = dt0
    )
    @test sol.retcode == ReturnCode.Success

    # Extract solution via accessor
    accessor = MHDSolutionAccessor(prob)
    U_final = get_conserved(accessor, sol, length(sol.t))
    @test size(U_final) == (nx, ny)

    # Convert to primitive and check density positivity
    W_final = [conserved_to_primitive(law, U_final[ix, iy]) for ix in 1:nx, iy in 1:ny]
    rho = [w[1] for w in W_final]
    P = [w[5] for w in W_final]

    @test all(rho .> 0)
    @test all(P .> 0)
    @test all(isfinite.(rho))

    # Density should develop structure in x (shock tube)
    rho_x = [rho[ix, 1] for ix in 1:nx]
    @test maximum(rho_x) > minimum(rho_x) + 0.01

    # Check div(B) via CT state
    ct = get_ct_state(accessor, sol, length(sol.t))
    @test ct isa CTData2D
    divB_max = max_divB(ct, mesh.dx, mesh.dy, nx, ny)
    @test divB_max < 1.0e-12
end

# ============================================================
# Test 2: Uniform Field Preservation
# ============================================================
@testset "Uniform field preservation (ODEProblem)" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)

    # Constant state: rho=1, v=0, P=1, B=(1,0,0)
    w_const = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    u_ref = primitive_to_conserved(law, w_const)

    uniform_ic(x, y) = w_const

    nx, ny = 20, 20
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    prob = HyperbolicProblem2D(
        law, mesh, HLLDSolver(), NoReconstruction(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        uniform_ic; final_time = 0.05, cfl = 0.3
    )

    ode_prob = ODEProblem(prob)
    cache = ode_prob.p
    dt0 = compute_initial_dt(cache, ode_prob.u0)

    limiter = mhd_stage_limiter(cache)
    sol = solve(
        ode_prob, SSPRK33(; stage_limiter! = limiter);
        adaptive = false, dt = dt0
    )
    @test sol.retcode == ReturnCode.Success

    # State should be preserved to near machine precision
    accessor = MHDSolutionAccessor(prob)
    U_final = get_conserved(accessor, sol, length(sol.t))
    for iy in 1:ny, ix in 1:nx
        @test U_final[ix, iy] ≈ u_ref atol = 1.0e-12
    end

    # div(B) should be near machine precision
    ct = get_ct_state(accessor, sol, length(sol.t))
    @test max_divB(ct, mesh.dx, mesh.dy, nx, ny) < 1.0e-13
end

# ============================================================
# Test 3: MHD Solution Accessors
# ============================================================
@testset "MHD solution accessors (ODEProblem)" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)

    function accessor_ic(x, y)
        rho = 1.0 + 0.1 * sin(2pi * x) * cos(2pi * y)
        return SVector(rho, 0.1, -0.1, 0.0, 1.0, 0.5, 0.3, 0.0)
    end

    nx, ny = 16, 12
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    prob = HyperbolicProblem2D(
        law, mesh, HLLDSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        accessor_ic; final_time = 0.01, cfl = 0.3
    )

    ode_prob = ODEProblem(prob)
    cache = ode_prob.p
    dt0 = compute_initial_dt(cache, ode_prob.u0)

    limiter = mhd_stage_limiter(cache)
    sol = solve(
        ode_prob, SSPRK33(; stage_limiter! = limiter);
        adaptive = false, dt = dt0
    )
    @test sol.retcode == ReturnCode.Success
    @test length(sol.t) > 1

    accessor = MHDSolutionAccessor(prob)

    @testset "get_conserved returns nx x ny matrix of SVector{8}" begin
        U = get_conserved(accessor, sol, length(sol.t))
        @test size(U) == (nx, ny)
        @test U[1, 1] isa SVector{8, Float64}

        # All entries should be finite
        for iy in 1:ny, ix in 1:nx
            @test all(isfinite, U[ix, iy])
        end
    end

    @testset "get_ct_state returns CTData2D with correct sizes" begin
        ct = get_ct_state(accessor, sol, length(sol.t))
        @test ct isa CTData2D
        @test size(ct.Bx_face) == (nx + 1, ny)
        @test size(ct.By_face) == (nx, ny + 1)
        @test size(ct.emf_z) == (nx + 1, ny + 1)

        # Face-B values should be finite
        @test all(isfinite, ct.Bx_face)
        @test all(isfinite, ct.By_face)
    end

    @testset "get_conserved at initial time matches IC" begin
        U0 = get_conserved(accessor, sol, 1)
        @test size(U0) == (nx, ny)

        # Check a few cells against the IC
        x0, y0 = cell_center(mesh, cell_idx(mesh, 1, 1))
        w0 = accessor_ic(x0, y0)
        u0_expected = primitive_to_conserved(law, w0)
        @test U0[1, 1] ≈ u0_expected atol = 1.0e-10
    end

    @testset "accessor field values" begin
        @test accessor.nx == nx
        @test accessor.ny == ny
        @test accessor.n_cell_vars == nx * ny * 8
        @test accessor.n_bx_face == (nx + 1) * ny
        @test accessor.n_by_face == nx * (ny + 1)
    end
end

# ============================================================
# Test 4: Compare ODEProblem with Legacy solve_hyperbolic
# ============================================================
@testset "ODEProblem vs legacy solve_hyperbolic" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{2}(eos)

    Bx_val = 0.75
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx_val, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, Bx_val, -1.0, 0.0)

    bw_ic(x, y) = x < 0.5 ? wL : wR

    nx, ny = 40, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    prob = HyperbolicProblem2D(
        law, mesh, HLLDSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        bw_ic; final_time = 0.05, cfl = 0.4
    )

    # Legacy solve
    coords_leg, U_leg, t_leg, ct_leg = solve_hyperbolic(prob)
    @test t_leg ≈ 0.05 atol = 1.0e-10

    # ODEProblem solve
    ode_prob = ODEProblem(prob)
    cache = ode_prob.p
    dt0 = compute_initial_dt(cache, ode_prob.u0)

    limiter = mhd_stage_limiter(cache)
    sol = solve(
        ode_prob, SSPRK33(; stage_limiter! = limiter);
        adaptive = false, dt = dt0
    )
    @test sol.retcode == ReturnCode.Success

    # Extract ODEProblem solution
    accessor = MHDSolutionAccessor(prob)
    U_ode = get_conserved(accessor, sol, length(sol.t))
    ct_ode = get_ct_state(accessor, sol, length(sol.t))

    # Both should reach the same final time
    @test abs(sol.t[end] - t_leg) < 1.0e-10

    # Solutions should be close (same RK3 scheme, same CFL, same RHS)
    max_diff = 0.0
    for iy in 1:ny, ix in 1:nx
        diff = maximum(abs.(U_ode[ix, iy] - U_leg[ix, iy]))
        max_diff = max(max_diff, diff)
    end
    @test max_diff < 1.0e-6

    # Both should maintain div(B) = 0
    @test max_divB(ct_leg, mesh.dx, mesh.dy, nx, ny) < 1.0e-12
    @test max_divB(ct_ode, mesh.dx, mesh.dy, nx, ny) < 1.0e-12

    # Both solutions should be physically valid
    for iy in 1:ny, ix in 1:nx
        w_ode = conserved_to_primitive(law, U_ode[ix, iy])
        w_leg = conserved_to_primitive(law, U_leg[ix, iy])
        @test w_ode[1] > 0  # density positive
        @test w_ode[5] > 0  # pressure positive
        @test w_leg[1] > 0
        @test w_leg[5] > 0
    end
end

# ============================================================
# SciMLStructures repack round-trip for MHD (Dim preserved)
# ============================================================
using SciMLBase: SciMLBase
import SciMLBase.SciMLStructures as SciMLStructuresMod

@testset "SciMLStructures Tunable repack for MHD caches" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 16)
    ic = x -> SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
    prob = HyperbolicProblem(
        law, mesh, HLLDSolver(), CellCenteredMUSCL(),
        TransmissiveBC(), TransmissiveBC(), ic;
        final_time = 0.1, cfl = 0.5
    )
    cache = FiniteVolumeMethod.build_cache(prob)

    vals, repack, alias = SciMLStructuresMod.canonicalize(SciMLStructuresMod.Tunable(), cache)
    @test vals == [5.0 / 3.0, 0.5]

    # Round-trip with unchanged values preserves everything
    cache_rt = repack(copy(vals))
    @test cache_rt.prob.law isa IdealMHDEquations{1}
    @test cache_rt.prob.law.eos.gamma == 5.0 / 3.0
    @test cache_rt.prob.cfl == 0.5

    # Repack with new values: Dim parameter must be preserved
    # (this used to throw a MethodError — IdealMHDEquations has no
    # Dim-less constructor).
    cache2 = repack([1.4, 0.9])
    @test cache2.prob.law isa IdealMHDEquations{1}
    @test cache2.prob.law.eos.gamma == 1.4
    @test cache2.prob.cfl == 0.9

    # SciMLStructures.replace also works
    cache3 = SciMLStructuresMod.replace(SciMLStructuresMod.Tunable(), cache, [2.0, 0.3])
    @test cache3.prob.law isa IdealMHDEquations{1}
    @test cache3.prob.law.eos.gamma == 2.0
end

# ============================================================
# Curved-spacetime GRMHD through the SciML ODEProblem path
# ============================================================
@testset "GRMHD curved atmosphere held (ODEProblem path)" begin
    # Same static Kerr-Schild Schwarzschild atmosphere gate as
    # test/grmhd_2d.jl, exercised through the SciML RHS (which shares
    # _grmhd_stage_rhs! with the legacy solver, so the paths agree).
    M_BH = 1.0
    GAM = 5.0 / 3.0
    K_POLY = 0.1
    R0 = 6.0
    H0 = 1.0 + GAM / (GAM - 1.0) * K_POLY

    atmosphere_w(x, y) = begin
        r = sqrt(x^2 + y^2)
        h = H0 * sqrt((1 - 2 * M_BH / R0) / (1 - 2 * M_BH / r))
        rho = ((h - 1) * (GAM - 1) / (GAM * K_POLY))^(1 / (GAM - 1))
        P = K_POLY * rho^GAM
        Hks = M_BH / r
        vmag = 2 * Hks / sqrt(1 + 2 * Hks)
        SVector(rho, vmag * x / r, vmag * y / r, 0.0, P, 0.0, 0.0, 0.0)
    end

    eos = IdealGasEOS(gamma = GAM)
    metric = SchwarzschildMetric(M_BH; r_min = 1.5)
    law = GRMHDEquations{2}(eos, metric)
    N = 24
    mesh = StructuredMesh2D(4.0, 8.0, -2.0, 2.0, N, N)
    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        atmosphere_w; final_time = 0.5, cfl = 0.3
    )
    ode_prob = ODEProblem(prob)
    sol = solve(ode_prob, SSPRK33(); adaptive = false, dt = 1.0e-3)
    @test sol.retcode == SciMLBase.ReturnCode.Success

    cache = ode_prob.p
    u = sol.u[end]
    u_sv = reinterpret(SVector{8, Float64}, @view u[1:(cache.n_cell_vars)])
    U_mat = [u_sv[(iy - 1) * N + ix] for ix in 1:N, iy in 1:N]
    W = FiniteVolumeMethod.grmhd_recover_primitive_field(law, U_mat, mesh)
    m = N ÷ 4
    drift = maximum(
        abs(W[ix, iy][1] - atmosphere_w(cell_center(mesh, cell_idx(mesh, ix, iy))...)[1]) /
            atmosphere_w(cell_center(mesh, cell_idx(mesh, ix, iy))...)[1]
            for iy in (m + 1):(N - m), ix in (m + 1):(N - m)
    )
    @test drift < 2.0e-3
end
