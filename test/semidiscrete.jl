using FiniteVolumeMethod
using FiniteVolumeMethod: HyperbolicCache1D, HyperbolicCache2D, HyperbolicCache3D, HyperbolicSolutionAccessor, MHDCTCache2D, MHDSolutionAccessor, UnstructuredCache, UnstructuredHyperbolicMesh, UnstructuredHyperbolicProblem, build_cache, cell_volume, fold_from_padded!, initial_state_flat, unfold_to_padded!
using OrdinaryDiffEq
using OrdinaryDiffEqSSPRK: SSPRK33
using StaticArrays
using Test

# ============================================================
# Test 1: ODEProblem construction for 1D Euler (Sod shock tube)
# ============================================================
@testset "1D ODEProblem construction + solve" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 100)

    function sod_ic(x)
        if x < 0.5
            return SVector(1.0, 0.0, 1.0)   # ρ, v, P
        else
            return SVector(0.125, 0.0, 0.1)
        end
    end

    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(),
        TransmissiveBC(), TransmissiveBC(), sod_ic;
        final_time = 0.2, cfl = 0.5
    )

    # Build ODEProblem
    ode_prob = ODEProblem(prob)
    @test ode_prob isa ODEProblem
    @test length(ode_prob.u0) == 100 * 3  # nc * N (1D Euler has 3 vars)
    @test ode_prob.tspan == (0.0, 0.2)
    @test ode_prob.p isa HyperbolicCache1D

    # Solve with SSP-RK3
    cache = ode_prob.p
    dt0 = compute_initial_dt(cache, ode_prob.u0)
    sol = solve(ode_prob, SSPRK33(); adaptive = false, dt = dt0)
    @test sol.retcode == ReturnCode.Success
    @test length(sol.t) > 1

    # Verify solution is reasonable (density should be positive)
    U_final = reinterpret(SVector{3, Float64}, copy(sol.u[end]))
    for u in U_final
        @test u[1] > 0  # density > 0
    end

    # The solve must reach the requested final time exactly
    @test abs(sol.t[end] - 0.2) < 1.0e-10
end

# ============================================================
# Test 2: ODEProblem construction for 2D Euler
# ============================================================
@testset "2D ODEProblem construction + solve" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 20, 20)

    function uniform_ic_2d(x, y)
        return SVector(1.0, 0.0, 0.0, 1.0)  # ρ, vx, vy, P
    end

    prob = HyperbolicProblem2D(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        uniform_ic_2d; final_time = 0.1, cfl = 0.4
    )

    ode_prob = ODEProblem(prob)
    @test ode_prob isa ODEProblem
    @test length(ode_prob.u0) == 20 * 20 * 4
    @test ode_prob.p isa HyperbolicCache2D

    cache = ode_prob.p
    dt0 = compute_initial_dt(cache, ode_prob.u0)
    sol = solve(ode_prob, SSPRK33(); adaptive = false, dt = dt0)
    @test sol.retcode == ReturnCode.Success

    # Uniform IC should remain uniform
    U_final = reinterpret(SVector{4, Float64}, copy(sol.u[end]))
    for u in U_final
        @test abs(u[1] - 1.0) < 1.0e-12
    end
end

# ============================================================
# Test 3: Solution accessors
# ============================================================
@testset "Solution accessors" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 50)

    function uniform_ic_1d(x)
        return SVector(1.0, 0.1, 1.0)  # ρ, v, P
    end

    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), uniform_ic_1d;
        final_time = 0.05, cfl = 0.5
    )

    ode_prob = ODEProblem(prob)
    cache = ode_prob.p
    dt0 = compute_initial_dt(cache, ode_prob.u0)
    sol = solve(ode_prob, SSPRK33(); adaptive = false, dt = dt0)

    accessor = HyperbolicSolutionAccessor(prob)

    # get_conserved
    U = get_conserved(accessor, sol, length(sol.t))
    @test length(U) == 50
    @test U[1] isa SVector{3}

    # get_primitive
    W = get_primitive(accessor, sol, length(sol.t))
    @test length(W) == 50
    @test W[1] isa SVector{3}

    # get_coordinates
    coords = get_coordinates(accessor)
    @test length(coords) == 50
end

# ============================================================
# Test 4: CFL callback correctness
# ============================================================
@testset "CFL callback" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 50)

    function sod_ic_cfl(x)
        if x < 0.5
            return SVector(1.0, 0.0, 1.0)
        else
            return SVector(0.125, 0.0, 0.1)
        end
    end

    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), sod_ic_cfl;
        final_time = 0.1, cfl = 0.3
    )

    ode_prob = ODEProblem(prob)
    cache = ode_prob.p
    dt0 = compute_initial_dt(cache, ode_prob.u0)
    @test dt0 > 0

    # The initial dt should respect CFL
    dx = cell_volume(mesh, 1)
    # Max wave speed from Sod is bounded
    @test dt0 < dx  # sanity check
end

# ============================================================
# Test 5: 3D ODEProblem
# ============================================================
@testset "3D ODEProblem" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 5, 5, 5)

    function uniform_ic_3d(x, y, z)
        return SVector(1.0, 0.0, 0.0, 0.0, 1.0)  # 3D: ρ, vx, vy, vz, P (5 variables)
    end

    prob = HyperbolicProblem3D(
        law, mesh, HLLCSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        uniform_ic_3d; final_time = 0.05, cfl = 0.3
    )

    ode_prob = ODEProblem(prob)
    @test ode_prob isa ODEProblem
    @test length(ode_prob.u0) == 5 * 5 * 5 * 5  # nx*ny*nz*N
    @test ode_prob.p isa HyperbolicCache3D
end

# ============================================================
# Test 6: MHD/CT augmented state ODEProblem
# ============================================================
@testset "MHD/CT ODEProblem" begin
    eos = IdealGasEOS(5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)
    nx, ny = 10, 10
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    function mhd_ic(x, y)
        return SVector(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    end

    prob = HyperbolicProblem2D(
        law, mesh, HLLDSolver(), NoReconstruction(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        mhd_ic; final_time = 0.05, cfl = 0.3
    )

    ode_prob = ODEProblem(prob)
    @test ode_prob isa ODEProblem
    # State = cell vars + face Bx + face By
    expected_len = nx * ny * 8 + (nx + 1) * ny + nx * (ny + 1)
    @test length(ode_prob.u0) == expected_len
    @test ode_prob.p isa MHDCTCache2D

    # Check solution accessor
    accessor = MHDSolutionAccessor(prob)
    @test accessor.n_cell_vars == nx * ny * 8
    @test accessor.n_bx_face == (nx + 1) * ny
    @test accessor.n_by_face == nx * (ny + 1)
end

# ============================================================
# Test 7: Unstructured ODEProblem
# ============================================================
@testset "Unstructured ODEProblem" begin
    using DelaunayTriangulation
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    # Create a simple triangular mesh via DelaunayTriangulation
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    boundary_nodes = [[1, 2], [2, 3], [3, 4], [4, 1]]
    tri = triangulate(points; boundary_nodes)

    umesh = UnstructuredHyperbolicMesh(tri)

    function uniform_ic_unstruct(x, y)
        return SVector(1.0, 0.0, 0.0, 1.0)
    end

    prob = UnstructuredHyperbolicProblem(
        law, umesh, LaxFriedrichsSolver(), NoReconstruction(),
        TransmissiveBC(), uniform_ic_unstruct; final_time = 0.01, cfl = 0.2
    )

    ode_prob = ODEProblem(prob)
    @test ode_prob isa ODEProblem
    @test length(ode_prob.u0) == umesh.ntri * 4
    @test ode_prob.p isa UnstructuredCache
end

# ============================================================
# Test 8: build_cache + state mapping round-trip
# ============================================================
@testset "State mapping round-trip" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 20)

    function ramp_ic(x)
        return SVector(1.0 + x, x, 1.0 + x)  # ρ, v, P
    end

    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), ramp_ic;
        final_time = 0.1, cfl = 0.5
    )

    cache = build_cache(prob)
    u0 = initial_state_flat(prob, cache)

    # Padding follows the reconstruction's ghost count (NoReconstruction
    # pads a single layer); interior cells live at ng+1 : nc+ng.
    ng = cache.ng
    @test ng == FiniteVolumeMethod._nghost_for_reconstruction(prob.reconstruction)

    # Unfold to padded
    unfold_to_padded!(cache, u0)

    # Check interior cells match
    u0_sv = reinterpret(SVector{3, Float64}, u0)
    for i in 1:20
        @test cache.padded_U[i + ng] == u0_sv[i]
    end

    # Set padded_dU to known values and fold back
    for i in (ng + 1):(20 + ng)
        cache.padded_dU[i] = SVector(Float64(i), 0.0, 0.0)
    end

    du = zeros(20 * 3)
    fold_from_padded!(du, cache)
    du_sv = reinterpret(SVector{3, Float64}, du)
    for i in 1:20
        @test du_sv[i][1] == Float64(i + ng)
    end
end

# ============================================================
# Test 9: 2D state mapping round-trip
# ============================================================
@testset "2D State mapping round-trip" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    nx, ny = 8, 6
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    function ramp_ic_2d(x, y)
        return SVector(1.0, x, y, 1.0)  # ρ, vx, vy, P
    end

    prob = HyperbolicProblem2D(
        law, mesh, HLLCSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        ramp_ic_2d; final_time = 0.1, cfl = 0.4
    )

    cache = build_cache(prob)
    u0 = initial_state_flat(prob, cache)

    # Padding follows the reconstruction's ghost count (NoReconstruction
    # pads a single layer).
    ng = cache.ng

    # Unfold and check
    unfold_to_padded!(cache, u0)
    u0_sv = reinterpret(SVector{4, Float64}, u0)
    for iy in 1:ny, ix in 1:nx
        flat_idx = (iy - 1) * nx + ix
        @test cache.padded_U[ix + ng, iy + ng] == u0_sv[flat_idx]
    end
end

# ============================================================
# CFL callback must control dt under adaptive = false
# ============================================================
# Regression for the audit finding that the callback only called
# set_proposed_dt!, which fixed-step integrators ignore — making the
# documented `solve(ode_prob, SSPRK33(); adaptive = false, dt = dt0)`
# usage run at a frozen dt forever.
using SciMLBase: SciMLBase

@testset "CFL callback drives dt with adaptive = false" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    mesh = StructuredMesh1D(0.0, 1.0, 100)
    # Blast-wave-like IC: the shock accelerates into the low-pressure
    # ambient, so the CFL-limited dt must change during the run.
    ic = x -> abs(x - 0.5) < 0.05 ? SVector(1.0, 0.0, 100.0) : SVector(1.0, 0.0, 1.0e-2)
    prob = HyperbolicProblem(
        law, mesh, HLLCSolver(), CellCenteredMUSCL(),
        TransmissiveBC(), TransmissiveBC(), ic;
        final_time = 0.02, cfl = 0.4
    )
    ode = ODEProblem(prob)
    dt0 = compute_initial_dt(ode.p, ode.u0)

    dts = Float64[]
    recorder = DiscreteCallback(
        (u, t, integrator) -> true,
        integrator -> begin
            push!(dts, integrator.dt)
            SciMLBase.derivative_discontinuity!(integrator, false)
        end;
        save_positions = (false, false)
    )
    sol = solve(ode, SSPRK33(); adaptive = false, dt = dt0, callback = recorder)

    @test SciMLBase.successful_retcode(sol)
    # The integrator's dt must actually track the CFL callback: at least
    # one step after the first must differ from the initial dt.
    @test length(dts) > 2
    @test any(d -> abs(d - dt0) > 1.0e-12 * dt0, dts[2:end])
    # And the solution must stay finite (no blow-up from a frozen dt)
    @test all(isfinite, sol.u[end])
end
