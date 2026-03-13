using FiniteVolumeMethod
using OrdinaryDiffEq
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

    # Compare with legacy solve_hyperbolic
    x_leg, U_leg, t_leg = solve_hyperbolic(prob; method = :ssprk3)
    @test abs(sol.t[end] - t_leg) < 1.0e-10

    # Solutions should be similar (not bitwise identical due to different
    # callback mechanics, but physically equivalent)
    U_sciml = reinterpret(SVector{3, Float64}, copy(sol.u[end]))
    max_diff = maximum(maximum(abs.(U_sciml[i] - U_leg[i])) for i in eachindex(U_leg))
    @test max_diff < 1.0e-8
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

    # Unfold to padded
    unfold_to_padded!(cache, u0)

    # Check interior cells match
    u0_sv = reinterpret(SVector{3, Float64}, u0)
    for i in 1:20
        @test cache.padded_U[i + 2] == u0_sv[i]
    end

    # Set padded_dU to known values and fold back
    for i in 3:22
        cache.padded_dU[i] = SVector(Float64(i), 0.0, 0.0)
    end

    du = zeros(20 * 3)
    fold_from_padded!(du, cache)
    du_sv = reinterpret(SVector{3, Float64}, du)
    for i in 1:20
        @test du_sv[i][1] == Float64(i + 2)
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

    # Unfold and check
    unfold_to_padded!(cache, u0)
    u0_sv = reinterpret(SVector{4, Float64}, u0)
    for iy in 1:ny, ix in 1:nx
        flat_idx = (iy - 1) * nx + ix
        @test cache.padded_U[ix + 2, iy + 2] == u0_sv[flat_idx]
    end
end
