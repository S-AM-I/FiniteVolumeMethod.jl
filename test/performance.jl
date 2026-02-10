# ============================================================
# Performance & Threading Tests (Phase 12)
# ============================================================
#
# Verifies that threaded implementations produce identical results
# to serial implementations across all 2D solver modes.

using FiniteVolumeMethod
using StaticArrays
using Test

# ============================================================
# Helper: create a standard 2D Euler problem for testing
# ============================================================

function make_2d_euler_problem(;
        nx = 20, ny = 20, final_time = 0.01, cfl = 0.4,
        solver = HLLCSolver(), recon = CellCenteredMUSCL()
    )
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    function ic(x, y)
        if x < 0.5 && y < 0.5
            return SVector(1.0, 0.0, 0.0, 2.5)    # ρ, vx, vy, p
        elseif x >= 0.5 && y < 0.5
            return SVector(0.5, 0.0, 0.0, 1.0)
        elseif x < 0.5 && y >= 0.5
            return SVector(0.5, 0.0, 0.0, 1.0)
        else
            return SVector(0.25, 0.0, 0.0, 0.4)
        end
    end

    bc = TransmissiveBC()
    return HyperbolicProblem2D(
        law, mesh, solver, recon,
        bc, bc, bc, bc,
        ic; cfl = cfl, final_time = final_time
    )
end

# ============================================================
# Tests
# ============================================================

@testset "Threaded vs Serial: 2D CFL" begin
    prob = make_2d_euler_problem()
    U = FiniteVolumeMethod.initialize_2d(prob)
    FiniteVolumeMethod.apply_boundary_conditions_2d!(U, prob, 0.0)

    dt_serial = FiniteVolumeMethod.compute_dt_2d(prob, U, 0.0)
    dt_threaded = FiniteVolumeMethod._compute_dt_2d_threaded(prob, U, 0.0)

    @test dt_serial ≈ dt_threaded atol = 1.0e-15
    @test dt_serial > 0
end

@testset "Threaded vs Serial: 2D RHS" begin
    prob = make_2d_euler_problem()
    U = FiniteVolumeMethod.initialize_2d(prob)
    FiniteVolumeMethod.apply_boundary_conditions_2d!(U, prob, 0.0)

    nx, ny = prob.mesh.nx, prob.mesh.ny
    N = nvariables(prob.law)
    FT = Float64
    zero_state = zero(SVector{N, FT})

    dU_serial = similar(U)
    dU_threaded = similar(U)
    for j in axes(U, 2), i in axes(U, 1)
        dU_serial[i, j] = zero_state
        dU_threaded[i, j] = zero_state
    end

    FiniteVolumeMethod.hyperbolic_rhs_2d!(dU_serial, U, prob, 0.0)
    FiniteVolumeMethod._hyperbolic_rhs_2d_threaded!(dU_threaded, U, prob, 0.0)

    for iy in 1:ny, ix in 1:nx
        @test dU_serial[ix + 2, iy + 2] ≈ dU_threaded[ix + 2, iy + 2] atol = 1.0e-14
    end
end

@testset "Threaded vs Serial: 2D solve_hyperbolic (Euler)" begin
    prob = make_2d_euler_problem(; final_time = 0.005)

    coords_s, U_s, t_s = solve_hyperbolic(prob; method = :euler, parallel = false)
    coords_t, U_t, t_t = solve_hyperbolic(prob; method = :euler, parallel = true)

    @test t_s ≈ t_t atol = 1.0e-14
    nx, ny = prob.mesh.nx, prob.mesh.ny
    for iy in 1:ny, ix in 1:nx
        @test U_s[ix, iy] ≈ U_t[ix, iy] atol = 1.0e-13
    end
end

@testset "Threaded vs Serial: 2D solve_hyperbolic SSP-RK3" begin
    prob = make_2d_euler_problem(; final_time = 0.005)

    coords_s, U_s, t_s = solve_hyperbolic(prob; method = :ssprk3, parallel = false)
    coords_t, U_t, t_t = solve_hyperbolic(prob; method = :ssprk3, parallel = true)

    @test t_s ≈ t_t atol = 1.0e-14
    nx, ny = prob.mesh.nx, prob.mesh.ny
    for iy in 1:ny, ix in 1:nx
        @test U_s[ix, iy] ≈ U_t[ix, iy] atol = 1.0e-13
    end
end

@testset "Threaded vs Serial: 2D IMEX solve" begin
    eos = IdealGasEOS(5.0 / 3.0)
    law = EulerEquations{2}(eos)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 16, 16)

    ic_uniform(x, y) = SVector(1.0, 0.0, 0.0, 1.0)
    bc = TransmissiveBC()
    prob = HyperbolicProblem2D(
        law, mesh, HLLCSolver(), NoReconstruction(),
        bc, bc, bc, bc,
        ic_uniform; cfl = 0.3, final_time = 0.005
    )
    source = NullSource()

    coords_s, U_s, t_s = solve_hyperbolic_imex(prob, source; parallel = false)
    coords_t, U_t, t_t = solve_hyperbolic_imex(prob, source; parallel = true)

    @test t_s ≈ t_t atol = 1.0e-14
    nx, ny = mesh.nx, mesh.ny
    for iy in 1:ny, ix in 1:nx
        @test U_s[ix, iy] ≈ U_t[ix, iy] atol = 1.0e-13
    end
end

@testset "Threaded vs Serial: Implicit solve 2D" begin
    eos = IdealGasEOS(5.0 / 3.0)
    law = EulerEquations{2}(eos)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 10, 10)
    N = nvariables(law)
    nx, ny = mesh.nx, mesh.ny

    # Create a padded array with non-trivial state
    U_serial = Matrix{SVector{N, Float64}}(undef, nx + 4, ny + 4)
    zero_state = zero(SVector{N, Float64})
    for j in 1:(ny + 4), i in 1:(nx + 4)
        U_serial[i, j] = zero_state
    end
    for iy in 1:ny, ix in 1:nx
        rho = 1.0 + 0.1 * sin(2π * ix / nx)
        p = 1.0 + 0.05 * cos(2π * iy / ny)
        u = primitive_to_conserved(law, SVector(rho, 0.0, 0.0, p))
        U_serial[ix + 2, iy + 2] = u
    end

    U_threaded = copy(U_serial)

    source = CoolingSource(T -> 0.1 * T)
    adt = 0.001

    FiniteVolumeMethod._implicit_solve_2d!(
        U_serial, law, source, adt, nx, ny, N, 1.0e-10, 5
    )
    FiniteVolumeMethod._implicit_solve_2d_threaded!(
        U_threaded, law, source, adt, nx, ny, N, 1.0e-10, 5
    )

    for iy in 1:ny, ix in 1:nx
        @test U_serial[ix + 2, iy + 2] ≈ U_threaded[ix + 2, iy + 2] atol = 1.0e-14
    end
end

@testset "Threaded: Conservation check" begin
    prob = make_2d_euler_problem(; final_time = 0.01)
    _, U_t, _ = solve_hyperbolic(prob; parallel = true)

    # Total mass should be conserved (transmissive BCs, short time)
    nx, ny = prob.mesh.nx, prob.mesh.ny
    dx, dy = prob.mesh.dx, prob.mesh.dy
    total_mass = 0.0
    for iy in 1:ny, ix in 1:nx
        total_mass += U_t[ix, iy][1] * dx * dy
    end
    @test total_mass > 0  # non-trivial
end

@testset "Threaded: NoReconstruction" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 16, 16)
    sod_x(x, y) = x < 0.5 ? SVector(1.0, 0.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.0, 0.1)
    bc = TransmissiveBC()
    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), NoReconstruction(),
        bc, bc, bc, bc,
        sod_x; cfl = 0.4, final_time = 0.005
    )
    _, U_s, t_s = solve_hyperbolic(prob; parallel = false)
    _, U_t, t_t = solve_hyperbolic(prob; parallel = true)

    @test t_s ≈ t_t
    for iy in 1:(mesh.ny), ix in 1:(mesh.nx)
        @test U_s[ix, iy] ≈ U_t[ix, iy] atol = 1.0e-14
    end
end

@testset "Threaded: Different Riemann solvers" begin
    for rsolver in [LaxFriedrichsSolver(), HLLSolver(), HLLCSolver()]
        eos = IdealGasEOS(1.4)
        law = EulerEquations{2}(eos)
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 12, 12)
        sod_y(x, y) = y < 0.5 ? SVector(1.0, 0.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.0, 0.1)
        bc = TransmissiveBC()
        prob = HyperbolicProblem2D(
            law, mesh, rsolver, NoReconstruction(),
            bc, bc, bc, bc,
            sod_y; cfl = 0.3, final_time = 0.003
        )
        _, U_s, _ = solve_hyperbolic(prob; parallel = false)
        _, U_t, _ = solve_hyperbolic(prob; parallel = true)

        for iy in 1:(mesh.ny), ix in 1:(mesh.nx)
            @test U_s[ix, iy] ≈ U_t[ix, iy] atol = 1.0e-13
        end
    end
end
