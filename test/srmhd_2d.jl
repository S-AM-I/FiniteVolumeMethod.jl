using FiniteVolumeMethod
using Test
using StaticArrays

# ============================================================
# 2D SRMHD Solver Tests
# ============================================================

@testset "2D SRMHD Sod x-direction" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 100, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)
    ic(x, y) = x < 0.5 ? wL : wR

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.1, cfl = 0.3
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    # Solution should be uniform in y
    for iy in 2:ny
        for ix in 1:nx
            @test W[ix, iy][1] ≈ W[ix, 1][1] atol = 1e-10
        end
    end

    # Sanity
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][5] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
end

@testset "2D SRMHD Sod y-direction" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 4, 100
    mesh = StructuredMesh2D(0.0, 0.1, 0.0, 1.0, nx, ny)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, -1.0, 0.5, 0.0)
    ic(x, y) = y < 0.5 ? wL : wR

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.1, cfl = 0.3
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    # Solution should be uniform in x
    for ix in 2:nx
        for iy in 1:ny
            @test W[ix, iy][1] ≈ W[1, iy][1] atol = 1e-10
        end
    end

    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][5] > 0 for ix in 1:nx, iy in 1:ny)
end

# ============================================================
# CT: ∇·B = 0 Preservation
# ============================================================
@testset "CT DivB Preservation" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 32, 32
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    # Use vector potential for divergence-free initialization
    Az(x, y) = cos(2π * x) * cos(2π * y) / (2π)

    ic(x, y) = SVector(1.0, 0.1, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0)

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.05, cfl = 0.3
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = Az)

    divB_max = max_divB(ct, mesh.dx, mesh.dy, nx, ny)
    @test divB_max < 1e-13
end

# ============================================================
# 2D Conservation with Periodic BCs
# ============================================================
@testset "2D Conservation" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 32, 32
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    ic(x, y) = SVector(
        1.0 + 0.1 * sin(2π * x) * cos(2π * y),
        0.05 * cos(2π * x), 0.05 * sin(2π * y), 0.0,
        1.0 + 0.05 * cos(2π * x),
        1.0, 0.5, 0.0
    )

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.05, cfl = 0.3
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)

    dV = mesh.dx * mesh.dy

    # Compute conserved totals
    D_total = sum(U[ix, iy][1] for ix in 1:nx, iy in 1:ny) * dV

    # Initial totals
    U0 = [FiniteVolumeMethod.primitive_to_conserved(law, ic(coords[ix, iy]...)) for ix in 1:nx, iy in 1:ny]
    D_total_0 = sum(U0[ix, iy][1] for ix in 1:nx, iy in 1:ny) * dV

    @test D_total ≈ D_total_0 rtol = 1e-10
end

# ============================================================
# Reflective BCs for 2D SRMHD
# ============================================================
@testset "Reflective BCs" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 20, 20
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    # Pressure pulse at center
    ic(x, y) = begin
        r = sqrt((x - 0.5)^2 + (y - 0.5)^2)
        P = r < 0.2 ? 10.0 : 1.0
        SVector(1.0, 0.0, 0.0, 0.0, P, 1.0, 0.0, 0.0)
    end

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), NoReconstruction(),
        ReflectiveBC(), ReflectiveBC(), ReflectiveBC(), ReflectiveBC(),
        ic; final_time = 0.05, cfl = 0.2
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][5] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
end
