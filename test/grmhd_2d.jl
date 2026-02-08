using FiniteVolumeMethod
using Test
using StaticArrays

# ============================================================
# 2D GRMHD Solver Tests
# ============================================================

@testset "Flat-Spacetime Conservation" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = MinkowskiMetric{2}()
    law = GRMHDEquations{2}(eos, metric)
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
    D_total = sum(U[ix, iy][1] for ix in 1:nx, iy in 1:ny) * dV

    U0 = [FiniteVolumeMethod.primitive_to_conserved(law, ic(coords[ix, iy]...)) for ix in 1:nx, iy in 1:ny]
    D_total_0 = sum(U0[ix, iy][1] for ix in 1:nx, iy in 1:ny) * dV

    # In Minkowski spacetime, geometric sources vanish → perfect conservation
    @test D_total ≈ D_total_0 rtol = 1e-10
end

@testset "CT DivB Preservation (GRMHD)" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = MinkowskiMetric{2}()
    law = GRMHDEquations{2}(eos, metric)
    nx, ny = 32, 32
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

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

@testset "GRMHD matches SRMHD in Minkowski" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = MinkowskiMetric{2}()
    law_gr = GRMHDEquations{2}(eos, metric)
    law_sr = SRMHDEquations{2}(eos)
    nx, ny = 40, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)
    ic(x, y) = x < 0.5 ? wL : wR

    prob_gr = HyperbolicProblem2D(
        law_gr, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.1, cfl = 0.3
    )
    prob_sr = HyperbolicProblem2D(
        law_sr, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.1, cfl = 0.3
    )

    _, U_gr, t_gr, _ = solve_hyperbolic(prob_gr)
    _, U_sr, t_sr, _ = solve_hyperbolic(prob_sr)

    W_gr = to_primitive(law_gr, U_gr)
    W_sr = to_primitive(law_sr, U_sr)

    # Density profiles should match closely
    for iy in 1:ny, ix in 1:nx
        @test W_gr[ix, iy][1] ≈ W_sr[ix, iy][1] rtol = 1e-10
    end
end

@testset "Schwarzschild basic stability" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = SchwarzschildMetric(1.0; r_min = 1.5)
    law = GRMHDEquations{2}(eos, metric)
    nx, ny = 20, 20
    mesh = StructuredMesh2D(2.0, 10.0, -4.0, 4.0, nx, ny)

    ic(x, y) = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0)

    prob = HyperbolicProblem2D(
        law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.5, cfl = 0.2
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    # Should not crash and should have finite values
    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
end
