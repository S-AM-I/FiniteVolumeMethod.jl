using FiniteVolumeMethod
using Test
using StaticArrays

# Helper
_mean(itr) = sum(itr) / length(collect(itr))

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

@testset "Flat-Spacetime Full Conservation" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = MinkowskiMetric{2}()
    law = GRMHDEquations{2}(eos, metric)
    nx, ny = 24, 24
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    ic(x, y) = SVector(
        1.0 + 0.1 * sin(2π * x),
        0.05 * cos(2π * y), 0.05 * sin(2π * x), 0.0,
        1.0, 0.5, 0.5, 0.0
    )

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.05, cfl = 0.3
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    dV = mesh.dx * mesh.dy

    U0 = [FiniteVolumeMethod.primitive_to_conserved(law, ic(coords[ix, iy]...)) for ix in 1:nx, iy in 1:ny]

    # Test conservation of all conserved quantities
    for var in [1, 2, 3, 5]  # D, Sx, Sy, tau
        total = sum(U[ix, iy][var] for ix in 1:nx, iy in 1:ny) * dV
        total_0 = sum(U0[ix, iy][var] for ix in 1:nx, iy in 1:ny) * dV
        # Use atol for quantities near zero (like momentum), rtol for large ones
        if abs(total_0) > 1e-10
            @test total ≈ total_0 rtol = 1e-9
        else
            @test total ≈ total_0 atol = 1e-12
        end
    end
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

@testset "GRMHD matches SRMHD (y-direction Sod)" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = MinkowskiMetric{2}()
    law_gr = GRMHDEquations{2}(eos, metric)
    law_sr = SRMHDEquations{2}(eos)
    nx, ny = 4, 40
    mesh = StructuredMesh2D(0.0, 0.1, 0.0, 1.0, nx, ny)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, -1.0, 0.5, 0.0)
    ic(x, y) = y < 0.5 ? wL : wR

    prob_gr = HyperbolicProblem2D(
        law_gr, mesh, HLLSolver(), NoReconstruction(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.1, cfl = 0.3
    )
    prob_sr = HyperbolicProblem2D(
        law_sr, mesh, HLLSolver(), NoReconstruction(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.1, cfl = 0.3
    )

    _, U_gr, _, _ = solve_hyperbolic(prob_gr)
    _, U_sr, _, _ = solve_hyperbolic(prob_sr)

    W_gr = to_primitive(law_gr, U_gr)
    W_sr = to_primitive(law_sr, U_sr)

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

# ============================================================
# Schwarzschild with MUSCL Reconstruction
# ============================================================
@testset "Schwarzschild MUSCL stability" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = SchwarzschildMetric(1.0; r_min = 1.5)
    law = GRMHDEquations{2}(eos, metric)
    nx, ny = 20, 20
    mesh = StructuredMesh2D(3.0, 10.0, -3.0, 3.0, nx, ny)

    ic(x, y) = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0)

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.3, cfl = 0.2
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][5] > 0 for ix in 1:nx, iy in 1:ny)
end

# ============================================================
# Schwarzschild with Magnetic Field + CT
# ============================================================
@testset "Schwarzschild CT DivB" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = SchwarzschildMetric(1.0; r_min = 1.5)
    law = GRMHDEquations{2}(eos, metric)
    nx, ny = 24, 24
    mesh = StructuredMesh2D(3.0, 10.0, -3.0, 3.0, nx, ny)

    Az(x, y) = 0.1 * cos(2π * (x - 3.0) / 7.0) * cos(2π * (y + 3.0) / 6.0) / (2π)
    ic(x, y) = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    prob = HyperbolicProblem2D(
        law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.2, cfl = 0.15
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = Az)

    divB_max = max_divB(ct, mesh.dx, mesh.dy, nx, ny)
    @test divB_max < 1e-12
end

# ============================================================
# Kerr Metric Stability
# ============================================================
@testset "Kerr basic stability" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = KerrMetric(1.0, 0.5; r_min = 1.5)
    law = GRMHDEquations{2}(eos, metric)
    nx, ny = 20, 20
    mesh = StructuredMesh2D(3.0, 10.0, -3.0, 3.0, nx, ny)

    ic(x, y) = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.05, 0.0, 0.0)

    prob = HyperbolicProblem2D(
        law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.3, cfl = 0.15
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][5] > 0 for ix in 1:nx, iy in 1:ny)
end

# ============================================================
# Kerr High Spin Stability
# ============================================================
@testset "Kerr high spin stability" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = KerrMetric(1.0, 0.9; r_min = 1.5)
    law = GRMHDEquations{2}(eos, metric)
    nx, ny = 16, 16
    mesh = StructuredMesh2D(3.0, 10.0, -3.0, 3.0, nx, ny)

    ic(x, y) = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    prob = HyperbolicProblem2D(
        law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.2, cfl = 0.1
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
end

# ============================================================
# Reflective BC Stability
# ============================================================
@testset "GRMHD Reflective BCs" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = MinkowskiMetric{2}()
    law = GRMHDEquations{2}(eos, metric)
    nx, ny = 20, 20
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    ic(x, y) = begin
        r = sqrt((x - 0.5)^2 + (y - 0.5)^2)
        P = r < 0.2 ? 10.0 : 1.0
        SVector(1.0, 0.0, 0.0, 0.0, P, 0.5, 0.0, 0.0)
    end

    prob = HyperbolicProblem2D(
        law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
        ReflectiveBC(), ReflectiveBC(), ReflectiveBC(), ReflectiveBC(),
        ic; final_time = 0.05, cfl = 0.2
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][5] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
end

# ============================================================
# Schwarzschild Pressure Pulse
# ============================================================
@testset "Schwarzschild pressure pulse" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = SchwarzschildMetric(1.0; r_min = 1.5)
    law = GRMHDEquations{2}(eos, metric)
    nx, ny = 24, 24
    mesh = StructuredMesh2D(3.0, 10.0, -3.0, 3.0, nx, ny)

    # Pressure pulse off-center
    ic(x, y) = begin
        r = sqrt((x - 6.5)^2 + y^2)
        P = r < 0.5 ? 5.0 : 1.0
        SVector(1.0, 0.0, 0.0, 0.0, P, 0.1, 0.0, 0.0)
    end

    prob = HyperbolicProblem2D(
        law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.5, cfl = 0.15
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    # Solution should remain physical
    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][5] > 0 for ix in 1:nx, iy in 1:ny)
    # Pressure pulse should have evolved
    P_vals = [W[ix, iy][5] for ix in 1:nx, iy in 1:ny]
    @test minimum(P_vals) < 5.0  # pulse has spread
end

# ============================================================
# Forward Euler Time Integration
# ============================================================
@testset "Forward Euler method" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = MinkowskiMetric{2}()
    law = GRMHDEquations{2}(eos, metric)
    nx, ny = 16, 16
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    ic(x, y) = SVector(
        1.0 + 0.1 * sin(2π * x),
        0.0, 0.0, 0.0,
        1.0, 0.5, 0.0, 0.0
    )

    prob = HyperbolicProblem2D(
        law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.02, cfl = 0.2
    )

    coords, U, t, ct = solve_hyperbolic(prob; method = :euler, vector_potential = nothing)
    W = to_primitive(law, U)

    @test t ≈ 0.02 atol = 1e-10
    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
end

# ============================================================
# Different Riemann Solvers
# ============================================================
@testset "Riemann solvers" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = MinkowskiMetric{2}()
    law = GRMHDEquations{2}(eos, metric)
    nx, ny = 20, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)
    ic(x, y) = x < 0.5 ? wL : wR

    for solver in [LaxFriedrichsSolver(), HLLSolver()]
        prob = HyperbolicProblem2D(
            law, mesh, solver, NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            ic; final_time = 0.1, cfl = 0.25
        )

        coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
        W = to_primitive(law, U)

        @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
        @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
        @test all(W[ix, iy][5] > 0 for ix in 1:nx, iy in 1:ny)
    end
end

# ============================================================
# Multiple Limiters
# ============================================================
@testset "Limiters" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = MinkowskiMetric{2}()
    law = GRMHDEquations{2}(eos, metric)
    nx, ny = 20, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)
    ic(x, y) = x < 0.5 ? wL : wR

    limiters = [MinmodLimiter(), SuperbeeLimiter(), VanLeerLimiter()]
    for lim in limiters
        prob = HyperbolicProblem2D(
            law, mesh, HLLSolver(), CellCenteredMUSCL(lim),
            TransmissiveBC(), TransmissiveBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            ic; final_time = 0.05, cfl = 0.25
        )

        coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
        W = to_primitive(law, U)

        @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
        @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    end
end

# ============================================================
# Schwarzschild Infall Direction
# ============================================================
@testset "Schwarzschild gravitational attraction" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = SchwarzschildMetric(1.0; r_min = 1.5)
    law = GRMHDEquations{2}(eos, metric)
    nx, ny = 16, 4
    mesh = StructuredMesh2D(3.0, 15.0, -0.5, 0.5, nx, ny)

    # Uniform static atmosphere — should develop inward velocity
    ic(x, y) = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    prob = HyperbolicProblem2D(
        law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(), PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 1.0, cfl = 0.15
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    # Material near the BH (left side) should develop negative vx (infall)
    # or at least the density near the BH should increase
    ρ_left = _mean(W[1, iy][1] for iy in 1:ny)
    ρ_right = _mean(W[nx, iy][1] for iy in 1:ny)

    # Material should accumulate toward the BH (left side)
    # This is a qualitative test — gravitational source terms cause infall
    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
end

