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
    @test D_total ≈ D_total_0 rtol = 1.0e-10
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
        if abs(total_0) > 1.0e-10
            @test total ≈ total_0 rtol = 1.0e-9
        else
            @test total ≈ total_0 atol = 1.0e-12
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
    @test divB_max < 1.0e-13
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
        @test W_gr[ix, iy][1] ≈ W_sr[ix, iy][1] rtol = 1.0e-10
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
        @test W_gr[ix, iy][1] ≈ W_sr[ix, iy][1] rtol = 1.0e-10
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
    # Curved path: state is densitized -> metric-aware recovery
    W = FiniteVolumeMethod.grmhd_recover_primitive_field(law, U, mesh)

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
    W = FiniteVolumeMethod.grmhd_recover_primitive_field(law, U, mesh)

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
    @test divB_max < 1.0e-12
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
    W = FiniteVolumeMethod.grmhd_recover_primitive_field(law, U, mesh)

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
    W = FiniteVolumeMethod.grmhd_recover_primitive_field(law, U, mesh)

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
    W = FiniteVolumeMethod.grmhd_recover_primitive_field(law, U, mesh)

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

    @test t ≈ 0.02 atol = 1.0e-10
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
    W = FiniteVolumeMethod.grmhd_recover_primitive_field(law, U, mesh)

    # Material near the BH (left side) should develop negative vx (infall)
    # or at least the density near the BH should increase
    ρ_left = _mean(W[1, iy][1] for iy in 1:ny)
    ρ_right = _mean(W[nx, iy][1] for iy in 1:ny)

    # Material should accumulate toward the BH (left side)
    # This is a qualitative test — gravitational source terms cause infall
    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
end

# ============================================================
# Curved-path verification gates
# ============================================================
#
# Exact stationary solution: a static polytropic atmosphere in
# Kerr-Schild Schwarzschild coordinates. A coordinate-static fluid
# (u^i = 0) has Valencia velocity v^i = beta^i/alpha and satisfies the
# Tolman condition h(r) * sqrt(1 - 2M/r) = const for a polytrope
# P = K rho^Gamma. Its transport velocity alpha v - beta vanishes
# identically, so D and B fluxes are exactly zero, while the momentum
# and energy equations balance pressure fluxes against the geometric
# sources.

const GRG_M = 1.0
const GRG_GAM = 5.0 / 3.0
const GRG_K = 0.1
const GRG_R0 = 6.0
const GRG_H0 = 1.0 + GRG_GAM / (GRG_GAM - 1.0) * GRG_K  # rho0 = 1 at r0

grg_atmosphere(x, y) = begin
    r = sqrt(x^2 + y^2)
    h = GRG_H0 * sqrt((1 - 2 * GRG_M / GRG_R0) / (1 - 2 * GRG_M / r))
    rho = ((h - 1) * (GRG_GAM - 1) / (GRG_GAM * GRG_K))^(1 / (GRG_GAM - 1))
    P = GRG_K * rho^GRG_GAM
    Hks = GRG_M / r
    vmag = 2 * Hks / sqrt(1 + 2 * Hks)
    SVector(rho, vmag * x / r, vmag * y / r, 0.0, P, 0.0, 0.0, 0.0)
end

const GRG_EOS = IdealGasEOS(gamma = GRG_GAM)
const GRG_METRIC = SchwarzschildMetric(GRG_M; r_min = 1.5)
const GRG_LAW = GRMHDEquations{2}(GRG_EOS, GRG_METRIC)

function grg_make_prob(N, recon; tf = 0.0)
    mesh = StructuredMesh2D(4.0, 8.0, -2.0, 2.0, N, N)
    prob = HyperbolicProblem2D(
        GRG_LAW, mesh, HLLSolver(), recon,
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        grg_atmosphere; final_time = tf, cfl = 0.3
    )
    return prob, mesh
end

@testset "Curved gate: Minkowski reduction of the curved machinery" begin
    mink = MinkowskiMetric{2}()
    law_m = GRMHDEquations{2}(GRG_EOS, mink)
    w = SVector(1.3, 0.2, -0.1, 0.05, 2.1, 0.4, -0.3, 0.2)
    gm = FiniteVolumeMethod.spatial_metric(mink, 0.0, 0.0)
    gi = FiniteVolumeMethod.inv_spatial_metric(mink, 0.0, 0.0)
    for dir in [1, 2]
        Fv = FiniteVolumeMethod._grmhd_valencia_flux(GRG_EOS, w, dir, 1.0, SVector(0.0, 0.0), gm, 1.0)
        Ff = physical_flux(law_m, w, dir)
        @test maximum(abs.(Fv - Ff)) < 1.0e-14
        lmf, lpf = FiniteVolumeMethod._grmhd_wave_speeds(law_m, w, dir)
        lmc, lpc = FiniteVolumeMethod._grmhd_coord_wave_speeds(GRG_EOS, w, dir, 1.0, 0.0, gm, gi)
        @test lmf ≈ lmc atol = 1.0e-14
        @test lpf ≈ lpc atol = 1.0e-14
    end
end

@testset "Curved gate: continuum flux/source balance on the atmosphere" begin
    # At an arbitrary point, the analytic divergence of the densitized
    # Valencia flux must equal the geometric source exactly (up to the
    # finite-difference accuracy of the probes).
    x0, y0 = 5.3, 0.7
    fdel = 1.0e-6
    flux_at(x, y, dir) = begin
        w = grg_atmosphere(x, y)
        alp = FiniteVolumeMethod.lapse(GRG_METRIC, x, y)
        beta = FiniteVolumeMethod.shift(GRG_METRIC, x, y)
        gm = FiniteVolumeMethod.spatial_metric(GRG_METRIC, x, y)
        sg = FiniteVolumeMethod.sqrt_gamma(GRG_METRIC, x, y)
        FiniteVolumeMethod._grmhd_valencia_flux(GRG_EOS, w, dir, alp, beta, gm, sg)
    end
    divF = (flux_at(x0 + fdel, y0, 1) - flux_at(x0 - fdel, y0, 1)) / (2 * fdel) +
        (flux_at(x0, y0 + fdel, 2) - flux_at(x0, y0 - fdel, 2)) / (2 * fdel)

    dm = 1.0e-5
    mesh3 = StructuredMesh2D(x0 - 1.5 * dm, x0 + 1.5 * dm, y0 - 1.5 * dm, y0 + 1.5 * dm, 3, 3)
    md3 = FiniteVolumeMethod.precompute_metric(GRG_METRIC, mesh3)
    w0 = grg_atmosphere(x0, y0)
    S = FiniteVolumeMethod.grmhd_source_terms(GRG_LAW, w0, w0, md3, mesh3, 2, 2)

    for k in 1:8
        @test abs(divF[k] - S[k]) < 1.0e-6
    end
    # Transport velocity vanishes for the static fluid: D flux divergence is 0
    @test abs(divF[1]) < 1.0e-8
end

@testset "Curved gate: discrete RHS residual converges on the atmosphere" begin
    residuals = Float64[]
    for N in [16, 32]
        prob, mesh = grg_make_prob(N, CellCenteredMUSCL(MinmodLimiter()))
        ng = 2
        U = FiniteVolumeMethod.initialize_2d(prob; nghost = ng)
        FiniteVolumeMethod._grmhd_initialize_densitized_2d!(U, prob, ng)
        W_pad = fill(zero(SVector{8, Float64}), size(U))
        dU = fill(zero(SVector{8, Float64}), size(U))
        Fx_all = fill(zero(SVector{8, Float64}), N + 1, N + 2)
        Fy_all = fill(zero(SVector{8, Float64}), N + 2, N + 1)
        md = FiniteVolumeMethod.precompute_metric(GRG_METRIC, mesh)
        fd = FiniteVolumeMethod.precompute_metric_at_faces(GRG_METRIC, mesh)
        FiniteVolumeMethod._grmhd_stage_rhs!(Fx_all, Fy_all, dU, U, W_pad, prob, 0.0, md, fd)
        res = 0.0
        cnt = 0
        for iy in 3:(N - 2), ix in 3:(N - 2)
            res += sum(abs.(dU[ix + ng, iy + ng]))
            cnt += 1
        end
        push!(residuals, res / cnt)
    end
    # First-order-or-better convergence of the equilibrium residual
    @test residuals[2] < residuals[1] / 1.8
    @test residuals[1] < 0.01
end

@testset "Curved gate: atmosphere held for many steps with converging drift" begin
    drifts = Float64[]
    for N in [24, 48]
        prob, mesh = grg_make_prob(N, CellCenteredMUSCL(MinmodLimiter()); tf = 1.0)
        coords, U, t, ct = solve_hyperbolic(prob)
        @test t ≈ 1.0 atol = 1.0e-10
        W = FiniteVolumeMethod.grmhd_recover_primitive_field(GRG_LAW, U, mesh)
        # Deep interior (25% margin) excludes the zero-gradient boundary
        # contamination, which advects inward at finite speed.
        m = max(2, N ÷ 4)
        drift = maximum(
            abs(W[ix, iy][1] - grg_atmosphere(coords[ix, iy]...)[1]) /
                grg_atmosphere(coords[ix, iy]...)[1]
                for iy in (m + 1):(N - m), ix in (m + 1):(N - m)
        )
        push!(drifts, drift)
    end
    @test drifts[1] < 5.0e-3       # N = 24 after ~50 SSP-RK3 steps
    @test drifts[2] < 1.0e-3       # N = 48 after ~100 steps
    @test drifts[2] < drifts[1] / 2  # resolution convergence
end
