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
            @test W[ix, iy][1] ≈ W[ix, 1][1] atol = 1.0e-10
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
            @test W[ix, iy][1] ≈ W[1, iy][1] atol = 1.0e-10
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
    @test divB_max < 1.0e-13
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

    @test D_total ≈ D_total_0 rtol = 1.0e-10
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

# ============================================================
# Balsara 1 in 2D (x-direction)
# ============================================================
@testset "Balsara 1 in 2D" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 200, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)
    ic(x, y) = x < 0.5 ? wL : wR

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.4, cfl = 0.3
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][5] > 0 for ix in 1:nx, iy in 1:ny)
    # Solution should have evolved
    @test !all(W[ix, 1][1] ≈ wL[1] for ix in 1:nx)
end

# ============================================================
# Balsara 2 in 2D (mildly relativistic)
# ============================================================
@testset "Balsara 2 in 2D" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 200, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 30.0, 5.0, 6.0, 6.0)
    wR = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.7, 0.7)
    ic(x, y) = x < 0.5 ? wL : wR

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.4, cfl = 0.25
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][5] > 0 for ix in 1:nx, iy in 1:ny)
end

# ============================================================
# Rotational Sod test (diagonal shock propagation)
# ============================================================
@testset "2D SRMHD diagonal shock" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 40, 40
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    # Diagonal Riemann problem: discontinuity along x + y = 1
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, 0.5, 0.0)
    ic(x, y) = (x + y) < 1.0 ? wL : wR

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.1, cfl = 0.25
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][5] > 0 for ix in 1:nx, iy in 1:ny)
end

# ============================================================
# Angled Flow Test
# ============================================================
@testset "Angled flow" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 32, 32
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    # Uniform flow at 45 degrees
    v_mag = 0.3
    vx0 = v_mag / sqrt(2.0)
    vy0 = v_mag / sqrt(2.0)
    w0 = SVector(1.0, vx0, vy0, 0.0, 1.0, 0.5, 0.5, 0.0)
    ic(x, y) = w0

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.1, cfl = 0.3
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    # Uniform flow should remain uniform with periodic BCs
    for iy in 1:ny, ix in 1:nx
        @test W[ix, iy][1] ≈ w0[1] atol = 1.0e-8
        @test W[ix, iy][2] ≈ w0[2] atol = 1.0e-8
        @test W[ix, iy][3] ≈ w0[3] atol = 1.0e-8
        @test W[ix, iy][5] ≈ w0[5] atol = 1.0e-8
    end
end

# ============================================================
# All Limiters in 2D
# ============================================================
@testset "All Limiters 2D" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 40, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)
    ic(x, y) = x < 0.5 ? wL : wR

    limiters = [
        MinmodLimiter(), SuperbeeLimiter(), VanLeerLimiter(),
        KorenLimiter(), OspreLimiter(),
    ]

    for lim in limiters
        prob = HyperbolicProblem2D(
            law, mesh, HLLSolver(), CellCenteredMUSCL(lim),
            TransmissiveBC(), TransmissiveBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            ic; final_time = 0.1, cfl = 0.25
        )

        coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
        W = to_primitive(law, U)

        @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
        @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    end
end

# ============================================================
# Riemann Solvers Comparison
# ============================================================
@testset "Riemann Solvers 2D" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 40, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)
    ic(x, y) = x < 0.5 ? wL : wR

    for solver in [LaxFriedrichsSolver(), HLLSolver()]
        prob = HyperbolicProblem2D(
            law, mesh, solver, CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            ic; final_time = 0.1, cfl = 0.25
        )

        coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
        W = to_primitive(law, U)

        @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
        @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    end
end

# ============================================================
# Newtonian Limit in 2D
# ============================================================
@testset "2D Newtonian Limit" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law_sr = SRMHDEquations{2}(eos)
    law_mhd = IdealMHDEquations{2}(eos)
    nx, ny = 60, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    # Low-velocity Riemann problem
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0)
    ic(x, y) = x < 0.5 ? wL : wR

    prob_sr = HyperbolicProblem2D(
        law_sr, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.1, cfl = 0.25
    )

    prob_mhd = HyperbolicProblem2D(
        law_mhd, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.1, cfl = 0.25
    )

    _, U_sr, _, _ = solve_hyperbolic(prob_sr; vector_potential = nothing)
    _, U_mhd, _, _ = solve_hyperbolic(prob_mhd; vector_potential = nothing)

    W_sr = to_primitive(law_sr, U_sr)
    W_mhd = to_primitive(law_mhd, U_mhd)

    # Density profiles should be similar in the Newtonian limit
    ρ_sr = [W_sr[ix, 1][1] for ix in 1:nx]
    ρ_mhd = [W_mhd[ix, 1][1] for ix in 1:nx]
    l2_err = sqrt(sum((ρ_sr .- ρ_mhd) .^ 2) / nx)
    @test l2_err < 0.15
end

# ============================================================
# Pressure Pulse (Cylindrical Blast)
# ============================================================
@testset "2D SRMHD Pressure Pulse" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 40, 40
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    ic(x, y) = begin
        r = sqrt((x - 0.5)^2 + (y - 0.5)^2)
        P = r < 0.1 ? 100.0 : 1.0
        SVector(1.0, 0.0, 0.0, 0.0, P, 1.0, 0.0, 0.0)
    end

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.05, cfl = 0.2
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][5] > 0 for ix in 1:nx, iy in 1:ny)

    # Symmetry: solution should be approximately symmetric about center
    # Compare density at symmetric points
    ρ_ul = W[nx ÷ 2 - 5, ny ÷ 2 + 5][1]
    ρ_lr = W[nx ÷ 2 + 5, ny ÷ 2 - 5][1]
    @test ρ_ul ≈ ρ_lr rtol = 0.05
end

# ============================================================
# Forward Euler Time Integration
# ============================================================
@testset "Forward Euler" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 40, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)
    ic(x, y) = x < 0.5 ? wL : wR

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.05, cfl = 0.3
    )

    coords, U, t, ct = solve_hyperbolic(prob; method = :euler, vector_potential = nothing)
    W = to_primitive(law, U)

    @test t ≈ 0.05 atol = 1.0e-10
    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
end

# ============================================================
# NoReconstruction Stability
# ============================================================
@testset "NoReconstruction" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 40, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)
    ic(x, y) = x < 0.5 ? wL : wR

    prob = HyperbolicProblem2D(
        law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.1, cfl = 0.4
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
end

# ============================================================
# High Lorentz Factor Stability
# ============================================================
@testset "High Lorentz factor" begin
    eos = IdealGasEOS(gamma = 4.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 40, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    # Moderately relativistic flow (W ~ 2)
    v = 0.866  # W ≈ 2
    wL = SVector(1.0, v, 0.0, 0.0, 10.0, 0.5, 1.0, 0.0)
    wR = SVector(0.1, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)
    ic(x, y) = x < 0.5 ? wL : wR

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.1, cfl = 0.2
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = nothing)
    W = to_primitive(law, U)

    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][5] > 0 for ix in 1:nx, iy in 1:ny)
end

# ============================================================
# Full Conservation (D, Sx, Sy, tau)
# ============================================================
@testset "Full Conservation" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 32, 32
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    ic(x, y) = SVector(
        1.0 + 0.2 * sin(2π * x) * cos(2π * y),
        0.1 * cos(2π * x), 0.05 * sin(4π * y), 0.0,
        2.0 + 0.1 * cos(2π * x) * sin(2π * y),
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

    U0 = [FiniteVolumeMethod.primitive_to_conserved(law, ic(coords[ix, iy]...)) for ix in 1:nx, iy in 1:ny]

    for var in [1, 5]  # D and tau
        total = sum(U[ix, iy][var] for ix in 1:nx, iy in 1:ny) * dV
        total_0 = sum(U0[ix, iy][var] for ix in 1:nx, iy in 1:ny) * dV
        @test total ≈ total_0 rtol = 1.0e-9
    end
end

# ============================================================
# Different EOS (gamma = 4/3, radiation dominated)
# ============================================================
@testset "Radiation-dominated EOS" begin
    eos = IdealGasEOS(gamma = 4.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 40, 4
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 0.1, nx, ny)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0)
    ic(x, y) = x < 0.5 ? wL : wR

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
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

# ============================================================
# Vector Potential CT with SRMHD Field Loop
# ============================================================
@testset "CT Field Loop" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = SRMHDEquations{2}(eos)
    nx, ny = 32, 32
    mesh = StructuredMesh2D(-0.5, 0.5, -0.5, 0.5, nx, ny)

    # Circular field loop via vector potential
    R0 = 0.3
    Az(x, y) = begin
        r = sqrt(x^2 + y^2)
        r < R0 ? 1.0e-3 * (R0 - r) : 0.0
    end

    ic(x, y) = SVector(1.0, 0.1, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0)

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.1, cfl = 0.3
    )

    coords, U, t, ct = solve_hyperbolic(prob; vector_potential = Az)

    # div(B) should be preserved to machine precision
    divB_max = max_divB(ct, mesh.dx, mesh.dy, nx, ny)
    @test divB_max < 1.0e-13

    # Solution should remain physical
    W = to_primitive(law, U)
    @test all(isfinite(W[ix, iy][1]) for ix in 1:nx, iy in 1:ny)
    @test all(W[ix, iy][1] > 0 for ix in 1:nx, iy in 1:ny)
end
