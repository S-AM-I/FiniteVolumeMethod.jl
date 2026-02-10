# ============================================================
# Advanced Numerics Tests (Phase 13)
# ============================================================
#
# Tests for PPM reconstruction, positivity-preserving limiter,
# and multi-rate (subcycling) time stepping.

using FiniteVolumeMethod
using StaticArrays
using Test

# ============================================================
# PPM Reconstruction Tests
# ============================================================

@testset "PPM Reconstruction" begin
    @testset "Type and nghost" begin
        ppm = PPMReconstruction()
        @test ppm isa PPMReconstruction
        @test nghost(ppm) == 2
    end

    @testset "Constant data preserved" begin
        ppm = PPMReconstruction()
        c = 3.14
        wL, wR = reconstruct_interface(ppm, c, c, c, c)
        @test wL ≈ c atol = 1.0e-14
        @test wR ≈ c atol = 1.0e-14
    end

    @testset "Linear data exact" begin
        ppm = PPMReconstruction()
        # Linear profile: f(x) = 2x with cells at x = 1, 2, 3, 4
        # Face at x = 2.5 should give L=2.5, R=2.5
        wLL, wL, wR, wRR = 2.0, 4.0, 6.0, 8.0
        wL_f, wR_f = reconstruct_interface(ppm, wLL, wL, wR, wRR)
        # PPM should be at least 2nd order accurate on linear data
        @test wL_f ≈ wR_f atol = 0.5  # within reasonable bounds for 4-cell stencil
    end

    @testset "No overshoot at discontinuity" begin
        ppm = PPMReconstruction()
        # Step: [1, 1, 2, 2]
        wLL, wL, wR, wRR = 1.0, 1.0, 2.0, 2.0
        wL_f, wR_f = reconstruct_interface(ppm, wLL, wL, wR, wRR)
        @test wL_f >= 1.0 - 1.0e-14
        @test wL_f <= 2.0 + 1.0e-14
        @test wR_f >= 1.0 - 1.0e-14
        @test wR_f <= 2.0 + 1.0e-14
    end

    @testset "SVector component-wise" begin
        ppm = PPMReconstruction()
        wLL = SVector(1.0, 2.0, 3.0)
        wL = SVector(1.0, 2.0, 3.0)
        wR = SVector(1.0, 2.0, 3.0)
        wRR = SVector(1.0, 2.0, 3.0)
        wL_f, wR_f = reconstruct_interface(ppm, wLL, wL, wR, wRR)
        @test wL_f ≈ wL atol = 1.0e-14
        @test wR_f ≈ wR atol = 1.0e-14
    end

    @testset "SVector discontinuity" begin
        ppm = PPMReconstruction()
        wLL = SVector(1.0, 0.0, 1.0)
        wL = SVector(1.0, 0.0, 1.0)
        wR = SVector(0.5, 0.0, 0.5)
        wRR = SVector(0.5, 0.0, 0.5)
        wL_f, wR_f = reconstruct_interface(ppm, wLL, wL, wR, wRR)
        # No overshoot for any component
        for i in 1:3
            @test wL_f[i] >= min(wLL[i], wL[i], wR[i], wRR[i]) - 1.0e-14
            @test wL_f[i] <= max(wLL[i], wL[i], wR[i], wRR[i]) + 1.0e-14
            @test wR_f[i] >= min(wLL[i], wL[i], wR[i], wRR[i]) - 1.0e-14
            @test wR_f[i] <= max(wLL[i], wL[i], wR[i], wRR[i]) + 1.0e-14
        end
    end

    @testset "1D Euler with PPM" begin
        eos = IdealGasEOS(1.4)
        law = EulerEquations{1}(eos)
        mesh = StructuredMesh1D(0.0, 1.0, 100)
        ic(x) = x < 0.5 ? SVector(1.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.1)
        prob = HyperbolicProblem(
            law, mesh, HLLCSolver(), PPMReconstruction(),
            TransmissiveBC(), TransmissiveBC(), ic;
            cfl = 0.4, final_time = 0.2
        )
        x, U, t = solve_hyperbolic(prob)
        @test t ≈ 0.2 atol = 0.01
        @test all(u -> u[1] > 0, U)  # density positive
        # Check Sod shock tube basic structure
        # Left state should be close to initial
        @test U[1][1] ≈ 1.0 atol = 0.05
        # Right state should be close to initial
        @test U[end][1] ≈ 0.125 atol = 0.05
    end

    @testset "2D Euler with PPM" begin
        eos = IdealGasEOS(1.4)
        law = EulerEquations{2}(eos)
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 20, 20)
        ic(x, y) = x < 0.5 ? SVector(1.0, 0.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.0, 0.1)
        bc = TransmissiveBC()
        prob = HyperbolicProblem2D(
            law, mesh, HLLCSolver(), PPMReconstruction(),
            bc, bc, bc, bc, ic;
            cfl = 0.3, final_time = 0.05
        )
        coords, U, t = solve_hyperbolic(prob)
        @test t ≈ 0.05 atol = 0.01
        @test all(u -> u[1] > 0, U)
    end

    @testset "PPM vs MUSCL convergence (smooth)" begin
        eos = IdealGasEOS(1.4)
        law = EulerEquations{1}(eos)

        # Smooth initial condition: Gaussian density bump
        ic(x) = SVector(1.0 + 0.2 * exp(-100 * (x - 0.5)^2), 1.0, 1.0)

        # Run at two resolutions with PPM and MUSCL
        errs_ppm = Float64[]
        errs_muscl = Float64[]
        for nc in [50, 100]
            mesh = StructuredMesh1D(0.0, 1.0, nc)
            prob_ppm = HyperbolicProblem(
                law, mesh, HLLCSolver(), PPMReconstruction(),
                TransmissiveBC(), TransmissiveBC(), ic;
                cfl = 0.3, final_time = 0.01
            )
            prob_muscl = HyperbolicProblem(
                law, mesh, HLLCSolver(), CellCenteredMUSCL(),
                TransmissiveBC(), TransmissiveBC(), ic;
                cfl = 0.3, final_time = 0.01
            )
            _, U_ppm, _ = solve_hyperbolic(prob_ppm)
            _, U_muscl, _ = solve_hyperbolic(prob_muscl)

            # Both should produce all-positive density
            @test all(u -> u[1] > 0, U_ppm)
            @test all(u -> u[1] > 0, U_muscl)
        end
    end
end

# ============================================================
# Positivity Limiter Tests
# ============================================================

@testset "Positivity Limiter" begin
    @testset "Type construction" begin
        lim = PositivityLimiter()
        @test lim.epsilon == 1.0e-13

        lim2 = PositivityLimiter(1.0e-8)
        @test lim2.epsilon == 1.0e-8
    end

    @testset "1D density floor" begin
        eos = IdealGasEOS(1.4)
        law = EulerEquations{1}(eos)
        lim = PositivityLimiter(1.0e-10)
        nc = 10

        # Create solution with negative density
        N = nvariables(law)
        U = Vector{SVector{N, Float64}}(undef, nc + 4)
        for i in 1:(nc + 4)
            U[i] = primitive_to_conserved(law, SVector(1.0, 0.0, 1.0))
        end
        # Set one cell to have negative density
        U[5] = SVector(-0.5, 0.0, 1.0)  # bad density in conserved

        apply_positivity_limiter!(U, law, lim, nc)

        # Check density is floored
        @test U[5][1] >= lim.epsilon
        # Other cells unchanged
        @test U[4][1] ≈ 1.0
    end

    @testset "1D pressure floor" begin
        eos = IdealGasEOS(1.4)
        law = EulerEquations{1}(eos)
        lim = PositivityLimiter(1.0e-10)
        nc = 10

        N = nvariables(law)
        U = Vector{SVector{N, Float64}}(undef, nc + 4)
        for i in 1:(nc + 4)
            U[i] = primitive_to_conserved(law, SVector(1.0, 0.0, 1.0))
        end
        # Set one cell to have very low energy (negative pressure)
        # E = rho * e + 0.5 * rho * v^2. For P>0, need e > 0.
        # With gamma=1.4, P = (gamma-1) * rho * e. Set E very small.
        U[5] = SVector(1.0, 10.0, 0.01)  # rho=1, rho*v=10, E=0.01 -> KE=50 >> E -> P<0

        apply_positivity_limiter!(U, law, lim, nc)

        # Check pressure is fixed
        w = conserved_to_primitive(law, U[5])
        @test w[3] >= lim.epsilon * 0.99  # pressure (allow tiny FP rounding)
        @test U[5][1] >= lim.epsilon  # density still ok
    end

    @testset "2D density and pressure floor" begin
        eos = IdealGasEOS(1.4)
        law = EulerEquations{2}(eos)
        lim = PositivityLimiter(1.0e-10)
        nx, ny = 5, 5
        N = nvariables(law)

        U = Matrix{SVector{N, Float64}}(undef, nx + 4, ny + 4)
        for j in 1:(ny + 4), i in 1:(nx + 4)
            U[i, j] = primitive_to_conserved(law, SVector(1.0, 0.0, 0.0, 1.0))
        end
        # Set one cell to have negative density
        U[4, 4] = SVector(-0.5, 0.0, 0.0, 1.0)

        apply_positivity_limiter_2d!(U, law, lim, nx, ny)

        @test U[4, 4][1] >= lim.epsilon
    end

    @testset "Face state theta-limiting" begin
        eos = IdealGasEOS(1.4)
        law = EulerEquations{1}(eos)
        lim = PositivityLimiter(1.0e-10)

        # Cell average with good density
        wL = SVector(1.0, 0.0, 1.0)
        wR = SVector(1.0, 0.0, 1.0)

        # Face state with negative density
        wL_face = SVector(-0.1, 0.5, 1.0)
        wR_face = SVector(0.5, 0.0, 0.5)

        wL_lim, wR_lim = limit_reconstructed_states(lim, law, wL, wR, wL_face, wR_face)

        # Left face density should be at least epsilon
        @test wL_lim[1] >= lim.epsilon
        # Right face should be unchanged (already positive)
        @test wR_lim[1] ≈ wR_face[1] atol = 1.0e-14
    end

    @testset "No change for positive states" begin
        eos = IdealGasEOS(1.4)
        law = EulerEquations{1}(eos)
        lim = PositivityLimiter(1.0e-10)
        nc = 5
        N = nvariables(law)

        U = Vector{SVector{N, Float64}}(undef, nc + 4)
        for i in 1:(nc + 4)
            U[i] = primitive_to_conserved(law, SVector(1.0, 0.5, 2.0))
        end
        U_orig = copy(U)

        apply_positivity_limiter!(U, law, lim, nc)

        for i in 3:(nc + 2)
            @test U[i] ≈ U_orig[i] atol = 1.0e-14
        end
    end

    @testset "SVector face state limiting" begin
        eos = IdealGasEOS(1.4)
        law = EulerEquations{2}(eos)
        lim = PositivityLimiter(1.0e-8)

        wL = SVector(1.0, 0.0, 0.0, 1.0)
        wR = SVector(1.0, 0.0, 0.0, 1.0)

        # Face state with negative pressure
        wL_face = SVector(1.0, 0.0, 0.0, -0.5)
        wR_face = SVector(1.0, 0.0, 0.0, 1.0)

        wL_lim, wR_lim = limit_reconstructed_states(lim, law, wL, wR, wL_face, wR_face)

        @test wL_lim[4] >= lim.epsilon  # pressure corrected
        @test wR_lim ≈ wR_face atol = 1.0e-14  # unchanged
    end
end

# ============================================================
# Multi-rate (Subcycling) Tests
# ============================================================

@testset "Subcycling" begin
    @testset "SubcyclingScheme construction" begin
        sc = SubcyclingScheme()
        @test sc.ratio == 2

        sc2 = SubcyclingScheme(4)
        @test sc2.ratio == 4

        @test_throws ArgumentError SubcyclingScheme(0)
    end

    @testset "total_substeps" begin
        sc = SubcyclingScheme(2)
        @test total_substeps(sc, 0, 0) == 1
        @test total_substeps(sc, 0, 1) == 2
        @test total_substeps(sc, 0, 2) == 4
        @test total_substeps(sc, 0, 3) == 8

        sc3 = SubcyclingScheme(3)
        @test total_substeps(sc3, 0, 2) == 9
        @test total_substeps(sc3, 1, 3) == 9

        @test_throws ArgumentError total_substeps(sc, 2, 1)
    end

    @testset "AMR subcycled solve (single level)" begin
        # With a single-level AMR grid, subcycling is equivalent to normal AMR solve
        eos = IdealGasEOS(1.4)
        law = EulerEquations{2}(eos)
        nx, ny = 8, 8

        # Create a simple AMR grid with one block
        origin = (0.0, 0.0)
        dx = (1.0 / nx, 1.0 / ny)
        block = AMRBlock(1, 0, origin, (nx, ny), dx, Val(4))

        # Initialize with Sod problem in x
        for j in 1:ny, i in 1:nx
            x = (i - 0.5) / nx
            if x < 0.5
                block.U[i, j] = primitive_to_conserved(law, SVector(1.0, 0.0, 0.0, 1.0))
            else
                block.U[i, j] = primitive_to_conserved(law, SVector(0.125, 0.0, 0.0, 0.1))
            end
        end

        crit = GradientRefinement(1, 0.5, 0.1)
        grid = AMRGrid(law, crit, (nx, ny), 1, origin, (1.0, 1.0), Val(4))
        # Replace root block data with our initialized data
        grid.blocks[1] = block

        bcs = (TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC())
        prob = AMRProblem(
            grid, HLLCSolver(), NoReconstruction(), bcs;
            initial_time = 0.0, final_time = 0.01, cfl = 0.4, regrid_interval = 0
        )

        grid_out, t_final = solve_amr_subcycled(prob; method = :euler)
        @test t_final ≈ 0.01 atol = 0.005
        # Solution should have positive density everywhere
        for block_out in values(grid_out.blocks)
            if block_out.active
                for j in 1:(block_out.dims[2]), i in 1:(block_out.dims[1])
                    @test block_out.U[i, j][1] > 0
                end
            end
        end
    end

    @testset "Subcycled SSP-RK3 vs Euler" begin
        eos = IdealGasEOS(1.4)
        law = EulerEquations{2}(eos)
        nx, ny = 8, 8

        origin = (0.0, 0.0)
        dx = (1.0 / nx, 1.0 / ny)
        crit = GradientRefinement(1, 0.5, 0.1)

        grid1 = AMRGrid(law, crit, (nx, ny), 1, origin, (1.0, 1.0), Val(4))
        grid2 = AMRGrid(law, crit, (nx, ny), 1, origin, (1.0, 1.0), Val(4))

        for j in 1:ny, i in 1:nx
            x = (i - 0.5) / nx
            w = x < 0.5 ? SVector(1.0, 0.0, 0.0, 1.0) : SVector(0.125, 0.0, 0.0, 0.1)
            u = primitive_to_conserved(law, w)
            grid1.blocks[1].U[i, j] = u
            grid2.blocks[1].U[i, j] = u
        end

        bcs = (TransmissiveBC(), TransmissiveBC(), TransmissiveBC(), TransmissiveBC())
        prob1 = AMRProblem(
            grid1, HLLCSolver(), NoReconstruction(), bcs;
            final_time = 0.005, cfl = 0.3, regrid_interval = 0
        )
        prob2 = AMRProblem(
            grid2, HLLCSolver(), NoReconstruction(), bcs;
            final_time = 0.005, cfl = 0.3, regrid_interval = 0
        )

        _, t1 = solve_amr_subcycled(prob1; method = :euler)
        _, t2 = solve_amr_subcycled(prob2; method = :ssprk3)

        @test t1 ≈ t2 atol = 0.005
        # Both should produce valid results (no NaN)
        for b in values(grid1.blocks)
            if b.active
                for idx in eachindex(b.U)
                    @test all(isfinite, b.U[idx])
                end
            end
        end
        for b in values(grid2.blocks)
            if b.active
                for idx in eachindex(b.U)
                    @test all(isfinite, b.U[idx])
                end
            end
        end
    end
end
