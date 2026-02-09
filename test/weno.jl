using FiniteVolumeMethod
using Test
using StaticArrays
using LinearAlgebra

# ============================================================
# Exact Riemann solver for Sod shock tube (used for verification)
# ============================================================

function sod_exact_weno(x, t; x0 = 0.5, gamma = 1.4)
    rhoL, vL, PL = 1.0, 0.0, 1.0
    rhoR, vR, PR = 0.125, 0.0, 0.1
    cL = sqrt(gamma * PL / rhoL)
    cR = sqrt(gamma * PR / rhoR)
    P_star = 0.30313017805064707
    v_star = 0.92745262004895057
    rho_star_L = 0.42631942817849544
    rho_star_R = 0.26557371170530708
    c_star_L = sqrt(gamma * P_star / rho_star_L)
    x_head = x0 - cL * t
    x_tail = x0 + (v_star - c_star_L) * t
    S_shock = vR + cR * sqrt((gamma + 1) / (2 * gamma) * P_star / PR + (gamma - 1) / (2 * gamma))
    x_shock = x0 + S_shock * t
    xi = (x - x0) / t
    if x <= x_head
        return rhoL, vL, PL
    elseif x <= x_tail
        gm1 = gamma - 1
        gp1 = gamma + 1
        v = 2 / gp1 * (cL + xi)
        c = cL - gm1 / 2 * v
        rho = rhoL * (c / cL)^(2 / gm1)
        P = PL * (c / cL)^(2 * gamma / gm1)
        return rho, v, P
    elseif x <= x0 + v_star * t
        return rho_star_L, v_star, P_star
    elseif x <= x_shock
        return rho_star_R, v_star, P_star
    else
        return rhoR, vR, PR
    end
end

# ============================================================
# 1. nghost tests
# ============================================================

@testset "nghost" begin
    @test nghost(NoReconstruction()) == 1
    @test nghost(CellCenteredMUSCL()) == 2
    @test nghost(CellCenteredMUSCL(MinmodLimiter())) == 2
    @test nghost(CellCenteredMUSCL(SuperbeeLimiter())) == 2
    @test nghost(WENO3()) == 2
    @test nghost(WENO3(1.0e-10)) == 2
    @test nghost(WENO5()) == 3
    @test nghost(WENO5(1.0e-10, 2)) == 3
    @test nghost(CharacteristicWENO(WENO3())) == 2
    @test nghost(CharacteristicWENO(WENO5())) == 3
end

# ============================================================
# 2. WENO3 scalar reconstruction
# ============================================================

@testset "WENO3 Scalar Reconstruction" begin
    recon = WENO3()

    @testset "Default epsilon" begin
        @test recon.epsilon == 1.0e-6
    end

    @testset "Custom epsilon" begin
        r = WENO3(1.0e-12)
        @test r.epsilon == 1.0e-12
    end

    @testset "Uniform data returns cell value" begin
        # For uniform data, reconstruction should return the cell value exactly
        val = 3.5
        wL_face, wR_face = reconstruct_interface(recon, val, val, val, val)
        @test wL_face ≈ val atol = 1.0e-14
        @test wR_face ≈ val atol = 1.0e-14
    end

    @testset "Linear data (exact for 3rd order)" begin
        # Linear function: f(x) = 2x + 1 on cells centered at x = 0, 1, 2, 3
        # Cell averages equal point values for linear functions
        wLL = 1.0   # f(0)
        wL = 3.0    # f(1)
        wR = 5.0    # f(2)
        wRR = 7.0   # f(3)

        # Interface is between wL and wR, at x = 1.5
        # Expected value: f(1.5) = 4.0
        wL_face, wR_face = reconstruct_interface(recon, wLL, wL, wR, wRR)

        # Both left and right reconstructions should give 4.0 for linear data
        @test wL_face ≈ 4.0 atol = 1.0e-10
        @test wR_face ≈ 4.0 atol = 1.0e-10
    end

    @testset "Smooth quadratic data" begin
        # Quadratic: f(x) = x^2, cells at x = -1, 0, 1, 2
        wLL = 1.0   # (-1)^2
        wL = 0.0    # 0^2
        wR = 1.0    # 1^2
        wRR = 4.0   # 2^2

        # Interface at x = 0.5, exact value = 0.25
        wL_face, wR_face = reconstruct_interface(recon, wLL, wL, wR, wRR)

        # WENO3 should be close to the quadratic value but not exact
        # (3rd order accuracy)
        @test abs(wL_face - 0.25) < 0.5
        @test abs(wR_face - 0.25) < 0.5
        # Both should be reasonable (non-negative for this function)
        @test isfinite(wL_face)
        @test isfinite(wR_face)
    end

    @testset "Discontinuous data (no oscillation)" begin
        # Step function: 0, 0, 1, 1
        wLL = 0.0
        wL = 0.0
        wR = 1.0
        wRR = 1.0

        wL_face, wR_face = reconstruct_interface(recon, wLL, wL, wR, wRR)

        # Left reconstruction should be between 0 and 1 (no undershoot)
        @test wL_face >= -0.01
        @test wL_face <= 1.01

        # Right reconstruction should be between 0 and 1 (no overshoot)
        @test wR_face >= -0.01
        @test wR_face <= 1.01

        # Left state should lean toward 0, right state toward 1
        @test wL_face < wR_face
    end

    @testset "Symmetry" begin
        # Symmetric data around the interface
        wLL = 1.0
        wL = 2.0
        wR = 2.0
        wRR = 1.0

        wL_face, wR_face = reconstruct_interface(recon, wLL, wL, wR, wRR)

        # By symmetry, both face values should be equal
        @test wL_face ≈ wR_face atol = 1.0e-14
    end

    @testset "Internal _weno3 functions" begin
        eps = 1.0e-6

        # Test left reconstruction with uniform data
        val = 5.0
        result = FiniteVolumeMethod._weno3_reconstruct_left(val, val, val, eps)
        @test result ≈ val atol = 1.0e-14

        # Test right reconstruction with uniform data
        result_r = FiniteVolumeMethod._weno3_reconstruct_right(val, val, val, eps)
        @test result_r ≈ val atol = 1.0e-14

        # Test left reconstruction with linear data: v0=0, v1=1, v2=2
        # Face at x=1.5 should give 1.5
        left_val = FiniteVolumeMethod._weno3_reconstruct_left(0.0, 1.0, 2.0, eps)
        @test left_val ≈ 1.5 atol = 1.0e-10

        # Test right reconstruction with linear data: v0=0, v1=1, v2=2
        # Face at x=0.5 should give 0.5
        right_val = FiniteVolumeMethod._weno3_reconstruct_right(0.0, 1.0, 2.0, eps)
        @test right_val ≈ 0.5 atol = 1.0e-10
    end
end

# ============================================================
# 3. WENO5 scalar reconstruction
# ============================================================

@testset "WENO5 Scalar Reconstruction" begin
    recon = WENO5()

    @testset "Default parameters" begin
        @test recon.epsilon == 1.0e-6
        @test recon.p == 2
    end

    @testset "Custom parameters" begin
        r = WENO5(1.0e-12, 1)
        @test r.epsilon == 1.0e-12
        @test r.p == 1
    end

    @testset "Uniform data" begin
        val = 2.7
        left_face, right_face = reconstruct_interface_weno5(recon, val, val, val, val, val, val)
        @test left_face ≈ val atol = 1.0e-14
        @test right_face ≈ val atol = 1.0e-14
    end

    @testset "Linear data (exact)" begin
        # Linear: f(x) = 3x + 1, cells at x = -2, -1, 0, 1, 2, 3
        v1 = -5.0   # f(-2)
        v2 = -2.0   # f(-1)
        v3 = 1.0    # f(0)
        v4 = 4.0    # f(1)
        v5 = 7.0    # f(2)
        v6 = 10.0   # f(3)

        # Interface between v3 and v4, at x = 0.5; exact = 2.5
        left_face, right_face = reconstruct_interface_weno5(recon, v1, v2, v3, v4, v5, v6)
        @test left_face ≈ 2.5 atol = 1.0e-10
        @test right_face ≈ 2.5 atol = 1.0e-10
    end

    @testset "Quadratic data" begin
        # Quadratic: f(x) = x^2, cells at x = -2, -1, 0, 1, 2, 3
        v1 = 4.0
        v2 = 1.0
        v3 = 0.0
        v4 = 1.0
        v5 = 4.0
        v6 = 9.0

        # Interface at x = 0.5, exact = 0.25
        left_face, right_face = reconstruct_interface_weno5(recon, v1, v2, v3, v4, v5, v6)

        # WENO5 should be quite close for smooth data
        @test abs(left_face - 0.25) < 0.1
        @test abs(right_face - 0.25) < 0.1
    end

    @testset "Discontinuous data (bounded)" begin
        # Step: 0, 0, 0, 1, 1, 1
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        v4 = 1.0
        v5 = 1.0
        v6 = 1.0

        left_face, right_face = reconstruct_interface_weno5(recon, v1, v2, v3, v4, v5, v6)

        # Should not produce large oscillations
        @test left_face >= -0.2
        @test left_face <= 1.2
        @test right_face >= -0.2
        @test right_face <= 1.2
        @test isfinite(left_face)
        @test isfinite(right_face)
    end

    @testset "WENO5 higher accuracy than WENO3 on smooth data" begin
        # Use a cubic: f(x) = x^3, cells at x = -2, -1, 0, 1, 2, 3
        # Interface at x = 0.5, exact = 0.125
        v1 = -8.0
        v2 = -1.0
        v3 = 0.0
        v4 = 1.0
        v5 = 8.0
        v6 = 27.0

        left5, right5 = reconstruct_interface_weno5(WENO5(), v1, v2, v3, v4, v5, v6)

        # WENO3 only uses 4 points: v2, v3, v4, v5
        left3, right3 = reconstruct_interface(WENO3(), v2, v3, v4, v5)

        exact = 0.125
        err5_left = abs(left5 - exact)
        err3_left = abs(left3 - exact)

        # WENO5 should be at least as accurate as WENO3 (or very close)
        @test err5_left < err3_left + 0.1
    end

    @testset "Internal _weno5 functions" begin
        eps_val = 1.0e-6
        p_val = 2

        # Uniform data
        val = 4.2
        result = FiniteVolumeMethod._weno5_reconstruct_left(val, val, val, val, val, eps_val, p_val)
        @test result ≈ val atol = 1.0e-14

        result_r = FiniteVolumeMethod._weno5_reconstruct_right(val, val, val, val, val, eps_val, p_val)
        @test result_r ≈ val atol = 1.0e-14

        # Linear data: v = [1, 2, 3, 4, 5]
        left_val = FiniteVolumeMethod._weno5_reconstruct_left(1.0, 2.0, 3.0, 4.0, 5.0, eps_val, p_val)
        @test left_val ≈ 3.5 atol = 1.0e-10

        right_val = FiniteVolumeMethod._weno5_reconstruct_right(1.0, 2.0, 3.0, 4.0, 5.0, eps_val, p_val)
        @test right_val ≈ 2.5 atol = 1.0e-10

        # Right = mirror of left
        left_mirror = FiniteVolumeMethod._weno5_reconstruct_left(5.0, 4.0, 3.0, 2.0, 1.0, eps_val, p_val)
        @test right_val ≈ left_mirror atol = 1.0e-14
    end
end

# ============================================================
# 4. SVector reconstruction
# ============================================================

@testset "SVector Reconstruction" begin
    @testset "WENO3 SVector{3}" begin
        recon = WENO3()

        # Uniform SVector data
        w = SVector(1.0, 2.0, 3.0)
        wL_face, wR_face = reconstruct_interface(recon, w, w, w, w)
        @test wL_face ≈ w atol = 1.0e-14
        @test wR_face ≈ w atol = 1.0e-14

        # Linear SVector data
        wLL = SVector(0.0, 0.0, 0.0)
        wL = SVector(1.0, 2.0, 3.0)
        wR = SVector(2.0, 4.0, 6.0)
        wRR = SVector(3.0, 6.0, 9.0)

        wL_face, wR_face = reconstruct_interface(recon, wLL, wL, wR, wRR)

        # For linear data, face value = 1.5 * (1,2,3)
        expected = SVector(1.5, 3.0, 4.5)
        @test wL_face ≈ expected atol = 1.0e-10
        @test wR_face ≈ expected atol = 1.0e-10
    end

    @testset "WENO3 SVector component-wise consistency" begin
        recon = WENO3()

        wLL = SVector(1.0, 4.0, 0.0)
        wL = SVector(2.0, 3.0, 1.0)
        wR = SVector(4.0, 1.0, 3.0)
        wRR = SVector(5.0, 0.5, 4.0)

        wL_face, wR_face = reconstruct_interface(recon, wLL, wL, wR, wRR)

        # Each component should match the scalar reconstruction
        for k in 1:3
            sL, sR = reconstruct_interface(recon, wLL[k], wL[k], wR[k], wRR[k])
            @test wL_face[k] ≈ sL atol = 1.0e-14
            @test wR_face[k] ≈ sR atol = 1.0e-14
        end
    end

    @testset "WENO5 SVector{3}" begin
        recon = WENO5()

        # Uniform SVector data
        w = SVector(1.0, 2.0, 3.0)
        left_face, right_face = reconstruct_interface_weno5(recon, w, w, w, w, w, w)
        @test left_face ≈ w atol = 1.0e-14
        @test right_face ≈ w atol = 1.0e-14

        # Linear SVector data
        v1 = SVector(0.0, 0.0, 0.0)
        v2 = SVector(1.0, 2.0, 3.0)
        v3 = SVector(2.0, 4.0, 6.0)
        v4 = SVector(3.0, 6.0, 9.0)
        v5 = SVector(4.0, 8.0, 12.0)
        v6 = SVector(5.0, 10.0, 15.0)

        left_face, right_face = reconstruct_interface_weno5(recon, v1, v2, v3, v4, v5, v6)

        # Face between v3 and v4 at midpoint: expected = 2.5 * (1,2,3)
        expected = SVector(2.5, 5.0, 7.5)
        @test left_face ≈ expected atol = 1.0e-10
        @test right_face ≈ expected atol = 1.0e-10
    end

    @testset "WENO5 SVector component-wise consistency" begin
        recon = WENO5()

        v1 = SVector(1.0, 5.0, 0.1)
        v2 = SVector(2.0, 3.0, 0.5)
        v3 = SVector(4.0, 1.0, 1.0)
        v4 = SVector(5.0, 0.5, 2.0)
        v5 = SVector(5.5, 0.2, 3.0)
        v6 = SVector(5.8, 0.1, 3.5)

        left_face, right_face = reconstruct_interface_weno5(recon, v1, v2, v3, v4, v5, v6)

        # Each component should match the scalar reconstruction
        for k in 1:3
            sL, sR = reconstruct_interface_weno5(recon, v1[k], v2[k], v3[k], v4[k], v5[k], v6[k])
            @test left_face[k] ≈ sL atol = 1.0e-14
            @test right_face[k] ≈ sR atol = 1.0e-14
        end
    end
end

# ============================================================
# 5. Characteristic WENO
# ============================================================

@testset "Characteristic WENO" begin
    eos = IdealGasEOS(1.4)

    @testset "1D Euler eigenvectors: R * L = I" begin
        law = EulerEquations{1}(eos)

        # Test at several states
        states = [
            SVector(1.0, 0.0, 1.0),
            SVector(1.0, 0.5, 1.0),
            SVector(0.5, -1.0, 2.0),
            SVector(2.0, 1.5, 0.5),
        ]

        for w in states
            L = left_eigenvectors(law, w, 1)
            R = right_eigenvectors(law, w, 1)

            product = L * R
            @test product ≈ I(3) atol = 1.0e-12

            # Also check R * L = I
            product2 = R * L
            @test product2 ≈ I(3) atol = 1.0e-12
        end
    end

    @testset "2D Euler eigenvectors: R * L = I" begin
        law = EulerEquations{2}(eos)

        states = [
            SVector(1.0, 0.0, 0.0, 1.0),
            SVector(1.0, 0.5, 0.3, 1.0),
            SVector(0.5, -1.0, 0.5, 2.0),
        ]

        for w in states
            for dir in [1, 2]
                L = left_eigenvectors(law, w, dir)
                R = right_eigenvectors(law, w, dir)

                product = L * R
                @test product ≈ I(4) atol = 1.0e-12
            end
        end
    end

    @testset "CharacteristicWENO construction" begin
        cw3 = CharacteristicWENO(WENO3())
        @test cw3.recon isa WENO3
        @test nghost(cw3) == 2

        cw5 = CharacteristicWENO(WENO5())
        @test cw5.recon isa WENO5
        @test nghost(cw5) == 3
    end

    @testset "CharacteristicWENO reduces to component-wise for identity eigenvectors" begin
        # For a generic law with identity eigenvectors (fallback),
        # characteristic WENO should give same result as plain WENO.

        # Use WENO3 with SVector data
        cw = CharacteristicWENO(WENO3())
        plain = WENO3()

        wLL = SVector(1.0, 2.0, 3.0)
        wL = SVector(2.0, 3.0, 4.0)
        wR = SVector(4.0, 5.0, 6.0)
        wRR = SVector(5.0, 6.0, 7.0)

        # Without law info, CharacteristicWENO should fall back to component-wise
        wL_char, wR_char = reconstruct_interface(cw, wLL, wL, wR, wRR)
        wL_plain, wR_plain = reconstruct_interface(plain, wLL, wL, wR, wRR)

        @test wL_char ≈ wL_plain atol = 1.0e-14
        @test wR_char ≈ wR_plain atol = 1.0e-14
    end

    @testset "Characteristic WENO3 1D dispatch" begin
        law = EulerEquations{1}(eos)
        cw = CharacteristicWENO(WENO3())
        plain = WENO3()

        # Build a small padded array (2 ghost cells each side) with smooth data
        nc = 10
        N = nvariables(law)
        U = Vector{SVector{N, Float64}}(undef, nc + 4)

        for i in 1:(nc + 4)
            x = (i - 2.5) / nc
            rho = 1.0 + 0.1 * sin(2 * pi * x)
            v = 0.1
            P = 1.0
            U[i] = primitive_to_conserved(law, SVector(rho, v, P))
        end

        # Reconstruct at a middle face
        face_idx = 5
        wL_char, wR_char = FiniteVolumeMethod.reconstruct_interface_1d(cw, law, U, face_idx, nc)
        wL_plain, wR_plain = FiniteVolumeMethod.reconstruct_interface_1d(plain, law, U, face_idx, nc)

        # Both should be finite
        @test all(isfinite, wL_char)
        @test all(isfinite, wR_char)
        @test all(isfinite, wL_plain)
        @test all(isfinite, wR_plain)

        # For smooth data they should be similar but not necessarily identical
        # (characteristic projection may slightly change results)
        @test norm(wL_char - wL_plain) < 0.5
        @test norm(wR_char - wR_plain) < 0.5
    end

    @testset "Eigenvector matrix dimensions" begin
        law1d = EulerEquations{1}(eos)
        law2d = EulerEquations{2}(eos)

        w1d = SVector(1.0, 0.5, 1.0)
        L1d = left_eigenvectors(law1d, w1d, 1)
        R1d = right_eigenvectors(law1d, w1d, 1)
        @test size(L1d) == (3, 3)
        @test size(R1d) == (3, 3)

        w2d = SVector(1.0, 0.5, 0.3, 1.0)
        L2d = left_eigenvectors(law2d, w2d, 1)
        R2d = right_eigenvectors(law2d, w2d, 1)
        @test size(L2d) == (4, 4)
        @test size(R2d) == (4, 4)
    end
end

# ============================================================
# 6. Smooth advection convergence (integration test)
# ============================================================

@testset "Smooth Advection Convergence" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)

    # Base state: uniform flow with small density perturbation
    rho0 = 1.0
    v0 = 1.0
    P0 = 1.0
    amp = 0.01  # small amplitude perturbation

    # Short final time so the wave does not wrap around more than once
    final_t = 0.05

    function advection_ic(x)
        rho = rho0 + amp * sin(2 * pi * x)
        return SVector(rho, v0, P0)
    end

    function compute_l1_error(N_cells, recon)
        mesh = StructuredMesh1D(0.0, 1.0, N_cells)
        prob = HyperbolicProblem(
            law, mesh, HLLCSolver(), recon,
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            advection_ic;
            final_time = final_t, cfl = 0.4
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)
        dx = 1.0 / N_cells

        # Exact solution: density wave advects at velocity v0
        err = 0.0
        for i in 1:N_cells
            x_shifted = mod(x[i] - v0 * t, 1.0)
            rho_exact = rho0 + amp * sin(2 * pi * x_shifted)
            err += abs(W[i][1] - rho_exact) * dx
        end
        return err
    end

    @testset "MUSCL convergence" begin
        err_32 = compute_l1_error(32, CellCenteredMUSCL(MinmodLimiter()))
        err_64 = compute_l1_error(64, CellCenteredMUSCL(MinmodLimiter()))
        @test err_64 < err_32
        @test err_32 < 0.01  # Should be small for smooth data
    end

    @testset "WENO3 convergence" begin
        err_32 = compute_l1_error(32, WENO3())
        err_64 = compute_l1_error(64, WENO3())
        @test err_64 < err_32
        @test err_32 < 0.01  # Should be small for smooth data
    end

    @testset "WENO3 error comparable to or smaller than MUSCL" begin
        # At same resolution, WENO3 should be at least comparable to MUSCL-minmod
        err_muscl = compute_l1_error(64, CellCenteredMUSCL(MinmodLimiter()))
        err_weno3 = compute_l1_error(64, WENO3())

        # WENO3 should not be significantly worse than MUSCL
        @test err_weno3 < 2.0 * err_muscl
    end
end

# ============================================================
# 7. 1D Sod Shock Tube with WENO3
# ============================================================

@testset "1D Sod with WENO3" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)

    wL = SVector(1.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.1)

    @testset "WENO3 + HLLC (N=$N)" for N in [32, 64]
        mesh = StructuredMesh1D(0.0, 1.0, N)

        prob = HyperbolicProblem(
            law, mesh, HLLCSolver(), WENO3(),
            DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
            x -> x < 0.5 ? wL : wR;
            final_time = 0.2, cfl = 0.4
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        # Should reach final time
        @test t ≈ 0.2 atol = 1.0e-10

        # No NaN or Inf
        for i in 1:N
            @test all(isfinite, W[i])
        end

        # Density and pressure should be positive
        for i in 1:N
            @test W[i][1] > 0  # rho > 0
            @test W[i][3] > 0  # P > 0
        end

        # Density jump should exist (discontinuity captured)
        rho_min = minimum(W[i][1] for i in 1:N)
        rho_max = maximum(W[i][1] for i in 1:N)
        @test rho_max > 0.5   # Left state rho ~ 1
        @test rho_min < 0.5   # Right state rho ~ 0.125

        # L1 error should be reasonable
        dx = 1.0 / N
        rho_err = sum(abs(W[i][1] - sod_exact_weno(x[i], 0.2)[1]) * dx for i in 1:N)
        @test rho_err < 0.1
    end

    @testset "WENO3 + HLL" begin
        N = 64
        mesh = StructuredMesh1D(0.0, 1.0, N)

        prob = HyperbolicProblem(
            law, mesh, HLLSolver(), WENO3(),
            DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
            x -> x < 0.5 ? wL : wR;
            final_time = 0.2, cfl = 0.4
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t ≈ 0.2 atol = 1.0e-10
        for i in 1:N
            @test all(isfinite, W[i])
            @test W[i][1] > 0
            @test W[i][3] > 0
        end
    end

    @testset "WENO3 + Lax-Friedrichs" begin
        N = 64
        mesh = StructuredMesh1D(0.0, 1.0, N)

        prob = HyperbolicProblem(
            law, mesh, LaxFriedrichsSolver(), WENO3(),
            DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
            x -> x < 0.5 ? wL : wR;
            final_time = 0.2, cfl = 0.4
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t ≈ 0.2 atol = 1.0e-10
        for i in 1:N
            @test all(isfinite, W[i])
            @test W[i][1] > 0
            @test W[i][3] > 0
        end
    end

    @testset "WENO3 conservation with periodic BCs" begin
        N = 64
        mesh = StructuredMesh1D(0.0, 1.0, N)
        dx = mesh.dx

        ic = x -> SVector(1.0 + 0.2 * sin(2 * pi * x), 0.5, 1.0)

        prob = HyperbolicProblem(
            law, mesh, HLLCSolver(), WENO3(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), ic;
            final_time = 0.05, cfl = 0.3
        )

        U0 = FiniteVolumeMethod.initialize_1d(prob)
        FiniteVolumeMethod.apply_boundary_conditions!(U0, prob, 0.0)

        mass0 = sum(U0[i][1] for i in 3:(N + 2)) * dx
        momentum0 = sum(U0[i][2] for i in 3:(N + 2)) * dx
        energy0 = sum(U0[i][3] for i in 3:(N + 2)) * dx

        x, U_final, t = solve_hyperbolic(prob)

        mass_final = sum(U_final[i][1] for i in 1:N) * dx
        momentum_final = sum(U_final[i][2] for i in 1:N) * dx
        energy_final = sum(U_final[i][3] for i in 1:N) * dx

        @test mass_final ≈ mass0 rtol = 1.0e-12
        @test momentum_final ≈ momentum0 rtol = 1.0e-12
        @test energy_final ≈ energy0 rtol = 1.0e-12
    end

    @testset "CharacteristicWENO3 on Sod" begin
        N = 64
        mesh = StructuredMesh1D(0.0, 1.0, N)

        prob = HyperbolicProblem(
            law, mesh, HLLCSolver(), CharacteristicWENO(WENO3()),
            DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
            x -> x < 0.5 ? wL : wR;
            final_time = 0.2, cfl = 0.4
        )

        x, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t ≈ 0.2 atol = 1.0e-10
        for i in 1:N
            @test all(isfinite, W[i])
            @test W[i][1] > 0
            @test W[i][3] > 0
        end

        # L1 error should be reasonable
        dx = 1.0 / N
        rho_err = sum(abs(W[i][1] - sod_exact_weno(x[i], 0.2)[1]) * dx for i in 1:N)
        @test rho_err < 0.1
    end
end
