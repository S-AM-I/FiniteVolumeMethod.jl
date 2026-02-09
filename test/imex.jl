using FiniteVolumeMethod, Test, StaticArrays, LinearAlgebra

# ============================================================
# 1. Tableau Tests
# ============================================================

@testset "IMEX Tableau Properties" begin
    schemes = [
        ("IMEX_SSP3_433", IMEX_SSP3_433(), 4),
        ("IMEX_ARS222", IMEX_ARS222(), 3),
        ("IMEX_Midpoint", IMEX_Midpoint(), 2),
    ]

    for (name, scheme, expected_stages) in schemes
        @testset "$name" begin
            tab = imex_tableau(scheme)
            s = tab.s

            # Number of stages is correct
            @test imex_nstages(scheme) == expected_stages
            @test s == expected_stages

            # Dimensions match
            @test length(tab.A_ex) == s
            @test length(tab.A_im) == s
            @test length(tab.b_ex) == s
            @test length(tab.b_im) == s
            @test length(tab.c_ex) == s
            @test length(tab.c_im) == s
            for k in 1:s
                @test length(tab.A_ex[k]) == s
                @test length(tab.A_im[k]) == s
            end

            # A_ex is strictly lower triangular (diagonal and upper entries are 0)
            for k in 1:s
                for j in k:s
                    @test tab.A_ex[k][j] == 0.0
                end
            end

            # A_im is lower triangular (upper entries are 0)
            for k in 1:s
                for j in (k + 1):s
                    @test tab.A_im[k][j] == 0.0
                end
            end

            # A_im has non-zero diagonal for at least some implicit stages
            # (except for the first stage which may be explicit)
            has_implicit_stage = false
            for k in 1:s
                if tab.A_im[k][k] != 0.0
                    has_implicit_stage = true
                end
            end
            @test has_implicit_stage

            # b_ex sums to 1 (consistency)
            @test sum(tab.b_ex) ≈ 1.0 atol = 1.0e-14

            # b_im sums to 1 (consistency)
            @test sum(tab.b_im) ≈ 1.0 atol = 1.0e-14

            # c_ex = A_ex * ones (row sums match abscissae)
            for k in 1:s
                row_sum = sum(tab.A_ex[k][j] for j in 1:s)
                @test row_sum ≈ tab.c_ex[k] atol = 1.0e-14
            end

            # c_im = A_im * ones (row sums match abscissae)
            for k in 1:s
                row_sum = sum(tab.A_im[k][j] for j in 1:s)
                @test row_sum ≈ tab.c_im[k] atol = 1.0e-14
            end
        end
    end
end

# ============================================================
# 2. NullSource Tests
# ============================================================

@testset "NullSource" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)

    u = SVector(1.0, 0.5, 2.5)  # conserved state
    w = conserved_to_primitive(law, u)

    src = NullSource()

    # evaluate_stiff_source returns zero
    S = evaluate_stiff_source(src, law, w, u)
    @test S == zero(u)
    @test S isa SVector{3, Float64}

    # stiff_source_jacobian returns zero matrix
    J = stiff_source_jacobian(src, law, w, u)
    @test J == zero(SMatrix{3, 3, Float64})
    @test J isa SMatrix{3, 3, Float64}

    # Also test with 2D Euler
    law2d = EulerEquations{2}(eos)
    u2 = SVector(1.0, 0.5, 0.3, 2.5)
    w2 = conserved_to_primitive(law2d, u2)

    S2 = evaluate_stiff_source(src, law2d, w2, u2)
    @test S2 == zero(u2)
    @test S2 isa SVector{4, Float64}

    J2 = stiff_source_jacobian(src, law2d, w2, u2)
    @test J2 == zero(SMatrix{4, 4, Float64})
    @test J2 isa SMatrix{4, 4, Float64}
end

# ============================================================
# 3. IMEX with NullSource = Pure Explicit
# ============================================================

@testset "IMEX with NullSource matches pure explicit (Sod)" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)

    N_cells = 64
    wL = SVector(1.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.1)
    t_final = 0.1
    cfl = 0.4

    mesh = StructuredMesh1D(0.0, 1.0, N_cells)

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        x -> x < 0.5 ? wL : wR;
        final_time = t_final, cfl = cfl
    )

    # Pure explicit solve
    x_ref, U_ref, t_ref = solve_hyperbolic(prob)
    W_ref = to_primitive(law, U_ref)

    # IMEX solve with NullSource (should behave like explicit)
    x_imex, U_imex, t_imex = solve_hyperbolic_imex(
        prob, NullSource();
        scheme = IMEX_Midpoint()
    )
    W_imex = to_primitive(law, U_imex)

    # Both should reach the final time
    @test t_ref ≈ t_final atol = 1.0e-10
    @test t_imex ≈ t_final atol = 1.0e-10

    # The solutions should be close (not identical due to different time
    # stepping structure, but the physics should match)
    for i in 1:N_cells
        # Density
        @test W_imex[i][1] ≈ W_ref[i][1] rtol = 0.15
        # Velocity
        @test W_imex[i][2] ≈ W_ref[i][2] atol = 0.15
        # Pressure
        @test W_imex[i][3] ≈ W_ref[i][3] rtol = 0.15
    end

    # L1 error between IMEX and explicit should be small
    dx = 1.0 / N_cells
    rho_diff = sum(abs(W_imex[i][1] - W_ref[i][1]) * dx for i in 1:N_cells)
    @test rho_diff < 0.1

    # Both should produce physical results (positive density and pressure)
    for i in 1:N_cells
        @test W_imex[i][1] > 0.0  # rho > 0
        @test W_imex[i][3] > 0.0  # P > 0
    end
end

# ============================================================
# 4. Stiff Relaxation Test with CoolingSource
# ============================================================

@testset "Stiff relaxation with CoolingSource" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)

    N_cells = 32

    # Target pressure and stiff cooling rate
    # CoolingSource uses cooling_function(T) -> Lambda
    # where T = P/rho * mu_mol, and S_E = -rho^2 * Lambda
    # We want pressure to relax toward P_target = 1.0
    # For a uniform rho=1 gas: T = P * mu_mol
    # So T_target = P_target * mu_mol
    P_target = 1.0
    mu_mol = 1.0
    T_target = P_target * mu_mol
    lambda_rate = 50.0  # stiff cooling rate

    # Cooling function: Lambda(T) = lambda_rate * (T - T_target)
    # When T > T_target, Lambda > 0 => S_E = -rho^2 * Lambda < 0 (cooling)
    # When T < T_target, Lambda < 0 => S_E = -rho^2 * Lambda > 0 (heating)
    cooling_func = T -> lambda_rate * (T - T_target)
    source = CoolingSource(cooling_func; mu_mol = mu_mol)

    # Uniform initial condition with P > P_target
    P_init = 3.0
    rho_init = 1.0
    v_init = 0.0
    w_init = SVector(rho_init, v_init, P_init)

    mesh = StructuredMesh1D(0.0, 1.0, N_cells)

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        x -> w_init;
        final_time = 0.05, cfl = 0.4
    )

    x, U_final, t_final = solve_hyperbolic_imex(
        prob, source;
        scheme = IMEX_ARS222(), newton_tol = 1.0e-12, newton_maxiter = 10
    )
    W_final = to_primitive(law, U_final)

    # After evolution, pressure should have relaxed toward P_target
    # It may not reach exactly P_target but should be closer than P_init
    for i in 1:N_cells
        P_i = W_final[i][3]
        rho_i = W_final[i][1]
        # Solution should remain physical
        @test rho_i > 0.0
        @test P_i > 0.0
        # Pressure should be closer to P_target than P_init was
        @test abs(P_i - P_target) < abs(P_init - P_target)
    end

    # Density should be essentially unchanged (cooling only affects energy)
    for i in 1:N_cells
        @test W_final[i][1] ≈ rho_init rtol = 0.05
    end

    # Velocity should remain near zero (no driving force)
    for i in 1:N_cells
        @test abs(W_final[i][2]) < 0.1
    end

    # The solution should not blow up (no instability)
    max_P = maximum(W_final[i][3] for i in 1:N_cells)
    min_P = minimum(W_final[i][3] for i in 1:N_cells)
    @test max_P < 10 * P_init
    @test min_P > 0.0
end

# ============================================================
# 5. Conservation Check with Periodic BCs
# ============================================================

@testset "Conservation with IMEX + periodic BCs + NullSource" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)

    N_cells = 64
    mesh = StructuredMesh1D(0.0, 1.0, N_cells)
    dx = mesh.dx

    # Smooth initial condition
    ic = x -> SVector(1.0 + 0.2 * sin(2 * pi * x), 0.5, 1.0)

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(), ic;
        final_time = 0.05, cfl = 0.3
    )

    # Compute initial conserved quantities
    U0 = FiniteVolumeMethod.initialize_1d(prob)
    FiniteVolumeMethod.apply_boundary_conditions!(U0, prob, 0.0)

    mass0 = sum(U0[i][1] for i in 3:(N_cells + 2)) * dx
    momentum0 = sum(U0[i][2] for i in 3:(N_cells + 2)) * dx
    energy0 = sum(U0[i][3] for i in 3:(N_cells + 2)) * dx

    # Solve with IMEX + NullSource (purely explicit, conservative)
    x, U_final, t_final = solve_hyperbolic_imex(
        prob, NullSource();
        scheme = IMEX_SSP3_433()
    )

    mass_final = sum(U_final[i][1] for i in 1:N_cells) * dx
    momentum_final = sum(U_final[i][2] for i in 1:N_cells) * dx
    energy_final = sum(U_final[i][3] for i in 1:N_cells) * dx

    # Conservation should hold (NullSource adds no energy/mass/momentum)
    @test mass_final ≈ mass0 rtol = 1.0e-10
    @test momentum_final ≈ momentum0 rtol = 1.0e-10
    @test energy_final ≈ energy0 rtol = 1.0e-10
end

# ============================================================
# 6. 2D IMEX Test
# ============================================================

@testset "2D IMEX with NullSource" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    nx, ny = 16, 16
    mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, nx, ny)

    # Smooth initial condition: uniform background with a density perturbation
    w_bg = SVector(1.0, 0.0, 0.0, 1.0)  # (rho, vx, vy, P)
    ic = (x, y) -> SVector(1.0 + 0.1 * sin(2 * pi * x) * sin(2 * pi * y), 0.0, 0.0, 1.0)

    prob = HyperbolicProblem2D(
        law, mesh, HLLSolver(), NoReconstruction(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.01, cfl = 0.3
    )

    coords, U_final, t_final = solve_hyperbolic_imex(
        prob, NullSource();
        scheme = IMEX_Midpoint()
    )

    # Should reach final time
    @test t_final ≈ 0.01 atol = 1.0e-10

    # Solution should have correct dimensions
    @test size(U_final) == (nx, ny)
    @test size(coords) == (nx, ny)

    # All states should be physical
    W_final = to_primitive(law, U_final)
    for iy in 1:ny, ix in 1:nx
        @test W_final[ix, iy][1] > 0.0  # rho > 0
        @test W_final[ix, iy][4] > 0.0  # P > 0
    end

    # Density should remain close to initial (short time, small perturbation)
    for iy in 1:ny, ix in 1:nx
        @test W_final[ix, iy][1] ≈ 1.0 atol = 0.2
    end

    # Pressure should remain near 1.0
    for iy in 1:ny, ix in 1:nx
        @test W_final[ix, iy][4] ≈ 1.0 atol = 0.2
    end
end

# ============================================================
# Additional: Scheme-specific integration tests
# ============================================================

@testset "All IMEX schemes run Sod without crash" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)
    N_cells = 32

    wL = SVector(1.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.1)

    mesh = StructuredMesh1D(0.0, 1.0, N_cells)

    prob = HyperbolicProblem(
        law, mesh, HLLSolver(), NoReconstruction(),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        x -> x < 0.5 ? wL : wR;
        final_time = 0.05, cfl = 0.3
    )

    for (name, scheme) in [
            ("SSP3_433", IMEX_SSP3_433()),
            ("ARS222", IMEX_ARS222()),
            ("Midpoint", IMEX_Midpoint()),
        ]
        @testset "$name" begin
            x, U, t = solve_hyperbolic_imex(prob, NullSource(); scheme = scheme)
            W = to_primitive(law, U)

            @test t ≈ 0.05 atol = 1.0e-10

            # Physical results
            for i in 1:N_cells
                @test W[i][1] > 0.0  # rho > 0
                @test W[i][3] > 0.0  # P > 0
            end

            # Left state should be near original left state
            @test W[1][1] ≈ 1.0 atol = 0.1
            @test W[1][3] ≈ 1.0 atol = 0.1

            # Right state should be near original right state
            @test W[N_cells][1] ≈ 0.125 atol = 0.05
            @test W[N_cells][3] ≈ 0.1 atol = 0.05
        end
    end
end

@testset "ResistiveSource basic evaluation" begin
    # Test that ResistiveSource works with MHD equations
    eos = IdealGasEOS(5.0 / 3.0)
    law = IdealMHDEquations{1}(eos)

    eta = 0.1
    src = ResistiveSource(eta)

    u = SVector(1.0, 0.5, 0.3, 0.1, 2.5, 0.8, 0.6, 0.2)
    w = conserved_to_primitive(law, u)

    S = evaluate_stiff_source(src, law, w, u)
    @test S isa SVector{8, Float64}
    # Source only acts on B components
    @test S[1] == 0.0  # no mass source
    @test S[2] == 0.0  # no momentum source
    @test S[3] == 0.0
    @test S[4] == 0.0
    @test S[5] == 0.0  # no energy source (just B)
    @test S[6] ≈ -eta * u[6]
    @test S[7] ≈ -eta * u[7]
    @test S[8] ≈ -eta * u[8]

    J = stiff_source_jacobian(src, law, w, u)
    @test J isa SMatrix{8, 8, Float64}
    # Check diagonal structure: only (6,6), (7,7), (8,8) are nonzero
    @test J[6, 6] ≈ -eta
    @test J[7, 7] ≈ -eta
    @test J[8, 8] ≈ -eta
    @test J[1, 1] == 0.0
    @test J[5, 5] == 0.0
end

@testset "CoolingSource basic evaluation" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{1}(eos)

    mu_mol = 1.0
    cooling_func = T -> 0.5 * T  # simple proportional cooling
    src = CoolingSource(cooling_func; mu_mol = mu_mol)

    # State with rho=1, v=0, P=2
    w = SVector(1.0, 0.0, 2.0)
    u = primitive_to_conserved(law, w)

    S = evaluate_stiff_source(src, law, w, u)
    @test S isa SVector{3, Float64}
    @test S[1] == 0.0  # no mass source
    @test S[2] == 0.0  # no momentum source

    # T = P/rho * mu_mol = 2.0, Lambda = 0.5 * 2.0 = 1.0
    # S_E = -rho^2 * Lambda = -1.0
    @test S[3] ≈ -1.0

    J = stiff_source_jacobian(src, law, w, u)
    @test J isa SMatrix{3, 3, Float64}
    # Only the (3,3) entry should be nonzero
    @test J[1, 1] == 0.0
    @test J[2, 2] == 0.0
    @test J[3, 3] != 0.0
end
