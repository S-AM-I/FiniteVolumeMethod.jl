using FiniteVolumeMethod
using Test
using StaticArrays
using LinearAlgebra

# ============================================================
# 1. Type and Physical Flux Tests
# ============================================================
@testset "3D MHD Type Tests" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)

    @test nvariables(law) == 8
    @test law.eos === eos

    # Different gamma
    eos2 = IdealGasEOS(gamma = 2.0)
    law2 = IdealMHDEquations{3}(eos2)
    @test law2.eos.gamma ≈ 2.0
    @test nvariables(law2) == 8
end

@testset "3D MHD Primitive <-> Conserved Roundtrip" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)

    @testset "Specific states" begin
        states = [
            SVector(1.0, 0.5, -0.3, 0.2, 2.0, 0.7, 0.8, 0.4),
            SVector(0.125, -1.0, 0.5, 0.3, 0.1, 0.75, -1.0, 0.0),
            SVector(3.0, 0.0, 0.0, 0.0, 10.0, 5.0, 3.0, 1.0),
        ]
        for w in states
            u = primitive_to_conserved(law, w)
            w2 = conserved_to_primitive(law, u)
            @test w2 ≈ w atol = 1.0e-12
        end
    end

    @testset "Random states" begin
        for _ in 1:20
            ρ = 0.1 + 3.0 * rand()
            vx = -2.0 + 4.0 * rand()
            vy = -2.0 + 4.0 * rand()
            vz = -2.0 + 4.0 * rand()
            P = 0.01 + 5.0 * rand()
            Bx = -3.0 + 6.0 * rand()
            By = -3.0 + 6.0 * rand()
            Bz = -3.0 + 6.0 * rand()
            w = SVector(ρ, vx, vy, vz, P, Bx, By, Bz)
            u = primitive_to_conserved(law, w)
            w2 = conserved_to_primitive(law, u)
            @test w2 ≈ w atol = 1.0e-12
        end
    end
end

@testset "3D MHD Physical Flux" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)
    gamma = eos.gamma

    w = SVector(1.0, 0.5, -0.3, 0.2, 2.0, 0.7, 0.8, 0.4)
    rho, vx, vy, vz, P, Bx, By, Bz = w
    v_dot_B = vx * Bx + vy * By + vz * Bz
    B_sq = Bx^2 + By^2 + Bz^2
    P_tot = P + 0.5 * B_sq
    KE = 0.5 * rho * (vx^2 + vy^2 + vz^2)
    E = P / (gamma - 1) + KE + 0.5 * B_sq

    @testset "x-flux (dir=1)" begin
        F = physical_flux(law, w, 1)
        @test F[1] ≈ rho * vx
        @test F[2] ≈ rho * vx^2 + P_tot - Bx^2
        @test F[3] ≈ rho * vx * vy - Bx * By
        @test F[4] ≈ rho * vx * vz - Bx * Bz
        @test F[5] ≈ (E + P_tot) * vx - Bx * v_dot_B
        @test F[6] ≈ 0.0  # Bx flux = 0
        @test F[7] ≈ By * vx - Bx * vy
        @test F[8] ≈ Bz * vx - Bx * vz
    end

    @testset "y-flux (dir=2)" begin
        G = physical_flux(law, w, 2)
        @test G[1] ≈ rho * vy
        @test G[2] ≈ rho * vx * vy - Bx * By
        @test G[3] ≈ rho * vy^2 + P_tot - By^2
        @test G[4] ≈ rho * vy * vz - By * Bz
        @test G[5] ≈ (E + P_tot) * vy - By * v_dot_B
        @test G[6] ≈ Bx * vy - By * vx
        @test G[7] ≈ 0.0  # By flux = 0
        @test G[8] ≈ Bz * vy - By * vz
    end

    @testset "z-flux (dir=3)" begin
        H = physical_flux(law, w, 3)
        @test H[1] ≈ rho * vz
        @test H[2] ≈ rho * vx * vz - Bx * Bz
        @test H[3] ≈ rho * vy * vz - By * Bz
        @test H[4] ≈ rho * vz^2 + P_tot - Bz^2
        @test H[5] ≈ (E + P_tot) * vz - Bz * v_dot_B
        @test H[6] ≈ Bx * vz - Bz * vx
        @test H[7] ≈ By * vz - Bz * vy
        @test H[8] ≈ 0.0  # Bz flux = 0
    end

    @testset "B=0 reduces to Euler flux" begin
        eos14 = IdealGasEOS(1.4)
        law_mhd = IdealMHDEquations{3}(eos14)
        law_euler = EulerEquations{3}(eos14)

        w_mhd = SVector(1.0, 0.5, -0.3, 0.2, 2.0, 0.0, 0.0, 0.0)
        w_euler = SVector(1.0, 0.5, -0.3, 0.2, 2.0)

        for dir in 1:3
            f_mhd = physical_flux(law_mhd, w_mhd, dir)
            f_euler = physical_flux(law_euler, w_euler, dir)
            @test f_mhd[1] ≈ f_euler[1] atol = 1.0e-14
            @test f_mhd[2] ≈ f_euler[2] atol = 1.0e-14
            @test f_mhd[3] ≈ f_euler[3] atol = 1.0e-14
            @test f_mhd[4] ≈ f_euler[4] atol = 1.0e-14
            @test f_mhd[5] ≈ f_euler[5] atol = 1.0e-14
        end
    end

    @testset "Flux consistency through roundtrip" begin
        for dir in 1:3
            w_test = SVector(2.0, 1.0, -0.5, 0.3, 3.0, 0.5, -0.7, 0.9)
            f1 = physical_flux(law, w_test, dir)
            u = primitive_to_conserved(law, w_test)
            w2 = conserved_to_primitive(law, u)
            f2 = physical_flux(law, w2, dir)
            @test f1 ≈ f2 atol = 1.0e-14
        end
    end
end

@testset "3D MHD Wave Speeds" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)
    gamma = eos.gamma

    w = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.5)

    @testset "Fast/slow ordering for all directions" begin
        for dir in 1:3
            cf = fast_magnetosonic_speed(law, w, dir)
            cs = slow_magnetosonic_speed(law, w, dir)
            @test cf > 0
            @test cs >= 0
            @test cf >= cs

            # Alfven speed for this direction
            Bn = dir == 1 ? w[6] : (dir == 2 ? w[7] : w[8])
            ca = abs(Bn) / sqrt(w[1])
            @test cf >= ca - 1.0e-14
            @test ca >= cs - 1.0e-14
        end
    end

    @testset "wave_speeds consistency" begin
        w2 = SVector(1.0, 0.5, -0.3, 0.2, 1.0, 0.75, 1.0, 0.5)
        for dir in 1:3
            lam_min, lam_max = wave_speeds(law, w2, dir)
            cf = fast_magnetosonic_speed(law, w2, dir)
            vn = dir == 1 ? w2[2] : (dir == 2 ? w2[3] : w2[4])
            @test lam_min ≈ vn - cf
            @test lam_max ≈ vn + cf
        end
    end

    @testset "B=0 reduces to sound speed" begin
        w_hydro = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        cs_hydro = sqrt(gamma * 1.0 / 1.0)
        for dir in 1:3
            cf = fast_magnetosonic_speed(law, w_hydro, dir)
            @test cf ≈ cs_hydro atol = 1.0e-14

            cs = slow_magnetosonic_speed(law, w_hydro, dir)
            @test cs ≈ 0.0 atol = 1.0e-14
        end
    end

    @testset "Symmetric speeds for v=0" begin
        w_sym = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.3, 0.2)
        for dir in 1:3
            lam_min, lam_max = wave_speeds(law, w_sym, dir)
            @test lam_min ≈ -lam_max atol = 1.0e-14
        end
    end
end

# ============================================================
# 2. 3D Brio-Wu Along x-axis
# ============================================================
@testset "3D Brio-Wu x-axis" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{3}(eos)

    Bx_val = 0.75
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx_val, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, Bx_val, -1.0, 0.0)
    bw_ic(x, y, z) = x < 0.5 ? wL : wR

    nx, ny, nz = 40, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)
    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        bw_ic; final_time = 0.05, cfl = 0.3
    )

    coords, U_final, t_final, ct = solve_hyperbolic(prob)
    W = to_primitive(law, U_final)

    @test t_final ≈ 0.05 atol = 1.0e-10
    @test size(U_final) == (nx, ny, nz)

    # Extract density along x at iy=1, iz=1
    rho_x = [W[ix, 1, 1][1] for ix in 1:nx]
    P_x = [W[ix, 1, 1][5] for ix in 1:nx]

    @test all(rho_x .> 0)
    @test all(P_x .> 0)
    @test all(isfinite.(rho_x))
    @test all(isfinite.(P_x))

    # Should have structure in x (not uniform)
    @test maximum(rho_x) > minimum(rho_x) + 0.01

    # Solution should be approximately uniform in y and z
    for ix in 1:nx, iy in 1:ny, iz in 1:nz
        @test W[ix, iy, iz][1] ≈ W[ix, 1, 1][1] atol = 1.0e-8
    end

    # Bx should remain approximately constant
    Bx_sol = [W[ix, 1, 1][6] for ix in 1:nx]
    @test all(b -> abs(b - Bx_val) < 0.1, Bx_sol)
end

# ============================================================
# 2b. 3D Brio-Wu Along y-axis
# ============================================================
@testset "3D Brio-Wu y-axis" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{3}(eos)

    By_val = 0.75
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, By_val, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, -1.0, By_val, 0.0)
    bw_y_ic(x, y, z) = y < 0.5 ? wL : wR

    nx, ny, nz = 4, 40, 4
    mesh = StructuredMesh3D(0.0, 0.1, 0.0, 1.0, 0.0, 0.1, nx, ny, nz)
    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        bw_y_ic; final_time = 0.05, cfl = 0.3
    )

    coords, U_final, t_final, ct = solve_hyperbolic(prob)
    W = to_primitive(law, U_final)

    @test t_final ≈ 0.05 atol = 1.0e-10

    # Extract density along y at ix=1, iz=1
    rho_y = [W[1, iy, 1][1] for iy in 1:ny]
    P_y = [W[1, iy, 1][5] for iy in 1:ny]

    @test all(rho_y .> 0)
    @test all(P_y .> 0)

    # Should have structure in y (not uniform)
    @test maximum(rho_y) > minimum(rho_y) + 0.01

    # Uniform in x and z
    for ix in 1:nx, iy in 1:ny, iz in 1:nz
        @test W[ix, iy, iz][1] ≈ W[1, iy, 1][1] atol = 1.0e-8
    end
end

# ============================================================
# 3. 3D Brio-Wu Along z-axis
# ============================================================
@testset "3D Brio-Wu z-axis" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{3}(eos)

    # Rotate the Brio-Wu problem to z-direction:
    # Normal B along z (Bz = const), By reverses
    Bz_val = 0.75
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, Bz_val)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.0, -1.0, Bz_val)
    bw_z_ic(x, y, z) = z < 0.5 ? wL : wR

    nx, ny, nz = 4, 4, 40
    mesh = StructuredMesh3D(0.0, 0.1, 0.0, 0.1, 0.0, 1.0, nx, ny, nz)
    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        bw_z_ic; final_time = 0.05, cfl = 0.3
    )

    coords, U_final, t_final, ct = solve_hyperbolic(prob)
    W = to_primitive(law, U_final)

    @test t_final ≈ 0.05 atol = 1.0e-10

    # Extract density along z at ix=1, iy=1
    rho_z = [W[1, 1, iz][1] for iz in 1:nz]
    P_z = [W[1, 1, iz][5] for iz in 1:nz]

    @test all(rho_z .> 0)
    @test all(P_z .> 0)
    @test all(isfinite.(rho_z))

    # Should have structure in z (not uniform)
    @test maximum(rho_z) > minimum(rho_z) + 0.01

    # Solution should be approximately uniform in x and y
    for ix in 1:nx, iy in 1:ny, iz in 1:nz
        @test W[ix, iy, iz][1] ≈ W[1, 1, iz][1] atol = 1.0e-8
    end
end

# ============================================================
# 3b. 3D Brio-Wu with MUSCL Reconstruction
# ============================================================
@testset "3D Brio-Wu MUSCL" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{3}(eos)

    Bx_val = 0.75
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx_val, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, Bx_val, -1.0, 0.0)
    bw_ic(x, y, z) = x < 0.5 ? wL : wR

    limiters = [
        ("Minmod", MinmodLimiter()),
        ("Superbee", SuperbeeLimiter()),
        ("VanLeer", VanLeerLimiter()),
        ("Koren", KorenLimiter()),
        ("Ospre", OspreLimiter()),
    ]

    @testset "$name" for (name, lim) in limiters
        nx, ny, nz = 30, 4, 4
        mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)
        prob = HyperbolicProblem3D(
            law, mesh, HLLSolver(), CellCenteredMUSCL(lim),
            TransmissiveBC(), TransmissiveBC(),
            TransmissiveBC(), TransmissiveBC(),
            TransmissiveBC(), TransmissiveBC(),
            bw_ic; final_time = 0.05, cfl = 0.3
        )

        coords, U_final, t_final, ct = solve_hyperbolic(prob)
        W = to_primitive(law, U_final)

        # Positivity
        for iz in 1:nz, iy in 1:ny, ix in 1:nx
            @test W[ix, iy, iz][1] > 0
            @test W[ix, iy, iz][5] > 0
        end

        # Shock structure
        @test W[1, 1, 1][1] > W[nx, 1, 1][1]

        # divB
        divB = max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz)
        @test divB < 1.0e-12
    end
end

# ============================================================
# 4. CT Divergence-Free Tests
# ============================================================
@testset "3D CT Divergence-Free" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)

    @testset "Vector potential initialization" begin
        # Simple vector potential: A = (0, 0, x*y) => Bx = dAz/dy = x, By = -dAz/dx = -y, Bz = 0
        # More generally: A = (sin(z), sin(x), sin(y)) gives a divergence-free B
        Ax_func(x, y, z) = sin(2 * pi * z)
        Ay_func(x, y, z) = sin(2 * pi * x)
        Az_func(x, y, z) = sin(2 * pi * y)

        nx, ny, nz = 8, 8, 8
        mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz)
        ct = CTData3D(nx, ny, nz)
        initialize_ct_3d_from_potential!(ct, Ax_func, Ay_func, Az_func, mesh)

        divB = max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz)
        @test divB < 1.0e-14
    end

    @testset "DivB preserved after evolution" begin
        # Uniform state with non-trivial B field initialized via vector potential
        Ax_func(x, y, z) = 0.1 * cos(2 * pi * z)
        Ay_func(x, y, z) = 0.1 * cos(2 * pi * x)
        Az_func(x, y, z) = 0.1 * cos(2 * pi * y)

        function smooth_ic(x, y, z)
            rho = 1.0
            vx = 0.1
            vy = 0.1
            vz = 0.1
            P = 1.0
            # These will be overwritten by CT face_to_cell_B
            Bx = 0.0
            By = 0.0
            Bz = 0.0
            return SVector(rho, vx, vy, vz, P, Bx, By, Bz)
        end

        nx, ny, nz = 8, 8, 8
        mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz)
        prob = HyperbolicProblem3D(
            law, mesh, HLLSolver(), NoReconstruction(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            smooth_ic; final_time = 0.02, cfl = 0.3
        )

        coords, U_final, t_final, ct = solve_hyperbolic(
            prob;
            vector_potential_x = Ax_func,
            vector_potential_y = Ay_func,
            vector_potential_z = Az_func
        )

        divB = max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz)
        @test divB < 1.0e-14
        @test t_final ≈ 0.02 atol = 1.0e-10
    end

    @testset "Uniform field: divB = 0" begin
        nx, ny, nz = 8, 8, 8
        mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz)
        ic(x, y, z) = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.3)

        prob = HyperbolicProblem3D(
            law, mesh, HLLSolver(), NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(),
            TransmissiveBC(), TransmissiveBC(),
            TransmissiveBC(), TransmissiveBC(),
            ic; final_time = 0.01, cfl = 0.3
        )

        coords, U_final, t_final, ct = solve_hyperbolic(prob)
        divB = max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz)
        @test divB < 1.0e-12
    end

    @testset "Euler method: divB = 0" begin
        nx, ny, nz = 8, 8, 8
        mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz)
        ic(x, y, z) = SVector(1.0, 0.3, -0.2, 0.1, 1.0, 0.5, 0.8, 0.3)

        prob = HyperbolicProblem3D(
            law, mesh, HLLSolver(), NoReconstruction(),
            TransmissiveBC(), TransmissiveBC(),
            TransmissiveBC(), TransmissiveBC(),
            TransmissiveBC(), TransmissiveBC(),
            ic; final_time = 0.01, cfl = 0.2
        )

        _, _, _, ct = solve_hyperbolic(prob; method = :euler)
        divB = max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz)
        @test divB < 1.0e-12
    end

    @testset "DivB with MUSCL reconstruction" begin
        nx, ny, nz = 8, 8, 8
        mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz)
        ic(x, y, z) = SVector(
            1.0 + 0.1 * sin(2π * x),
            0.1, -0.1, 0.05, 1.0, 0.5, 0.3, 0.2
        )

        prob = HyperbolicProblem3D(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            ic; final_time = 0.01, cfl = 0.3
        )

        _, _, _, ct = solve_hyperbolic(prob)
        divB = max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz)
        @test divB < 1.0e-12
    end
end

# ============================================================
# 5. Field Loop Advection 3D
# ============================================================
@testset "3D Field Loop Advection" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)

    # A cylindrical field loop aligned with the z-axis,
    # advected by a uniform flow. Uses vector potential for initialization.
    vx_bg = 1.0
    vy_bg = 0.5
    vz_bg = 0.0
    rho_bg = 1.0
    P_bg = 1.0
    R0 = 0.3
    A0 = 1.0e-3

    function loop_ic_3d(x, y, z)
        # B will be overwritten by CT, just set hydrodynamic part
        return SVector(rho_bg, vx_bg, vy_bg, vz_bg, P_bg, 0.0, 0.0, 0.0)
    end

    # Vector potential for a z-aligned field loop centered at (0.5, 0.5, _):
    # Az = A0 * max(R0 - r, 0) where r = sqrt((x-0.5)^2 + (y-0.5)^2)
    # Ax = Ay = 0
    # => Bx = dAz/dy - dAy/dz = dAz/dy, By = dAx/dz - dAz/dx = -dAz/dx, Bz = 0
    Ax_loop(x, y, z) = 0.0
    Ay_loop(x, y, z) = 0.0
    function Az_loop(x, y, z)
        r = sqrt((x - 0.5)^2 + (y - 0.5)^2)
        return r < R0 ? A0 * (R0 - r) : 0.0
    end

    @testset "DivB preserved" begin
        nx, ny, nz = 8, 8, 4
        mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 0.5, nx, ny, nz)
        prob = HyperbolicProblem3D(
            law, mesh, HLLSolver(), NoReconstruction(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            loop_ic_3d; final_time = 0.02, cfl = 0.3
        )

        coords, U_final, t_final, ct = solve_hyperbolic(
            prob;
            vector_potential_x = Ax_loop,
            vector_potential_y = Ay_loop,
            vector_potential_z = Az_loop
        )

        divB = max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz)
        @test divB < 1.0e-14
        @test t_final ≈ 0.02 atol = 1.0e-10
    end

    @testset "Density and pressure remain positive and nearly uniform" begin
        nx, ny, nz = 8, 8, 4
        mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 0.5, nx, ny, nz)
        prob = HyperbolicProblem3D(
            law, mesh, HLLSolver(), NoReconstruction(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            loop_ic_3d; final_time = 0.02, cfl = 0.3
        )

        coords, U_final, t_final, ct = solve_hyperbolic(
            prob;
            vector_potential_x = Ax_loop,
            vector_potential_y = Ay_loop,
            vector_potential_z = Az_loop
        )

        W = to_primitive(law, U_final)
        rho_vals = [w[1] for w in W]
        P_vals = [w[5] for w in W]

        @test all(rho_vals .> 0)
        @test all(P_vals .> 0)
        # Weak field: density and pressure should remain near-uniform
        @test maximum(rho_vals) - minimum(rho_vals) < 0.01
        @test maximum(P_vals) - minimum(P_vals) < 0.01
    end
end

# ============================================================
# 6. Conservation with Periodic BCs
# ============================================================
@testset "3D MHD Conservation" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)

    function smooth_ic_3d(x, y, z)
        rho = 1.0 + 0.1 * sin(2 * pi * x) * cos(2 * pi * y) * cos(2 * pi * z)
        vx = 0.05 * sin(2 * pi * y)
        vy = -0.05 * sin(2 * pi * x)
        vz = 0.05 * sin(2 * pi * z)
        P = 1.0
        Bx = 0.5
        By = 0.3
        Bz = 0.2
        return SVector(rho, vx, vy, vz, P, Bx, By, Bz)
    end

    nx, ny, nz = 8, 8, 8
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz)
    dV = mesh.dx * mesh.dy * mesh.dz

    # Initial state (zero final_time to get IC)
    prob0 = HyperbolicProblem3D(
        law, mesh, HLLSolver(), NoReconstruction(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        smooth_ic_3d; final_time = 0.0, cfl = 0.3
    )
    _, U0, _, _ = solve_hyperbolic(prob0)

    # Evolved state
    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), NoReconstruction(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        smooth_ic_3d; final_time = 0.02, cfl = 0.3
    )
    _, U_final, t_final, ct = solve_hyperbolic(prob)

    @test t_final ≈ 0.02 atol = 1.0e-10

    # Compute conserved totals
    mass_0 = sum(u[1] for u in U0) * dV
    mass_f = sum(u[1] for u in U_final) * dV
    momx_0 = sum(u[2] for u in U0) * dV
    momx_f = sum(u[2] for u in U_final) * dV
    momy_0 = sum(u[3] for u in U0) * dV
    momy_f = sum(u[3] for u in U_final) * dV
    momz_0 = sum(u[4] for u in U0) * dV
    momz_f = sum(u[4] for u in U_final) * dV
    energy_0 = sum(u[5] for u in U0) * dV
    energy_f = sum(u[5] for u in U_final) * dV

    @test mass_f ≈ mass_0 atol = 1.0e-10
    @test momx_f ≈ momx_0 atol = 1.0e-10
    @test momy_f ≈ momy_0 atol = 1.0e-10
    @test momz_f ≈ momz_0 atol = 1.0e-10
    @test energy_f ≈ energy_0 atol = 1.0e-8

    # divB should also be preserved
    divB = max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz)
    @test divB < 1.0e-12
end

# ============================================================
# 6b. Conservation with MUSCL + Forward Euler
# ============================================================
@testset "3D MHD Conservation MUSCL + Euler" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)

    ic(x, y, z) = SVector(
        1.0 + 0.1 * sin(2π * x) * sin(2π * y),
        0.1, -0.05, 0.05, 1.0, 0.3, 0.5, 0.2
    )

    nx, ny, nz = 8, 8, 8
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz)
    dV = mesh.dx * mesh.dy * mesh.dz

    prob0 = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(VanLeerLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.0, cfl = 0.2
    )
    _, U0, _, _ = solve_hyperbolic(prob0)

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(VanLeerLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.01, cfl = 0.2
    )
    _, U_final, _, ct = solve_hyperbolic(prob; method = :euler)

    mass_0 = sum(u[1] for u in U0) * dV
    mass_f = sum(u[1] for u in U_final) * dV
    @test mass_f ≈ mass_0 atol = 1.0e-10

    divB = max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz)
    @test divB < 1.0e-12
end

# ============================================================
# CTData3D Structure Tests
# ============================================================
@testset "CTData3D" begin
    @testset "Construction" begin
        ct = CTData3D(10, 8, 6)
        @test size(ct.Bx_face) == (11, 8, 6)
        @test size(ct.By_face) == (10, 9, 6)
        @test size(ct.Bz_face) == (10, 8, 7)
        @test size(ct.emf_x) == (10, 9, 7)
        @test size(ct.emf_y) == (11, 8, 7)
        @test size(ct.emf_z) == (11, 9, 6)
    end

    @testset "copy_ct and copyto_ct!" begin
        ct1 = CTData3D(5, 5, 5)
        ct1.Bx_face .= 1.0
        ct1.By_face .= 2.0
        ct1.Bz_face .= 3.0

        ct2 = copy_ct(ct1)
        @test ct2.Bx_face == ct1.Bx_face
        @test ct2.By_face == ct1.By_face
        @test ct2.Bz_face == ct1.Bz_face

        ct3 = CTData3D(5, 5, 5)
        copyto_ct!(ct3, ct1)
        @test ct3.Bx_face == ct1.Bx_face
        @test ct3.By_face == ct1.By_face
        @test ct3.Bz_face == ct1.Bz_face
    end

    @testset "divB diagnostics" begin
        nx, ny, nz = 10, 10, 10
        ct = CTData3D(nx, ny, nz)
        dx, dy, dz = 0.1, 0.1, 0.1

        # Uniform Bx, zero By, Bz -> divB = 0
        ct.Bx_face .= 1.0
        ct.By_face .= 0.0
        ct.Bz_face .= 0.0
        @test max_divB_3d(ct, dx, dy, dz, nx, ny, nz) ≈ 0.0 atol = 1.0e-15

        # Introduce non-zero divergence
        ct.Bx_face[5, 5, 5] = 2.0
        divB_val = max_divB_3d(ct, dx, dy, dz, nx, ny, nz)
        @test divB_val > 0

        # compute_divB_3d returns the right shape
        divB_arr = compute_divB_3d(ct, dx, dy, dz, nx, ny, nz)
        @test size(divB_arr) == (nx, ny, nz)
    end
end

# ============================================================
# 3D MHD with Multiple Riemann Solvers
# ============================================================
@testset "3D MHD All Solvers" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)

    ic(x, y, z) = SVector(1.0, 0.3, -0.2, 0.1, 1.0, 0.5, 0.8, 0.3)

    solvers = [
        ("HLL", HLLSolver()),
        ("Lax-Friedrichs", LaxFriedrichsSolver()),
        ("HLLD", HLLDSolver()),
    ]

    for (name, solver) in solvers
        @testset "$name" begin
            nx, ny, nz = 8, 8, 8
            mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz)
            prob = HyperbolicProblem3D(
                law, mesh, solver, NoReconstruction(),
                TransmissiveBC(), TransmissiveBC(),
                TransmissiveBC(), TransmissiveBC(),
                TransmissiveBC(), TransmissiveBC(),
                ic; final_time = 0.01, cfl = 0.2
            )

            _, U, t, ct = solve_hyperbolic(prob)
            W = to_primitive(law, U)
            @test all(w -> w[1] > 0, W)
            @test all(w -> w[5] > 0, W)
            @test all(w -> all(isfinite, w), W)
            @test max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz) < 1.0e-12
        end
    end
end

# ============================================================
# Uniform State Stays Uniform
# ============================================================
@testset "3D MHD Uniform State" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)

    nx, ny, nz = 8, 8, 8
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz)
    w_const = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    ic(x, y, z) = w_const

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.02, cfl = 0.3
    )

    coords, U_final, t_final, ct = solve_hyperbolic(prob)
    u_ref = primitive_to_conserved(law, w_const)

    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test U_final[ix, iy, iz] ≈ u_ref atol = 1.0e-12
    end
    @test max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz) < 1.0e-13
end

# ============================================================
# 3D MHD Uniform Flow Preservation (with B)
# ============================================================
@testset "3D MHD Uniform Flow Preservation" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)

    nx, ny, nz = 8, 8, 8
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz)
    # Moving uniform state with non-trivial B
    w_uniform = SVector(1.0, 0.5, -0.3, 0.2, 1.0, 0.5, 0.3, 0.1)
    ic(x, y, z) = w_uniform

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.05, cfl = 0.3
    )

    coords, U_final, t_final, ct = solve_hyperbolic(prob)
    W = to_primitive(law, U_final)
    u_ref = primitive_to_conserved(law, w_uniform)

    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test U_final[ix, iy, iz] ≈ u_ref atol = 1.0e-10
    end
    @test max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz) < 1.0e-13
end

# ============================================================
# 3D MHD Reflective BC Symmetry
# ============================================================
@testset "3D MHD Reflective BC Symmetry" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)

    N = 8
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, N, N, N)

    # Symmetric MHD pressure pulse at center with B=0
    # (Non-zero uniform B breaks reflective BC symmetry since Bn flips)
    ic = function (x, y, z)
        r = sqrt((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2)
        P = r < 0.2 ? 5.0 : 1.0
        return SVector(1.0, 0.0, 0.0, 0.0, P, 0.0, 0.0, 0.0)
    end

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), NoReconstruction(),
        ReflectiveBC(), ReflectiveBC(),
        ReflectiveBC(), ReflectiveBC(),
        ReflectiveBC(), ReflectiveBC(),
        ic; final_time = 0.02, cfl = 0.3
    )

    coords, U_final, t_final, ct = solve_hyperbolic(prob)
    W = to_primitive(law, U_final)

    # Positivity
    for iz in 1:N, iy in 1:N, ix in 1:N
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    # x-symmetry for density (B=0 → pure hydro symmetry)
    for iz in 1:N, iy in 1:N, ix in 1:div(N, 2)
        @test W[ix, iy, iz][1] ≈ W[N + 1 - ix, iy, iz][1] rtol = 1.0e-8
    end
end

# ============================================================
# 3D MHD Blast Wave
# ============================================================
@testset "3D MHD Blast Wave" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)

    N = 10
    mesh = StructuredMesh3D(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, N, N, N)
    dV = mesh.dx * mesh.dy * mesh.dz

    ic = function (x, y, z)
        r = sqrt(x^2 + y^2 + z^2)
        P = r < 0.3 ? 10.0 : 0.1
        # Uniform B field
        return SVector(1.0, 0.0, 0.0, 0.0, P, 1.0, 0.0, 0.0)
    end

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.05, cfl = 0.3
    )

    coords, U_final, t_final, ct = solve_hyperbolic(prob)
    W = to_primitive(law, U_final)

    # Positivity
    for iz in 1:N, iy in 1:N, ix in 1:N
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    # divB preserved
    divB = max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, N, N, N)
    @test divB < 1.0e-12

    # Max pressure should be near center
    P_max, idx_max = -Inf, (1, 1, 1)
    for iz in 1:N, iy in 1:N, ix in 1:N
        if W[ix, iy, iz][5] > P_max
            P_max = W[ix, iy, iz][5]
            idx_max = (ix, iy, iz)
        end
    end
    center_lo = div(N, 2)
    center_hi = div(N, 2) + 1
    @test center_lo <= idx_max[1] <= center_hi
end

# ============================================================
# 3D MHD with HLLD Solver - Brio-Wu
# ============================================================
@testset "3D MHD HLLD Brio-Wu" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{3}(eos)

    Bx_val = 0.75
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx_val, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, Bx_val, -1.0, 0.0)
    bw_ic(x, y, z) = x < 0.5 ? wL : wR

    nx, ny, nz = 30, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    prob = HyperbolicProblem3D(
        law, mesh, HLLDSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        bw_ic; final_time = 0.05, cfl = 0.3
    )

    coords, U_final, t_final, ct = solve_hyperbolic(prob)
    W = to_primitive(law, U_final)

    # Positivity
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    # Shock structure
    @test W[1, 1, 1][1] > W[nx, 1, 1][1]

    # divB
    divB = max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz)
    @test divB < 1.0e-12
end

# ============================================================
# 3D MHD HLLD vs HLL Sharpness
# ============================================================
@testset "3D MHD HLLD vs HLL" begin
    eos = IdealGasEOS(gamma = 2.0)
    law = IdealMHDEquations{3}(eos)

    Bx_val = 0.75
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx_val, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, Bx_val, -1.0, 0.0)
    bw_ic(x, y, z) = x < 0.5 ? wL : wR

    nx, ny, nz = 40, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)
    recon = NoReconstruction()

    prob_hlld = HyperbolicProblem3D(
        law, mesh, HLLDSolver(), recon,
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        bw_ic; final_time = 0.1, cfl = 0.3
    )
    prob_hll = HyperbolicProblem3D(
        law, mesh, HLLSolver(), recon,
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        bw_ic; final_time = 0.1, cfl = 0.3
    )

    _, U_hlld, _, _ = solve_hyperbolic(prob_hlld)
    _, U_hll, _, _ = solve_hyperbolic(prob_hll)
    W_hlld = to_primitive(law, U_hlld)
    W_hll = to_primitive(law, U_hll)

    # HLLD should have sharper features (higher max gradient)
    max_grad_hlld = maximum(abs(W_hlld[ix + 1, 1, 1][1] - W_hlld[ix, 1, 1][1]) for ix in 1:(nx - 1))
    max_grad_hll = maximum(abs(W_hll[ix + 1, 1, 1][1] - W_hll[ix, 1, 1][1]) for ix in 1:(nx - 1))
    @test max_grad_hlld >= max_grad_hll * 0.95
end

# ============================================================
# 3D MHD Different Gamma (gamma=1.4)
# ============================================================
@testset "3D MHD gamma=1.4" begin
    eos = IdealGasEOS(1.4)
    law = IdealMHDEquations{3}(eos)

    ic(x, y, z) = SVector(
        1.0 + 0.1 * sin(2π * x),
        0.1, 0.0, 0.0, 1.0, 0.5, 0.3, 0.0
    )

    nx, ny, nz = 8, 8, 8
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz)

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.02, cfl = 0.3
    )

    coords, U_final, t_final, ct = solve_hyperbolic(prob)
    W = to_primitive(law, U_final)

    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    divB = max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz)
    @test divB < 1.0e-12
end

# ============================================================
# 3D MHD Forward Euler
# ============================================================
@testset "3D MHD Forward Euler" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)

    Bx_val = 0.75
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx_val, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, Bx_val, -1.0, 0.0)
    bw_ic(x, y, z) = x < 0.5 ? wL : wR

    nx, ny, nz = 20, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        bw_ic; final_time = 0.05, cfl = 0.2
    )

    coords, U_final, t_final, ct = solve_hyperbolic(prob; method = :euler)

    @test t_final ≈ 0.05 atol = 1.0e-10

    W = to_primitive(law, U_final)
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    # Shock direction
    @test W[1, 1, 1][1] > W[nx, 1, 1][1]

    # divB
    divB = max_divB_3d(ct, mesh.dx, mesh.dy, mesh.dz, nx, ny, nz)
    @test divB < 1.0e-12
end

# ============================================================
# 3D MHD Strong Shock
# ============================================================
@testset "3D MHD Strong Shock" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    law = IdealMHDEquations{3}(eos)

    Bx_val = 1.0
    wL = SVector(1.0, 0.0, 0.0, 0.0, 100.0, Bx_val, 2.0, 0.0)
    wR = SVector(1.0, 0.0, 0.0, 0.0, 1.0, Bx_val, -2.0, 0.0)
    ic(x, y, z) = x < 0.5 ? wL : wR

    nx, ny, nz = 30, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.01, cfl = 0.2
    )

    coords, U_final, t_final, ct = solve_hyperbolic(prob)
    W = to_primitive(law, U_final)

    # Positivity
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    # Left pressure > right
    @test W[1, 1, 1][5] > W[nx, 1, 1][5]
end

# ============================================================
# 3D MHD B=0 Matches Euler
# ============================================================
@testset "3D MHD B=0 Matches Euler" begin
    eos = IdealGasEOS(1.4)
    law_mhd = IdealMHDEquations{3}(eos)
    law_euler = EulerEquations{3}(eos)

    wL_mhd = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    wR_mhd = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0)
    wL_euler = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
    wR_euler = SVector(0.125, 0.0, 0.0, 0.0, 0.1)

    nx, ny, nz = 20, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    prob_mhd = HyperbolicProblem3D(
        law_mhd, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        (x, y, z) -> x < 0.5 ? wL_mhd : wR_mhd;
        final_time = 0.1, cfl = 0.3
    )
    prob_euler = HyperbolicProblem3D(
        law_euler, mesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        (x, y, z) -> x < 0.5 ? wL_euler : wR_euler;
        final_time = 0.1, cfl = 0.3
    )

    _, U_mhd, _, _ = solve_hyperbolic(prob_mhd)
    _, U_euler, _ = solve_hyperbolic(prob_euler)

    W_mhd = to_primitive(law_mhd, U_mhd)
    W_euler = to_primitive(law_euler, U_euler)

    # Hydrodynamic variables should match
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test W_mhd[ix, iy, iz][1] ≈ W_euler[ix, iy, iz][1] atol = 1.0e-10  # ρ
        @test W_mhd[ix, iy, iz][2] ≈ W_euler[ix, iy, iz][2] atol = 1.0e-10  # vx
        @test W_mhd[ix, iy, iz][5] ≈ W_euler[ix, iy, iz][5] atol = 1.0e-10  # P
    end
end
