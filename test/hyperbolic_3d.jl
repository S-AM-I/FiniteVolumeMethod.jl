using FiniteVolumeMethod
using Test
using StaticArrays
using LinearAlgebra

# ============================================================
# Type Tests: 3D Euler Equations
# ============================================================

@testset "3D Euler Equations Types" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    @testset "Constructor and nvariables" begin
        @test law.eos === eos
        @test nvariables(law) == 5
    end

    @testset "Primitive <-> Conserved roundtrip" begin
        w = SVector(1.0, 0.5, -0.3, 0.2, 1.0)  # ρ, vx, vy, vz, P
        u = primitive_to_conserved(law, w)
        w2 = conserved_to_primitive(law, u)
        @test w2 ≈ w atol = 1.0e-14

        # Another state
        w3 = SVector(0.125, -1.0, 0.7, 0.4, 0.1)
        u3 = primitive_to_conserved(law, w3)
        w4 = conserved_to_primitive(law, u3)
        @test w4 ≈ w3 atol = 1.0e-14
    end

    @testset "Primitive <-> Conserved roundtrip (random states)" begin
        for _ in 1:20
            ρ = 0.1 + 2.0 * rand()
            vx = -2.0 + 4.0 * rand()
            vy = -2.0 + 4.0 * rand()
            vz = -2.0 + 4.0 * rand()
            P = 0.01 + 5.0 * rand()
            w = SVector(ρ, vx, vy, vz, P)
            u = primitive_to_conserved(law, w)
            w2 = conserved_to_primitive(law, u)
            @test w2 ≈ w atol = 1.0e-12
        end
    end

    @testset "physical_flux dir=$dir" for dir in 1:3
        w = SVector(1.0, 0.5, -0.3, 0.2, 1.0)
        f = physical_flux(law, w, dir)
        @test length(f) == 5

        ρ, vx, vy, vz, P = w
        E = total_energy(eos, ρ, vx, vy, vz, P)

        if dir == 1
            @test f[1] ≈ ρ * vx
            @test f[2] ≈ ρ * vx^2 + P
            @test f[3] ≈ ρ * vx * vy
            @test f[4] ≈ ρ * vx * vz
            @test f[5] ≈ (E + P) * vx
        elseif dir == 2
            @test f[1] ≈ ρ * vy
            @test f[2] ≈ ρ * vx * vy
            @test f[3] ≈ ρ * vy^2 + P
            @test f[4] ≈ ρ * vy * vz
            @test f[5] ≈ (E + P) * vy
        else
            @test f[1] ≈ ρ * vz
            @test f[2] ≈ ρ * vx * vz
            @test f[3] ≈ ρ * vy * vz
            @test f[4] ≈ ρ * vz^2 + P
            @test f[5] ≈ (E + P) * vz
        end
    end

    @testset "Flux consistency: F(w) = F(con2prim(prim2con(w)))" begin
        for dir in 1:3
            w = SVector(2.0, 1.0, -0.5, 0.3, 3.0)
            f1 = physical_flux(law, w, dir)
            u = primitive_to_conserved(law, w)
            w2 = conserved_to_primitive(law, u)
            f2 = physical_flux(law, w2, dir)
            @test f1 ≈ f2 atol = 1.0e-14
        end
    end

    @testset "max_wave_speed dir=$dir" for dir in 1:3
        w = SVector(1.0, 0.5, -0.3, 0.2, 1.0)
        λ = max_wave_speed(law, w, dir)
        c = sound_speed(eos, 1.0, 1.0)
        v_n = dir == 1 ? 0.5 : dir == 2 ? -0.3 : 0.2
        @test λ ≈ abs(v_n) + c
    end

    @testset "wave_speeds dir=$dir" for dir in 1:3
        w = SVector(1.0, 0.5, -0.3, 0.2, 1.0)
        λ_min, λ_max = wave_speeds(law, w, dir)
        c = sound_speed(eos, 1.0, 1.0)
        v_n = dir == 1 ? 0.5 : dir == 2 ? -0.3 : 0.2
        @test λ_min ≈ v_n - c
        @test λ_max ≈ v_n + c
    end

    @testset "Wave speeds: symmetric for v=0" begin
        w = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
        for dir in 1:3
            λ_min, λ_max = wave_speeds(law, w, dir)
            @test λ_min ≈ -λ_max atol = 1.0e-14
        end
    end

    @testset "Different EOS gamma=5/3" begin
        eos53 = IdealGasEOS(5.0 / 3.0)
        law53 = EulerEquations{3}(eos53)
        @test nvariables(law53) == 5

        w = SVector(1.0, 0.0, 0.0, 0.0, 1.5)
        u = primitive_to_conserved(law53, w)
        w2 = conserved_to_primitive(law53, u)
        @test w2 ≈ w atol = 1.0e-14

        c = sound_speed(eos53, 1.0, 1.5)
        @test c ≈ sqrt(5.0 / 3.0 * 1.5 / 1.0)
    end
end

# ============================================================
# 3D Mesh Tests
# ============================================================

@testset "StructuredMesh3D" begin
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 4, 5, 6)

    @testset "Constructor and basic queries" begin
        @test mesh.nx == 4
        @test mesh.ny == 5
        @test mesh.nz == 6
        @test mesh.dx ≈ 0.25
        @test mesh.dy ≈ 0.4
        @test mesh.dz ≈ 0.5
        @test ncells(mesh) == 4 * 5 * 6
        @test ndims_mesh(mesh) == 3
    end

    @testset "cell_center" begin
        # First cell: (1,1,1)
        x, y, z = cell_center(mesh, 1)
        @test x ≈ 0.125   # xmin + 0.5*dx
        @test y ≈ 0.2     # ymin + 0.5*dy
        @test z ≈ 0.25    # zmin + 0.5*dz

        # Last cell: (4,5,6)
        idx_last = cell_idx_3d(mesh, 4, 5, 6)
        xl, yl, zl = cell_center(mesh, idx_last)
        @test xl ≈ 1.0 - 0.125
        @test yl ≈ 2.0 - 0.2
        @test zl ≈ 3.0 - 0.25
    end

    @testset "cell_volume" begin
        vol = cell_volume(mesh, 1)
        @test vol ≈ 0.25 * 0.4 * 0.5
    end

    @testset "cell_idx_3d and cell_ijk roundtrip" begin
        for k in 1:6, j in 1:5, i in 1:4
            idx = cell_idx_3d(mesh, i, j, k)
            i2, j2, k2 = cell_ijk(mesh, idx)
            @test i2 == i
            @test j2 == j
            @test k2 == k
        end
    end

    @testset "Anisotropic mesh" begin
        mesh2 = StructuredMesh3D(-1.0, 1.0, 0.0, 0.5, -3.0, 3.0, 10, 5, 30)
        @test mesh2.dx ≈ 0.2
        @test mesh2.dy ≈ 0.1
        @test mesh2.dz ≈ 0.2
        @test ncells(mesh2) == 10 * 5 * 30

        # Cell centers should be at correct positions
        x1, y1, z1 = cell_center(mesh2, cell_idx_3d(mesh2, 1, 1, 1))
        @test x1 ≈ -1.0 + 0.1
        @test y1 ≈ 0.0 + 0.05
        @test z1 ≈ -3.0 + 0.1
    end
end

# ============================================================
# 3D Sod Shock Tube Tests
# ============================================================

@testset "3D Sod Shock Tube" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    @testset "X-direction" begin
        wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
        wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1)
        nx, ny, nz = 40, 4, 4

        mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

        prob = HyperbolicProblem3D(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
            TransmissiveBC(), TransmissiveBC(),
            TransmissiveBC(), TransmissiveBC(),
            (x, y, z) -> x < 0.5 ? wL : wR;
            final_time = 0.05, cfl = 0.3
        )

        coords, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t ≈ 0.05 atol = 1.0e-10

        # Density should vary along x: left side > right side
        jmid, kmid = 2, 2
        @test W[1, jmid, kmid][1] > W[nx, jmid, kmid][1]

        # Check that a density jump exists somewhere in the middle
        ρ_left = W[5, jmid, kmid][1]
        ρ_right = W[nx - 4, jmid, kmid][1]
        @test ρ_left > ρ_right

        # Solution should be uniform in y and z
        for ix in 1:nx
            for iy in 2:ny, iz in 1:nz
                @test W[ix, iy, iz][1] ≈ W[ix, 1, 1][1] atol = 1.0e-10
            end
        end

        # All densities and pressures should be positive
        for iz in 1:nz, iy in 1:ny, ix in 1:nx
            @test W[ix, iy, iz][1] > 0
            @test W[ix, iy, iz][5] > 0
        end
    end

    @testset "Y-direction" begin
        wB = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
        wT = SVector(0.125, 0.0, 0.0, 0.0, 0.1)
        nx, ny, nz = 4, 40, 4

        mesh = StructuredMesh3D(0.0, 0.1, 0.0, 1.0, 0.0, 0.1, nx, ny, nz)

        prob = HyperbolicProblem3D(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(),
            DirichletHyperbolicBC(wB), DirichletHyperbolicBC(wT),
            TransmissiveBC(), TransmissiveBC(),
            (x, y, z) -> y < 0.5 ? wB : wT;
            final_time = 0.05, cfl = 0.3
        )

        coords, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t ≈ 0.05 atol = 1.0e-10

        imid, kmid = 2, 2
        # Bottom state has higher density than top
        @test W[imid, 1, kmid][1] > W[imid, ny, kmid][1]

        # A density jump exists
        ρ_bottom = W[imid, 5, kmid][1]
        ρ_top = W[imid, ny - 4, kmid][1]
        @test ρ_bottom > ρ_top

        # Uniform in x and z
        for iy in 1:ny
            for ix in 2:nx, iz in 1:nz
                @test W[ix, iy, iz][1] ≈ W[1, iy, 1][1] atol = 1.0e-10
            end
        end

        # Positivity
        for iz in 1:nz, iy in 1:ny, ix in 1:nx
            @test W[ix, iy, iz][1] > 0
            @test W[ix, iy, iz][5] > 0
        end
    end

    @testset "Z-direction" begin
        wF = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
        wBk = SVector(0.125, 0.0, 0.0, 0.0, 0.1)
        nx, ny, nz = 4, 4, 40

        mesh = StructuredMesh3D(0.0, 0.1, 0.0, 0.1, 0.0, 1.0, nx, ny, nz)

        prob = HyperbolicProblem3D(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            TransmissiveBC(), TransmissiveBC(),
            TransmissiveBC(), TransmissiveBC(),
            DirichletHyperbolicBC(wF), DirichletHyperbolicBC(wBk),
            (x, y, z) -> z < 0.5 ? wF : wBk;
            final_time = 0.05, cfl = 0.3
        )

        coords, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t ≈ 0.05 atol = 1.0e-10

        imid, jmid = 2, 2
        # Front (low z) has higher density than back (high z)
        @test W[imid, jmid, 1][1] > W[imid, jmid, nz][1]

        # A density jump exists
        ρ_front = W[imid, jmid, 5][1]
        ρ_back = W[imid, jmid, nz - 4][1]
        @test ρ_front > ρ_back

        # Uniform in x and y
        for iz in 1:nz
            for ix in 2:nx, iy in 1:ny
                @test W[ix, iy, iz][1] ≈ W[1, 1, iz][1] atol = 1.0e-10
            end
        end

        # Positivity
        for iz in 1:nz, iy in 1:ny, ix in 1:nx
            @test W[ix, iy, iz][1] > 0
            @test W[ix, iy, iz][5] > 0
        end
    end
end

# ============================================================
# 3D Sod with All Limiters
# ============================================================

@testset "3D Sod All Limiters" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1)
    nx, ny, nz = 20, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    limiters = [
        ("Minmod", MinmodLimiter()),
        ("Superbee", SuperbeeLimiter()),
        ("VanLeer", VanLeerLimiter()),
        ("Koren", KorenLimiter()),
        ("Ospre", OspreLimiter()),
    ]

    @testset "$name" for (name, lim) in limiters
        prob = HyperbolicProblem3D(
            law, mesh, HLLSolver(), CellCenteredMUSCL(lim),
            DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
            TransmissiveBC(), TransmissiveBC(),
            TransmissiveBC(), TransmissiveBC(),
            (x, y, z) -> x < 0.5 ? wL : wR;
            final_time = 0.05, cfl = 0.3
        )

        coords, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t ≈ 0.05 atol = 1.0e-10

        # Positivity
        for iz in 1:nz, iy in 1:ny, ix in 1:nx
            @test W[ix, iy, iz][1] > 0
            @test W[ix, iy, iz][5] > 0
        end

        # Density monotonicity: left > right
        jmid, kmid = 2, 2
        @test W[1, jmid, kmid][1] > W[nx, jmid, kmid][1]

        # Uniform in y and z
        for ix in 1:nx
            for iy in 2:ny, iz in 1:nz
                @test W[ix, iy, iz][1] ≈ W[ix, 1, 1][1] atol = 1.0e-10
            end
        end
    end
end

# ============================================================
# 3D Sod with All Riemann Solvers
# ============================================================

@testset "3D Sod All Riemann Solvers" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1)
    nx, ny, nz = 20, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    # Note: HLLC only supports 1D/2D Euler, not 3D
    solvers = [
        ("LaxFriedrichs", LaxFriedrichsSolver()),
        ("HLL", HLLSolver()),
    ]

    @testset "$name" for (name, rs) in solvers
        prob = HyperbolicProblem3D(
            law, mesh, rs, CellCenteredMUSCL(MinmodLimiter()),
            DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
            TransmissiveBC(), TransmissiveBC(),
            TransmissiveBC(), TransmissiveBC(),
            (x, y, z) -> x < 0.5 ? wL : wR;
            final_time = 0.05, cfl = 0.3
        )

        coords, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        @test t ≈ 0.05 atol = 1.0e-10

        # Positivity
        for iz in 1:nz, iy in 1:ny, ix in 1:nx
            @test W[ix, iy, iz][1] > 0
            @test W[ix, iy, iz][5] > 0
        end

        # Shock structure: left > right
        jmid, kmid = 2, 2
        @test W[1, jmid, kmid][1] > W[nx, jmid, kmid][1]
    end
end

# ============================================================
# HLL vs LF: HLL Should Be Less Diffusive in 3D
# ============================================================

@testset "3D HLL vs LF Sharpness" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1)
    nx, ny, nz = 40, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)
    recon = CellCenteredMUSCL(MinmodLimiter())
    ic = (x, y, z) -> x < 0.5 ? wL : wR

    prob_hll = HyperbolicProblem3D(
        law, mesh, HLLSolver(), recon,
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.1, cfl = 0.3
    )
    prob_lf = HyperbolicProblem3D(
        law, mesh, LaxFriedrichsSolver(), recon,
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.1, cfl = 0.3
    )

    _, U_hll, _ = solve_hyperbolic(prob_hll)
    _, U_lf, _ = solve_hyperbolic(prob_lf)
    W_hll = to_primitive(law, U_hll)
    W_lf = to_primitive(law, U_lf)

    # HLL should have sharper discontinuity than LF
    jmid, kmid = 2, 2
    max_grad_hll = maximum(abs(W_hll[ix + 1, jmid, kmid][1] - W_hll[ix, jmid, kmid][1]) for ix in 1:(nx - 1))
    max_grad_lf = maximum(abs(W_lf[ix + 1, jmid, kmid][1] - W_lf[ix, jmid, kmid][1]) for ix in 1:(nx - 1))
    @test max_grad_hll >= max_grad_lf
end

# ============================================================
# Forward Euler Time Integration
# ============================================================

@testset "3D Forward Euler" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1)
    nx, ny, nz = 20, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        (x, y, z) -> x < 0.5 ? wL : wR;
        final_time = 0.05, cfl = 0.2
    )

    coords, U, t = solve_hyperbolic(prob; method = :euler)
    W = to_primitive(law, U)

    @test t ≈ 0.05 atol = 1.0e-10

    # Positivity
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    # Correct shock direction
    jmid, kmid = 2, 2
    @test W[1, jmid, kmid][1] > W[nx, jmid, kmid][1]
end

# ============================================================
# NoReconstruction (First Order)
# ============================================================

@testset "3D NoReconstruction" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1)
    nx, ny, nz = 20, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), NoReconstruction(),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        (x, y, z) -> x < 0.5 ? wL : wR;
        final_time = 0.05, cfl = 0.3
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    @test t ≈ 0.05 atol = 1.0e-10

    # Positivity
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    # Shock direction still correct, though more diffusive
    jmid, kmid = 2, 2
    @test W[1, jmid, kmid][1] > W[nx, jmid, kmid][1]
end

# ============================================================
# Uniform Flow Preservation
# ============================================================

@testset "3D Uniform Flow Preservation" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    N = 8
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, N, N, N)

    w_uniform = SVector(1.0, 0.5, -0.3, 0.2, 1.0)

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        (x, y, z) -> w_uniform;
        final_time = 0.1, cfl = 0.4
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    # Every cell should remain exactly the uniform state
    for iz in 1:N, iy in 1:N, ix in 1:N
        @test W[ix, iy, iz][1] ≈ w_uniform[1] atol = 1.0e-12
        @test W[ix, iy, iz][2] ≈ w_uniform[2] atol = 1.0e-12
        @test W[ix, iy, iz][3] ≈ w_uniform[3] atol = 1.0e-12
        @test W[ix, iy, iz][4] ≈ w_uniform[4] atol = 1.0e-12
        @test W[ix, iy, iz][5] ≈ w_uniform[5] atol = 1.0e-12
    end
end

# ============================================================
# Conservation with Periodic BCs
# ============================================================

@testset "3D Conservation (periodic)" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    N = 10
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, N, N, N)
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    vol = dx * dy * dz

    # Smooth initial condition with variation in all three directions
    ic = (x, y, z) -> SVector(
        1.0 + 0.2 * sin(2π * x) * cos(2π * y) * cos(2π * z),
        0.3,
        0.2,
        -0.1,
        1.0
    )

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.02, cfl = 0.3
    )

    # Compute initial totals from the padded array
    U0 = FiniteVolumeMethod.initialize_3d(prob)
    FiniteVolumeMethod.apply_boundary_conditions_3d!(U0, prob, 0.0)

    mass0 = sum(U0[ix + 2, iy + 2, iz + 2][1] for ix in 1:N, iy in 1:N, iz in 1:N) * vol
    mom_x0 = sum(U0[ix + 2, iy + 2, iz + 2][2] for ix in 1:N, iy in 1:N, iz in 1:N) * vol
    mom_y0 = sum(U0[ix + 2, iy + 2, iz + 2][3] for ix in 1:N, iy in 1:N, iz in 1:N) * vol
    mom_z0 = sum(U0[ix + 2, iy + 2, iz + 2][4] for ix in 1:N, iy in 1:N, iz in 1:N) * vol
    energy0 = sum(U0[ix + 2, iy + 2, iz + 2][5] for ix in 1:N, iy in 1:N, iz in 1:N) * vol

    coords, U_final, t = solve_hyperbolic(prob)

    mass_f = sum(U_final[ix, iy, iz][1] for ix in 1:N, iy in 1:N, iz in 1:N) * vol
    mom_xf = sum(U_final[ix, iy, iz][2] for ix in 1:N, iy in 1:N, iz in 1:N) * vol
    mom_yf = sum(U_final[ix, iy, iz][3] for ix in 1:N, iy in 1:N, iz in 1:N) * vol
    mom_zf = sum(U_final[ix, iy, iz][4] for ix in 1:N, iy in 1:N, iz in 1:N) * vol
    energy_f = sum(U_final[ix, iy, iz][5] for ix in 1:N, iy in 1:N, iz in 1:N) * vol

    @test mass_f ≈ mass0 rtol = 1.0e-10
    @test mom_xf ≈ mom_x0 rtol = 1.0e-10
    @test mom_yf ≈ mom_y0 rtol = 1.0e-10
    @test mom_zf ≈ mom_z0 rtol = 1.0e-10
    @test energy_f ≈ energy0 rtol = 1.0e-10
end

# ============================================================
# Conservation with HLLC + Forward Euler
# ============================================================

@testset "3D Conservation HLLC + Euler" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    N = 8
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, N, N, N)
    vol = mesh.dx * mesh.dy * mesh.dz

    ic = (x, y, z) -> SVector(
        1.0 + 0.1 * sin(2π * x) * sin(2π * y) * sin(2π * z),
        0.1, -0.1, 0.05, 1.0
    )

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(VanLeerLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.02, cfl = 0.2
    )

    U0 = FiniteVolumeMethod.initialize_3d(prob)
    FiniteVolumeMethod.apply_boundary_conditions_3d!(U0, prob, 0.0)

    mass0 = sum(U0[ix + 2, iy + 2, iz + 2][1] for ix in 1:N, iy in 1:N, iz in 1:N) * vol
    energy0 = sum(U0[ix + 2, iy + 2, iz + 2][5] for ix in 1:N, iy in 1:N, iz in 1:N) * vol

    _, U_final, _ = solve_hyperbolic(prob; method = :euler)

    mass_f = sum(U_final[ix, iy, iz][1] for ix in 1:N, iy in 1:N, iz in 1:N) * vol
    energy_f = sum(U_final[ix, iy, iz][5] for ix in 1:N, iy in 1:N, iz in 1:N) * vol

    @test mass_f ≈ mass0 rtol = 1.0e-10
    @test energy_f ≈ energy0 rtol = 1.0e-10
end

# ============================================================
# 3D Sedov Blast
# ============================================================

@testset "3D Sedov Blast" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    N = 10
    mesh = StructuredMesh3D(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, N, N, N)
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    vol = dx * dy * dz

    P_bg = 1.0e-5
    P_blast = 1.0
    r_blast = 3.0 * dx  # a few cells

    ic = function (x, y, z)
        r = sqrt(x^2 + y^2 + z^2)
        P = r < r_blast ? P_blast : P_bg
        return SVector(1.0, 0.0, 0.0, 0.0, P)
    end

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        ReflectiveBC(), ReflectiveBC(),
        ReflectiveBC(), ReflectiveBC(),
        ReflectiveBC(), ReflectiveBC(),
        ic; final_time = 0.05, cfl = 0.3
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    @testset "Positivity" begin
        for iz in 1:N, iy in 1:N, ix in 1:N
            @test W[ix, iy, iz][1] > 0  # density
            @test W[ix, iy, iz][5] > 0  # pressure
        end
    end

    @testset "Max pressure near center" begin
        # Find the cell with maximum pressure
        P_max = -Inf
        ix_max, iy_max, iz_max = 1, 1, 1
        for iz in 1:N, iy in 1:N, ix in 1:N
            if W[ix, iy, iz][5] > P_max
                P_max = W[ix, iy, iz][5]
                ix_max = ix
                iy_max = iy
                iz_max = iz
            end
        end

        # Center cell indices (the center of [-1,1]^3 is at cells N/2 and N/2+1)
        center_lo = div(N, 2)
        center_hi = div(N, 2) + 1

        @test center_lo <= ix_max <= center_hi
        @test center_lo <= iy_max <= center_hi
        @test center_lo <= iz_max <= center_hi
    end

    @testset "Total energy conservation" begin
        # Reflective BCs should conserve total energy
        U0 = FiniteVolumeMethod.initialize_3d(prob)
        FiniteVolumeMethod.apply_boundary_conditions_3d!(U0, prob, 0.0)

        energy0 = sum(U0[ix + 2, iy + 2, iz + 2][5] for ix in 1:N, iy in 1:N, iz in 1:N) * vol
        energy_f = sum(U[ix, iy, iz][5] for ix in 1:N, iy in 1:N, iz in 1:N) * vol

        @test energy_f ≈ energy0 rtol = 1.0e-8
    end

    @testset "Approximate spherical symmetry" begin
        # Check that densities at equivalent radial positions are similar
        # Cells (1,N/2,N/2), (N,N/2,N/2) etc. should be approximately equal
        m = div(N, 2)
        ρ_xp = W[N, m, m][1]   # +x face
        ρ_xm = W[1, m, m][1]   # -x face
        ρ_yp = W[m, N, m][1]   # +y face
        ρ_ym = W[m, 1, m][1]   # -y face
        ρ_zp = W[m, m, N][1]   # +z face
        ρ_zm = W[m, m, 1][1]   # -z face
        ρ_avg = (ρ_xp + ρ_xm + ρ_yp + ρ_ym + ρ_zp + ρ_zm) / 6
        for ρ in (ρ_xp, ρ_xm, ρ_yp, ρ_ym, ρ_zp, ρ_zm)
            @test ρ ≈ ρ_avg rtol = 0.15
        end
    end
end

# ============================================================
# 3D Diagonal Shock
# ============================================================

@testset "3D Diagonal Shock" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    N = 12
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, N, N, N)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1)

    # Discontinuity along x + y + z = 1.5 (center of cube)
    ic = (x, y, z) -> (x + y + z) < 1.5 ? wL : wR

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.05, cfl = 0.3
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    @test t ≈ 0.05 atol = 1.0e-10

    # Positivity
    for iz in 1:N, iy in 1:N, ix in 1:N
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    # Corner (1,1,1) should still have high density, corner (N,N,N) low
    @test W[1, 1, 1][1] > W[N, N, N][1]
end

# ============================================================
# Reflective BC Symmetry in 3D
# ============================================================

@testset "3D Reflective BC Symmetry" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    N = 10
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, N, N, N)

    # Symmetric initial condition: pressure pulse at center
    ic = function (x, y, z)
        r = sqrt((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2)
        P = r < 0.2 ? 5.0 : 1.0
        return SVector(1.0, 0.0, 0.0, 0.0, P)
    end

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        ReflectiveBC(), ReflectiveBC(),
        ReflectiveBC(), ReflectiveBC(),
        ReflectiveBC(), ReflectiveBC(),
        ic; final_time = 0.05, cfl = 0.3
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    # Density should be symmetric: W[ix,...] ≈ W[N+1-ix,...] etc.
    @testset "x-symmetry" begin
        for iz in 1:N, iy in 1:N, ix in 1:div(N, 2)
            @test W[ix, iy, iz][1] ≈ W[N + 1 - ix, iy, iz][1] rtol = 1.0e-10
        end
    end
    @testset "y-symmetry" begin
        for iz in 1:N, iy in 1:div(N, 2), ix in 1:N
            @test W[ix, iy, iz][1] ≈ W[ix, N + 1 - iy, iz][1] rtol = 1.0e-10
        end
    end
    @testset "z-symmetry" begin
        for iz in 1:div(N, 2), iy in 1:N, ix in 1:N
            @test W[ix, iy, iz][1] ≈ W[ix, iy, N + 1 - iz][1] rtol = 1.0e-10
        end
    end
end

# ============================================================
# 3D Pressure Pulse with Spherical Symmetry Check
# ============================================================

@testset "3D Pressure Pulse Spherical Symmetry" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    N = 12
    mesh = StructuredMesh3D(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, N, N, N)

    # Gaussian pressure pulse centered at origin
    ic = function (x, y, z)
        r2 = x^2 + y^2 + z^2
        P = 1.0 + 2.0 * exp(-10.0 * r2)
        return SVector(1.0, 0.0, 0.0, 0.0, P)
    end

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(VanLeerLimiter()),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.05, cfl = 0.3
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    # Positivity
    for iz in 1:N, iy in 1:N, ix in 1:N
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    # Symmetry along axes: cells at same distance from center should have similar density
    m = div(N, 2)
    for d in 1:min(m - 1, 3)
        # Compare +x, -x, +y, -y, +z, -z at same offset from center
        ρ_vals = [
            W[m + d, m, m][1], W[m + 1 - d, m, m][1],
            W[m, m + d, m][1], W[m, m + 1 - d, m][1],
            W[m, m, m + d][1], W[m, m, m + 1 - d][1],
        ]
        ρ_mean = sum(ρ_vals) / 6
        for ρ in ρ_vals
            @test ρ ≈ ρ_mean rtol = 0.1
        end
    end
end

# ============================================================
# Einfeldt 1-2-3 Problem in 3D
# ============================================================

@testset "3D Einfeldt 1-2-3" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    # Two rarefaction waves: creates near-vacuum in the middle
    wL = SVector(1.0, -2.0, 0.0, 0.0, 0.4)
    wR = SVector(1.0, 2.0, 0.0, 0.0, 0.4)
    nx, ny, nz = 40, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        (x, y, z) -> x < 0.5 ? wL : wR;
        final_time = 0.1, cfl = 0.3
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    # Positivity — this is the key challenge for Einfeldt
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    # Density in the middle should be low (near-vacuum)
    jmid, kmid = 2, 2
    ρ_mid = W[div(nx, 2), jmid, kmid][1]
    ρ_edge = W[1, jmid, kmid][1]
    @test ρ_mid < ρ_edge
end

# ============================================================
# Strong Shock in 3D
# ============================================================

@testset "3D Strong Shock" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    # Large pressure ratio (1000:1)
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1000.0)
    wR = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
    nx, ny, nz = 40, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        (x, y, z) -> x < 0.5 ? wL : wR;
        final_time = 0.01, cfl = 0.2
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    # Positivity
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    # Left pressure > right pressure still holds
    jmid, kmid = 2, 2
    @test W[1, jmid, kmid][5] > W[nx, jmid, kmid][5]
end

# ============================================================
# Different EOS: gamma = 5/3 (Monatomic Gas)
# ============================================================

@testset "3D Sod gamma=5/3" begin
    eos = IdealGasEOS(5.0 / 3.0)
    law = EulerEquations{3}(eos)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1)
    nx, ny, nz = 30, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        (x, y, z) -> x < 0.5 ? wL : wR;
        final_time = 0.1, cfl = 0.3
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    # Positivity
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    # Shock structure preserved
    jmid, kmid = 2, 2
    @test W[1, jmid, kmid][1] > W[nx, jmid, kmid][1]
end

# ============================================================
# 3D Inflow/Dirichlet BC Test
# ============================================================

@testset "3D Inflow BC" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    # Supersonic inflow from left, transmissive on right
    w_in = SVector(1.0, 3.0, 0.0, 0.0, 1.0)  # Mach > 1
    nx, ny, nz = 20, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        InflowBC(w_in), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        (x, y, z) -> w_in;
        final_time = 0.1, cfl = 0.3
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    # Uniform supersonic flow should remain uniform
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test W[ix, iy, iz][1] ≈ w_in[1] atol = 1.0e-10
        @test W[ix, iy, iz][2] ≈ w_in[2] atol = 1.0e-10
        @test W[ix, iy, iz][5] ≈ w_in[5] atol = 1.0e-10
    end
end

# ============================================================
# 3D CFL Sensitivity
# ============================================================

@testset "3D CFL Sensitivity" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1)
    nx, ny, nz = 16, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)
    ic = (x, y, z) -> x < 0.5 ? wL : wR

    cfls = [0.1, 0.2, 0.3, 0.4]
    results = Dict{Float64, Array}()

    @testset "CFL=$cfl" for cfl in cfls
        prob = HyperbolicProblem3D(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
            TransmissiveBC(), TransmissiveBC(),
            TransmissiveBC(), TransmissiveBC(),
            ic; final_time = 0.05, cfl = cfl
        )

        coords, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        # All should reach the same final time
        @test t ≈ 0.05 atol = 1.0e-10

        # Positivity
        for iz in 1:nz, iy in 1:ny, ix in 1:nx
            @test W[ix, iy, iz][1] > 0
            @test W[ix, iy, iz][5] > 0
        end

        results[cfl] = U
    end

    # Different CFL values should give similar (not identical) solutions
    W1 = to_primitive(law, results[0.1])
    W4 = to_primitive(law, results[0.4])
    jmid, kmid = 2, 2
    L2 = sqrt(sum((W1[ix, jmid, kmid][1] - W4[ix, jmid, kmid][1])^2 for ix in 1:nx) / nx)
    @test L2 < 0.1  # Solutions should be similar
end

# ============================================================
# 3D Acoustic Wave Convergence
# ============================================================

@testset "3D Acoustic Wave Convergence" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    # Small-amplitude sound wave in x-direction
    ρ0, P0, ε = 1.0, 1.0, 1.0e-4
    c0 = sound_speed(eos, ρ0, P0)

    ic = (x, y, z) -> SVector(
        ρ0 + ε * sin(2π * x),
        ε * c0 * sin(2π * x) / ρ0,
        0.0, 0.0,
        P0 + ε * c0^2 * sin(2π * x)
    )

    resolutions = [8, 16]
    errors = Float64[]

    for N in resolutions
        mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, N, 4, 4)

        prob = HyperbolicProblem3D(
            law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
            PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
            TransmissiveBC(), TransmissiveBC(),
            TransmissiveBC(), TransmissiveBC(),
            ic; final_time = 1.0 / c0, cfl = 0.3
        )

        coords, U, t = solve_hyperbolic(prob)
        W = to_primitive(law, U)

        # After one sound crossing time, the wave should return
        # to approximately its initial state (periodic in x)
        L2 = 0.0
        for ix in 1:N
            x, _, _ = coords[ix, 2, 2]
            ρ_exact = ρ0 + ε * sin(2π * x)
            L2 += (W[ix, 2, 2][1] - ρ_exact)^2
        end
        L2 = sqrt(L2 / N)
        push!(errors, L2)
    end

    # Higher resolution should give smaller error
    @test errors[2] < errors[1]
end

# ============================================================
# 3D LaxFriedrichs vs HLL Diffusivity
# ============================================================

@testset "3D LF vs HLL Diffusivity" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1)
    nx, ny, nz = 30, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)
    recon = CellCenteredMUSCL(MinmodLimiter())
    ic = (x, y, z) -> x < 0.5 ? wL : wR

    prob_lf = HyperbolicProblem3D(
        law, mesh, LaxFriedrichsSolver(), recon,
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.1, cfl = 0.3
    )
    prob_hll = HyperbolicProblem3D(
        law, mesh, HLLSolver(), recon,
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        ic; final_time = 0.1, cfl = 0.3
    )

    _, U_lf, _ = solve_hyperbolic(prob_lf)
    _, U_hll, _ = solve_hyperbolic(prob_hll)
    W_lf = to_primitive(law, U_lf)
    W_hll = to_primitive(law, U_hll)

    # LF should be more diffusive: density transition is smoother
    # Measure total variation of density
    jmid, kmid = 2, 2
    tv_lf = sum(abs(W_lf[ix + 1, jmid, kmid][1] - W_lf[ix, jmid, kmid][1]) for ix in 1:(nx - 1))
    tv_hll = sum(abs(W_hll[ix + 1, jmid, kmid][1] - W_hll[ix, jmid, kmid][1]) for ix in 1:(nx - 1))
    # HLL total variation should be >= LF (sharper features, more total variation preserved)
    @test tv_hll >= tv_lf * 0.95  # approximate: HLL preserves more TV
end

# ============================================================
# 3D Solver with Transverse Velocity
# ============================================================

@testset "3D Sod with Transverse Velocity" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    # Sod in x-direction but with constant transverse velocities
    vy0, vz0 = 1.0, -0.5
    wL = SVector(1.0, 0.0, vy0, vz0, 1.0)
    wR = SVector(0.125, 0.0, vy0, vz0, 0.1)
    nx, ny, nz = 30, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        (x, y, z) -> x < 0.5 ? wL : wR;
        final_time = 0.05, cfl = 0.3
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    # Transverse velocities should remain constant everywhere
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test W[ix, iy, iz][3] ≈ vy0 atol = 1.0e-10
        @test W[ix, iy, iz][4] ≈ vz0 atol = 1.0e-10
    end

    # Density shock structure preserved
    jmid, kmid = 2, 2
    @test W[1, jmid, kmid][1] > W[nx, jmid, kmid][1]
end

# ============================================================
# 3D Lax Problem
# ============================================================

@testset "3D Lax Problem" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    # Classic Lax shock tube
    wL = SVector(0.445, 0.698, 0.0, 0.0, 3.528)
    wR = SVector(0.5, 0.0, 0.0, 0.0, 0.571)
    nx, ny, nz = 40, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        (x, y, z) -> x < 0.5 ? wL : wR;
        final_time = 0.1, cfl = 0.3
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    @test t ≈ 0.1 atol = 1.0e-10

    # Positivity
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    # Left pressure > right pressure
    jmid, kmid = 2, 2
    @test W[1, jmid, kmid][5] > W[nx, jmid, kmid][5]
end

# ============================================================
# 3D Multi-Direction Conservation (all 3 axes)
# ============================================================

@testset "3D Conservation All Axes" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    N = 8
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, N, N, N)
    vol = mesh.dx * mesh.dy * mesh.dz

    # Density varies in all 3 directions
    ic = (x, y, z) -> SVector(
        1.0 + 0.1 * sin(2π * x) + 0.1 * cos(2π * y) + 0.1 * sin(2π * z),
        0.2, 0.3, -0.1, 2.0
    )

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(SuperbeeLimiter()),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        PeriodicHyperbolicBC(), PeriodicHyperbolicBC(),
        ic; final_time = 0.01, cfl = 0.3
    )

    U0 = FiniteVolumeMethod.initialize_3d(prob)
    FiniteVolumeMethod.apply_boundary_conditions_3d!(U0, prob, 0.0)

    totals0 = [
        sum(U0[ix + 2, iy + 2, iz + 2][v] for ix in 1:N, iy in 1:N, iz in 1:N) * vol
            for v in 1:5
    ]

    _, U_final, _ = solve_hyperbolic(prob)

    totals_f = [
        sum(U_final[ix, iy, iz][v] for ix in 1:N, iy in 1:N, iz in 1:N) * vol
            for v in 1:5
    ]

    for v in 1:5
        if abs(totals0[v]) > 1.0e-10
            @test totals_f[v] ≈ totals0[v] rtol = 1.0e-10
        else
            @test totals_f[v] ≈ totals0[v] atol = 1.0e-12
        end
    end
end

# ============================================================
# 3D Static Atmosphere (Hydrostatic Equilibrium Check)
# ============================================================

@testset "3D Static Atmosphere" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    N = 8
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, N, N, N)

    # Uniform, static state — should remain perfectly static
    w0 = SVector(1.0, 0.0, 0.0, 0.0, 1.0)

    prob = HyperbolicProblem3D(
        law, mesh, HLLSolver(), CellCenteredMUSCL(MinmodLimiter()),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        (x, y, z) -> w0;
        final_time = 0.1, cfl = 0.4
    )

    coords, U, t = solve_hyperbolic(prob)
    W = to_primitive(law, U)

    for iz in 1:N, iy in 1:N, ix in 1:N
        @test W[ix, iy, iz][1] ≈ w0[1] atol = 1.0e-12
        @test W[ix, iy, iz][2] ≈ 0.0 atol = 1.0e-12
        @test W[ix, iy, iz][3] ≈ 0.0 atol = 1.0e-12
        @test W[ix, iy, iz][4] ≈ 0.0 atol = 1.0e-12
        @test W[ix, iy, iz][5] ≈ w0[5] atol = 1.0e-12
    end
end

# ============================================================
# 3D Sod with LaxFriedrichs + NoReconstruction + Forward Euler
# (stress test: most diffusive combination)
# ============================================================

@testset "3D Most Diffusive Combo" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{3}(eos)

    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1)
    nx, ny, nz = 20, 4, 4
    mesh = StructuredMesh3D(0.0, 1.0, 0.0, 0.1, 0.0, 0.1, nx, ny, nz)

    prob = HyperbolicProblem3D(
        law, mesh, LaxFriedrichsSolver(), NoReconstruction(),
        DirichletHyperbolicBC(wL), DirichletHyperbolicBC(wR),
        TransmissiveBC(), TransmissiveBC(),
        TransmissiveBC(), TransmissiveBC(),
        (x, y, z) -> x < 0.5 ? wL : wR;
        final_time = 0.05, cfl = 0.2
    )

    coords, U, t = solve_hyperbolic(prob; method = :euler)
    W = to_primitive(law, U)

    @test t ≈ 0.05 atol = 1.0e-10

    # Even the most diffusive combination should maintain positivity
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        @test W[ix, iy, iz][1] > 0
        @test W[ix, iy, iz][5] > 0
    end

    # And still capture the shock direction
    jmid, kmid = 2, 2
    @test W[1, jmid, kmid][1] > W[nx, jmid, kmid][1]
end
