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
        @test w2 ≈ w atol = 1e-14

        # Another state
        w3 = SVector(0.125, -1.0, 0.7, 0.4, 0.1)
        u3 = primitive_to_conserved(law, w3)
        w4 = conserved_to_primitive(law, u3)
        @test w4 ≈ w3 atol = 1e-14
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

        @test t ≈ 0.05 atol = 1e-10

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
                @test W[ix, iy, iz][1] ≈ W[ix, 1, 1][1] atol = 1e-10
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

        @test t ≈ 0.05 atol = 1e-10

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
                @test W[ix, iy, iz][1] ≈ W[1, iy, 1][1] atol = 1e-10
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

        @test t ≈ 0.05 atol = 1e-10

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
                @test W[ix, iy, iz][1] ≈ W[1, 1, iz][1] atol = 1e-10
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

    @test mass_f ≈ mass0 rtol = 1e-10
    @test mom_xf ≈ mom_x0 rtol = 1e-10
    @test mom_yf ≈ mom_y0 rtol = 1e-10
    @test mom_zf ≈ mom_z0 rtol = 1e-10
    @test energy_f ≈ energy0 rtol = 1e-10
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

    P_bg = 1e-5
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

        @test energy_f ≈ energy0 rtol = 1e-8
    end
end
