using Test
using FiniteVolumeMethod
using LinearAlgebra

# Regression coverage for the ParabolicRobin handlers in
# src/parabolic/assembly/assembly_cylindrical.jl. These previously carried a
# spurious sign flip on :right (1D) and :right/:top (2D) that produced large
# errors at the Robin boundary — see the CRUD.jl side repro.

@testset "Cylindrical ParabolicRobin sign correctness" begin
    @testset "1D radial conduction with right-Robin" begin
        # Steady k * (1/r) d/dr (r dT/dr) = 0 between r_inner..r_outer
        # with Neumann inner (q_inner) and Robin outer (h, T_coolant).
        r_inner = 4.75e-3
        r_outer = 4.8e-3
        n_cells = 80
        k = 0.5
        q_inner = 1.0e6
        h_outer = 4.0e4
        T_coolant = 600.0

        T_analytic(r) = T_coolant + q_inner * r_inner *
            (log(r_outer / r) / k + 1 / (h_outer * r_outer))

        nodes = collect(range(r_inner, r_outer; length = n_cells + 1))
        mesh = generate_mesh_1d_nonuniform(nodes)

        bc_inner = ParabolicNeumann(-q_inner)
        bc_outer = ParabolicRobin(h_outer, 1.0, h_outer * T_coolant)

        A, b = assemble_system(CylindricalDiffusion1D(k), mesh, bc_inner, bc_outer)
        T = A \ Vector(b)

        r_centers = [mesh.cells[i].center for i in 1:n_cells]
        T_ref = T_analytic.(r_centers)

        @test maximum(abs.(T .- T_ref)) < 1.0e-3  # ~2e-5 on this mesh
    end

    @testset "2D axisymmetric solid cylinder with right-Robin (uniform source)" begin
        # γ ∇²T + Q = 0 in solid cylinder r ∈ [0, R], z ∈ [0, H].
        # BCs: symmetry at r=0, Robin at r=R, insulated top/bottom.
        # → 1D radial: T(r) = T_∞ + Q R/(2h) + Q/(4γ) (R² - r²)
        R = 5.0e-3
        H = 1.0e-2
        nx = 40
        ny = 8
        γ = 0.5
        Q = 1.0e8
        h = 4.0e4
        T_∞ = 600.0

        T_analytic(r) = T_∞ + Q * R / (2h) + Q / (4γ) * (R^2 - r^2)

        mesh = generate_mesh_2d(nx, ny, R, H)

        bcs = (
            ParabolicNeumann(0.0),                        # left (r=0): symmetry
            ParabolicRobin(h, 1.0, h * T_∞),              # right (r=R): convection
            ParabolicNeumann(0.0),                        # bottom (z=0): insulated
            ParabolicNeumann(0.0),                        # top (z=H): insulated
        )

        A, b = assemble_system(
            CylindricalDiffusion2D(γ), mesh, bcs;
            source = ConstantSource(Q),
        )
        T = A \ Vector(b)

        # Cell centers: r at (i-0.5)*dx, indexed k = (i-1)*ny + j.
        dx = R / nx
        max_err = 0.0
        for i in 1:nx, j in 1:ny
            r_c = (i - 0.5) * dx
            k = (i - 1) * ny + j
            err = abs(T[k] - T_analytic(r_c))
            max_err = max(max_err, err)
        end

        # Discretization error scales like (Q R²/γ) * O(1/nx²); generous bound.
        @test max_err < 5.0
    end

    @testset "2D axisymmetric slab with top-Robin (uniform source)" begin
        # Insulated radial sides, insulated bottom, Robin at top.
        # Reduces to 1D axial conduction:
        # γ T''(z) + Q = 0, T'(0)=0, γ T'(H) + h(T(H)-T_∞) = 0
        # → T(z) = T_∞ + Q H/h + Q/(2γ) (H² - z²)
        R = 5.0e-3
        H = 5.0e-3
        nx = 6
        ny = 40
        γ = 0.5
        Q = 1.0e8
        h = 4.0e4
        T_∞ = 600.0

        T_analytic(z) = T_∞ + Q * H / h + Q / (2γ) * (H^2 - z^2)

        mesh = generate_mesh_2d(nx, ny, R, H)

        bcs = (
            ParabolicNeumann(0.0),                        # left (r=0): symmetry
            ParabolicNeumann(0.0),                        # right (r=R): insulated
            ParabolicNeumann(0.0),                        # bottom (z=0): insulated
            ParabolicRobin(h, 1.0, h * T_∞),              # top (z=H): convection
        )

        A, b = assemble_system(
            CylindricalDiffusion2D(γ), mesh, bcs;
            source = ConstantSource(Q),
        )
        T = A \ Vector(b)

        dy = H / ny
        max_err = 0.0
        for i in 1:nx, j in 1:ny
            z_c = (j - 0.5) * dy
            k = (i - 1) * ny + j
            err = abs(T[k] - T_analytic(z_c))
            max_err = max(max_err, err)
        end

        @test max_err < 5.0
    end
end
