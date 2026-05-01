using Test
using FiniteVolumeMethod
using LinearAlgebra

# Tests for VariableCylindricalAdvectionDiffusion2D — spatially varying
# velocity AND diffusion coefficient on an annular (r_inner > 0) 2D mesh.

@testset "VariableCylindricalAdvectionDiffusion2D" begin
    r_inner = 4.75e-3
    r_outer = 4.8e-3
    L = 1.0e-2

    @testset "Recovers constant CylindricalAdvectionDiffusion2D" begin
        # γ and (vr, vz) constant via Function: matrices and RHS must match
        # the constant-coefficient assembler bit-for-bit.
        nx, ny = 30, 12
        γ₀, vr0, vz0 = 0.5, 1.0e-3, 2.0e-3

        x_nodes = collect(range(r_inner, r_outer; length = nx + 1))
        y_nodes = collect(range(0.0, L; length = ny + 1))
        mesh = generate_mesh_2d_nonuniform(nx, ny, r_outer - r_inner, L, x_nodes, y_nodes)

        bcs = (
            ParabolicDirichlet(1.0),
            ParabolicDirichlet(0.0),
            ParabolicDirichlet(0.5),
            ParabolicDirichlet(0.5),
        )
        Q(r, z) = 1.0e-3

        m_const = CylindricalAdvectionDiffusion2D(
            CylindricalAdvection2D(vr0, vz0),
            CylindricalDiffusion2D(γ₀),
        )
        A1, b1 = assemble_system(m_const, mesh, bcs; source = FunctionSource(Q))

        m_var = VariableCylindricalAdvectionDiffusion2D(
            VariableCylindricalAdvection2D((r, z) -> vr0, (r, z) -> vz0),
            VariableCylindricalDiffusion2D((r, z) -> γ₀),
        )
        A2, b2 = assemble_system(m_var, mesh, bcs; source = FunctionSource(Q))

        @test maximum(abs.(A1 - A2)) < 1e-12
        @test maximum(abs.(b1 - b2)) < 1e-12
    end

    @testset "MMS — variable γ(r, z) and axial advection vz(r)" begin
        # T(r, z) = sin(α(r-r_i)) sin(βz) → vanishes on all 4 boundaries.
        # γ(r, z) = γ₀ (1 + a₁(r-r_i) + a₂ z)        (50% spread axially)
        # vr(r, z) = 0,  vz(r, z) = w₀ (1 + b₁(r-r_i)) — spatially varying.
        # Source Q = vz·∂T/∂z - γ ∇²T - (∂γ/∂r) ∂T/∂r - (∂γ/∂z) ∂T/∂z.
        α = π / (r_outer - r_inner)
        β = π / L

        γ₀, a₁, a₂ = 1.0e-3, 1.0e3, 50.0
        w₀, b₁     = 2.0e-3, 5.0e2

        γ_fn(r, z)  = γ₀ * (1 + a₁ * (r - r_inner) + a₂ * z)
        vr_fn(r, z) = 0.0
        vz_fn(r, z) = w₀ * (1 + b₁ * (r - r_inner))
        γr(r, z)    = γ₀ * a₁
        γz(r, z)    = γ₀ * a₂

        T_exact(r, z) = sin(α * (r - r_inner)) * sin(β * z)
        T_r(r, z)     = α * cos(α * (r - r_inner)) * sin(β * z)
        T_z(r, z)     = β * sin(α * (r - r_inner)) * cos(β * z)

        function lap_cyl(r, z)
            return -(α^2 + β^2) * T_exact(r, z) + (α / r) * cos(α * (r - r_inner)) * sin(β * z)
        end

        Q_exact(r, z) = vz_fn(r, z) * T_z(r, z) -
                        γ_fn(r, z) * lap_cyl(r, z) -
                        γr(r, z) * T_r(r, z) -
                        γz(r, z) * T_z(r, z)

        function solve_on(nx, ny)
            x_nodes = collect(range(r_inner, r_outer; length = nx + 1))
            y_nodes = collect(range(0.0, L; length = ny + 1))
            mesh = generate_mesh_2d_nonuniform(nx, ny, r_outer - r_inner, L, x_nodes, y_nodes)
            bcs = ntuple(_ -> ParabolicDirichlet(0.0), 4)

            model = VariableCylindricalAdvectionDiffusion2D(
                VariableCylindricalAdvection2D(vr_fn, vz_fn),
                VariableCylindricalDiffusion2D(γ_fn),
            )
            A, b = assemble_system(model, mesh, bcs; source = FunctionSource(Q_exact))
            T = A \ Vector(b)

            err = 0.0
            for i in 1:nx, j in 1:ny
                k = (i - 1) * ny + j
                r_c, z_c = mesh.cells[k].center
                err = max(err, abs(T[k] - T_exact(r_c, z_c)))
            end
            return err
        end

        sizes = [(20, 8), (40, 16), (80, 32), (160, 64)]
        errs  = Float64[solve_on(nx, ny) for (nx, ny) in sizes]
        rates = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]

        # Upwind advection is first-order in space; combined with second-order
        # diffusion the asymptotic rate sits between 1 and 2 depending on
        # which term dominates. Pe number here is small (γ ~ 1e-3 vs uL ~ 2e-5)
        # so diffusion dominates — expect close to second order.
        @test rates[end] > 1.5
        @test errs[end] < errs[1] / 10
    end
end
