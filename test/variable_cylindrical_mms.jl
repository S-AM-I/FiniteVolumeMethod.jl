using Test
using FiniteVolumeMethod
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC, RobinBC
using LinearAlgebra

# MMS for VariableCylindricalDiffusion2D. Manufactured solution
# T(r, z) = sin(α (r - r_i)) sin(β z) on the annular domain, with a
# spatially varying diffusion coefficient
#
#   γ(r, z) = γ₀ (1 + a₁ (r - r_i) + a₂ z)
#
# Source term derived from γ ∇²T + ∇γ · ∇T + Q = 0, i.e.
#   Q = -[ γ ∇²T + (∂γ/∂r) (∂T/∂r) + (∂γ/∂z) (∂T/∂z) ]
# In cylindrical coords the Laplacian piece is
#   ∇²T = ∂²T/∂r² + (1/r) ∂T/∂r + ∂²T/∂z².

@testset "VariableCylindricalDiffusion2D — MMS at second order" begin
    r_inner = 4.75e-3
    r_outer = 4.8e-3
    L = 1.0e-2

    α = π / (r_outer - r_inner)
    β = π / L

    γ₀ = 0.5
    a₁ = 1.0e3   # γ varies ~5% across the radial extent
    a₂ = 50.0    # γ varies ~50% along the axial extent
    γ_fn(r, z) = γ₀ * (1 + a₁ * (r - r_inner) + a₂ * z)

    T_exact(r, z) = sin(α * (r - r_inner)) * sin(β * z)

    function T_r(r, z)   # ∂T/∂r
        return α * cos(α * (r - r_inner)) * sin(β * z)
    end
    function T_z(r, z)   # ∂T/∂z
        return β * sin(α * (r - r_inner)) * cos(β * z)
    end
    function T_rr(r, z)  # ∂²T/∂r²
        return -α^2 * T_exact(r, z)
    end
    function T_zz(r, z)  # ∂²T/∂z²
        return -β^2 * T_exact(r, z)
    end

    γr_fn(r, z) = γ₀ * a₁
    γz_fn(r, z) = γ₀ * a₂

    function Q_exact(r, z)
        γ = γ_fn(r, z)
        lap = T_rr(r, z) + (1.0 / r) * T_r(r, z) + T_zz(r, z)
        return -(γ * lap + γr_fn(r, z) * T_r(r, z) + γz_fn(r, z) * T_z(r, z))
    end

    function solve_on(nx, ny)
        x_nodes = collect(range(r_inner, r_outer; length = nx + 1))
        y_nodes = collect(range(0.0, L; length = ny + 1))
        mesh = generate_mesh_2d_nonuniform(nx, ny, r_outer - r_inner, L, x_nodes, y_nodes)

        bcs = ntuple(_ -> DirichletBC(0.0), 4)
        model = VariableCylindricalDiffusion2D(γ_fn)
        A, b = assemble_system(model, mesh, bcs; source = FunctionSource(Q_exact))
        T = A \ Vector(b)

        max_err = 0.0
        for i in 1:nx, j in 1:ny
            k = (i - 1) * ny + j
            r_c = mesh.cells[k].center[1]
            z_c = mesh.cells[k].center[2]
            err = abs(T[k] - T_exact(r_c, z_c))
            max_err = max(max_err, err)
        end
        return max_err
    end

    sizes = [(20, 8), (40, 16), (80, 32), (160, 64)]
    errors = Float64[solve_on(nx, ny) for (nx, ny) in sizes]
    rates = [log2(errors[i] / errors[i + 1]) for i in 1:(length(errors) - 1)]

    @test rates[end] > 1.7    # asymptotic ≥ ~1.8 in practice
    @test errors[end] < errors[1] / 30
end

@testset "VariableCylindricalDiffusion2D — recovers constant-γ result" begin
    # With γ(r, z) = γ₀ everywhere, the variable solver must produce the
    # same matrix as the constant CylindricalDiffusion2D solver.
    r_inner, r_outer, L = 4.75e-3, 4.8e-3, 1.0e-2
    γ₀ = 0.5
    nx, ny = 30, 12

    x_nodes = collect(range(r_inner, r_outer; length = nx + 1))
    y_nodes = collect(range(0.0, L; length = ny + 1))
    mesh = generate_mesh_2d_nonuniform(nx, ny, r_outer - r_inner, L, x_nodes, y_nodes)

    Q_const(r, z) = 1.0e8

    bcs = (
        NeumannBC(-1.0e6),
        RobinBC(4.0e4, 1.0, 4.0e4 * 600.0),
        NeumannBC(0.0),
        NeumannBC(0.0),
    )

    A_const, b_const = assemble_system(
        CylindricalDiffusion2D(γ₀), mesh, bcs;
        source = FunctionSource(Q_const)
    )
    A_var, b_var = assemble_system(
        VariableCylindricalDiffusion2D((r, z) -> γ₀), mesh, bcs;
        source = FunctionSource(Q_const)
    )

    @test maximum(abs.(A_const - A_var)) < 1.0e-12
    @test maximum(abs.(b_const - b_var)) < 1.0e-9
end

@testset "VariableCylindricalDiffusion1D — recovers constant-γ result" begin
    r_inner, r_outer = 4.75e-3, 4.8e-3
    γ₀ = 0.5
    n_cells = 80
    nodes = collect(range(r_inner, r_outer; length = n_cells + 1))
    mesh = generate_mesh_1d_nonuniform(nodes)
    bc_inner = NeumannBC(-1.0e6)
    bc_outer = RobinBC(4.0e4, 1.0, 4.0e4 * 600.0)

    A_const, b_const = assemble_system(CylindricalDiffusion1D(γ₀), mesh, bc_inner, bc_outer)
    A_var, b_var = assemble_system(VariableCylindricalDiffusion1D(r -> γ₀), mesh, bc_inner, bc_outer)

    @test maximum(abs.(A_const - A_var)) < 1.0e-12
    @test maximum(abs.(b_const - b_var)) < 1.0e-9
end
