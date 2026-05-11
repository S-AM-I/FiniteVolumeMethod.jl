using Test
using FiniteVolumeMethod
using LinearAlgebra

# Method of manufactured solutions for the 2D axisymmetric (r-z) diffusion
# operator on an annular domain. Confirms that the assembly converges at
# second order in L∞ as the mesh is refined.
#
# Manufactured solution (vanishes on all four boundaries):
#   T(r, z) = sin(α (r - r_i)) · sin(β z)
# with α = π / (r_o - r_i) and β = π / L.
#
# In cylindrical coordinates (axisymmetric):
#   ∇²T = ∂²T/∂r² + (1/r) ∂T/∂r + ∂²T/∂z²
# For γ ∇²T + Q = 0 to hold with this T:
#   Q(r, z) = γ [(α² + β²) T - (α / r) cos(α (r - r_i)) sin(β z)]

@testset "Cylindrical 2D annular MMS — second-order convergence" begin
    r_inner = 4.75e-3
    r_outer = 4.8e-3
    L = 1.0e-2
    γ = 0.5

    α = π / (r_outer - r_inner)
    β = π / L

    T_exact(r, z) = sin(α * (r - r_inner)) * sin(β * z)
    Q_exact(r, z) = γ * (
        (α^2 + β^2) * T_exact(r, z) -
            (α / r) * cos(α * (r - r_inner)) * sin(β * z)
    )

    function solve_on(nx, ny)
        x_nodes = collect(range(r_inner, r_outer; length = nx + 1))
        y_nodes = collect(range(0.0, L; length = ny + 1))
        mesh = generate_mesh_2d_nonuniform(nx, ny, r_outer - r_inner, L, x_nodes, y_nodes)

        bcs = (
            ParabolicDirichlet(0.0),
            ParabolicDirichlet(0.0),
            ParabolicDirichlet(0.0),
            ParabolicDirichlet(0.0),
        )

        A, b = assemble_system(
            CylindricalDiffusion2D(γ), mesh, bcs;
            source = FunctionSource(Q_exact),
        )
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

    # The asymptotic convergence rate must be at least 1.8 (close to the
    # theoretical 2nd order; allow some slack at the coarsest pair).
    @test rates[end] > 1.8
    @test errors[end] < errors[1] / 30  # 4× refinement should drop ~16x for 2nd order
end
