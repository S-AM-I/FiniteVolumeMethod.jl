# test/v_and_v_lsq_gradient.jl — Weighted least-squares gradient V&V
#
# Verifies `least_squares_gradient` on affine, constant, and quadratic
# fields on both Cartesian and skewed meshes. The LSQ gradient is exact
# for linear fields on arbitrary polyhedral meshes (Jasak 1996 Ch. 4);
# for quadratic fields it is first-order accurate in cell size.

using FiniteVolumeMethod
using LinearAlgebra: norm
using StaticArrays: SVector
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

const lsq_gradient = FiniteVolumeMethod.least_squares_gradient

# Shared skewed-mesh helper (duplicated from v_and_v_over_relaxed.jl
# because `safe_include` isolates each test file in its own module).
function build_skewed_mesh(N::Int, skew::Float64)
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    new_centers = copy(mesh.cell_centers)
    nc = size(new_centers, 2)
    for c in 1:nc
        x = new_centers[1, c]
        y = new_centers[2, c]
        if 0.05 < x < 0.95 && 0.05 < y < 0.95
            new_centers[1, c] = x + skew * sin(3π * x) * sin(2π * y)
            new_centers[2, c] = y + skew * sin(2π * x) * sin(3π * y)
        end
    end
    return FiniteVolumeMethod.UnstructuredFVMMesh{2, Float64}(
        new_centers,
        mesh.cell_volumes,
        mesh.face_cells,
        mesh.face_centers,
        mesh.face_areas,
        mesh.face_normals,
        mesh.face_tags,
        mesh.face_velocity,
        mesh.cell_faces,
    )
end

"""Fill `phi` with the values of scalar function `f(x, y)` at cell
centres and boundary-face centres."""
function fill_phi!(phi, mesh, f)
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        phi.internal[c] = f(x, y)
    end
    for (k, face) in enumerate(phi.boundary_face_indices)
        x = mesh.face_centers[1, face]
        y = mesh.face_centers[2, face]
        phi.boundary[k] = f(x, y)
    end
    return phi
end

@testset "V&V: LSQ gradient is exact on a linear field" begin
    N = 12
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    phi = CollocatedScalarField(:phi, mesh)
    fill_phi!(phi, mesh, (x, y) -> 2.0 * x + 3.0 * y + 0.5)

    grad = lsq_gradient(phi, mesh)
    max_err = 0.0
    for g in grad
        max_err = max(max_err, norm(g - SVector(2.0, 3.0)))
    end
    @test max_err < 1.0e-12
end

@testset "V&V: LSQ gradient is exact on a linear field on a skewed mesh" begin
    # Core LSQ property: exactness on linear fields holds for ARBITRARY
    # polyhedral meshes. Skewed cell centres should not introduce error.
    mesh = build_skewed_mesh(20, 0.05)
    phi = CollocatedScalarField(:phi, mesh)
    fill_phi!(phi, mesh, (x, y) -> -1.7 * x + 4.3 * y - 2.1)

    grad = lsq_gradient(phi, mesh)
    max_err = 0.0
    for g in grad
        max_err = max(max_err, norm(g - SVector(-1.7, 4.3)))
    end
    @test max_err < 1.0e-12
end

@testset "V&V: LSQ gradient of a constant field is zero" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    phi = CollocatedScalarField(:phi, mesh; value = 3.14)
    grad = lsq_gradient(phi, mesh)
    max_err = maximum(norm.(grad))
    @test max_err < 1.0e-14
end

@testset "V&V: LSQ gradient recovers an arbitrary direction exactly" begin
    # Two-point-stencil degenerate case is impossible in 2D (every cell
    # has ≥ 2 faces), but we can still verify that any user-specified
    # linear field is recovered exactly along the stencil directions.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    phi = CollocatedScalarField(:phi, mesh)
    a = SVector(1.234, -5.678)
    fill_phi!(phi, mesh, (x, y) -> a[1] * x + a[2] * y)

    grad = lsq_gradient(phi, mesh)
    for g in grad
        @test norm(g - a) < 1.0e-12
    end
end

"""L²-error of LSQ gradient against the analytic gradient, using only
interior cells (skips the outermost ring where boundary-face offsets
dominate the truncation error)."""
function lsq_gradient_error(N::Int, f::Function, grad_f::Function)
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    phi = CollocatedScalarField(:phi, mesh)
    fill_phi!(phi, mesh, f)

    grad = lsq_gradient(phi, mesh)

    # Restrict to interior cells: (i, j) with 2 ≤ i ≤ N-1 and 2 ≤ j ≤ N-1.
    err_sq = 0.0
    vol = 0.0
    for j in 2:(N - 1), i in 2:(N - 1)
        c = (j - 1) * N + i
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        exact = grad_f(x, y)
        err_sq += mesh.cell_volumes[c] * sum(abs2, grad[c] - exact)
        vol += mesh.cell_volumes[c]
    end
    return sqrt(err_sq / vol)
end

@testset "V&V: LSQ gradient of a quadratic field is exact on a symmetric Cartesian stencil" begin
    # Weighted LSQ with symmetric neighbour offsets is algebraically exact
    # for φ = x² + y² on a uniform Cartesian grid (the O(h²) truncation
    # terms cancel by symmetry at interior cells). This is a useful
    # sanity check that `least_squares_gradient` is not accidentally
    # adding a finite-precision bias.
    for N in (16, 32, 64)
        err = lsq_gradient_error(N, (x, y) -> x^2 + y^2, (x, y) -> SVector(2.0 * x, 2.0 * y))
        @test err < 1.0e-12
    end
end

@testset "V&V: LSQ gradient of a smooth non-symmetric field is ≥ first-order accurate" begin
    # Use a field with no special stencil symmetry so the truncation error
    # does not vanish at interior cells. φ(x, y) = sin(π x) sin(π y);
    # ∇φ = (π cos(π x) sin(π y), π sin(π x) cos(π y)).
    f(x, y) = sin(π * x) * sin(π * y)
    gf(x, y) = SVector(π * cos(π * x) * sin(π * y), π * sin(π * x) * cos(π * y))
    err16 = lsq_gradient_error(16, f, gf)
    err32 = lsq_gradient_error(32, f, gf)
    err64 = lsq_gradient_error(64, f, gf)
    order_a = log2(err16 / err32)
    order_b = log2(err32 / err64)
    @test err16 > err32 > err64 > 0
    # Inverse-distance-squared weights on a uniform Cartesian grid give
    # ≈ 2nd-order convergence on a smooth field; the contract for an
    # LSQ gradient on an arbitrary polyhedral mesh is just first-order.
    @test order_a > 0.9
    @test order_b > 0.9
end

@testset "V&V: LSQ and Green-Gauss agree within 5% on a smooth field" begin
    N = 40
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    phi = CollocatedScalarField(:phi, mesh)
    fill_phi!(phi, mesh, (x, y) -> sin(π * x) * cos(π * y))

    grad_lsq = lsq_gradient(phi, mesh)
    grad_gg = gradient(phi, mesh)

    # Compare on interior cells only (boundary-adjacent cells see larger
    # stencil differences because the two schemes weight boundary
    # contributions differently).
    max_ref = 0.0
    max_diff = 0.0
    for j in 3:(N - 2), i in 3:(N - 2)
        c = (j - 1) * N + i
        max_ref = max(max_ref, norm(grad_gg[c]))
        max_diff = max(max_diff, norm(grad_lsq[c] - grad_gg[c]))
    end
    @test max_ref > 0
    @test max_diff / max_ref < 0.05
end
