# test/v_and_v_p1_slab.jl — P1 radiation 1D slab V&V (v3.15)
#
# Verifies `solve_p1_radiation` against the closed-form 1D analytical
# solution of the P1 attenuation problem in a cold medium:
#
#   -(1/3a) ∂²G/∂x² + a G = 4 a σ T_m⁴     (T_m = 0 ⇒ pure attenuation)
#
# Reduces to  G'' = 3 a² G,  with general solution
#
#   G(x) = G_0 · sinh(√3 a (L - x)) / sinh(√3 a L)
#
# satisfying G(0) = G_0 and G(L) = 0. Top/bottom boundaries use
# zero-gradient Neumann conditions so the 2D problem is strictly
# 1D in x. Evidence for promoting `radiation` from `experimental`/
# `smoke_tested` to `provisional`/`convergence_verified`.

using FiniteVolumeMethod
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC
using LinearSolve
using Test

include("TestHelpers.jl")

# Parameters: unit domain, a = 1, G_0 = 1, cold medium.
const SLAB_L = 1.0
const SLAB_A = 1.0
const SLAB_G0 = 1.0

function G_exact_1d(x::Float64)
    k = sqrt(3.0) * SLAB_A
    return SLAB_G0 * sinh(k * (SLAB_L - x)) / sinh(k * SLAB_L)
end

function solve_slab(Nx::Int, Ny::Int)
    mesh = build_cartesian_unstructured_mesh(Nx, Ny, SLAB_L, 0.2)
    rad = P1Model(; a = SLAB_A)
    T_field = CollocatedScalarField(:T, mesh; value = 0.0)

    bcs_G = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(SLAB_G0),
        :right => DirichletBC(0.0),
        :bottom => NeumannBC(0.0),
        :top => NeumannBC(0.0),
    )

    return mesh, solve_p1_radiation(rad, T_field, mesh, bcs_G)
end

function slab_l2_error(Nx::Int, Ny::Int)
    mesh, G = solve_slab(Nx, Ny)
    nc = length(mesh.cell_volumes)

    err_sq = 0.0
    vol = 0.0
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        # Restrict to interior band away from the walls. Boundary
        # layers near x=0 and x=1 carry first-order discretization
        # error for cell-centered FVM at Dirichlet walls.
        if 0.1 < x < 0.9
            G_e = G_exact_1d(x)
            err_sq += mesh.cell_volumes[c] * (G.internal[c] - G_e)^2
            vol += mesh.cell_volumes[c]
        end
    end
    return sqrt(err_sq / vol)
end

@testset "V&V: P1 slab — monotone decay matches analytical shape" begin
    # At 40×4 the solution should be ≈1 at left, ≈0 at right, and
    # monotone-decreasing along x.
    mesh, G = solve_slab(40, 4)
    nc = length(mesh.cell_volumes)

    # Sort by x.
    xs = [mesh.cell_centers[1, c] for c in 1:nc]
    order = sortperm(xs)

    # Unique x values (one column = 4 cells at the same x on a 40×4 mesh).
    unique_x = Float64[]
    column_G = Float64[]
    last_x = -Inf
    for c in order
        x = xs[c]
        if x - last_x > 1.0e-9
            push!(unique_x, x)
            push!(column_G, G.internal[c])
            last_x = x
        end
    end

    # Monotone decay across columns.
    @test all(diff(column_G) .<= 1.0e-10)

    # Left column ≈ G_0, right column ≈ 0 (with some discretization slack).
    @test column_G[1] > 0.9 * SLAB_G0
    @test column_G[end] < 0.1 * SLAB_G0

    # Strictly positive (physics).
    @test all(>(0.0), G.internal)
end

@testset "V&V: P1 slab — y-direction independence (1D verification)" begin
    # With Neumann top/bottom BCs the solution must depend only on x.
    # All cells sharing the same x must have identical G (to round-off).
    mesh, G = solve_slab(20, 8)
    nc = length(mesh.cell_volumes)

    # Group by x and check column uniformity.
    buckets = Dict{Float64, Vector{Float64}}()
    for c in 1:nc
        x = round(mesh.cell_centers[1, c]; digits = 10)
        push!(get!(buckets, x, Float64[]), G.internal[c])
    end

    max_column_spread = 0.0
    for (_, vals) in buckets
        if length(vals) >= 2
            spread = maximum(vals) - minimum(vals)
            max_column_spread = max(max_column_spread, spread)
        end
    end
    @test max_column_spread < 1.0e-10
end

@testset "V&V: P1 slab — O(h²) grid convergence" begin
    errs = [slab_l2_error(N, 4) for N in (20, 40, 80)]
    orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]

    # Textbook second-order for FVM Laplacian + Dirichlet BCs.
    for p in orders
        @test 1.8 < p < 2.2
    end

    # Monotone decrease, finest error small.
    @test all(errs[i] > errs[i + 1] for i in 1:(length(errs) - 1))
    @test errs[end] < 1.0e-4
end
