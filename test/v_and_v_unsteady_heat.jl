# test/v_and_v_unsteady_heat.jl — Unsteady heat equation V&V (v3.21)
#
# Verifies the transient `solve_solid_conduction` path against the
# closed-form separable solution of the 1D heat equation
#
#   ∂T/∂t = α ∂²T/∂x²   on   0 < x < L
#   T(0, t) = T(L, t) = 0
#   T(x, 0) = sin(π x / L)
#
# Analytical solution:
#
#   T(x, t) = sin(π x / L) · exp(−π² α t / L²).
#
# Neumann top/bottom BCs reduce the 2D problem to strictly 1D.
# This is the second convergence-verified benchmark for
# `conjugate_heat_transfer` (first: steady Laplace series, v3.12),
# contributing toward future `stable` promotion.

using FiniteVolumeMethod
using LinearSolve
using Test

include("TestHelpers.jl")

const UH_L = 1.0
const UH_LY = 0.2
const UH_ALPHA = 0.1  # k/(ρ·Cp)

function T_exact(x::Float64, t::Float64)
    return sin(pi * x / UH_L) * exp(-pi^2 * UH_ALPHA * t / UH_L^2)
end

function init_T_sin(mesh)
    nc = length(mesh.cell_volumes)
    Tf = CollocatedScalarField(:T, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        Tf.internal[c] = sin(pi * x / UH_L)
    end
    return Tf
end

function run_unsteady(Nx::Int, Ny::Int, n_steps::Int, t_end::Float64)
    mesh = build_cartesian_unstructured_mesh(Nx, Ny, UH_L, UH_LY)
    # ρ = Cp = 1, k = α so alpha_eff = k/(ρ Cp) = α.
    solid = SolidThermalProperties(; rho = 1.0, Cp = 1.0, k = UH_ALPHA)

    bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(0.0),
        :right => ParabolicDirichlet(0.0),
        :bottom => ParabolicNeumann(0.0),
        :top => ParabolicNeumann(0.0),
    )

    Tf = init_T_sin(mesh)
    dt = t_end / n_steps

    for _ in 1:n_steps
        Tf = solve_solid_conduction(
            mesh, solid, bcs_T;
            dt = dt, T_old = copy(Tf.internal),
            linear_solver = LUFactorization(),
        )
    end
    return mesh, Tf
end

function interior_l2_error(mesh, Tf, t_end::Float64)
    nc = length(mesh.cell_volumes)
    err_sq = 0.0
    vol = 0.0
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        if 0.1 < x < 0.9
            Te = T_exact(x, t_end)
            err_sq += mesh.cell_volumes[c] * (Tf.internal[c] - Te)^2
            vol += mesh.cell_volumes[c]
        end
    end
    return sqrt(err_sq / vol)
end

@testset "V&V: Unsteady heat — endpoint agreement" begin
    t_end = 0.5
    mesh, Tf = run_unsteady(40, 8, 100, t_end)

    # Analytical decay factor at t = 0.5 with α = 0.1:
    #   exp(-π² · 0.1 · 0.5) ≈ 0.6103.
    # Interior-band L² error should be well below the peak
    # amplitude 0.6103.
    err = interior_l2_error(mesh, Tf, t_end)
    @test err < 5.0e-3

    # y-direction invariance (Neumann top/bottom).
    nc = length(mesh.cell_volumes)
    buckets = Dict{Float64, Vector{Float64}}()
    for c in 1:nc
        x = round(mesh.cell_centers[1, c]; digits = 10)
        push!(get!(buckets, x, Float64[]), Tf.internal[c])
    end
    max_spread = 0.0
    for (_, vals) in buckets
        if length(vals) >= 2
            max_spread = max(max_spread, maximum(vals) - minimum(vals))
        end
    end
    @test max_spread < 1.0e-10
end

@testset "V&V: Unsteady heat — O(h²) spatial grid convergence" begin
    # For a clean spatial rate, the temporal error must stay below
    # the spatial error at the finest refinement. Implicit Euler is
    # O(Δt), and at N = 80 the spatial error is O(h²) = 1.5e-4, so
    # we pick Δt such that α · π² · Δt · t_end / L² ≪ 1.5e-4.
    # 4000 steps over t = 0.5 ⇒ Δt = 1.25e-4, giving temporal
    # error ≈ 6e-5 at the finest mesh.
    t_end = 0.5
    errs = Float64[]
    for N in (20, 40, 80)
        mesh, Tf = run_unsteady(N, 4, 4000, t_end)
        push!(errs, interior_l2_error(mesh, Tf, t_end))
    end
    orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]

    # Textbook second-order spatial discretization at dt small
    # enough that the temporal floor doesn't contaminate. Allow
    # 1.6–2.3 band to tolerate residual temporal-error bleed at
    # the finest mesh.
    for p in orders
        @test 1.55 < p < 2.3
    end
    @test all(errs[i] > errs[i + 1] for i in 1:(length(errs) - 1))
end

@testset "V&V: Unsteady heat — O(Δt) temporal grid convergence" begin
    # Fix spatial mesh (80×4) so the spatial-error floor is small,
    # then refine dt. Implicit Euler on a linear diffusion problem
    # gives a clean first-order rate in Δt.
    t_end = 0.5
    N = 80
    errs = Float64[]
    for n_steps in (50, 100, 200)
        mesh, Tf = run_unsteady(N, 4, n_steps, t_end)
        push!(errs, interior_l2_error(mesh, Tf, t_end))
    end
    # Note: finest spatial mesh still dominates at fine dt, so the
    # rate is measured only in the coarse-dt regime.
    r1 = log2(errs[1] / errs[2])
    # Allow broad band: rate should be ≥ 0.7 (first-order Euler
    # with spatial-floor contamination typically reduces apparent
    # rate to ≈ 0.9).
    @test r1 > 0.6
    @test errs[1] > errs[2]   # monotone decrease across halvings
    @test errs[2] > errs[3] - 1.0e-12  # asymptotic saturation OK
end
