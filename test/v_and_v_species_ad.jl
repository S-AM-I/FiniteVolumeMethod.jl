# test/v_and_v_species_ad.jl — Species advection-diffusion V&V (v3.17)
#
# Verifies `assemble_species!` against the closed-form 1D steady
# advection-diffusion solution for a scalar Y with Dirichlet BCs:
#
#   u ∂Y/∂x − D ∂²Y/∂x² = 0,    Y(0) = Y_L,  Y(L) = Y_R
#
# Analytical solution (Pe = u·L/D):
#
#   Y(x) = Y_L + (Y_R − Y_L) · (exp(Pe · x / L) − 1) / (exp(Pe) − 1)
#
# At moderate Pe the steady profile is a curved transition from
# Y_L at the inflow to Y_R at the outflow; as Pe → ∞ it degenerates
# into a boundary layer near x = L. This is the canonical
# verification problem for scalar convection-diffusion operators —
# the same kinematic skeleton that drives the species transport
# step in `solve_species!`. Evidence for promoting `combustion`
# from `experimental`/`smoke_tested` to `provisional`/
# `convergence_verified` on the transport side.

using FiniteVolumeMethod
using FiniteVolumeMethod: CONV_UPWIND, CollocatedEquation, assemble_convection!, assemble_species!, face_normal_area, solve_species!, to_linear_problem
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC
using LinearSolve
using StaticArrays
using Test

include("TestHelpers.jl")

const AD_L = 1.0
const AD_LY = 0.2
const AD_U = 1.0
const AD_D = 0.5
const AD_PE = AD_U * AD_L / AD_D  # = 2.0
const AD_YL = 1.0
const AD_YR = 0.0

function Y_exact_1d(x::Float64)
    num = exp(AD_PE * x / AD_L) - 1
    den = exp(AD_PE) - 1
    return AD_YL + (AD_YR - AD_YL) * num / den
end

function solve_species_steady(Nx::Int, Ny::Int)
    mesh = build_cartesian_unstructured_mesh(Nx, Ny, AD_L, AD_LY)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    Y = CollocatedScalarField(:Y, mesh; value = 0.5)

    # Uniform velocity u = (U, 0) ⇒ phi = U · S_f[1].
    phi = FaceFluxField(:phi, mesh)
    for f in 1:nf
        phi.values[f] = AD_U * face_normal_area(mesh, f)[1]
    end

    # Dirichlet inflow/outflow, Neumann top/bottom for strict 1D.
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(AD_YL),
        :right => DirichletBC(AD_YR),
        :bottom => NeumannBC(0.0),
        :top => NeumannBC(0.0),
    )

    eq = CollocatedEquation(mesh)
    # Steady state: dt = nothing ⇒ no temporal term ⇒ convection +
    # diffusion balance = Dirichlet source.
    assemble_species!(eq, Y, phi, AD_D, mesh, bcs; dt = nothing)

    lp = to_linear_problem(eq)
    sol = LinearSolve.solve(lp, LUFactorization())
    for c in 1:nc
        Y.internal[c] = sol.u[c]
    end
    return mesh, Y
end

function ad_l2_error(Nx::Int, Ny::Int)
    mesh, Y = solve_species_steady(Nx, Ny)
    nc = length(mesh.cell_volumes)

    err_sq = 0.0
    vol = 0.0
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        # Interior band to avoid boundary-layer-at-x=L rasterization.
        if 0.1 < x < 0.9
            Y_e = Y_exact_1d(x)
            err_sq += mesh.cell_volumes[c] * (Y.internal[c] - Y_e)^2
            vol += mesh.cell_volumes[c]
        end
    end
    return sqrt(err_sq / vol)
end

@testset "V&V: Species AD — boundary values and monotonicity" begin
    # Coarse 40×8 run: endpoint values and monotone shape.
    mesh, Y = solve_species_steady(40, 8)
    nc = length(mesh.cell_volumes)

    # Sort by x to extract the 1D profile.
    xs = [mesh.cell_centers[1, c] for c in 1:nc]
    order = sortperm(xs)
    unique_x = Float64[]
    col_avg = Float64[]
    last_x = -Inf
    col_acc = 0.0
    col_n = 0
    for c in order
        x = xs[c]
        if x - last_x > 1.0e-9
            if col_n > 0
                push!(unique_x, last_x)
                push!(col_avg, col_acc / col_n)
            end
            last_x = x
            col_acc = Y.internal[c]
            col_n = 1
        else
            col_acc += Y.internal[c]
            col_n += 1
        end
    end
    push!(unique_x, last_x)
    push!(col_avg, col_acc / col_n)

    # Monotone decay from Y_L to Y_R.
    @test all(diff(col_avg) .<= 1.0e-10)

    # Left / right columns bracket the Dirichlet values.
    @test col_avg[1] > 0.85 * AD_YL
    @test col_avg[end] < 0.15 * AD_YL

    # No overshoot: Y ∈ [Y_R, Y_L] at every cell (upwind + Laplacian
    # on a Cartesian mesh preserves the maximum principle at Pe = 2).
    @test all(x -> AD_YR - 1.0e-12 <= x <= AD_YL + 1.0e-12, Y.internal)
end

@testset "V&V: Species AD — y-direction independence" begin
    # Neumann top/bottom ⇒ strict 1D solution.
    mesh, Y = solve_species_steady(20, 10)
    nc = length(mesh.cell_volumes)

    buckets = Dict{Float64, Vector{Float64}}()
    for c in 1:nc
        x = round(mesh.cell_centers[1, c]; digits = 10)
        push!(get!(buckets, x, Float64[]), Y.internal[c])
    end
    max_spread = 0.0
    for (_, vals) in buckets
        if length(vals) >= 2
            max_spread = max(max_spread, maximum(vals) - minimum(vals))
        end
    end
    @test max_spread < 1.0e-10
end

@testset "V&V: Species AD — first-order upwind convergence" begin
    errs = [ad_l2_error(N, 4) for N in (20, 40, 80)]
    orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]

    # The default scheme in `assemble_convection!` is CONV_UPWIND
    # (formally first-order in smooth regions). At Pe = 2 the
    # convection term dominates the discretization error; the
    # observed rate is ≈1.0 (classical first-order upwind on
    # exp(2x), which has non-trivial curvature). The diffusive
    # Laplacian correction is second-order but sub-dominant here.
    for p in orders
        @test 0.8 < p < 1.5
    end

    # Monotone decrease.
    @test all(errs[i] > errs[i + 1] for i in 1:(length(errs) - 1))

    # Finest-grid interior-band L² error < 5 × 10⁻³ (≈0.5 % of the
    # profile swing 0 → 1, consistent with the observed O(h) rate
    # at N = 80).
    @test errs[end] < 5.0e-3
end
