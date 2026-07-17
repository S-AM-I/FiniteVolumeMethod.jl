# test/v_and_v_laplacian_mms.jl — Laplacian operator MMS convergence (v3.4)
#
# Verifies that the collocated Laplacian operator `assemble_laplacian!`
# achieves the expected second-order spatial convergence on a Cartesian
# mesh. This is a UNIT test of the discretization, independent of the
# SIMPLE pressure-velocity-coupling loop — the latter has its own
# convergence-rate limitations (Stage 3d over-relaxed correction).
#
# Manufactured solution:
#   φ(x, y) = sin(π x) · sin(π y)     on [0, 1]²
#   ∇²φ = -2π² · sin(π x) · sin(π y)
#
# For Γ = 1 and forcing f(x, y) = -∇²φ, solving `div(Γ ∇φ) = -f`
# recovers φ. Discretization error ||φ_num - φ_exact||_∞ should decrease
# as O(h²) on a uniform refinement sequence.

using FiniteVolumeMethod
using FiniteVolumeMethod: CollocatedEquation, assemble_laplacian!, to_linear_problem
using FiniteVolumeMethod.Parabolic: DirichletBC
using LinearSolve
using StaticArrays: SVector
using LinearAlgebra: norm
using Test

include("TestHelpers.jl")

# Exact solution and forcing
phi_exact(x, y) = sin(π * x) * sin(π * y)
f_forcing(x, y) = 2π^2 * sin(π * x) * sin(π * y)

function solve_laplacian_mms(N::Int)
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
        :top => DirichletBC(0.0),
    )
    eq = CollocatedEquation(mesh)
    assemble_laplacian!(eq, 1.0, mesh, bcs)

    # Add the forcing source term: we are solving `-∇²φ = f` ↔
    # the Laplacian assembly computed `div(Γ ∇φ)` on the LHS (with Γ = 1).
    # The convention here: eq.A · φ corresponds to `∫_V div(grad φ) dV`
    # ≈ V_c · ∇²φ. For `-∇²φ = f`, we need `eq.A · φ = V_c · (-f) = -V_c · f`.
    # Actually `assemble_laplacian!` assembles `A φ = b` where `A φ ≈ -div(Γ ∇φ) V_c`
    # (Laplacian is SPD on the cell-centered stencil). So for `-∇²φ = f`,
    # the RHS is `V_c · f`.
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        eq.b[c] += mesh.cell_volumes[c] * f_forcing(x, y)
    end

    lp = to_linear_problem(eq)
    sol = solve(lp)
    phi_num = sol.u

    # L∞ error
    err_inf = 0.0
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        err_inf = max(err_inf, abs(phi_num[c] - phi_exact(x, y)))
    end
    # L2 error (volume-weighted)
    err_sq = 0.0
    vol_tot = 0.0
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        err_sq += mesh.cell_volumes[c] * (phi_num[c] - phi_exact(x, y))^2
        vol_tot += mesh.cell_volumes[c]
    end
    err_L2 = sqrt(err_sq / vol_tot)

    return err_inf, err_L2
end

@testset "V&V: Laplacian operator MMS — spatial order of accuracy" begin
    Ns = [10, 20, 40, 80]
    errs_inf = Float64[]
    errs_L2 = Float64[]
    for N in Ns
        einf, eL2 = solve_laplacian_mms(N)
        push!(errs_inf, einf)
        push!(errs_L2, eL2)
    end

    # Order of accuracy: p = log(err_i / err_{i+1}) / log(h_i / h_{i+1})
    # Uniform refinement h_{i+1} = h_i / 2 → log(2) in denominator.
    orders_L2 = [log2(errs_L2[i] / errs_L2[i + 1]) for i in 1:(length(Ns) - 1)]
    orders_inf = [log2(errs_inf[i] / errs_inf[i + 1]) for i in 1:(length(Ns) - 1)]

    # Second-order Laplacian on Cartesian mesh: expect p ≈ 2.
    # We verify at the FINEST transition (80-cell vs 40-cell) where
    # the asymptotic rate dominates over boundary/discretization noise.
    @test orders_L2[end] > 1.8
    @test orders_L2[end] < 2.2
    @test orders_inf[end] > 1.7  # L∞ a bit noisier

    # Absolute errors should also shrink monotonically.
    @test all(errs_L2[i] > errs_L2[i + 1] for i in 1:(length(Ns) - 1))
    @test all(errs_inf[i] > errs_inf[i + 1] for i in 1:(length(Ns) - 1))

    # Expected ballpark on the finest grid: O(h²) = O((1/80)²) = 1.6e-4.
    @test errs_L2[end] < 1.0e-3
end
