# test/v_and_v_partitioned_fsi.jl — Partitioned FSI loop V&V (Wave 3)
#
# Verifies the outer Dirichlet-Neumann loop in `src/fsi/partitioned.jl`
# against a 1-DOF spring-damper problem whose exact fixed point is
# analytically available.
#
# Mock fluid  : F_fluid(x) = k · x + F_ext                (linear load).
# Mock solid  : x = F / k_s                               (static spring).
#
# Fixed point of the coupled system: x = F_ext / (k_s − k).
#
# Four gates:
#   1. Strictly contractive coupling (k/k_s < 1) ⇒ partitioned solve
#      with Aitken converges to the analytical x within 1e-8.
#   2. Zero coupling strength (k = 0) ⇒ convergence in a single
#      outer iteration.
#   3. Divergent case (k/k_s > 1): fixed ω = 1 (no Aitken) oscillates
#      / diverges; Aitken stabilizes it and brings it to the analytical
#      fixed point.
#   4. Residual history monotonically decreases (non-increasing) under
#      Aitken for the contractive case.

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: AitkenRelaxation, FSIInterface
using Test
using LinearAlgebra: norm
using StaticArrays: SVector

const solve_partitioned_fsi = FiniteVolumeMethod.solve_partitioned_fsi
const update_aitken_omega! = FiniteVolumeMethod.update_aitken_omega!

# ---------------------------------------------------------------------
# Mock solver builders. Each pair (fluid, solid) captures a scalar 1-DOF
# problem inside a closure so the same outer loop can drive arbitrary
# coupling stiffnesses.
# ---------------------------------------------------------------------

function make_mock_fluid(; k::Float64, F_ext::Float64)
    # Given interface displacement (the fluid sees x as the Dirichlet
    # mesh-motion BC), return the scalar traction k·x + F_ext.
    return function fluid_solver(displacement::AbstractVector{SVector{1, Float64}})
        x = displacement[1][1]
        traction = SVector{1, Float64}(k * x + F_ext)
        return ([traction], (traction = k * x + F_ext,))
    end
end

function make_mock_solid(; k_s::Float64)
    # Given scalar Neumann traction F, return the static-spring
    # displacement x = F / k_s.
    return function solid_solver(traction::AbstractVector{SVector{1, Float64}})
        F = traction[1][1]
        x = F / k_s
        return ([SVector{1, Float64}(x)], (x = x,))
    end
end

function single_face_interface()
    return FSIInterface{1, Float64}([1], [1])
end

@testset "V&V: Partitioned FSI — contractive case converges to analytic x" begin
    k = 0.4
    k_s = 1.0
    F_ext = 2.0
    x_exact = F_ext / (k_s - k)   # 2 / 0.6 ≈ 3.333…

    fluid = make_mock_fluid(; k = k, F_ext = F_ext)
    solid = make_mock_solid(; k_s = k_s)
    interface = single_face_interface()
    relax = AitkenRelaxation(; omega0 = 0.5, omega_min = 1.0e-3, omega_max = 2.0)

    result = solve_partitioned_fsi(
        fluid, solid, interface;
        max_outer = 100, tol = 1.0e-10, relaxation = relax,
    )

    @test result.converged
    @test isapprox(result.displacement[1][1], x_exact; atol = 1.0e-8)
end

@testset "V&V: Partitioned FSI — k=0 ⇒ converges in one iteration" begin
    F_ext = 1.5
    k_s = 2.0
    x_exact = F_ext / k_s

    fluid = make_mock_fluid(; k = 0.0, F_ext = F_ext)
    solid = make_mock_solid(; k_s = k_s)
    interface = single_face_interface()
    relax = AitkenRelaxation(; omega0 = 1.0, omega_min = 1.0e-3, omega_max = 1.0)

    result = solve_partitioned_fsi(
        fluid, solid, interface;
        max_outer = 50, tol = 1.0e-10, relaxation = relax,
    )

    @test result.converged
    # First iterate jumps from 0 → x_exact because traction is
    # independent of x. Update Δu = |ω|·|δ| = 1·x_exact, which is
    # > tol on iter 1. The second iterate has δ = 0, so update = 0 and
    # the loop exits at iter 2.
    @test result.iterations ≤ 2
    @test isapprox(result.displacement[1][1], x_exact; atol = 1.0e-12)
end

# ---------------------------------------------------------------------
# Fixed-ω driver (no Aitken). Used to show that a divergent linear
# coupling (k/k_s > 1) oscillates at ω = 1, and that Aitken stabilizes
# the same problem. We deliberately re-implement a minimal fixed-ω
# iterator here so the comparison doesn't depend on the outer-loop
# code under test.
# ---------------------------------------------------------------------

function fixed_omega_iterate(
        k::Float64, k_s::Float64, F_ext::Float64;
        ω::Float64, max_iter::Int = 50
    )
    x = 0.0
    history = Float64[]
    for _ in 1:max_iter
        x_tilde = (k * x + F_ext) / k_s
        δ = x_tilde - x
        x = x + ω * δ
        push!(history, abs(δ))
    end
    return x, history
end

@testset "V&V: Partitioned FSI — Aitken stabilizes a divergent case" begin
    # k/k_s = 1.5 ⇒ bare fixed-point iterate diverges.
    k = 1.5
    k_s = 1.0
    F_ext = 0.5
    x_exact = F_ext / (k_s - k)   # = 0.5 / -0.5 = -1.0

    # (a) Fixed ω = 1 diverges / oscillates violently.
    x_no_relax, hist_no_relax = fixed_omega_iterate(k, k_s, F_ext; ω = 1.0, max_iter = 20)
    @test !isapprox(x_no_relax, x_exact; atol = 1.0e-3)
    @test maximum(hist_no_relax) > 1.0       # residuals explode

    # (b) Aitken + partitioned driver converges. For k/k_s > 1 the
    # optimal relaxation is negative (ω ≈ -2 for this pair), so the
    # clamp must admit negative ω.
    fluid = make_mock_fluid(; k = k, F_ext = F_ext)
    solid = make_mock_solid(; k_s = k_s)
    interface = single_face_interface()
    relax = AitkenRelaxation(; omega0 = 0.5, omega_min = -5.0, omega_max = 5.0)

    result = solve_partitioned_fsi(
        fluid, solid, interface;
        max_outer = 200, tol = 1.0e-10, relaxation = relax,
    )

    @test result.converged
    @test isapprox(result.displacement[1][1], x_exact; atol = 1.0e-6)
end

@testset "V&V: Partitioned FSI — residual history non-increasing in contractive case" begin
    k = 0.5
    k_s = 1.0
    F_ext = 1.0

    fluid = make_mock_fluid(; k = k, F_ext = F_ext)
    solid = make_mock_solid(; k_s = k_s)
    interface = single_face_interface()
    relax = AitkenRelaxation(; omega0 = 0.5, omega_min = 1.0e-4, omega_max = 2.0)

    result = solve_partitioned_fsi(
        fluid, solid, interface;
        max_outer = 50, tol = 1.0e-10, relaxation = relax,
    )
    @test result.converged

    # After the seed iterate the residual should shrink monotonically
    # (we allow equality — Aitken drives it to zero on step 2 for a
    # scalar linear problem).
    hist = result.residual_history
    @test length(hist) ≥ 1
    for i in 2:length(hist)
        @test hist[i] ≤ hist[i - 1] + 1.0e-12
    end
end
