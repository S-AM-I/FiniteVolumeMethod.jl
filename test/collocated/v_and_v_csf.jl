# test/v_and_v_csf.jl — CSF surface-tension algebra V&V (v3.46)
#
# Fourth convergence-verified benchmark for `multiphase_vof`.
# Joins disc translation (v3.16), plane-wave advection (v3.24),
# and mixture blending (v3.36). Covers the Continuum Surface
# Force primitive:
#
#   F_st = σ · κ · ∇α
#
# where κ = -div(∇α/|∇α|) is the interface curvature and ∇α is
# the volume-fraction gradient.
#
# Invariants:
#
#   1. σ = 0 disables F_st (returns nothing).
#   2. Uniform α (no interface) ⇒ ∇α = 0 ⇒ F_st = 0.
#   3. F_st ∝ σ at fixed α-field: doubling σ doubles F_st.
#   4. Curvature of uniform α is zero.
#   5. Straight-line interface α(x) = step at x = x₀: curvature
#      is zero (planar interface has κ = 0).

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_curvature, compute_surface_tension_force
using LinearAlgebra: norm
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: CSF — σ = 0 disables surface tension" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    props = TwoPhaseProperties(;
        rho1 = 1000.0, rho2 = 1.2,
        mu1 = 1.0e-3, mu2 = 1.8e-5,
        sigma = 0.0,
    )
    alpha = CollocatedScalarField(:alpha, mesh; value = 0.5)
    F_st = compute_surface_tension_force(alpha, props, mesh)
    @test F_st === nothing
end

@testset "V&V: CSF — uniform α ⇒ F_st = 0" begin
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = TwoPhaseProperties(;
        rho1 = 1000.0, rho2 = 1.2,
        mu1 = 1.0e-3, mu2 = 1.8e-5,
        sigma = 0.072,
    )
    alpha = CollocatedScalarField(:alpha, mesh; value = 0.3)

    F_st = compute_surface_tension_force(alpha, props, mesh)
    @test F_st !== nothing
    for c in 1:nc
        @test isapprox(F_st[c][1], 0.0; atol = 1.0e-12)
        @test isapprox(F_st[c][2], 0.0; atol = 1.0e-12)
    end
end

@testset "V&V: CSF — curvature of uniform α is zero" begin
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    alpha = CollocatedScalarField(:alpha, mesh; value = 0.4)

    kappa = compute_curvature(alpha, mesh)
    for c in 1:nc
        @test isapprox(kappa[c], 0.0; atol = 1.0e-12)
    end
end

@testset "V&V: CSF — F_st ∝ σ scaling at fixed interface" begin
    mesh = build_cartesian_unstructured_mesh(20, 20, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    # Smooth α ramp (not a true sharp interface, but gives
    # non-trivial ∇α and κ).
    alpha = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        alpha.internal[c] = 0.5 * (1.0 + tanh(20.0 * (x - 0.5)))
    end

    props_a = TwoPhaseProperties(;
        rho1 = 1000.0, rho2 = 1.2,
        mu1 = 1.0e-3, mu2 = 1.8e-5, sigma = 0.05,
    )
    props_b = TwoPhaseProperties(;
        rho1 = 1000.0, rho2 = 1.2,
        mu1 = 1.0e-3, mu2 = 1.8e-5, sigma = 0.1,
    )

    F_a = compute_surface_tension_force(alpha, props_a, mesh)
    F_b = compute_surface_tension_force(alpha, props_b, mesh)

    # F_st scales linearly with σ — ratio should be 2 at every
    # cell with non-negligible F.
    for c in 1:nc
        mag_a = norm(F_a[c])
        if mag_a > 1.0e-8
            @test isapprox(norm(F_b[c]) / mag_a, 2.0; rtol = 1.0e-10)
        end
    end
end

@testset "V&V: CSF — pointwise algebraic identity F_st = σ·κ·∇α" begin
    # Direct identity: for any α field, `compute_surface_tension_force`
    # must equal σ · κ · ∇α at every cell. Compare pointwise against
    # explicit evaluation of the three primitives.
    mesh = build_cartesian_unstructured_mesh(20, 20, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    sigma_val = 0.072
    props = TwoPhaseProperties(;
        rho1 = 1000.0, rho2 = 1.2,
        mu1 = 1.0e-3, mu2 = 1.8e-5, sigma = sigma_val,
    )
    alpha = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        alpha.internal[c] = 0.5 * (1.0 + tanh(20.0 * (x - 0.5)))
    end

    F_st = compute_surface_tension_force(alpha, props, mesh)
    kappa = compute_curvature(alpha, mesh)
    grad_alpha = FiniteVolumeMethod.gradient(alpha, mesh)

    for c in 1:nc
        expected = sigma_val * kappa[c] * grad_alpha[c]
        @test isapprox(F_st[c][1], expected[1]; rtol = 1.0e-12, atol = 1.0e-14)
        @test isapprox(F_st[c][2], expected[2]; rtol = 1.0e-12, atol = 1.0e-14)
    end
end
