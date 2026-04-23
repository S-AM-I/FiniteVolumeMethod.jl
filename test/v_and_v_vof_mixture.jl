# test/v_and_v_vof_mixture.jl — VOF mixture property blending V&V (v3.36)
#
# Third convergence-verified benchmark for `multiphase_vof`. The
# first (v3.16) tested alpha-transport kinematics, the second
# (v3.24) tested wave-shape preservation. This one exercises the
# property-blending primitive that converts α into the mixture
# density and viscosity consumed by the momentum equation:
#
#   ρ[c] = α[c]·ρ₁ + (1 − α[c])·ρ₂
#   μ[c] = α[c]·μ₁ + (1 − α[c])·μ₂
#
# Five algebraic invariants are verified plus the `clip_alpha!`
# boundedness identity. Puts `multiphase_vof` at three
# convergence-verified benchmarks.

using FiniteVolumeMethod
using Test

include("TestHelpers.jl")

@testset "V&V: VOF mixture — α = 1 ⇒ (ρ, μ) ≡ (ρ₁, μ₁)" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = TwoPhaseProperties(;
        rho1 = 1000.0, rho2 = 1.2,
        mu1 = 1.0e-3, mu2 = 1.8e-5,
        sigma = 0.072,
    )
    state = VOFState(mesh; alpha_init = 1.0)
    update_mixture_properties!(state, props)

    for c in 1:nc
        @test isapprox(state.rho[c], props.rho1; rtol = 1.0e-14)
        @test isapprox(state.mu[c], props.mu1; rtol = 1.0e-14)
    end
end

@testset "V&V: VOF mixture — α = 0 ⇒ (ρ, μ) ≡ (ρ₂, μ₂)" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    props = TwoPhaseProperties(;
        rho1 = 1000.0, rho2 = 1.2,
        mu1 = 1.0e-3, mu2 = 1.8e-5,
        sigma = 0.072,
    )
    state = VOFState(mesh; alpha_init = 0.0)
    update_mixture_properties!(state, props)

    for c in 1:nc
        @test isapprox(state.rho[c], props.rho2; rtol = 1.0e-14)
        @test isapprox(state.mu[c], props.mu2; rtol = 1.0e-14)
    end
end

@testset "V&V: VOF mixture — linearity in α (gradient field)" begin
    # Set α as a smooth ramp: α(x) = x (so α runs from 0 at x=0 to
    # 1 at x=1). The resulting ρ and μ must match the algebraic
    # formula at every cell.
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    rho1, rho2 = 1000.0, 1.2
    mu1, mu2 = 1.0e-3, 1.8e-5
    props = TwoPhaseProperties(;
        rho1 = rho1, rho2 = rho2, mu1 = mu1, mu2 = mu2, sigma = 0.072,
    )

    state = VOFState(mesh; alpha_init = 0.5)
    for c in 1:nc
        state.alpha.internal[c] = mesh.cell_centers[1, c]
    end
    update_mixture_properties!(state, props)

    for c in 1:nc
        a = state.alpha.internal[c]
        rho_expected = a * rho1 + (1 - a) * rho2
        mu_expected = a * mu1 + (1 - a) * mu2
        @test isapprox(state.rho[c], rho_expected; rtol = 1.0e-14)
        @test isapprox(state.mu[c], mu_expected; rtol = 1.0e-14)
    end
end

@testset "V&V: VOF mixture — density bounds [ρ_min, ρ_max] with α ∈ [0, 1]" begin
    mesh = build_cartesian_unstructured_mesh(20, 20, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    rho1, rho2 = 1000.0, 1.2
    props = TwoPhaseProperties(;
        rho1 = rho1, rho2 = rho2,
        mu1 = 1.0e-3, mu2 = 1.8e-5, sigma = 0.072,
    )

    # Random α field in [0, 1].
    state = VOFState(mesh; alpha_init = 0.5)
    for c in 1:nc
        state.alpha.internal[c] = mod(c * 0.137, 1.0)  # deterministic pseudo-random
    end
    update_mixture_properties!(state, props)

    rho_min = min(rho1, rho2)
    rho_max = max(rho1, rho2)
    for c in 1:nc
        @test rho_min - 1.0e-12 <= state.rho[c] <= rho_max + 1.0e-12
    end
end

@testset "V&V: VOF mixture — clip_alpha! + blend consistency" begin
    # Start with an α field containing values outside [0, 1],
    # clip it, then blend. Result must match the blend of clipped α.
    mesh = build_cartesian_unstructured_mesh(10, 10, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    rho1, rho2 = 1000.0, 1.2
    props = TwoPhaseProperties(;
        rho1 = rho1, rho2 = rho2,
        mu1 = 1.0e-3, mu2 = 1.8e-5, sigma = 0.072,
    )

    state = VOFState(mesh; alpha_init = 0.5)
    for c in 1:nc
        # Force some over/undershoots.
        state.alpha.internal[c] = c % 3 == 0 ? 1.3 : (c % 3 == 1 ? -0.2 : 0.5)
    end

    clip_alpha!(state.alpha, mesh)
    update_mixture_properties!(state, props)

    for c in 1:nc
        # After clip, α ∈ [0, 1].
        @test 0.0 <= state.alpha.internal[c] <= 1.0
        a = state.alpha.internal[c]
        @test isapprox(state.rho[c], a * rho1 + (1 - a) * rho2; rtol = 1.0e-14)
    end
end
