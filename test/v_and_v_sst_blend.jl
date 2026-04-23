# test/v_and_v_sst_blend.jl — k-ω SST blending function algebra V&V (v3.89)

using FiniteVolumeMethod
using Test

include("TestHelpers.jl")

const _F1 = FiniteVolumeMethod._sst_F1
const _F2 = FiniteVolumeMethod._sst_F2
const _blend = FiniteVolumeMethod._blend

@testset "V&V: KappaOmegaSST default coefficients (Menter 1994)" begin
    co = KappaOmegaSST()
    @test co.a1 == 0.31
    @test co.beta_star == 0.09
    @test co.sigma_k1 == 0.85
    @test co.sigma_k2 == 1.0
    @test co.sigma_omega1 == 0.5
    @test co.sigma_omega2 == 0.856
    @test co.beta1 == 0.075
    @test co.beta2 == 0.0828
    @test co.kappa == 0.41
end

@testset "V&V: _blend — boundary cases and linearity" begin
    # F1 = 1 ⇒ return phi1; F1 = 0 ⇒ return phi2.
    @test _blend(0.85, 1.0, 1.0) == 0.85
    @test _blend(0.85, 1.0, 0.0) == 1.0
    @test _blend(0.5, 0.856, 1.0) == 0.5
    @test _blend(0.5, 0.856, 0.0) == 0.856
    # Midpoint F1 = 0.5 ⇒ mean of phi1 and phi2.
    @test _blend(0.85, 1.0, 0.5) ≈ 0.925 rtol = 1.0e-14
    # Exact linearity phi(F1) = F1·phi1 + (1-F1)·phi2 at 11 sample points.
    phi1, phi2 = 0.075, 0.0828
    for i in 0:10
        F1 = i / 10
        expected = F1 * phi1 + (1.0 - F1) * phi2
        @test _blend(phi1, phi2, F1) ≈ expected rtol = 1.0e-14
    end
end

@testset "V&V: _sst_F1 ∈ [0, 1] (tanh bound)" begin
    # F1 = tanh(arg1⁴) ∈ [0, 1] by construction.
    co = KappaOmegaSST()
    nu = 1.0e-5
    # Sweep a range of (k, omega, d) triplets.
    for k in (1.0e-4, 1.0e-2, 1.0, 1.0e2)
        for omega in (1.0, 10.0, 100.0, 1.0e4)
            for d in (1.0e-5, 1.0e-3, 1.0e-1, 1.0)
                F1 = _F1(k, omega, nu, d, co, 0.0)
                @test 0.0 <= F1 <= 1.0
            end
        end
    end
end

@testset "V&V: _sst_F2 ∈ [0, 1] (tanh bound)" begin
    co = KappaOmegaSST()
    nu = 1.0e-5
    for k in (1.0e-4, 1.0e-2, 1.0, 1.0e2)
        for omega in (1.0, 10.0, 100.0, 1.0e4)
            for d in (1.0e-5, 1.0e-3, 1.0e-1, 1.0)
                F2 = _F2(k, omega, nu, d, co)
                @test 0.0 <= F2 <= 1.0
            end
        end
    end
end

@testset "V&V: _sst_F2 — algebraic closed form" begin
    # F2 = tanh(max(2√k/(β*·ω·d), 500ν/(d²·ω))²). Verify the identity
    # at explicit sample points by reconstructing the argument by hand.
    co = KappaOmegaSST()
    nu = 1.0e-5
    for (k, omega, d) in (
            (1.0e-2, 10.0, 1.0e-2),
            (1.0e-1, 100.0, 1.0e-1),
            (1.0, 1.0e3, 1.0e-3),
            (1.0e-3, 5.0, 1.0),
        )
        a = 2.0 * sqrt(k) / (co.beta_star * omega * d)
        b = 500.0 * nu / (d^2 * omega)
        expected = tanh(max(a, b)^2)
        @test _F2(k, omega, nu, d, co) ≈ expected rtol = 1.0e-12
    end
end

@testset "V&V: _sst_F1 — near-wall limit (viscous sublayer)" begin
    # Close enough to the wall, the 500·ν/(d²·ω) term dominates
    # and F1 saturates to 1.0 (tanh of a very large argument).
    co = KappaOmegaSST()
    nu = 1.0e-5
    # d = 1e-6 so d² = 1e-12; with ω = 1 and grad_k·grad_omega = 0 this
    # forces arg1 very large ⇒ F1 ≈ 1.
    F1_wall = _F1(1.0e-3, 1.0, nu, 1.0e-6, co, 0.0)
    @test F1_wall > 0.99
end

@testset "V&V: _sst_F1 — free-stream limit (grad_k·grad_omega large)" begin
    # Large grad_k·grad_omega drives CDkw large ⇒ arg1_c → 0 ⇒ arg1 → 0
    # ⇒ tanh(arg1⁴) → 0, so F1 → 0 in the free stream.
    co = KappaOmegaSST()
    nu = 1.0e-5
    F1_free = _F1(1.0e-4, 1.0e4, nu, 1.0, co, 1.0e10)
    @test F1_free < 0.01
end

@testset "V&V: blended constants — sigma_k, beta, alpha" begin
    # When F1 = 1 (near wall) blended constants equal the inner-set values;
    # when F1 = 0 (free stream) they equal the outer-set values. Uses the
    # exact same _blend call site pattern as turbulent_viscosity!.
    co = KappaOmegaSST()
    @test _blend(co.sigma_k1, co.sigma_k2, 1.0) == co.sigma_k1
    @test _blend(co.sigma_k1, co.sigma_k2, 0.0) == co.sigma_k2
    @test _blend(co.sigma_omega1, co.sigma_omega2, 1.0) == co.sigma_omega1
    @test _blend(co.sigma_omega1, co.sigma_omega2, 0.0) == co.sigma_omega2
    @test _blend(co.beta1, co.beta2, 1.0) == co.beta1
    @test _blend(co.beta1, co.beta2, 0.0) == co.beta2
end
