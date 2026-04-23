# test/v_and_v_spalart_allmaras.jl — Spalart-Allmaras V&V (v3.44)
#
# Fourth convergence-verified benchmark for `turbulence_rans`,
# joining k-ε DHIT (v3.18), k-ε log-layer (v3.23), and
# k-ω decay (v3.38). Covers the one-equation
# Spalart-Allmaras closure — the third canonical RANS model
# used in external-aero workflows.
#
# SA helpers to verify:
#
#   χ = ν̃ / ν                                   (viscosity ratio)
#   fv1 = χ³ / (χ³ + cv1³)                      (near-wall damping)
#   ν_t = ν̃ · fv1                               (eddy viscosity)
#
# Invariants:
#
#   1. High-χ asymptote: χ → ∞ ⇒ fv1 → 1 ⇒ ν_t → ν̃.
#   2. Low-χ asymptote: χ → 0 ⇒ fv1 → χ³/cv1³ ⇒ ν_t → ν̃·χ³/cv1³.
#   3. ν̃ = 0 ⇒ ν_t = 0.
#   4. Monotone ν_t(χ) at fixed ν̃: increasing χ increases ν_t.

using FiniteVolumeMethod
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: Spalart-Allmaras — ν̃ = 0 ⇒ ν_t = 0" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = SpalartAllmaras(mesh, Symbol[])

    turb_state = RANSTurbulenceState(model, mesh; nu_tilde = 0.0)
    nu = 1.0e-5

    nu_t = zeros(Float64, nc)
    FiniteVolumeMethod.turbulent_viscosity_sa!(nu_t, model, turb_state, mesh, nu)

    for c in 1:nc
        @test isapprox(nu_t[c], 0.0; atol = 1.0e-14)
    end
end

@testset "V&V: Spalart-Allmaras — high-χ limit ν_t ≈ ν̃" begin
    # When ν̃ >> ν (fully turbulent region), χ = ν̃/ν >> 1, so
    # fv1 → 1 and ν_t → ν̃.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = SpalartAllmaras(mesh, Symbol[])

    nu_tilde_val = 1.0e-2   # large
    nu = 1.0e-6              # small ⇒ χ = 10⁴

    turb_state = RANSTurbulenceState(model, mesh; nu_tilde = nu_tilde_val)
    nu_t = zeros(Float64, nc)
    FiniteVolumeMethod.turbulent_viscosity_sa!(nu_t, model, turb_state, mesh, nu)

    # ν_t ≈ ν̃ with fv1 ≈ 1 - (cv1/χ)³ ≈ 1.
    chi = nu_tilde_val / nu
    fv1_expected = chi^3 / (chi^3 + model.cv1^3)
    expected = nu_tilde_val * fv1_expected

    for c in 1:nc
        @test isapprox(nu_t[c], expected; rtol = 1.0e-12)
    end

    # fv1 very close to 1.
    @test fv1_expected > 0.9999
end

@testset "V&V: Spalart-Allmaras — low-χ limit ν_t ≈ ν̃·χ³/cv1³" begin
    # When ν̃ << ν, χ << 1, so fv1 ≈ χ³/cv1³.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = SpalartAllmaras(mesh, Symbol[])

    nu_tilde_val = 1.0e-9   # very small
    nu = 1.0e-5              # ⇒ χ = 1e-4

    turb_state = RANSTurbulenceState(model, mesh; nu_tilde = nu_tilde_val)
    nu_t = zeros(Float64, nc)
    FiniteVolumeMethod.turbulent_viscosity_sa!(nu_t, model, turb_state, mesh, nu)

    chi = nu_tilde_val / nu
    fv1_analytical = chi^3 / (chi^3 + model.cv1^3)
    expected = nu_tilde_val * fv1_analytical

    for c in 1:nc
        @test isapprox(nu_t[c], expected; rtol = 1.0e-12)
    end

    # fv1 very small.
    @test fv1_analytical < 1.0e-10
end

@testset "V&V: Spalart-Allmaras — ν_t monotone in ν̃ at fixed ν" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = SpalartAllmaras(mesh, Symbol[])
    nu = 1.0e-5

    results = Float64[]
    for nt_val in (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)
        turb_state = RANSTurbulenceState(model, mesh; nu_tilde = nt_val)
        nu_t = zeros(Float64, nc)
        FiniteVolumeMethod.turbulent_viscosity_sa!(nu_t, model, turb_state, mesh, nu)
        push!(results, nu_t[1])
    end

    # Monotone increase.
    for i in 1:(length(results) - 1)
        @test results[i] < results[i + 1]
    end

    # All non-negative (realizability).
    @test all(>=(0.0), results)
end

@testset "V&V: Spalart-Allmaras — fv1 function algebraic identity" begin
    # Direct check of fv1(χ) = χ³/(χ³ + cv1³) formula.
    cv1 = 7.1
    for chi in (0.01, 0.5, 1.0, 5.0, 50.0, 500.0)
        fv1_computed = FiniteVolumeMethod._sa_fv1(chi, cv1)
        fv1_expected = chi^3 / (chi^3 + cv1^3)
        @test isapprox(fv1_computed, fv1_expected; rtol = 1.0e-14)
    end
end
