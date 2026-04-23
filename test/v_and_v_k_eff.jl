# test/v_and_v_k_eff.jl — update_k_eff! + alpha_eff V&V (v3.56)
#
# Fifth convergence-verified benchmark for
# `conjugate_heat_transfer`, joining Laplace conduction (v3.12),
# unsteady decay (v3.21), Boussinesq (v3.32), and interface flux
# (v3.50). Covers the thermal-property primitives:
#
#   k_eff[c] = k_laminar + ρ · Cp · ν_t[c] / Pr_t
#   α_eff[c] = k_eff[c] / (ρ · Cp)                (thermal diffusivity)
#
# Invariants:
#
#   1. ν_t = nothing ⇒ k_eff ≡ k_laminar (laminar fallback).
#   2. ν_t = 0 ⇒ k_eff = k_laminar.
#   3. k_eff linear in ν_t at fixed ρ, Cp, Pr_t.
#   4. α_eff = k_eff / (ρ · Cp) algebraic identity.
#   5. Pr_t scaling: doubling Pr_t halves the turbulent contribution.
#   6. ρ·Cp scaling: doubling ρ·Cp doubles the turbulent contribution.

using FiniteVolumeMethod
using StaticArrays
using Test

include("TestHelpers.jl")

function setup_thermal_state(mesh)
    props = FluidThermalProperties{2}(;
        Cp = 1000.0, k = 0.026, Pr_t = 0.9,
        beta = 0.0, T_ref = 300.0,
        g = SVector(0.0, -9.81),
    )
    thermal = FiniteVolumeMethod.ThermalState(mesh)
    return props, thermal
end

@testset "V&V: k_eff — ν_t = nothing ⇒ k_eff ≡ k_laminar" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    props, thermal = setup_thermal_state(mesh)

    update_k_eff!(thermal, props, nothing, 1.2)

    for c in 1:length(thermal.k_eff)
        @test isapprox(thermal.k_eff[c], props.k; rtol = 1.0e-14)
    end
end

@testset "V&V: k_eff — ν_t = 0 ⇒ k_eff = k_laminar" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    props, thermal = setup_thermal_state(mesh)

    nu_t = zeros(Float64, length(mesh.cell_volumes))
    update_k_eff!(thermal, props, nu_t, 1.2)

    for c in 1:length(thermal.k_eff)
        @test isapprox(thermal.k_eff[c], props.k; rtol = 1.0e-14)
    end
end

@testset "V&V: k_eff — linear in ν_t at fixed ρ, Cp, Pr_t" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    props, thermal = setup_thermal_state(mesh)
    rho = 1.2
    nc = length(mesh.cell_volumes)

    nu_t_a = fill(1.0e-5, nc)
    update_k_eff!(thermal, props, nu_t_a, rho)
    k_eff_a = copy(thermal.k_eff)

    nu_t_b = fill(2.0e-5, nc)
    update_k_eff!(thermal, props, nu_t_b, rho)
    k_eff_b = copy(thermal.k_eff)

    # k_eff_b - k_lam should be 2 × (k_eff_a - k_lam).
    for c in 1:nc
        delta_a = k_eff_a[c] - props.k
        delta_b = k_eff_b[c] - props.k
        @test isapprox(delta_b / delta_a, 2.0; rtol = 1.0e-12)
    end
end

@testset "V&V: k_eff — algebraic closed-form at every cell" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    props, thermal = setup_thermal_state(mesh)
    rho = 1.5
    nc = length(mesh.cell_volumes)

    nu_t = [1.0e-6 * (c / nc) for c in 1:nc]
    update_k_eff!(thermal, props, nu_t, rho)

    for c in 1:nc
        expected = props.k + rho * props.Cp * nu_t[c] / props.Pr_t
        @test isapprox(thermal.k_eff[c], expected; rtol = 1.0e-14)
    end
end

@testset "V&V: compute_alpha_eff — α = k / (ρ·Cp) algebraic identity" begin
    nc = 20
    k_eff_vec = [0.03 + 0.01 * i for i in 1:nc]
    rho = 1.2
    Cp = 1005.0

    alpha = compute_alpha_eff(k_eff_vec, rho, Cp)

    for c in 1:nc
        @test isapprox(alpha[c], k_eff_vec[c] / (rho * Cp); rtol = 1.0e-14)
    end
end

@testset "V&V: k_eff — Pr_t inverse scaling" begin
    # Doubling Pr_t halves the turbulent contribution.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    thermal = FiniteVolumeMethod.ThermalState(mesh)
    rho = 1.2
    nu_t = fill(1.0e-5, length(mesh.cell_volumes))

    props_a = FluidThermalProperties{2}(;
        Cp = 1000.0, k = 0.026, Pr_t = 0.5,
        beta = 0.0, T_ref = 300.0, g = SVector(0.0, -9.81),
    )
    props_b = FluidThermalProperties{2}(;
        Cp = 1000.0, k = 0.026, Pr_t = 1.0,
        beta = 0.0, T_ref = 300.0, g = SVector(0.0, -9.81),
    )

    update_k_eff!(thermal, props_a, nu_t, rho)
    k_eff_a = copy(thermal.k_eff)

    update_k_eff!(thermal, props_b, nu_t, rho)
    k_eff_b = copy(thermal.k_eff)

    # (k_eff_a - k_lam) / (k_eff_b - k_lam) should be Pr_t_b / Pr_t_a = 2.
    for c in 1:length(thermal.k_eff)
        ratio = (k_eff_a[c] - props_a.k) / (k_eff_b[c] - props_b.k)
        @test isapprox(ratio, 2.0; rtol = 1.0e-12)
    end
end
