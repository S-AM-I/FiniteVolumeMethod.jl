# test/v_and_v_wale.jl — WALE LES model V&V (v3.28)
#
# Second analytical benchmark for `turbulence_les`. Smagorinsky
# (v3.19) tested the ν_t = (C_s·Δ)²·|S| algebra; this one tests
# the qualitatively different **WALE invariant properties** that
# motivated its development:
#
#   WALE was designed so that ν_sgs vanishes at a wall without
#   explicit damping. The key identity is that WALE evaluates to
#   exactly zero on all 2D "shear-only" velocity fields
#   (gradient tensor that squares to zero deviatoric part), which
#   includes:
#
#     • pure shear U = (A·y, 0),
#     • solid-body rotation U = (−Ω·y, Ω·x),
#     • parabolic profile U = (A·y², 0).
#
# On 2D flows with non-trivial second velocity gradients (e.g.
# U = (x·y, 0)) WALE gives a non-zero ν_t and obeys the expected
# (Cw·Δ)² scaling. Evidence toward future `stable` promotion.

using FiniteVolumeMethod
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: WALE — zero velocity ⇒ ν_t ≡ 0" begin
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = WALE(mesh; Cw = 0.325)

    U = CollocatedVectorField(:U, mesh; value = SVector(0.0, 0.0))

    nu_t = zeros(Float64, nc)
    FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)

    @test all(isapprox.(nu_t, 0.0; atol = 1.0e-14))
end

@testset "V&V: WALE — pure shear ⇒ ν_t vanishes (design property)" begin
    # U = (A·y, 0). Gradient tensor g_ij has g_12 = A as the only
    # non-zero entry. (g·g)_ij ≡ 0 ⇒ S_d:S_d = 0 ⇒ ν_t = 0.
    # This is the defining feature of WALE: unlike Smagorinsky,
    # it gives zero eddy viscosity in pure shear and therefore
    # vanishes at walls in channel flow without damping.
    A = 3.0
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = WALE(mesh; Cw = 0.325)

    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(A * y, 0.0)
    end

    nu_t = zeros(Float64, nc)
    FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)

    # Interior cells (FVM gradient is exact on linear fields).
    count_checked = 0
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.2 < x < 0.8 && 0.2 < y < 0.8
            @test abs(nu_t[c]) < 1.0e-12
            count_checked += 1
        end
    end
    @test count_checked > 50
end

@testset "V&V: WALE — solid-body rotation ⇒ ν_t vanishes" begin
    # U = (−Ω·y, Ω·x). g²_ij = −Ω²·δ_ij (isotropic), so the
    # traceless symmetric part S_d_ij ≡ 0 ⇒ ν_t = 0.
    Omega = 2.5
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    model = WALE(mesh; Cw = 0.325)

    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c] - 0.5
        y = mesh.cell_centers[2, c] - 0.5
        U.internal[c] = SVector(-Omega * y, Omega * x)
    end

    nu_t = zeros(Float64, nc)
    FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)

    count_checked = 0
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.2 < x < 0.8 && 0.2 < y < 0.8
            @test abs(nu_t[c]) < 1.0e-10
            count_checked += 1
        end
    end
    @test count_checked > 50
end

@testset "V&V: WALE — non-trivial flow gives ν_t > 0 with Δ² scaling" begin
    # U = (x·y, 0). ∂u/∂x = y, ∂u/∂y = x, gradient tensor has
    # g²_11 = y², g²_12 = xy, non-zero traceless symmetric part.
    # ν_t > 0 here and must scale as Δ² under mesh refinement.
    results = Float64[]
    for N in (8, 16)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        model = WALE(mesh; Cw = 0.325)

        U = CollocatedVectorField(:U, mesh)
        for c in 1:nc
            x = mesh.cell_centers[1, c]
            y = mesh.cell_centers[2, c]
            U.internal[c] = SVector(x * y, 0.0)
        end

        nu_t = zeros(Float64, nc)
        FiniteVolumeMethod.turbulent_viscosity!(nu_t, model, U, mesh)

        # All cells non-negative.
        @test all(>=(0.0), nu_t)
        # Some cells strictly positive (the flow has non-zero
        # WALE structure away from boundaries).
        @test maximum(nu_t) > 0.0

        # Sample interior average (avoids boundary noise).
        sample = 0.0
        count = 0
        for c in 1:nc
            x = mesh.cell_centers[1, c]
            y = mesh.cell_centers[2, c]
            if 0.3 < x < 0.7 && 0.3 < y < 0.7
                sample += nu_t[c]
                count += 1
            end
        end
        push!(results, sample / count)
    end

    # Δ_fine / Δ_coarse = 1/2 ⇒ ν_t ratio should be (1/2)² = 1/4.
    # WALE has Δ² scaling (Cw·Δ)² × dimensionless gradient factor.
    ratio = results[2] / results[1]
    @test isapprox(ratio, 0.25; rtol = 5.0e-2)
end

@testset "V&V: WALE — Cw² scaling at fixed mesh + flow" begin
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(x * y, 0.0)
    end

    models = (WALE(mesh; Cw = 0.1), WALE(mesh; Cw = 0.2), WALE(mesh; Cw = 0.4))
    samples = Float64[]
    for m in models
        nu_t = zeros(Float64, nc)
        FiniteVolumeMethod.turbulent_viscosity!(nu_t, m, U, mesh)
        # Interior sample.
        sample = 0.0
        count = 0
        for c in 1:nc
            x = mesh.cell_centers[1, c]
            y = mesh.cell_centers[2, c]
            if 0.3 < x < 0.7 && 0.3 < y < 0.7
                sample += nu_t[c]
                count += 1
            end
        end
        push!(samples, sample / count)
    end

    # Cw doubles ⇒ ν_t scales by 4.
    r1 = samples[2] / samples[1]
    r2 = samples[3] / samples[2]
    @test isapprox(r1, 4.0; rtol = 1.0e-10)
    @test isapprox(r2, 4.0; rtol = 1.0e-10)
end
