# test/v_and_v_p1_equilibrium.jl — P1 radiative equilibrium V&V (v3.25)
#
# Second analytical benchmark for `radiation`. The first benchmark
# (v3.15, cold-slab attenuation) tested the diffusion + absorption
# kernel; this one tests the emission + Marshak-wall BC pathway
# through the **radiative-equilibrium** invariant:
#
#   In a medium at uniform temperature T_m with walls at the same
#   T_m, the P1 equation
#
#     −(1/3a) ∇²G + a·G = 4·a·σ·T_m⁴
#
#   admits the trivial uniform solution G ≡ 4·σ·T_m⁴, and this
#   solution is consistent with Marshak BCs
#
#     G_wall + (2/a) · n̂·∇G_wall / 3 = 4σT_wall⁴
#
#   because ∇G ≡ 0 in equilibrium. This is the canonical
#   "Modest radiative enclosure" check: an isolated cavity in
#   thermal equilibrium must reach G = 4σT⁴ identically.
#
# Evidence toward future `stable` promotion of `radiation`.

using FiniteVolumeMethod
using LinearSolve
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

const SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant

function solve_equilibrium(Nx::Int, Ny::Int, T_wall::Float64, T_medium::Float64, a::Float64)
    mesh = build_cartesian_unstructured_mesh(Nx, Ny, 1.0, 1.0)
    rad = P1Model(; a = a)
    T_field = CollocatedScalarField(:T, mesh; value = T_medium)

    marshak = marshak_wall_bc(rad, T_wall)
    bcs_G = Dict{Symbol, AbstractBoundaryCondition}(
        :left => marshak, :right => marshak,
        :bottom => marshak, :top => marshak,
    )

    G = solve_p1_radiation(rad, T_field, mesh, bcs_G; linear_solver = LUFactorization())
    return mesh, G
end

@testset "V&V: P1 equilibrium — G ≡ 4σT⁴ when T_wall = T_medium" begin
    T_m = 500.0
    G_eq_analytical = 4 * SIGMA_SB * T_m^4

    for a in (0.1, 1.0, 10.0)
        mesh, G = solve_equilibrium(32, 32, T_m, T_m, a)
        nc = length(mesh.cell_volumes)

        # Every interior cell should equal G_eq_analytical to high
        # relative precision (the equilibrium is an algebraic
        # identity, so discretization does not degrade it).
        mask_interior = 0
        for c in 1:nc
            x = mesh.cell_centers[1, c]
            y = mesh.cell_centers[2, c]
            if 0.2 < x < 0.8 && 0.2 < y < 0.8
                @test isapprox(G.internal[c], G_eq_analytical; rtol = 1.0e-2)
                mask_interior += 1
            end
        end
        @test mask_interior > 50
    end
end

@testset "V&V: P1 equilibrium — solution is uniform (no spurious gradient)" begin
    T_m = 800.0
    a = 1.0
    mesh, G = solve_equilibrium(32, 32, T_m, T_m, a)
    nc = length(mesh.cell_volumes)

    # Field should be spatially uniform (zero gradient). Measure
    # max - min in the interior.
    interior_vals = Float64[]
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.2 < x < 0.8 && 0.2 < y < 0.8
            push!(interior_vals, G.internal[c])
        end
    end
    spread = maximum(interior_vals) - minimum(interior_vals)
    mean_val = sum(interior_vals) / length(interior_vals)
    @test spread / mean_val < 1.0e-4
end

@testset "V&V: P1 equilibrium — T⁴ scaling at fixed a" begin
    # Equilibrium G scales exactly as T⁴. Verify at three
    # temperatures; ratios should be (T₂/T₁)⁴.
    a = 1.0
    temps = (300.0, 600.0, 1200.0)
    means = Float64[]
    for T_m in temps
        mesh, G = solve_equilibrium(16, 16, T_m, T_m, a)
        nc = length(mesh.cell_volumes)
        sample = 0.0
        count = 0
        for c in 1:nc
            x = mesh.cell_centers[1, c]
            y = mesh.cell_centers[2, c]
            if 0.3 < x < 0.7 && 0.3 < y < 0.7
                sample += G.internal[c]
                count += 1
            end
        end
        push!(means, sample / count)
    end

    # T doubles ⇒ G scales by 16 (= 2⁴).
    r1 = means[2] / means[1]
    r2 = means[3] / means[2]
    @test isapprox(r1, 16.0; rtol = 5.0e-3)
    @test isapprox(r2, 16.0; rtol = 5.0e-3)
end
