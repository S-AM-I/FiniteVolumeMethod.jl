# test/v_and_v_postprocessing.jl — Vorticity + Q-criterion V&V (v3.20)
#
# Verifies `compute_vorticity` and `compute_q_criterion` against
# closed-form values on three canonical flows:
#
#   1. Uniform flow          U = (U₀, 0)
#      ∇×U = 0, Q = 0.
#
#   2. Simple shear          U = (A·y, 0)
#      ω_z = ∂v/∂x − ∂u/∂y = −A
#      S_12 = A/2, Ω_12 = −A/2
#      Q = (1/2)(|Ω|² − |S|²) = (1/2)(2·(A/2)² − 2·(A/2)²) = 0.
#
#   3. Solid-body rotation   U = (−Ω·y, Ω·x)
#      ω_z = ∂v/∂x − ∂u/∂y = Ω − (−Ω) = 2Ω
#      S = 0, Ω_12 = Ω
#      Q = (1/2)(2·Ω² − 0) = Ω² > 0.
#
# These are exact on a Cartesian mesh because the velocity fields
# are linear in (x, y) and the FVM gradient is exact on linear
# fields. Evidence for promoting `postprocessing` from
# `experimental`/`smoke_tested` to `provisional`/
# `convergence_verified`.

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_enstrophy, compute_q_criterion, compute_vorticity
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

function interior_mask(mesh, margin::Float64 = 0.2)
    nc = length(mesh.cell_volumes)
    # Determine domain extent from cell centers.
    xmin = minimum(mesh.cell_centers[1, c] for c in 1:nc)
    xmax = maximum(mesh.cell_centers[1, c] for c in 1:nc)
    ymin = minimum(mesh.cell_centers[2, c] for c in 1:nc)
    ymax = maximum(mesh.cell_centers[2, c] for c in 1:nc)
    Lx = xmax - xmin
    Ly = ymax - ymin
    mask = falses(nc)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if (xmin + margin * Lx < x < xmax - margin * Lx) &&
                (ymin + margin * Ly < y < ymax - margin * Ly)
            mask[c] = true
        end
    end
    return mask
end

@testset "V&V: Postprocessing — uniform flow has zero vorticity and Q" begin
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    U = CollocatedVectorField(:U, mesh; value = SVector(2.0, 0.0))
    omega = compute_vorticity(U, mesh)
    Q = compute_q_criterion(U, mesh)

    mask = interior_mask(mesh)
    for c in 1:nc
        if mask[c]
            @test isapprox(omega[c], 0.0; atol = 1.0e-10)
            @test isapprox(Q[c], 0.0; atol = 1.0e-10)
        end
    end
end

@testset "V&V: Postprocessing — simple shear ω = −A, Q = 0" begin
    A = 4.0
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(A * y, 0.0)
    end

    omega = compute_vorticity(U, mesh)
    Q = compute_q_criterion(U, mesh)

    mask = interior_mask(mesh)
    for c in 1:nc
        if mask[c]
            @test isapprox(omega[c], -A; rtol = 1.0e-8)
            # Pure shear: |Ω|² = |S|² ⇒ Q = 0.
            @test isapprox(Q[c], 0.0; atol = 1.0e-8)
        end
    end
end

@testset "V&V: Postprocessing — solid-body rotation ω = 2Ω, Q = Ω²" begin
    Omega = 3.0
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c] - 0.5
        y = mesh.cell_centers[2, c] - 0.5
        U.internal[c] = SVector(-Omega * y, Omega * x)
    end

    omega = compute_vorticity(U, mesh)
    Q = compute_q_criterion(U, mesh)

    mask = interior_mask(mesh)
    for c in 1:nc
        if mask[c]
            @test isapprox(omega[c], 2 * Omega; rtol = 1.0e-8)
            # Pure rotation: S = 0, |Ω|² = 2·Ω² ⇒ Q = Ω².
            @test isapprox(Q[c], Omega^2; rtol = 1.0e-8)
        end
    end
end

@testset "V&V: Postprocessing — enstrophy under solid-body rotation" begin
    # The solver defines enstrophy density as |ω|² (no 1/2 factor;
    # see `compute_enstrophy` in src/postprocessing/field_operations.jl).
    # For solid-body rotation ω_z = 2Ω ⇒ enstrophy density = 4Ω².
    Omega = 2.0
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c] - 0.5
        y = mesh.cell_centers[2, c] - 0.5
        U.internal[c] = SVector(-Omega * y, Omega * x)
    end

    enst = compute_enstrophy(U, mesh)

    mask = interior_mask(mesh)
    expected = (2 * Omega)^2
    for c in 1:nc
        if mask[c]
            @test isapprox(enst[c], expected; rtol = 1.0e-8)
        end
    end
end
