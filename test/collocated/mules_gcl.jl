# test/collocated/mules_gcl.jl — MULES flux limiter bounds and the geometric

using FiniteVolumeMethod
using FiniteVolumeMethod: mules_limit_flux!, verify_gcl
using Test
using StaticArrays: SVector

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "MULES flux limiter bounds output between upwind/high-order" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    alpha = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        alpha.internal[c] = mesh.cell_centers[1, c] < 0.5 ? 1.0 : 0.0
    end

    phi_up = FaceFluxField(:phi_up, mesh; value = 0.0)
    phi_hi = FaceFluxField(:phi_hi, mesh; value = 0.0)
    for f in 1:nf
        nx = mesh.face_normals[1, f]
        phi_up.values[f] = 0.1 * nx * mesh.face_areas[f]
        phi_hi.values[f] = 0.15 * nx * mesh.face_areas[f]   # overshoots by 50%
    end

    limited = FaceFluxField(:phi_lim, mesh; value = 0.0)
    mules_limit_flux!(limited, alpha, phi_up, phi_hi, mesh, 0.01)

    # Per-face: limited must sit between upwind and high.
    for f in 1:nf
        lo = min(phi_up.values[f], phi_hi.values[f])
        hi = max(phi_up.values[f], phi_hi.values[f])
        @test lo - 1.0e-12 <= limited.values[f] <= hi + 1.0e-12
    end
end

@testset "MULES with identical inputs is identity" begin
    # When phi_upwind == phi_high, MULES adds zero anti-diffusion → result
    # equals input regardless of alpha.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    alpha = CollocatedScalarField(:alpha, mesh; value = 0.5)

    phi_up = FaceFluxField(:phi_up, mesh; value = 0.1)
    phi_hi = FaceFluxField(:phi_hi, mesh; value = 0.1)

    limited = FaceFluxField(:phi_lim, mesh; value = 0.0)
    mules_limit_flux!(limited, alpha, phi_up, phi_hi, mesh, 0.01)

    @test all(limited.values .== phi_up.values)
end

@testset "verify_gcl returns zero residual for a perfect GCL pair" begin
    # Construct a GCL-consistent trio: phi_mesh, V_old, V_new.
    mesh = build_cartesian_unstructured_mesh(5, 5, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Start with V_old = actual cell volumes.
    V_old = copy(mesh.cell_volumes)

    # Assign an arbitrary face sweep flux.
    phi_mesh = [0.1 * (f / nf) for f in 1:nf]

    # Construct V_new consistent with phi_mesh via the GCL relation:
    # V_new[c] = V_old[c] + dt * Σ_f ε(c, f) · phi_mesh[f]
    dt = 0.05
    V_new = copy(V_old)
    for f in 1:nf
        P = mesh.face_cells[1, f]
        N = mesh.face_cells[2, f]
        V_new[P] += dt * phi_mesh[f]
        if N != 0
            V_new[N] -= dt * phi_mesh[f]
        end
    end

    residuals, max_res = FiniteVolumeMethod.verify_gcl(
        phi_mesh, V_old, V_new, mesh, dt,
    )
    @test length(residuals) == nc
    @test max_res ≈ 0.0 atol = 1.0e-12
end

@testset "verify_gcl detects inconsistent mesh motion" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    V_old = copy(mesh.cell_volumes)
    V_new = copy(V_old)
    # Arbitrarily grow cell 1 by 10% without a consistent flux — GCL should
    # be violated.
    V_new[1] *= 1.1

    phi_mesh = zeros(Float64, nf)
    _, max_res = FiniteVolumeMethod.verify_gcl(phi_mesh, V_old, V_new, mesh, 0.01)
    @test max_res > 0.01  # clearly non-zero
end
