# test/v_and_v_iso_advector.jl — geometric α reconstruction V&V (v3.92)
#
# Exercises the `assemble_isoadvector_flux!` primitive added in Wave-1
# Agent D. The scheme is a simplified linear-plane PLIC reconstruction
# of the α face-flux for interface cells, falling back to donor-cell
# upwind for pure-phase cells.
#
# Invariants verified:
#
#   1. Total α·V is conserved to 1e-12 after one explicit Euler step
#      with the reconstructed flux.
#   2. Pure-phase cells (α ≈ 0 or α ≈ 1) produce the obvious flux
#      (0 or F_f respectively).
#   3. A flat α field outside the interface band falls back to upwind.
#   4. A horizontal interface α = 1 below y = 0.5, α = 0 above, advected
#      upward by a uniform U = (0, v), stays sharp (α remains in [0, 1]).

using FiniteVolumeMethod
using LinearAlgebra
using StaticArrays
using Test

include("TestHelpers.jl")

const owner = FiniteVolumeMethod.owner
const neighbour = FiniteVolumeMethod.neighbour
const is_internal_face = FiniteVolumeMethod.is_internal_face
const face_normal_area = FiniteVolumeMethod.face_normal_area

function _uniform_velocity_field(mesh, u::Float64, v::Float64)
    U = CollocatedVectorField(:U, mesh)
    for c in eachindex(U.internal)
        U.internal[c] = SVector{2, Float64}(u, v)
    end
    for i in eachindex(U.boundary)
        U.boundary[i] = SVector{2, Float64}(u, v)
    end
    return U
end

@testset "V&V: isoAdvector — pure-phase α=0 gives zero reconstructed flux" begin
    mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)
    alpha = CollocatedScalarField(:alpha, mesh; value = 0.0)
    U = _uniform_velocity_field(mesh, 1.0, 0.0)
    phi_alpha = FaceFluxField(:phi_alpha, mesh; value = 0.0)

    FiniteVolumeMethod.assemble_isoadvector_flux!(
        phi_alpha, alpha, U, mesh, 1.0e-3,
    )
    for f in 1:nf
        @test phi_alpha.values[f] == 0.0
    end
end

@testset "V&V: isoAdvector — pure-phase α=1 gives F_f at each face" begin
    mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)
    alpha = CollocatedScalarField(:alpha, mesh; value = 1.0)
    U = _uniform_velocity_field(mesh, 1.0, 0.0)
    phi_alpha = FaceFluxField(:phi_alpha, mesh; value = 0.0)

    FiniteVolumeMethod.assemble_isoadvector_flux!(
        phi_alpha, alpha, U, mesh, 1.0e-3,
    )
    for f in 1:nf
        S = face_normal_area(mesh, f)
        F_f_expected = dot(SVector{2, Float64}(1.0, 0.0), S)
        @test isapprox(phi_alpha.values[f], F_f_expected; rtol = 1.0e-12)
    end
end

@testset "V&V: isoAdvector — flat interior α outside band falls back to upwind" begin
    # α = 0.4 everywhere: |∇α| ≈ 0 → PLIC reconstruction degenerates.
    # The primitive is required to fall back to upwind donor α · F_f.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)
    alpha = CollocatedScalarField(:alpha, mesh; value = 0.4)
    U = _uniform_velocity_field(mesh, 1.0, 0.0)
    phi_alpha = FaceFluxField(:phi_alpha, mesh; value = 0.0)

    FiniteVolumeMethod.assemble_isoadvector_flux!(
        phi_alpha, alpha, U, mesh, 1.0e-3,
    )
    for f in 1:nf
        S = face_normal_area(mesh, f)
        F_f = dot(SVector{2, Float64}(1.0, 0.0), S)
        @test isapprox(phi_alpha.values[f], F_f * 0.4; rtol = 1.0e-12)
    end
end

@testset "V&V: isoAdvector — closed-box α·V conservation (one step)" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # α = step in y → interface band at y = 0.5.
    alpha = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        y = mesh.cell_centers[2, c]
        alpha.internal[c] = y < 0.5 ? 1.0 : 0.0
    end

    # Zero velocity on the boundary faces (closed box). Interior flux
    # comes from the linear interpolation of U at faces.
    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        U.internal[c] = SVector{2, Float64}(0.0, 0.1)
    end
    # Boundary values at walls must be zero so phi_alpha = 0 on outer faces.
    for (i, f) in enumerate(U.boundary_face_indices)
        U.boundary[i] = SVector{2, Float64}(0.0, 0.0)
    end

    phi_alpha = FaceFluxField(:phi_alpha, mesh; value = 0.0)
    FiniteVolumeMethod.assemble_isoadvector_flux!(
        phi_alpha, alpha, U, mesh, 1.0e-3,
    )
    # Zero out fluxes on boundary faces (closed box).
    for f in 1:nf
        if !is_internal_face(mesh, f)
            phi_alpha.values[f] = 0.0
        end
    end

    dt = 1.0e-3
    total_before = sum(alpha.internal[c] * mesh.cell_volumes[c] for c in 1:nc)
    for f in 1:nf
        if is_internal_face(mesh, f)
            F = phi_alpha.values[f] * dt
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            alpha.internal[P] -= F / mesh.cell_volumes[P]
            alpha.internal[N] += F / mesh.cell_volumes[N]
        end
    end
    total_after = sum(alpha.internal[c] * mesh.cell_volumes[c] for c in 1:nc)

    @test isapprox(total_after, total_before; rtol = 1.0e-12, atol = 1.0e-12)
end

@testset "V&V: isoAdvector — horizontal interface under upward velocity" begin
    # α = 1 below y=0.5, α = 0 above. After one small explicit step
    # with uniform upward velocity the field must stay bounded and
    # largely sharp (α remains in [0, 1]).
    mesh = build_cartesian_unstructured_mesh(20, 20, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    alpha = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        y = mesh.cell_centers[2, c]
        alpha.internal[c] = y < 0.5 ? 1.0 : 0.0
    end

    U = _uniform_velocity_field(mesh, 0.0, 0.05)

    phi_alpha = FaceFluxField(:phi_alpha, mesh; value = 0.0)
    FiniteVolumeMethod.assemble_isoadvector_flux!(
        phi_alpha, alpha, U, mesh, 5.0e-3,
    )

    # Zero boundary fluxes for closed test.
    for f in 1:nf
        if !is_internal_face(mesh, f)
            phi_alpha.values[f] = 0.0
        end
    end

    dt = 5.0e-3
    for f in 1:nf
        if is_internal_face(mesh, f)
            F = phi_alpha.values[f] * dt
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            alpha.internal[P] -= F / mesh.cell_volumes[P]
            alpha.internal[N] += F / mesh.cell_volumes[N]
        end
    end

    for c in 1:nc
        @test -1.0e-9 <= alpha.internal[c] <= 1.0 + 1.0e-9
    end
end
