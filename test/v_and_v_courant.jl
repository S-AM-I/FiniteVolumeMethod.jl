# test/v_and_v_courant.jl — Courant number + derived-field invariants V&V (v3.30)
#
# Second analytical benchmark for `postprocessing`. The first
# benchmark (v3.20, vorticity + Q on canonical flows) tested
# the derived-field kernels. This one covers the remaining
# post-processing primitives: the Courant number (face-flux
# stencil) and the Q-criterion sign invariant.
#
# Courant number on a Cartesian mesh with uniform flow
# u = (U, 0):
#
#   |phi|_f = U · dy on streamwise faces (left + right); zero
#   on cross-stream faces (top + bottom). Total per cell:
#
#     Σ_f |phi_f| = 2 · U · dy,
#
#   so
#
#     Co = dt · 2·U·dy / (2·V_c) = dt · U / dx.
#
# Diagonal flow u = (U, U) activates all four faces equally:
#
#     Σ_f |phi_f| = 4·U·h,   Co = 2·dt·U / h.
#
# These identities can be verified cell-by-cell on a
# Cartesian mesh. Evidence toward future `stable` promotion
# of `postprocessing`.

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_courant_number, compute_q_criterion, face_normal_area
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: Courant — zero flow ⇒ Co ≡ 0" begin
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    phi = FaceFluxField(:phi, mesh; value = 0.0)

    Co = compute_courant_number(phi, mesh, 0.01)
    @test all(isapprox.(Co, 0.0; atol = 1.0e-14))
end

@testset "V&V: Courant — uniform flow matches dt·U/dx analytical" begin
    # Cartesian N×N mesh on [0, 1] × [0, 1] ⇒ dx = dy = 1/N.
    # For u = (U, 0), Co = dt·U/dx.
    U = 2.0
    for N in (8, 16, 32)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        phi = FaceFluxField(:phi, mesh)
        for f in 1:size(mesh.face_cells, 2)
            phi.values[f] = U * face_normal_area(mesh, f)[1]
        end

        dt = 0.1
        Co = compute_courant_number(phi, mesh, dt)

        dx = 1.0 / N
        Co_analytical = dt * U / dx

        # Interior cells only — boundary cells have one missing
        # external face so the stencil sum is smaller. Interior
        # cells have both left and right streamwise faces and the
        # formula gives Co = dt·U/dx exactly.
        count_checked = 0
        for c in 1:nc
            x = mesh.cell_centers[1, c]
            y = mesh.cell_centers[2, c]
            if 0.2 < x < 0.8 && 0.2 < y < 0.8
                @test isapprox(Co[c], Co_analytical; rtol = 1.0e-10)
                count_checked += 1
            end
        end
        @test count_checked > 0
    end
end

@testset "V&V: Courant — diagonal flow activates all four faces" begin
    # u = (U, U): each cell has |phi_left| = |phi_right| = U·dy,
    # |phi_bottom| = |phi_top| = U·dx. On a square Cartesian mesh
    # with dx = dy = h, Σ|phi| = 4 U h, Co = 2·dt·U/h.
    U = 1.5
    N = 20
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    phi = FaceFluxField(:phi, mesh)
    for f in 1:size(mesh.face_cells, 2)
        S_f = face_normal_area(mesh, f)
        phi.values[f] = U * (S_f[1] + S_f[2])  # uniform velocity (U, U)
    end

    dt = 0.05
    Co = compute_courant_number(phi, mesh, dt)
    Co_analytical = 2 * dt * U / (1.0 / N)

    count_checked = 0
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        if 0.2 < x < 0.8 && 0.2 < y < 0.8
            @test isapprox(Co[c], Co_analytical; rtol = 1.0e-10)
            count_checked += 1
        end
    end
    @test count_checked > 20
end

@testset "V&V: Q-criterion — solid-body rotation Q = Ω² > 0 identifies vortex" begin
    # Q > 0 identifies vortex cores. Verify via a two-flow
    # comparison: solid-body rotation vs. pure shear. The Q sign
    # must distinguish the two cases at every interior cell.
    Omega = 2.0
    A = 2.0

    mesh_rot = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    mesh_shear = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)

    U_rot = CollocatedVectorField(:U, mesh_rot)
    U_shear = CollocatedVectorField(:U, mesh_shear)
    for c in 1:length(mesh_rot.cell_volumes)
        x = mesh_rot.cell_centers[1, c] - 0.5
        y = mesh_rot.cell_centers[2, c] - 0.5
        U_rot.internal[c] = SVector(-Omega * y, Omega * x)
    end
    for c in 1:length(mesh_shear.cell_volumes)
        y = mesh_shear.cell_centers[2, c]
        U_shear.internal[c] = SVector(A * y, 0.0)
    end

    Q_rot = compute_q_criterion(U_rot, mesh_rot)
    Q_shear = compute_q_criterion(U_shear, mesh_shear)

    # Interior: rotation Q = Ω² > 0, shear Q = 0 (balance of
    # production and destruction).
    for c in 1:length(mesh_rot.cell_volumes)
        x = mesh_rot.cell_centers[1, c]
        y = mesh_rot.cell_centers[2, c]
        if 0.25 < x < 0.75 && 0.25 < y < 0.75
            @test Q_rot[c] > 0.9 * Omega^2
            @test abs(Q_shear[c]) < 1.0e-8
        end
    end
end
