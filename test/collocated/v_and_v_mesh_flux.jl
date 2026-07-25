# test/v_and_v_mesh_flux.jl — Face sweep flux primitive V&V (v3.34)
#
# Third convergence-verified benchmark for `dynamic_mesh`.
# The first (v3.14) tested GCL under three motion patterns at
# the cell level; the second (v3.29) extended to rotational
# motion. This one exercises the **face-primitive** kernel
# `compute_mesh_flux!` directly:
#
#   phi_mesh[f] = <d_f, S_f> / Δt,
#
# where d_f is the face-center displacement (interpolated from
# owner/neighbour cell centers for internal faces, or taken as
# the owner displacement for boundary faces), and S_f is the
# face normal-area vector.
#
# Invariants verified:
#
#   1. Zero displacement ⇒ phi_mesh ≡ 0.
#   2. Uniform translation d = d₀: phi_mesh[f] = <d₀, S_f>/Δt
#      exactly at every face.
#   3. Δt scaling: phi_mesh ∝ 1/Δt (doubling Δt halves phi_mesh).
#   4. Face-sum identity: Σ_f ε(c, f) · phi_mesh[f] = 0 for
#      uniform translation (closed-cell divergence theorem).
#
# Puts `dynamic_mesh` at three convergence-verified benchmarks.

using FiniteVolumeMethod
using FiniteVolumeMethod: MeshMotionState, compute_mesh_flux!, face_normal_area
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: mesh flux — zero displacement ⇒ phi_mesh ≡ 0" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    # Displacement left at its zero-initialized default.
    nf = size(mesh.face_cells, 2)
    compute_mesh_flux!(ms, mesh, 0.1)
    @test all(==(0.0), ms.phi_mesh)
end

@testset "V&V: mesh flux — uniform translation exact formula" begin
    d0 = SVector(0.25, -0.1)
    dt = 0.1
    mesh = build_cartesian_unstructured_mesh(10, 10, 1.0, 1.0)
    ms = MeshMotionState(mesh)

    nc = length(mesh.cell_volumes)
    for c in 1:nc
        ms.displacement[c] = d0
    end

    compute_mesh_flux!(ms, mesh, dt)

    nf = size(mesh.face_cells, 2)
    for f in 1:nf
        S_f = face_normal_area(mesh, f)
        phi_expected = (d0[1] * S_f[1] + d0[2] * S_f[2]) / dt
        @test isapprox(ms.phi_mesh[f], phi_expected; rtol = 1.0e-12, atol = 1.0e-14)
    end
end

@testset "V&V: mesh flux — Δt scaling (halving Δt doubles flux)" begin
    d0 = SVector(0.1, 0.15)
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    ms_a = MeshMotionState(mesh)
    ms_b = MeshMotionState(mesh)

    nc = length(mesh.cell_volumes)
    for c in 1:nc
        ms_a.displacement[c] = d0
        ms_b.displacement[c] = d0
    end

    compute_mesh_flux!(ms_a, mesh, 0.2)
    compute_mesh_flux!(ms_b, mesh, 0.1)

    nf = size(mesh.face_cells, 2)
    count_nonzero = 0
    for f in 1:nf
        if abs(ms_a.phi_mesh[f]) > 1.0e-14
            @test isapprox(ms_b.phi_mesh[f] / ms_a.phi_mesh[f], 2.0; rtol = 1.0e-12)
            count_nonzero += 1
        end
    end
    @test count_nonzero > 10   # sanity: some faces have non-zero flux
end

@testset "V&V: mesh flux — closed-cell sum ≡ 0 under uniform translation" begin
    # For uniform d, Σ_f ε(c,f) <d, S_f> / dt = <d, Σ_f ε(c,f) S_f> / dt
    #                                         = <d, 0> / dt = 0,
    # by the divergence theorem applied to any closed cell.
    d0 = SVector(0.4, 0.2)
    dt = 0.1
    mesh = build_cartesian_unstructured_mesh(12, 12, 1.0, 1.0)
    ms = MeshMotionState(mesh)

    nc = length(mesh.cell_volumes)
    for c in 1:nc
        ms.displacement[c] = d0
    end
    compute_mesh_flux!(ms, mesh, dt)

    nf = size(mesh.face_cells, 2)
    net_per_cell = zeros(nc)
    for f in 1:nf
        P = mesh.face_cells[1, f]
        N = mesh.face_cells[2, f]
        net_per_cell[P] -= ms.phi_mesh[f]
        if N != 0
            net_per_cell[N] += ms.phi_mesh[f]
        end
    end
    @test maximum(abs, net_per_cell) < 1.0e-12
end
