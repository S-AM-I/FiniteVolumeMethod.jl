# test/v_and_v_gcl_rotation.jl — GCL under rotational mesh motion (v3.29)
#
# Second analytical benchmark for `dynamic_mesh`. The first
# benchmark (v3.14, GCL patterns) tested zero motion, rigid
# translation, and isotropic scaling — all motion patterns where
# `_approximate_volumes!` gives the exact answer. This one
# extends coverage to **infinitesimal rigid rotation**, a
# non-uniform displacement field that a naive divergence-based
# volume approximation could mis-handle.
#
# For 2D rotation by a small angle θ about the domain center:
#
#   d(x, y) = (−θ·(y − y_0), θ·(x − x_0))
#
# which has ∇·d = 0 exactly ⇒ volumes preserved to first order.
# The cross-product with face area vectors, Σ_f ε(c,f)·<d_f, S_f>,
# also vanishes per cell by the closed-cell divergence theorem
# applied to the linear field d, so the GCL residual is
# machine-zero at every cell — the same exactness level as the
# v3.14 translation test.
#
# Evidence toward future `stable` promotion of `dynamic_mesh`.

using FiniteVolumeMethod
using FiniteVolumeMethod: MeshMotionState, update_mesh!, verify_gcl
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

function set_rotation_displacement!(ms, mesh, theta, cx, cy)
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        x = mesh.cell_centers[1, c] - cx
        y = mesh.cell_centers[2, c] - cy
        ms.displacement[c] = SVector(-theta * y, theta * x)
    end
    return nothing
end

@testset "V&V: GCL rotation — small-angle volume preservation" begin
    # θ = 0.05 rad ≈ 2.9°. At this magnitude the linear
    # approximation d = θ·(−y, x) closely tracks the exact
    # rotation (R − I = [[cos θ − 1, −sin θ], [sin θ, cos θ − 1]])
    # but is not identical. `_approximate_volumes!` uses
    # V_new ≈ V_old·(1 + div(d)); since ∇·d = 0 exactly for
    # linear rotational displacement, V_new ≡ V_old.
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    set_rotation_displacement!(ms, mesh, 0.05, 0.5, 0.5)

    V_before = copy(mesh.cell_volumes)
    dt = 0.1
    update_mesh!(mesh, ms, dt)

    # Every cell volume preserved (linear displacement has zero
    # divergence, which is what `_approximate_volumes!` measures).
    for c in 1:length(V_before)
        @test isapprox(mesh.cell_volumes[c], V_before[c]; atol = 1.0e-12)
    end

    # Total volume preserved (trivially from above).
    @test isapprox(sum(mesh.cell_volumes), sum(V_before); rtol = 1.0e-14)
end

@testset "V&V: GCL rotation — residual machine-zero at every cell" begin
    # Σ_f ε(c,f)·<d_f, S_f> = 0 per cell for any linear
    # displacement, by the divergence theorem. Since
    # `phi_mesh[f] = <d_f, S_f>/dt`, the signed face-flux
    # sum divided by dt equals (V_new − V_old)/dt (which is
    # zero here), so GCL holds to round-off.
    mesh = build_cartesian_unstructured_mesh(20, 20, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    set_rotation_displacement!(ms, mesh, 0.03, 0.5, 0.5)

    dt = 0.1
    update_mesh!(mesh, ms, dt)
    _, max_res = verify_gcl(ms.phi_mesh, ms.V_old, mesh.cell_volumes, mesh, dt)

    V_cell_mean = sum(ms.V_old) / length(ms.V_old)
    @test max_res * dt / V_cell_mean < 1.0e-10
end

@testset "V&V: GCL rotation — residual invariance across refinement" begin
    # Rotation is a smooth linear displacement; the divergence-
    # theorem identity is mesh-independent. Refinement must not
    # degrade GCL exactness.
    for N in (8, 16, 32)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        ms = MeshMotionState(mesh)
        set_rotation_displacement!(ms, mesh, 0.02, 0.5, 0.5)

        dt = 0.1
        update_mesh!(mesh, ms, dt)

        _, max_res = verify_gcl(ms.phi_mesh, ms.V_old, mesh.cell_volumes, mesh, dt)

        V_cell = 1.0 / (N * N)
        @test max_res * dt / V_cell < 1.0e-10
    end
end

@testset "V&V: GCL rotation — cell centers rotate as prescribed" begin
    theta = 0.01
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    ms = MeshMotionState(mesh)

    # Save unrotated coords.
    x_before = copy(mesh.cell_centers)

    set_rotation_displacement!(ms, mesh, theta, 0.5, 0.5)
    update_mesh!(mesh, ms, 0.1)

    # Each cell center should land at x_before + d.
    for c in 1:length(mesh.cell_volumes)
        x_old = x_before[1, c] - 0.5
        y_old = x_before[2, c] - 0.5
        x_new_expected = (x_before[1, c] - 0.5) - theta * y_old + 0.5
        y_new_expected = (x_before[2, c] - 0.5) + theta * x_old + 0.5
        @test isapprox(mesh.cell_centers[1, c], x_new_expected; atol = 1.0e-14)
        @test isapprox(mesh.cell_centers[2, c], y_new_expected; atol = 1.0e-14)
    end
end
