# test/v_and_v_gcl.jl — Geometric Conservation Law V&V (v3.14)
#
# The GCL is the cornerstone invariance of ALE transport: the
# discrete swept-face flux `phi_mesh` must satisfy, for every cell,
#
#   (V_new - V_old) / Δt  =  Σ_f ε(c, f) · phi_mesh[f].
#
# Failure of this identity manifests as artificial mass/energy
# creation under mesh motion — a notorious source of spurious
# transport in legacy ALE codes. This V&V establishes that the
# present implementation recovers GCL exactness (to round-off) on
# three analytically tractable motion patterns:
#
#   1. Zero motion               — trivial fixed point.
#   2. Rigid-body translation    — exact by the divergence theorem.
#   3. Isotropic linear scaling  — finite volume change, finite flux,
#                                  exact volume identity for linear d(x).
#
# Evidence for promoting `dynamic_mesh` from `experimental`/
# `smoke_tested` to `provisional`/`convergence_verified`.

using FiniteVolumeMethod
using FiniteVolumeMethod: MeshMotionState, compute_displacement!, update_mesh!, verify_gcl
using LinearAlgebra: norm
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: GCL — zero-motion invariance" begin
    # No displacement ⇒ no geometry change, phi_mesh ≡ 0, and the
    # GCL residual is identically zero at every cell.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    motion = SolidBodyMotion{2, Float64}(t -> SVector(0.0, 0.0))

    xc_before = copy(mesh.cell_centers)
    V_before = copy(mesh.cell_volumes)

    dt = 1.0e-3
    compute_displacement!(ms, motion, mesh, 1.0)
    update_mesh!(mesh, ms, dt)

    # Geometry is machine-precision invariant.
    @test mesh.cell_centers == xc_before
    @test all(isapprox.(mesh.cell_volumes, V_before; atol = 1.0e-14))

    # phi_mesh is identically zero.
    @test all(==(0.0), ms.phi_mesh)

    # GCL residual is identically zero.
    _, max_res = verify_gcl(ms.phi_mesh, ms.V_old, mesh.cell_volumes, mesh, dt)
    @test max_res < 1.0e-14
end

@testset "V&V: GCL — rigid translation preserves volumes exactly" begin
    # Uniform displacement d moves every cell by the same vector.
    # Cell volumes are preserved exactly (volumes are translation
    # invariant). The face-flux is phi_mesh[f] = <d, S_f>/Δt; since
    # Σ_{f ∈ ∂c} ε(c,f) S_f = 0 (divergence theorem on a closed cell),
    # the net face flux per cell is zero — GCL holds to round-off.
    mesh = build_cartesian_unstructured_mesh(10, 10, 1.0, 1.0)
    ms = MeshMotionState(mesh)
    motion = SolidBodyMotion{2, Float64}(t -> SVector(0.25 * t, -0.1 * t))

    V_before = copy(mesh.cell_volumes)

    dt = 0.1
    compute_displacement!(ms, motion, mesh, 1.0)
    update_mesh!(mesh, ms, dt)

    # Volumes preserved to within round-off (the _approximate_volumes!
    # routine uses the divergence of a uniform field, which is
    # exactly zero in continuous calculus and O(h²) after
    # discretization on a Cartesian grid).
    for c in 1:length(V_before)
        @test isapprox(mesh.cell_volumes[c], V_before[c]; atol = 1.0e-12)
    end

    # GCL residual is identically zero (not merely small).
    _, max_res = verify_gcl(ms.phi_mesh, ms.V_old, mesh.cell_volumes, mesh, dt)
    @test max_res < 1.0e-11

    # phi_mesh on boundary faces of opposite sides must sum to zero
    # per cell (closed-cell invariant). Check by re-summing:
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    net_flux = zeros(nc)
    for f in 1:nf
        P = mesh.face_cells[1, f]
        N = mesh.face_cells[2, f]
        net_flux[P] -= ms.phi_mesh[f]
        if N != 0
            net_flux[N] += ms.phi_mesh[f]
        end
    end
    @test maximum(abs, net_flux) < 1.0e-12
end

@testset "V&V: GCL — isotropic scaling d(x) = α(x - x₀)" begin
    # Linear displacement field d(x) = α(x - c) about the domain
    # center c = (0.5, 0.5) produces finite volume change per cell:
    # for linear d, div(d) = Dim·α exactly (continuous and
    # discrete via midpoint-face). So
    #
    #   ΔV / V_old = Σ_f <d_f, S_f> / V_old = div(d) = Dim·α
    #
    # and the mesh-update routine should recover V_new = V_old(1 + Dim·α)
    # to O(α²) accuracy (the 1st-order Taylor truncation is what the
    # `_approximate_volumes!` routine implements). The GCL residual
    # should remain at round-off because phi_mesh is consistent with
    # the approximated ΔV by construction.
    alpha = 0.05
    Nx = 16

    mesh = build_cartesian_unstructured_mesh(Nx, Nx, 1.0, 1.0)
    ms = MeshMotionState(mesh)

    # Apply d(x) = α(x - 0.5).
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        x = mesh.cell_centers[1, c] - 0.5
        y = mesh.cell_centers[2, c] - 0.5
        ms.displacement[c] = SVector(alpha * x, alpha * y)
    end

    V_before = copy(mesh.cell_volumes)
    dt = 0.1
    update_mesh!(mesh, ms, dt)

    # GCL residual is machine-zero by construction: phi_mesh is built
    # from the same divergence that approximates V_new - V_old.
    _, max_res = verify_gcl(ms.phi_mesh, ms.V_old, mesh.cell_volumes, mesh, dt)
    V_mean = sum(V_before) / nc
    @test max_res < 1.0e-10 * V_mean / dt

    # Leading-order volume ratio matches 1 + Dim·α with a small
    # α²-correction tolerance.
    Dim = 2
    V_ratio = sum(mesh.cell_volumes) / sum(V_before)
    @test abs(V_ratio - (1 + Dim * alpha)) < 5 * alpha^2
end

@testset "V&V: GCL — translation monotone order-of-accuracy in h" begin
    # GCL residuals under uniform translation should scale as
    # machine-precision (the closed-cell identity is exact), so
    # as we refine the mesh, the residual should remain bounded by
    # a small multiple of eps. This test checks non-regression of
    # the exact invariant across mesh refinement.
    for N in (8, 16, 32)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        ms = MeshMotionState(mesh)
        motion = SolidBodyMotion{2, Float64}(t -> SVector(0.3 * t, 0.2 * t))

        dt = 0.1
        compute_displacement!(ms, motion, mesh, 1.0)
        update_mesh!(mesh, ms, dt)

        _, max_res = verify_gcl(ms.phi_mesh, ms.V_old, mesh.cell_volumes, mesh, dt)

        # Per-cell residual scales with cell-volume/dt; express in
        # non-dimensional terms. Cell volume on an N×N Cartesian mesh
        # is (1/N)² = 1/N². We expect max_res · dt / V_cell ≤ 1e-10.
        V_cell = 1.0 / (N * N)
        @test max_res * dt / V_cell < 1.0e-10
    end
end
