# test/v_and_v_verify_gcl.jl — verify_gcl diagnostic V&V (v3.95)

using FiniteVolumeMethod
using Test

include("TestHelpers.jl")

const _verify_gcl = FiniteVolumeMethod.verify_gcl

@testset "V&V: verify_gcl — zero motion ⇒ zero residual" begin
    # V_new = V_old, phi_mesh ≡ 0 ⇒ residuals all exactly zero.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    V_old = copy(mesh.cell_volumes)
    V_new = copy(mesh.cell_volumes)
    phi_mesh = zeros(Float64, nf)
    residuals, max_res = _verify_gcl(phi_mesh, V_old, V_new, mesh, 1.0e-2)
    @test length(residuals) == nc
    @test max_res == 0.0
    for r in residuals
        @test r == 0.0
    end
end

@testset "V&V: verify_gcl — residual = (ΔV/dt) with zero phi_mesh" begin
    # Set V_new = V_old + δV per cell with phi_mesh ≡ 0.
    # Residual[c] should equal δV[c] / dt exactly.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    V_old = copy(mesh.cell_volumes)
    V_new = [V_old[c] + 0.01 * c for c in 1:nc]
    phi_mesh = zeros(Float64, nf)
    dt = 0.5
    residuals, _ = _verify_gcl(phi_mesh, V_old, V_new, mesh, dt)
    for c in 1:nc
        @test residuals[c] ≈ (0.01 * c) / dt rtol = 1.0e-14
    end
end

@testset "V&V: verify_gcl — dt scaling of residual" begin
    # For fixed V_old, V_new, phi_mesh, halving dt doubles |residuals|.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    V_old = copy(mesh.cell_volumes)
    V_new = [V_old[c] + 0.02 for c in 1:nc]
    phi_mesh = zeros(Float64, nf)
    residuals1, _ = _verify_gcl(phi_mesh, V_old, V_new, mesh, 0.1)
    residuals2, _ = _verify_gcl(phi_mesh, V_old, V_new, mesh, 0.05)
    for c in 1:nc
        @test residuals2[c] ≈ 2.0 * residuals1[c] rtol = 1.0e-14
    end
end

@testset "V&V: verify_gcl — consistent phi_mesh gives zero residual" begin
    # Construct phi_mesh such that Σ_f ε(c,f)·phi_mesh[f] = (V_new-V_old)/dt
    # exactly for every cell, then check that verify_gcl returns zero.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    V_old = copy(mesh.cell_volumes)
    # Uniform isotropic expansion: V_new = V_old · (1 + α).
    alpha = 0.01
    V_new = V_old .* (1.0 + alpha)
    dt = 0.1
    # For uniform volume change, the "consistent" phi_mesh for a
    # Cartesian mesh distributes (V_new - V_old)/dt equally among the
    # four faces of each cell. We construct this directly by solving
    # the per-face constraint from the cell-accumulate. An explicit
    # closed form: for a uniform mesh with all internal faces shared by
    # two owner cells of equal sign-flip pattern, phi_mesh ≡ 0 on
    # internal faces and phi_mesh = (α·V_cell) / (n_boundary_faces_per_cell · dt)
    # at boundary faces captures the expansion. But this is tricky —
    # instead, directly verify the identity V_new[c] - V_old[c] =
    # dt · Σ_f ε(c,f)·phi_mesh[f] has zero residual iff phi_mesh is
    # chosen self-consistently.
    #
    # Here we just check the trivial case V_new = V_old + dt·Σ(phi)
    # holds when we construct phi_mesh from a particular δV field.
    phi_mesh = zeros(Float64, nf)
    # Add an arbitrary phi_mesh.
    for f in 1:nf
        phi_mesh[f] = 0.01 * f
    end
    # Reverse-engineer V_new consistent with phi_mesh and dt = 0.1.
    V_new_consistent = copy(V_old)
    for f in 1:nf
        P = mesh.face_cells[1, f]
        N = mesh.face_cells[2, f]
        V_new_consistent[P] += phi_mesh[f] * dt
        if N != 0
            V_new_consistent[N] -= phi_mesh[f] * dt
        end
    end
    residuals, max_res = _verify_gcl(phi_mesh, V_old, V_new_consistent, mesh, dt)
    @test max_res < 1.0e-12
    for r in residuals
        @test abs(r) < 1.0e-12
    end
end

@testset "V&V: verify_gcl — error on length mismatch" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    V_old = copy(mesh.cell_volumes)
    V_new = copy(mesh.cell_volumes)
    phi_mesh = zeros(Float64, nf)
    # Wrong V_old length.
    @test_throws ErrorException _verify_gcl(
        phi_mesh, V_old[1:(nc - 1)], V_new, mesh, 0.1,
    )
    # Wrong V_new length.
    @test_throws ErrorException _verify_gcl(
        phi_mesh, V_old, V_new[1:(nc - 1)], mesh, 0.1,
    )
    # Wrong phi_mesh length.
    @test_throws ErrorException _verify_gcl(
        zeros(Float64, nf - 1), V_old, V_new, mesh, 0.1,
    )
end

@testset "V&V: verify_gcl — error on dt ≤ 0" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nf = size(mesh.face_cells, 2)
    V_old = copy(mesh.cell_volumes)
    V_new = copy(mesh.cell_volumes)
    phi_mesh = zeros(Float64, nf)
    @test_throws ErrorException _verify_gcl(phi_mesh, V_old, V_new, mesh, 0.0)
    @test_throws ErrorException _verify_gcl(phi_mesh, V_old, V_new, mesh, -0.1)
end

@testset "V&V: verify_gcl — max_residual = maximum(abs, residuals)" begin
    # The second return value must match maximum(abs, residuals) exactly.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    V_old = copy(mesh.cell_volumes)
    V_new = [V_old[c] + 0.03 * sin(c) for c in 1:nc]
    phi_mesh = zeros(Float64, nf)
    residuals, max_res = _verify_gcl(phi_mesh, V_old, V_new, mesh, 0.1)
    @test max_res == maximum(abs, residuals)
end
