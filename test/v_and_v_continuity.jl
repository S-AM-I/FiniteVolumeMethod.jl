# test/v_and_v_continuity.jl — continuity residual primitives V&V (v3.100)

using FiniteVolumeMethod
using StaticArrays
using Test

include("TestHelpers.jl")

const _continuity_res = FiniteVolumeMethod.continuity_residual
const _continuity_interior = FiniteVolumeMethod.continuity_residual_interior

@testset "V&V: continuity_residual — zero flux ⇒ zero residual" begin
    # With phi ≡ 0, every cell imbalance is zero ⇒ residual = 0.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    state = IncompressibleState(mesh)
    for f in eachindex(state.phi.values)
        state.phi.values[f] = 0.0
    end
    @test _continuity_res(state, mesh) == 0.0
end

@testset "V&V: continuity_residual — zero flux ⇒ interior zero residual" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    state = IncompressibleState(mesh)
    for f in eachindex(state.phi.values)
        state.phi.values[f] = 0.0
    end
    @test _continuity_interior(state, mesh) == 0.0
end

@testset "V&V: continuity_residual ≥ 0 (L¹ norm)" begin
    # residual is a sum of absolute values — always non-negative.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    state = IncompressibleState(mesh)
    for f in eachindex(state.phi.values)
        state.phi.values[f] = 0.1 * sin(f)
    end
    r = _continuity_res(state, mesh)
    @test r >= 0.0
    r_int = _continuity_interior(state, mesh)
    @test r_int >= 0.0
    # Interior residual cannot exceed the full residual (strict subset).
    @test r_int <= r + 1.0e-12
end

@testset "V&V: continuity_residual — linear flux scaling" begin
    # continuity_residual is linear in phi (|Σε·αφ| = α·|Σε·φ|), so
    # doubling phi doubles the residual.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    state1 = IncompressibleState(mesh)
    state2 = IncompressibleState(mesh)
    for f in eachindex(state1.phi.values)
        state1.phi.values[f] = 0.1 * sin(f)
        state2.phi.values[f] = 0.2 * sin(f)
    end
    r1 = _continuity_res(state1, mesh)
    r2 = _continuity_res(state2, mesh)
    @test r2 ≈ 2.0 * r1 rtol = 1.0e-14
end

@testset "V&V: continuity_residual — single-cell imbalance identity" begin
    # Construct a phi field with a single positive flux leaving cell 1
    # through one face, with zero elsewhere. Residual must equal
    # |phi_f| exactly.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    state = IncompressibleState(mesh)
    for f in eachindex(state.phi.values)
        state.phi.values[f] = 0.0
    end
    # Pick one internal face.
    target_face = 0
    for f in eachindex(state.phi.values)
        if FiniteVolumeMethod.is_internal_face(mesh, f)
            target_face = f
            break
        end
    end
    @test target_face > 0
    state.phi.values[target_face] = 0.5
    # Both owner and neighbour contribute |0.5| each ⇒ residual = 1.0.
    @test _continuity_res(state, mesh) ≈ 1.0 rtol = 1.0e-14
end

@testset "V&V: continuity_residual_interior — larger band excludes more cells" begin
    # Default band is 0.1; a band of 0.45 excludes almost everything.
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    state = IncompressibleState(mesh)
    for f in eachindex(state.phi.values)
        state.phi.values[f] = 0.1 * sin(f)
    end
    r_small = _continuity_interior(state, mesh, 0.1)
    r_large = _continuity_interior(state, mesh, 0.45)
    # Larger band ⇒ stricter interior subset ⇒ smaller residual.
    @test r_large <= r_small + 1.0e-12
end

@testset "V&V: compute_max_courant — zero flow ⇒ zero" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    state = IncompressibleState(mesh)
    for f in eachindex(state.phi.values)
        state.phi.values[f] = 0.0
    end
    @test FiniteVolumeMethod.compute_max_courant(state, mesh, 0.1) == 0.0
end

@testset "V&V: compute_max_courant — dt-linear scaling" begin
    # Co_max ∝ dt (identity-direct property of the maximum).
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    state = IncompressibleState(mesh)
    for f in eachindex(state.phi.values)
        state.phi.values[f] = 0.5 + 0.1 * f
    end
    c1 = FiniteVolumeMethod.compute_max_courant(state, mesh, 0.01)
    c2 = FiniteVolumeMethod.compute_max_courant(state, mesh, 0.02)
    @test c2 ≈ 2.0 * c1 rtol = 1.0e-14
end

@testset "V&V: compute_max_courant — |phi|-invariant (sign-symmetric)" begin
    # Flipping the sign of every flux cannot change max|phi|·dt/V.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    state_pos = IncompressibleState(mesh)
    state_neg = IncompressibleState(mesh)
    for f in eachindex(state_pos.phi.values)
        state_pos.phi.values[f] = 0.5 + 0.1 * f
        state_neg.phi.values[f] = -(0.5 + 0.1 * f)
    end
    dt = 0.01
    @test FiniteVolumeMethod.compute_max_courant(state_pos, mesh, dt) ==
        FiniteVolumeMethod.compute_max_courant(state_neg, mesh, dt)
end
