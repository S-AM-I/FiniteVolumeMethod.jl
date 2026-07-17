# test/v_and_v_polyhedral_volumes.jl — polyhedral volume primitives V&V (v3.101)

using FiniteVolumeMethod
using FiniteVolumeMethod: volume_hex, volume_prism, volume_pyramid, volume_tet
using Test

include("TestHelpers.jl")

const _volume_tet = FiniteVolumeMethod.volume_tet
const _volume_hex = FiniteVolumeMethod.volume_hex
const _volume_prism = FiniteVolumeMethod.volume_prism
const _volume_pyramid = FiniteVolumeMethod.volume_pyramid
const _Node3D = FiniteVolumeMethod.Node3D

@testset "V&V: volume_tet — unit tetrahedron = 1/6" begin
    # Tet with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1) ⇒ V = 1/6.
    nodes = [_Node3D(0.0, 0.0, 0.0), _Node3D(1.0, 0.0, 0.0), _Node3D(0.0, 1.0, 0.0), _Node3D(0.0, 0.0, 1.0)]
    @test _volume_tet(nodes) ≈ 1.0 / 6.0 rtol = 1.0e-14
end

@testset "V&V: volume_tet — degenerate (coplanar) ⇒ zero" begin
    # Four coplanar vertices give zero tet volume.
    nodes = [_Node3D(0.0, 0.0, 0.0), _Node3D(1.0, 0.0, 0.0), _Node3D(0.0, 1.0, 0.0), _Node3D(1.0, 1.0, 0.0)]
    @test _volume_tet(nodes) < 1.0e-14
end

@testset "V&V: volume_tet — scaling invariance" begin
    # Scaling all vertices by factor α scales volume by α³.
    for alpha in (0.5, 2.0, 3.0)
        base = [_Node3D(0.0, 0.0, 0.0), _Node3D(1.0, 0.0, 0.0), _Node3D(0.0, 1.0, 0.0), _Node3D(0.0, 0.0, 1.0)]
        scaled = [_Node3D(n.x * alpha, n.y * alpha, n.z * alpha) for n in base]
        V_base = _volume_tet(base)
        V_scaled = _volume_tet(scaled)
        @test V_scaled ≈ alpha^3 * V_base rtol = 1.0e-14
    end
end

@testset "V&V: volume_tet — translation invariance" begin
    # Translating all vertices by the same vector preserves volume.
    base = [_Node3D(0.0, 0.0, 0.0), _Node3D(1.0, 0.0, 0.0), _Node3D(0.0, 1.0, 0.0), _Node3D(0.0, 0.0, 1.0)]
    dx, dy, dz = 5.0, -3.0, 7.2
    translated = [_Node3D(n.x + dx, n.y + dy, n.z + dz) for n in base]
    @test _volume_tet(base) ≈ _volume_tet(translated) rtol = 1.0e-14
end

@testset "V&V: volume_tet — reflection (vertex permutation) preserves magnitude" begin
    # Swapping any two vertices flips the signed volume, but volume_tet
    # returns |…|/6 so the result is unchanged.
    nodes = [_Node3D(0.0, 0.0, 0.0), _Node3D(1.0, 0.0, 0.0), _Node3D(0.0, 1.0, 0.0), _Node3D(0.0, 0.0, 1.0)]
    swap12 = [nodes[2], nodes[1], nodes[3], nodes[4]]
    @test _volume_tet(swap12) == _volume_tet(nodes)
end

@testset "V&V: volume_hex — unit hex = 1" begin
    # Axis-aligned unit hexahedron with Gmsh-ordered vertices.
    nodes = [
        _Node3D(0.0, 0.0, 0.0), _Node3D(1.0, 0.0, 0.0),
        _Node3D(1.0, 1.0, 0.0), _Node3D(0.0, 1.0, 0.0),
        _Node3D(0.0, 0.0, 1.0), _Node3D(1.0, 0.0, 1.0),
        _Node3D(1.0, 1.0, 1.0), _Node3D(0.0, 1.0, 1.0),
    ]
    @test _volume_hex(nodes) ≈ 1.0 rtol = 1.0e-14
end

@testset "V&V: volume_hex — Lx·Ly·Lz closed form" begin
    # Axis-aligned box of dimensions (Lx, Ly, Lz) has volume Lx·Ly·Lz.
    for (Lx, Ly, Lz) in ((2.0, 3.0, 5.0), (0.5, 1.0, 0.25), (10.0, 1.0, 0.1))
        nodes = [
            _Node3D(0.0, 0.0, 0.0), _Node3D(Lx, 0.0, 0.0),
            _Node3D(Lx, Ly, 0.0), _Node3D(0.0, Ly, 0.0),
            _Node3D(0.0, 0.0, Lz), _Node3D(Lx, 0.0, Lz),
            _Node3D(Lx, Ly, Lz), _Node3D(0.0, Ly, Lz),
        ]
        @test _volume_hex(nodes) ≈ Lx * Ly * Lz rtol = 1.0e-14
    end
end

@testset "V&V: volume_hex — scaling invariance α³" begin
    base = [
        _Node3D(0.0, 0.0, 0.0), _Node3D(1.0, 0.0, 0.0),
        _Node3D(1.0, 1.0, 0.0), _Node3D(0.0, 1.0, 0.0),
        _Node3D(0.0, 0.0, 1.0), _Node3D(1.0, 0.0, 1.0),
        _Node3D(1.0, 1.0, 1.0), _Node3D(0.0, 1.0, 1.0),
    ]
    for alpha in (0.5, 2.0, 3.0)
        scaled = [_Node3D(n.x * alpha, n.y * alpha, n.z * alpha) for n in base]
        @test _volume_hex(scaled) ≈ alpha^3 * _volume_hex(base) rtol = 1.0e-14
    end
end

@testset "V&V: volume_hex — wrong node count errors" begin
    @test_throws ErrorException _volume_hex(
        [_Node3D(0.0, 0.0, 0.0), _Node3D(1.0, 0.0, 0.0)]
    )
end

@testset "V&V: volume_pyramid — unit pyramid = 1/3" begin
    # Unit-square base + apex at (0.5, 0.5, 1) ⇒ V = (1/3)·1·1 = 1/3.
    nodes = [
        _Node3D(0.0, 0.0, 0.0), _Node3D(1.0, 0.0, 0.0),
        _Node3D(1.0, 1.0, 0.0), _Node3D(0.0, 1.0, 0.0),
        _Node3D(0.5, 0.5, 1.0),
    ]
    @test _volume_pyramid(nodes) ≈ 1.0 / 3.0 rtol = 1.0e-14
end

@testset "V&V: volume_pyramid — h linear scaling" begin
    # V = (1/3)·base_area·h ⇒ V is linear in apex height h.
    for h in (0.5, 1.0, 2.5, 5.0)
        nodes = [
            _Node3D(0.0, 0.0, 0.0), _Node3D(1.0, 0.0, 0.0),
            _Node3D(1.0, 1.0, 0.0), _Node3D(0.0, 1.0, 0.0),
            _Node3D(0.5, 0.5, h),
        ]
        @test _volume_pyramid(nodes) ≈ h / 3.0 rtol = 1.0e-14
    end
end

@testset "V&V: volume_pyramid / volume_prism — wrong node count errors" begin
    @test_throws ErrorException _volume_pyramid(
        [_Node3D(0.0, 0.0, 0.0), _Node3D(1.0, 0.0, 0.0)]
    )
    @test_throws ErrorException _volume_prism(
        [_Node3D(0.0, 0.0, 0.0), _Node3D(1.0, 0.0, 0.0)]
    )
end

@testset "V&V: volume_prism — unit-triangle prism = 0.5·h" begin
    # Triangular prism with unit right-triangle base (area 0.5) and
    # height h in z has volume 0.5·h.
    for h in (0.5, 1.0, 2.0)
        nodes = [
            _Node3D(0.0, 0.0, 0.0), _Node3D(1.0, 0.0, 0.0), _Node3D(0.0, 1.0, 0.0),
            _Node3D(0.0, 0.0, h), _Node3D(1.0, 0.0, h), _Node3D(0.0, 1.0, h),
        ]
        @test _volume_prism(nodes) ≈ 0.5 * h rtol = 1.0e-14
    end
end
