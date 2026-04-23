# test/sciml_contract_uniform.jl — Stage 1d+1f contract test
#
# Asserts that every solver family exposes a common umbrella-type surface:
#   - mesh types subtype `AbstractFiniteVolumeMesh` and answer `dim_of`,
#     `n_cells`, `n_faces` without any family-specific knowledge.
#   - BC types subtype `AbstractFVMBoundaryCondition`.
#
# This is the minimum generic contract downstream consumers can rely on
# when dispatching across parabolic, hyperbolic, and collocated families.

using FiniteVolumeMethod
using Test
using DelaunayTriangulation

include("TestHelpers.jl")

@testset "Stage 1d: AbstractFiniteVolumeMesh umbrella" begin
    # Collocated unstructured mesh
    um = build_cartesian_unstructured_mesh(4, 3, 1.0, 1.0)
    @test um isa AbstractFiniteVolumeMesh
    @test um isa AbstractFVMMesh
    @test dim_of(um) == 2
    @test n_cells(um) == 12
    # 2 × (internal) + 4 sides × boundary = ... check against mesh directly
    @test n_faces(um) == size(um.face_cells, 2)

    # Parabolic vertex-centered (FVMGeometry)
    tri = triangulate_rectangle(0.0, 1.0, 0.0, 1.0, 5, 5; single_boundary = true)
    geo = FVMGeometry(tri)
    @test geo isa AbstractFiniteVolumeMesh
    @test dim_of(geo) == 2
    @test n_cells(geo) == 25
    @test n_faces(geo) > 0

    # Hyperbolic structured meshes
    sm2d = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 6, 4)
    @test sm2d isa AbstractFiniteVolumeMesh
    @test dim_of(sm2d) == 2
    @test ncells(sm2d) == 6 * 4  # hyperbolic uses lowercase `ncells`
end

@testset "Stage 1d: AbstractFVMBoundaryCondition umbrella" begin
    # Parabolic BCs
    @test ParabolicDirichlet(1.0) isa AbstractFVMBoundaryCondition
    @test ParabolicNeumann(0.0) isa AbstractFVMBoundaryCondition
    @test ParabolicRobin(1.0, 2.0, 3.0) isa AbstractFVMBoundaryCondition

    # Collocated BCs (subtype AbstractBoundaryCondition which subtypes the umbrella)
    @test NoSlipWallBC() isa AbstractFVMBoundaryCondition
    @test SlipWallBC() isa AbstractFVMBoundaryCondition
    @test FixedPressureBC(0.0) isa AbstractFVMBoundaryCondition

    # Hyperbolic BCs
    @test TransmissiveBC() isa AbstractFVMBoundaryCondition
    @test ReflectiveBC() isa AbstractFVMBoundaryCondition
end

@testset "Stage 1d: Generic dispatch on umbrella type" begin
    # A downstream consumer can write one method on `::AbstractFiniteVolumeMesh`
    # and have it match every mesh family without knowing concrete types.
    summarize(m::AbstractFiniteVolumeMesh) = (dim_of(m), n_cells(m))

    um = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
    tri = triangulate_rectangle(0.0, 1.0, 0.0, 1.0, 4, 4; single_boundary = true)
    geo = FVMGeometry(tri)

    @test summarize(um) == (2, 9)
    @test summarize(geo) == (2, 16)
end
