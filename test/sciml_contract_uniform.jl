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
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC, RobinBC
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
    @test DirichletBC(1.0) isa AbstractFVMBoundaryCondition
    @test NeumannBC(0.0) isa AbstractFVMBoundaryCondition
    @test RobinBC(1.0, 2.0, 3.0) isa AbstractFVMBoundaryCondition

    # Collocated BCs (subtype AbstractBoundaryCondition which subtypes the umbrella)
    @test NoSlipWallBC() isa AbstractFVMBoundaryCondition
    @test SlipWallBC() isa AbstractFVMBoundaryCondition
    @test FixedPressureBC(0.0) isa AbstractFVMBoundaryCondition

    # Hyperbolic BCs
    @test TransmissiveBC() isa AbstractFVMBoundaryCondition
    @test ReflectiveBC() isa AbstractFVMBoundaryCondition
end

@testset "Stage 1e: Extensible SciMLStructures.Tunable schema" begin
    using LinearSolve
    using SciMLStructures: Tunable, canonicalize, replace as ss_replace

    mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(),
        :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(),
        :top => NoSlipWallBC(),
    )
    prob = IncompressibleProblem(mesh, bcs, SIMPLE(); nu = 1.0e-3, density = 1.25)

    # Named schema matches the canonical vector length.
    names = FiniteVolumeMethod.tunable_names(prob)
    @test names == [:nu, :density, :alpha_U, :alpha_p, :tolerance]

    # NamedTuple introspection matches the registered getters.
    nt = FiniteVolumeMethod.tunable_namedtuple(prob)
    @test nt.nu == 1.0e-3
    @test nt.density == 1.25
    @test nt.alpha_U == prob.algorithm.alpha_U
    @test nt.alpha_p == prob.algorithm.alpha_p
    @test nt.tolerance == prob.algorithm.tolerance

    # Canonicalize returns the same-ordered values.
    vals, repack, aliasing = canonicalize(Tunable(), prob)
    @test length(vals) == length(names)
    @test vals ≈ [nt[n] for n in names]
    @test !aliasing

    # repack(new_vals) reconstructs with updated values.
    new_vals = copy(vals)
    new_vals[1] = 5.0e-4  # new nu
    new_vals[3] = 0.8     # new alpha_U
    new_prob = repack(new_vals)
    @test new_prob.nu == 5.0e-4
    @test new_prob.algorithm.alpha_U == 0.8
    @test new_prob.density == 1.25  # untouched

    # Adding a new tunable at runtime extends the schema without breaking
    # existing consumers (the new entry just appends to the end).
    # Use a dummy "roughness" tunable on a scratch scalar field of the problem.
    const_box = Ref(0.7)
    FiniteVolumeMethod.register_tunable!(
        IncompressibleProblem, :demo_const,
        _ -> const_box[],
        (p, v) -> (const_box[] = v; p),
    )
    names2 = FiniteVolumeMethod.tunable_names(prob)
    @test :demo_const in names2
    @test length(FiniteVolumeMethod.tunable_namedtuple(prob)) == length(names2)

    # Clean up the registry so the test is idempotent.
    entries = FiniteVolumeMethod._TUNABLE_REGISTRY[IncompressibleProblem]
    filter!(e -> e.name !== :demo_const, entries)
end

@testset "Stage 1h: AbstractLinearOperator wrapper" begin
    using LinearAlgebra: mul!
    using SparseArrays: sparse

    A = sparse([1 2 0; 0 3 4; 5 0 6] .|> Float64)
    op = SparseMatrixLinearOperator(A)

    @test op isa AbstractLinearOperator{Float64}
    @test size(op) == (3, 3)
    @test size(op, 1) == 3
    @test eltype(op) == Float64

    # mul!(y, op, x) matches mul!(y, A, x)
    x = [1.0, 2.0, 3.0]
    y_op = zeros(3)
    y_A = zeros(3)
    mul!(y_op, op, x)
    mul!(y_A, A, x)
    @test y_op ≈ y_A

    # underlying_matrix round-trips
    @test underlying_matrix(op) === A

    # as_linear_operator idempotent on operators; wraps raw matrices
    @test as_linear_operator(op) === op
    wrapped = as_linear_operator(A)
    @test wrapped isa SparseMatrixLinearOperator
    @test underlying_matrix(wrapped) === A

    # MatrixFreeError path: an abstract subtype without underlying_matrix overload
    # should throw (caught by catch_backtrace in user code). Simulate by
    # defining a no-op matrix-free subtype.
    struct _TestMatrixFreeOp{T} <: AbstractLinearOperator{T}
        n::Int
    end
    @test_throws MatrixFreeError underlying_matrix(_TestMatrixFreeOp{Float64}(4))
end

@testset "Stage 1g: Abstract-array-parameterized field types" begin
    mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
    nc = length(mesh.cell_volumes)

    # Default constructors still give Vector-backed fields.
    s = CollocatedScalarField(:phi, mesh; value = 0.0)
    @test s.internal isa Vector{Float64}
    @test s isa CollocatedScalarField{Float64}  # UnionAll dispatch matches

    v = CollocatedVectorField(:U, mesh)
    @test v.internal isa Vector  # concrete SVector element type
    @test v isa CollocatedVectorField{2, Float64}

    phi = FaceFluxField(:phi, mesh; value = 0.0)
    @test phi.values isa Vector{Float64}
    @test phi isa FaceFluxField{Float64}

    # Custom container type — use a `Base.ReinterpretArray` wrapper to
    # prove the API accepts any `AbstractVector{T}` for `internal` /
    # `boundary` as long as types agree. A real GPU port would swap in a
    # `CuVector{T}` here without any other change.
    bface_idxs = [f for f in 1:size(mesh.face_cells, 2) if mesh.face_cells[2, f] == 0]
    internal = view(zeros(Float64, nc), :)
    boundary = view(zeros(Float64, length(bface_idxs)), :)
    s2 = CollocatedScalarField{Float64}(:alt, internal, boundary, bface_idxs)
    @test s2 isa CollocatedScalarField{Float64}
    @test s2.internal === internal
    @test s2.boundary === boundary
end

@testset "Stage 1f: is_fvm_solution trait + AbstractFVMSolution" begin
    # Non-FVM values: false
    @test !is_fvm_solution(42)
    @test !is_fvm_solution("hello")
    @test !is_fvm_solution(nothing)

    # IncompressibleSolution: true
    mesh = build_cartesian_unstructured_mesh(3, 3, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(),
        :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(),
        :top => NoSlipWallBC(),
    )
    prob = IncompressibleProblem(mesh, bcs, SIMPLE(); nu = 1.0e-3, density = 1.0)
    sol = solve(prob, SIMPLE())
    @test sol isa IncompressibleSolution
    @test sol isa AbstractFVMSolution
    @test is_fvm_solution(sol)
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
