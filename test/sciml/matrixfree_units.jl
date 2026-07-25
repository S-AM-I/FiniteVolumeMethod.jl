# test/sciml/matrixfree_units.jl — matrix-free linear operator interface and

using FiniteVolumeMethod
using FiniteVolumeMethod: AbstractLinearOperator, MatrixFreeError, MatrixFreeLinearOperator, SparseMatrixLinearOperator, as_si_density, as_si_temperature, as_si_velocity, as_si_viscosity, is_dimensionless, strip_units, underlying_matrix
using Test
using LinearAlgebra
using SparseArrays: sparse

@testset "MatrixFreeLinearOperator implements AbstractLinearOperator interface" begin
    # Wrap a simple diagonal matrix-vector product y_i = 2·x_i as a
    # matrix-free operator.
    n = 5
    matvec! = (y, x) -> (
        @inbounds for i in 1:n
            y[i] = 2 * x[i]
        end; nothing
    )
    op = MatrixFreeLinearOperator{Float64}(n, matvec!)

    @test op isa AbstractLinearOperator{Float64}
    @test size(op) == (5, 5)
    @test size(op, 1) == 5
    @test size(op, 2) == 5
    @test eltype(op) == Float64

    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = zeros(5)
    mul!(y, op, x)
    @test y == 2x

    # Non-square operator
    rect = MatrixFreeLinearOperator{Float64}(3, 5, (y, x) -> (y .= x[1:3]; nothing))
    @test size(rect) == (3, 5)
    y3 = zeros(3)
    mul!(y3, rect, x)
    @test y3 == x[1:3]

    # underlying_matrix throws MatrixFreeError
    @test_throws MatrixFreeError underlying_matrix(op)
end

@testset "MatrixFreeLinearOperator equivalent to SparseMatrixLinearOperator" begin
    A = sparse([2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0])
    sparse_op = SparseMatrixLinearOperator(A)
    matfree_op = MatrixFreeLinearOperator{Float64}(
        3, (y, x) -> (y[1] = 2x[1]; y[2] = 3x[2]; y[3] = 4x[3]; nothing),
    )

    x = [1.0, 1.0, 1.0]
    y_sparse = zeros(3); y_free = zeros(3)
    mul!(y_sparse, sparse_op, x)
    mul!(y_free, matfree_op, x)
    @test y_sparse == y_free
end

@testset "strip_units handles plain reals and dimensionless inputs" begin
    # Plain Real inputs pass through.
    @test strip_units(2.5, 1.0) ≈ 2.5
    @test strip_units(0.0, 1.0) == 0.0
    @test strip_units(100.0, 10.0) ≈ 10.0

    # Non-Real target_scale: 1.0 is the default dimensionless case.
    @test as_si_velocity(5.0) == 5.0
    @test as_si_density(1.2) == 1.2
    @test as_si_viscosity(1.8e-5) == 1.8e-5
    @test as_si_temperature(300.0) == 300.0
end

@testset "is_dimensionless traits plain numbers correctly" begin
    @test is_dimensionless(1.0)
    @test is_dimensionless(42)
    @test is_dimensionless(1.5f0)
    @test !is_dimensionless("not a number")
    @test !is_dimensionless(nothing)
end
