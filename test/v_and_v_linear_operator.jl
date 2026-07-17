# test/v_and_v_linear_operator.jl — AbstractLinearOperator wrapper V&V (v3.97)

using FiniteVolumeMethod
using FiniteVolumeMethod: AbstractLinearOperator, MatrixFreeError, SparseMatrixLinearOperator, as_linear_operator, underlying_matrix
using LinearAlgebra
using SparseArrays
using Test

include("TestHelpers.jl")

@testset "V&V: SparseMatrixLinearOperator — size + eltype" begin
    # The wrapper forwards size / eltype to the underlying matrix.
    A = sparse([1 2 0; 0 3 4; 5 0 6.0])
    op = SparseMatrixLinearOperator(A)
    @test op isa AbstractLinearOperator
    @test size(op) == (3, 3)
    @test size(op, 1) == 3
    @test size(op, 2) == 3
    @test eltype(op) == Float64
end

@testset "V&V: SparseMatrixLinearOperator — mul! matches matrix product" begin
    # op · x must equal A · x exactly (wrapper is a no-op on the arithmetic).
    A = sparse([1 2 0; 0 3 4; 5 0 6.0])
    op = SparseMatrixLinearOperator(A)
    x = [1.0, 2.0, 3.0]
    expected = A * x
    y = zeros(3)
    mul!(y, op, x)
    for i in 1:3
        @test y[i] == expected[i]
    end
end

@testset "V&V: SparseMatrixLinearOperator — underlying_matrix round-trip" begin
    # underlying_matrix returns the wrapped matrix identically.
    A = sparse([2.0 0; 0 3.0])
    op = SparseMatrixLinearOperator(A)
    @test underlying_matrix(op) === A
end

@testset "V&V: as_linear_operator — SparseMatrixCSC passthrough" begin
    # A raw SparseMatrixCSC gets wrapped in SparseMatrixLinearOperator.
    A = sparse([1.0 0; 0 1.0])
    op = as_linear_operator(A)
    @test op isa SparseMatrixLinearOperator
    @test underlying_matrix(op) === A
end

@testset "V&V: as_linear_operator — AbstractLinearOperator passthrough" begin
    # An existing AbstractLinearOperator should pass through unchanged.
    A = sparse([1.0 0; 0 1.0])
    op = SparseMatrixLinearOperator(A)
    op2 = as_linear_operator(op)
    @test op2 === op
end

@testset "V&V: MatrixFreeError — fallback underlying_matrix throws" begin
    # A custom AbstractLinearOperator without an overridden
    # underlying_matrix must throw MatrixFreeError. Define a minimal
    # matrix-free subtype here.
    struct _MatrixFreeOp{T} <: AbstractLinearOperator{T} end
    op = _MatrixFreeOp{Float64}()
    @test_throws MatrixFreeError underlying_matrix(op)
end

@testset "V&V: SparseMatrixLinearOperator — size / mul! for non-square" begin
    # Wrapper preserves shape for rectangular matrices.
    A = sparse([1.0 2.0 3.0; 4.0 5.0 6.0])
    op = SparseMatrixLinearOperator(A)
    @test size(op) == (2, 3)
    x = [1.0, 1.0, 1.0]
    y = zeros(2)
    mul!(y, op, x)
    @test y == [6.0, 15.0]
end

@testset "V&V: AbstractLinearOperator — eltype propagates type param" begin
    # eltype(op) is set from the T parameter, not computed from the matrix.
    A32 = sparse([1.0f0 2.0f0; 3.0f0 4.0f0])
    op32 = SparseMatrixLinearOperator(A32)
    @test eltype(op32) == Float32
    A64 = sparse([1.0 2.0; 3.0 4.0])
    op64 = SparseMatrixLinearOperator(A64)
    @test eltype(op64) == Float64
end

@testset "V&V: SparseMatrixLinearOperator — linearity via mul!" begin
    # mul!(y, op, α·x + β·z) matches α·(op·x) + β·(op·z) via superposition.
    A = sparse([2.0 0.5; 0.5 2.0])
    op = SparseMatrixLinearOperator(A)
    x = [1.0, 2.0]
    z = [-0.5, 1.5]
    y_x = zeros(2); mul!(y_x, op, x)
    y_z = zeros(2); mul!(y_z, op, z)
    y_sum = zeros(2); mul!(y_sum, op, x .+ z)
    for i in 1:2
        @test y_sum[i] ≈ y_x[i] + y_z[i] rtol = 1.0e-14
    end
    y_scaled = zeros(2); mul!(y_scaled, op, 3.0 .* x)
    for i in 1:2
        @test y_scaled[i] ≈ 3.0 * y_x[i] rtol = 1.0e-14
    end
end
