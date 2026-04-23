# test/assembly_bench.jl
#
# Stage 1a gate: make sure the `CollocatedEquation` sparsity pattern is built
# once and re-used across `assemble_*!` calls. Regressing this would put
# CSC random-pattern insertion back into the SIMPLE inner loop, which at
# industrial cell counts (10^5–10^6) dominates wall-clock.
#
# Two checks:
#   1. `reset!` + `assemble_laplacian!` on a 10^4-cell mesh allocates 0 bytes
#      (the sparsity pattern is NOT rebuilt; only `A.nzval` is zeroed and
#      re-filled).
#   2. Repeated assembly produces the same matrix entries bit-for-bit (no
#      drift from floating-point noise due to insertion order).

using FiniteVolumeMethod
using Test
using BenchmarkTools
using SparseArrays
using StaticArrays: SVector

include("TestHelpers.jl")

@testset "Stage 1a: CollocatedEquation sparsity pattern reuse" begin
    mesh = build_cartesian_unstructured_mesh(100, 100, 1.0, 1.0)  # 10_000 cells
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => ParabolicDirichlet(0.0),
        :right => ParabolicDirichlet(0.0),
        :bottom => ParabolicDirichlet(0.0),
        :top => ParabolicDirichlet(1.0),
    )
    eq = CollocatedEquation(mesh)

    # Warm up JIT
    reset!(eq)
    assemble_laplacian!(eq, 1.0, mesh, bcs)
    snapshot = copy(eq.A.nzval)
    colptr_before = copy(eq.A.colptr)
    rowval_before = copy(eq.A.rowval)

    # ── Allocation gate ─────────────────────────────────────────────
    # reset!+assemble must not allocate — sparsity is pre-built and every
    # assembly site writes into `A.nzval` by index. Zero allocations here
    # is the critical CI gate for Stage 1a.
    bench = @benchmark begin
        reset!($eq)
        assemble_laplacian!($eq, 1.0, $mesh, $bcs)
    end samples = 50 evals = 1
    @test bench.memory == 0
    @test bench.allocs == 0

    # ── Determinism gate ────────────────────────────────────────────
    # A repeated assembly (with reset) yields the same nzval and the
    # same structural arrays — no row/col reshuffling from implicit
    # sparse insertion.
    reset!(eq)
    assemble_laplacian!(eq, 1.0, mesh, bcs)
    @test eq.A.nzval == snapshot
    @test eq.A.colptr == colptr_before
    @test eq.A.rowval == rowval_before

    # ── Structure-preservation gate ────────────────────────────────
    # The number of structural nonzeros is 1 per cell (diagonal) + 2 per
    # internal face (A[P,N] and A[N,P]). Anything larger means a stray
    # structural insertion leaked in during assembly.
    nf_internal = count(!iszero, @view(mesh.face_cells[2, :]))
    expected_nnz = 10_000 + 2 * nf_internal
    @test nnz(eq.A) == expected_nnz
end

@testset "Stage 1b: Zero-allocation gradient! with cached scratch + bmap" begin
    mesh = build_cartesian_unstructured_mesh(100, 100, 1.0, 1.0)
    phi = CollocatedScalarField(:phi, mesh; value = 1.0)
    nc = length(mesh.cell_volumes)
    grad = Vector{SVector{2, Float64}}(undef, nc)
    scratch = Vector{SVector{2, Float64}}(undef, nc)
    bmap = build_boundary_map(phi, mesh)

    # Warm up
    gradient!(grad, phi, mesh; n_corrections = 0, scratch = scratch, bmap = bmap)
    gradient!(grad, phi, mesh; n_corrections = 2, scratch = scratch, bmap = bmap)

    # Zero-corrections path: no scratch needed, bmap re-used
    bench0 = @benchmark gradient!(
        $grad, $phi, $mesh; n_corrections = 0, bmap = $bmap,
    ) samples = 20 evals = 1
    @test bench0.memory == 0
    @test bench0.allocs == 0

    # Corrected path: scratch re-used
    bench2 = @benchmark gradient!(
        $grad, $phi, $mesh; n_corrections = 2, scratch = $scratch, bmap = $bmap,
    ) samples = 20 evals = 1
    @test bench2.memory == 0
    @test bench2.allocs == 0
end

@testset "Stage 1b: build_boundary_map returns O(1)-lookup Vector" begin
    mesh = build_cartesian_unstructured_mesh(10, 10, 1.0, 1.0)
    phi = CollocatedScalarField(:phi, mesh; value = 0.0)
    bmap = build_boundary_map(phi, mesh)
    nf = size(mesh.face_cells, 2)

    @test bmap isa Vector{Int}
    @test length(bmap) == nf
    # Internal faces map to 0; boundary faces map to a valid phi.boundary index.
    for f in 1:nf
        if mesh.face_cells[2, f] == 0  # boundary face
            @test 1 <= bmap[f] <= length(phi.boundary)
            @test phi.boundary_face_indices[bmap[f]] == f
        else
            @test bmap[f] == 0
        end
    end
end
