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
using FiniteVolumeMethod: BlockCollocatedEquation, CollocatedEquation, add_block_diag!, add_block_offdiag_NP!, add_block_offdiag_PN!, assemble_laplacian!, build_boundary_map, gradient!, nblocks, reset!
using FiniteVolumeMethod.Parabolic: DirichletBC
using Test
using BenchmarkTools
using SparseArrays
using StaticArrays: SVector

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "Stage 1a: CollocatedEquation sparsity pattern reuse" begin
    mesh = build_cartesian_unstructured_mesh(100, 100, 1.0, 1.0)  # 10_000 cells
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
        :top => DirichletBC(1.0),
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

@testset "Stage 1c: BlockCollocatedEquation structure + helpers" begin
    mesh = build_cartesian_unstructured_mesh(5, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    nf_internal = count(!iszero, @view(mesh.face_cells[2, :]))

    # NBlocks = 1 should produce the same structural nnz as the scalar case.
    eq1 = BlockCollocatedEquation(mesh, Val(1))
    @test nblocks(eq1) == 1
    @test size(eq1.A) == (nc, nc)
    @test nnz(eq1.A) == nc + 2 * nf_internal

    # NBlocks = 2: nc × 2² on the diagonal, nf_internal × 2² × 2 off-diagonal.
    eq2 = BlockCollocatedEquation(mesh, Val(2))
    @test nblocks(eq2) == 2
    @test size(eq2.A) == (2 * nc, 2 * nc)
    @test nnz(eq2.A) == nc * 4 + 2 * nf_internal * 4

    # Sanity-check fast-path helpers on a 2-block equation: assembling a
    # diagonal + internal-face symmetric coupling produces a matrix whose
    # inner block structure matches the naive `A[i, j] +=` path.
    reset!(eq2)
    for c in 1:nc
        add_block_diag!(eq2, c, 1, 1, 1.0)
        add_block_diag!(eq2, c, 2, 2, 2.0)
        add_block_diag!(eq2, c, 1, 2, 0.1)
    end
    # Reference via A[i, j] += on a fresh spzeros of the same size
    ref = SparseArrays.spzeros(Float64, 2 * nc, 2 * nc)
    for c in 1:nc
        ref[(c - 1) * 2 + 1, (c - 1) * 2 + 1] += 1.0
        ref[(c - 1) * 2 + 2, (c - 1) * 2 + 2] += 2.0
        ref[(c - 1) * 2 + 1, (c - 1) * 2 + 2] += 0.1
    end
    @test eq2.A == ref

    # Off-diagonal helpers
    reset!(eq2)
    f_internal = findfirst(f -> mesh.face_cells[2, f] != 0, 1:nf)
    P = mesh.face_cells[1, f_internal]
    N = mesh.face_cells[2, f_internal]
    add_block_offdiag_PN!(eq2, f_internal, 1, 1, 3.5)
    add_block_offdiag_NP!(eq2, f_internal, 2, 2, -2.0)
    @test eq2.A[(P - 1) * 2 + 1, (N - 1) * 2 + 1] == 3.5
    @test eq2.A[(N - 1) * 2 + 2, (P - 1) * 2 + 2] == -2.0
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
