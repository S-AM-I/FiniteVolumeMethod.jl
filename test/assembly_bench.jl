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
