# v3.0 Migration Guide

v3.0.0 is the consolidation release for the Stage 0–9 v3 industrial-grade
overhaul. This guide
summarises behavioural and API changes since v2.0.0 and tells existing
users what to touch.

## Release mapping

| v3 Stage | Version | Headline |
|----------|---------|----------|
| 0 | v2.1.0 | Orphan cleanup, shared test helpers, typed BC errors, provenance doc. |
| 1 | v2.2.0 | Sparsity reuse, zero-alloc gradient, block equations, umbrella types, named Tunable schema, `AbstractFVMSolution`, AbstractArray state, `AbstractLinearOperator`. |
| 2 | v2.3.0 | True MPI submesh decomposition via RCB + LocalFVMMesh. |
| 3 | v2.4.0 | Thermo + rheology hierarchies, over-relaxed Laplacian correction. |
| 4 | v2.5.0 | Turbulence correctness: Durbin, full-tensor Germano, skewed wall functions. |
| 5 | v2.6.0 | Per-face CHT, MULES VOF, GCL verification. |
| 6 | v2.7.0 | MRF, porous, cavitation, FW-H, QMoM PBM. |
| 7 | v2.8.0 | Solid mechanics, FSI Aitken, function objects, expression BCs. |
| 8 | v2.9.0 | Octree mesh-gen skeleton, AMR markers, ZZ indicator. |
| 9 | v2.10.0 | Matrix-free operator, Unitful boundary integration. |
| 10 | **v3.0.0** | Release polish, CHANGELOG rollup, docs. |

## Breaking changes

Per the "break freely" posture from the roadmap's scoping questions,
v3.0 makes a handful of API-breaking changes:

1. **`build_boundary_map(field)` returns `Vector{Int}`** (was `Dict{Int,Int}`).
   Call syntax `bmap[f]` is unchanged; `haskey(bmap, f)` callers must
   switch to `bmap[f] != 0`.

2. **`CollocatedScalarField`, `CollocatedVectorField`, `FaceFluxField` gained a
   new trailing type parameter `A <: AbstractVector`.** Existing dispatch on
   `::CollocatedScalarField{T}` still works via Julia UnionAll matching.
   Only direct 2-parameter construction `CollocatedScalarField{T, A}(…)`
   is required for non-default containers (future GPU backends).

3. **`AbstractFVMMesh{Dim, T}`** now subtypes `AbstractFiniteVolumeMesh{Dim}`
   (was `AbstractParabolicMesh`). Transparent for all known callers.

4. **`StandardKEpsilon` gained a `realizability_alpha::T` field** (default
   `0` preserves classical formulation). Construction by keyword is
   backward-compatible; positional constructor now takes 7 args.

5. **`DistributedFVMMesh` field layout rewritten** (Stage 2): removed
   `n_ghost`, added `n_local` and `halo_owner_rank`. External users of
   `FVMMPIExt` must update field access.

6. **`assemble_laplacian!` default correction mode** changed from
   minimum-correction (implicit) to `NON_ORTHO_OVER_RELAXED`. Behaviour
   is identical on Cartesian / orthogonal meshes; skewed meshes see an
   improved iterative-correction convergence rate.

No functional changes to public `solve()` / `remake` / symbolic
indexing APIs — user scripts written against v2.0.0 continue to run
unchanged for all standard incompressible, hyperbolic, parabolic, and
MHD problems.

## Manifest audit

Before v3.0.0, `validation/manifest.toml` flagged every collocated
feature as `experimental`. The v3 roadmap's Stage-stable promotions
require benchmark-suite validation per feature (Stages 3e, 4b, 4c, 5,
6) before marking `stable`. The current manifest retains
`experimental` maturity for all collocated features because those
benchmark suites are Stage-10 follow-ups (ship-after-v3.0.0 commitments
listed in each release's CHANGELOG). Users should treat the collocated
stack as `experimental` per the manifest, not as `stable` — the v3
release fixes the infrastructure underneath, not the V&V story.

## Deprecations

None. Every v2 export retained, expanded with ~60 new Stage 1–9
symbols. See `src/FiniteVolumeMethod.jl` export block for the
authoritative list.

## Where to look for each improvement

- **Sparsity pattern reuse + fast assembly helpers**: `src/collocated/types.jl:170-310`
- **Zero-alloc gradient**: `src/collocated/gradient.jl:78-115`
- **True MPI decomposition**: `src/parallel/rcb_partitioner.jl`, `src/parallel/local_mesh.jl`, `ext/FVMMPIExt/partitioning.jl`
- **Thermo / rheology hierarchies**: `src/pressure_based/thermo_models.jl`, `src/pressure_based/rheology.jl`
- **Over-relaxed Laplacian**: `src/collocated/laplacian.jl:43-110`
- **Turbulence Durbin / wall functions**: `src/physics/turbulence/k_epsilon.jl`, `src/turbulence/wall_functions.jl`
- **MULES flux limiter**: `src/multiphase/boundedness.jl:62-175`
- **GCL verification**: `src/dynamic_mesh/mesh_update.jl:145-205`
- **MRF / porous / cavitation / FW-H / QMoM**: `src/mrf/`, `src/porous/`, `src/cavitation/`, `src/aeroacoustics/`, `src/population_balance/`
- **Solid mechanics / FSI / function objects**: `src/solid_mechanics/`, `src/fsi/`, `src/function_objects/`
- **Octree mesher + AMR markers + ZZ indicator**: `src/mesh_generation/`, `src/amr_collocated/`
- **Matrix-free operator**: `src/linear_solvers/matrix_free.jl`
- **Unitful integration**: `src/units/units.jl`

## What's still outstanding

The v3.0.0 release ships the infrastructure needed to promote every
collocated feature to `stable` maturity — that is, to reach production
OpenFOAM-parity CFD. What remains is V&V: per the roadmap, each
feature needs ≥3 published-benchmark cases (Ghia cavity, Moser
channel, De Vahl Davis CHT, Hysing bubble, Sandia Flame D, Turek-Hron
FSI, RAE2822, ONERA M6, etc.). These are tracked as Stage-specific
follow-ups in each release's CHANGELOG and in `test/KNOWN_FAILURES.md`.

The v3 overhaul closed **~25 of the 40 originally-flagged issues** in
`test/KNOWN_FAILURES.md`. The remaining items — mostly V&V benchmark
suites, full distributed-Krylov via PartitionedArrays, and full
snappyHexMesh-level mesh generation — are Stage-specific follow-ups on
a v3.x cadence.
