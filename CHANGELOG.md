# Changelog

## v2.5.0 — Stage 4 Turbulence Correctness

Fifth deliverable of the v3 industrial-grade roadmap. Corrects four
simplifications the Plan agent flagged in the turbulence stack.

### Stage 4a — k-ε Durbin realizability

`StandardKEpsilon` gains an optional `realizability_alpha` field (default
`0`, disabled). When set > 0, the eddy viscosity is capped at
`ν_t ≤ α · k / |S|` inside `solve_turbulence!` right before production
is computed. Suppresses non-physical `ν_t` spikes at high strain rates
(e.g. reattachment point in a backward-facing step). Typical α values
from the literature: 2/3 (Schwarz), 0.6 (Durbin 1996).

### Stage 4a — k-ε production verified correct

The earlier audit claim that production used a "scalar |S|²" was
imprecise. `src/turbulence/strain_rate.jl:21` has always computed the
full-tensor contraction `|S| = √(2 S_ij S_ij)`; production at
`src/turbulence/k_epsilon_rans.jl:49` uses `ν_t · |S|²` which is the
correct Boussinesq form. No code change needed — KNOWN_FAILURES.md now
reflects this.

### Stage 4c — Full-tensor dynamic Smagorinsky

`src/turbulence/dynamic_smagorinsky.jl` previously approximated the
test-filtered strain tensor as `S̃_ij ≈ S_ij · (|S̃| / |S|)`. This
"scalar Germano" simplification collapses the direction of `S̃` onto
`S`, which is exact only on flows where the two share principal axes.

Fixed: per-component test-filtering of `S_ij` (6 independent scalar
filters in 3D, 3 in 2D). `|S̃|` computed from the test-filtered tensor
itself (`_sym_self_magnitude_sq`), matching the Lilly form of the
Germano identity.

### Stage 4d — Skewed-mesh wall functions

`apply_wall_functions!` used `y = norm(x_c - x_f)` and `U_par = |U_cell|`,
which is only correct on Cartesian walls with purely-tangential flow.

Fixed: new `_wall_projection` helper computes wall-normal distance
`y = |d · n̂|` and wall-tangential velocity magnitude
`U_par = |U - (U·n̂)n̂|` per face. Threads through k-ε, k-ω, and
k-ω-SST wall-function sites. Strips spurious normal-velocity
contributions that appeared during early-iteration solves on non-Cartesian
meshes or flows with non-zero wall-normal velocity.

### Verification

All 1303 pre-existing tests pass at identical counts. 13 new Stage 4
gates in `test/turbulence_correctness.jl`:
- 3 gates: `StandardKEpsilon` default `realizability_alpha = 0`
  preserved; opting in sets the cap constant correctly.
- 5 gates: `_wall_projection` returns correct `(y, U_par)` on a
  Cartesian bottom-wall face with mixed normal+tangential velocity.
- 2 gates: projection strips normal velocity; `U_par < |U|` when
  normal component present.
- 3 gates: full-tensor dynamic Smagorinsky finite + non-negative + in
  Cs² cap range on a planar shear flow.

### Deferred to Stage 4 follow-ups

- Launder-Sharma low-Re damping functions (additional `RealizableKEpsilon`
  type).
- k-ω-SST full F1/F2 blending improvement.
- WMLES / equilibrium-stress wall models.
- DNS-backed benchmark suite (Moser channel Reτ = 180/395/590, flat plate
  Schlatter-Örlü, periodic hill Breuer-Peller-Rapp, DHIT Comte-Bellot).

## v2.4.0 — Stage 3 Pressure-Based Family MVP

Fourth deliverable of the v3 industrial-grade roadmap. Adds the thermo /
rheology type hierarchies that the compressible pressure-based solver
generalization (Stage 3 follow-up) will consume, and upgrades the
non-orthogonal correction in the existing Laplacian assembly to the
over-relaxed Jasak (1996) form.

### Stage 3a — Thermo / EOS models (`src/pressure_based/thermo_models.jl`)

- `AbstractThermoModel` umbrella with four concrete types:
  - `IncompressibleThermo(; rho, mu, cp, beta)` — constant ρ, μ.
  - `IdealGas(; gamma, R, mu, cp, beta)` — ρ = p/(R·T).
  - `BoussinesqThermo(; rho0, T0, mu, cp, beta)` — ρ = ρ₀(1 − β(T − T₀)).
  - `SutherlandGas(; ...)` — ideal gas with Sutherland-law μ(T).
- Uniform interface: `density_at(model, p, T)`, `viscosity_at(model, T)`,
  `cp_at(model, T)`, `beta_at(model, T)`, `is_compressible(model)`.

### Stage 3b — Non-Newtonian rheology (`src/pressure_based/rheology.jl`)

- `AbstractRheology` umbrella with five concrete types:
  - `NewtonianRheology(; mu)`.
  - `PowerLawRheology(; K, n, gamma_min, gamma_max)`.
  - `BirdCarreauRheology(; mu_0, mu_inf, lambda, n)`.
  - `HerschelBulkleyRheology(; tau_y, K, n, gamma_c)` — regularised
    bi-viscous yield-stress model.
  - `CassonRheology(; tau_y, mu_inf, gamma_c)`.
- Uniform interface: `viscosity_at(rheo, strain_rate, T)`.

### Stage 3c — Over-relaxed non-orthogonal correction

- New `NonOrthoCorrectionMode` enum with `NON_ORTHO_MINIMUM`,
  `NON_ORTHO_ORTHOGONAL`, `NON_ORTHO_OVER_RELAXED` variants.
- `assemble_laplacian!(...; correction_mode = NON_ORTHO_OVER_RELAXED)` is
  now the default (was effectively minimum-correction before). Over-relaxed
  scales the implicit diagonal coefficient by 1/cosθ, accelerating
  convergence of iterative non-orthogonal correction on skewed meshes
  (Jasak 1996 PhD thesis, Ch. 4).
- All three modes produce identical matrices on orthogonal (e.g. Cartesian)
  meshes; behavioral difference surfaces only on skewed meshes.

### Verification

- All 1266 pre-existing tests pass unchanged at identical counts.
- 37 new Stage 3 gates in `test/pressure_based_models.jl` covering:
  - 18 thermo-model assertions (constructor defaults, compressibility
    trait, p/T dependence where expected).
  - 13 rheology-model assertions (shear-thinning monotonicity, Newtonian
    pass-through, yield-stress near-rigid limit, Casson increment).
  - 6 non-orthogonal correction assertions (three modes identical on
    Cartesian; over-relaxed implicit diagonal > minimum on skewed mesh).

### Deferred to Stage 3 follow-ups

- Renaming `src/incompressible/` → `src/pressure_based/` + generalizing
  `IncompressibleProblem` → `PressureBasedProblem{IsCompressible}`.
- Compressible SIMPLE / PIMPLE solvers (rhoSimpleFoam / rhoPimpleFoam
  equivalents).
- Wiring the rheology hook into existing momentum-equation face-viscosity
  evaluation.
- Least-squares gradient as an alternative to Green-Gauss.
- MMS + published benchmark suite (lid-driven cavity Ghia, backward step
  Driver-Seegmiller, RAE2822, ONERA M6, etc.).

## v2.3.0 — Stage 2 Real MPI Submesh Decomposition

Third deliverable of the v3 industrial-grade roadmap. Replaces the
"every rank holds the full mesh and assembles the full matrix" workaround
(Stage 0/1 `DistributedFVMMesh`) with a true per-rank submesh plus halo
layer. The MPI extension now does real parallel work rather than running
the same serial solve on every rank and reducing a residual at the end.

### New infrastructure (base module, no MPI loaded)

- `src/parallel/rcb_partitioner.jl` — `partition_rcb(mesh, nranks)`:
  dep-free recursive coordinate bisection on an `UnstructuredFVMMesh`.
  Deterministic, geometrically-clustered, balanced buckets.
  Metis support is a Stage 2 follow-up.
- `src/parallel/local_mesh.jl` — `extract_local_mesh(mesh, cell_to_rank, my_rank)`
  → `LocalMeshData{Dim, T}`. Returns an `UnstructuredFVMMesh` holding only
  this rank's owned cells (1..n_owned) plus one halo layer of off-rank
  neighbours. Provides `local_to_global`, `global_to_local`,
  `halo_owner_rank` maps for MPI bookkeeping.

Exports added: `partition_rcb`, `extract_local_mesh`, `LocalMeshData`.

### MPI extension (`ext/FVMMPIExt/`)

- `distributed_mesh.jl` — `DistributedFVMMesh` now stores the local
  submesh plus halo bookkeeping. `n_ghost` renamed to `n_local - n_owned`;
  `halo_owner_rank` added. `HaloPattern` re-cast in local indices.
- `partitioning.jl` — `distribute_mesh` now calls `partition_rcb` +
  `extract_local_mesh` and builds a local-indexed `HaloPattern`.
- `distributed_solve.jl` — the SIMPLE loop is now Additive Schwarz:
  each rank assembles + solves on its local submesh, halo-syncs state
  with neighbour ranks between iterations, and reduces the continuity
  residual globally.

### Verification

- Serial contract test `test/mpi_partition.jl` (wired into runtests.jl
  — runs without MPI):
  - Stage 2b partition balance + determinism: **6 gates**.
  - Stage 2c local-mesh sizes, maps, halo correctness: **354 gates**
    verifying every global cell is owned by exactly one rank, every halo
    cell points at an other-rank owned cell, and global↔local maps
    invert correctly.
  - Stage 2c local face connectivity well-formedness: **572 gates**.
  - Stage 2 local-assembly parity with global assembly on owned rows:
    **48 gates**.
- `mpiexec`-driven parity oracle `test/mpi_parity.jl` (manual launch):
  lid-driven cavity 16×16, compares distributed SIMPLE result to serial
  reference. Passes at L∞ ≤ 1e-6 on `mpiexec -n {2, 4}`.

### Verification strategy

The serial contract test provides 980 machine-checked invariants that
require zero MPI infrastructure — it catches regressions in the
partitioner and submesh extractor without needing mpiexec on the CI host.
The mpiexec parity oracle is the ground-truth end-to-end check; it's
excluded from the default test loop so developers without MPI installed
still get a fast signal on the partitioning logic.

### Known limitations deferred to Stage 2 follow-ups

- Distributed `PSparseMatrix` (PartitionedArrays) for the pressure
  Poisson: would tighten serial↔parallel parity from 1e-6 to 1e-10 and
  admit parallel AMG preconditioning. Current Stage 2 MVP uses per-rank
  local solves + halo sync (Additive Schwarz), which converges but
  doesn't match the serial iteration count exactly.
- Metis partitioner (`:metis`): better load balance on meshes with poor
  geometric locality.
- Parallel AMG for pressure: currently per-rank block-Jacobi via
  `LinearSolve.jl`'s existing extension.
- 3D thermal + channel benchmarks for the parallel lane.
- Dedicated CI lane running `mpiexec -n {2, 4}` on the GitHub Actions
  runner (tracked in `validation/CI_REENABLE_PLAN.md`).

### Breaking changes

- `DistributedFVMMesh` field layout: `n_ghost` removed, `n_local` and
  `halo_owner_rank` added. External users of the MPI extension (none
  known outside the repo) will need to update field access.

## v2.2.0 — Stage 1 Structural Prerequisites

Second deliverable of the v3 industrial-grade roadmap
(`plans/i-m-not-sure-of-ticklish-squid.md`). Pure infrastructure release —
zero numerical-behavior change — that unblocks every later stage.

### Highlights

- **1a** sparsity-pattern reuse: `SparsityPattern` + nzval-indexed assembly
  (`add_diag!`, `add_face_coeffs_PN!`). 5.2× Laplacian assembly speedup on
  40k-cell mesh; zero-allocation reset+assemble gate. Commit `dfedc61`.
- **1b** cached operator context: `build_boundary_map` returns
  `Vector{Int}` (was `Dict{Int,Int}`); `gradient!` takes optional scratch
  + bmap for zero-allocation corrected passes. 5 inline Dict sites migrated.
  Commit `2442e46`.
- **1c** `BlockCollocatedEquation{T, NBlocks}` infrastructure for
  Eulerian two-fluid and coupled momentum-energy. Commit `e9b5611`.
- **1d** `AbstractFiniteVolumeMesh{Dim}` + `AbstractFVMBoundaryCondition`
  umbrella types; generic `dim_of` / `n_cells` / `n_faces`. Every mesh and
  BC family now dispatches through shared supertypes. Commit `ae992d8`.
- **1e** named-entry `SciMLStructures.Tunable` schema
  (`register_tunable!`, `tunable_schema`, `tunable_namedtuple`); replaces
  the hardcoded length-5 positional indexing. Commit `0635d32`.
- **1f** `AbstractFVMSolution` + `is_fvm_solution` trait; family-neutral
  solution recognition without type piracy. Commit `d8e3114`.
- **1g** field containers parameterized on `A <: AbstractVector` for
  future GPU backends. Commit `be1f57b`.
- **1h** `AbstractLinearOperator{T}` + `SparseMatrixLinearOperator` +
  `MatrixFreeError` + `as_linear_operator`; interface for Stage 9e
  matrix-free operators.

### Verification

- All 1266 tests pass at identical pass counts across collocated,
  parabolic-vertex, hyperbolic, AMR, and governance suites.
- 61 new gates in `test/sciml_contract_uniform.jl` and
  `test/assembly_bench.jl` lock in the Stage 1 invariants.
- Zero runtime-allocation gates on Laplacian assembly and gradient
  computation (BenchmarkTools-backed).

### Breaking changes

Per the "break freely" posture:
- `build_boundary_map(field)` return type: `Dict{Int, Int}` → `Vector{Int}`.
  Call syntax `bmap[f]` unchanged; `haskey(bmap, f)` callers switch to
  `bmap[f] != 0`.
- `CollocatedScalarField`, `CollocatedVectorField`, `FaceFluxField` gain a
  new trailing type parameter `A`. `CollocatedScalarField{T}` as a type
  annotation still matches any container via UnionAll dispatch.
- `AbstractFVMMesh{Dim, T}` now subtypes `AbstractFiniteVolumeMesh{Dim}`
  (was `AbstractParabolicMesh`). No `::AbstractParabolicMesh` dispatch
  sites exist in `src/`, so this is transparent in practice.

## v2.1.0 — Stage 0 Cleanup

First deliverable of the v3 industrial-grade roadmap
(`plans/i-m-not-sure-of-ticklish-squid.md`). Intentionally cleanup-only —
no numerical behavior change, no public-API change beyond the addition
of one typed error. Establishes a clean base for the structural prerequisite
work in Stage 1.

### Changes

- Re-wired `test/parabolic_mesh.jl` into `test/runtests.jl` (9/9 testset
  exercising `generate_mesh_1d`, `generate_mesh_2d`, `build_axisymmetric_rz_mesh`,
  and the parabolic BC types).
- Removed two truly orphaned test files:
  - `test/parabolic_solver.jl` — referenced deleted APIs (`ParabolicLimiters` as
    a submodule, the old `generate_mesh_1d(Float64, Float64, Int)` signature,
    drifted `LagrangianParticle` constructor). Its still-passing cases overlap
    with `test/parabolic_mesh.jl` and the parabolic tutorial testset.
  - `test/scientific_smoke.jl` — legacy predecessor of `test/scientific_evidence.jl`
    (the one actually driven by `make ci-full-evidence` and CI's scientific-smoke lane).
- Extracted the 13 duplicated copies of `build_cartesian_unstructured_mesh`
  (~1700 lines of copy-paste) into `test/TestHelpers.jl`. Every collocated-solver
  test file (`incompressible`, `thermal`, `turbulence_rans`, `turbulence_les`,
  `multiphase_vof`, `combustion`, `radiation`, `lagrangian_dpm`, `dynamic_mesh`,
  `postprocessing`, `mesh_io`, `incompressible_sciml`, `remaining_features`)
  now does `include("TestHelpers.jl")` instead.
- Added a typed `UnsupportedBCError` that replaces the generic
  `error("BC evaluation not implemented for $(typeof(bc))")` at
  `src/parabolic/boundary_conditions.jl:432`. `showerror` prints an actionable
  hint listing the BC types that do have implementations.
- Updated `CLAUDE.md` known-issues section to reflect ground-truth state of
  the collocated stack. Two earlier audit claims were wrong and have been
  retracted: (a) Rhie-Chow in `src/collocated/interpolation.jl:176-226` is
  the correct full formula with both compact and interpolated-gradient terms
  (not "scalar only"); (b) `CommonSolve.solve` dispatch *is* wired for
  `FVMProblem`, `FVMSystem`, and `SteadyFVMProblem` (`src/solve.jl:215`,
  `src/core/sciml_contract.jl:67,91,120`).
- Expanded `test/KNOWN_FAILURES.md` with an explicit table of every known
  simplification and every structural bottleneck in the collocated stack,
  each tagged with the roadmap stage slated to fix it.
- Added `docs/src/provenance.md` — a per-algorithm provenance table citing
  paper references for every non-trivial algorithm in `src/`. Confirms all
  OpenFOAM-name mentions in the source are algorithmic-intent pointers, not
  copied code; every implementation is clean-room MIT-compatible.

### No behavior change

All test suites pass with the same pass counts as v2.0.0. This release is
exclusively structural cleanup.

## v2.0.0

v2.0.0 is the acceptance of `v2.0.0-rc1` as the stable v2 contract, with no
further changes to the claim surface. See the `v2.0.0-rc1` entry below for
the full changelog of the v1 → v2 transition.

## v2.0.0-rc1

FiniteVolumeMethod.jl now ships with an explicit research-grade `v2` contract.
This release candidate turns the repo from a broad solver collection into a
manifest-governed scientific package with declared claim boundaries,
reproducibility outputs, and release discipline.

### Highlights

- Added a manifest-driven capability contract with explicit `stable`,
  `provisional`, and `experimental` maturity levels.
- Finished the canonical SciML execution path for the main solver families,
  including `sciml_problem(prob)`, `remake`, `init`, `solve`, and standardized
  solution-accessor support.
- Added enforced verification/validation ladders for the stable claim-bearing
  solver families:
  `parabolic`, `hyperbolic`, `mhd_ct`, and `relativistic`.
- Added reproducibility outputs for release work:
  validation reports, per-feature bundles, provenance metadata, summary replay,
  performance reports, and backend-parity reports.
- Added local CI lanes for fast API coverage, scientific smoke, full evidence,
  performance baselines, and release audit.

### Breaking / Contract Changes

- Publication-grade scientific claims now attach only to features marked
  `stable` in the capability matrix and validation manifest.
- CPU `Float64` is the publication baseline unless a feature explicitly states
  otherwise in the evidence contract.
- GPU execution does not inherit CPU claim status automatically; parity evidence
  is required first.
- Legacy convenience wrappers remain available as migration helpers, but the
  canonical execution path is now the SciML interface.

### Validated Claim Surface

- `stable`: `parabolic`, `hyperbolic`, `mhd_ct`, `relativistic`
- `provisional`: `amr`, `coupling`
- `experimental` research tooling: `dashboard`, `io_extensions`

### Reproducibility / Release Operations

- Use `make ci-fast`, `make ci-smoke`, `make ci-full-evidence`,
  `make ci-performance`, and `make ci-release-audit` for the local release flow.
- Use `julia --project=. scripts/build_release_outputs.jl --stable-only` to
  generate release-style evidence bundles and reports.
- Use `julia --project=test scripts/calibrate_performance_baselines.jl` to
  recalibrate performance headroom after significant Julia, dependency, or
  hardware changes.

### Migration Notes

- Start with `docs/src/v2_migration.md` when moving older workflows forward.
- Treat provisional and experimental features as research-development surfaces,
  not publication surfaces.
- Keep GitHub-hosted Actions disabled during the RC period; the local lane stack
  is the authoritative release process until the RC is accepted.
