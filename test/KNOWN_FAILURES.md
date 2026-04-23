# Known Failures

This file documents known test failures and their status.
The authoritative machine-readable source of truth for exclusions and demotions
is `validation/manifest.toml`; this document is a human-readable companion.

## Pre-existing

| Test | Status | Notes |
|------|--------|-------|
| `Aqua.test_unbound_args` | Broken (`broken = true`) | `Val{N}` pattern in AMR constructors is a known false positive. Tracked in `test/QUALITY_LEDGER.toml`. |
| `keller_segel_chemotaxis.jl` | Skipped | Excluded from tutorial test loop (marked `manual_review` in manifest). |

## Demoted From V&V Claims

| Test | Status | Notes |
|------|--------|-------|
| `heated_cavity.jl` | Demoted | Uses a simplified compressible surrogate, not a De Vahl Davis validation case. |
| `fishbone_moncrief_torus.jl` | Demoted | Uses an approximate torus initial condition, not a Fishbone-Moncrief equilibrium solution. |
| `lid_driven_cavity.jl` | Demoted | Does not impose the literature benchmark boundary treatment or compare against published profiles quantitatively. |
| `bondi_accretion_schwarzschild.jl` | Demoted | Current setup is not an actual Bondi solution and therefore cannot support a Bondi validation claim. |
| `amr_convergence.jl` | Demoted | Current assertions are regression/smoke checks, not a rigorous AMR convergence study. |
| `mhd_solver_comparison.jl` | Demoted | Relative solver comparison without external truth is not treated as scientific evidence. |
| `premixed_flame_1d.jl` | Demoted | Current checks are qualitative combustion regression checks, not a literature-backed validation case. |

## Validation Level Notes

- Scripts marked `run_in_ci = false` in `validation/manifest.toml` are excluded from CI due to memory or runtime constraints. They are exercised in the Nightly and Release workflows.
- All numerical acceptance criteria use fixed `@test` assertions. Image regression tests use `JULIA_REFERENCETESTS_UPDATE=true` and are not part of the scientific contract.

## Simplifications in the Collocated / OpenFOAM-Style Solver Stack

Every item below is a known simplification or incorrect implementation; each
is scheduled for a specific stage of the v3 roadmap
(`plans/i-m-not-sure-of-ticklish-squid.md`). Promotion of a feature from
`experimental` to `stable` in `validation/manifest.toml` requires the
corresponding entry to be fixed *and* a 3+ published-benchmark suite to be
green in CI.

### Numerical correctness

| Component | File:Line | Simplification | Fix Stage |
|-----------|-----------|----------------|-----------|
| Non-orthogonal correction | `src/collocated/gradient.jl:144-149` | Interpolated-gradient only; no over-relaxed variant, no least-squares gradient fallback | 3d |
| Laplacian skewness | `src/collocated/laplacian.jl` | No face-skewness correction term; accuracy drops on non-orthogonal meshes | 3d |
| k-ε realizability | `src/turbulence/k_epsilon_rans.jl:24` | `ν_t = C_μ k²/ε` with simple `max()` floor; no Durbin bound | 4a |
| k-ε production | `src/turbulence/k_epsilon_rans.jl` | Scalar strain magnitude `|S|²`, not full tensor contraction `S_ij S_ij` | 4a |
| k-ε / k-ω low-Re | — | High-Re form only; no Launder-Sharma, Abid, or other low-Re damping functions | 4a |
| k-ω-SST blending | `src/turbulence/k_omega_sst.jl` | Simplified scalar blending; should be full F1/F2 blending with proper limiter | 4a |
| Dynamic Smagorinsky | `src/turbulence/dynamic_smagorinsky.jl` | Scalar Germano identity, not full tensor form | 4a |
| Wall functions | `src/turbulence/wall_functions.jl` | Assumes cells aligned with boundary normal; no skew/tangential projection | 4a |
| Conjugate HT interface | `src/thermal/conjugate.jl` | Scalar face-averaged interface temperature, not per-face | 5a |
| VOF boundedness | `src/multiphase/boundedness.jl` | Hard clipping `clamp(α, 0, 1)` — not MULES (Multidimensional Universal Limiter with Explicit Solution) | 5b |
| VOF interface reconstruction | `src/multiphase/` | No isoAdvector / sharp interface reconstruction | 5b |
| VOF contact angles | `src/multiphase/surface_tension.jl` | Static/dynamic contact-angle models absent | 5b |
| Combustion chemistry | `src/combustion/edm.jl` | One-step EDM only; no multi-step mechanisms, no FGM, no Cantera interface | 5c |
| Combustion diffusion | `src/combustion/species_transport.jl` | Lewis-unity implicit; no per-species Le exposure | 5c |
| Radiation quadrature | `src/radiation/fvdom.jl` | fvDOM angular quadrature is skeleton; LSn/Tn sets absent | 5d |
| Radiation scattering | `src/radiation/fvdom.jl` | Scattering term absent | 5d |
| Radiation wall BCs | `src/radiation/fvdom.jl` | Basic Dirichlet/Neumann only; no wavelength-banded emissivity | 5d |
| DPM collision | `src/lagrangian/collisions.jl` | Binary elastic only; no hard/soft-sphere DEM, no agglomeration/coalescence | 5e |
| DPM breakup | `src/lagrangian/spray.jl` | Secondary breakup only (TAB/KHRT); no primary breakup (KH-ACT, LISA) | 5e, 7c |
| DPM injection | — | No cone/hollow-cone/flat-fan injection patterns or rate-of-injection profiles | 5e |
| Dynamic-mesh GCL | `src/dynamic_mesh/ale.jl` | Geometric conservation law not verified for large deformation | 5f |
| Dynamic-mesh 6-DOF | — | No 6-DOF rigid-body solver | 5f |
| Dynamic-mesh topology | — | No dynamic refinement/coarsening or topology changes during a run | 5f |
| Overset / chimera | — | Absent | 5f |

### Structural / performance

| Component | File:Line | Issue | Fix Stage |
|-----------|-----------|-------|-----------|
| ~~CollocatedEquation assembly~~ | ~~`src/collocated/types.jl:181,192`, `src/collocated/laplacian.jl`, every `assemble_*!`~~ | ~~Random-pattern CSC insertion `A[P,N] += …` on every SIMPLE outer iteration~~ | **Fixed in v2.2.0-dev (Stage 1a)**: `SparsityPattern` pre-computes nzval indices at mesh-bind time; `add_diag!` / `add_face_coeffs_PN!` write `A.nzval[idx]` in O(1). 5× speedup on 40k-cell Laplacian; zero-allocation gate in `test/assembly_bench.jl`. Cyclic BCs + pressure ref-cell pinning still use slow path until cyclic pairs are plumbed into `build_collocated_sparsity` (Stage 1a follow-up). |
| ~~Operator hot-loop allocation~~ | ~~`src/collocated/gradient.jl:126-130`, `src/collocated/interpolation.jl:96`~~ | ~~`fill(…)` buffer and `Dict{Int,Int}` constructed on every call~~ | **Fixed in v2.2.0-dev (Stage 1b)**: `build_boundary_map` now returns `Vector{Int}` (O(1) indexed lookup, single allocation) instead of `Dict{Int,Int}`. `gradient!` accepts optional `scratch` + `bmap` kwargs for full zero-allocation use. The 5 inline `Dict(f => i for …)` constructions in `interpolation.jl`, `pressure.jl`, and `boundary_conditions.jl` migrated to `build_boundary_map(field, mesh)`. Verified zero-alloc gate in `test/assembly_bench.jl`. |
| ~~CollocatedEquation is scalar-only~~ | ~~`src/collocated/types.jl:181`~~ | ~~Single `Vector{T}` for `b`; two-fluid and coupled momentum-energy need a `BlockCollocatedEquation`~~ | **Fixed in v2.2.0-dev (Stage 1c)**: `BlockCollocatedEquation{T, NBlocks}` with `BlockSparsityPattern` + `add_block_diag!` / `add_block_offdiag_PN/NP!` helpers added alongside the scalar type. Cell-major layout, eagerly-built `N×N` CSC, same O(1) nzval-indexed write pattern. Infrastructure only — actual use by Eulerian two-fluid (Stage 6e) and coupled momentum-energy (Stage 3) wires on top. Verified in `test/assembly_bench.jl`. |
| No AbstractFVMMesh supertype | `src/mesh/abstract_mesh.jl` | `FVMGeometry`, `StructuredMesh{1,2,3}D`, `UnstructuredFVMMesh` have no common supertype; conversion paths sparse | 1d |
| SciMLStructures.Tunable length-5 | `src/core/sciml_structures.jl:130-144` | Hardcoded `[nu, density, alpha_U, alpha_p, tolerance]`; adding one tunable breaks all `remake` callers | 1e |
| State containers non-generic | `src/incompressible/types.jl` | `Vector{T}` baked in; blocks KA.jl / GPU port without a rewrite | 1g |
| No AbstractLinearOperator | `src/linear_solvers/` | `_dispatch_solve` takes `SparseMatrixCSC` directly; no matrix-free path | 1h |
| MPI is full-mesh-per-rank | `ext/FVMMPIExt/distributed_mesh.jl:44`, `ext/FVMMPIExt/distributed_solve.jl:49-53` | Each rank stores full mesh AND assembles full matrix; halo exchange is decorative; only residual `Allreduce` uses MPI | 2 |
| `test/mpi_test.jl` not in runtests.jl | — | Requires manual `mpiexec -n 2 julia …`; no parallel/serial parity check | 2 |

### Missing OpenFOAM features

Each slated for the stage noted in the roadmap:

| Feature | Status | Stage |
|---------|--------|-------|
| Compressible pressure-based solvers (rhoSimpleFoam, rhoPimpleFoam, rhoReactingFoam) | Absent | 3 |
| Real-gas EOS (Peng-Robinson, Redlich-Kwong, tabulated) | Absent | 3b |
| Non-Newtonian rheology (power-law, Bird-Carreau, Herschel-Bulkley, Casson) | Absent | 3c |
| Moving Reference Frame (MRF) | Absent | 6a |
| Arbitrary Mesh Interface (AMI) / sliding mesh | Absent | 6b |
| Porous media (Darcy-Forchheimer) | Absent | 6c |
| Cavitation (Kunz, Schnerr-Sauer, Merkle) | Absent | 6d |
| Eulerian two-fluid | Absent | 6e |
| Aeroacoustics (FW-H, sponge zones) | Absent | 6f |
| Population balance modeling | Absent | 6g |
| Wall-modeled LES (WMLES) | Absent | 4a |
| Solid mechanics / FSI | Absent | 7a/b |
| Function objects / coded BCs / expression BCs | Absent | 7d |
| snappyHexMesh-equivalent mesh generation | Absent | 8a |
| Gmsh automation pipeline | Absent | 8b |
| AMR on collocated side | Absent | 8c |
| Error indicators | Absent | 8d |
| Full adjoint (SciMLSensitivity integration) | Absent | 9a–c |
| GPU backends for collocated | Absent | 9d |
| Matrix-free linear operators | Absent | 9e |
| Unitful integration | Absent | 9f |
| Binary OpenFOAM polyMesh reader | ASCII only (`src/mesh/openfoam_io.jl:22`) | 3 |
