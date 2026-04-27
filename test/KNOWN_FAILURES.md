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

## Published-Benchmark Suite Status (2026-04-27, v3.120)

Run via `./scripts/run_benchmarks.sh` with `FVM_RUN_BENCHMARKS=true`. The 5-case harness lands as follows on M3 / Julia 1.12.4:

| Benchmark | Status | Notes |
|-----------|--------|-------|
| `sod_shock_tube` | ✓ pass | HLLC + MUSCL on N=400 hits L¹ density error < 0.05 vs. analytical Riemann. |
| `moser_re180` | ✓ pass | Channel flow Re_τ=180. |
| `martin_moyce_dam_break` | ✓ pass | VOF + MULES dam-break front position vs. Martin-Moyce. |
| `ghia_re400` | ✓ RESOLVED | Grid bumped from N=64 to N=128 (matches Ghia 1982's 129×129), iterations reduced to 4000 with αU=0.5, αP=0.2. All 28 assertions now pass. |
| `rayleigh_benard_1e4` | CONFIGURATION UPDATED | Grid bumped from N=40 to N=80 with increased iterations. Needs a verification run to confirm all 9 assertions pass. Previously failing: De Vahl Davis Nu=2.243 ±10% and ±25% velocity tolerances. |

Promotion of `incompressible`, `thermal`, `multiphase`, `hyperbolic`, or `turbulence` from `provisional` to `stable` in `validation/manifest.toml` requires the corresponding benchmarks to pass.

**Pkg.test() baseline (v3.120):** 1,428,433 passed / 0 failed / 0 errored. The full failure-sweep wave is documented in `plans/open-work.md` §E.

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
| ~~WENO5 ghost-cell BC~~ | ~~`src/hyperbolic/boundary_conditions_{1d,2d,3d}.jl`~~ | ~~All BC fill functions hardcoded `ng=2`; WENO5 reconstruction requires `nghost=3` and failed on structured meshes~~ | **RESOLVED in stabilization session**: all 1D/2D/3D hyperbolic BC fill functions now accept an `ng` (ghost count) parameter; WENO5 (nghost=3) works on structured meshes. Remaining hardcodes: `positivity_limiter.jl`, `multirate.jl`, AMR ghost fill, and MHD/CT caches still use ng=2 and are follow-up items. |
| ~~Non-orthogonal correction~~ | ~~`src/collocated/gradient.jl:144-149`~~ | ~~Interpolated-gradient only; no over-relaxed variant~~ | **Fixed in v2.4.0 + v3.102.0 (Wave 1)**: `assemble_laplacian!` supports `NON_ORTHO_MINIMUM` / `NON_ORTHO_ORTHOGONAL` / `NON_ORTHO_OVER_RELAXED`; **over-relaxed (Jasak 1996 Ch. 4) is now the default in v3.102** for all collocated assembly. LSQ gradient alternative also lands in v3.102 as a peer to the Green-Gauss path. |
| Laplacian skewness | `src/collocated/laplacian.jl` | No face-skewness correction term; accuracy drops on heavily skewed meshes | 3 follow-up |
| ~~k-ε realizability~~ | ~~`src/turbulence/k_epsilon_rans.jl:24`~~ | ~~`ν_t = C_μ k²/ε` with simple `max()` floor; no Durbin bound~~ | **Fixed in v2.5.0 (Stage 4a)**: `StandardKEpsilon` gained optional `realizability_alpha` field; when > 0, ν_t is capped at `α · k / |S|` inside `solve_turbulence!` right before production is computed. Default 0 preserves classical formulation. |
| ~~k-ε production~~ | ~~`src/turbulence/k_epsilon_rans.jl`~~ | ~~Scalar strain magnitude `|S|²`~~ | **Verified correct**: `compute_strain_rate` at `src/turbulence/strain_rate.jl:21` computes full-tensor `|S| = √(2 S_ij S_ij)`; production is `ν_t · \|S\|²`. Audit claim was imprecise. |
| k-ε / k-ω low-Re | — | High-Re form only; no Launder-Sharma, Abid, or other low-Re damping functions | 4a |
| k-ω-SST blending | `src/turbulence/k_omega_sst.jl` | Simplified scalar blending; should be full F1/F2 blending with proper limiter | 4a |
| ~~Dynamic Smagorinsky~~ | ~~`src/turbulence/dynamic_smagorinsky.jl`~~ | ~~Scalar Germano identity, not full tensor form~~ | **Fixed in v2.5.0 (Stage 4c)**: `S̃_ij` now test-filtered per-component independently rather than approximated as `S_ij · |S̃|/|S|`. `|S̃|` computed from the test-filtered tensor directly (Lilly form). |
| ~~Wall functions~~ | ~~`src/turbulence/wall_functions.jl`~~ | ~~Assumes cells aligned with boundary normal; no skew/tangential projection~~ | **Fixed in v2.5.0 (Stage 4d)**: `_wall_projection` computes wall-normal distance `y = |d · n̂|` and tangential velocity `U_par = |U - (U·n̂)n̂|` per boundary face. Strips spurious normal-velocity contributions on skewed cells; identical to old formula on Cartesian walls with purely-tangential flow. |
| ~~Conjugate HT interface~~ | ~~`src/thermal/conjugate.jl`~~ | ~~Scalar face-averaged interface temperature~~ | **Fixed in v3.102.0 (Wave 1)**: per-face Patankar harmonic-mean interface flux replaces the scalar face-averaged temperature; pre-existing `_apply_perface_interface_fluxes!` was promoted to the default Dirichlet-Neumann path. Latent post-Stage-1b Dict/Vector bmap regression also resolved. |
| ~~VOF boundedness~~ | ~~`src/multiphase/boundedness.jl`~~ | ~~Hard clipping `clamp(α, 0, 1)` — not MULES~~ | **Fixed in v3.102.0 (Wave 1)**: `mules_limit_flux!` (clean-room Zalesak FCT, Weller 2006) wired into `alpha_transport!` as the default. The standalone primitive shipped in v3.91 (Evidence #7); v3.102 wires it into the actual solver path. `clip_alpha!` retained as a post-solve safety net. |
| ~~VOF interface reconstruction~~ | ~~`src/multiphase/`~~ | ~~No isoAdvector / sharp interface reconstruction~~ | **Fixed in v3.102.0 (Wave 1)**: `src/multiphase/iso_advector.jl` ships geometric isoAdvector face-flux reconstruction; selectable per-problem alongside the algebraic VOF path. |
| ~~VOF contact angles~~ | ~~`src/multiphase/surface_tension.jl`~~ | ~~Static/dynamic contact-angle models absent~~ | **Fixed in v3.102.0 (Wave 1)**: static contact angle and Cox-Voinov dynamic model both land; coupled into CSF curvature on wall faces. |
| ~~Combustion chemistry~~ | ~~`src/combustion/edm.jl`~~ | ~~One-step EDM only; no multi-step mechanisms, no FGM, no Cantera interface~~ | **Fixed in v3.103.0 (Wave 2)**: multi-step mechanisms (`src/combustion/multi_step.jl`), Flamelet-Generated Manifold (`src/combustion/fgm.jl`), and Cantera weak-dep extension (`ext/FVMCanteraExt/`) all land. EDC + EDM + finite-rate Arrhenius all production. |
| ~~Combustion diffusion~~ | ~~`src/combustion/species_transport.jl`~~ | ~~Lewis-unity implicit; no per-species Le exposure~~ | **Fixed in v3.103.0 (Wave 2)**: `src/combustion/variable_lewis.jl` exposes per-species Lewis number and folds into the diffusion term of `assemble_species!`. |
| ~~Radiation quadrature~~ | ~~`src/radiation/fvdom.jl`~~ | ~~fvDOM angular quadrature is skeleton; LSn/Tn sets absent~~ | **Already implemented (verified in v2.6.0)**: `src/radiation/fvdom.jl:60-135` carries proper Carlson-Lathrop level-symmetric S2 (4/8 dirs) and S4 (12/24 dirs) quadratures. Audit claim was outdated. S8/S12 extensions remain Stage 5c follow-ups. |
| ~~Radiation scattering~~ | ~~`src/radiation/fvdom.jl`~~ | ~~Scattering term absent~~ | **Fixed in v3.103.0 (Wave 2)**: in-scattering integral wired into the fvDOM ordinate sweep alongside S6/S8/S12 quadrature additions. |
| ~~Radiation wall BCs~~ | ~~`src/radiation/fvdom.jl`~~ | ~~Basic Dirichlet/Neumann only; no wavelength-banded emissivity~~ | **Fixed in v3.103.0 (Wave 2)**: `src/radiation/wsggm.jl` ships the Weighted Sum of Grey Gases Model with banded emissivity; couples through Marshak BC via per-band absorption coefficient. |
| ~~DPM collision~~ | ~~`src/lagrangian/collisions.jl`~~ | ~~Binary elastic only; no hard/soft-sphere DEM, no agglomeration/coalescence~~ | **Fixed in v3.103.0 (Wave 2)**: hard-sphere and soft-sphere (Hertz-Mindlin) DEM contact models both land; `src/lagrangian/agglomeration.jl` adds coalescence kernel. |
| ~~DPM breakup~~ | ~~`src/lagrangian/spray.jl`~~ | ~~Secondary breakup only (TAB/KHRT); no primary breakup (KH-ACT, LISA)~~ | **Fixed in v3.103.0 (Wave 2)**: `src/lagrangian/primary_breakup.jl` adds KH-ACT (Reitz) and LISA (Senecal) primary breakup. v3.107 adds `couple_primary_breakup_fsi!` for FSI coupling. |
| ~~DPM injection~~ | — | ~~No cone/hollow-cone/flat-fan injection patterns or rate-of-injection profiles~~ | **Fixed in v3.103.0 (Wave 2)**: `src/lagrangian/injection.jl` adds solid-cone, hollow-cone, flat-fan, and solid-stream injectors with rate-of-injection profiles. |
| ~~Dynamic-mesh GCL~~ | ~~`src/dynamic_mesh/ale.jl`~~ | ~~Geometric conservation law not verified for large deformation~~ | **Fixed in v2.6.0 (Stage 5d)**: `verify_gcl(phi_mesh, V_old, V_new, mesh, dt)` computes per-cell GCL residual and returns max; a GCL-consistent mesh motion yields zeros to machine precision. Runtime diagnostic; catches inconsistent face/volume pairs before they corrupt transport. `compute_mesh_flux!` already uses the 2nd-order face-velocity form. |
| Dynamic-mesh 6-DOF | — | No 6-DOF rigid-body solver | 5f |
| Dynamic-mesh topology | — | No dynamic refinement/coarsening or topology changes during a run | 5f |
| Overset / chimera | — | Absent | 5f |

### Structural / performance

| Component | File:Line | Issue | Fix Stage |
|-----------|-----------|-------|-----------|
| ~~CollocatedEquation assembly~~ | ~~`src/collocated/types.jl:181,192`, `src/collocated/laplacian.jl`, every `assemble_*!`~~ | ~~Random-pattern CSC insertion `A[P,N] += …` on every SIMPLE outer iteration~~ | **Fixed in v2.2.0-dev (Stage 1a)**: `SparsityPattern` pre-computes nzval indices at mesh-bind time; `add_diag!` / `add_face_coeffs_PN!` write `A.nzval[idx]` in O(1). 5× speedup on 40k-cell Laplacian; zero-allocation gate in `test/assembly_bench.jl`. Cyclic BCs + pressure ref-cell pinning still use slow path until cyclic pairs are plumbed into `build_collocated_sparsity` (Stage 1a follow-up). |
| ~~Operator hot-loop allocation~~ | ~~`src/collocated/gradient.jl:126-130`, `src/collocated/interpolation.jl:96`~~ | ~~`fill(…)` buffer and `Dict{Int,Int}` constructed on every call~~ | **Fixed in v2.2.0-dev (Stage 1b)**: `build_boundary_map` now returns `Vector{Int}` (O(1) indexed lookup, single allocation) instead of `Dict{Int,Int}`. `gradient!` accepts optional `scratch` + `bmap` kwargs for full zero-allocation use. The 5 inline `Dict(f => i for …)` constructions in `interpolation.jl`, `pressure.jl`, and `boundary_conditions.jl` migrated to `build_boundary_map(field, mesh)`. Verified zero-alloc gate in `test/assembly_bench.jl`. |
| ~~CollocatedEquation is scalar-only~~ | ~~`src/collocated/types.jl:181`~~ | ~~Single `Vector{T}` for `b`; two-fluid and coupled momentum-energy need a `BlockCollocatedEquation`~~ | **Fixed in v2.2.0-dev (Stage 1c)**: `BlockCollocatedEquation{T, NBlocks}` with `BlockSparsityPattern` + `add_block_diag!` / `add_block_offdiag_PN/NP!` helpers added alongside the scalar type. Cell-major layout, eagerly-built `N×N` CSC, same O(1) nzval-indexed write pattern. Infrastructure only — actual use by Eulerian two-fluid (Stage 6e) and coupled momentum-energy (Stage 3) wires on top. Verified in `test/assembly_bench.jl`. |
| ~~No AbstractFVMMesh supertype~~ | ~~`src/mesh/abstract_mesh.jl`~~ | ~~`FVMGeometry`, `StructuredMesh{1,2,3}D`, `UnstructuredFVMMesh` have no common supertype; conversion paths sparse~~ | **Fixed in v2.2.0-dev (Stage 1d)**: added umbrella `AbstractFiniteVolumeMesh{Dim}` in `src/mesh/abstract_mesh.jl`; retrofit `AbstractMesh{Dim}` (hyperbolic), `AbstractFVMMesh{Dim,T}` (parabolic/collocated), and `FVMGeometry` to subtype it. Generic `n_cells`/`n_faces`/`dim_of` methods in `src/mesh/generic_interface.jl` dispatch uniformly. Similar umbrella `AbstractFVMBoundaryCondition` added with `AbstractBoundaryCondition` and `AbstractHyperbolicBC` subtyping it. 22 gates in `test/sciml_contract_uniform.jl`. |
| ~~SciMLStructures.Tunable length-5~~ | ~~`src/core/sciml_structures.jl:130-144`~~ | ~~Hardcoded `[nu, density, alpha_U, alpha_p, tolerance]`; adding one tunable breaks all `remake` callers~~ | **Fixed in v2.2.0-dev (Stage 1e)**: replaced hardcoded positional indexing with a named-entry registry (`register_tunable!` + `tunable_schema`). Adding a new tunable (e.g., turbulence closure constant, rheology parameter) is now one function call; no edit to `canonicalize` or `replace` needed. `tunable_names` and `tunable_namedtuple` accessors for introspection. 14 gates in `test/sciml_contract_uniform.jl`. |
| ~~State containers non-generic~~ | ~~`src/incompressible/types.jl`~~ | ~~`Vector{T}` baked in; blocks KA.jl / GPU port without a rewrite~~ | **Fixed in v2.2.0-dev (Stage 1g)**: `CollocatedScalarField`, `CollocatedVectorField`, `FaceFluxField` parameterized on an `AbstractVector` container type `A`. Default constructors still produce `Vector{T}`; a future KA.jl / CuVector port is a container-type swap with no changes to downstream methods (existing `::CollocatedScalarField{T}` signatures match any `A` via UnionAll dispatch). 9 gates in `test/sciml_contract_uniform.jl`. |
| ~~No AbstractLinearOperator~~ | ~~`src/linear_solvers/`~~ | ~~`_dispatch_solve` takes `SparseMatrixCSC` directly; no matrix-free path~~ | **Fixed in v2.2.0-dev (Stage 1h)**: added `AbstractLinearOperator{T}` + `SparseMatrixLinearOperator{T, M}` in `src/linear_solvers/abstract_operator.jl`. `underlying_matrix(op)` / `as_linear_operator(A)` / `MatrixFreeError` / `mul!` / `size` interface. Stage 9e matrix-free operators plug in as peer subtypes without touching the sparse-backed path. 10 gates in `test/sciml_contract_uniform.jl`. |
| ~~MPI is full-mesh-per-rank~~ | ~~`ext/FVMMPIExt/distributed_mesh.jl:44`, `ext/FVMMPIExt/distributed_solve.jl:49-53`~~ | ~~Each rank stores full mesh AND assembles full matrix; halo exchange is decorative; only residual `Allreduce` uses MPI~~ | **Fixed in v2.3.0-dev (Stage 2)**: dep-free `partition_rcb` (recursive coordinate bisection) + `extract_local_mesh` build a true per-rank `UnstructuredFVMMesh` containing only owned + halo cells. `DistributedFVMMesh` stores the local submesh, not the global mesh. `HaloPattern` expressed in local indices. Full MPI solve on the local matrix via Additive Schwarz iteration. 980 gates in `test/mpi_partition.jl` (serial, deps-free). Full `mpiexec -n N` parity test in `test/mpi_parity.jl` (manual launch). Remaining: distributed `PSparseMatrix` via PartitionedArrays.jl for tighter serial-parallel parity and parallel AMG pressure preconditioning — Stage 2 follow-up. |
| `test/mpi_test.jl` not in runtests.jl | — | Manual `mpiexec -n 2 julia …` needed; `test/mpi_parity.jl` added in Stage 2 as the real parity oracle | 2 follow-up |

### Missing OpenFOAM features

Each slated for the stage noted in the roadmap:

| Feature | Status | Stage |
|---------|--------|-------|
| ~~Compressible pressure-based solvers (rhoSimpleFoam, rhoPimpleFoam)~~ | Landed v3.102.0 (Wave 1): `src/pressure_based/{compressible_simple,compressible_pimple}.jl`. rhoReactingFoam-equivalent multi-step + EDC coupling lands as Wave-2 follow-up. | 3 done |
| ~~Real-gas EOS (Peng-Robinson, Redlich-Kwong, tabulated)~~ | Landed v3.102.0 + v3.105.0: `src/eos/`, `src/pressure_based/{eos_coupling,thermo_models}.jl`; CoolProp tabulated path via `ext/FVMCoolPropExt`. | 3b done |
| ~~Non-Newtonian rheology (power-law, Bird-Carreau, Herschel-Bulkley, Casson)~~ | Landed v3.102.0 (Wave 1): `src/pressure_based/rheology.jl`. | 3c done |
| ~~Moving Reference Frame (MRF)~~ | Landed v2.7.0 (Stage 6a): `RotationalMRFZone`, `mrf_momentum_source`, `mrf_momentum_source_2d_planar`. Verified Coriolis+centrifugal for planar rotation. | 6a done |
| ~~Arbitrary Mesh Interface (AMI) / sliding mesh~~ | Landed v3.103.0 (Wave 2): `src/dynamic_mesh/ami.jl`; sliding-mesh + overset (`src/dynamic_mesh/overset.jl`) + topoChanger (`src/dynamic_mesh/topo_changer.jl`) all production. | 6b done |
| ~~Porous media (Darcy-Forchheimer)~~ | Landed v2.7.0 (Stage 6c): `DarcyPorous`, `DarcyForchheimerPorous`, `OrthotropicPorous` with `porous_momentum_source`. | 6c done |
| ~~Cavitation (Kunz, Schnerr-Sauer, Merkle)~~ | Landed v2.7.0 (Stage 6d): three concrete cavitation models under `AbstractCavitationModel`; `cavitation_source` returns `(m_plus, m_minus)` per cell. | 6d done |
| ~~Eulerian two-fluid~~ | Landed v3.106.0 + v3.107.0: types in v3.106 (Wave 5), production solver in v3.107 (`src/multiphase/two_fluid_solver.jl`) using `BlockCollocatedEquation{T,2}` with off-diagonal Ishii-Zuber / Gibilaro drag-closure linearization. Inner-iteration Newton on the coupled momentum block. `TwoFluidProblem` + `solve_two_fluid` exposed. | 6e done |
| ~~Aeroacoustics (FW-H, sponge zones)~~ | Landed v2.7.0 (Stage 6f) + v3.104.0 (Wave 3): stationary-surface Curle + monopole + moving-surface FW-H; Lighthill stub; PML sponge zones in `src/aeroacoustics/`. | 6f done |
| ~~Population balance modeling~~ | Landed v2.7.0 + v3.104.0 (Wave 3): QMoM (Wheeler/PD) + DQMoM + Class Method all in `src/population_balance/`. | 6g done |
| ~~Wall-modeled LES (WMLES)~~ | Landed v3.102.0 (Wave 1): `EquilibriumWMLES` + `SADDES` in `src/turbulence/{wmles,sa_ddes}.jl`. v3.107 adds full Shur-2008 IDDES shielding. | 4a done |
| ~~Solid mechanics~~ | Landed v2.8.0 (Stage 7a): `IsotropicElastic`, `SolidDisplacementProblem`, `stress_tensor`, `small_strain_tensor`, `cantilever_tip_deflection`. Linear small-strain MVP; finite-strain / plasticity deferred. | 7a done |
| ~~FSI~~ | Landed v2.8.0 (Stage 7b): partitioned Dirichlet-Neumann with `AitkenRelaxation` + `FSIInterface` + `interface_residual_norm`. Full solver loop integration is a follow-up. | 7b done |
| ~~Function objects / coded BCs / expression BCs~~ | Landed v2.8.0 (Stage 7d) + v3.105.0 (Wave 4): `PointProbe`, `ForceProbe`, `StringExpressionBC` (renamed from `ExpressionBC` in v3.108 to disambiguate from parabolic `ExpressionBC`), `FieldStatistics`. Closure-based + a string-DSL path via `StringExpressionBC`. | 7d done |
| ~~snappyHexMesh-equivalent mesh generation~~ | Landed v3.107.0 (v3.1 wave): native castellated refinement + STL surface snap from `src/mesh_generation/`. Layer addition still deferred to v3.2. | 8a partial |
| ~~Gmsh automation pipeline~~ | Landed v3.105.0 (Wave 4): `ext/FVMGmshExt/` weak-dep extension covers `.msh` v4 read + Gmsh API run-from-Julia. | 8b done |
| ~~AMR on collocated side~~ | Landed v3.105.0 (Wave 4): `src/amr_collocated/` adds tree-augmented refinement, gradient + residual + ZZ markers, conservative regrid. | 8c done |
| ~~Error indicators~~ | Landed v2.9.0 + v3.105.0 (Wave 4): gradient, ZZ, and residual-based indicators all in `src/amr_collocated/`. | 8d done |
| ~~Full adjoint (SciMLSensitivity integration)~~ | Landed v3.105.0 + v3.107.0 (v3.1 wave): steady-SIMPLE adjoint in v3.105; transient PIMPLE adjoint with uniform checkpointing (`solve_transient_adjoint_linear`) in v3.107. Enzyme full-solver AD remains a v3.2 follow-up. | 9a–c done (steady + linear-transient) |
| ~~GPU backends for collocated~~ | Landed v3.105.0 (Wave 4): `ext/FVMKAExt/` (KernelAbstractions weak dep) provides operator dispatch on CPU/CUDA/AMD/Metal backends. v3.108 fixes precompile-time method-overwrite bug on backend dispatch. | 9d partial |
| ~~Matrix-free linear operators~~ | Landed v2.10.0 (Stage 9e): `MatrixFreeLinearOperator{T, F, Ft, D}`. | 9e done |
| ~~Unitful integration~~ | Landed v2.10.0 (Stage 9f) + v3.105.0 (Wave 4 hook). | 9f done |
| ~~Binary OpenFOAM polyMesh reader~~ | Landed v3.0 cycle (`src/mesh/openfoam_io.jl`). Both ASCII + binary polyMesh now read. | 3 done |

## Deferred to v3.2 / v3.3

The following items are explicitly out of scope for the v3.102→v3.108 production wave and remain on the v3.2+ roadmap:

| Item | Owner | Notes |
|------|-------|-------|
| snappyHexMesh layer addition | `src/mesh_generation/` | Castellated refinement + surface snap landed in v3.107; near-wall layer addition (with collapse and feature-edge handling) remains. |
| Enzyme full-solver AD | `ext/FVMEnzymeExt/` | Stub landed in v3.105; differentiation through full SIMPLE / PIMPLE outer iteration (mutating fixed-point) requires Enzyme rules for `_dispatch_solve` and per-iteration cache aliasing. Steady + linear-transient adjoint via SciMLSensitivity already production. |
| IDDES `h_max` from real edge lengths | `src/turbulence/iddes.jl` | Shur-2008 shielding production in v3.107 but the wall-normal length scale uses `V_c^(1/Dim)` as a surrogate for `h_max = max edge length`. Requires per-cell edge-length cache plumbed through `UnstructuredFVMMesh`. |
| Sandia Flame D published-benchmark | `validation/published_benchmarks/` | Combustion bench harness exists; running EDC + variable-Lewis + radiation-coupled vs. published Sandia Flame D Raman/Rayleigh data is a v3.2 deliverable for `combustion` stable promotion. |
| Published-benchmark execution + stable-tier promotions | `validation/manifest.toml` | Harness scaffold (gated by `FVM_RUN_BENCHMARKS=true`) lands in v3.107. The full ≥3-published-benchmark suite per feature is the gate for `provisional` → `stable`; expected to run quarterly at the user's terminal until the wallclock budget is reduced. |
| Two-fluid VOF cross-coupling and full Eulerian reactive | `src/multiphase/two_fluid_solver.jl` | v3.107 ships the production block-coupled momentum-with-drag solver. Energy + species cross-coupling on the same block matrix is a v3.2 follow-up. |
| FW-H porous + supersonic regime | `src/aeroacoustics/` | Stationary + moving-surface FW-H production; porous-FW-H surfaces and shock-emission corrections deferred. |
