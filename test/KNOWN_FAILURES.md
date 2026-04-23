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
| ~~Non-orthogonal correction~~ | ~~`src/collocated/gradient.jl:144-149`~~ | ~~Interpolated-gradient only; no over-relaxed variant~~ | **Partially fixed in v2.4.0 (Stage 3c)**: `assemble_laplacian!` now supports `NON_ORTHO_MINIMUM` / `NON_ORTHO_ORTHOGONAL` / `NON_ORTHO_OVER_RELAXED` modes via a `correction_mode` keyword; default is over-relaxed (Jasak 1996 Ch. 4). On orthogonal meshes all three are identical; on skewed meshes the over-relaxed implicit coefficient scales by 1/cosθ. Least-squares gradient alternative is a Stage 3 follow-up. |
| Laplacian skewness | `src/collocated/laplacian.jl` | No face-skewness correction term; accuracy drops on heavily skewed meshes | 3 follow-up |
| ~~k-ε realizability~~ | ~~`src/turbulence/k_epsilon_rans.jl:24`~~ | ~~`ν_t = C_μ k²/ε` with simple `max()` floor; no Durbin bound~~ | **Fixed in v2.5.0 (Stage 4a)**: `StandardKEpsilon` gained optional `realizability_alpha` field; when > 0, ν_t is capped at `α · k / |S|` inside `solve_turbulence!` right before production is computed. Default 0 preserves classical formulation. |
| ~~k-ε production~~ | ~~`src/turbulence/k_epsilon_rans.jl`~~ | ~~Scalar strain magnitude `|S|²`~~ | **Verified correct**: `compute_strain_rate` at `src/turbulence/strain_rate.jl:21` computes full-tensor `|S| = √(2 S_ij S_ij)`; production is `ν_t · \|S\|²`. Audit claim was imprecise. |
| k-ε / k-ω low-Re | — | High-Re form only; no Launder-Sharma, Abid, or other low-Re damping functions | 4a |
| k-ω-SST blending | `src/turbulence/k_omega_sst.jl` | Simplified scalar blending; should be full F1/F2 blending with proper limiter | 4a |
| ~~Dynamic Smagorinsky~~ | ~~`src/turbulence/dynamic_smagorinsky.jl`~~ | ~~Scalar Germano identity, not full tensor form~~ | **Fixed in v2.5.0 (Stage 4c)**: `S̃_ij` now test-filtered per-component independently rather than approximated as `S_ij · |S̃|/|S|`. `|S̃|` computed from the test-filtered tensor directly (Lilly form). |
| ~~Wall functions~~ | ~~`src/turbulence/wall_functions.jl`~~ | ~~Assumes cells aligned with boundary normal; no skew/tangential projection~~ | **Fixed in v2.5.0 (Stage 4d)**: `_wall_projection` computes wall-normal distance `y = |d · n̂|` and tangential velocity `U_par = |U - (U·n̂)n̂|` per boundary face. Strips spurious normal-velocity contributions on skewed cells; identical to old formula on Cartesian walls with purely-tangential flow. |
| ~~Conjugate HT interface~~ | ~~`src/thermal/conjugate.jl`~~ | ~~Scalar face-averaged interface temperature~~ | **Fixed in v2.6.0 (Stage 5a)**: per-face heat-flux correction in `_apply_perface_interface_fluxes!` was already present; fixed latent post-Stage-1b Dict/Vector bmap regression. |
| ~~VOF boundedness~~ | ~~`src/multiphase/boundedness.jl`~~ | ~~Hard clipping `clamp(α, 0, 1)` — not MULES~~ | **Fixed in v2.6.0 (Stage 5b)**: `mules_limit_flux!` implements the Zalesak FCT limiter (clean-room from Weller 2006). Takes upwind + high-order flux and returns λ_f-blended flux guaranteeing α stays in [0, 1] after one explicit Euler step. `clip_alpha!` retained as a post-solve safety net. |
| VOF interface reconstruction | `src/multiphase/` | No isoAdvector / sharp interface reconstruction | 5b |
| VOF contact angles | `src/multiphase/surface_tension.jl` | Static/dynamic contact-angle models absent | 5b |
| Combustion chemistry | `src/combustion/edm.jl` | One-step EDM only; no multi-step mechanisms, no FGM, no Cantera interface | 5c |
| Combustion diffusion | `src/combustion/species_transport.jl` | Lewis-unity implicit; no per-species Le exposure | 5c |
| ~~Radiation quadrature~~ | ~~`src/radiation/fvdom.jl`~~ | ~~fvDOM angular quadrature is skeleton; LSn/Tn sets absent~~ | **Already implemented (verified in v2.6.0)**: `src/radiation/fvdom.jl:60-135` carries proper Carlson-Lathrop level-symmetric S2 (4/8 dirs) and S4 (12/24 dirs) quadratures. Audit claim was outdated. S8/S12 extensions remain Stage 5c follow-ups. |
| Radiation scattering | `src/radiation/fvdom.jl` | Scattering term absent | 5d |
| Radiation wall BCs | `src/radiation/fvdom.jl` | Basic Dirichlet/Neumann only; no wavelength-banded emissivity | 5d |
| DPM collision | `src/lagrangian/collisions.jl` | Binary elastic only; no hard/soft-sphere DEM, no agglomeration/coalescence | 5e |
| DPM breakup | `src/lagrangian/spray.jl` | Secondary breakup only (TAB/KHRT); no primary breakup (KH-ACT, LISA) | 5e, 7c |
| DPM injection | — | No cone/hollow-cone/flat-fan injection patterns or rate-of-injection profiles | 5e |
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
| Compressible pressure-based solvers (rhoSimpleFoam, rhoPimpleFoam, rhoReactingFoam) | Absent | 3 |
| Real-gas EOS (Peng-Robinson, Redlich-Kwong, tabulated) | Absent | 3b |
| Non-Newtonian rheology (power-law, Bird-Carreau, Herschel-Bulkley, Casson) | Absent | 3c |
| ~~Moving Reference Frame (MRF)~~ | Landed v2.7.0 (Stage 6a): `RotationalMRFZone`, `mrf_momentum_source`, `mrf_momentum_source_2d_planar`. Verified Coriolis+centrifugal for planar rotation. | 6a done |
| Arbitrary Mesh Interface (AMI) / sliding mesh | Absent — still Stage 6 follow-up | 6b |
| ~~Porous media (Darcy-Forchheimer)~~ | Landed v2.7.0 (Stage 6c): `DarcyPorous`, `DarcyForchheimerPorous`, `OrthotropicPorous` with `porous_momentum_source`. | 6c done |
| ~~Cavitation (Kunz, Schnerr-Sauer, Merkle)~~ | Landed v2.7.0 (Stage 6d): three concrete cavitation models under `AbstractCavitationModel`; `cavitation_source` returns `(m_plus, m_minus)` per cell. | 6d done |
| Eulerian two-fluid | Absent — requires block-coupled equation (Stage 1c present, solver wiring is Stage 6 follow-up) | 6e |
| ~~Aeroacoustics (FW-H, sponge zones)~~ | Landed v2.7.0 (Stage 6f): `FWHSurface`, `FWHObserver`, `curle_dipole_pressure`, `fwh_monopole_pressure`. Stationary-surface Curle + monopole; moving-surface + porous-FW-H are follow-ups. Sponge-zones still pending. | 6f partial |
| ~~Population balance modeling~~ | Landed v2.7.0 (Stage 6g): `qmom_recover_abscissae_weights` (Wheeler/PD algorithm) + moment sources for growth, binary aggregation, binary breakage. CM + DQMoM extensions are follow-ups. | 6g done |
| Wall-modeled LES (WMLES) | Absent | 4a |
| ~~Solid mechanics~~ | Landed v2.8.0 (Stage 7a): `IsotropicElastic`, `SolidDisplacementProblem`, `stress_tensor`, `small_strain_tensor`, `cantilever_tip_deflection`. Linear small-strain MVP; finite-strain / plasticity deferred. | 7a done |
| ~~FSI~~ | Landed v2.8.0 (Stage 7b): partitioned Dirichlet-Neumann with `AitkenRelaxation` + `FSIInterface` + `interface_residual_norm`. Full solver loop integration is a follow-up. | 7b done |
| ~~Function objects / coded BCs / expression BCs~~ | Landed v2.8.0 (Stage 7d): `PointProbe`, `ForceProbe`, `ExpressionBC`, `FieldStatistics` with shared `AbstractFunctionObject` + `run!` interface. Closure-based (no string DSL). | 7d done |
| ~~snappyHexMesh-equivalent mesh generation~~ | Landed v2.9.0 (Stage 8a): `Octree{Dim, T}` + `build_octree` + `refine_near_sphere!` + STL-bandwidth-refinement infrastructure. Full STL snapping + layer addition + topology healing are follow-ups. | 8a partial |
| Gmsh automation pipeline | Absent — still 8b follow-up | 8b |
| ~~AMR on collocated side~~ | Landed v2.9.0 (Stage 8c): `mark_cells_by_gradient` refinement markers + `flux_correction_factor` conservation check. Tree-augmented mesh structure for actual h-refinement is a follow-up. | 8c partial |
| ~~Error indicators~~ | Landed v2.9.0 (Stage 8d): `zz_error_indicator` (Zienkiewicz-Zhu recovery-based); gradient-based marking already in Stage 8c. Residual-based indicator is a follow-up. | 8d done |
| Full adjoint (SciMLSensitivity integration) | Absent | 9a–c |
| GPU backends for collocated | Absent | 9d |
| ~~Matrix-free linear operators~~ | Landed v2.10.0 (Stage 9e): `MatrixFreeLinearOperator{T, F, Ft, D}` subtypes `AbstractLinearOperator`; user closure `matvec!(y, x)` with optional transpose and diagonal. Plugs into existing `_dispatch_solve` path. | 9e done |
| ~~Unitful integration~~ | Landed v2.10.0 (Stage 9f): `strip_units`, `is_dimensionless`, `as_si_velocity/density/viscosity/temperature`. Unit-checking at problem-setup boundary; hot-path remains `Float64`. | 9f done |
| Binary OpenFOAM polyMesh reader | ASCII only (`src/mesh/openfoam_io.jl:22`) | 3 |
