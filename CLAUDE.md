# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FiniteVolumeMethod.jl is a Julia package for solving PDEs with three solver families:
- **Parabolic/elliptic**: Cell-vertex solver on unstructured triangular meshes (via DelaunayTriangulation.jl)
- **Hyperbolic**: Cell-centered solver on structured 1D/2D/3D meshes (Euler, MHD, Navier-Stokes, GRMHD, etc.)
- **Collocated incompressible**: OpenFOAM-style cell-centered solver on unstructured polyhedral meshes with SIMPLE/PISO/PIMPLE pressure-velocity coupling, turbulence (RANS/LES/hybrid), heat transfer, radiation, combustion, multiphase VOF, Lagrangian DPM, and dynamic mesh

Requires Julia 1.10+. Supports current stable + LTS releases. Targets eventual inclusion in the SciML ecosystem. Currently at `4.0.0-DEV`: the v2→v3 research-grade overhaul is complete through v3.114, and the v4 structural rework (submodules, curated exports, SciML contract alignment — Stages 3–7b) has landed on `main`. Only features marked `stable` in the capability matrix and validation manifest are publication-grade. The collocated solver stack is `experimental`.

The capability matrix (`docs/src/capability_matrix.md`) and validation manifest (`validation/manifest.toml`) are the authoritative contracts for feature maturity, V&V status, and CI inclusion.

## Common Commands

### Running Tests
```bash
# Full test suite (slow — runs tutorials, verification, and governance checks)
julia --project -e 'using Pkg; Pkg.test()'

# Single test file (test env must have FiniteVolumeMethod dev'd)
# Recommended fast-iteration loop — collocated test files pull
# `build_cartesian_unstructured_mesh` from test/TestHelpers.jl
# (centralized in v2.1.0) so they run standalone.
julia --project=test test/<filename>.jl

# Single test file via Docker
TEST_FILE=test/geometry/geometry.jl make ci-test-file

# Scientific evidence subset (used by CI scientific-smoke lane)
julia --project=test test/governance/scientific_evidence.jl
```

### Formatting (Runic)
All `.jl` files must pass Runic formatting. A pre-commit hook at `.git/hooks/pre-commit` enforces this locally; CI runs `fredrikekre/runic-action@v1` on every push/PR.
```bash
# Check formatting (Runic lives in the global/default Julia env, not the project env)
julia -e 'using Runic; Runic.main(["--check", "."])'

# Auto-fix formatting
julia -e 'using Runic; Runic.main(["--inplace", "."])'

# Via Docker
make ci-format       # check only
make ci-format-fix   # auto-fix
```
Key Runic rules: spaces around `=` in kwargs (`atol = 0.01`), 4-space continuation indent (not aligned to `(`), spaces around `/` in arithmetic.

### Local CI (Docker-based)
GitHub Actions status (see `.github/workflows/README.md`): `CI.yml` is active on pushes/PRs with five jobs — environment-integrity, unit-interop, scientific-smoke, published-benchmarks (all 5 benchmarks must execute their physics assertions; deferrals count as broken and fail the job), and docs (build + GitHub Pages deploy). `FormatCheck.yml` runs on pushes/PRs. `Nightly.yml` runs weekly (Monday 03:00 UTC cron) with the `FVM_RUN_VANDV`-gated collocated V&V cases. `benchmarks.yml`, `Docs.yml`, `docs-quality.yml`, and `jet.yml` are manual-dispatch only. `Release.yml.disabled` and `TagBot.yml.disabled` are disabled (TagBot cannot fire — the fork is unregistered). For local iteration, use the Makefile Docker lanes:
```bash
make ci-fast              # Fast API/interop lane
make ci-smoke             # Scientific smoke tests
make ci-full-evidence     # Full scientific evidence
make ci-performance       # Performance baselines
make ci-release-audit     # Release audit lane
make ci-format            # Format check only
make ci-published-benchmarks  # 5-case published-benchmark suite (native, no Docker)
make ci-all               # All lanes
make ci-repl              # Interactive Julia REPL in container
```
First run requires `make ci-build` (downloads + precompiles all deps). Docker Desktop needs ≥12 GB memory.

### Documentation
```bash
# Build docs locally (executes examples — slow)
make ci-docs

# Build docs with CI subset only
make ci-docs-ci

# Live-server for docs development
julia --project=docs docs/liveserver.jl
```

## Architecture

### Module Structure (Stage 3 of the v4 overhaul)
`src/FiniteVolumeMethod.jl` is the single include site (the former `src/layers/` files were dissolved in Stage 3g). It loads real submodules in dependency order, with flat cross-family glue between them — never import backwards in this chain:

1. `Geometry` (`src/geometry/`) — all mesh types + geometry; then `Numerics` (`src/numerics/`) — backends, EOS, schemes, kernels, linear-solver config
2. `VertexConditions` (`src/vertex_conditions/`) → `Parabolic` (`src/parabolic/`) → `Collocated` (`src/collocated/`, with nested `Collocated.Physics` for turbulence/thermal/radiation/combustion)
3. `Hyperbolic` (`src/hyperbolic/`, incl. the semidiscrete SciML bridge `src/hyperbolic/core/` and `coupling/`) → flat `src/sciml/` glue (symbolic indexing, SciMLStructures, `remake`, `solve.jl`)
4. `Experimental` (`src/experimental/`) — quarantined scaffolds (pressure_based, aeroacoustics, population_balance, solid_mechanics, fsi, mesh_generation, adjoint, parallel), landed in Stage 3h; entry points warn once per feature
5. `FVMIO` (`src/io/`) — dashboard session types, output management, diagnostics, VTK/HDF5/checkpoint extension stubs; then flat `capabilities.jl`

After each submodule the main module has `import .Sub: ...` guard blocks — these keep unexported internals resolving as `FiniteVolumeMethod.name` for tests/docs/extensions and prevent dispatch fracture where flat code extends submodule generics. Do not remove them casually.

### Key Source Directories
- `src/parabolic/` — Cell-vertex solver: types, mesh variants, assembly, boundary conditions, gradients, limiters, turbulence models
- `src/hyperbolic/` — Cell-centered solver: conservation laws, Riemann solvers (HLL/HLLC/HLLD), reconstruction (MUSCL/PPM/WENO), plus advanced physics (Navier-Stokes, MHD variants, GRMHD, IMEX)
- `src/sciml/` — flat cross-family SciML glue: symbolic indexing (`symbolic_indexing.jl`), SciMLStructures parameter partitioning (`sciml_structures.jl`), `remake` support (`remake.jl`), and the parabolic `solve.jl`. The semidiscrete caches/state mapping/CFL callback/ODE construction moved into `src/hyperbolic/core/` (Stage 3e). `sciml_problem(prob)` returns the underlying `ODEProblem` for hyperbolic solvers (used to access `p`/`u0` for e.g. `compute_initial_dt`).
- `src/collocated/` — Phase 0: Cell-centered operators on `UnstructuredFVMMesh` (types, interpolation, gradient, laplacian, divergence, ddt, cyclic BC assembly). Foundation for all OpenFOAM-style solvers
- `src/incompressible/` — Phase 1: SIMPLE/PISO/PIMPLE pressure-velocity coupling (types, BCs, momentum, pressure, correction, residuals, solver loops, SciML interface with `CommonSolve.solve` dispatch and `IncompressibleSolution`)
- `src/turbulence/` — Phase 2a/2b: RANS (k-ε, k-ω, k-ω SST, SA), LES (Smagorinsky, WALE, dynamic), hybrid (DDES). Interface, strain rate, wall distance, wall functions, solver wrappers
- `src/thermal/` — Phase 3: Energy equation, buoyancy (Boussinesq), solid conduction, conjugate heat transfer (Dirichlet-Neumann iteration)
- `src/mesh/` — Phase 4: OpenFOAM polyMesh reader/writer, `UnstructuredMesh3D` → `UnstructuredFVMMesh` converter, polyhedral volumes (hex/prism/pyramid), mesh quality metrics
- `src/linear_solvers/` — Phase 5: `FVMSolverConfig` per-field solver/preconditioner config, `_dispatch_solve` for field-aware solver selection, preconditioner dispatch with Val-based extension pattern
- `src/multiphase/` — Phase 7: VOF alpha transport with interface compression, boundedness limiter, mixture properties, CSF surface tension
- `src/combustion/` — Phase 8: Species transport, EDM and EDC reaction models, heat release coupling
- `src/radiation/` — Phase 9: P1 and fvDOM radiation models, Marshak wall BCs, thermal coupling
- `src/dynamic_mesh/` — Phase 10: ALE formulation, solid body motion, Laplacian motion solver, mesh update, face sweep flux
- `src/lagrangian/` — Phase 11: Drag models (Stokes, Schiller-Naumann), Ranz-Marshall heat transfer, spray breakup (TAB, KHRT), two-way coupling (PSI-cell)
- `src/postprocessing/` — Phase 12: Vorticity, Q-criterion, wall shear stress, y+, Nusselt, force coefficients, line sampling, field statistics
- `src/experimental/` — Stage 3h: quarantined scaffolds (pressure_based, aeroacoustics, population_balance, solid_mechanics, fsi, mesh_generation, adjoint, parallel MPI stubs) wrapped in `module Experimental`; entry points warn once per feature
- `src/amr/` — Block-structured AMR with prolongation, restriction, flux correction, subcycling
- `src/constrained_transport/` — Divergence-free magnetic field evolution (div-B preservation for 2D and 3D)
- `src/coupling/` — Multi-physics operator splitting (Lie-Trotter, Strang)

### Key Design Patterns

**Ghost-cell state mapping**: The hyperbolic solver maintains two representations of the solution. The ODE integrator sees a flat `Vector{SVector{N,FT}}` of interior cells only. Inside the RHS function, `unfold_to_padded!()` copies this into `cache.padded_U` (an array with ghost cells for boundary stencils), the RHS is computed on the padded array, and `fold_from_padded!()` copies the result back. This pattern enables allocation-free time-stepping and appears identically for 1D/2D/3D/AMR.

**Cache-as-parameter**: All hyperbolic solvers use pre-allocated `AbstractSemidiscreteCache` subtypes (`HyperbolicCache1D`, `HyperbolicCache2D`, `MHDCTCache2D`, `AMRCache`, etc.) that hold padded arrays, flux buffers, the problem object, and grid metadata. The cache is passed as the ODE parameter `p` so the RHS function never allocates.

**Conservation law interface**: The hyperbolic solver is built on `AbstractConservationLaw{Dim}`. To add new physics, subtype it and implement: `nvariables(law)`, `physical_flux(law, u, dir)`, `max_wave_speed(law, u, dir)`, `conserved_to_primitive(law, u)`, `primitive_to_conserved(law, w)`. Existing laws: `EulerEquations`, `IdealMHDEquations`, `NavierStokesEquations`, `GRMHDEquations`, `ShallowWaterEquations`, `ReactiveEulerEquations`, etc.

**Parabolic assembly**: The parabolic solver assembles `M du/dt + A u = b` matrices (dimension-specific files in `src/parabolic/assembly/`), then converts to `ODEProblem`/`LinearProblem` via Layer 3 helpers. Problem types are `FVMProblem` (single-field), `FVMSystem` (multi-field coupled), and `SteadyFVMProblem` (steady-state wrapper).

**Collocated incompressible assembly**: The collocated solver assembles `CollocatedEquation{T}` (sparse A + RHS b) using Phase 0 operators (`assemble_convection!`, `assemble_laplacian!`, `assemble_ddt_euler!`), then converts to `LinearProblem` via `to_linear_problem()`. Pressure-velocity coupling via SIMPLE (steady) / PISO / PIMPLE (transient). All sub-solves dispatch through `_dispatch_solve(lp, linear_solver, solver_config, field_name)` which routes to `FVMSolverConfig` when provided.

**SciML integration for incompressible**: `CommonSolve.solve(prob::IncompressibleProblem, alg)` returns `IncompressibleSolution` with symbolic indexing (`sol[:U]`, `sol[:p]`). `SciMLBase.remake(prob; nu=...)` modifies problem parameters. `SciMLStructures.Tunable` exposes `[nu, density, alpha_U, alpha_p, tolerance]` for parameter sensitivity workflows.

### Package Extensions
Defined in `Project.toml` under `[extensions]`:
- `FVMCUDAExt` (CUDA) — GPU backend (currently only 2D Euler; most solvers are CPU-only)
- `FVMVTKExt` (WriteVTK) — VTK output
- `FVMHdf5Ext` (HDF5) — HDF5 I/O
- `FVMCheckpointExt` (JLD2) — Checkpointing
- `FVMDashboardExt` / `FVMDashboardServerExt` (JSON3, HTTP) — Live dashboard
- `FVMRecipesExt` (RecipesBase) — Plot recipes for 1D/2D hyperbolic solutions
- `FVMAMGExt` (AlgebraicMultigrid) — AMG preconditioner for pressure Poisson
- `FVMILUExt` (IncompleteLU) — ILU preconditioner for velocity/scalar equations
- `FVMLinearSolveExt` (LinearSolve) — Krylov solver constructors (CG, BiCGSTAB, GMRES)
- `FVMMPIExt` (MPI + PartitionedArrays) — Distributed mesh, halo exchange, parallel SIMPLE

### Test Organization
`test/runtests.jl` orchestrates all tests via `safe_include()`, which runs each test file in its own anonymous module to prevent namespace pollution between tests. The test suite includes:

- **Unit tests** — `test/geometry/geometry.jl`, `test/parabolic/conditions.jl`, `test/hyperbolic/hyperbolic.jl`, `test/hyperbolic/mhd.jl`, `test/parabolic/advanced_bcs.jl` (parabolic boundary gradient / segment utilities), `test/hyperbolic/advanced_numerics.jl` (Phase 13: PPM, positivity-preserving limiter), `test/hyperbolic/extended_physics.jl` (extended conservation laws), etc.
- **Collocated solver tests** — `test/collocated/incompressible.jl` (94 tests), `test/collocated/incompressible_sciml.jl` (58 tests), `test/collocated/turbulence_rans.jl` (127), `test/collocated/turbulence_les.jl` (92), `test/collocated/thermal.jl` (132), `test/geometry/mesh_io.jl` (37), `test/collocated/linear_solvers.jl` (35), `test/collocated/multiphase_vof.jl` (57), `test/collocated/combustion.jl` (49), `test/collocated/radiation.jl` (71), `test/collocated/lagrangian_dpm.jl` (53), `test/collocated/dynamic_mesh.jl` (72), `test/collocated/postprocessing.jl` (100), `test/collocated/remaining_features.jl` (116)
- **Tutorials as tests** — Literate.jl scripts from `docs/src/literate_tutorials/` and `docs/src/literate_wyos/` are executed as testsets (docs are tested code)
- **Verification cases** — driven by `validation/manifest.toml` via the `RepoValidationManifest` module; scripts from `docs/src/literate_verification/`
- **Governance** — Aqua.jl quality, environment integrity, repository governance, reproducibility bundles, quality ledger
- **MPI tests** — `test/experimental/mpi_test.jl` (NOT in runtests.jl — requires `mpiexec -n 2 julia --project=test test/experimental/mpi_test.jl`)

Note: `keller_segel_chemotaxis.jl` is explicitly excluded from the tutorials testset. Collocated solver test files get `build_cartesian_unstructured_mesh` from `test/TestHelpers.jl` (each file `include`s it; centralized in v2.1.0).

`test/` mirrors the module tree: `test/{geometry,parabolic,hyperbolic,collocated,sciml,experimental,governance}/`, with the shared helpers (`TestHelpers.jl`, `test_functions.jl`, `verification_utils.jl`) and `runtests.jl` at the root. Test files reach the helpers via `include(joinpath(@__DIR__, "..", "TestHelpers.jl"))` and the repo root via `dirname(dirname(@__DIR__))`.

`runtests.jl` is a thin dispatcher over a `TESTS` table of `(group, name, path)` rows. `FVM_TEST_GROUP` selects which groups run — `all` (default), or a comma-separated subset:
```bash
FVM_TEST_GROUP=collocated,sciml julia --project -e 'using Pkg; Pkg.test()'
```
Groups: `geometry`, `parabolic`, `hyperbolic`, `collocated`, `sciml`, `experimental`, `governance`, `tutorials`, `verification`.

To add a new test file, create it in the family directory it belongs to and add a row to the `TESTS` table in `test/runtests.jl`. Test files NOT included in `runtests.jl` (run manually or by dedicated lanes): `test/hyperbolic/cuda_hyperbolic_2d.jl` (requires CUDA), `test/experimental/mpi_test.jl` and `test/experimental/mpi_parity.jl` (require `mpiexec`), `test/governance/test_jet.jl` (JET lane, `jet.yml`), `test/governance/scientific_evidence.jl` (CI scientific-smoke lane), `test/governance/performance_baselines.jl` and `test/governance/release_audit.jl` (Makefile lanes), and the helpers `TestHelpers.jl`/`test_functions.jl`/`verification_utils.jl`. `test/parabolic/parabolic_mesh.jl` and `test/parabolic/io.jl` ARE included in `runtests.jl`.

### Validation Infrastructure
- `validation/manifest.toml` — Machine-readable source of truth for feature maturity, V&V status, and CI inclusion. Features are `stable`, `provisional`, or `experimental`.
- `validation/manifest.jl` — Julia module (`RepoValidationManifest`) that parses the manifest; used by both tests and docs builds
- `test/KNOWN_FAILURES.md` — Documents known broken/skipped/demoted tests

### Collocated Solver Key Types
- `IncompressibleProblem{Dim, T}` — problem definition (mesh, BCs, algorithm, nu, density, model)
- `IncompressibleModel` — physics composition carried by the problem: `TurbulenceComponent`, `ThermalComponent`, `RadiationComponent`, `CombustionComponent`, `porous_zones`, `mrf_zones`. Component dependencies are validated on construction; query with `has_turbulence`/`has_thermal`/`has_radiation`/`has_combustion`/`has_porous_zones`/`has_mrf_zones`/`is_plain_flow`
- `IncompressibleState{Dim, T}` — mutable flow state (U, p, phi, A_P, H_U)
- `IncompressibleSolution{Dim, T}` — SciML-compatible result with `sol[:U]`, `sol[:p]` access
- `SolveResult{Dim, T}` — raw result with converged, iterations, residuals
- `SIMPLE{T}`, `PISO{T}`, `PIMPLE{T}` — pressure-velocity coupling algorithms
- `CollocatedEquation{T}` — assembled linear system (A, b, source) → `to_linear_problem()` → `LinearProblem`
- `CollocatedScalarField{T}`, `CollocatedVectorField{Dim,T}`, `FaceFluxField{T}` — cell-centered fields
- `FVMSolverConfig` + `FieldSolverConfig` — per-field solver/preconditioner configuration
- 15 boundary condition types: `FixedVelocityBC`, `FixedPressureBC`, `NoSlipWallBC`, `SlipWallBC`, `InletOutletBC`, `ZeroGradientBC`, `TotalPressureBC`, `SymmetryBC`, `FlowRateInletBC`, `TimeDependentVelocityBC`, `WallFunctionBC`, `ConvectiveOutletBC`, `PressureInletVelocityBC`, `CyclicBC`, `CustomBC`

### Collocated Solver SciML Integration
- `CommonSolve.solve(prob, SIMPLE())` / `solve(prob, PISO(); tspan, dt)` — standard SciML dispatch
- `sol[:U]`, `sol[:p]`, `sol[:Ux]`, `sol[:Uy]`, `sol[:phi]` — symbolic field access
- `SciMLBase.remake(prob; nu=1e-4, density=2.0)` — immutable problem modification
- `SciMLStructures.Tunable` — parameter extraction `[nu, density, alpha_U, alpha_p, tolerance]` (the algorithm-dependent entries are filtered by an `applies` predicate, so `PISO` exposes only `[nu, density]`)
- Optional physics comes from `prob.model`, not keyword arguments (Stage 5d): build an `IncompressibleModel` and pass it to `IncompressibleProblem(...; model = ...)`, or override per-solve with `solve(prob, alg; model = ...)`. `solve` itself carries numerics only (`linear_solver`, `solver_config`, `scheme`, `blend`, `tspan`, `dt`, `save_every`, `U0`, `p0`, `verbose`)

## Known Issues

The authoritative issue list is `test/KNOWN_FAILURES.md`. High-level summary (fix waves v3.112/v3.114 are history; current `Project.toml` version is `4.0.0-DEV`):

### Fixed in v3.112 (solver-correctness wave)
Hyperbolic/core: HLLD uses signed `Bn` in star states and has a real `dir=3` branch; HLLC extended to 3D Euler; generic `ReflectiveBC` via the `normal_velocity_index(law, dir)` interface; the CFL callback actually enforces dt under `adaptive=false` (dtcache/dtmax, `u_modified!(false)`); `remake(ode_prob; u0/tspan/p)` is honored or throws — never silently dropped; MHD energy goes through the EOS interface (non-ideal-EOS consistent); CT caches reject `nghost>2` reconstructions and derive ghost offsets from the padded array (also fixed pre-existing `NoReconstruction` BoundsErrors); 1D RHS computes each face flux once and CFL/flux differencing handle nonuniform meshes; `srmhd_con2prim` non-convergence errors instead of silently proceeding.

Collocated: transient ddt assembles against a `state.U_old` time-level snapshot (PISO/PIMPLE are now consistent transient schemes); `SymmetryBC`/`SlipWallBC` project out the wall-normal velocity (no boundary mass leak); non-orthogonal explicit correction sign fixed and wired through momentum/pressure assembly; Rhie-Chow uses the same harmonic face `D_f` as the pressure operator; `H`/`A_P` extracted after the relaxed momentum solve with `∇p` removed from `H` (fixed a double pressure application that made reordered PISO blow up); kinematic-form consistency — solutions are density-invariant at fixed ν, buoyancy is per unit mass; the Durbin cap survives to the momentum-visible `ν_t`; `WallFunctionBC` is no-slip + Spalding wall `ν_t` (nonzero drag); time/space-varying velocity BCs are evaluated each step into the matrix (`ParabolicDirichletFunc`); MULES honors inflow alpha BCs; the pressure-reference fix is a symmetric elimination (matrix stays SPD for CG/AMG); the continuity residual is flux-normalized; turbulent SIMPLE/PISO/PIMPLE loops regained cyclic-BC handling; equations/sparsity are allocated once per solve with pattern-indexed writes in hot loops.

### Fixed in v3.114 (backlog wave)
- WENO5 1D ghost handling: the documented `nghost=3` bug no longer reproduces after the v3.112 ghost parameterization (96-config sweep in `test/hyperbolic/weno.jl`); the last 2-ghost hardcode (positivity limiter) now takes `ng`
- AMR 2D inter-block ghost exchange is real: same-level copy, coarse→fine prolongation, conservative fine→coarse averaging, and conservative seam-flux replacement at single-level jumps (same-level multi-block matches a single-block reference bitwise). 3D multi-block still throws
- GRMHD has a validated curved-spacetime path: densitized Valencia formulation with metric-aware con2prim, metric-aware source terms (two physics bugs fixed), Minkowski results bitwise unchanged, static Kerr-Schild atmosphere held with resolution-converging drift
- Pressure-based compressible SIMPLE/PIMPLE solve a real compressible pressure equation (see per-module status above)
- Cavitation/porous/MRF wired into the solver loops (see per-module status above)
- `fwh_farassat1a`: real retarded-time FW-H validated against analytic monopole/dipole (see per-module status above)
- Collocated SIMPLE Uy residual plateau RESOLVED by the v3.112 Rhie-Chow harmonic-`D_f` fix: 80×80 lid cavity now reaches Uy ≈ 5e-10 (was floored at ~3e-3); the binding residual is now the flux-normalized continuity (~2e-5) from lid-corner singularity cells
- VOF pressure correction: factor-ρ overcorrection fixed (`D` weighting was inconsistent between Laplacian and correction); VOF body force is now kinematic (`g + F_σ/ρ`)

### Still-open correctness items
- Vertex-centered FVM on unstructured meshes converges at ~O(h^1.5) in L∞, not O(h^2) (property of the scheme's boundary treatment; research item, not a bug)
- CyclicBC face matching converges slowly on coarse meshes (Stage 1a follow-up)
- IDDES uses `V_c^(1/Dim)` as a surrogate for `h_max`; full real-edge-length variant is a v3.2 follow-up
- AMR: 3D multi-block still unsupported (throws); ΔL≥2 seam fluxes uncorrected (warned); AMR domain BCs are zero-gradient only
- GRMHD curved path: HLL only, zero-gradient domain BCs, magnetized-curved cases validated for stability/div(B) only (scope stated by a one-time `@info`)
- Compressible pressure-based solvers are subsonic-only (no `div(phid,p)`); momentum ddt neglects ∂ρ/∂t
- PBM is 0-D and FSI has no solver adapters (deliberately deferred — do not wire without dedicated V&V)
- The published-benchmark harness lives in `test/benchmarks/` (NOT `validation/published_benchmarks/`, which does not exist). The CI `published-benchmarks` job in `.github/workflows/CI.yml` runs all 5 cases with `FVM_RUN_BENCHMARKS=true` and FAILS unless all 5 executed their physics assertions — `mark_deferred_compute` records `@test_broken`, so a deferral can never masquerade as a pass

### v3 fast-path modules: honest per-module status
The v3.102→v3.108 waves landed a large amount of code under `src/`. The per-module reality (audited v3.111) is more modest than the wave logs claimed. None of these are `stable`; most are thin kernels or scaffolds:

- **aeroacoustics** (`src/experimental/aeroacoustics/`) — `fwh_farassat1a` (v3.114) is a real retarded-time Farassat 1A implementation for static surfaces (time-series API; thickness ∂Uₙ/∂τ + loading far-field ∂Δp/∂τ/(cr) and near-field 1/r² terms; validated against analytic monopole/dipole to ≲0.1% amplitude). The legacy static-sum functions remain as documented near-field snapshot approximations. Lighthill quadrupole is still a stub; "PML" is a plain damping sponge zone
- **FSI** (`src/experimental/fsi/`) — generic Aitken-Δ² fixed-point accelerator over user-supplied callbacks; no adapters to the package's PISO or elasticity solvers exist; only exercised against a mock 1-DOF spring-damper. Interface transfer supports matching meshes only (1:1 copy)
- **adjoint** (`src/experimental/adjoint/`) — dense linear adjoint identities only (transposed solve for given A, b; checkpointed linear-transient variant). Not wired into SIMPLE/PIMPLE; no SciMLSensitivity integration (it is not a dependency)
- **solid_mechanics** (`src/experimental/solid_mechanics/`) — decoupled per-component Poisson solve by default (not full coupled elasticity); `traction_bcs` are unused by the solvers and now throw if supplied
- **mesh_generation** (`src/experimental/mesh_generation/`) — octree castellated refinement + STL snap prototype; there is NO octree → `UnstructuredFVMMesh` extraction, so it cannot produce a solver-usable mesh; no layer addition
- **MPI** (`src/experimental/parallel/`, `ext/FVMMPIExt/`) — per-rank local solves with halo exchange between outer iterations (additive-Schwarz-style); no distributed matrix is ever constructed (the `PSparseMatrix` claims were wrong — a PartitionedArrays row partition is carried as metadata only)
- **pressure-based compressible SIMPLE/PIMPLE** (`src/experimental/pressure_based/`) — real subsonic compressible pressure equation as of v3.114 (ρ_f mass fluxes, implicit ψ = ∂ρ/∂p diagonal, (1/ρ)∇p momentum): closed-box mass conserved to machine precision, low-Mach limit matches the incompressible solver, finite acoustic propagation verified. Subsonic only (no `div(phid,p)` shock treatment); momentum ddt neglects ∂ρ/∂t
- **population_balance** (`src/experimental/population_balance/`) — 0-D moment/class kernel library (QMoM/DQMoM/Class Method); not coupled to transport
- **cavitation / porous / MRF** (`src/cavitation/`, `src/porous/`, `src/mrf/`) — wired into the solver loops as of v3.114: `solve_vof(...; cavitation_model)` (Patankar-implicit α source + implicit pressure dilatation), `porous_zones = [...]` (implicit Darcy-Forchheimer diagonal, verified vs analytic Δp), `mrf_zones = [...]` (absolute-velocity formulation with makeRelative/makeAbsolute zone-face flux conversion, verified vs solid-body rotation). Porous/MRF kwargs throw on the combustion/radiation solve paths (not yet threaded there); MRF interfaces must be surfaces of revolution

### Structural items already addressed
The following structural items previously listed here have been resolved during the v2→v3 overhaul. See `test/KNOWN_FAILURES.md` for fix-stage details and assertions:
- `CollocatedEquation` random-pattern CSC insertion (Stage 1a — `SparsityPattern` pre-computes `nzval` indices, O(1) writes)
- Operator hot-loop allocations (Stage 1b — `Dict{Int,Int}` replaced with `Vector{Int}`; audit confirmed gradients/interpolation are zero-alloc on the hot path)
- `BlockCollocatedEquation` for vector-valued / two-fluid systems (Stage 1c, in production use by v3.107 two-fluid solver)
- `SciMLStructures.Tunable` named registry (Stage 1e — `register_tunable!` + `tunable_schema`, no longer hardcoded length-5)
- MPI full-mesh-per-rank (Stage 2 + Wave 5 — `LocalFVMMesh`, RCB + Metis partitioners; solves remain per-rank local, see honest status above)
- OpenFOAM binary polyMesh reader landed alongside Gmsh v4 reader

### Validation status (v3.112)
- `incompressible_ns`, `turbulence_rans`, `conjugate_heat_transfer`, and `turbulence_les` are back at `provisional` (re-promoted in v4.0.0-DEV) with machine-linked `[[scientific_evidence]]` entries under `docs/src/literate_verification/` executed by `validation/evidence_runner.jl`: Poiseuille O(h²) convergence + Ghia 1982 cavity; DHIT exact-ODE decay + log-layer P_k = ε equilibrium; Laplace-series conduction + separable unsteady-heat decay; Smagorinsky shear algebra + WALE operator invariants. The remaining 9 collocated features are still `experimental` (demoted from `provisional` in v3.112): their "Evidence #N" items are real `test/` files run by `Pkg.test()`, but none are machine-linked `[[scientific_evidence]]` entries, so the governance ladder gate — enforced for `provisional` as well as `stable` in `test/governance/repository_governance.jl` — is not satisfied. Re-promotion follows the same evidence-script pattern
- Published-benchmark execution (≥3 per feature) remains the explicit gate for `stable` promotion; the 5-case suite now runs in CI (`published-benchmarks` job)
- Outstanding deferrals to v3.2 / v3.3: layer addition + mesh extraction in the octree mesher, Enzyme full-solver AD, IDDES `h_max` from real edge lengths, Sandia Flame D combustion benchmark, all `stable`-tier promotions
- `test/KNOWN_FAILURES.md` is the authoritative per-item status list (the historical `plans/` and `specs/` dev-planning directories were removed in v3.114 — see git history if you need the original phase plans)
