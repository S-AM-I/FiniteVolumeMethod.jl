# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FiniteVolumeMethod.jl is a Julia package for solving PDEs with two solver families:
- **Parabolic/elliptic**: Cell-vertex solver on unstructured triangular meshes (via DelaunayTriangulation.jl)
- **Hyperbolic**: Cell-centered solver on structured 1D/2D/3D meshes (Euler, MHD, Navier-Stokes, GRMHD, etc.)

The package is undergoing a v2 transition. The capability matrix (`docs/src/capability_matrix.md`) and validation manifest (`validation/manifest.toml`) are the authoritative contracts for feature maturity. Only `stable` features are publication-grade.

## Common Commands

### Running Tests
```bash
# Full test suite (slow — runs tutorials, verification, and governance checks)
julia --project -e 'using Pkg; Pkg.test()'

# Single test file
julia --project=test test/<filename>.jl

# Single test file via Docker
TEST_FILE=test/geometry.jl make ci-test-file
```

### Formatting (Runic)
All `.jl` files must pass Runic formatting. A pre-commit hook enforces this.
```bash
# Check formatting
julia --project -e 'using Runic; Runic.main(["--check", "."])'

# Auto-fix formatting
julia --project -e 'using Runic; Runic.main(["--inplace", "."])'
```
Runic is in the global Julia environment, not the project env. Key rules: spaces around `=` in kwargs (`atol = 0.01`), 4-space continuation indent (not aligned to `(`), spaces around `/` in arithmetic.

### Local CI (Docker-based)
GitHub Actions CI is enabled (`.github/workflows/CI.yml`) with four lanes: environment-integrity, unit-interop, scientific-smoke, and docs. For local iteration, use the Makefile Docker lanes:
```bash
make ci-fast              # Fast API/interop lane
make ci-smoke             # Scientific smoke tests
make ci-full-evidence     # Full scientific evidence
make ci-performance       # Performance baselines
make ci-release-audit     # Release audit lane
make ci-format            # Format check only
make ci-all               # All lanes
make ci-repl              # Interactive Julia REPL in container
```
First run requires `make ci-build` (downloads + precompiles all deps).

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

### Layered Include System
The main module (`src/FiniteVolumeMethod.jl`) loads code through four layer files in `src/layers/`. The ordering is a strict dependency chain — never import backwards across layers.

1. **`domain_problem_definitions.jl`** — Foundational types, mesh definitions (parabolic and structured), coordinate systems, geometry, conditions, problem types
2. **`discretization_assembly_kernels.jl`** — FVM equation assembly, reconstruction schemes, all hyperbolic solvers, AMR, WENO/PPM/IMEX, coupling infrastructure. New physics goes here.
3. **`sciml_adapters_and_accessors.jl`** — SciML integration: cache types, state mapping (fold/unfold), CFL callbacks, ODE/SplitODE construction, solution accessors, `remake`
4. **`extensions_tooling_output.jl`** — Dashboard types, I/O (VTK, HDF5, CSV), diagnostics, checkpointing, capability matrix

### Key Source Directories
- `src/parabolic/` — Cell-vertex solver: types, mesh variants, assembly (`assembly/` has dimension-specific files: `assembly_1d.jl`, `assembly_2d.jl`, `assembly_unstructured.jl`, `assembly_curvilinear.jl`, etc.), boundary conditions, gradients, limiters, turbulence models
- `src/hyperbolic/` — Cell-centered solver: conservation laws, Riemann solvers (HLL/HLLC/HLLD), reconstruction (MUSCL/PPM/WENO), plus advanced physics (Navier-Stokes, MHD variants, GRMHD, IMEX)
- `src/core/` — SciML bridge: semidiscrete caches, state mapping (fold/unfold between flat ODE vectors and padded arrays), CFL callback, ODE problem construction, backend abstraction (`backends.jl`)
- `src/amr/` — Block-structured AMR with prolongation, restriction, flux correction, subcycling
- `src/constrained_transport/` — Divergence-free magnetic field evolution (div-B preservation for 2D and 3D)
- `src/coupling/` — Multi-physics operator splitting (Lie-Trotter, Strang)

### Key Design Patterns

**Ghost-cell state mapping**: The hyperbolic solver maintains two representations of the solution. The ODE integrator sees a flat `Vector{SVector{N,FT}}` of interior cells only. Inside the RHS function, `unfold_to_padded!()` copies this into `cache.padded_U` (an array with ghost cells for boundary stencils), the RHS is computed on the padded array, and `fold_from_padded!()` copies the result back. This pattern enables allocation-free time-stepping and appears identically for 1D/2D/3D/AMR.

**Cache-as-parameter**: All hyperbolic solvers use pre-allocated `AbstractSemidiscreteCache` subtypes (`HyperbolicCache1D`, `HyperbolicCache2D`, `MHDCTCache2D`, `AMRCache`, etc.) that hold padded arrays, flux buffers, the problem object, and grid metadata. The cache is passed as the ODE parameter `p` so the RHS function never allocates.

**Conservation law interface**: The hyperbolic solver is built on `AbstractConservationLaw{Dim}`. To add new physics, subtype it and implement: `nvariables(law)`, `physical_flux(law, u, dir)`, `max_wave_speed(law, u, dir)`, `conserved_to_primitive(law, u)`, `primitive_to_conserved(law, w)`. Existing laws: `EulerEquations`, `IdealMHDEquations`, `NavierStokesEquations`, `GRMHDEquations`, `ShallowWaterEquations`, `ReactiveEulerEquations`, etc.

**Parabolic assembly**: The parabolic solver assembles `M du/dt + A u = b` matrices (dimension-specific files in `src/parabolic/assembly/`), then converts to `ODEProblem`/`LinearProblem` via Layer 3 helpers. Problem types are `FVMProblem` (single-field), `FVMSystem` (multi-field coupled), and `SteadyFVMProblem` (steady-state wrapper).

### Package Extensions
Defined in `Project.toml` under `[extensions]`:
- `FVMCUDAExt` (CUDA) — GPU backend (currently only 2D Euler; most solvers are CPU-only)
- `FVMVTKExt` (WriteVTK) — VTK output
- `FVMHdf5Ext` (HDF5) — HDF5 I/O
- `FVMCheckpointExt` (JLD2) — Checkpointing
- `FVMDashboardExt` / `FVMDashboardServerExt` (JSON3, HTTP) — Live dashboard

### Test Organization
`test/runtests.jl` orchestrates all tests via `safe_include()`, which runs each test file in its own anonymous module to prevent namespace pollution between tests. The test suite includes:

- **Unit tests** — `test/geometry.jl`, `test/conditions.jl`, `test/hyperbolic.jl`, `test/mhd.jl`, etc.
- **Tutorials as tests** — Literate.jl scripts from `docs/src/literate_tutorials/` and `docs/src/literate_wyos/` are executed as testsets (docs are tested code)
- **Verification cases** — driven by `validation/manifest.toml` via the `RepoValidationManifest` module; scripts from `docs/src/literate_verification/`
- **Governance** — Aqua.jl quality, environment integrity, repository governance, reproducibility bundles, quality ledger

Note: `keller_segel_chemotaxis.jl` is explicitly excluded from the tutorials testset.

### Validation Infrastructure
- `validation/manifest.toml` — Machine-readable source of truth for feature maturity, V&V status, and CI inclusion. Features are `stable`, `experimental`, or `deprecated`.
- `validation/manifest.jl` — Julia module (`RepoValidationManifest`) that parses the manifest; used by both tests and docs builds
- `test/KNOWN_FAILURES.md` — Documents known broken/skipped/demoted tests

## Known Issues
- WENO5 has a ghost cell bug in the 1D solver (`nghost=3` not supported at small grid sizes)
- Vertex-centered FVM on unstructured meshes converges at ~O(h^1.5) in L-inf norm, not O(h^2)
