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
GitHub Actions are disabled during the v2 overhaul. Use the Makefile lanes:
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
Docs use Literate.jl — tutorial source files live in `docs/src/literate_tutorials/`, `docs/src/literate_hyperbolic/`, `docs/src/literate_wyos/`, and `docs/src/literate_verification/`.

## Architecture

### Layered Include System
The main module (`src/FiniteVolumeMethod.jl`) loads code through four layer files in `src/layers/`:

1. **`domain_problem_definitions.jl`** — Foundational types, mesh definitions (parabolic and structured), coordinate systems, geometry, conditions, problem types
2. **`discretization_assembly_kernels.jl`** — FVM equation assembly, reconstruction schemes, all hyperbolic solvers (Euler, MHD, GRMHD, SRMHD, etc.), AMR, WENO/PPM/IMEX, coupling infrastructure
3. **`sciml_adapters_and_accessors.jl`** — SciML integration: cache types, state mapping, CFL callbacks, ODE/SplitODE construction, solution accessors, `remake`
4. **`extensions_tooling_output.jl`** — Dashboard types, I/O (VTK, HDF5, CSV), diagnostics, checkpointing, capability matrix

### Key Source Directories
- `src/parabolic/` — Cell-vertex parabolic solver: types, mesh (structured/curvilinear/unstructured), assembly (`assembly/`), boundary conditions, gradients, limiters, turbulence models
- `src/hyperbolic/` — Cell-centered hyperbolic solver: conservation laws, Riemann solvers (HLL/HLLC/HLLD), reconstruction (MUSCL/PPM/WENO), Navier-Stokes viscous fluxes, MHD variants, GRMHD, IMEX
- `src/core/` — SciML bridge: semidiscrete caches, state mapping (fold/unfold between flat ODE vectors and padded arrays), CFL callback, ODE problem construction
- `src/constrained_transport/` — Divergence-free magnetic field evolution (2D and 3D)
- `src/amr/` — Block-structured AMR with prolongation, restriction, flux correction, subcycling
- `src/mesh/` — Abstract mesh interface and structured mesh implementations for the hyperbolic solver
- `src/eos/` — Equations of state (ideal gas, stiffened gas)
- `src/metric/` — Spacetime metrics for GRMHD (Minkowski, Schwarzschild, Kerr)
- `src/coupling/` — Multi-physics operator splitting (Lie-Trotter, Strang)
- `src/io/` — Output management, VTK writers, diagnostics, checkpointing
- `src/schemes/` — Shared limiters and gradient reconstruction used by both solver families

### Package Extensions
Defined in `Project.toml` under `[extensions]`:
- `FVMCUDAExt` (CUDA) — GPU backend
- `FVMVTKExt` (WriteVTK) — VTK output
- `FVMHdf5Ext` (HDF5) — HDF5 I/O
- `FVMCheckpointExt` (JLD2) — Checkpointing
- `FVMDashboardExt` / `FVMDashboardServerExt` (JSON3, HTTP) — Live dashboard

### Test Organization
`test/runtests.jl` orchestrates all tests via `safe_include()`. Tests span unit tests, tutorial execution (Literate.jl scripts from `docs/src/`), verification cases (driven by `validation/manifest.toml`), Aqua.jl quality checks, and governance/reproducibility audits. The `Tutorials`, `Custom Templates`, and `Verification` testsets execute the actual documentation scripts.

### Validation Infrastructure
- `validation/manifest.toml` — Machine-readable source of truth for feature maturity, V&V status, and CI inclusion
- `validation/manifest.jl` — Julia module (`RepoValidationManifest`) that parses the manifest
- `test/KNOWN_FAILURES.md` — Documents known broken/skipped/demoted tests

## Known Issues
- WENO5 has a ghost cell bug in the 1D solver (`nghost=3` not supported at small grid sizes)
- Vertex-centered FVM on unstructured meshes converges at ~O(h^1.5) in L-inf norm, not O(h^2)
- `Aqua.test_unbound_args` is marked `broken = true` due to `Val{N}` false positive in AMR constructors
