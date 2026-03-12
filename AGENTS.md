# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the package entrypoint `FiniteVolumeMethod.jl` and feature areas such as `hyperbolic/`, `parabolic/`, `mesh/`, `io/`, `amr/`, and `specific_problems/`. Keep new code close to the existing solver family it extends. `test/` mirrors that structure with focused files like `hyperbolic.jl`, `mhd_2d.jl`, and `io.jl`, all orchestrated from `test/runtests.jl`. `docs/` holds the Documenter build (`make.jl`) plus literate tutorials and verification cases. Docker and local CI helpers live in `docker/`, `docker-compose.yml`, and `Makefile`.

## Build, Test, and Development Commands
For a direct Julia workflow:

- `julia --project -e 'using Pkg; Pkg.instantiate()'` installs dependencies.
- `julia --project -e 'using Pkg; Pkg.test()'` runs the full test suite.
- `julia --project=docs docs/make.jl` builds documentation with examples.

For the repo’s CI-like local workflow:

- `make ci-build` builds the Docker base image after dependency changes.
- `make ci-test` runs the full suite in the container.
- `TEST_FILE=test/geometry.jl make ci-test-file` runs one targeted test file.
- `make ci-format` checks formatting; `make ci-format-fix` applies it.
- `make ci-docs-ci` runs the faster docs build used in CI.

## Coding Style & Naming Conventions
This is a Julia package; follow the existing style and let Runic be the formatter of record. Use 4-space indentation, keep filenames lowercase with underscores (`structured_mesh_3d.jl`), and use descriptive type/module names in `CamelCase` (`FVMProblem`, `BoundaryConditions`). Prefer small, domain-focused methods and place exports/includes in the relevant solver area rather than creating catch-all files.

## Testing Guidelines
Add or update tests in the matching file under `test/`; if a feature spans subsystems, extend `test/runtests.jl` intentionally rather than creating hidden coverage gaps. The suite uses `Test` plus `Aqua` checks, and it executes docs-backed tutorial and verification scripts, so avoid brittle examples. Run `julia --project -e 'using Pkg; Pkg.test()'` before opening a PR; use `make ci-test` when you need parity with GitHub Actions.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `Fix docs build and Aqua formatting failures` and `Add parabolic solver, engine, and I/O modules...`. Keep commit titles concise, sentence-style, and action-led. PRs should explain the numerical or API change, note affected areas (`src/hyperbolic`, `docs/`, `test/`), and mention any follow-up work. Include plots or screenshots only when a dashboard or docs output changes.
