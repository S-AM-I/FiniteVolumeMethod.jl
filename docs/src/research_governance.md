# Verification and Validation

FiniteVolumeMethod.jl distinguishes between solver capabilities that have been
numerically verified and validated, and supporting infrastructure (I/O,
dashboards, checkpointing) that facilitates research workflows but does not
itself constitute solver verification.

## Solver Maturity Levels

Each solver capability is assigned a maturity level reflecting its
verification and validation (V&V) status:

- **Stable** — the solver has passed code verification (convergence studies
  and/or manufactured solutions), has been validated against published
  benchmarks, and is suitable for publication-grade results.
- **Provisional** — automated verification exists and results are promising,
  but the V&V coverage is incomplete. Suitable for internal research and
  method development.
- **Experimental** — early-stage or opt-in work without formal V&V. Not
  suitable for drawing scientific conclusions.

See the [Capability Matrix](capability_matrix.md) for the current status of
each solver family.

## V&V Methodology

The package uses a three-level verification and validation approach:

1. **Code verification** — convergence studies on problems with known exact
   solutions or manufactured solutions (MMS). Confirms that the discrete
   solution converges at the expected order of accuracy.
2. **Solution verification** — grid-refinement studies on problems without
   analytical solutions, using Richardson extrapolation or GCI to estimate
   discretization error.
3. **Validation** — comparison against published experimental data or
   established computational benchmarks from the literature.

Verification cases and their expected convergence rates are recorded in the
validation manifest (`validation/manifest.toml`), which serves as the
machine-readable source of truth.

## Reproducibility

- `julia --project=. scripts/verification_validation_report.jl` regenerates
  the V&V report with executed summaries in `validation/reports/`.
- `julia --project=. scripts/build_reproduction_bundles.jl` creates
  per-feature benchmark archives in `validation/reproduction_bundles/`.
- `julia --project=. scripts/build_release_outputs.jl --stable-only` produces
  a release archive with V&V reports, benchmark data, and reproducibility
  metadata (git commit, Julia version, dependency versions).

## Running Tests

All tests run locally via Docker (GitHub Actions are disabled during the v2
development cycle):

| Command                 | What it runs                                        |
|-------------------------|-----------------------------------------------------|
| `make ci-fast`          | API tests, SciML integration, mesh operations       |
| `make ci-smoke`         | One verification case per stable solver family      |
| `make ci-full-evidence` | Full V&V suite (all convergence studies)             |
| `make ci-performance`   | Performance baselines for stable solver families    |
| `make ci-release-audit` | Release gate: V&V + performance + reproducibility   |

## Support Policy

- Supported Julia versions: current stable release and current LTS.
- CPU `Float64` is the reference baseline for all V&V results.
- GPU (CUDA) execution is an extension that must demonstrate parity against
  the CPU baseline before it inherits the same validation status.
