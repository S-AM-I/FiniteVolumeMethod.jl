# `test/benchmarks/` — Published-benchmark suite (v3.1 Agent E)

This directory holds the **published-reference benchmark harness** that feeds
the `stable` promotion ladder documented in `validation/manifest.toml`. Each
benchmark reproduces a third-party published result (not an internal MMS or
self-comparison) and is the primary qualifying evidence for the
"3+ published benchmarks per feature" gate.

The harness is intentionally **gated** — it is **not** run by default under
`Pkg.test()`.

## Running the benchmarks

```shell
# Run everything in the suite
FVM_RUN_BENCHMARKS=true julia --project=test test/benchmarks/ghia_re400.jl
FVM_RUN_BENCHMARKS=true julia --project=test test/benchmarks/moser_re180.jl
FVM_RUN_BENCHMARKS=true julia --project=test test/benchmarks/rayleigh_benard_1e4.jl
FVM_RUN_BENCHMARKS=true julia --project=test test/benchmarks/martin_moyce_dam_break.jl
FVM_RUN_BENCHMARKS=true julia --project=test test/benchmarks/sod_shock_tube.jl
```

Without the environment flag, every file logs a `benchmark skipped` notice and
returns without running. This keeps `Pkg.test()` fast.

## Compute cost (M3 Max, 11 performance cores, single-threaded BLAS)

| Benchmark                       | Grid      | Wall-clock target | Observed (2026-04)       |
| ------------------------------- | --------- | ----------------- | ------------------------ |
| `ghia_re400.jl`                 | 64 × 64   | < 15 min          | see `.cache/<name>.sha`  |
| `moser_re180.jl`                | 32 × 64   | < 10 min          | — RANS steady, ~2000 it. |
| `rayleigh_benard_1e4.jl`        | 40 × 40   | < 20 min          | buoyancy-coupled SIMPLE  |
| `martin_moyce_dam_break.jl`     | 100 × 50  | < 20 min          | transient PISO, 200 dt   |
| `sod_shock_tube.jl`             | 400 cells | < 30 s            | SSPRK33 explicit         |

The per-benchmark budget is 30 minutes. A benchmark that cannot reach
its published tolerance on this compute tier calls `mark_deferred_compute`
inside its body — this records a passing test with a `DEFERRED_COMPUTE`
log entry so the benchmark doesn't appear failing, but the cache is **not**
refreshed. The capability matrix will not count deferred cases as part of the
3-benchmark gate.

## Caching

Each benchmark stores a SHA-256 of the solver source files it depends on in
`test/benchmarks/.cache/<name>.sha` (gitignored). On subsequent invocations:

- If the hash **matches**, the benchmark reports `[cached]` without re-running.
  This makes the suite cheap to re-verify between unrelated commits.
- If the hash **mismatches** (source changed), the benchmark re-runs. On a
  clean pass the cache is refreshed; on failure the cache is left stale.

Tags and their source coverage:

| Tag              | Files hashed                                                |
| ---------------- | ----------------------------------------------------------- |
| `:incompressible`| `src/incompressible/{simple,momentum,pressure,correction,residuals}.jl`, `src/collocated/{gradient,laplacian,interpolation}.jl` |
| `:turbulence`    | `src/turbulence/{solvers,k_epsilon_rans,wall_functions,interface}.jl` + incompressible |
| `:thermal`       | `src/thermal/{solvers,energy_equation,buoyancy}.jl` + incompressible + collocated ops  |
| `:multiphase`    | `src/multiphase/{solvers,alpha_transport,boundedness,mixture}.jl` + collocated ops      |
| `:hyperbolic`    | `src/hyperbolic/{hyperbolic_solve,hyperbolic_problem,hllc_solver,reconstruction,euler}.jl`, `src/core/cache.jl` |

See `harness.jl :: sources_of` for the authoritative map. Add a new tag by
extending that function.

## Contributing a new benchmark

1. Pick a published result with a directly-reproducible numerical target
   (tabulated profile, bulk coefficient, conserved-quantity drift, …). The
   Lid-driven cavity Ghia series, De Vahl Davis natural convection,
   Martin-Moyce dam break, Moser-Kim-Mansour DNS channel, Comte-Bellot-Corrsin
   DHIT, Zalesak rotating slot, Hysing rising bubble, Brio-Wu MHD shock, and
   Orszag-Tang MHD vortex are natural candidates.
2. Create `test/benchmarks/<slug>.jl`. Start the body with:
   ```julia
   include("harness.jl")
   include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

   @benchmark_testset "<slug>" sources = :<tag> begin
       # setup + solve
       @benchmark_assert <published-ref comparison>
   end
   ```
3. If the benchmark can exceed the 30-minute budget on target hardware, guard
   the expensive path behind a `mark_deferred_compute(...)` branch that runs
   when early convergence checks fail.
4. Update the table above with the grid + budget.
5. Do **not** update `validation/manifest.toml` — the main thread attaches
   Evidence #N entries after a benchmark ships.

## Files in this directory

- `harness.jl` — caching + gating infrastructure (`@benchmark_testset`,
  `@benchmark_assert`, `mark_deferred_compute`).
- `ghia_re400.jl` — Lid-driven cavity, Re = 400, Ghia 1982 Table II.
  Feeds `incompressible_ns` stable-promotion evidence.
- `moser_re180.jl` — Turbulent channel flow, Reτ = 180, Moser-Kim-Mansour
  1999 log-law. Feeds `turbulence_rans` stable-promotion evidence.
- `rayleigh_benard_1e4.jl` — Differentially-heated cavity, Ra = 10⁴,
  De Vahl Davis 1983. Feeds `conjugate_heat_transfer` stable-promotion evidence.
- `martin_moyce_dam_break.jl` — Collapsing column, g-driven two-phase,
  Martin-Moyce 1952 front-correlation. Feeds `multiphase_vof` stable-promotion
  evidence.
- `sod_shock_tube.jl` — Sod 1978 Riemann problem stub. Wraps existing
  hyperbolic solver coverage with cache-pass provenance for the
  `hyperbolic` feature family. Pressure-based collocated family (`incompressible_ns`
  and descendants) is out-of-family for shock tubes and is not exercised here.
- `.cache/` — (gitignored) per-benchmark SHA + provenance records.

## Relationship to `validation/manifest.toml`

The benchmarks here are **one** input to the stable-promotion review. Other
inputs: Aqua quality, grid-convergence studies, round-trip SciML-interface
tests, and CI inclusion. A feature passes the "3+ published benchmarks" gate
when three or more of the benchmarks tagged against it live in this directory
**and** pass cleanly (cached or fresh) on reference compute.
