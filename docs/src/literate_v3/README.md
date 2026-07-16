# v3 Collocated Stack — Runnable Tutorials

This directory contains twelve worked examples that exercise each major
solver family in the v3 OpenFOAM-style collocated stack. Each tutorial
is self-contained, uses a small mesh so it completes in under 30 s on a
laptop, and prints a sanity-check value at the end. They are written
as Literate.jl-style `.jl` files, so the same file doubles as a
`julia` script and as the source for a generated Markdown page.

## Running a single tutorial

```bash
julia --project=docs docs/src/literate_v3/01_lid_driven_cavity.jl
```

All twelve tutorials use the same top-level imports
(`FiniteVolumeMethod`, `LinearSolve`, `StaticArrays`, `Printf`) and
pull the mesh builder `build_cartesian_unstructured_mesh` from
`test/TestHelpers.jl`.

## Running them all

```bash
for f in docs/src/literate_v3/*.jl; do
    echo "=== $f ==="
    julia --project=docs "$f"
done
```

On a first run, expect an additional ~45 s for `FiniteVolumeMethod` to
precompile. Subsequent runs are bounded by the per-tutorial budget
below.

## Tutorial index

| # | File | Feature area | Runtime | Status |
|---|------|-------------|---------|--------|
| 01 | `01_lid_driven_cavity.jl` | Incompressible SIMPLE (Phase 1) | ~3 s | runs clean |
| 02 | `02_compressible_channel.jl` | CompressibleSIMPLE (Stage 3) | ~3 s | runs clean |
| 03 | `03_kepsilon_channel.jl` | k-ε RANS (Phase 2a) | ~3 s | runs clean (see note) |
| 04 | `04_rayleigh_benard.jl` | Boussinesq thermal (Phase 3) | ~3 s | runs clean (see note) |
| 05 | `05_dam_break.jl` | VOF multiphase (Phase 7) | ~3 s | runs clean |
| 06 | `06_combustion_one_step.jl` | EDM reacting flow (Phase 8) | ~3 s | runs clean |
| 07 | `07_radiation_p1.jl` | P1 radiation (Phase 9) | ~2 s | runs clean |
| 08 | `08_dpm_stokes.jl` | Lagrangian DPM (Phase 11) | ~1 s | runs clean |
| 09 | `09_dynamic_mesh_oscillator.jl` | ALE dynamic mesh (Phase 10) | ~3 s | runs clean |
| 10 | `10_two_fluid_bubble_column.jl` | Eulerian two-fluid (Phase 7) | ~3 s | runs clean |
| 11 | `11_solid_mechanics_beam.jl` | Linear elasticity (Stage 7a) | ~2 s | runs clean |
| 12 | `12_aeroacoustics_fwh.jl` | FW-H surface integration (Stage 6f) | < 1 s | runs clean |

Runtimes exclude Julia startup + precompile (~10 s after a cold start).

## Known caveats flagged in the tutorials

Several tutorials include a `# KNOWN ISSUE:` comment block where the
underlying feature is deliberately simplified:

- `03_kepsilon_channel.jl` — k-ε ν_t floor via `max()` without Durbin
  realizability; the tutorial therefore reports centreline k, ε and
  ν_t/ν but does not assert log-law agreement.
- `04_rayleigh_benard.jl` — the solver diverges for Ra ≳ 10³ on a
  12×12 mesh with default under-relaxation, so the tutorial runs at
  Ra ≈ 40 (subcritical) and prints a coarse Nusselt estimate rather
  than the published-benchmark Ra = 10⁴ value.
- `05_dam_break.jl` — α transport uses hard clipping, not MULES or
  isoAdvector; front position is sensitive to the short time span.
- `06_combustion_one_step.jl` — EDM is one-step only with a Lewis-
  unity implicit assumption.
- `09_dynamic_mesh_oscillator.jl` — only prescribed motion is
  implemented; a full 6-DOF rigid-body coupling is future work.
- `10_two_fluid_bubble_column.jl` — upwind + clip α transport, no
  non-orthogonal correction.

Each of these matches a line item in the root `CLAUDE.md` "Known
Issues" section or `test/KNOWN_FAILURES.md`.

## Relationship to the test / V&V suites

Every tutorial carries a footer comment pointing at:

- the `validation/manifest.toml` feature entry (e.g.
  `phase1.simple_collocated`), which records `stable` /
  `experimental` / `smoke_tested` status; and
- the corresponding tests under `test/` (e.g. `test/incompressible.jl`
  and `test/incompressible_sciml.jl`).

The tutorials are deliberately a subset of the V&V cases — they trade
publishable benchmark agreement for a <30 s per-tutorial budget on
small meshes. Treat them as API onboarding, not as validation.
