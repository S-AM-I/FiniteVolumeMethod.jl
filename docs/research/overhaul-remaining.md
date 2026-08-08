---
tags: [repo/FiniteVolumeMethod.jl, roadmap]
---

# Remaining Overhaul Tasks

Status snapshot (2026-08-07): the v2 structural overhaul (Stages 0–9), the v3
research-grade cycle (v3.0 → v3.114), and the v4 structural rework
(Stages 3–7b: submodules, curated exports, SciML contract alignment, docs and
test restructure) are complete on `main`. All 13 collocated features are back
at `provisional` with machine-linked evidence. This document records what
remains, so it can be picked up deliberately rather than rediscovered.

Authoritative companions: `test/KNOWN_FAILURES.md` (per-item status),
`validation/manifest.toml` (maturity contract), `docs/research/vv-plan.md`
(V&V strategy), CLAUDE.md "Known Issues" (summary).

Closed in the 2026-08-07 closeout session (removed from the lists below):

- Aqua `test_unbound_args` re-enabled — the `FixedVelocityBC` /
  `FlowRateInletBC` tuple constructors now bind all type parameters; the
  quality-ledger exception is retired.
- `shallow_water_dam_break.jl` mass assertion replaced by an efflux-aware
  check (no-mass-creation + regression band around the measured 2.08%
  open-boundary loss).
- LinearSolve macOS default-solver bug: correctly attributed — fixed by
  LinearSolve **5.5.0**, not 5.3.0; the test env now floors at 5.5.
- Clean full-suite baseline rerun (in progress at the time of writing; result
  recorded in `test/KNOWN_FAILURES.md`).

## Track 1 — `stable`-tier promotion programme (the main outstanding work)

Gate: **≥3 published benchmarks per feature executing in CI** (enforced by
`test/governance/repository_governance.jl`; the 5-case suite in
`test/benchmarks/` is the current ceiling).

1. Extend the published-benchmark suite beyond the current five cases
   (Sod, Moser Re180, Martin–Moyce, Ghia Re400, Rayleigh–Bénard 1e4) until
   each feature slated for `stable` has ≥3.
2. Sandia Flame D for `combustion` promotion: EDC + variable-Lewis +
   radiation-coupled vs published Raman/Rayleigh data.
3. Relativistic promotion: a genuine Bondi solution case and full SRMHD
   eigenmode convergence (the existing Bondi / Fishbone–Moncrief scripts are
   demoted — approximate initial conditions, not equilibrium solutions).
4. Wallclock budget: the full benchmark suite is expected to run quarterly at
   the terminal until CI wallclock is reduced.

## Track 2 — Deferred features (v3.2 / v3.3 roadmap)

1. Octree mesher: near-wall layer addition, and octree →
   `UnstructuredFVMMesh` extraction (`extract_unstructured_mesh` does not
   exist; the mesher cannot yet emit a solver-usable mesh).
2. Enzyme full-solver AD (`ext/FVMEnzymeExt/` is a stub); wiring the adjoint
   into SIMPLE/PIMPLE; SciMLSensitivity integration (currently the adjoint is
   a dense linear-identity library only).
3. IDDES `h_max` from real edge lengths (currently `V_c^(1/Dim)` surrogate;
   needs a per-cell edge-length cache in `UnstructuredFVMMesh`).
4. Two-fluid energy + species cross-coupling on the
   `BlockCollocatedEquation` momentum block.
5. FW-H porous surfaces and supersonic emission corrections; Lighthill
   quadrupole is a stub.
6. Dynamic mesh: 6-DOF rigid-body motion; run-time topology changes.
7. FSI solver adapters and PBM–transport coupling (deliberately deferred —
   do not wire without dedicated V&V).

## Track 3 — Open correctness items

1. Vertex-centred FVM on unstructured meshes converges at ~O(h^1.5) in L∞,
   not O(h²) (boundary treatment; research item, not a bug).
2. CyclicBC: slow face matching on coarse meshes; cyclic pairs not yet in
   `build_collocated_sparsity` (Stage 1a follow-up — cyclic assembly still
   takes the slow path).
3. AMR: 3D multi-block throws; ΔL≥2 seam fluxes uncorrected (warned);
   AMR domain BCs are zero-gradient only.
4. GRMHD curved path: HLL only, zero-gradient domain BCs,
   magnetised-curved cases validated for stability/div(B) only.
5. Compressible pressure-based solvers: subsonic only (no `div(phid,p)`);
   momentum ddt neglects ∂ρ/∂t.
6. Collocated Laplacian: no face-skewness correction term.
7. Turbulence: no low-Re damping for k-ε/k-ω (Launder–Sharma, Abid);
   k-ω SST blending is simplified scalar, not full F1/F2.
8. MPI: distributed `PSparseMatrix` assembly + parallel AMG pressure
   preconditioning (Stage 2 follow-up). Current parity floor is the
   one-cell-overlap Schwarz transmission error (~1e-4 at 2–4 ranks).
9. MHD positivity floor: cells floored to `ρ = ε` keep their momentum, so
   KE ~ |m|²/(2ε) and the recovered pressure carries O(eps(KE)) ≈ 1e-4
   absolute cancellation noise. A velocity-capping vacuum treatment
   (Athena-style) would make floored states round-trip exactly.

## Track 4 — V&V plan gaps (`docs/research/vv-plan.md`)

1. `tgv_kinetic_energy_decay.jl` (Phase 2B) — KE decay tracking vs
   `KE_0 exp(-4νk²t)`.
2. `srmhd_eigenmode_convergence.jl` (Phase 2C) — all four SRMHD wave
   families.
3. Reclaiming the seven demoted scripts as literature-grade validation:
   heated cavity (De Vahl Davis), lid-driven cavity (Ghia, hyperbolic path),
   Bondi accretion, Fishbone–Moncrief torus, AMR convergence rigour,
   premixed flame, MHD solver comparison.

## Track 5 — Housekeeping still open

1. ReferenceTests baselines were rendered on macOS; the first honest Linux CI
   run may need baselines regenerated on CI hardware (do NOT re-add the
   auto-update env var).
2. `literate_hyperbolic/` is not executed by any test loop — its `#src`
   assertions (including the repaired dam-break check) only run manually.
3. `Release.yml` / TagBot remain disabled (unregistered fork) — revisit only
   when registration/SciML inclusion becomes a goal.

## Relevance to CRUD.jl

None of the tracks above block CRUD.jl, which uses only the parabolic
structured-mesh path (`Mesh1D`/`Mesh2D`, cylindrical models,
`Parabolic*` BCs, `assemble_system`/`assemble_mass_matrix`) — all `stable`
tier with an empty backlog. If CRUD later moves to unstructured triangulated
meshes, Track 3 item 1 (the O(h^1.5) L∞ boundary order) becomes relevant.
The v4 export surface preserves CRUD's entire import list (verified
2026-08-07; CRUD runs on FVM `4.0.0-DEV` via a dev'd path).
