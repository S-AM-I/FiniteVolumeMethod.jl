# V&V Suite Implementation Plan for FiniteVolumeMethod.jl

## Context

Two detailed verification plan documents describe an exhaustive V&V strategy aligned with ASME V&V 20-2009 standards:
- **Plan 1** ("Comprehensive V&V Suite"): general CFD — MMS, analytical benchmarks (Burgers, Barenblatt-Pattle, Smith-Hutton, Poiseuille), experimental validation (Ghia cavity, Armaly BFS, cylinder), heat transfer (De Vahl Davis)
- **Plan 2** ("Relativistic & Newtonian MHD"): physics hierarchy — GRMHD→SRMHD→Newtonian asymptotic reductions, Taylor-Green, KHI, Brio-Wu/Orszag-Tang, Bondi accretion, Fishbone-Moncrief torus

The codebase already has 19 literate verification scripts covering parabolic MMS, Euler convergence/conservation, Toro problems, MHD div(B)/convergence, and SRMHD/GRMHD convergence. The plan below fills the highest-value gaps while respecting CI constraints (8GB swap) and avoiding tests that would require significant new solver features.

---

## Phase 0: Infrastructure — GCI Utility

**File:** `test/verification_utils.jl`

Add ASME V&V 20-2009 Grid Convergence Index computation:

```julia
grid_convergence_index(e1, e2, e3, r; safety_factor=1.25)
  → (p, gci_fine, gci_coarse, asymptotic_ratio)

assert_gci_asymptotic(ratio; tol=0.1)

assert_conservation(initial, final; rtol=1e-10, labels=nothing)
```

- `p` = observed order via 3-grid Richardson extrapolation
- `gci_fine` = fine-grid GCI (uncertainty band)
- `asymptotic_ratio` should be ~1.0 in convergence range
- Validates with synthetic data: exact 1st-order and 2nd-order sequences

---

## Phase 1: Code Verification — Fill Critical Gaps

All scripts in `docs/src/literate_verification/`, following existing literate format.

### 1A. `euler_mms_convergence.jl` — MMS for Hyperbolic Euler
- **Gap:** No MMS exists for the cell-centered hyperbolic solver (only parabolic)
- **Setup:** 1D Euler with smooth manufactured solution `(rho, v, P)(x, t) = (1 + 0.2*sin(pi*x)*cos(t), ...)`. Derive source analytically, inject via `SourceOperator`
- **Grid sizes:** N ∈ {32, 64, 128, 256}
- **Accept:** L1 convergence rate > 1.5 for MUSCL+HLLC. GCI asymptotic ratio ~1.0
- **CI cost:** 1D, lightweight

### 1B. `mms_spatial_temporal_decoupled.jl` — Decoupled MMS (Parabolic)
- **Gap:** Existing `mms_convergence.jl` conflates spatial and temporal errors
- **Setup:** (a) Spatial-only: polynomial-in-time exact solution so ODE integrator is exact; refine h only. (b) Temporal-only: coarse fixed mesh, refine dt only
- **Accept:** Spatial O(h^2) independent of dt. Temporal matches Tsit5 order (~4-5) on fixed mesh
- **CI cost:** Same as existing MMS

### 1C. `mhd_solver_comparison.jl` — HLL vs HLLD Programmatic Comparison
- **Gap:** Plan 2 requires programmatic proof that HLLD < HLL in L1 error
- **Setup:** Circularly polarized Alfven wave (same as `mhd_convergence.jl`) with both solvers
- **Accept:** `@test l1_hlld < l1_hll` at N=64, 128. Rate difference > 0.2
- **CI cost:** Same as existing MHD convergence

### 1D. `balsara_mhd_suite.jl` — Full Balsara MHD Test Suite
- **Gap:** `test/reference_data/balsara_mhd_tests.json` exists but has no literate verification script
- **Setup:** All Balsara problems at N=400 with HLLD, compare against reference
- **Accept:** Shock locations within 2 cells, plateau values within 5% of reference
- **CI cost:** 1D, moderate

### 1E. `grmhd_asymptotic_flat.jl` — GRMHD Minkowski Source Terms = 0
- **Gap:** Plan 2's tier-1 asymptotic reduction (GRMHD → SRMHD in flat space)
- **Setup:** Evaluate GRMHD source terms with `MinkowskiMetric` for comprehensive state set (low/high velocity, low/high B). Verify all sources are machine-zero. Also verify flux match with SRMHD
- **Accept:** `max(|source|) < 1e-13` for all states. GRMHD flux = SRMHD flux to `atol=1e-12`
- **CI cost:** No PDE solve, just function evaluations. Negligible

### 1F. `grmhd_newtonian_limit.jl` — Low-Velocity Con2Prim Stability
- **Gap:** Plan 2's Newtonian limit asymptotic reduction
- **Setup:** Con2Prim at v ~ 1e-6, 1e-8, 1e-10 in Schwarzschild. Static atmosphere stationarity test
- **Accept:** Con2Prim converges for v < 1e-10. Static atmosphere drift < 1e-8 after 100 steps
- **CI cost:** Small grid, short run

---

## Phase 2: Analytical Benchmarks & Physics Validation

### 2A. `porous_medium_barenblatt.jl` — Barenblatt-Pattle Self-Similar Solution
- **Gap:** Plan 1 requests PME verification. Tutorial exists but no convergence study
- **Setup:** Parabolic solver with `D(u) = m * u^(m-1)`, m=2. Compare against exact self-similar profile
- **Accept:** L2 error decreases monotonically. Compact support radius matches analytical ±2 cells
- **CI cost:** Parabolic, small meshes

### 2B. `tgv_kinetic_energy_decay.jl` — Taylor-Green Vortex KE Decay
- **Gap:** Plan 2 requests explicit KE decay verification. `ns_convergence.jl` only checks velocity at one instant
- **Setup:** Track KE(t) at multiple timesteps, compare against `KE_0 * exp(-4*nu*k^2*t)`
- **Accept:** Relative error in KE decay rate < 5% at N=64. GCI on decay rate
- **CI cost:** 2D NS, resolutions [16, 32, 64]

### 2C. `srmhd_eigenmode_convergence.jl` — All SRMHD Linear Eigenmodes
- **Gap:** Plan 2 requests all eigenmodes. `srmhd_convergence.jl` only tests fast wave/density
- **Setup:** Fast magnetosonic, slow magnetosonic, Alfven, entropy — each as small perturbation along eigenvector, propagated one period
- **Accept:** Each eigenmode converges at rate > 0.8 with MUSCL+HLL
- **CI cost:** 1D, four small convergence studies

### 2D. `bondi_accretion_schwarzschild.jl` — Steady-State Bondi Accretion
- **Gap:** Plan 2's primary GRMHD validation benchmark
- **Setup:** GRMHD with `SchwarzschildMetric`, analytical Bondi profile, B=0. Run many steps, verify stationarity and mass accretion rate constancy across shells
- **Accept:** Density drift < 1% after 200 CFL steps. Accretion rate constant across shells ±5%
- **CI cost:** 2D GRMHD ~32x32. Set `run_in_ci = false` due to memory

---

## Phase 3: Experimental Validation (stretch goals)

### 3A. `lid_driven_cavity.jl` — Ghia et al. 1982
- **Reference data:** `test/reference_data/ghia_cavity_1982.json` (new file, data from Plan 1 tables)
- **Setup:** `NavierStokesEquations{2}` with no-slip walls + moving lid at Re=100, Re=400
- **Accept:** Centerline velocities match Ghia ±5% at Re=100 (N=64)
- **CI:** `run_in_ci = false` — memory-intensive, needs many timesteps to steady state

### 3B. `fishbone_moncrief_torus.jl` — FM Torus Equilibrium (Phase 1 of 3)
- **Setup:** GRMHD with `KerrMetric`, hydro-only FM torus, verify equilibrium maintained
- **Accept:** Density drift < 5% after 1 orbital period. Velocity < 1e-3
- **CI:** `run_in_ci = false` — GRMHD + Kerr metric is very memory-intensive

### 3C. `heated_cavity.jl` — De Vahl Davis 1983
- **Reference data:** `test/reference_data/devahldavis_1983.json`
- **Setup:** NS + Boussinesq buoyancy via operator splitting. Ra = 10^3, 10^4
- **Accept:** Average Nusselt number within 5% of reference at Ra=10^3
- **CI:** `run_in_ci = false`. May require new Boussinesq source term implementation — defer if invasive

---

## Phase 4: Registration & CI Integration

For each new script:

1. **`validation/manifest.toml`** — add `[[generated_pages]]` and optionally `[[scientific_evidence]]` entries
2. **`test/runtests.jl`** — add to `file_names` array in Verification testset, update `@test length(files) == length(file_names)`
3. **Promote to scientific evidence** (run in CI's dedicated job):
   - `euler_mms_convergence.jl` — fills biggest gap
   - `grmhd_asymptotic_flat.jl` — lightweight, validates GR→SR reduction
   - `srmhd_eigenmode_convergence.jl` — validates all wave families
4. **Update feature maturity** in manifest.toml after passing:
   - `relativistic`: `provisional` → `stable` (after Bondi + eigenmode pass)
   - `hyperbolic` validation: `executed_examples` → `convergence_verified` (after Euler MMS)

---

## What Is Excluded (And Why)

| Requested | Reason for exclusion |
|-----------|---------------------|
| Smith-Hutton problem | Requires deprecated Simu.jl parabolic/engine solver |
| Circular cylinder / backward-facing step | Requires body-fitted mesh or immersed boundary (not in solver) |
| Israel-Stewart relaxation τ→0 | Viscous Israel-Stewart equations not implemented |
| KHI growth rate measurement | Depends heavily on numerical viscosity; not a well-posed convergence test |
| FM torus MRI growth (phases 2-3) | Requires 3D + very long runtime; impractical for CI/testing |
| Symbolics.jl for MMS source generation | Heavy dependency; hand-derivation is straightforward for smooth profiles used |
| Poiseuille flow | Compressible solver lacks classical pressure-velocity coupling |
| Heated cylinder crossflow | Same geometry limitation as circular cylinder |

---

## Dependency Order

```
Phase 0 (GCI utility)
  ├─ Phase 1A (Euler MMS)
  ├─ Phase 1B (Decoupled MMS)
  ├─ Phase 1C (HLL vs HLLD)
  ├─ Phase 1D (Balsara suite)
  ├─ Phase 1E (GRMHD flat limit)
  └─ Phase 1F (GRMHD Newtonian limit)
       ├─ Phase 2D (Bondi accretion)
       └─ Phase 3B (FM torus)

Phase 1A ─── Phase 2A (Barenblatt-Pattle)
Phase 1C ─── Phase 2C (SRMHD eigenmodes)
existing ns_convergence ─── Phase 2B (TGV KE decay)

Phase 2 complete ─── Phase 3A (Lid-driven cavity)
                 ─── Phase 3C (Heated cavity)

All phases ─── Phase 4 (registration)
```

All Phase 1 items are independent of each other and can be parallelized.

---

## Verification Approach

After implementation, verify by:
1. Run `julia --project -e 'using Runic; Runic.main(["--check", "."])'` — all new files must pass
2. Run each new script individually: `julia --project docs/src/literate_verification/<script>.jl`
3. Run full verification testset: `julia --project -e 'using Pkg; Pkg.test()'`
4. Confirm manifest consistency: check that `validation/manifest.toml` parses correctly and new entries are picked up by `test/scientific_evidence.jl`
5. Verify GCI utility with synthetic data before using in convergence scripts
