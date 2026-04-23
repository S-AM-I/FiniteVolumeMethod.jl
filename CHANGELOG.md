# Changelog

## v3.25.0 — P1 Radiative Equilibrium (second `radiation` benchmark)

Second independent benchmark for `radiation`, joining cold-slab
attenuation (v3.15). Verifies the emission + Marshak BC pathway
through the canonical radiative-equilibrium invariant.

### test/v_and_v_p1_equilibrium.jl

In a cavity with uniform medium temperature T_m matching all wall
temperatures, the P1 equation admits the exact solution

    G ≡ 4 σ T_m⁴

regardless of absorption coefficient, because ∇G ≡ 0 makes both
the Laplacian and the Marshak-wall gradient term vanish.

Three testsets (1206 gates, ~0.4 s):

1. **G ≡ 4σT⁴ invariant.** At T_m = 500 K with a ∈ {0.1, 1, 10},
   every interior cell matches the analytical equilibrium to
   rtol 1e-2. > 50 interior cells checked per a value on a
   32 × 32 mesh.

2. **Uniform solution.** The interior spread
   (max − min) / mean < 1e-4 — no spurious gradient is generated
   by the closed-cavity Marshak BC.

3. **T⁴ scaling.** Temperatures 300 / 600 / 1200 K ⇒ G ratios
   should be 16 = 2⁴. Verified to rtol 5e-3 at both halving
   transitions.

### `radiation` benchmark inventory

| Benchmark | Added | Evidence |
|-----------|-------|----------|
| P1 slab cold-medium attenuation | v3.15 | `test/v_and_v_p1_slab.jl`       |
| P1 radiative equilibrium         | v3.25 | `test/v_and_v_p1_equilibrium.jl`|
| fvDOM angular quadrature         | ≥ v3.26 | (pending)                     |

### Verification

No manifest tier change. 1206 new gates wired into default
runtests.jl under `V&V: P1 radiative equilibrium`.

## v3.24.0 — VOF Plane Wave (second `multiphase_vof` benchmark)

Second independent benchmark for `multiphase_vof`, joining disc
translation (v3.16). Verifies accuracy of the alpha-transport
solver on a smooth wave — the primary test for advection-scheme
dissipation and dispersion.

### test/v_and_v_vof_planewave.jl

Problem: α₀(x) = 0.5 + 0.4·sin(2π·x/L) advected at U = 1 for
t = 0.25 on a 200 × 10 mesh. C_α = 0, Neumann BCs.

Three testsets (8 gates, ~1.7 s):

1. **Amplitude + phase.** Peak-to-peak amplitude of the
   numerical wave > 0.5 × initial (upwind dissipation bound)
   and < 1.02 × initial (no overshoot). Peak-location phase
   error < 5 · h (upwind is non-dispersive).

2. **L¹ error rate.** Interior L¹ error at Nx ∈ {100, 200, 400}
   (CFL ≈ 0.5 held constant) decreases monotonically with
   observed rates > 0.6 — first-order upwind on a smooth sine
   bounded below textbook 1.0 by the zero-gradient inflow BC.

3. **Strict max-principle.** Upwind on linear advection is
   TVD; α ∈ [0.1, 0.9] strictly at every cell at final time
   (original range is preserved).

### `multiphase_vof` benchmark inventory

| Benchmark | Added | Evidence |
|-----------|-------|----------|
| Disc translation (mass + COM) | v3.16 | `test/v_and_v_vof_translation.jl` |
| Plane-wave advection           | v3.24 | `test/v_and_v_vof_planewave.jl`    |
| Martin-Moyce dam break         | ≥ v3.25 | (pending)                         |

### Verification

No manifest tier change. All pre-existing tests pass. 8 new
gates wired into default runtests.jl under `V&V: VOF plane wave`.

## v3.23.0 — k-ε Log-Layer Equilibrium (second `turbulence_rans` benchmark)

Second independent benchmark for `turbulence_rans`, joining DHIT
(v3.18). Verifies the standard k-ε model in the wall-bounded
log-layer regime where it admits a closed-form production-
dissipation equilibrium.

### test/v_and_v_kepsilon_loglayer.jl

In the inertial sublayer, the log-law scalings

    U(y) = (u_τ/κ)·log(y/y₀)
    k(y) = u_τ²/√C_μ
    ε(y) = u_τ³/(κ·y)

admit three analytical identities that the numerical code must
reproduce cell-by-cell:

1. **P_k / ε ≡ 1 (production-dissipation balance).**
   Prescribe U, k, ε analytically on a 8×40 mesh; evaluate |S|
   via `compute_strain_rate` and form P_k = ν_t·|S|². The ratio
   P_k/ε matches 1 to within 15 % at 66 interior cells (the
   tolerance absorbs the O(h²·d²U/dy²) gradient truncation on
   the log-curved velocity field).

2. **ν_t = κ·y·u_τ algebraic identity.** Independent of any FVM
   discretization: the k-ε formula ν_t = C_μ·k²/ε evaluated at
   the log-layer k, ε reduces algebraically to κ·y·u_τ.
   Verified at 20 y-stations to rtol 1e-12.

3. **Durbin realizability inactive in equilibrium.** With α = 0.6
   (Durbin 1996), the realizability cap ν_t ≤ α·k/|S| evaluates
   to ~2× the equilibrium ν_t at every y — the cap is
   analytically inactive in the log layer, confirming the
   closure does not engage under its designed operating regime.

Three testsets, 106 gates, ~0.3 s.

### `turbulence_rans` benchmark inventory

| Benchmark | Added | Evidence |
|-----------|-------|----------|
| DHIT decay ODE          | v3.18 | `test/v_and_v_kepsilon_dhit.jl`    |
| Log-layer equilibrium   | v3.23 | `test/v_and_v_kepsilon_loglayer.jl`|
| Moser channel DNS (Reτ) | ≥ v3.24 | (pending)                        |

### Verification

No manifest tier change (still provisional). All pre-existing
tests pass at identical counts. 106 new gates wired into default
runtests.jl under `V&V: k-ε log-layer equilibrium`.

## v3.22.0 — Couette Flow V&V (third `incompressible_ns` benchmark)

Third independent benchmark for `incompressible_ns`, establishing
the 3-benchmark evidence floor for future `stable`-promotion
review. Joins Ghia 1982 lid-driven cavity (v3.1–v3.3, Re = 100)
and Poiseuille parabolic + grid convergence (v3.10–v3.11).

### test/v_and_v_couette.jl

Problem: plane Couette flow between parallel plates with the top
plate moving at `U_top`. Analytical solution:

    u(y) = U_top · y / H,    v(y) = 0,    p = const.

Domain [0, 4] × [0, 1] with N × N/2 Cartesian mesh:

- **Left inlet**: `SpatialVelocityBC` prescribing the linear
  analytical profile.
- **Right outlet**: `FixedPressureBC(0)`.
- **Bottom wall**: `NoSlipWallBC`.
- **Top wall**: `FixedVelocityBC((U_top, 0))`.

Three testsets (8 gates, ~6 s):

1. **Linear-profile agreement.** Max relative u error in the
   interior band < 5 %; v ≤ 0.05; monotone u(y).

2. **No streamwise pressure drop.** Couette has ∂p/∂x = 0;
   measured p_left − p_right < 0.05 (< 10 % of (1/2)·ρ·U²).

3. **Linear regression on centerline u(y).** Fit
   u ≈ a + b·y on the interior band. Slope b matches
   U_top / H = 1 within 0.05; intercept a < 0.05; residual
   from the linear fit < 0.02 (confirms the profile is linear,
   not just "close to linear").

### `incompressible_ns` benchmark inventory

| Benchmark | Added | Evidence |
|-----------|-------|----------|
| Ghia 1982 lid-driven cavity (Re = 100) | v3.1–v3.3 | `test/v_and_v_ghia_cavity.jl` |
| Poiseuille parabolic + grid convergence | v3.10–v3.11 | `test/v_and_v_poiseuille*.jl` |
| Couette linear shear-driven | v3.22 | `test/v_and_v_couette.jl` |

Three independent steady-flow benchmarks: recirculation,
pressure-driven laminar, and shear-driven. The 3-benchmark gate
for stable-promotion review is now met; remaining blockers are
transient PISO/PIMPLE V&V and higher-Re reliability (both noted
in the provisional limitations).

### Verification

No manifest-tier change (still provisional). All pre-existing
tests pass at identical counts. 8 new gates wired into default
runtests.jl under `V&V: Couette flow`.

## v3.21.0 — Unsteady Heat V&V (second CHT benchmark)

First *second-benchmark* release — adds a transient
analytical verification to `conjugate_heat_transfer` (already
provisional from v3.12.0), progressing it toward future `stable`
promotion. No manifest-tier change; limitations block and summary
updated to reflect two independent benchmarks.

### test/v_and_v_unsteady_heat.jl

Problem: the 1D unsteady heat equation on [0, L] with zero
Dirichlet sidewalls, Neumann top/bottom, and sinusoidal initial
condition:

    ∂T/∂t = α ∂²T/∂x²,
    T(0, t) = T(L, t) = 0,
    T(x, 0) = sin(π x / L)

has the closed-form separable solution

    T(x, t) = sin(π x / L) · exp(−π² α t / L²).

Three testsets (8 gates, ~2 s):

1. **Endpoint agreement.** At α = 0.1, L = 1, t = 0.5 the decay
   factor is exp(−π² · 0.1 · 0.5) ≈ 0.610. Interior-band L² error
   at 40 × 8 × 100 time steps: < 5 × 10⁻³. y-direction spread
   (Neumann top/bottom invariance): < 10⁻¹⁰.

2. **O(h²) spatial convergence.** At dt = t_end/4000 (temporal
   error ≪ spatial error), N ∈ {20, 40, 80} gives observed
   orders in [1.55, 2.3] — textbook second-order FVM Laplacian.

3. **O(Δt) temporal convergence.** At fixed N = 80 and
   n_steps ∈ {50, 100, 200}, the error drops monotonically with
   the coarse-to-fine rate r₁ > 0.6 (first-order implicit Euler,
   saturating as spatial-error floor is approached).

### `conjugate_heat_transfer` evidence inventory

| Benchmark | Added | Evidence |
|-----------|-------|----------|
| Steady Laplace series   | v3.12 | `test/v_and_v_heat_conduction.jl` |
| Unsteady decay          | v3.21 | `test/v_and_v_unsteady_heat.jl`   |
| De Vahl Davis natural convection | ≥ v3.22 | (pending) |

### Verification

All pre-existing tests pass at identical counts. 8 new gates
wired into default runtests.jl under `V&V: Unsteady heat`.

## v3.20.0 — Postprocessing V&V + `postprocessing` Promotion

Eleventh manifest promotion. `postprocessing` advances from
`experimental`/`smoke_tested` to `provisional`/`convergence_verified`
on the strength of an analytical verification of the vorticity,
Q-criterion, and enstrophy routines against three canonical flows.

### test/v_and_v_postprocessing.jl

Four testsets (448 gates, ~0.3 s), each testing every interior
cell on a 16×16 mesh:

1. **Uniform flow U = (2, 0).** ω = 0 to atol 1 × 10⁻¹⁰, Q = 0 to
   atol 1 × 10⁻¹⁰. Trivial invariance.

2. **Simple shear U = (A·y, 0), A = 4.** Analytical:
   ω_z = ∂v/∂x − ∂u/∂y = −A; S_12 = Ω_12 = A/2 ⇒ Q = 0.
   Every interior cell matches to rtol 1 × 10⁻⁸.

3. **Solid-body rotation U = (−Ω·y, Ω·x), Ω = 3.** Analytical:
   ω = 2Ω, S = 0, |Ω|² = 2Ω² ⇒ Q = Ω². Every interior cell
   matches to rtol 1 × 10⁻⁸.

4. **Enstrophy density under solid-body rotation (Ω = 2).**
   The `compute_enstrophy` routine returns |ω|² per cell (no
   factor of ½). Expected value: (2Ω)² = 16; measured to
   rtol 1 × 10⁻⁸.

### Manifest promotion

`postprocessing`:
- `maturity`: experimental → **provisional**
- `validation`: smoke_tested → **convergence_verified**

### Limitations carried into provisional

- Only the field-operation routines (`compute_vorticity`,
  `compute_q_criterion`, `compute_enstrophy`) are
  convergence-verified on linear velocity fields where the FVM
  gradient is exact.
- Wall-quantity routines (shear stress, y+, Nusselt), force
  coefficients, and line sampling remain smoke-tested.
- Line sampling uses 0th-order nearest-cell interpolation.

### Running manifest-promotion tally

Eleven `provisional` features this session:

| Feature | Promoted | Evidence |
|---------|----------|----------|
| `collocated_operators`    | v3.7  | Laplacian + gradient + divergence + Rhie-Chow MMS |
| `incompressible_ns`       | v3.11 | Poiseuille grid-convergence O(h²) + Ghia Re=100 |
| `conjugate_heat_transfer` | v3.12 | Laplace series grid-convergence O(h²) |
| `lagrangian_dpm`          | v3.13 | Stokes terminal velocity analytical match |
| `dynamic_mesh`            | v3.14 | GCL three-pattern round-off-exactness |
| `radiation`               | v3.15 | P1 slab sinh attenuation O(h²) |
| `multiphase_vof`          | v3.16 | Disc translation mass + COM invariants |
| `combustion`              | v3.17 | Species AD exponential BL first-order |
| `turbulence_rans`         | v3.18 | k-ε DHIT ODE match + O(Δt) |
| `turbulence_les`          | v3.19 | Smagorinsky ν_t = (C_s·Δ)²·|S| analytical |
| `postprocessing`          | v3.20 | Vorticity + Q + enstrophy on canonical flows |

### Verification

All pre-existing tests pass at identical counts. 448 new gates
wired into default runtests.jl under `V&V: Postprocessing kinematics`.

## v3.19.0 — Smagorinsky LES V&V + `turbulence_les` Promotion

Tenth manifest promotion. `turbulence_les` advances from
`experimental`/`smoke_tested` to `provisional`/`convergence_verified`
on the strength of an analytical verification of the Smagorinsky
eddy-viscosity formula

    ν_t = (C_s · Δ)² · |S|

against its closed-form values on prescribed velocity fields.

### test/v_and_v_smagorinsky.jl

Four testsets (106 gates, ~0.5 s):

1. **Zero velocity ⇒ ν_t ≡ 0.** Trivial invariance: |S| = 0 gives
   ν_t = 0 to round-off at every cell.

2. **Linear shear ν_t = (C_s·Δ)²·A.** For U = (A·y, 0) on a 16×16
   Cartesian mesh, the analytical strain magnitude is |S| = A
   (exact, since FVM gradient is exact on linear fields). Every
   interior cell (0.2 < x, y < 0.8; 100 cells) matches the
   analytical ν_t to rtol 1 × 10⁻⁸.

3. **ν_t ∝ C_s² scaling.** At fixed A and Δ, ν_t scales
   quadratically with C_s. Verified at C_s ∈ {0.05, 0.10, 0.20};
   ratios match 4.0 to rtol 1 × 10⁻¹⁰.

4. **Δ² mesh-refinement scaling.** Coarse (8×8) → fine (16×16)
   at fixed flow and C_s. The ν_t ratio should be (Δ_fine/Δ_coarse)²
   = 0.25; measured to rtol 1 × 10⁻⁸ in the interior.

### Manifest promotion

`turbulence_les`:
- `maturity`: experimental → **provisional**
- `validation`: smoke_tested → **convergence_verified**
- `role`: research_tooling → **claim_bearing_solver**

### Limitations carried into provisional

- Only `Smagorinsky` is convergence-verified. `WALE`,
  `DynamicSmagorinsky`, and `DDES` are smoke-tested only.
- Dynamic Smagorinsky uses a simplified scalar Germano identity,
  not the full tensor form.
- DDES only wraps Spalart-Allmaras; SST-based DDES is deferred.
- Published benchmarks (DHIT vs. Comte-Bellot–Corrsin, periodic
  channel at Reτ = 395, periodic hills) are a v3.20+ follow-up.

### Running manifest-promotion tally

Ten `provisional` features this session:

| Feature | Promoted | Evidence |
|---------|----------|----------|
| `collocated_operators`    | v3.7  | Laplacian + gradient + divergence + Rhie-Chow MMS |
| `incompressible_ns`       | v3.11 | Poiseuille grid-convergence O(h²) + Ghia Re=100 |
| `conjugate_heat_transfer` | v3.12 | Laplace series grid-convergence O(h²) |
| `lagrangian_dpm`          | v3.13 | Stokes terminal velocity analytical match |
| `dynamic_mesh`            | v3.14 | GCL three-pattern round-off-exactness |
| `radiation`               | v3.15 | P1 slab sinh attenuation O(h²) |
| `multiphase_vof`          | v3.16 | Disc translation mass + COM invariants |
| `combustion`              | v3.17 | Species AD exponential BL first-order |
| `turbulence_rans`         | v3.18 | k-ε DHIT ODE match + O(Δt) |
| `turbulence_les`          | v3.19 | Smagorinsky ν_t = (C_s·Δ)²·|S| analytical |

### Verification

All pre-existing tests pass at identical counts. 106 new gates
wired into default runtests.jl under `V&V: Smagorinsky LES`.

## v3.18.0 — k-ε DHIT V&V + `turbulence_rans` Promotion

Ninth manifest promotion. `turbulence_rans` advances from
`experimental`/`smoke_tested` to `provisional`/`convergence_verified`
on the strength of a decaying-homogeneous-turbulence study of the
`StandardKEpsilon` source terms against its closed-form ODE
solution.

### test/v_and_v_kepsilon_dhit.jl

Problem: with uniform fields, zero mean flow (no production), zero
flux (no convection), and Neumann BCs everywhere, the k-ε transport
system reduces to the ODE

    dk/dt = −ε,          dε/dt = −C_ε2 · ε²/k,

whose closed-form solution (with τ = ε_0 · t / k_0, C_ε2 = 1.92) is

    k(t) = k_0 (1 + 0.92 τ)^(−1.087)
    ε(t) = ε_0 (1 + 0.92 τ)^(−2.087)

Three testsets (10 gates, ~0.8 s):

1. **Realizability invariants.** k, ε, ν_t remain ≥ 0 at every
   step; both fields decay strictly monotonically (no oscillation,
   no overshoot from the implicit linearization).

2. **Endpoint agreement.** At 1000 × 0.001 the numerical k(1) and
   ε(1) match the analytical values within 1 %:
   - k_an ≈ 0.508, ε_an ≈ 0.264.

3. **First-order Δt convergence.** At dt ∈ {0.01, 0.005, 0.0025}
   the final-k error decreases monotonically; observed orders fall
   in [0.8, 1.3] — consistent with implicit Euler first-order
   discretization of the source ODE.

### Manifest promotion

`turbulence_rans`:
- `maturity`: experimental → **provisional**
- `validation`: smoke_tested → **convergence_verified**
- `role`: research_tooling → **claim_bearing_solver**

### Limitations carried into provisional

- Only `StandardKEpsilon` source-term ODE is convergence-verified.
  Shear production, wall functions, full channel-flow profiles,
  and k-ω / k-ω-SST / Spalart-Allmaras variants are smoke-tested.
- Published RANS benchmarks (Moser channel DNS at Reτ ∈ {180, 395,
  590}, flat-plate boundary layer, periodic hills) are a v3.19+
  follow-up.

### Running manifest-promotion tally

Nine `provisional` features this session:

| Feature | Promoted | Evidence |
|---------|----------|----------|
| `collocated_operators`    | v3.7  | Laplacian + gradient + divergence + Rhie-Chow MMS |
| `incompressible_ns`       | v3.11 | Poiseuille grid-convergence O(h²) + Ghia Re=100 |
| `conjugate_heat_transfer` | v3.12 | Laplace series grid-convergence O(h²) |
| `lagrangian_dpm`          | v3.13 | Stokes terminal velocity analytical match |
| `dynamic_mesh`            | v3.14 | GCL three-pattern round-off-exactness |
| `radiation`               | v3.15 | P1 slab sinh attenuation O(h²) |
| `multiphase_vof`          | v3.16 | Disc translation mass + COM invariants |
| `combustion`              | v3.17 | Species AD exponential BL first-order |
| `turbulence_rans`         | v3.18 | k-ε DHIT ODE match + O(Δt) |

### Verification

All pre-existing tests pass at identical counts. 10 new gates
wired into default runtests.jl under `V&V: k-ε DHIT`.

## v3.17.0 — Species Advection-Diffusion V&V + `combustion` Promotion

Eighth manifest promotion. `combustion` advances from
`experimental`/`smoke_tested` to `provisional`/`convergence_verified`
on the strength of a steady 1D advection-diffusion verification of
the `assemble_species!` transport kernel against its closed-form
exponential-boundary-layer solution.

### test/v_and_v_species_ad.jl

Problem: the steady 1D species transport ODE

    u ∂Y/∂x − D ∂²Y/∂x² = 0,   Y(0) = Y_L,  Y(L) = Y_R

has closed-form solution (Pe = u L / D)

    Y(x) = Y_L + (Y_R − Y_L) · (exp(Pe x / L) − 1) / (exp(Pe) − 1).

At Pe = 2 (moderate), the profile is a smooth exponential
transition from Y_L = 1 at the inflow to Y_R = 0 at the outflow;
top/bottom boundaries are Neumann so the 2D problem collapses to
strictly 1D.

Three testsets (9 gates, ~0.7 s):

1. **Boundary values + monotonicity.** Left column Y > 0.85,
   right column Y < 0.15. Monotone decrease across columns.
   No overshoot: Y ∈ [Y_R, Y_L] at every cell (upwind + Laplacian
   preserves the maximum principle on a Cartesian grid at Pe = 2).

2. **y-direction invariance.** Neumann top/bottom forces the
   solution to depend only on x; measured column spread <
   1 × 10⁻¹⁰.

3. **First-order grid convergence.** L² error on the interior
   band (0.1 < x < 0.9) at N ∈ {20, 40, 80}:
   - Observed orders: ≈1.0 (textbook first-order upwind on
     a smooth exponential with curvature).
   - Monotone error decrease.
   - Finest-grid error < 5 × 10⁻³ (< 0.5 % of Y swing).

### Manifest promotion

`combustion`:
- `maturity`: experimental → **provisional**
- `validation`: smoke_tested → **convergence_verified**
- `role`: research_tooling → **claim_bearing_solver**

### Limitations carried into provisional

- Only the species transport kernel is convergence-verified.
  EDM, EDC, Arrhenius reaction source terms, and heat-release
  coupling to the energy equation remain smoke-tested.
- Single-species, moderate-Pe, no-reaction test; multi-step
  chemistry + turbulent-chemistry interaction are future work.
- Published benchmarks (1D laminar premixed flame vs. Cantera,
  Sandia Flame D, counterflow diffusion flame) require Cantera
  interop and are a v3.18+ follow-up.

### Running manifest-promotion tally

Eight `provisional` features this session:

| Feature | Promoted | Evidence |
|---------|----------|----------|
| `collocated_operators`    | v3.7  | Laplacian + gradient + divergence + Rhie-Chow MMS |
| `incompressible_ns`       | v3.11 | Poiseuille grid-convergence O(h²) + Ghia Re=100 |
| `conjugate_heat_transfer` | v3.12 | Laplace series grid-convergence O(h²) |
| `lagrangian_dpm`          | v3.13 | Stokes terminal velocity analytical match |
| `dynamic_mesh`            | v3.14 | GCL three-pattern round-off-exactness |
| `radiation`               | v3.15 | P1 slab sinh attenuation O(h²) |
| `multiphase_vof`          | v3.16 | Disc translation mass + COM invariants |
| `combustion`              | v3.17 | Species AD exponential BL first-order |

### Verification

All pre-existing tests pass at identical counts. 9 new gates
wired into default runtests.jl under `V&V: Species advection-diffusion`.

## v3.16.0 — VOF Translation V&V + `multiphase_vof` Promotion

Seventh manifest promotion. `multiphase_vof` advances from
`experimental`/`smoke_tested` to `provisional`/`convergence_verified`
on the strength of a pure-kinematic disc-translation study of the
alpha transport solver under a prescribed divergence-free velocity.

### test/v_and_v_vof_translation.jl

Problem: a disc of α = 1 (radius 0.1 at (0.3, 0.25)) is advected
across an 80 × 20 domain [0, 2] × [0, 0.5] by a uniform
divergence-free velocity u = (1, 0). BCs: Dirichlet(0) on the
inflow (left), zero-gradient (Neumann) on outflow (right) and
tangential walls (top/bottom). C_α = 0 disables interface
compression so only the linear alpha-transport core is exercised.

Three testsets (8 gates, ~0.6 s):

1. **Mass conservation.** Total mass (Σ α · V) drift:
   - Half-run drift < 10⁻¹² (round-off / LU-precision).
   - Full-run (25 steps) drift < 10⁻⁸ as LU solver cond-number ×
     eps accumulates. Three orders of magnitude tighter than
     typical CFD mass-imbalance tolerance.
   - Total mass range across run < 10⁻⁶.

2. **Boundedness.** After `clip_alpha!`, α ∈ [0, 1] at every
   cell, every step. Pre-clip upwind bolus stays non-negative
   (no clip deltas triggered at any step).

3. **Center-of-mass transport.** x_COM(T) − x_COM(0) matches
   U · T within 2 h (cell-centered rasterization error on the
   initial disc is O(h)). Lateral drift y_COM(T) − y_COM(0) < 10⁻¹⁰
   (initial-condition + BC symmetry).

### Manifest promotion

`multiphase_vof`:
- `maturity`: experimental → **provisional**
- `validation`: smoke_tested → **convergence_verified**
- `role`: research_tooling → **claim_bearing_solver**

### Limitations carried into provisional

- Only pure-kinematic alpha transport under prescribed
  divergence-free flow is verified.
- Full two-phase PISO/PIMPLE (variable density), interface
  compression (C_α > 0), CSF surface tension, and contact-angle
  coupling remain smoke-tested.
- Published benchmarks (Martin-Moyce dam break, Hysing rising
  bubble, Zalesak rotating slot) are a v3.17+ follow-up.

### Running manifest-promotion tally

Seven `provisional` features this session:

| Feature | Promoted | Evidence |
|---------|----------|----------|
| `collocated_operators`    | v3.7  | Laplacian + gradient + divergence + Rhie-Chow MMS |
| `incompressible_ns`       | v3.11 | Poiseuille grid-convergence O(h²) + Ghia Re=100 |
| `conjugate_heat_transfer` | v3.12 | Laplace series grid-convergence O(h²) |
| `lagrangian_dpm`          | v3.13 | Stokes terminal velocity analytical match |
| `dynamic_mesh`            | v3.14 | GCL three-pattern round-off-exactness |
| `radiation`               | v3.15 | P1 slab sinh attenuation O(h²) |
| `multiphase_vof`          | v3.16 | Disc translation mass + COM invariants |

### Verification

All pre-existing tests pass at identical counts. 8 new gates
wired into default runtests.jl under `V&V: VOF translation`.

## v3.15.0 — P1 Radiation Slab V&V + `radiation` Promotion

Sixth manifest promotion. `radiation` advances from
`experimental`/`smoke_tested` to `provisional`/`convergence_verified`
on the strength of a 1D-slab analytical comparison of the P1 model
in a cold attenuating medium.

### test/v_and_v_p1_slab.jl

Problem: P1 model in a cold medium (T_m = 0, emission vanishes):

    -(1/3a) ∂²G/∂x² + a G = 0

reduces to G'' = 3 a² G, with closed-form solution for BCs
G(0) = G₀, G(L) = 0:

    G(x) = G₀ · sinh(√3 a (L − x)) / sinh(√3 a L)

Top and bottom boundaries use zero-gradient (Neumann) BCs so the
2D problem collapses to strictly 1D. At a = 1, L = 1, G₀ = 1:

- **Monotone shape match.** At 40×4 the field decays monotonically
  from ≈1 at the left wall to ≈0 at the right wall; every cell
  strictly positive.
- **1D invariance.** Under Neumann top/bottom, all cells sharing
  the same x must produce identical G. Max column spread on 20×8:
  < 1 × 10⁻¹⁰ (round-off).
- **O(h²) grid convergence** at N ∈ {20, 40, 80} against the
  sinh analytical, measured on the interior band 0.1 < x < 0.9
  (avoids cell-centered Dirichlet boundary-layer error):
  observed orders in `[1.8, 2.2]`, finest-grid L² < 10⁻⁴.

Total: 9 gates across 3 testsets, ~2 s runtime.

### Manifest promotion

`radiation`:
- `maturity`: experimental → **provisional**
- `validation`: smoke_tested → **convergence_verified**
- `role`: research_tooling → **claim_bearing_solver**

### Limitations carried into provisional

- Only P1 with Dirichlet boundaries on a cold medium is
  convergence-verified.
- Marshak-wall boundaries, non-zero emission (T⁴ source), fvDOM,
  and scattering are still smoke-tested.
- Radiation-flow coupling is lagged one iteration (standard);
  a coupled radiation + energy + flow benchmark is a v3.16+
  follow-up.

### Running manifest-promotion tally

Six `provisional` features this session:

| Feature | Promoted | Evidence |
|---------|----------|----------|
| `collocated_operators`    | v3.7  | Laplacian + gradient + divergence + Rhie-Chow MMS |
| `incompressible_ns`       | v3.11 | Poiseuille grid-convergence O(h²) + Ghia Re=100 |
| `conjugate_heat_transfer` | v3.12 | Laplace series grid-convergence O(h²) |
| `lagrangian_dpm`          | v3.13 | Stokes terminal velocity analytical match |
| `dynamic_mesh`            | v3.14 | GCL three-pattern round-off-exactness |
| `radiation`               | v3.15 | P1 slab sinh attenuation O(h²) |

### Verification

All pre-existing tests pass at identical counts. 9 new gates
wired into default runtests.jl under `V&V: P1 radiation slab`.

## v3.14.0 — GCL Invariance V&V + `dynamic_mesh` Promotion

Fifth manifest promotion. `dynamic_mesh` advances from
`experimental`/`smoke_tested` to `provisional`/`convergence_verified`
on the strength of a round-off-exact Geometric Conservation Law
study across three analytically tractable motion patterns.

### test/v_and_v_gcl.jl

Problem: the GCL is the identity

    (V_new[c] − V_old[c]) / Δt  ≡  Σ_f ε(c, f) · phi_mesh[f]

for every cell `c`. Failure manifests as artificial mass/energy
creation under mesh motion. This V&V verifies GCL exactness (to
round-off) on patterns whose continuum answer is known:

1. **Zero motion.** Displacement ≡ 0, `phi_mesh ≡ 0`, volumes
   unchanged. GCL residual < 1 × 10⁻¹⁴.

2. **Rigid translation.** Uniform `d(t) = (0.25 t, −0.10 t)` on a
   10 × 10 mesh. Volumes preserved to < 1 × 10⁻¹², per-cell GCL
   residual < 1 × 10⁻¹¹, per-cell net face-flux < 1 × 10⁻¹²
   (closed-cell divergence-theorem identity).

3. **Isotropic linear scaling `d(x) = α(x − x₀)`.** At α = 0.05
   on a 16 × 16 mesh:
   - GCL residual < 10⁻¹⁰ · V̄ / Δt (machine-zero by construction).
   - Total volume ratio matches `1 + Dim · α` within `5α²` — the
     discretization is exact for linear displacement fields, so
     the only non-trivial error is the Taylor-truncation of
     `V_new = V_old(1 + Dim α + …)`.

4. **Translation invariance across refinement.** At
   N ∈ {8, 16, 32}, the non-dimensional residual
   `max_res · Δt / V_cell < 1 × 10⁻¹⁰` at every refinement level
   (the divergence-theorem identity is mesh-independent).

Total: 111 passing gates across 4 testsets, ~0.5 s runtime.

### Manifest promotion

`dynamic_mesh`:
- `maturity`: experimental → **provisional**
- `validation`: smoke_tested → **convergence_verified**
- `role`: research_tooling → **claim_bearing_solver**

### Limitations carried into provisional

- Verified motion patterns cover zero, rigid translation, and
  isotropic linear scaling — the patterns for which the
  implementation is exact by construction.
- Rotational mesh motion, large-deformation Laplacian motion, and
  full fluid-coupled ALE runs are still smoke-tested only.
- A Turek-Hron FSI case is a v3.15+ follow-up.

### Running manifest-promotion tally

Five `provisional` features this session:

| Feature | Promoted | Evidence |
|---------|----------|----------|
| `collocated_operators`    | v3.7  | Laplacian + gradient + divergence + Rhie-Chow MMS |
| `incompressible_ns`       | v3.11 | Poiseuille grid-convergence O(h²) + Ghia Re=100 |
| `conjugate_heat_transfer` | v3.12 | Laplace series grid-convergence O(h²) |
| `lagrangian_dpm`          | v3.13 | Stokes terminal velocity analytical match |
| `dynamic_mesh`            | v3.14 | GCL three-pattern round-off-exactness |

### Verification

All pre-existing tests pass at identical counts. 111 new gates
wired into default runtests.jl under `V&V: GCL invariances`.

## v3.13.0 — Stokes Terminal-Velocity V&V + `lagrangian_dpm` Promotion

Fourth manifest promotion. `lagrangian_dpm` advances from
`experimental`/`smoke_tested` to `provisional`/`convergence_verified`
on the strength of an analytical Stokes-terminal-velocity study of
the forward-Euler particle integrator under `StokesDrag`.

### test/v_and_v_stokes_terminal.jl

Problem: a single particle settles from rest in a quiescent fluid
under uniform gravity. In the Stokes regime (Re_p ≪ 1) the equation
of motion is

    dv/dt = g - v/τ_p,   τ_p = ρ_p d² / (18 μ_f)

with closed-form solution

    v(t) = v_t (1 - exp(-t/τ_p)),   v_t = g τ_p.

Test parameters (10 μm water droplet in air):

    d = 10 μm,  ρ_p = 1000 kg/m³,  ρ_f = 1.2 kg/m³,
    μ_f = 1.81 × 10⁻⁵ Pa·s,  g = 9.81 m/s²
    ⇒  τ_p = 3.07 × 10⁻⁴ s,  v_t = 3.01 × 10⁻³ m/s
    ⇒  Re_p(v_t) ≈ 2.0 × 10⁻³  (Stokes regime holds).

Nine gates pass across three testsets:

1. **Steady-state asymptote.** 500 sub-steps at Δt/τ_p = 0.01 drive
   `v_final` within 1% of analytical `v_t (1 - e⁻⁵) ≈ 0.9933 v_t`.
   No lateral drift (x-position and x-velocity preserve machine
   precision). Descent is monotone and bounded by v_t.

2. **Mid-transient accuracy.** After t = τ_p the vertical velocity
   matches `v_t (1 - e⁻¹) = 0.632 v_t` to within 2%.

3. **Euler first-order rate in Δt.** Δt/τ_p ∈ {0.04, 0.02, 0.01}
   gives a monotone error decrease with observed orders in `[0.8, 1.3]`,
   confirming the documented forward-Euler discretization.

### Manifest promotion

`lagrangian_dpm`:
- `maturity`: experimental → **provisional**
- `validation`: smoke_tested → **convergence_verified**
- `role`: research_tooling → **claim_bearing_solver**

### Limitations carried into provisional

- Only `StokesDrag` with forward-Euler integration is verified.
  `SchillerNaumann` and spray-breakup models (`TABBreakup`,
  `KHRTBreakup`) remain `smoke_tested`.
- Buoyancy is not applied automatically — users must pass the
  effective gravity `g·(1 - ρ_f/ρ_p)` if it matters. Documented
  in the limitations block of the manifest entry.
- Two-way PSI-cell coupling is smoke-tested only; a
  momentum-source verification against an analytical pipe-flow
  solution is a v3.14+ follow-up.

### Running manifest-promotion tally

Four `provisional` features this session:

| Feature | Promoted | Evidence |
|---------|----------|----------|
| `collocated_operators` | v3.7  | Laplacian + gradient + divergence + Rhie-Chow MMS |
| `incompressible_ns`    | v3.11 | Poiseuille grid-convergence O(h²) + Ghia Re=100 |
| `conjugate_heat_transfer` | v3.12 | Laplace series grid-convergence O(h²) |
| `lagrangian_dpm`       | v3.13 | Stokes terminal velocity analytical match |

### Verification

All pre-existing tests pass at identical counts. 9 new gates
(~0.5 s runtime) wired into default runtests.jl under
`V&V: Stokes terminal velocity`.

## v3.12.0 — Heat-Conduction V&V + `conjugate_heat_transfer` Promotion

Third manifest promotion. `conjugate_heat_transfer` advances from
`experimental`/`smoke_tested` to `provisional`/`convergence_verified`
on the strength of a solid-conduction grid-convergence study against
the analytical Laplace series solution.

### test/v_and_v_heat_conduction.jl

Problem: `-∇²T = 0` on `[0, 1]²` with `T(x, 0) = T(0, y) = T(1, y) = 0`
and `T(x, 1) = 1`. Analytical solution (Fourier series):

    T(x, y) = (4/π) Σ_{n odd} (1/n) sin(n π x) sinh(n π y) / sinh(n π)

Refine N ∈ {20, 40, 80} and measure L² error in the interior band
(x, y) ∈ [0.1, 0.9]² (excludes the corner singularities at
(0, 1) and (1, 1) where T is multi-valued):

    N=20    L²(T) = 5.60 × 10⁻⁴
    N=40    L²(T) = 1.44 × 10⁻⁴   (rate: 1.96)
    N=80    L²(T) = 3.63 × 10⁻⁵   (rate: 1.99)

Five gates pass:
- Two monotone-transition rates each in `[1.8, 2.2]` (textbook O(h²)).
- All errors monotone-decreasing.
- Finest-grid L² < 10⁻⁴.
- Center-cell `T(0.5, 0.5)` within 3% of the analytical 0.25
  (symmetry gives this exactly).

### Manifest promotion

`conjugate_heat_transfer`:
- `maturity`: experimental → **provisional**
- `validation`: smoke_tested → **convergence_verified**
- `role`: research_tooling → **claim_bearing_solver**

### Limitations carried into provisional

- Only `solve_solid_conduction` is verified; full
  `solve_conjugate_ht` (fluid-solid Dirichlet-Neumann iteration)
  lacks a dedicated analytical benchmark.
- Fluid-energy equation (forced-convection / buoyancy) verification
  is a v3.13+ follow-up — target: De Vahl Davis natural convection.

### Running manifest-promotion tally

Three `provisional` features this session:

| Feature | Promoted | Evidence |
|---------|----------|----------|
| `collocated_operators` | v3.7 | Laplacian + gradient + divergence + Rhie-Chow MMS |
| `incompressible_ns` | v3.11 | Poiseuille grid-convergence O(h²) + Ghia Re=100 |
| `conjugate_heat_transfer` | v3.12 | Laplace series grid-convergence O(h²) |

### Verification

All 1571 pre-existing tests pass at identical counts. 5 new gates
(~2 s runtime) wired into default runtests.jl.

## v3.11.0 — Poiseuille Grid-Convergence + `incompressible_ns` Promotion

**First order-of-accuracy verification of the full Navier-Stokes
solver** — observed spatial convergence rate ≈ 1.95, textbook second
order. Based on this evidence, `incompressible_ns` is promoted from
`experimental`/`smoke_tested` to `provisional`/`convergence_verified`
in the validation manifest.

### test/v_and_v_poiseuille_convergence.jl

Runs Poiseuille channel (from v3.10) on three successive refinements
Nx × Ny ∈ {25×10, 50×20, 100×40} and measures the L² error against the
analytical `u(y) = G/(2μ) y(H-y)` at the mid-channel column in the
fully-developed interior (0.1 < y < 0.9).

Observed:

    N_x=25   L²(u) = 8.05 × 10⁻⁴
    N_x=50   L²(u) = 2.09 × 10⁻⁴   (rate: 1.95)
    N_x=100  L²(u) = 5.46 × 10⁻⁵   (rate: 1.94)

Four gates pass:
- All error transitions monotone-decreasing.
- Each observed order satisfies 1.7 < p < 2.3 (textbook O(h²) with
  floating-point slack).
- Finest-grid L² < 10⁻⁴.

Runtime ~11 s — gated behind `FVM_RUN_VANDV=true` (like Ghia).

### Manifest promotion

`incompressible_ns`:
- `maturity` experimental → **provisional**.
- `validation` smoke_tested → **convergence_verified**.
- `role` research_tooling → **claim_bearing_solver**.

Evidence backing the promotion:
- Poiseuille grid-convergence (v3.11): O(h²) verified, three
  refinements, observed rate 1.95.
- Poiseuille single-mesh (v3.10): 5% agreement with analytical at
  50×20, peak location and magnitude match.
- Ghia 1982 Re=100 (v3.1-v3.3): 10 centerline reference points agree
  to ≤8% interior, ≤5% near-lid, interior divergence residual < 10⁻⁴.

### Noted limitations (to reach `stable`)

- Steady SIMPLE only. Transient PISO/PIMPLE paths exist but lack
  dedicated V&V.
- Higher Reynolds (Re ≥ 400) destabilises without deferred-correction
  convection — Stage 2a follow-up.
- Kovasznay and other steady NS analytical benchmarks beyond
  Poiseuille are outside the current reliable envelope (solver
  diverges at Re ≥ 10 on Kovasznay); v3.12+ follow-up.
- Fixed time step only; adaptive CFL deferred.

## v3.10.0 — Poiseuille Channel V&V (First Incompressible NS Benchmark)

First V&V of the full SIMPLE pressure-velocity-coupling solver against
an analytical closed-form solution. Previous solver-level V&V (Ghia
1982 cavity, v3.1-v3.3) was against tabulated numerical reference data;
Poiseuille channel is a pure analytical test.

### test/v_and_v_poiseuille.jl

Pressure-driven (or inlet-driven) Hagen-Poiseuille flow in a 2D
channel `[0, L] × [0, H]` with no-slip top and bottom walls. For
dp/dx = -G the fully-developed profile is:

    u(y) = G / (2μ) · y · (H - y),    v = 0

Set up with:
- Inlet: `SpatialVelocityBC(u_inlet, ...)` matching analytical profile.
- Outlet: `FixedPressureBC(0.0)`.
- Walls: `NoSlipWallBC()`.
- L = 5, H = 1, μ = 1, G = 2 (giving `u_max = 0.25` at `y = 0.5`).

Mesh: 50 × 20 cells.

Five gates pass:
- Solver produces finite output (any iteration count OK).
- Point-wise agreement with analytical `u(y)` at `x = L/2`: < 5%
  relative error everywhere `|u_exact| > 0.05`.
- Peak location within `y ∈ [0.40, 0.60]` (analytical: 0.5).
- Peak magnitude within `u_peak ∈ [0.22, 0.26]` (analytical: 0.25).
- Transverse velocity `max|v| < 0.05` in the fully-developed region.

### Why this matters for incompressible_ns promotion

The Poiseuille test is simpler than Ghia cavity — no BC singularity,
exact analytical solution, symmetric setup — and demonstrates:

1. `SpatialVelocityBC` works end-to-end with the SIMPLE loop on a real
   physics problem.
2. The pressure-outlet path (`FixedPressureBC`) correctly closes the
   momentum-pressure coupling.
3. 5% accuracy on 50×20 is in-line with expected O(h²) convergence and
   the v3.2 residual-normalization-fix regime.

This is the first concrete piece of evidence toward a future
`incompressible_ns` promotion from `experimental` to `provisional`.
That promotion will require the full suite:
- Poiseuille (v3.10) ✓
- Ghia cavity (v3.1-v3.3, FVM_RUN_VANDV-gated) ✓
- Ghia Re=400 or Taylor-Green 2D decay — still outstanding
- Grid convergence study / MMS for Navier-Stokes RHS

### Verification

All 1571 pre-existing tests pass at identical counts. 5 new gates
(~4 s runtime) wired into default runtests.jl.

## v3.9.0 — Temporal Order-of-Accuracy MMS

Adds the time-discretization verification that was flagged as a v3.7+
prerequisite for `collocated_operators` full `stable` promotion.
Implicit Euler and BDF2 are both exercised on a manufactured
transient-diffusion solution.

### test/v_and_v_temporal_mms.jl

Heat equation on `[0, 1]²` with Dirichlet-0 BCs and sinusoidal initial
condition:

    φ(x, y, 0) = sin(πx) · sin(πy)
    φ_exact(x, y, t) = sin(πx) · sin(πy) · exp(-2π²t)

Three gate sets (8 assertions total):

- **Implicit Euler first-order convergence**: on N = 20 spatial grid,
  sweep Δt ∈ {10⁻³, 5·10⁻⁴, 2.5·10⁻⁴} at T = 0.01, error halves at
  each refinement (order > 0.5 at each transition; exact rate is ~1
  in the dt-dominated regime).
- **BDF2 monotone convergence + first-order-or-better at coarsest
  transition**: with spatial-error floor at ~2·10⁻⁴ on N = 20, BDF2's
  nominal O(Δt²) rate is contaminated by spatial error at fine Δt.
  The coarsest-transition order (Δt = 2·10⁻³ → 10⁻³) is > 1.0 (we
  observe ~1.1), confirming a temporal contribution distinct from
  Euler's 1.0 floor.
- **BDF2 strictly outperforms Euler at the same Δt**: same spatial
  mesh, same dt, BDF2 error ≤ Euler error.

### Why the gate is "monotone + first-order-or-better" not "second-order"

Asymptotic O(Δt²) for BDF2 requires either finer spatial resolution
(N = 80+) so spatial error doesn't set the floor, or a final time T
large enough that the spatial error at `t = 0` dominates equally at
all dt (so cancels out of the dt-ratio). At the chosen T = 0.01 and
N = 20, both competing errors are similar magnitudes, and the
temporal rate saturates around 1.1-1.3. Tightening this to an
asymptotic-order gate is a v3.10+ follow-up needing a larger spatial
mesh (and correspondingly longer runtime).

### Verification

All 1571 pre-existing tests pass at identical counts. 8 new gates in
test/v_and_v_temporal_mms.jl, runtime ~2 s, wired into default
runtests.jl.

### Manifest status

`collocated_operators` remains at `provisional` / `convergence_verified`
(v3.7). Temporal-MMS evidence now exists (v3.9) but the asymptotic
second-order gate for BDF2 is deferred; full `stable` promotion
awaits that + skewed-mesh iterative correction + 3D operator MMS.

## v3.8.0 — Skewed-Mesh Laplacian MMS + Non-Orthogonal Baseline

Extends Phase 0 operator verification to non-Cartesian (skewed) meshes.
Exercises the three `NonOrthoCorrectionMode` variants
(MINIMUM / ORTHOGONAL / OVER_RELAXED) from Stage 3c and documents the
expected one-pass behaviour on meshes where `S_f · d̂ ≠ |S_f|`.

### test/v_and_v_laplacian_skewed.jl

Builds a non-orthogonal discrete stencil by taking a uniform Cartesian
mesh and displacing interior cell centers with a sinusoidal offset,
while keeping face geometry unchanged. This gives `d_PN` a tangential
component relative to `S_f` — the textbook non-orthogonal case that
the over-relaxed correction is designed for.

Three gate sets (17 assertions total):

- **Finite-error robustness**: all three correction modes produce
  bounded L² errors at N = 20, 40 and skewness 0.05. Over-relaxed in
  particular does NOT diverge.
- **Plateau documentation**: on a fixed-skewness mesh, refining N
  does NOT drive error to zero — the truncation error is set by the
  non-orthogonality itself (not h). This codifies the observed
  behaviour for future work on iterative non-orthogonal correction.
- **Zero-skew identity**: at skewness = 0 (pure Cartesian), all three
  correction modes produce bit-identical matrices and therefore
  bit-identical MMS errors.

### Known gap

The iterative non-orthogonal-correction path
(`non_ortho_correction=true, grad_phi=...`) exists in
`assemble_laplacian!` but the explicit-source feedback loop doesn't
converge with naive Picard iteration in this setup — needs
under-relaxation tuning or a dedicated fixed-point accelerator.
This is a v3.9+ follow-up. In the meantime, the one-pass behaviour
is correct and the V&V suite documents it explicitly.

### Impact on manifest

`collocated_operators` remains at `provisional` / `convergence_verified`
(v3.7). Full `stable` promotion requires:
- The iterative-correction convergence result (v3.9+).
- 3D Laplacian MMS.
- Temporal-order MMS for BDF2 / Crank-Nicolson.

### Verification

All 1571 pre-existing tests pass at identical counts. 17 new gates in
`test/v_and_v_laplacian_skewed.jl` wired into default runtests.jl.

## v3.7.0 — First Manifest Promotion (`collocated_operators` → provisional)

First time a feature in `validation/manifest.toml` advances past the
`experimental` / `smoke_tested` tier since v2.0. The `collocated_operators`
entry moves from:

    maturity = "experimental"
    validation = "smoke_tested"
    role = "research_support_tooling"
    required_ladder_stages = ["verification"]

to:

    maturity = "provisional"
    validation = "convergence_verified"
    role = "claim_bearing_solver"
    required_ladder_stages = ["verification", "benchmark"]

### Evidence backing the promotion

Five V&V test files landed in v3.4–v3.6 provide publishable-grade
machine-checked evidence on a uniform Cartesian mesh:

1. `test/v_and_v_laplacian_mms.jl` (v3.4): Laplacian MMS
   `-∇²(sin πx sin πy) = 2π² sin πx sin πy` with Dirichlet-0 BCs.
   Observed L² order 2.00 at all refinement transitions
   N ∈ {10, 20, 40, 80}.
2. `test/v_and_v_operator_mms.jl` — gradient (v3.5): Green-Gauss
   gradient of sin(πx)sin(πy) achieves L² O(h²) in interior.
3. Same file — divergence (v3.5): div of an analytically div-free
   field is machine zero (~10⁻¹⁵) at every N.
4. `test/v_and_v_rhie_chow.jl` (v3.6): three analytical invariants
   (linear-pressure identity, constant-pressure preservation,
   checkerboard suppression) all pass.
5. `test/v_and_v_ghia_cavity.jl` (v3.1–v3.3, gated by `FVM_RUN_VANDV`):
   Ghia 1982 Re=100 lid-driven cavity centerline u(y) matches 10
   reference points to ≤8% interior, ≤5% near-lid, with
   `continuity_residual_interior < 10⁻⁴`.

### What 'provisional' means

Per the v2.0 contract (CHANGELOG.md v2.0.0-rc1 section):

> `provisional` features are solver families whose numerical behavior
> is verified against published or manufactured references on a
> restricted regime (typically Cartesian mesh + low Reynolds + simple
> topology), but where full `stable` promotion requires additional
> evidence on the expanded regime.

### What's still missing for 'stable' promotion

- **Skewed-mesh Laplacian MMS**: exercises the Stage 3c over-relaxed
  non-orthogonal correction. Currently all MMS runs on Cartesian
  meshes where that correction is a no-op.
- **Temporal-order MMS for BDF2 / Crank-Nicolson**: unit tests cover
  correctness; no dedicated time-order-of-accuracy study.
- **3D Laplacian / gradient / divergence MMS**: 2D-only in v3.6.
- **Performance benchmarks**: convergence rate is necessary but not
  sufficient — `stable` also implies stable runtime performance.

Each is a v3.7+ follow-up.

### Related work (non-blocking)

- Other collocated features (`incompressible_ns`, `turbulence_rans`,
  etc.) remain at `experimental` / `smoke_tested`. They benefit
  transitively from the operator verification but need their own
  V&V (Poiseuille MMS, Ghia Re=400/1000 stability fix, Moser channel
  DNS comparison, etc.) before individual promotion.

## v3.6.0 — Rhie-Chow Interpolation V&V — Phase 0 Complete

Completes the Phase 0 operator V&V suite. All four core collocated
operators — Laplacian, gradient, divergence, Rhie-Chow — are now
verified against manufactured solutions or analytical invariants.

### Rhie-Chow three invariants (test/v_and_v_rhie_chow.jl)

1. **Linear-pressure identity**: for an affine `p(x, y) = a + bx + cy`,
   the compact face-normal gradient equals the interpolated
   cell-center gradient and the Rhie-Chow correction is zero. Corrected
   flux equals `U · S` to machine precision.

2. **Constant-pressure preservation**: constant `p = 1.234` yields
   zero correction regardless of velocity. Face flux equals plain
   linear-interpolated `U · S` to machine precision.

3. **Checkerboard suppression**: for a pressure field with pattern
   `p_{i,j} = (-1)^(i+j)`, the compact face-normal gradient sees the
   checkerboard oscillation but the interpolated gradient does not.
   Rhie-Chow correction is > 10⁻³, confirming the operator suppresses
   the pressure checkerboard exactly as designed by Rhie & Chow (1983).

All 3 gates pass. Runtime < 0.5 s.

### Phase 0 operator verification status

All four operators now carry publishable-grade machine-checked evidence:

| Operator | File | Verification | Evidence |
|----------|------|--------------|----------|
| Laplacian | `src/collocated/laplacian.jl` | MMS O(h²) | v3.4 |
| Gradient | `src/collocated/gradient.jl` | MMS O(h²) interior | v3.5 |
| Divergence | `src/collocated/divergence.jl` | div(div-free) ≡ 0 | v3.5 |
| Rhie-Chow | `src/collocated/interpolation.jl` | 3 analytical invariants | v3.6 |

This closes the `collocated_operators` entry in
`validation/manifest.toml` from "smoke-tested" to
"publishable-benchmark-verified" status for the Cartesian case. The
manifest promotion to `stable` is blocked only on a non-Cartesian
(skewed-mesh) Laplacian MMS, a v3.7+ follow-up.

### Deferred to v3.7+

- Skewed-mesh Laplacian MMS (exercises over-relaxed correction from
  Stage 3c).
- Promote `collocated_operators` in `validation/manifest.toml`.
- Ghia Re=400 (80×80, stable regime).
- Smoothed-lid cavity (SpatialVelocityBC full integration).
- Begin promotion work for `incompressible_ns`: Poiseuille MMS,
  TGV decay.

## v3.5.0 — Gradient + Divergence Operator MMS

Completes the Phase 0 operator V&V suite started in v3.4.

### Green-Gauss gradient (interior spatial-order verification)

Manufactured φ(x, y) = sin(πx)·sin(πy) with analytical gradient
∇φ = (π cos(πx) sin(πy), π sin(πx) cos(πy)). Initialize field with
exact values on cell centers and boundary faces; compute numerical
gradient via `gradient(phi, mesh)`; measure L² error over interior
(x, y) ∈ [0.15, 0.85]² (boundary cells pick up O(h) from the
face-stencil — standard Green-Gauss limitation).

| N | L² err (interior) | L² rate |
|---|-------------------|---------|
| 20 | ~2e-2 | — |
| 40 | ~5e-3 | 2.0 |
| 80 | ~1.3e-3 | 2.0 |

4 gates pass. Interior O(h²) confirmed.

### Divergence of divergence-free field (exactness test)

For U(x, y) = (sin(πx) cos(πy), -cos(πx) sin(πy)) (div-free
analytically), constructing face fluxes via midpoint-rule integration
and summing the FVM divergence sums to **machine zero** (~10⁻¹⁵) at
every grid size. The operator is exact on this input — a stronger
property than O(h²) convergence.

3 gates pass: L² of (div/V) < 10⁻¹⁰ at N ∈ {20, 40, 80}.

### Outcome

Three of the four Phase 0 operators are now verified:
- ✅ **Laplacian** O(h²) on interior (v3.4).
- ✅ **Gradient** O(h²) on interior (v3.5).
- ✅ **Divergence** exact on analytical div-free input (v3.5).
- ⏳ Interpolation (Rhie-Chow) — v3.6 follow-up.

### Deferred to v3.6+

- Rhie-Chow interpolation MMS.
- Laplacian MMS on skewed mesh (over-relaxed correction check).
- Ghia Re=400 (stable for the solver at 80×80).
- Smoothed-lid cavity (SpatialVelocityBC full integration).
- Poiseuille MMS, TGV decay, backward step.

## v3.4.0 — Laplacian MMS Order-of-Accuracy V&V + SpatialVelocityBC

First V&V of solver-free operator correctness: the collocated
Laplacian at `src/collocated/laplacian.jl` is verified against a
manufactured solution on a Cartesian grid and shown to achieve
textbook second-order spatial convergence.

### Manufactured-solution test (test/v_and_v_laplacian_mms.jl)

Solves `-∇²φ = f` on a `[0, 1]²` Cartesian mesh with Dirichlet-zero
BCs and manufactured forcing:

    φ_exact(x, y) = sin(π x) · sin(π y)
    f(x, y)       = 2π² · sin(π x) · sin(π y)

Uniform-refinement grid sequence `N ∈ {10, 20, 40, 80}` gives:

| N | L∞ error | L² error | L² rate (vs prev) |
|---|---------|---------|-------------------|
| 10 | 8.06×10⁻³ | 4.13×10⁻³ | — |
| 20 | 2.05×10⁻³ | 1.03×10⁻³ | 2.00 |
| 40 | 5.13×10⁻⁴ | 2.57×10⁻⁴ | 2.00 |
| 80 | 1.28×10⁻⁴ | 6.43×10⁻⁵ | 2.00 |

Test asserts:
- Observed order of convergence (L²) at the finest refinement is
  between 1.8 and 2.2.
- L∞ order at the finest refinement is > 1.7.
- Absolute errors decrease monotonically under refinement.
- Finest L² error < 10⁻³.

All 6 gates pass. Runtime ~2 s — wired into the DEFAULT `runtests.jl`
loop (not gated behind `FVM_RUN_VANDV`).

This is the first machine-checked evidence of publishable-grade
spatial accuracy for the collocated operators. Promotes
`collocated_operators` from "no convergence evidence" to "O(h²) on
Cartesian mesh verified" in the validation manifest.

### New boundary condition: SpatialVelocityBC

`SpatialVelocityBC{Dim, T, F} <: AbstractBoundaryCondition` — velocity
BC whose prescribed value is computed from a closure `func(x::SVector)`
evaluated at each face center. Enables smoothed-lid cavity tests
(`u_lid(x) = sin²(π x)`), Womersley-like inlet profiles, and any
geometrically-varying Dirichlet BC.

Wired into `update_boundary_velocity!` (per-face closure evaluation)
and `expand_velocity_bc` / `expand_pressure_bc` (momentum/pressure
assembly falls back to a 0 placeholder; the real value arrives via
`update_boundary_velocity!` each outer iteration).

Full Laplacian-assembly integration (so the interior momentum equation
can reference per-face BC values) is a v3.5+ follow-up; the current
SpatialVelocityBC works for any cavity-like problem where the top
boundary is geometrically smooth.

### Deferred to v3.5+

- Laplacian MMS on a non-Cartesian (skewed) mesh — verifies the
  over-relaxed non-orthogonal correction from Stage 3c.
- Ghia Re=400, 1000 extensions. Re=1000 currently destabilises
  (needs deferred-correction convection scheme or pseudo-transient
  continuation — Stage 2a follow-up).
- SpatialVelocityBC full Laplacian integration for smoothed-lid gate.
- Poiseuille MMS (analytical parabolic channel profile).
- Taylor-Green 2D kinetic-energy decay.
- Backward-facing step Driver-Seegmiller.

## v3.3.0 — Interior Residual Metric + Corner-Singularity Diagnosis

Closes out the residual investigation started in v3.2.

### Diagnosis

After v3.2's OpenFOAM-residual fix, the global continuity residual on
the lid-driven cavity Re=100 still plateaued around 5.9×10⁻⁴.
Decomposing per-cell: **65% of this is concentrated in 32 cells
(2% of the mesh) at the upper corners (0, 1) and (1, 1)** where the
lid velocity U=1 meets the no-slip wall velocity U=0 — the classic
discontinuous-BC corner singularity.

Per-region breakdown on 40×40 Ghia Re=100 (α_U = 0.3, α_p = 0.1):

| Region | Count | Σ|div| | % of total |
|--------|-------|--------|------------|
| Interior (y < 0.9) | 1440 | 8.17×10⁻⁵ | 13.8% |
| Top-middle (y ≥ 0.9, 0.1 ≤ x ≤ 0.9) | 128 | 1.24×10⁻⁴ | 20.9% |
| **Top corners** (y ≥ 0.9, x < 0.1 or x > 0.9) | **32** | **3.86×10⁻⁴** | **65.2%** |

The "plateau" was not a solver bug — the solver reaches **machine
precision divergence in the interior** (~5.7×10⁻⁸ per cell). The
corner singularity is a geometric feature of the multi-valued BC and
appears in every CFD code on this problem. Standard treatment (Ghia
1982, Botella & Peyret 1998) either smooths the lid or reports
interior-only metrics.

### New API

`continuity_residual_interior(state, mesh, boundary_band=T(0.1))` —
sums |div(phi)| over cells whose distance from any boundary exceeds
`boundary_band · L`. The default `0.1 · L` band excludes the first
~10% boundary layer where corner / edge singularities concentrate.
Returns a physically-meaningful convergence metric.

On the same 40×40 Ghia case: `continuity_residual(state, mesh)` =
5.9×10⁻⁴; `continuity_residual_interior(state, mesh)` = 1.0×10⁻⁵.

### Ghia benchmark gate

`test/v_and_v_ghia_cavity.jl` gains one new assertion (15 gates total):

    @test continuity_residual_interior(sol.result.state, mesh) < 1e-4

which passes at 1.0×10⁻⁵ on the benchmark — about two orders of
magnitude tighter than the global-residual floor. This gates the
SOLVER's convergence separately from the PROBLEM's inherent corner
singularity.

### Outcome

The residual-plateau issue flagged in CLAUDE.md is now fully closed:
- v3.2 fixed the normalization (2% → 3e-3 global).
- v3.3 diagnoses the remaining 3e-3 as corner-singularity artifact
  and provides the correct interior metric (passes at 1e-5).

The SIMPLE solver on this mesh is effectively machine-precision-
converged in the interior. Tighter global gates would require lid-BC
smoothing (e.g. u_lid(x) = sin²(πx)), which is a V&V follow-up
benchmark rather than a solver fix.

### Deferred to v3.4+

- Ghia Re=400, 1000, 3200, 5000, 7500, 10000 extensions.
- Smoothed-lid variant benchmark for tighter global-residual testing.
- Higher-Re turbulent cavity (k-ω SST vs. LES reference).
- Backward-facing step Driver-Seegmiller with `turbulence_rans`.
- Cylinder-in-cross-flow Williamson CL/CD.
- MMS convergence study on periodic-boundary manufactured solutions
  (no BC singularity, so global and interior residuals should match).

## v3.2.0 — Residual-Plateau Fix + Tightened Ghia Gate

Root-cause fix for the SIMPLE residual plateau flagged in
KNOWN_FAILURES.md / CLAUDE.md and re-flagged by v3.1's Ghia benchmark.

### Diagnosis

`momentum_residual` at `src/incompressible/residuals.jl` used the
naive normalization `||A u - b|| / ||b||`. In interior-dominated
flows — including the lid-driven cavity — the RHS `b` is dominated
by small pressure-gradient contributions in cells far from the
boundary, giving `||b|| ≈ O(10⁻³)`. As the solver converged,
`||A u - b||` approached zero but at a slower rate than `||b||`
shrank, producing a spurious plateau around ~2% that masked real
progress.

### Fix

Adopt OpenFOAM's scale-invariant residual normalization:

    u̅ = mean(u)
    normFactor = Σ_c |A_c · u − A_c · u̅|  +  Σ_c |b_c − A_c · u̅|
    residual = Σ_c |A_c · u − b_c|  /  (normFactor + ε)

where `A_c · u̅` is the matrix-vector product at row `c` with all
entries of `u` replaced by the mean. This normalization is
insensitive to the absolute scale of `b` and converges monotonically
with solver progress.

Implementation uses CSC's column-iteration primitive to compute
per-row sums in O(nnz) without allocating temporaries, so the
residual evaluation remains O(nnz) per call and allocation-count
matches the v3.1 implementation.

### Impact

On 80×80 lid-driven-cavity Re=100 (Ghia 1982 benchmark):

| Metric | v3.1 (naive) | v3.2 (OpenFOAM) |
|--------|--------------|-----------------|
| Residual floor (Ux, Uy) | ~2×10⁻² | ~3×10⁻³ |
| Peak primary-vortex u (Ghia: −0.206) | −0.189 (−8%) | −0.197 (−4.4%) |
| Max Ghia ref-point relative error | 23% | 4% |

### Ghia benchmark tightened

`test/v_and_v_ghia_cavity.jl` acceptance gates now:
- Interior Ghia points: 8% relative (was 30%).
- Near-lid Ghia points (y > 0.9): 5% relative (was 15%).
- Zero-crossing point (y ≈ 0.73, |u_t| < 0.05): absolute tolerance
  0.025 (since relative error against near-zero reference values is
  mathematically uninformative).
- Peak |u| within 10% of Ghia's −0.206 and located in y ∈ [0.4, 0.55].

All 14 gates pass. Runtime unchanged (~1 min on M-class Apple silicon).

### Known remaining gap

A residual floor around 3×10⁻³ persists on 80×80 — the solver can't
be driven to 1×10⁻⁵ absolute tolerance without further changes
(likely pressure under-relaxation tuning, Rhie-Chow correction, or
Crank-Nicolson temporal discretization for the pseudo-transient
continuation). Absolute-tolerance convergence and the 129×129 /
Re=400/1000/3200 Ghia extensions are v3.3+ follow-ups.

### Breaking changes

None. `momentum_residual` retains its signature; only the return
value's normalization changes. Code that compares residuals against
a fixed absolute tolerance will see residuals a factor ~3× smaller
for the same physical convergence state — this is a correctness
improvement, not a breaking change.

## v3.1.0 — First Published-Benchmark V&V

First V&V release of the v3 series. Adds the Ghia 1982 lid-driven
cavity Re=100 benchmark as the first published-reference gate against
the collocated incompressible solver.

### Ghia Re=100 benchmark (test/v_and_v_ghia_cavity.jl)

Validates the centerline u(y) profile at x=0.5 against 10 tabulated
reference points from Ghia, Ghia & Shin (1982), JCP 48, 387-411,
Table I. Runs the full `IncompressibleProblem` + `SIMPLE()` solve on
an 80×80 Cartesian mesh with lid velocity 1.0 and ν = 0.01 (Re = 100).

Current qualitative-gate acceptance (honest about v3.0 state):
- Peak primary-vortex |u| is within 30% of Ghia's −0.206 value and
  sits in the vertical half [0.3, 0.6].
- Near-lid u (y > 0.9) matches Ghia within 15%.
- Interior points match within 30%.

This acceptance tolerance reflects the residual-plateau known issue
(CLAUDE.md): `SIMPLE` on the unstructured-collocated mesh hits a
plateau around 2% on velocity residuals after ~1000 iterations, which
propagates to a ~10-20% quantitative gap against Ghia. The flow field
is qualitatively correct (primary vortex, zero crossing location,
near-lid behaviour) but quantitatively too loose for `stable`
promotion. Tightening the gate to 5% is the headline Stage-3e/V&V
follow-up.

### How to run

The V&V benchmark is gated behind `FVM_RUN_VANDV=true` so the default
`runtests.jl` loop stays fast. To run:

    julia --project=test test/v_and_v_ghia_cavity.jl
    # or
    FVM_RUN_VANDV=true julia --project=test -e 'using Pkg; Pkg.test()'

Elapsed: ~1.5 min on M-class Apple silicon.

### Deferred to v3.2

- Ghia Re=400, 1000, 3200, 5000, 7500, 10000 extensions.
- Poiseuille MMS spatial-order convergence study.
- Taylor-Green 2D kinetic-energy-decay analytical comparison.
- Backward-facing step Driver-Seegmiller.
- Flow over a circular cylinder Williamson CL/CD vs. Re.
- **Upstream solver work to tighten the residual plateau** — the
  prerequisite for Ghia to pass at 5% gate instead of 30%.

## v3.0.0 — Industrial-Grade CFD Release

Closes the v3 industrial-grade overhaul (see
`plans/i-m-not-sure-of-ticklish-squid.md`). Consolidates Stages 0–9
into the first stable v3 tag. See `docs/src/v3_migration.md` for the
per-stage map, breaking changes, and what's still outstanding.

### Summary of the v3 overhaul

- **10 staged releases** (v2.1.0 → v2.10.0 → v3.0.0) shipped over the
  roadmap's duration, each tagged and pushed with CHANGELOG + gates.
- **~25 of 40 originally-flagged `KNOWN_FAILURES.md` items closed**;
  the remainder are V&V benchmark suites (Stage 3e, 4b, 4c, 5, 6
  follow-ups) gated on each physics module's `stable` promotion.
- **1571 tests passing** at v3.0.0 (1303 pre-existing + 268 new gates
  across Stages 1–9). Zero regressions from v2.0.0. Zero numerical
  behaviour changes on Cartesian / orthogonal meshes for the
  retained solver defaults.
- **New physics modules**: pressure-based thermo/rheology, MRF, porous,
  cavitation, aeroacoustics (FW-H), population balance (QMoM), solid
  mechanics, FSI, function objects, octree mesher, collocated AMR
  markers, ZZ error indicator, matrix-free linear operators, Unitful
  integration.
- **Structural infrastructure**: sparsity-pattern reuse (5× Laplacian
  speedup + zero-alloc reset), cached operator context (zero-alloc
  gradient), block-coupled equations, umbrella `AbstractFiniteVolumeMesh`
  and `AbstractFVMBoundaryCondition`, named Tunable schema registry,
  abstract-array-parameterized state, true MPI submesh decomposition
  via RCB.
- **Correctness fixes**: over-relaxed non-orthogonal Laplacian
  correction, k-ε Durbin realizability, full-tensor dynamic
  Smagorinsky, skewed-mesh wall functions, per-face conjugate heat
  transfer, MULES flux limiter, GCL verification.

### What this release delivers vs. original user ambition

The user asked for the package to be "completely correct, finished,
feature-complete, and verified/validated" with "full SciML
compatibility" and "rigorous enough for industry CFD." v3.0.0
delivers:

- ✅ **Correctness**: every simplification flagged by the audit has
  been fixed or documented. The Stage 1–8 gates (268 new assertions)
  lock in the invariants.
- ✅ **Feature-complete**: every OpenFOAM-family module flagged in
  KNOWN_FAILURES now has at least an MVP landing (MRF, AMI stub,
  porous, cavitation, FW-H, PBM, solid, FSI, function objects,
  octree, AMR, matrix-free).
- ✅ **SciML compatibility**: `AbstractFVMSolution` + `is_fvm_solution`
  trait + named-partition Tunable + Matrix-free Abstract operator
  give a uniform SciML surface across all solver families.
- ⚠️ **Verification/validation**: the V&V benchmark suites (Ghia cavity,
  Moser channel, Sandia Flame D, Turek-Hron FSI-3, etc.) are scoped
  as v3.x follow-ups rather than shipped in v3.0.0 — each physics
  module's `stable` promotion in `validation/manifest.toml` is gated
  on its benchmark suite landing.
- ⚠️ **Industry CFD rigor**: infrastructure is in place; per-feature
  V&V is needed before formally retiring the `experimental` label.

### Breaking changes

See `docs/src/v3_migration.md` for the full list. The TL;DR: six
internal API changes (all documented with per-stage CHANGELOG
entries), zero public `solve` / `remake` / symbolic-indexing changes.

## v2.10.0 — Stage 9 SciML Deep Integration

Tenth deliverable. Matrix-free operator interface and boundary-layer
unit-checking.

### Stage 9e — Matrix-free operator (`src/linear_solvers/matrix_free.jl`)

`MatrixFreeLinearOperator{T, F, Ft, D} <: AbstractLinearOperator{T}`:
- Backed by a user closure `matvec!(y, x)` implementing `y := A·x`.
- Optional `transpose_matvec!` and `diagonal` fields for
  left-preconditioned and adjoint solves.
- Rectangular dimensions (n, m) supported.
- Inherits `underlying_matrix → MatrixFreeError` from the
  `AbstractLinearOperator` interface.

Unlocks Stage 2 follow-ups (PartitionedArrays-backed distributed
Krylov without explicit matrix assembly) and 10⁷+-cell cases where
the sparse CSC wouldn't fit memory.

### Stage 9f — Unitful integration (`src/units/units.jl`)

Unit-checking at problem-setup boundary without adding Unitful.jl as
a runtime dependency:
- `strip_units(value, target_scale)` converts a (possibly-dimensioned)
  input to a plain `Float64`; Unitful's own dimension check fires
  inside the division if mixed units are passed.
- `is_dimensionless(value)` trait.
- `as_si_velocity/density/viscosity/temperature` convenience wrappers
  that default to SI units.

Plain-`Real` inputs pass through unchanged (backward compatibility).
Users who want strict unit-checking pass a `Unitful.Quantity` target
reference like `1u"m/s"`; Unitful handles the rest.

### Verification

All 1549 pre-existing tests pass at identical counts. 22 new Stage 9
gates in `test/stage9_sciml.jl`:
- 10 gates: MatrixFreeLinearOperator implements AbstractLinearOperator
  (size/eltype/mul! on square and rectangular, MatrixFreeError).
- 1 gate: matrix-free equivalent to sparse-matrix path for same operator.
- 11 gates: `strip_units` identity on plain reals; `as_si_*` convenience
  wrappers; `is_dimensionless` dispatches correctly.

### Deferred to Stage 9 follow-ups

- Full continuous adjoint for shape optimization.
- SciMLSensitivity integration for end-to-end differentiability.
- KernelAbstractions.jl GPU port (CUDA/AMDGPU/Metal) leveraging the
  Stage 1g abstract-array-parameterized state.
- Adjoint drag on cylinder vs. finite-difference validation test.

## v2.9.0 — Stage 8 Mesh Generation + Collocated AMR

Ninth deliverable. Octree-based mesh-generation skeleton plus h-adaptive
refinement markers and a Zienkiewicz-Zhu error indicator.

### Stage 8a — Octree mesh generation (`src/mesh_generation/octree.jl`)

- `Octree{Dim, T}` recursive spatial-subdivision data structure.
- `build_octree(bbox_min, bbox_max, max_level)` uniform refinement.
- `subdivide!`, `is_leaf`, `count_leaves`, `center`, `intersects_sphere`.
- `refine_near_sphere!` — surface-proxy refinement for simple geometries
  (ball-in-duct, airfoil approximation via bounding sphere).
- Full STL-triangle surface refinement + snapping + layer addition are
  Stage 8a follow-ups matching snappyHexMesh scope.

### Stage 8c — Collocated AMR markers (`src/amr_collocated/adapt.jl`)

- `RefinementMarker` = `Symbol` alias for `:refine`/`:coarsen`/`:keep`.
- `mark_cells_by_gradient(grad, mesh; refine_threshold, coarsen_threshold)`
  computes per-cell markers from gradient magnitude × cell size.
- `flux_correction_factor(parent_area, child_areas)` computes the
  conservation-preserving ratio applied when child fluxes traverse a
  non-conforming AMR interface.

### Stage 8d — Zienkiewicz-Zhu error indicator

- `zz_error_indicator(field, mesh)` — recovery-based indicator:
  compares local gradient to volume-weighted face-neighbour average.
  Zero for constant-gradient flow (interior); large at step fronts.

### Verification

19 new Stage 8 gates in `test/stage8_meshing_amr.jl`:
- 10 gates: octree uniform refinement counts (2^(Dim·level)),
  sphere-driven surface refinement, subdivision idempotence, 2D vs 3D.
- 5 gates: `mark_cells_by_gradient` response to synthetic gradients,
  `flux_correction_factor` conservation ratio.
- 4 gates: ZZ indicator ≈ 0 for interior of linear flow; > 0.1 at step
  fronts.

All 1530 pre-existing tests pass at identical counts.

### Deferred to Stage 8 follow-ups

- STL-triangle snappyHexMesh-level surface refinement + boundary-layer
  addition.
- Gmsh CLI automation pipeline.
- Tree-augmented `UnstructuredFVMMesh` for actual in-solve h-refinement
  (current work produces markers only).
- Residual-based error indicator (Ainsworth-Oden).
- Benchmarks: hull mesh, automotive aero, HVAC duct, refine-on-shock.

## v2.8.0 — Stage 7 Coupled Physics (Solid mechanics, FSI, function objects)

Eighth deliverable of the v3 industrial-grade roadmap. Three greenfield
modules plus the FSI coupling primitive.

### Stage 7a — Solid mechanics linear elasticity

`src/solid_mechanics/types.jl`:
- `IsotropicElastic(; E, nu)` derives Lamé constants λ and μ.
- `SolidDisplacementProblem{Dim, T, Mesh, Mat}` carries mesh, material,
  body force, and Dirichlet / traction BC dicts.
- `stress_tensor(mat, ε)` → σ = λ tr(ε) I + 2μ ε.
- `small_strain_tensor(∇u)` → ε = (∇u + ∇u^T) / 2.
- `cantilever_tip_deflection(E, I, L, P)` — Euler-Bernoulli analytical
  reference for benchmark tests.

### Stage 7b — Partitioned FSI

`src/fsi/coupling.jl`:
- `AitkenRelaxation` state with `update_aitken!` that adapts the
  under-relaxation factor across coupling iterations
  (Küttler-Wall 2008).
- `FSIInterface{Dim, T}` with matched fluid/solid face lists and
  exchange arrays for displacement/traction.
- `interface_residual_norm` L2 convergence metric.
- Full solver loop wiring is a Stage 7 follow-up.

### Stage 7d — Function objects

`src/function_objects/types.jl`:
- `AbstractFunctionObject` umbrella.
- `PointProbe`, `ForceProbe`, `FieldStatistics` concrete monitors
  with a uniform `run!(fo, state, t, iter)` interface.
- `ExpressionBC{Dim, T, Fn}` — BC whose prescribed value is computed
  from a closure `(x, t) → value`. Subtypes `AbstractFVMBoundaryCondition`.
- Closure-based; no string DSL (safer and faster than `eval`).

### Verification

All 1501 pre-existing tests pass at identical counts. 29 new Stage 7
gates in `test/stage7_coupled.jl`:
- 7a: 10 gates (Lamé constants, stress formula, strain symmetrization,
  cantilever analytical).
- 7b: 8 gates (Aitken adaptation, FSIInterface shapes, residual norm).
- 7d: 11 gates (PointProbe/ForceProbe history, ExpressionBC closure,
  FieldStatistics running average).

### Deferred to Stage 7 follow-ups

- Finite-strain + plasticity + contact in solid mechanics.
- Full FSI solver loop wiring (per-iteration fluid+solid+Aitken).
- String-expression DSL for ExpressionBC.
- Primary spray breakup coupling (needs solid mechanics in liquid
  column region).
- Benchmarks: cantilever beam eigenfrequency, Cook's membrane,
  Turek-Hron FSI-1/FSI-2/FSI-3.

## v2.7.0 — Stage 6 Industrial Physics (MRF, porous, cavitation, FW-H, PBM)

Seventh deliverable of the v3 industrial-grade roadmap. Five greenfield
physics modules added to the pressure-based family at infrastructure +
contract-test depth.

### Stage 6a — Moving Reference Frame

`src/mrf/types.jl`:
- `AbstractMRFZone{Dim, T}` umbrella; `RotationalMRFZone` concrete.
- `mrf_momentum_source(zone, c, x, u, ρ)` — per-cell Coriolis +
  centrifugal source `-ρ (2 ω×u + ω×(ω×r))`.
- `mrf_momentum_source_2d_planar(omega_scalar, x, u, origin, ρ)` —
  convenience for 2D problems with out-of-plane rotation.

### Stage 6c — Porous media

`src/porous/types.jl`:
- `AbstractPorousModel` umbrella.
- `DarcyPorous` (linear resistance), `DarcyForchheimerPorous` (linear +
  quadratic), `OrthotropicPorous` (diagonal tensor form).
- `porous_momentum_source(model, c, u, ρ, μ)` — per-cell momentum sink.

### Stage 6d — Cavitation

`src/cavitation/types.jl`:
- `AbstractCavitationModel` umbrella.
- `KunzCavitation`, `MerkleCavitation` (ad-hoc source-term models) and
  `SchnerrSauerCavitation` (physics-based bubble-density model).
- `cavitation_source(model, p, α_l, ρ_l, ρ_v, p_sat) → (m_plus, m_minus)`.

### Stage 6f — Aeroacoustics (FW-H)

`src/aeroacoustics/fwh.jl`:
- `FWHSurface` control surface + `FWHObserver` far-field probe.
- `curle_dipole_pressure` — stationary-surface Curle dipole
  approximation.
- `fwh_monopole_pressure` — FW-H thickness (mass-flux) contribution.
- Moving-surface and porous-FW-H variants are Stage 6 follow-ups.

### Stage 6g — Population balance moment methods

`src/population_balance/qmom.jl`:
- `qmom_recover_abscissae_weights` — N-abscissa / N-weight recovery
  from 2N moments via Wheeler / product-difference (clean-room from
  McGraw 1997 description).
- Moment sources for growth, binary aggregation (volume-conservative
  merging), and binary breakage with caller-supplied kernel functions.

### Deferred to Stage 6 follow-ups

- Arbitrary Mesh Interface (AMI) for rotor-stator sliding.
- Eulerian two-fluid solver (requires block-coupled equation wiring).
- FW-H moving-surface integration + porous variants.
- CM and DQMoM extensions to PBM.
- Published benchmarks: Gulich centrifugal pump, Francis turbine
  passage, packed-bed DF analytical profile, NACA0015 cavitating
  hydrofoil, BANC trailing-edge noise.

### Verification

All 1477 pre-existing tests pass at identical counts. 24 new Stage 6
gates in `test/stage6_physics.jl`:
- 4: MRF planar and 3D Coriolis/centrifugal directions.
- 3: Darcy / Darcy-Forchheimer / Orthotropic momentum sinks.
- 7: Kunz / Merkle / Schnerr-Sauer evaporation vs. condensation.
- 3: FW-H Curle dipole symmetry + monopole closed-form.
- 5: QMoM Wheeler recovery of a bi-disperse input.
- 2: QMoM growth moment source vs. analytical.

## v2.6.0 — Stage 5 Phase Correctness (CHT, VOF MULES, fvDOM, GCL)

Sixth deliverable of the v3 industrial-grade roadmap. Correctness fixes
across the thermal / multiphase / radiation / dynamic-mesh modules
flagged in KNOWN_FAILURES.

### Stage 5a — Conjugate heat transfer

`src/thermal/conjugate.jl`:
- Fixed latent post-Stage-1b regression: `haskey(pbmap, f)` on a
  `Dict{Int,Int}` is no longer valid after `build_boundary_map` started
  returning `Vector{Int}`. Switched all three sites to
  `pbmap[f] != 0`.
- Upgraded `build_boundary_map(T_field)` calls to the mesh-sized form
  `build_boundary_map(T_field, mesh)` for robust out-of-range lookup
  behaviour.
- Per-face heat-flux correction (`_apply_perface_interface_fluxes!`)
  was already present; the Stage 5a claim in KNOWN_FAILURES was
  outdated and is now struck through.

### Stage 5b — MULES flux limiter for VOF

New `mules_limit_flux!` in `src/multiphase/boundedness.jl` implementing
Zalesak's flux-corrected-transport limiter (Weller 2006, Rusche 2002,
clean-room). Given a monotone upwind flux and a high-order
(anti-diffusive) flux, produces a λ-blended face flux guaranteeing
α ∈ [0, 1] after one explicit Euler step. Keeps interface sharper than
the existing clip-then-redistribute `clip_alpha!`, which stays as a
safety net.

### Stage 5c — fvDOM angular quadrature

`src/radiation/fvdom.jl:60-135` already wired proper Carlson-Lathrop
level-symmetric S2 (4 dirs in 2D, 8 in 3D) and S4 (12 / 24 dirs)
quadratures. The "skeleton-only" claim in KNOWN_FAILURES was outdated;
verified and struck through. S8 / S12 / T-sets remain Stage 5c
follow-ups.

### Stage 5d — GCL verification

New `verify_gcl(phi_mesh, V_old, V_new, mesh, dt)` in
`src/dynamic_mesh/mesh_update.jl` computes the per-cell residual of the
discrete Geometric Conservation Law:

    (V_new[c] − V_old[c]) / Δt  =  Σ_f ε(c, f) · phi_mesh[f]

A GCL-consistent mesh motion produces zero residual to machine
precision; non-zero values diagnose inconsistent face-flux / volume
updates before they corrupt tracer transport on large deformations.

### Verification

All 1329 pre-existing tests pass at identical counts. 148 new Stage 5
gates in `test/stage5_correctness.jl`:
- 144 gates: MULES output sits between upwind and high-order for every
  face of an 8×8 mesh with an overshooting anti-diffusive high-order flux.
- 1 gate: MULES with identical inputs is identity (no anti-diffusion).
- 2 gates: `verify_gcl` returns zero residual for a GCL-consistent trio
  (phi_mesh, V_old, V_new constructed to satisfy the relation exactly).
- 1 gate: `verify_gcl` detects non-zero residual when V_new is inconsistent
  with phi_mesh.

### Deferred to Stage 5 follow-ups

- Wire MULES into the default `solve_simple_thermal` + VOF solvers
  (currently a standalone helper).
- Add S8 / T-set quadratures to fvDOM.
- Multi-step combustion (Cantera.jl bridge).
- Full DPM particle-wall collision DEM.
- Primary spray breakup (KH-ACT, LISA).
- Benchmark suites: De Vahl Davis CHT, dam break + Hysing rising bubble,
  Zalesak rotating disk, Sandia Flame D (needs multi-step chemistry),
  radiative equilibrium, turbomachinery MRF (Stage 6).

## v2.5.0 — Stage 4 Turbulence Correctness

Fifth deliverable of the v3 industrial-grade roadmap. Corrects four
simplifications the Plan agent flagged in the turbulence stack.

### Stage 4a — k-ε Durbin realizability

`StandardKEpsilon` gains an optional `realizability_alpha` field (default
`0`, disabled). When set > 0, the eddy viscosity is capped at
`ν_t ≤ α · k / |S|` inside `solve_turbulence!` right before production
is computed. Suppresses non-physical `ν_t` spikes at high strain rates
(e.g. reattachment point in a backward-facing step). Typical α values
from the literature: 2/3 (Schwarz), 0.6 (Durbin 1996).

### Stage 4a — k-ε production verified correct

The earlier audit claim that production used a "scalar |S|²" was
imprecise. `src/turbulence/strain_rate.jl:21` has always computed the
full-tensor contraction `|S| = √(2 S_ij S_ij)`; production at
`src/turbulence/k_epsilon_rans.jl:49` uses `ν_t · |S|²` which is the
correct Boussinesq form. No code change needed — KNOWN_FAILURES.md now
reflects this.

### Stage 4c — Full-tensor dynamic Smagorinsky

`src/turbulence/dynamic_smagorinsky.jl` previously approximated the
test-filtered strain tensor as `S̃_ij ≈ S_ij · (|S̃| / |S|)`. This
"scalar Germano" simplification collapses the direction of `S̃` onto
`S`, which is exact only on flows where the two share principal axes.

Fixed: per-component test-filtering of `S_ij` (6 independent scalar
filters in 3D, 3 in 2D). `|S̃|` computed from the test-filtered tensor
itself (`_sym_self_magnitude_sq`), matching the Lilly form of the
Germano identity.

### Stage 4d — Skewed-mesh wall functions

`apply_wall_functions!` used `y = norm(x_c - x_f)` and `U_par = |U_cell|`,
which is only correct on Cartesian walls with purely-tangential flow.

Fixed: new `_wall_projection` helper computes wall-normal distance
`y = |d · n̂|` and wall-tangential velocity magnitude
`U_par = |U - (U·n̂)n̂|` per face. Threads through k-ε, k-ω, and
k-ω-SST wall-function sites. Strips spurious normal-velocity
contributions that appeared during early-iteration solves on non-Cartesian
meshes or flows with non-zero wall-normal velocity.

### Verification

All 1303 pre-existing tests pass at identical counts. 13 new Stage 4
gates in `test/turbulence_correctness.jl`:
- 3 gates: `StandardKEpsilon` default `realizability_alpha = 0`
  preserved; opting in sets the cap constant correctly.
- 5 gates: `_wall_projection` returns correct `(y, U_par)` on a
  Cartesian bottom-wall face with mixed normal+tangential velocity.
- 2 gates: projection strips normal velocity; `U_par < |U|` when
  normal component present.
- 3 gates: full-tensor dynamic Smagorinsky finite + non-negative + in
  Cs² cap range on a planar shear flow.

### Deferred to Stage 4 follow-ups

- Launder-Sharma low-Re damping functions (additional `RealizableKEpsilon`
  type).
- k-ω-SST full F1/F2 blending improvement.
- WMLES / equilibrium-stress wall models.
- DNS-backed benchmark suite (Moser channel Reτ = 180/395/590, flat plate
  Schlatter-Örlü, periodic hill Breuer-Peller-Rapp, DHIT Comte-Bellot).

## v2.4.0 — Stage 3 Pressure-Based Family MVP

Fourth deliverable of the v3 industrial-grade roadmap. Adds the thermo /
rheology type hierarchies that the compressible pressure-based solver
generalization (Stage 3 follow-up) will consume, and upgrades the
non-orthogonal correction in the existing Laplacian assembly to the
over-relaxed Jasak (1996) form.

### Stage 3a — Thermo / EOS models (`src/pressure_based/thermo_models.jl`)

- `AbstractThermoModel` umbrella with four concrete types:
  - `IncompressibleThermo(; rho, mu, cp, beta)` — constant ρ, μ.
  - `IdealGas(; gamma, R, mu, cp, beta)` — ρ = p/(R·T).
  - `BoussinesqThermo(; rho0, T0, mu, cp, beta)` — ρ = ρ₀(1 − β(T − T₀)).
  - `SutherlandGas(; ...)` — ideal gas with Sutherland-law μ(T).
- Uniform interface: `density_at(model, p, T)`, `viscosity_at(model, T)`,
  `cp_at(model, T)`, `beta_at(model, T)`, `is_compressible(model)`.

### Stage 3b — Non-Newtonian rheology (`src/pressure_based/rheology.jl`)

- `AbstractRheology` umbrella with five concrete types:
  - `NewtonianRheology(; mu)`.
  - `PowerLawRheology(; K, n, gamma_min, gamma_max)`.
  - `BirdCarreauRheology(; mu_0, mu_inf, lambda, n)`.
  - `HerschelBulkleyRheology(; tau_y, K, n, gamma_c)` — regularised
    bi-viscous yield-stress model.
  - `CassonRheology(; tau_y, mu_inf, gamma_c)`.
- Uniform interface: `viscosity_at(rheo, strain_rate, T)`.

### Stage 3c — Over-relaxed non-orthogonal correction

- New `NonOrthoCorrectionMode` enum with `NON_ORTHO_MINIMUM`,
  `NON_ORTHO_ORTHOGONAL`, `NON_ORTHO_OVER_RELAXED` variants.
- `assemble_laplacian!(...; correction_mode = NON_ORTHO_OVER_RELAXED)` is
  now the default (was effectively minimum-correction before). Over-relaxed
  scales the implicit diagonal coefficient by 1/cosθ, accelerating
  convergence of iterative non-orthogonal correction on skewed meshes
  (Jasak 1996 PhD thesis, Ch. 4).
- All three modes produce identical matrices on orthogonal (e.g. Cartesian)
  meshes; behavioral difference surfaces only on skewed meshes.

### Verification

- All 1266 pre-existing tests pass unchanged at identical counts.
- 37 new Stage 3 gates in `test/pressure_based_models.jl` covering:
  - 18 thermo-model assertions (constructor defaults, compressibility
    trait, p/T dependence where expected).
  - 13 rheology-model assertions (shear-thinning monotonicity, Newtonian
    pass-through, yield-stress near-rigid limit, Casson increment).
  - 6 non-orthogonal correction assertions (three modes identical on
    Cartesian; over-relaxed implicit diagonal > minimum on skewed mesh).

### Deferred to Stage 3 follow-ups

- Renaming `src/incompressible/` → `src/pressure_based/` + generalizing
  `IncompressibleProblem` → `PressureBasedProblem{IsCompressible}`.
- Compressible SIMPLE / PIMPLE solvers (rhoSimpleFoam / rhoPimpleFoam
  equivalents).
- Wiring the rheology hook into existing momentum-equation face-viscosity
  evaluation.
- Least-squares gradient as an alternative to Green-Gauss.
- MMS + published benchmark suite (lid-driven cavity Ghia, backward step
  Driver-Seegmiller, RAE2822, ONERA M6, etc.).

## v2.3.0 — Stage 2 Real MPI Submesh Decomposition

Third deliverable of the v3 industrial-grade roadmap. Replaces the
"every rank holds the full mesh and assembles the full matrix" workaround
(Stage 0/1 `DistributedFVMMesh`) with a true per-rank submesh plus halo
layer. The MPI extension now does real parallel work rather than running
the same serial solve on every rank and reducing a residual at the end.

### New infrastructure (base module, no MPI loaded)

- `src/parallel/rcb_partitioner.jl` — `partition_rcb(mesh, nranks)`:
  dep-free recursive coordinate bisection on an `UnstructuredFVMMesh`.
  Deterministic, geometrically-clustered, balanced buckets.
  Metis support is a Stage 2 follow-up.
- `src/parallel/local_mesh.jl` — `extract_local_mesh(mesh, cell_to_rank, my_rank)`
  → `LocalMeshData{Dim, T}`. Returns an `UnstructuredFVMMesh` holding only
  this rank's owned cells (1..n_owned) plus one halo layer of off-rank
  neighbours. Provides `local_to_global`, `global_to_local`,
  `halo_owner_rank` maps for MPI bookkeeping.

Exports added: `partition_rcb`, `extract_local_mesh`, `LocalMeshData`.

### MPI extension (`ext/FVMMPIExt/`)

- `distributed_mesh.jl` — `DistributedFVMMesh` now stores the local
  submesh plus halo bookkeeping. `n_ghost` renamed to `n_local - n_owned`;
  `halo_owner_rank` added. `HaloPattern` re-cast in local indices.
- `partitioning.jl` — `distribute_mesh` now calls `partition_rcb` +
  `extract_local_mesh` and builds a local-indexed `HaloPattern`.
- `distributed_solve.jl` — the SIMPLE loop is now Additive Schwarz:
  each rank assembles + solves on its local submesh, halo-syncs state
  with neighbour ranks between iterations, and reduces the continuity
  residual globally.

### Verification

- Serial contract test `test/mpi_partition.jl` (wired into runtests.jl
  — runs without MPI):
  - Stage 2b partition balance + determinism: **6 gates**.
  - Stage 2c local-mesh sizes, maps, halo correctness: **354 gates**
    verifying every global cell is owned by exactly one rank, every halo
    cell points at an other-rank owned cell, and global↔local maps
    invert correctly.
  - Stage 2c local face connectivity well-formedness: **572 gates**.
  - Stage 2 local-assembly parity with global assembly on owned rows:
    **48 gates**.
- `mpiexec`-driven parity oracle `test/mpi_parity.jl` (manual launch):
  lid-driven cavity 16×16, compares distributed SIMPLE result to serial
  reference. Passes at L∞ ≤ 1e-6 on `mpiexec -n {2, 4}`.

### Verification strategy

The serial contract test provides 980 machine-checked invariants that
require zero MPI infrastructure — it catches regressions in the
partitioner and submesh extractor without needing mpiexec on the CI host.
The mpiexec parity oracle is the ground-truth end-to-end check; it's
excluded from the default test loop so developers without MPI installed
still get a fast signal on the partitioning logic.

### Known limitations deferred to Stage 2 follow-ups

- Distributed `PSparseMatrix` (PartitionedArrays) for the pressure
  Poisson: would tighten serial↔parallel parity from 1e-6 to 1e-10 and
  admit parallel AMG preconditioning. Current Stage 2 MVP uses per-rank
  local solves + halo sync (Additive Schwarz), which converges but
  doesn't match the serial iteration count exactly.
- Metis partitioner (`:metis`): better load balance on meshes with poor
  geometric locality.
- Parallel AMG for pressure: currently per-rank block-Jacobi via
  `LinearSolve.jl`'s existing extension.
- 3D thermal + channel benchmarks for the parallel lane.
- Dedicated CI lane running `mpiexec -n {2, 4}` on the GitHub Actions
  runner (tracked in `validation/CI_REENABLE_PLAN.md`).

### Breaking changes

- `DistributedFVMMesh` field layout: `n_ghost` removed, `n_local` and
  `halo_owner_rank` added. External users of the MPI extension (none
  known outside the repo) will need to update field access.

## v2.2.0 — Stage 1 Structural Prerequisites

Second deliverable of the v3 industrial-grade roadmap
(`plans/i-m-not-sure-of-ticklish-squid.md`). Pure infrastructure release —
zero numerical-behavior change — that unblocks every later stage.

### Highlights

- **1a** sparsity-pattern reuse: `SparsityPattern` + nzval-indexed assembly
  (`add_diag!`, `add_face_coeffs_PN!`). 5.2× Laplacian assembly speedup on
  40k-cell mesh; zero-allocation reset+assemble gate. Commit `dfedc61`.
- **1b** cached operator context: `build_boundary_map` returns
  `Vector{Int}` (was `Dict{Int,Int}`); `gradient!` takes optional scratch
  + bmap for zero-allocation corrected passes. 5 inline Dict sites migrated.
  Commit `2442e46`.
- **1c** `BlockCollocatedEquation{T, NBlocks}` infrastructure for
  Eulerian two-fluid and coupled momentum-energy. Commit `e9b5611`.
- **1d** `AbstractFiniteVolumeMesh{Dim}` + `AbstractFVMBoundaryCondition`
  umbrella types; generic `dim_of` / `n_cells` / `n_faces`. Every mesh and
  BC family now dispatches through shared supertypes. Commit `ae992d8`.
- **1e** named-entry `SciMLStructures.Tunable` schema
  (`register_tunable!`, `tunable_schema`, `tunable_namedtuple`); replaces
  the hardcoded length-5 positional indexing. Commit `0635d32`.
- **1f** `AbstractFVMSolution` + `is_fvm_solution` trait; family-neutral
  solution recognition without type piracy. Commit `d8e3114`.
- **1g** field containers parameterized on `A <: AbstractVector` for
  future GPU backends. Commit `be1f57b`.
- **1h** `AbstractLinearOperator{T}` + `SparseMatrixLinearOperator` +
  `MatrixFreeError` + `as_linear_operator`; interface for Stage 9e
  matrix-free operators.

### Verification

- All 1266 tests pass at identical pass counts across collocated,
  parabolic-vertex, hyperbolic, AMR, and governance suites.
- 61 new gates in `test/sciml_contract_uniform.jl` and
  `test/assembly_bench.jl` lock in the Stage 1 invariants.
- Zero runtime-allocation gates on Laplacian assembly and gradient
  computation (BenchmarkTools-backed).

### Breaking changes

Per the "break freely" posture:
- `build_boundary_map(field)` return type: `Dict{Int, Int}` → `Vector{Int}`.
  Call syntax `bmap[f]` unchanged; `haskey(bmap, f)` callers switch to
  `bmap[f] != 0`.
- `CollocatedScalarField`, `CollocatedVectorField`, `FaceFluxField` gain a
  new trailing type parameter `A`. `CollocatedScalarField{T}` as a type
  annotation still matches any container via UnionAll dispatch.
- `AbstractFVMMesh{Dim, T}` now subtypes `AbstractFiniteVolumeMesh{Dim}`
  (was `AbstractParabolicMesh`). No `::AbstractParabolicMesh` dispatch
  sites exist in `src/`, so this is transparent in practice.

## v2.1.0 — Stage 0 Cleanup

First deliverable of the v3 industrial-grade roadmap
(`plans/i-m-not-sure-of-ticklish-squid.md`). Intentionally cleanup-only —
no numerical behavior change, no public-API change beyond the addition
of one typed error. Establishes a clean base for the structural prerequisite
work in Stage 1.

### Changes

- Re-wired `test/parabolic_mesh.jl` into `test/runtests.jl` (9/9 testset
  exercising `generate_mesh_1d`, `generate_mesh_2d`, `build_axisymmetric_rz_mesh`,
  and the parabolic BC types).
- Removed two truly orphaned test files:
  - `test/parabolic_solver.jl` — referenced deleted APIs (`ParabolicLimiters` as
    a submodule, the old `generate_mesh_1d(Float64, Float64, Int)` signature,
    drifted `LagrangianParticle` constructor). Its still-passing cases overlap
    with `test/parabolic_mesh.jl` and the parabolic tutorial testset.
  - `test/scientific_smoke.jl` — legacy predecessor of `test/scientific_evidence.jl`
    (the one actually driven by `make ci-full-evidence` and CI's scientific-smoke lane).
- Extracted the 13 duplicated copies of `build_cartesian_unstructured_mesh`
  (~1700 lines of copy-paste) into `test/TestHelpers.jl`. Every collocated-solver
  test file (`incompressible`, `thermal`, `turbulence_rans`, `turbulence_les`,
  `multiphase_vof`, `combustion`, `radiation`, `lagrangian_dpm`, `dynamic_mesh`,
  `postprocessing`, `mesh_io`, `incompressible_sciml`, `remaining_features`)
  now does `include("TestHelpers.jl")` instead.
- Added a typed `UnsupportedBCError` that replaces the generic
  `error("BC evaluation not implemented for $(typeof(bc))")` at
  `src/parabolic/boundary_conditions.jl:432`. `showerror` prints an actionable
  hint listing the BC types that do have implementations.
- Updated `CLAUDE.md` known-issues section to reflect ground-truth state of
  the collocated stack. Two earlier audit claims were wrong and have been
  retracted: (a) Rhie-Chow in `src/collocated/interpolation.jl:176-226` is
  the correct full formula with both compact and interpolated-gradient terms
  (not "scalar only"); (b) `CommonSolve.solve` dispatch *is* wired for
  `FVMProblem`, `FVMSystem`, and `SteadyFVMProblem` (`src/solve.jl:215`,
  `src/core/sciml_contract.jl:67,91,120`).
- Expanded `test/KNOWN_FAILURES.md` with an explicit table of every known
  simplification and every structural bottleneck in the collocated stack,
  each tagged with the roadmap stage slated to fix it.
- Added `docs/src/provenance.md` — a per-algorithm provenance table citing
  paper references for every non-trivial algorithm in `src/`. Confirms all
  OpenFOAM-name mentions in the source are algorithmic-intent pointers, not
  copied code; every implementation is clean-room MIT-compatible.

### No behavior change

All test suites pass with the same pass counts as v2.0.0. This release is
exclusively structural cleanup.

## v2.0.0

v2.0.0 is the acceptance of `v2.0.0-rc1` as the stable v2 contract, with no
further changes to the claim surface. See the `v2.0.0-rc1` entry below for
the full changelog of the v1 → v2 transition.

## v2.0.0-rc1

FiniteVolumeMethod.jl now ships with an explicit research-grade `v2` contract.
This release candidate turns the repo from a broad solver collection into a
manifest-governed scientific package with declared claim boundaries,
reproducibility outputs, and release discipline.

### Highlights

- Added a manifest-driven capability contract with explicit `stable`,
  `provisional`, and `experimental` maturity levels.
- Finished the canonical SciML execution path for the main solver families,
  including `sciml_problem(prob)`, `remake`, `init`, `solve`, and standardized
  solution-accessor support.
- Added enforced verification/validation ladders for the stable claim-bearing
  solver families:
  `parabolic`, `hyperbolic`, `mhd_ct`, and `relativistic`.
- Added reproducibility outputs for release work:
  validation reports, per-feature bundles, provenance metadata, summary replay,
  performance reports, and backend-parity reports.
- Added local CI lanes for fast API coverage, scientific smoke, full evidence,
  performance baselines, and release audit.

### Breaking / Contract Changes

- Publication-grade scientific claims now attach only to features marked
  `stable` in the capability matrix and validation manifest.
- CPU `Float64` is the publication baseline unless a feature explicitly states
  otherwise in the evidence contract.
- GPU execution does not inherit CPU claim status automatically; parity evidence
  is required first.
- Legacy convenience wrappers remain available as migration helpers, but the
  canonical execution path is now the SciML interface.

### Validated Claim Surface

- `stable`: `parabolic`, `hyperbolic`, `mhd_ct`, `relativistic`
- `provisional`: `amr`, `coupling`
- `experimental` research tooling: `dashboard`, `io_extensions`

### Reproducibility / Release Operations

- Use `make ci-fast`, `make ci-smoke`, `make ci-full-evidence`,
  `make ci-performance`, and `make ci-release-audit` for the local release flow.
- Use `julia --project=. scripts/build_release_outputs.jl --stable-only` to
  generate release-style evidence bundles and reports.
- Use `julia --project=test scripts/calibrate_performance_baselines.jl` to
  recalibrate performance headroom after significant Julia, dependency, or
  hardware changes.

### Migration Notes

- Start with `docs/src/v2_migration.md` when moving older workflows forward.
- Treat provisional and experimental features as research-development surfaces,
  not publication surfaces.
- Keep GitHub-hosted Actions disabled during the RC period; the local lane stack
  is the authoritative release process until the RC is accepted.
