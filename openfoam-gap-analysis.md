---
date: 2026-05-27
---

# OpenFOAM Gap Analysis — FiniteVolumeMethod.jl

Status snapshot. Reflects the v3.0 fast-path consolidation through v3.107
and subsequent stabilisation work (current release: v3.111).

## Implemented Phases

| Phase | Feature                        | Status         |
|-------|--------------------------------|----------------|
| 0     | Collocated FVM operators       | Stable         |
| 1     | Incompressible NS (SIMPLE/PISO/PIMPLE) | Stable  |
| 2a    | RANS turbulence (k-omega SST, SA) | Stable      |
| 2b    | LES & hybrid (Smagorinsky, WALE, DDES) | Experimental |
| 3     | Conjugate heat transfer & buoyancy | Stable      |
| 4     | Mesh I/O (Gmsh, OpenFOAM polyMesh) | Stable      |
| 5     | Linear solver infrastructure   | Stable         |
| 7     | Multiphase VOF                 | Experimental   |
| 8     | Combustion & species transport (EDM/EDC) | Experimental |
| 9     | Radiation (P1)                 | Experimental   |
| 10    | Dynamic/ALE mesh               | Experimental   |
| 11    | Lagrangian DPM                 | Experimental   |
| 12    | Post-processing (forces, y+, Q) | Stable        |
| —     | Mesh generation (Gmsh ext)     | Experimental   |
| —     | AMR (residual + Zienkiewicz-Zhu) | Provisional  |
| —     | Eulerian two-fluid (Ishii-Zuber, Gibilaro) | Experimental |
| —     | Solid mechanics + FSI (D-N, Aitken) | Experimental |
| —     | Aeroacoustics (FW-H Curle/Lighthill, PML) | Experimental |
| —     | Population balance (QMoM, DQMoM, Class) | Experimental |
| —     | MRFZone multi-zone Coriolis    | Experimental   |
| —     | Discrete adjoint (steady-SIMPLE, transient-PIMPLE) | Experimental |

## Major Remaining Gaps

### 1. Density-based compressible collocated solvers

OpenFOAM ships `rhoSimpleFoam`, `rhoPimpleFoam`, `sonicFoam`, and
`rhoCentralFoam`. FiniteVolumeMethod.jl has compressible support in
the hyperbolic family (Euler/NS Riemann-based) and supersonic FW-H
post-processing. A density-based pressure-velocity coupling for the
collocated SIMPLE/PIMPLE loop is still missing — this blocks transonic
internal aerodynamics and compressible-RANS workflows.

### 2. Boundary conditions (~85 still unimplemented)

A handful of high-impact BCs and v3.105's runtime expression BCs cover
the common cases. Most-wanted remaining:

| OpenFOAM BC               | Purpose                                  |
|---------------------------|------------------------------------------|
| `turbulentIntensityKineticEnergyInlet` | TI-based k inlet         |
| `turbulentMixingLengthDissipationRateInlet` | ML-based epsilon inlet |
| `kqRWallFunction`         | Wall-function for k                      |
| `omegaWallFunction`       | Wall-function for omega                  |
| `nutUSpaldingWallFunction`| Continuous wall function for nut         |
| `mappedField`             | Inter-region field mapping               |
| `waveTransmissive`        | Non-reflecting acoustic outlet           |

`groovyBC` (swak4Foam expression BCs) and `codedFixedValue` are
covered by the v3.105 runtime expression BC extension.

### 3. snappyHexMesh feature parity

v3.107 ships castellated + snap stages. Layer addition is deferred to
v3.2 (`KNOWN_FAILURES.md` and `plans/open-work.md` C-table).

### 4. Two-fluid cross-coupling completeness

v3.106 ships block-coupled momentum + Ishii-Zuber / Gibilaro drag. Energy
and species cross-coupling on the same block matrix is the v3.2
deliverable; until then, `twoPhaseEulerFoam`-style coupled flows ship as
experimental.

## Minor Gaps

| Feature                          | OpenFOAM solver / equivalent |
|----------------------------------|--------------------------|
| Shallow water                    | (contrib only)           |
| Overset / chimera mesh           | `overSimpleFoam`         |
| Topology-changing dynamic mesh   | `interDyMFoam` (`fvOptions`) |
| Electrochemistry                 | `electrochemicalFoam`    |
| FW-H porous + supersonic regime  | porous + shock corrections deferred |
| Sandia Flame D combustion benchmark | EDC + variable-Lewis + radiation-coupled validation pending |

## SciML Advantages over OpenFOAM

1. **Automatic differentiation** — Forward-mode (ForwardDiff.jl) and
   reverse-mode (Enzyme.jl, Zygote.jl) AD through the entire solver,
   including mesh and BC parameters. OpenFOAM requires hand-coded
   adjoints or finite differences. Steady-SIMPLE and transient-PIMPLE
   discrete adjoints landed v3.105/v3.107; full-solver Enzyme AD is the
   v3.2 deferral.

2. **Sensitivity analysis & parameter estimation** — Composable with
   `SciMLSensitivity.jl` for forward/adjoint sensitivity of any
   ODE-wrapped problem. Enables gradient-based calibration of
   turbulence model constants, inlet profiles, or material properties.

3. **Uncertainty quantification** — Direct integration with
   `PolyChaos.jl`, `Surrogates.jl`, and Bayesian frameworks
   (`Turing.jl`) through the SciML problem interface.

4. **Julia composability** — Any user-defined function (flux, source,
   BC) is a plain Julia callable — no code generation step, no
   string-based `codedFixedValue`. GPU kernels via the v3.105
   KernelAbstractions extension, custom linear algebra, and new
   conservation laws compose without framework modifications.

5. **Time-stepper ecosystem** — Access to 300+ ODE/DAE solvers in
   `OrdinaryDiffEq.jl` with adaptive stepping, implicit-explicit (IMEX)
   schemes, and multirate methods — far beyond OpenFOAM's
   Euler / CrankNicolson / backward.

6. **Reproducibility** — Julia's package manager provides exact
   environment snapshots (`Manifest.toml`); the v3 release-audit lane
   captures provenance + replay reports. OpenFOAM builds depend on
   system compilers, MPI versions, and third-party library paths.
