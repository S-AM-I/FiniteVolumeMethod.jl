# OpenFOAM Gap Analysis — FiniteVolumeMethod.jl

Status snapshot as of 2026-04-06.

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
| 8     | Combustion & species transport | Experimental   |
| 9     | Radiation (P1 model)           | Experimental   |
| 11    | Lagrangian DPM                 | Experimental   |
| 12    | Post-processing (forces, y+, Q) | Stable        |

## Major Remaining Gaps

### 1. Compressible Flow Solvers

OpenFOAM ships `rhoSimpleFoam`, `rhoPimpleFoam`, `sonicFoam`, and
`rhoCentralFoam`.  FiniteVolumeMethod.jl has compressible support only in
the hyperbolic solver family (Euler/NS Riemann-based).  A density-based
pressure-velocity coupling for the collocated SIMPLE/PIMPLE loop is
missing — this blocks transonic internal aerodynamics and compressible
RANS workflows.

### 2. Eulerian–Eulerian Multiphase

`twoPhaseEulerFoam` / `multiphaseEulerFoam` solve separate momentum
equations per phase with inter-phase drag, virtual mass, and lift.  The
current VOF solver (Phase 7) handles immiscible free-surface flows only.
Dispersed bubbly or granular flows are not supported.

### 3. Mesh Generation

OpenFOAM provides `blockMesh` (structured hex), `snappyHexMesh`
(automatic body-fitted hex-dominant), and `cfMesh`.
FiniteVolumeMethod.jl can *read* these meshes (Phase 4) but has no
built-in mesh generator beyond structured Cartesian and simple
unstructured triangulations via DelaunayTriangulation.jl.

### 4. Boundary Conditions (~85 remaining)

10 BCs are implemented.  The most impactful missing ones:

| OpenFOAM BC               | Purpose                                  |
|---------------------------|------------------------------------------|
| `turbulentIntensityKineticEnergyInlet` | TI-based k inlet         |
| `turbulentMixingLengthDissipationRateInlet` | ML-based epsilon inlet |
| `kqRWallFunction`         | Wall-function for k                      |
| `omegaWallFunction`       | Wall-function for omega                  |
| `nutUSpaldingWallFunction`| Continuous wall function for nut         |
| `mappedField`             | Inter-region field mapping               |
| `codedFixedValue`         | User-defined via inline C++              |
| `uniformFixedValue`       | Spatially uniform, time-varying          |
| `waveTransmissive`        | Non-reflecting acoustic outlet           |
| `groovyBC`                | Expression-based (swak4Foam)             |

### 5. Adjoint Optimization

OpenFOAM includes `adjointOptimisationFoam` for shape and topology
optimization.  No adjoint solver exists in FiniteVolumeMethod.jl.
However, SciML's AD ecosystem (see below) can provide discrete adjoints
through `SciMLSensitivity.jl` without a hand-written adjoint solver.

## Minor Gaps

| Feature                          | OpenFOAM solver          |
|----------------------------------|--------------------------|
| Acoustic solver                  | `acousticFoam`           |
| Shallow water                    | (contrib only)           |
| Structural / solid mechanics     | `solidFoam`, `solids4Foam` |
| Overset / chimera mesh           | `overSimpleFoam`         |
| Dynamic mesh (topology changes)  | `interDyMFoam`           |
| Electrochemistry                 | `electrochemicalFoam`    |

## SciML Advantages over OpenFOAM

1. **Automatic differentiation** — Forward-mode (ForwardDiff.jl) and
   reverse-mode (Enzyme.jl, Zygote.jl) AD through the entire solver,
   including mesh and BC parameters.  OpenFOAM requires hand-coded
   adjoints or finite differences.

2. **Sensitivity analysis & parameter estimation** — Composable with
   `SciMLSensitivity.jl` for forward/adjoint sensitivity of any
   ODE-wrapped problem.  Enables gradient-based calibration of
   turbulence model constants, inlet profiles, or material properties.

3. **Uncertainty quantification** — Direct integration with
   `PolyChaos.jl`, `Surrogates.jl`, and Bayesian frameworks
   (`Turing.jl`) through the SciML problem interface.

4. **Julia composability** — Any user-defined function (flux, source,
   BC) is a plain Julia callable — no code generation step, no
   string-based `codedFixedValue`.  GPU kernels, custom linear algebra,
   and new conservation laws compose without framework modifications.

5. **Time-stepper ecosystem** — Access to 300+ ODE/DAE solvers in
   `OrdinaryDiffEq.jl` with adaptive stepping, implicit-explicit (IMEX)
   schemes, and multirate methods — far beyond OpenFOAM's Euler / CrankNicolson / backward.

6. **Reproducibility** — Julia's package manager provides exact
   environment snapshots (`Manifest.toml`).  OpenFOAM builds depend on
   system compilers, MPI versions, and third-party library paths.
