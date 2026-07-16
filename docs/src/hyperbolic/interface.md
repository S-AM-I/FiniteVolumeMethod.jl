```@meta
CurrentModule = FiniteVolumeMethod
```

# Hyperbolic Solver Interface

```@contents
Pages = ["interface.md"]
```

This page documents the public API for the cell-centered finite volume solver for hyperbolic conservation laws. For the parabolic (cell-vertex) solver interface, see the [main interface page](../interface.md).

The typical workflow is:

1. Create a mesh using `StructuredMesh1D`, `StructuredMesh2D`, or `StructuredMesh3D`.
2. Choose an equation of state (`IdealGasEOS` or `StiffenedGasEOS`).
3. Select a conservation law (`EulerEquations`, `IdealMHDEquations`, `NavierStokesEquations`, `SRMHDEquations`, or `GRMHDEquations`).
4. Choose a Riemann solver and reconstruction scheme.
5. Specify boundary conditions.
6. Construct a `HyperbolicProblem` (1D), `HyperbolicProblem2D`, or `HyperbolicProblem3D`.
7. Solve with `solve(prob, alg; ...)` or build `sciml_problem(prob)` explicitly.

The fixed-step `solve_hyperbolic` and `solve_hyperbolic_imex` loops are internal as of v4.0; `solve_amr` remains public as the dynamic-regridding/subcycling AMR driver (see "Non-SciML Drivers" below).

## Meshes

```@docs
AbstractMesh
StructuredMesh1D
StructuredMesh2D
StructuredMesh3D
ndims_mesh
ncells
nfaces
cell_center
cell_volume
face_area
face_owner
face_neighbor
```

## Equations of State

```@docs
AbstractEOS
IdealGasEOS
StiffenedGasEOS
pressure
sound_speed
internal_energy
total_energy
```

## Conservation Laws

```@docs
AbstractConservationLaw
EulerEquations
IdealMHDEquations
NavierStokesEquations
nvariables
physical_flux
max_wave_speed
wave_speeds
conserved_to_primitive
primitive_to_conserved
fast_magnetosonic_speed
```

## Riemann Solvers

```@docs
AbstractRiemannSolver
LaxFriedrichsSolver
HLLSolver
HLLCSolver
HLLDSolver
solve_riemann
```

## Reconstruction

```@docs
NoReconstruction
CellCenteredMUSCL
reconstruct_interface
WENO3
WENO5
CharacteristicWENO
nghost
left_eigenvectors
right_eigenvectors
```

## Boundary Conditions

```@docs
AbstractHyperbolicBC
TransmissiveBC
ReflectiveBC
NoSlipBC
DirichletHyperbolicBC
InflowBC
PeriodicHyperbolicBC
```

## Problems and Solvers

```@docs
HyperbolicProblem
HyperbolicProblem2D
HyperbolicProblem3D
mhd_stage_limiter
compute_dt
compute_dt_2d
compute_dt_3d
```

The canonical SciML execution helpers `sciml_problem`, `solution_accessor`, and
`solution_snapshot` are documented on the package-level interface page so they
can serve parabolic, hyperbolic, AMR, coupling, and relativistic workflows from
one place.

For direct `solve(prob, alg; ...)` calls, SciML callbacks supplied via the `callback` keyword are merged with the package's internal CFL controller. For constrained-transport MHD, pass a stage limiter such as `SSPRK33(; stage_limiter! = mhd_stage_limiter(sciml_problem(prob).p))`.

## Navier-Stokes (Viscous)

```@docs
thermal_conductivity
viscous_flux_1d
viscous_flux_x_2d
viscous_flux_y_2d
```

## Constrained Transport

```@docs
CTData2D
CTData3D
initialize_ct!
initialize_ct_from_potential!
face_to_cell_B!
compute_emf_2d!
ct_update!
compute_divB
max_divB
l2_divB
```

## IMEX Time Integration

```@docs
AbstractIMEXScheme
IMEX_SSP3_433
IMEX_ARS222
IMEX_Midpoint
imex_tableau
imex_nstages
```

## Stiff Sources

```@docs
AbstractStiffSource
NullSource
CoolingSource
ResistiveSource
evaluate_stiff_source
stiff_source_jacobian
```

## Adaptive Mesh Refinement

```@docs
AMRBlock
AMRGrid
AbstractRefinementCriterion
GradientRefinement
CurrentSheetRefinement
AMRProblem
compute_dt_amr
is_leaf
active_blocks
blocks_at_level
max_active_level
refine_block!
coarsen_block!
regrid!
prolongate!
restrict!
FluxRegister
reset_flux_register!
accumulate_fine_flux!
store_coarse_flux!
apply_flux_correction_2d!
apply_flux_correction_3d!
```

## Non-SciML Drivers

The fixed-step convenience loops `solve_hyperbolic` and `solve_hyperbolic_imex`
are internal as of v4.0 (kept, unexported, for threaded execution, the GPU
backend dispatch, and the CPU-vs-CUDA parity baseline); the public execution
path is `sciml_problem(prob)` + `solve`. Dynamically-regridding and subcycled
AMR have no SciML equivalent yet, so their driver remains public:

```@docs
solve_amr
```

## Spacetime Metrics (GRMHD)

```@docs
AbstractMetric
MinkowskiMetric
SchwarzschildMetric
KerrMetric
lapse
shift
spatial_metric
sqrt_gamma
inv_spatial_metric
MetricData2D
precompute_metric
```

## SRMHD and GRMHD

```@docs
SRMHDEquations
GRMHDEquations
lorentz_factor
srmhd_con2prim
Con2PrimResult
grmhd_con2prim
grmhd_source_terms
```
