# Algorithm Provenance

FiniteVolumeMethod.jl is MIT-licensed. Every algorithm in this repository is
a clean-room Julia implementation derived from published papers, textbooks,
or algorithm descriptions in public documentation — not from the OpenFOAM
(GPLv3) source tree or any other GPL'd project.

Comments in the source that say "follows OpenFOAM semantics" or "equivalent
to OpenFOAM X" are algorithmic-intent pointers, not code provenance. Where
an algorithm has a well-known name (e.g. SIMPLE, PISO, MULES, HLLC), we
refer to the name; readers should consult the cited references for full
mathematical treatment.

The table below is the single source of truth for provenance. Any future
addition to `src/` that uses a non-trivial algorithm must add an entry
here. Reviewers are expected to refuse PRs that introduce algorithms
without provenance.

## Hyperbolic solver

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/hyperbolic/riemann_solvers.jl` | HLL Riemann solver | Harten, Lax, van Leer (1983), *SIAM Review* 25(1), 35–61. |
| `src/hyperbolic/riemann_solvers.jl` | HLLC Riemann solver | Toro, Spruce, Speares (1994), *Shock Waves* 4(1), 25–34. DOI 10.1007/BF01414629. |
| `src/hyperbolic/riemann_solvers.jl` | HLLD Riemann solver (MHD) | Miyoshi & Kusano (2005), *J. Comput. Phys.* 208, 315–344. DOI 10.1016/j.jcp.2005.02.017. |
| `src/hyperbolic/reconstruction.jl` | MUSCL slope limiter framework | van Leer (1979), *J. Comput. Phys.* 32, 101–136. |
| `src/schemes/limiters.jl` | Minmod, Superbee, Van Leer, Koren, Ospre, Venkatakrishnan limiters | Sweby (1984), *SIAM J. Numer. Anal.* 21(5), 995–1011; Venkatakrishnan (1995), *J. Comput. Phys.* 118, 120–130. |
| `src/schemes/weno.jl` | WENO3/WENO5 reconstruction | Jiang & Shu (1996), *J. Comput. Phys.* 126, 202–228. DOI 10.1006/jcph.1996.0130. |
| `src/schemes/ppm.jl` | PPM (piecewise parabolic method) | Colella & Woodward (1984), *J. Comput. Phys.* 54, 174–201. |
| `src/constrained_transport/` | Constrained transport for ∇·B = 0 | Evans & Hawley (1988), *Astrophys. J.* 332, 659; Balsara & Spicer (1999), *J. Comput. Phys.* 149, 270–292. |
| `src/hyperbolic/grmhd.jl` | GRMHD primitive recovery | Noble et al. (2006), *Astrophys. J.* 641, 626; Siegel et al. (2018), *Astrophys. J.* 859, 71. |
| `src/hyperbolic/srhydro.jl` | SR hydro primitive recovery | Mignone & Bodo (2005), *MNRAS* 364, 126–136. |
| `src/amr/` | Block-structured AMR with flux correction | Berger & Oliger (1984), *J. Comput. Phys.* 53, 484; Berger & Colella (1989), *J. Comput. Phys.* 82, 64–84. |

## Parabolic solver

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/parabolic/assembly/` | Cell-vertex FVM for unstructured triangulations | Versteeg & Malalasekera (2007), *An Introduction to Computational Fluid Dynamics: The Finite Volume Method*, 2nd ed., Ch. 11. |
| `src/parabolic/boundary_conditions.jl` | Dirichlet / Neumann / Robin ghost-cell assembly | Ferziger & Perić (2002), *Computational Methods for Fluid Dynamics*, 3rd ed., Ch. 4. |
| `src/parabolic/turbulence/parabolic_k_epsilon.jl` | Standard k-ε (parabolic variant) | Launder & Spalding (1974), *Comput. Methods Appl. Mech. Engrg.* 3, 269–289. |

## Collocated (unstructured) operators

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/collocated/types.jl` | CSC sparsity pre-build + `nzval`-indexed assembly via `SparsityPattern` | Standard sparse-matrix techniques: Saad (2003), *Iterative Methods for Sparse Linear Systems*, Ch. 3. The pre-compute-structure-once-then-fill-in-place pattern is how every mature FVM/FEM code avoids CSC random-pattern insertion in its inner loop. |
| `src/collocated/gradient.jl` | Green-Gauss gradient with iterative non-orthogonal correction | Jasak (1996), *Error analysis and estimation for the finite volume method with applications to fluid flows*, PhD thesis, Imperial College, Ch. 3. |
| `src/collocated/laplacian.jl` | Cell-centered Laplacian with non-orthogonal correction | Jasak (1996), Ch. 3–4. |
| `src/collocated/divergence.jl` | Cell-centered divergence via Gauss's theorem | Ferziger & Perić (2002), Ch. 8. |
| `src/collocated/interpolation.jl` | Linear / upwind / blended face interpolation | Darwish & Moukalled (2003), *Int. J. Heat Mass Transfer* 46, 599–611. |
| `src/collocated/interpolation.jl` | Rhie-Chow momentum interpolation | Rhie & Chow (1983), *AIAA J.* 21(11), 1525–1532. DOI 10.2514/3.8284. |
| `src/collocated/ddt.jl` | First-order implicit Euler, second-order BDF2, Crank-Nicolson temporal schemes | Ferziger & Perić (2002), Ch. 6. |

## Incompressible pressure-velocity coupling

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/incompressible/simple.jl` | SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) | Patankar & Spalding (1972), *Int. J. Heat Mass Transfer* 15, 1787–1806; Patankar (1980), *Numerical Heat Transfer and Fluid Flow*, Ch. 6. |
| `src/incompressible/piso.jl` | PISO (Pressure-Implicit with Splitting of Operators) | Issa (1986), *J. Comput. Phys.* 62, 40–65. |
| `src/incompressible/pimple.jl` | PIMPLE (merged SIMPLE + PISO for transient) | Jasak (1996), PhD thesis, Ch. 5; widely documented in CFD textbooks. |

## Turbulence — RANS

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/turbulence/k_epsilon_rans.jl` | Standard k-ε (high-Re form) | Launder & Spalding (1974), *Comput. Methods Appl. Mech. Engrg.* 3, 269–289. |
| `src/turbulence/k_omega.jl` | Standard k-ω | Wilcox (1988), *AIAA J.* 26, 1299–1310. |
| `src/turbulence/k_omega_sst.jl` | k-ω SST | Menter (1994), *AIAA J.* 32(8), 1598–1605. DOI 10.2514/3.12149. |
| `src/turbulence/spalart_allmaras.jl` | Spalart-Allmaras one-equation model | Spalart & Allmaras (1992), AIAA Paper 92-0439. |
| `src/turbulence/wall_functions.jl` | Spalding's continuous wall function | Spalding (1961), *J. Appl. Mech.* 28(3), 455–458. |
| `src/turbulence/wall_functions.jl` | Standard log-law wall function | Launder & Spalding (1974). |

## Turbulence — LES

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/turbulence/smagorinsky.jl` | Smagorinsky-Lilly model | Smagorinsky (1963), *Mon. Weather Rev.* 91, 99–164. |
| `src/turbulence/wale.jl` | WALE (Wall-Adapting Local Eddy-viscosity) | Nicoud & Ducros (1999), *Flow, Turbulence and Combustion* 62, 183–200. DOI 10.1023/A:1009995426001. |
| `src/turbulence/dynamic_smagorinsky.jl` | Germano-identity dynamic Smagorinsky (scalar form, simplified) | Germano et al. (1991), *Phys. Fluids A* 3, 1760–1765; Lilly (1992), *Phys. Fluids A* 4, 633–635. **Note:** current implementation uses scalar Germano; full tensor form is a v3.0 correctness fix (Stage 4a). |

## Turbulence — hybrid

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/turbulence/ddes.jl` | DDES (Delayed Detached-Eddy Simulation) | Spalart et al. (2006), *Theor. Comput. Fluid Dyn.* 20, 181–195. |

## Thermal

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/thermal/energy_equation.jl` | Convection-diffusion energy transport (temperature formulation) | Ferziger & Perić (2002), Ch. 12. |
| `src/thermal/buoyancy.jl` | Boussinesq buoyancy approximation | Boussinesq (1903); standard treatment in Spiegel & Veronis (1960), *Astrophys. J.* 131, 442. |
| `src/thermal/conjugate.jl` | Dirichlet-Neumann conjugate-heat-transfer iteration | Giles (1997), *J. Comput. Phys.* 137, 65–82. **Note:** current implementation uses scalar (face-averaged) interface temperature; per-face is a v3.0 Stage 5a fix. |

## Multiphase VOF

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/multiphase/alpha_transport.jl` | Volume-of-fluid α transport with interface compression | Hirt & Nichols (1981), *J. Comput. Phys.* 39, 201–225; Weller's compressive-velocity modification is described in Rusche (2002), PhD thesis, Imperial College. |
| `src/multiphase/surface_tension.jl` | CSF (Continuum Surface Force) surface tension | Brackbill, Kothe, Zemach (1992), *J. Comput. Phys.* 100, 335–354. |
| `src/multiphase/boundedness.jl` | Boundedness by `clamp(α, 0, 1)` | **Note:** placeholder; MULES (Multidimensional Universal Limiter with Explicit Solution) replacement is a v3.0 Stage 5b fix. Reference for MULES: Weller (2006), OpenFOAM Technical Report (clean-room reimplementation from the algorithm description, not from the OpenFOAM source). |

## Combustion

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/combustion/arrhenius.jl` | Arrhenius kinetics | Standard chemistry: see Turns (2011), *An Introduction to Combustion*, 3rd ed., Ch. 4. |
| `src/combustion/edm.jl` | Eddy Dissipation Model | Magnussen & Hjertager (1977), *16th Symposium (International) on Combustion*, 719–729. |
| `src/combustion/edc.jl` | Eddy Dissipation Concept | Magnussen (1981), AIAA paper 81-0042; Magnussen (2005). |
| `src/combustion/species_transport.jl` | Species convection-diffusion with unity Lewis number | Poinsot & Veynante (2005), *Theoretical and Numerical Combustion*, 2nd ed., Ch. 5. |

## Radiation

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/radiation/p1.jl` | P1 (spherical harmonics) radiation model | Modest (2013), *Radiative Heat Transfer*, 3rd ed., Ch. 16. |
| `src/radiation/fvdom.jl` | Finite-volume discrete ordinates method (fvDOM, skeleton) | Chui & Raithby (1993), *Numer. Heat Transfer B* 23, 269–288. |
| Marshak boundary condition | Marshak wall | Modest (2013), Ch. 16. |

## Lagrangian DPM

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/lagrangian/drag_models.jl` | Stokes drag | Classical: Stokes (1851). |
| `src/lagrangian/drag_models.jl` | Schiller-Naumann drag | Schiller & Naumann (1933), *VDI-Zeitschrift* 77, 318–320. |
| `src/lagrangian/heat_transfer.jl` | Ranz-Marshall Nusselt correlation | Ranz & Marshall (1952), *Chem. Eng. Prog.* 48, 141–146, 173–180. |
| `src/lagrangian/spray.jl` | TAB (Taylor Analogy Breakup) | O'Rourke & Amsden (1987), SAE paper 872089. |
| `src/lagrangian/spray.jl` | KHRT (Kelvin-Helmholtz / Rayleigh-Taylor) breakup | Beale & Reitz (1999), *Atomization and Sprays* 9, 623–650. |
| `src/lagrangian/two_way_coupling.jl` | PSI-cell (particle-source-in-cell) two-way coupling | Crowe, Sharma, Stock (1977), *J. Fluids Eng.* 99, 325–332. |

## Dynamic mesh / ALE

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/dynamic_mesh/ale.jl` | Arbitrary Lagrangian-Eulerian formulation | Hirt, Amsden, Cook (1974), *J. Comput. Phys.* 14, 227–253; Demirdžić & Perić (1988), *Int. J. Numer. Meth. Fluids* 8, 1037–1050. |
| `src/dynamic_mesh/laplacian_motion.jl` | Laplacian mesh motion (vertex smoothing) | Jasak & Tuković (2006), *Trans. FAMENA* 30(2), 1–18. |
| `src/dynamic_mesh/solid_body.jl` | Solid-body translation/rotation | Classical rigid-body motion; standard treatment. |

## Linear solvers / preconditioners

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/linear_solvers/` | Krylov solvers (CG, BiCGSTAB, GMRES) via LinearSolve.jl | delegated to [LinearSolve.jl](https://docs.sciml.ai/LinearSolve/stable/); underlying algorithms per LinearSolve's documentation. |
| Preconditioner extensions | ILU, AMG | delegated to IncompleteLU.jl and AlgebraicMultigrid.jl. |

## Meshing / I/O

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `src/mesh/openfoam_io.jl` | OpenFOAM polyMesh ASCII format reader | Format documented in the public OpenFOAM User Guide (OpenCFD Ltd., CC-BY-NC-SA 4.0). Implementation is clean-room based on the format description only; no code is lifted from the OpenFOAM source. |
| `src/mesh/gmsh_reader.jl` | Gmsh `.msh` v4 format reader | Format documented in the public Gmsh manual (Geuzaine & Remacle). Clean-room implementation. |
| `src/mesh/polyhedral_volumes.jl` | Cell volumes via divergence theorem applied to polyhedra | Standard construction; see Barth (1991), *Tech. Report RNR-91-023*. |

## Extension-provided algorithms

The following algorithms are *not* implemented in this repo; they are
delegated to upstream SciML ecosystem packages via Julia package extensions.
Provenance for these lives upstream:

- CUDA kernels (FVMCUDAExt) — upstream CUDA.jl and KernelAbstractions.jl.
- HDF5 I/O (FVMHdf5Ext) — upstream HDF5.jl.
- VTK I/O (FVMVTKExt) — upstream WriteVTK.jl.
- JLD2 checkpointing (FVMCheckpointExt) — upstream JLD2.jl.
- Plot recipes (FVMRecipesExt) — upstream RecipesBase.jl.
- LinearSolve integration (FVMLinearSolveExt) — upstream LinearSolve.jl.
- AMG preconditioner (FVMAMGExt) — upstream AlgebraicMultigrid.jl.
- ILU preconditioner (FVMILUExt) — upstream IncompleteLU.jl.
- MPI parallelism (FVMMPIExt) — upstream MPI.jl and PartitionedArrays.jl.
- Dashboard server (FVMDashboardExt, FVMDashboardServerExt) — upstream JSON3.jl and HTTP.jl.
