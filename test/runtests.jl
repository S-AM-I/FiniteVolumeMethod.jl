using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: LocalFVMMesh, SnappyMesher
using FiniteVolumeMethod: AbstractLinearOperator, CollocatedEquation, RadiationState, clip_alpha!, find_nearest_cell, update_mesh!, verify_gcl
using Test
using Dates
using Aqua

include(joinpath(dirname(@__DIR__), "validation", "manifest.jl"))
using .RepoValidationManifest

ct() = Dates.format(now(), "HH:MM:SS")
function safe_include(filename; name = filename) # Workaround for not being able to interpolate into SafeTestset test names
    mod = @eval module $(gensym()) end
    @info "[$(ct())] Testing $name"
    return @testset verbose = true "Example: $name" begin
        Base.include(mod, filename)
    end
end

# ── Test groups ──────────────────────────────────────────────────────
#
# `FVM_TEST_GROUP` selects which part of the suite to run, mirroring the
# `src/` module tree so a group maps to the code it covers. The default,
# "all", runs everything, so a bare `Pkg.test()` is unchanged.
#
#   geometry parabolic hyperbolic collocated sciml experimental
#   governance tutorials verification
#
# Several groups may be given, comma-separated:
#   FVM_TEST_GROUP=collocated,sciml julia --project -e 'using Pkg; Pkg.test()'
const TEST_GROUP = get(ENV, "FVM_TEST_GROUP", "all")
const _WANTED = Set(strip.(split(TEST_GROUP, ",")))

# The "vandv" group is opt-in even under "all": those cases run full SIMPLE
# solves (1-2 min each) and are driven by the nightly workflow via
# `FVM_RUN_VANDV`. Naming the group explicitly also runs them.
const RUN_VANDV = get(ENV, "FVM_RUN_VANDV", "false") == "true"
function want(group)
    group == "vandv" && return RUN_VANDV || ("vandv" in _WANTED)
    return ("all" in _WANTED) || (group in _WANTED)
end

# (group, testset name, path relative to test/)
const TESTS = [
    ("geometry", "Geometry", "geometry/geometry.jl"),
    ("parabolic", "Conditions", "parabolic/conditions.jl"),
    ("parabolic", "Robin BCs", "parabolic/robin.jl"),
    ("parabolic", "Cylindrical Robin sign", "parabolic/cylindrical_robin.jl"),
    ("parabolic", "Cylindrical 2D annular MMS", "parabolic/cylindrical_2d_mms.jl"),
    ("parabolic", "Variable Cylindrical Diffusion MMS", "parabolic/variable_cylindrical_mms.jl"),
    ("parabolic", "Variable Cylindrical Advection-Diffusion", "parabolic/variable_cylindrical_advection_diffusion.jl"),
    ("parabolic", "Parabolic Mesh", "parabolic/parabolic_mesh.jl"),
    ("parabolic", "Parabolic CRUD Paths", "parabolic/parabolic_crud_paths.jl"),
    ("governance", "Collocated Assembly Benchmark", "governance/assembly_bench.jl"),
    ("sciml", "SciML Contract Uniform", "sciml/sciml_contract_uniform.jl"),
    ("experimental", "MPI Partition", "experimental/mpi_partition.jl"),
    ("experimental", "Pressure-Based Thermo + Rheology", "experimental/pressure_based_models.jl"),
    ("collocated", "Turbulence Correctness", "collocated/turbulence_correctness.jl"),
    ("collocated", "MULES + GCL", "collocated/mules_gcl.jl"),
    ("collocated", "MRF + Porous + Cavitation", "collocated/mrf_porous_cavitation.jl"),
    ("experimental", "Aeroacoustics Surface Integration", "experimental/aeroacoustics_surface.jl"),
    ("experimental", "Population Balance (QMoM)", "experimental/population_balance_qmom.jl"),
    ("experimental", "Solid Mechanics (Linear Elasticity)", "experimental/solid_mechanics_elasticity.jl"),
    ("experimental", "FSI Coupling", "experimental/fsi_coupling.jl"),
    ("collocated", "Function Objects", "collocated/function_objects.jl"),
    ("experimental", "Octree Meshing", "experimental/octree_meshing.jl"),
    ("collocated", "AMR Markers + ZZ Indicator", "collocated/amr_markers.jl"),
    ("sciml", "Matrix-Free Operator + Unitful", "sciml/matrixfree_units.jl"),
    ("collocated", "V&V: Laplacian operator MMS", "collocated/v_and_v_laplacian_mms.jl"),
    ("collocated", "V&V: Gradient + Divergence MMS", "collocated/v_and_v_operator_mms.jl"),
    ("collocated", "V&V: Rhie-Chow interpolation", "collocated/v_and_v_rhie_chow.jl"),
    ("collocated", "V&V: Laplacian on skewed mesh", "collocated/v_and_v_laplacian_skewed.jl"),
    ("collocated", "V&V: Temporal ddt MMS", "collocated/v_and_v_temporal_mms.jl"),
    ("collocated", "V&V: Poiseuille channel", "collocated/v_and_v_poiseuille.jl"),
    ("collocated", "V&V: Heat conduction analytical", "collocated/v_and_v_heat_conduction.jl"),
    ("collocated", "V&V: Stokes terminal velocity", "collocated/v_and_v_stokes_terminal.jl"),
    ("collocated", "V&V: GCL invariances", "collocated/v_and_v_gcl.jl"),
    ("collocated", "V&V: P1 radiation slab", "collocated/v_and_v_p1_slab.jl"),
    ("collocated", "V&V: VOF translation", "collocated/v_and_v_vof_translation.jl"),
    ("collocated", "V&V: Species advection-diffusion", "collocated/v_and_v_species_ad.jl"),
    ("collocated", "V&V: k-ε DHIT", "collocated/v_and_v_kepsilon_dhit.jl"),
    ("collocated", "V&V: Smagorinsky LES", "collocated/v_and_v_smagorinsky.jl"),
    ("collocated", "V&V: Postprocessing kinematics", "collocated/v_and_v_postprocessing.jl"),
    ("collocated", "V&V: Unsteady heat", "collocated/v_and_v_unsteady_heat.jl"),
    ("collocated", "V&V: Couette flow", "collocated/v_and_v_couette.jl"),
    ("collocated", "V&V: k-ε log-layer equilibrium", "collocated/v_and_v_kepsilon_loglayer.jl"),
    ("collocated", "V&V: VOF plane wave", "collocated/v_and_v_vof_planewave.jl"),
    ("collocated", "V&V: P1 radiative equilibrium", "collocated/v_and_v_p1_equilibrium.jl"),
    ("collocated", "V&V: Schiller-Naumann drag", "collocated/v_and_v_schiller_naumann.jl"),
    ("collocated", "V&V: EDM combustion algebra", "collocated/v_and_v_edm.jl"),
    ("collocated", "V&V: WALE LES model", "collocated/v_and_v_wale.jl"),
    ("collocated", "V&V: GCL rotation", "collocated/v_and_v_gcl_rotation.jl"),
    ("collocated", "V&V: Courant + Q-sign", "collocated/v_and_v_courant.jl"),
    ("collocated", "V&V: Boussinesq buoyancy", "collocated/v_and_v_boussinesq.jl"),
    ("collocated", "V&V: Ranz-Marshall particle heat", "collocated/v_and_v_ranz_marshall.jl"),
    ("collocated", "V&V: Mesh sweep flux", "collocated/v_and_v_mesh_flux.jl"),
    ("collocated", "V&V: Radiation source algebra", "collocated/v_and_v_radiation_source.jl"),
    ("collocated", "V&V: VOF mixture properties", "collocated/v_and_v_vof_mixture.jl"),
    ("collocated", "V&V: Arrhenius kinetics", "collocated/v_and_v_arrhenius.jl"),
    ("collocated", "V&V: k-ω turbulence", "collocated/v_and_v_komega.jl"),
    ("collocated", "V&V: LES filter width", "collocated/v_and_v_filter_width.jl"),
    ("collocated", "V&V: Field statistics", "collocated/v_and_v_field_stats.jl"),
    ("collocated", "V&V: PISO transient stability", "collocated/v_and_v_piso_decay.jl"),
    ("collocated", "V&V: Linear solvers", "collocated/v_and_v_linear_solvers.jl"),
    ("geometry", "V&V: Mesh geometry invariants", "geometry/v_and_v_mesh_geometry.jl"),
    ("collocated", "V&V: Spalart-Allmaras", "collocated/v_and_v_spalart_allmaras.jl"),
    ("collocated", "V&V: Spray breakup", "collocated/v_and_v_spray.jl"),
    ("collocated", "V&V: CSF surface tension", "collocated/v_and_v_csf.jl"),
    ("collocated", "V&V: FR/ED combustion", "collocated/v_and_v_fred.jl"),
    ("collocated", "V&V: fvDOM quadrature", "collocated/v_and_v_fvdom_quadrature.jl"),
    ("collocated", "V&V: ALE-corrected flux", "collocated/v_and_v_ale_flux.jl"),
    ("collocated", "V&V: CHT interface flux", "collocated/v_and_v_cht_interface.jl"),
    ("collocated", "V&V: Wall quantities", "collocated/v_and_v_wall_quantities.jl"),
    ("collocated", "V&V: Solver config dispatch", "collocated/v_and_v_solver_config.jl"),
    ("collocated", "V&V: Strain rate primitive", "collocated/v_and_v_strain_rate.jl"),
    ("collocated", "V&V: Wall functions", "collocated/v_and_v_wall_functions.jl"),
    ("collocated", "V&V: Incompressible SciML interface", "collocated/v_and_v_incompressible_sciml.jl"),
    ("collocated", "V&V: Effective conductivity k_eff", "collocated/v_and_v_k_eff.jl"),
    ("collocated", "V&V: Particle state", "collocated/v_and_v_particle_state.jl"),
    ("collocated", "V&V: Laplacian mesh motion", "collocated/v_and_v_laplacian_motion.jl"),
    ("collocated", "V&V: VOF compression flux", "collocated/v_and_v_vof_compression.jl"),
    ("collocated", "V&V: Marshak wall BC", "collocated/v_and_v_marshak.jl"),
    ("collocated", "V&V: Combustion properties", "collocated/v_and_v_combustion_props.jl"),
    ("collocated", "V&V: Nusselt + y+", "collocated/v_and_v_nusselt.jl"),
    ("collocated", "V&V: LES turbulence state", "collocated/v_and_v_les_state.jl"),
    ("collocated", "V&V: Incompressible remake", "collocated/v_and_v_incompressible_remake.jl"),
    ("collocated", "V&V: Thermal types", "collocated/v_and_v_thermal_types.jl"),
    ("collocated", "V&V: RANS turbulence state", "collocated/v_and_v_rans_state.jl"),
    ("collocated", "V&V: SolidBodyMotion", "collocated/v_and_v_solid_body_motion.jl"),
    ("collocated", "V&V: RadiationState + P1Model", "collocated/v_and_v_radiation_state.jl"),
    ("collocated", "V&V: VOFState + TwoPhaseProperties", "collocated/v_and_v_vof_state.jl"),
    ("collocated", "V&V: DPM dispatch", "collocated/v_and_v_dpm_dispatch.jl"),
    ("collocated", "V&V: Heat release primitive", "collocated/v_and_v_heat_release.jl"),
    ("collocated", "V&V: Force coefficients", "collocated/v_and_v_force_coefficients.jl"),
    ("collocated", "V&V: LES dispatch", "collocated/v_and_v_les_dispatch.jl"),
    ("geometry", "V&V: Mesh accessors", "geometry/v_and_v_mesh_accessors.jl"),
    ("collocated", "V&V: IncompressibleState", "collocated/v_and_v_inc_state.jl"),
    ("collocated", "V&V: CHT problem", "collocated/v_and_v_cht_problem.jl"),
    ("collocated", "V&V: Turbulence inlet BCs", "collocated/v_and_v_turbulence_inlet.jl"),
    ("collocated", "V&V: DDES hybrid", "collocated/v_and_v_ddes.jl"),
    ("collocated", "V&V: clip_alpha!", "collocated/v_and_v_clip_alpha.jl"),
    ("collocated", "V&V: update_mesh!", "collocated/v_and_v_mesh_update.jl"),
    ("collocated", "V&V: FvDOMModel", "collocated/v_and_v_fvdom_model.jl"),
    ("collocated", "V&V: Species index lookup", "collocated/v_and_v_species_index.jl"),
    ("geometry", "V&V: find_nearest_cell", "geometry/v_and_v_cell_lookup.jl"),
    ("collocated", "V&V: Courant edge cases", "collocated/v_and_v_courant_edge.jl"),
    ("collocated", "V&V: Field constructors", "collocated/v_and_v_field_constructors.jl"),
    ("collocated", "V&V: PV coupling constructors", "collocated/v_and_v_pv_coupling.jl"),
    ("sciml", "V&V: CollocatedEquation", "sciml/v_and_v_equation_types.jl"),
    ("collocated", "V&V: Q_gen volumetric heat", "collocated/v_and_v_qgen.jl"),
    ("collocated", "V&V: SST blending algebra", "collocated/v_and_v_sst_blend.jl"),
    ("collocated", "V&V: LES primitives (test filter + contract)", "collocated/v_and_v_test_filter.jl"),
    ("collocated", "V&V: MULES flux limiter", "collocated/v_and_v_mules.jl"),
    ("collocated", "V&V: P1 solver invariants", "collocated/v_and_v_p1_solver.jl"),
    ("collocated", "V&V: EDC reaction rates", "collocated/v_and_v_edc.jl"),
    ("collocated", "V&V: PSI-cell two-way coupling", "collocated/v_and_v_psi_cell.jl"),
    ("collocated", "V&V: verify_gcl diagnostic", "collocated/v_and_v_verify_gcl.jl"),
    ("collocated", "V&V: field sampling", "collocated/v_and_v_sampling.jl"),
    ("sciml", "V&V: AbstractLinearOperator", "sciml/v_and_v_linear_operator.jl"),
    ("geometry", "V&V: mesh quality report", "geometry/v_and_v_mesh_quality.jl"),
    ("collocated", "V&V: temporal ddt assembly", "collocated/v_and_v_ddt.jl"),
    ("collocated", "V&V: continuity residual primitives", "collocated/v_and_v_continuity.jl"),
    ("geometry", "V&V: polyhedral volumes", "geometry/v_and_v_polyhedral_volumes.jl"),
    ("experimental", "V&V: compressible pressure-based", "experimental/v_and_v_compressible.jl"),
    ("collocated", "V&V: Durbin realizability + full-tensor P_k", "collocated/v_and_v_durbin.jl"),
    ("collocated", "V&V: equilibrium WMLES", "collocated/v_and_v_wmles.jl"),
    ("collocated", "V&V: full-tensor Germano invariants", "collocated/v_and_v_full_germano.jl"),
    ("collocated", "V&V: wall-function skew projection", "collocated/v_and_v_wall_skew.jl"),
    ("collocated", "V&V: per-face CHT coupling", "collocated/v_and_v_per_face_cht.jl"),
    ("collocated", "V&V: enthalpy energy equation", "collocated/v_and_v_enthalpy.jl"),
    ("collocated", "V&V: MULES-wired α transport", "collocated/v_and_v_mules_integration.jl"),
    ("collocated", "V&V: isoAdvector interface flux", "collocated/v_and_v_iso_advector.jl"),
    ("collocated", "V&V: contact angle (static + Cox-Voinov)", "collocated/v_and_v_contact_angle.jl"),
    ("collocated", "V&V: over-relaxed non-orthogonal correction", "collocated/v_and_v_over_relaxed.jl"),
    ("collocated", "V&V: least-squares gradient", "collocated/v_and_v_lsq_gradient.jl"),
    ("collocated", "V&V: variable Lewis number", "collocated/v_and_v_variable_lewis.jl"),
    ("collocated", "V&V: multi-step mechanism", "collocated/v_and_v_multi_step.jl"),
    ("collocated", "V&V: FGM tabulated chemistry", "collocated/v_and_v_fgm.jl"),
    ("collocated", "V&V: fvDOM scattering", "collocated/v_and_v_scattering.jl"),
    ("collocated", "V&V: SN quadratures (S6/S8/S12)", "collocated/v_and_v_sn_quadratures.jl"),
    ("collocated", "V&V: WSGGM weighted-sum-of-grey-gases", "collocated/v_and_v_wsggm.jl"),
    ("collocated", "V&V: hard-sphere DEM collision", "collocated/v_and_v_hard_sphere_dem.jl"),
    ("collocated", "V&V: agglomeration", "collocated/v_and_v_agglomeration.jl"),
    ("collocated", "V&V: primary breakup (KH-ACT + LISA)", "collocated/v_and_v_primary_breakup.jl"),
    ("collocated", "V&V: injection patterns", "collocated/v_and_v_injection_patterns.jl"),
    ("collocated", "V&V: 6-DOF rigid body", "collocated/v_and_v_six_dof.jl"),
    ("collocated", "V&V: topoChanger", "collocated/v_and_v_topo_changer.jl"),
    ("collocated", "V&V: overset/chimera interpolation", "collocated/v_and_v_overset.jl"),
    ("collocated", "V&V: AMI sliding interface", "collocated/v_and_v_ami.jl"),
    ("collocated", "V&V: Kunz cavitation", "collocated/v_and_v_kunz.jl"),
    ("collocated", "V&V: Schnerr-Sauer cavitation", "collocated/v_and_v_schnerr_sauer.jl"),
    ("collocated", "V&V: Merkle cavitation", "collocated/v_and_v_merkle.jl"),
    ("collocated", "V&V: Darcy-Forchheimer porous", "collocated/v_and_v_darcy_forchheimer.jl"),
    ("collocated", "V&V: MRF single-zone Coriolis+centrifugal", "collocated/v_and_v_mrf_single_zone.jl"),
    ("collocated", "V&V: MRF multi-zone", "collocated/v_and_v_mrf_multi_zone.jl"),
    ("experimental", "V&V: linear elasticity", "experimental/v_and_v_linear_elasticity.jl"),
    ("experimental", "V&V: finite strain (updated-Lagrangian)", "experimental/v_and_v_finite_strain.jl"),
    ("experimental", "V&V: Aitken relaxation", "experimental/v_and_v_aitken.jl"),
    ("experimental", "V&V: partitioned FSI", "experimental/v_and_v_partitioned_fsi.jl"),
    ("experimental", "V&V: FW-H aeroacoustics", "experimental/v_and_v_fwh.jl"),
    ("experimental", "V&V: PML sponge zones", "experimental/v_and_v_pml.jl"),
    ("experimental", "V&V: QMoM Wheeler inversion", "experimental/v_and_v_qmom.jl"),
    ("experimental", "V&V: DQMoM", "experimental/v_and_v_dqmom.jl"),
    ("experimental", "V&V: Class Method PBM", "experimental/v_and_v_class_method.jl"),
    ("experimental", "V&V: octree primitive", "experimental/v_and_v_octree.jl"),
    ("experimental", "V&V: SnappyMesher stub", "experimental/v_and_v_snappy_stub.jl"),
    ("experimental", "V&V: Gmsh pipeline stub", "experimental/v_and_v_gmsh_pipeline.jl"),
    ("collocated", "V&V: ZZ error indicator", "collocated/v_and_v_zz_indicator.jl"),
    ("collocated", "V&V: residual error indicator", "collocated/v_and_v_residual_indicator.jl"),
    ("collocated", "V&V: collocated refinement/coarsening", "collocated/v_and_v_collocated_refine.jl"),
    ("experimental", "V&V: steady adjoint identity", "experimental/v_and_v_steady_adjoint.jl"),
    ("experimental", "V&V: transient adjoint stub", "experimental/v_and_v_transient_adjoint_stub.jl"),
    ("experimental", "V&V: KA backend CPU path", "experimental/v_and_v_ka_backend.jl"),
    ("experimental", "V&V: Enzyme full-solver stub", "experimental/v_and_v_enzyme_stub.jl"),
    ("collocated", "V&V: runtime expression BC", "collocated/v_and_v_expression_bc.jl"),
    ("sciml", "V&V: Unitful hook", "sciml/v_and_v_unitful.jl"),
    ("experimental", "V&V: CoolProp stub", "experimental/v_and_v_coolprop_stub.jl"),
    ("experimental", "V&V: PETSc stub", "experimental/v_and_v_petsc_stub.jl"),
    ("experimental", "V&V: LocalFVMMesh partition view", "experimental/v_and_v_local_mesh.jl"),
    ("experimental", "V&V: Metis partition stub", "experimental/v_and_v_metis_partition_stub.jl"),
    ("collocated", "V&V: two-fluid drag closures", "collocated/v_and_v_drag_closures.jl"),
    ("collocated", "V&V: Eulerian two-fluid (experimental)", "collocated/v_and_v_two_fluid.jl"),
    ("collocated", "V&V: two-fluid coupled solver (v3.1)", "collocated/v_and_v_two_fluid_solver.jl"),
    ("experimental", "V&V: transient adjoint (checkpointed)", "experimental/v_and_v_transient_adjoint.jl"),
    ("collocated", "V&V: IDDES full Shur-2008 shielding", "collocated/v_and_v_iddes_full.jl"),
    ("experimental", "V&V: primary-breakup FSI coupling", "experimental/v_and_v_primary_breakup_fsi.jl"),
    ("geometry", "V&V: STL reader", "geometry/v_and_v_stl_reader.jl"),
    ("experimental", "V&V: snappyHexMesh native (castellated + snap)", "experimental/v_and_v_snappy_native.jl"),
    ("vandv", "V&V: Poiseuille grid convergence", "collocated/v_and_v_poiseuille_convergence.jl"),
    ("vandv", "V&V: Ghia lid-driven cavity Re=100", "collocated/v_and_v_ghia_cavity.jl"),
    ("parabolic", "Problem", "parabolic/problem.jl"),
    ("parabolic", "Equations", "parabolic/equations.jl"),
    ("hyperbolic", "Schemes", "hyperbolic/schemes.jl"),
    ("parabolic", "Advanced BCs", "parabolic/advanced_bcs.jl"),
    ("parabolic", "Physics Models", "parabolic/physics.jl"),
    ("hyperbolic", "Hyperbolic Solver", "hyperbolic/hyperbolic.jl"),
    ("hyperbolic", "Hyperbolic 2D + HLLC", "hyperbolic/hyperbolic_2d.jl"),
    ("hyperbolic", "MHD + HLLD", "hyperbolic/mhd.jl"),
    ("hyperbolic", "MHD 2D + CT", "hyperbolic/mhd_2d.jl"),
    ("hyperbolic", "Navier-Stokes", "hyperbolic/navier_stokes.jl"),
    ("hyperbolic", "SRMHD", "hyperbolic/srmhd.jl"),
    ("hyperbolic", "SRMHD 2D", "hyperbolic/srmhd_2d.jl"),
    ("hyperbolic", "GRMHD", "hyperbolic/grmhd.jl"),
    ("hyperbolic", "GRMHD 2D", "hyperbolic/grmhd_2d.jl"),
    ("hyperbolic", "Hyperbolic 3D", "hyperbolic/hyperbolic_3d.jl"),
    ("hyperbolic", "MHD 3D", "hyperbolic/mhd_3d.jl"),
    ("hyperbolic", "AMR", "hyperbolic/amr.jl"),
    ("hyperbolic", "WENO", "hyperbolic/weno.jl"),
    ("hyperbolic", "IMEX", "hyperbolic/imex.jl"),
    ("hyperbolic", "Unstructured Hyperbolic", "hyperbolic/unstructured_hyperbolic.jl"),
    ("hyperbolic", "Multi-Physics Coupling", "hyperbolic/coupling.jl"),
    ("governance", "Performance & Threading", "governance/performance.jl"),
    ("governance", "Performance Calibration", "governance/performance_calibration.jl"),
    ("hyperbolic", "Advanced Numerics", "hyperbolic/advanced_numerics.jl"),
    ("hyperbolic", "Extended Physics", "hyperbolic/extended_physics.jl"),
    ("hyperbolic", "Reactive Euler", "hyperbolic/reactive_euler.jl"),
    ("collocated", "Incompressible NS", "collocated/incompressible.jl"),
    ("collocated", "Incompressible SciML Compliance", "collocated/incompressible_sciml.jl"),
    ("collocated", "Incompressible SciML Compliance", "collocated/incompressible_integrator.jl"),
    ("collocated", "RANS Turbulence", "collocated/turbulence_rans.jl"),
    ("collocated", "Conjugate Heat Transfer", "collocated/thermal.jl"),
    ("geometry", "Polyhedral Mesh I/O", "geometry/mesh_io.jl"),
    ("collocated", "Post-Processing", "collocated/postprocessing.jl"),
    ("collocated", "Linear Solvers", "collocated/linear_solvers.jl"),
    ("collocated", "LES Turbulence", "collocated/turbulence_les.jl"),
    ("collocated", "Multiphase VOF", "collocated/multiphase_vof.jl"),
    ("collocated", "Radiation", "collocated/radiation.jl"),
    ("collocated", "Combustion", "collocated/combustion.jl"),
    ("collocated", "Lagrangian DPM", "collocated/lagrangian_dpm.jl"),
    ("collocated", "Dynamic Mesh", "collocated/dynamic_mesh.jl"),
    ("collocated", "Remaining Features", "collocated/remaining_features.jl"),
    ("governance", "README", "governance/README.jl"),
    ("geometry", "Coordinate Systems", "geometry/test_coordinate_systems.jl"),
    ("parabolic", "Dashboard", "parabolic/test_dashboard.jl"),
    ("parabolic", "I/O", "parabolic/io.jl"),
    ("sciml", "Remake", "sciml/test_remake.jl"),
    ("sciml", "Semidiscrete Core", "sciml/semidiscrete.jl"),
    ("sciml", "Semidiscrete MHD", "sciml/semidiscrete_mhd.jl"),
    ("sciml", "Semidiscrete AMR", "sciml/semidiscrete_amr.jl"),
    ("sciml", "Semidiscrete IMEX", "sciml/semidiscrete_imex.jl"),
    ("sciml", "SciML Contract", "sciml/sciml_contract.jl"),
    ("governance", "Environment Integrity", "governance/environment_integrity.jl"),
    ("governance", "SciML Audit", "governance/sciml_audit.jl"),
    ("governance", "Repository Governance", "governance/repository_governance.jl"),
    ("governance", "Reproducibility Bundles", "governance/reproducibility_bundle.jl"),
    ("governance", "Reference Artifacts", "governance/reference_artifacts.jl"),
    ("sciml", "Backend Parity", "sciml/backend_parity.jl"),
    ("governance", "Summary Replay", "governance/summary_replay.jl"),
    ("governance", "Quality Ledger", "governance/quality_ledger.jl"),
    ("governance", "Explicit Imports", "governance/explicit_imports.jl"),
]

@testset verbose = true "FiniteVolumeMethod.jl" begin
    for (group, name, file) in TESTS
        want(group) || continue
        @testset verbose = true "$name" begin
            safe_include(file)
        end
    end

    if want("tutorials")
        @testset verbose = true "Tutorials" begin
            dir = joinpath(dirname(@__DIR__), "docs", "src", "literate_tutorials")
            files = filter(!=("keller_segel_chemotaxis.jl"), readdir(dir))
            file_names = [
                "diffusion_equation_in_a_wedge_with_mixed_boundary_conditions.jl",
                "diffusion_equation_on_a_square_plate.jl",
                "diffusion_equation_on_an_annulus.jl",
                "equilibrium_temperature_distribution_with_mixed_boundary_conditions_and_using_ensembleproblems.jl",
                "helmholtz_equation_with_inhomogeneous_boundary_conditions.jl",
                "laplaces_equation_with_internal_dirichlet_conditions.jl",
                "mean_exit_time.jl",
                "piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation.jl",
                "porous_fisher_equation_and_travelling_waves.jl",
                "porous_medium_equation.jl",
                "reaction_diffusion_brusselator_system_of_pdes.jl",
                "reaction_diffusion_equation_with_a_time_dependent_dirichlet_boundary_condition_on_a_disk.jl",
                "solving_mazes_with_laplaces_equation.jl",
                "gray_scott_model_turing_patterns_from_a_coupled_reaction_diffusion_system.jl",
            ] # do it manually just to make it easier for testing individual files rather than in a loop, e.g. one like
            #=
            for file in files
                @testset "Example: $file" begin
                    safe_include(joinpath(dir, file))
                end
            end
            =#
            @test length(files) == length(file_names) # make sure we didn't miss any
            safe_include(joinpath(dir, file_names[1]); name = file_names[1]) # diffusion_equation_in_a_wedge_with_mixed_boundary_conditions
            safe_include(joinpath(dir, file_names[2]); name = file_names[2]) # diffusion_equation_on_a_square_plate
            safe_include(joinpath(dir, file_names[3]); name = file_names[3]) # diffusion_equation_on_an_annulus
            safe_include(joinpath(dir, file_names[4]); name = file_names[4]) # equilibrium_temperature_distribution_with_mixed_boundary_conditions_and_using_ensembleproblems
            safe_include(joinpath(dir, file_names[5]); name = file_names[5]) # helmholtz_equation_with_inhomogeneous_boundary_conditions
            safe_include(joinpath(dir, file_names[6]); name = file_names[6]) # laplaces_equation_with_internal_dirichlet_conditions
            safe_include(joinpath(dir, file_names[7]); name = file_names[7]) # mean_exit_time
            safe_include(joinpath(dir, file_names[8]); name = file_names[8]) # piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation
            safe_include(joinpath(dir, file_names[9]); name = file_names[9]) # porous_fisher_equation_and_travelling_waves
            safe_include(joinpath(dir, file_names[10]); name = file_names[10]) # porous_medium_equation
            safe_include(joinpath(dir, file_names[11]); name = file_names[11]) # reaction_diffusion_brusselator_system_of_pdes
            safe_include(joinpath(dir, file_names[12]); name = file_names[12]) # reaction_diffusion_equation_with_a_time_dependent_dirichlet_boundary_condition_on_a_disk
            safe_include(joinpath(dir, file_names[13]); name = file_names[13]) # solving_mazes_with_laplaces_equation
            safe_include(joinpath(dir, file_names[14]); name = file_names[14]) # gray_scott_model_turing_patterns_from_a_coupled_reaction_diffusion_system
            # safe_include(joinpath(dir, file_names[15]); name=file_names[15]) # keller_segel_chemotaxis
        end
    end

    if want("tutorials")
        @testset verbose = true "Custom Templates" begin
            dir = joinpath(dirname(@__DIR__), "docs", "src", "literate_wyos")
            files = readdir(dir)
            file_names = [
                "diffusion_equations.jl",
                "mean_exit_time.jl",
                "linear_reaction_diffusion_equations.jl",
                "poissons_equation.jl",
                "laplaces_equation.jl",
            ]
            @test length(files) == length(file_names) # make sure we didn't miss any
            safe_include(joinpath(dir, file_names[1]); name = file_names[1]) # diffusion_equations
            safe_include(joinpath(dir, file_names[2]); name = file_names[2]) # mean_exit_time
            safe_include(joinpath(dir, file_names[3]); name = file_names[3]) # linear_reaction_diffusion_equations
            safe_include(joinpath(dir, file_names[4]); name = file_names[4]) # poissons_equation
            safe_include(joinpath(dir, file_names[5]); name = file_names[5]) # laplaces_equation
        end
    end

    if want("verification")
        @testset verbose = true "Verification" begin
            dir = joinpath(dirname(@__DIR__), "docs", "src", "literate_verification")
            manifest = RepoValidationManifest.load_manifest(joinpath(dirname(@__DIR__), "validation", "manifest.toml"))
            file_names = sort!(
                [basename(entry.source) for entry in RepoValidationManifest.verification_pages(manifest)];
                by = identity,
            )
            @test !isempty(file_names)
            for file_name in file_names
                safe_include(joinpath(dir, file_name); name = file_name)
            end
        end
    end

    if want("governance")
        @testset verbose = true "Aqua" begin
            # piracy disabled: Aqua 0.7 traverses Core.TypeName.mt which was
            # removed in Julia 1.12 (FieldError). Re-enable once Aqua >= 0.8.
            # project_extras re-enabled 2026-07-15 after JET moved out of the
            # root [deps] and the dead [extras]/[targets] block was removed
            # (test/Project.toml governs the test env on Julia >= 1.2).
            # Aqua 0.8 renamed `piracy` → `piracies`; kept disabled for now, the
            # Stage-7 test-suite tightening pass re-evaluates enabling it.
            Aqua.test_all(
                FiniteVolumeMethod;
                ambiguities = false, unbound_args = false,
                piracies = false,
            )
            # Aqua.test_unbound_args remains *disabled* (re-checked 2026-07-15,
            # still fails) because of the NTuple{Dim,T}-parametrised
            # constructors in src/incompressible/boundary_conditions.jl
            # (FixedVelocityBC, FlowRateInletBC). Tracked in
            # test/QUALITY_LEDGER.toml.
            Aqua.test_ambiguities(FiniteVolumeMethod) # don't pick up Base and Core...
        end
    end
end
