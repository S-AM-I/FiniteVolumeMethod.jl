using FiniteVolumeMethod
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

@testset verbose = true "FiniteVolumeMethod.jl" begin
    @testset verbose = true "Geometry" begin
        safe_include("geometry.jl")
    end
    @testset verbose = true "Conditions" begin
        safe_include("conditions.jl")
    end
    @testset verbose = true "Robin BCs" begin
        safe_include("robin.jl")
    end
    @testset verbose = true "Parabolic Mesh" begin
        safe_include("parabolic_mesh.jl")
    end
    @testset verbose = true "Collocated Assembly Benchmark (Stage 1a-c)" begin
        safe_include("assembly_bench.jl")
    end
    @testset verbose = true "SciML Contract Uniform (Stage 1d)" begin
        safe_include("sciml_contract_uniform.jl")
    end
    @testset verbose = true "MPI Partition (Stage 2)" begin
        safe_include("mpi_partition.jl")
    end
    @testset verbose = true "Pressure-Based Thermo + Rheology (Stage 3)" begin
        safe_include("pressure_based_models.jl")
    end
    @testset verbose = true "Turbulence Correctness (Stage 4)" begin
        safe_include("turbulence_correctness.jl")
    end
    @testset verbose = true "Phase Correctness (Stage 5)" begin
        safe_include("stage5_correctness.jl")
    end
    @testset verbose = true "Industrial Physics (Stage 6)" begin
        safe_include("stage6_physics.jl")
    end
    @testset verbose = true "Coupled Physics (Stage 7)" begin
        safe_include("stage7_coupled.jl")
    end
    @testset verbose = true "Meshing + AMR (Stage 8)" begin
        safe_include("stage8_meshing_amr.jl")
    end
    @testset verbose = true "SciML Deep Integration (Stage 9)" begin
        safe_include("stage9_sciml.jl")
    end
    # Fast V&V gates (pure-operator MMS; < 5s) run unconditionally.
    @testset verbose = true "V&V: Laplacian operator MMS" begin
        safe_include("v_and_v_laplacian_mms.jl")
    end
    @testset verbose = true "V&V: Gradient + Divergence MMS" begin
        safe_include("v_and_v_operator_mms.jl")
    end
    @testset verbose = true "V&V: Rhie-Chow interpolation" begin
        safe_include("v_and_v_rhie_chow.jl")
    end
    @testset verbose = true "V&V: Laplacian on skewed mesh" begin
        safe_include("v_and_v_laplacian_skewed.jl")
    end
    @testset verbose = true "V&V: Temporal ddt MMS" begin
        safe_include("v_and_v_temporal_mms.jl")
    end
    @testset verbose = true "V&V: Poiseuille channel" begin
        safe_include("v_and_v_poiseuille.jl")
    end
    @testset verbose = true "V&V: Heat conduction analytical" begin
        safe_include("v_and_v_heat_conduction.jl")
    end
    @testset verbose = true "V&V: Stokes terminal velocity" begin
        safe_include("v_and_v_stokes_terminal.jl")
    end
    @testset verbose = true "V&V: GCL invariances" begin
        safe_include("v_and_v_gcl.jl")
    end
    @testset verbose = true "V&V: P1 radiation slab" begin
        safe_include("v_and_v_p1_slab.jl")
    end
    @testset verbose = true "V&V: VOF translation" begin
        safe_include("v_and_v_vof_translation.jl")
    end
    @testset verbose = true "V&V: Species advection-diffusion" begin
        safe_include("v_and_v_species_ad.jl")
    end
    @testset verbose = true "V&V: k-ε DHIT" begin
        safe_include("v_and_v_kepsilon_dhit.jl")
    end
    @testset verbose = true "V&V: Smagorinsky LES" begin
        safe_include("v_and_v_smagorinsky.jl")
    end
    @testset verbose = true "V&V: Postprocessing kinematics" begin
        safe_include("v_and_v_postprocessing.jl")
    end
    @testset verbose = true "V&V: Unsteady heat" begin
        safe_include("v_and_v_unsteady_heat.jl")
    end
    @testset verbose = true "V&V: Couette flow" begin
        safe_include("v_and_v_couette.jl")
    end
    @testset verbose = true "V&V: k-ε log-layer equilibrium" begin
        safe_include("v_and_v_kepsilon_loglayer.jl")
    end
    @testset verbose = true "V&V: VOF plane wave" begin
        safe_include("v_and_v_vof_planewave.jl")
    end
    @testset verbose = true "V&V: P1 radiative equilibrium" begin
        safe_include("v_and_v_p1_equilibrium.jl")
    end
    @testset verbose = true "V&V: Schiller-Naumann drag" begin
        safe_include("v_and_v_schiller_naumann.jl")
    end
    @testset verbose = true "V&V: EDM combustion algebra" begin
        safe_include("v_and_v_edm.jl")
    end
    @testset verbose = true "V&V: WALE LES model" begin
        safe_include("v_and_v_wale.jl")
    end
    @testset verbose = true "V&V: GCL rotation" begin
        safe_include("v_and_v_gcl_rotation.jl")
    end
    @testset verbose = true "V&V: Courant + Q-sign" begin
        safe_include("v_and_v_courant.jl")
    end
    @testset verbose = true "V&V: Boussinesq buoyancy" begin
        safe_include("v_and_v_boussinesq.jl")
    end
    @testset verbose = true "V&V: Ranz-Marshall particle heat" begin
        safe_include("v_and_v_ranz_marshall.jl")
    end
    @testset verbose = true "V&V: Mesh sweep flux" begin
        safe_include("v_and_v_mesh_flux.jl")
    end
    @testset verbose = true "V&V: Radiation source algebra" begin
        safe_include("v_and_v_radiation_source.jl")
    end
    @testset verbose = true "V&V: VOF mixture properties" begin
        safe_include("v_and_v_vof_mixture.jl")
    end
    @testset verbose = true "V&V: Arrhenius kinetics" begin
        safe_include("v_and_v_arrhenius.jl")
    end
    @testset verbose = true "V&V: k-ω turbulence" begin
        safe_include("v_and_v_komega.jl")
    end
    @testset verbose = true "V&V: LES filter width" begin
        safe_include("v_and_v_filter_width.jl")
    end
    @testset verbose = true "V&V: Field statistics" begin
        safe_include("v_and_v_field_stats.jl")
    end
    @testset verbose = true "V&V: PISO transient stability" begin
        safe_include("v_and_v_piso_decay.jl")
    end
    @testset verbose = true "V&V: Linear solvers" begin
        safe_include("v_and_v_linear_solvers.jl")
    end
    @testset verbose = true "V&V: Mesh geometry invariants" begin
        safe_include("v_and_v_mesh_geometry.jl")
    end
    @testset verbose = true "V&V: Spalart-Allmaras" begin
        safe_include("v_and_v_spalart_allmaras.jl")
    end
    @testset verbose = true "V&V: Spray breakup" begin
        safe_include("v_and_v_spray.jl")
    end
    @testset verbose = true "V&V: CSF surface tension" begin
        safe_include("v_and_v_csf.jl")
    end
    @testset verbose = true "V&V: FR/ED combustion" begin
        safe_include("v_and_v_fred.jl")
    end
    @testset verbose = true "V&V: fvDOM quadrature" begin
        safe_include("v_and_v_fvdom_quadrature.jl")
    end
    @testset verbose = true "V&V: ALE-corrected flux" begin
        safe_include("v_and_v_ale_flux.jl")
    end
    @testset verbose = true "V&V: CHT interface flux" begin
        safe_include("v_and_v_cht_interface.jl")
    end
    @testset verbose = true "V&V: Wall quantities" begin
        safe_include("v_and_v_wall_quantities.jl")
    end
    @testset verbose = true "V&V: Solver config dispatch" begin
        safe_include("v_and_v_solver_config.jl")
    end
    @testset verbose = true "V&V: Strain rate primitive" begin
        safe_include("v_and_v_strain_rate.jl")
    end
    @testset verbose = true "V&V: Wall functions" begin
        safe_include("v_and_v_wall_functions.jl")
    end
    @testset verbose = true "V&V: Incompressible SciML interface" begin
        safe_include("v_and_v_incompressible_sciml.jl")
    end
    @testset verbose = true "V&V: Effective conductivity k_eff" begin
        safe_include("v_and_v_k_eff.jl")
    end
    @testset verbose = true "V&V: Particle state" begin
        safe_include("v_and_v_particle_state.jl")
    end
    @testset verbose = true "V&V: Laplacian mesh motion" begin
        safe_include("v_and_v_laplacian_motion.jl")
    end
    @testset verbose = true "V&V: VOF compression flux" begin
        safe_include("v_and_v_vof_compression.jl")
    end
    @testset verbose = true "V&V: Marshak wall BC" begin
        safe_include("v_and_v_marshak.jl")
    end
    @testset verbose = true "V&V: Combustion properties" begin
        safe_include("v_and_v_combustion_props.jl")
    end
    @testset verbose = true "V&V: Nusselt + y+" begin
        safe_include("v_and_v_nusselt.jl")
    end
    @testset verbose = true "V&V: LES turbulence state" begin
        safe_include("v_and_v_les_state.jl")
    end
    @testset verbose = true "V&V: Incompressible remake" begin
        safe_include("v_and_v_incompressible_remake.jl")
    end
    @testset verbose = true "V&V: Thermal types" begin
        safe_include("v_and_v_thermal_types.jl")
    end
    @testset verbose = true "V&V: RANS turbulence state" begin
        safe_include("v_and_v_rans_state.jl")
    end
    @testset verbose = true "V&V: SolidBodyMotion" begin
        safe_include("v_and_v_solid_body_motion.jl")
    end
    @testset verbose = true "V&V: RadiationState + P1Model" begin
        safe_include("v_and_v_radiation_state.jl")
    end
    @testset verbose = true "V&V: VOFState + TwoPhaseProperties" begin
        safe_include("v_and_v_vof_state.jl")
    end
    @testset verbose = true "V&V: DPM dispatch" begin
        safe_include("v_and_v_dpm_dispatch.jl")
    end
    @testset verbose = true "V&V: Heat release primitive" begin
        safe_include("v_and_v_heat_release.jl")
    end
    @testset verbose = true "V&V: Force coefficients" begin
        safe_include("v_and_v_force_coefficients.jl")
    end
    @testset verbose = true "V&V: LES dispatch" begin
        safe_include("v_and_v_les_dispatch.jl")
    end
    @testset verbose = true "V&V: Mesh accessors" begin
        safe_include("v_and_v_mesh_accessors.jl")
    end
    @testset verbose = true "V&V: IncompressibleState" begin
        safe_include("v_and_v_inc_state.jl")
    end
    @testset verbose = true "V&V: CHT problem" begin
        safe_include("v_and_v_cht_problem.jl")
    end
    @testset verbose = true "V&V: Turbulence inlet BCs" begin
        safe_include("v_and_v_turbulence_inlet.jl")
    end
    @testset verbose = true "V&V: DDES hybrid" begin
        safe_include("v_and_v_ddes.jl")
    end
    @testset verbose = true "V&V: clip_alpha!" begin
        safe_include("v_and_v_clip_alpha.jl")
    end
    @testset verbose = true "V&V: update_mesh!" begin
        safe_include("v_and_v_mesh_update.jl")
    end
    @testset verbose = true "V&V: FvDOMModel" begin
        safe_include("v_and_v_fvdom_model.jl")
    end
    @testset verbose = true "V&V: Species index lookup" begin
        safe_include("v_and_v_species_index.jl")
    end
    @testset verbose = true "V&V: find_nearest_cell" begin
        safe_include("v_and_v_cell_lookup.jl")
    end
    @testset verbose = true "V&V: Courant edge cases" begin
        safe_include("v_and_v_courant_edge.jl")
    end
    @testset verbose = true "V&V: Field constructors" begin
        safe_include("v_and_v_field_constructors.jl")
    end
    @testset verbose = true "V&V: PV coupling constructors" begin
        safe_include("v_and_v_pv_coupling.jl")
    end
    @testset verbose = true "V&V: CollocatedEquation" begin
        safe_include("v_and_v_equation_types.jl")
    end
    @testset verbose = true "V&V: Q_gen volumetric heat" begin
        safe_include("v_and_v_qgen.jl")
    end
    @testset verbose = true "V&V: SST blending algebra" begin
        safe_include("v_and_v_sst_blend.jl")
    end
    @testset verbose = true "V&V: LES primitives (test filter + contract)" begin
        safe_include("v_and_v_test_filter.jl")
    end
    @testset verbose = true "V&V: MULES flux limiter" begin
        safe_include("v_and_v_mules.jl")
    end
    @testset verbose = true "V&V: P1 solver invariants" begin
        safe_include("v_and_v_p1_solver.jl")
    end
    @testset verbose = true "V&V: EDC reaction rates" begin
        safe_include("v_and_v_edc.jl")
    end
    @testset verbose = true "V&V: PSI-cell two-way coupling" begin
        safe_include("v_and_v_psi_cell.jl")
    end
    @testset verbose = true "V&V: verify_gcl diagnostic" begin
        safe_include("v_and_v_verify_gcl.jl")
    end
    @testset verbose = true "V&V: field sampling" begin
        safe_include("v_and_v_sampling.jl")
    end
    @testset verbose = true "V&V: AbstractLinearOperator" begin
        safe_include("v_and_v_linear_operator.jl")
    end
    @testset verbose = true "V&V: mesh quality report" begin
        safe_include("v_and_v_mesh_quality.jl")
    end
    @testset verbose = true "V&V: temporal ddt assembly" begin
        safe_include("v_and_v_ddt.jl")
    end
    @testset verbose = true "V&V: continuity residual primitives" begin
        safe_include("v_and_v_continuity.jl")
    end
    @testset verbose = true "V&V: polyhedral volumes" begin
        safe_include("v_and_v_polyhedral_volumes.jl")
    end
    # Wave 1 (v3.102) — pressure-based + turbulence + thermal + VOF + gradient
    @testset verbose = true "V&V: compressible pressure-based" begin
        safe_include("v_and_v_compressible.jl")
    end
    @testset verbose = true "V&V: Durbin realizability + full-tensor P_k" begin
        safe_include("v_and_v_durbin.jl")
    end
    @testset verbose = true "V&V: equilibrium WMLES" begin
        safe_include("v_and_v_wmles.jl")
    end
    @testset verbose = true "V&V: full-tensor Germano invariants" begin
        safe_include("v_and_v_full_germano.jl")
    end
    @testset verbose = true "V&V: wall-function skew projection" begin
        safe_include("v_and_v_wall_skew.jl")
    end
    @testset verbose = true "V&V: per-face CHT coupling" begin
        safe_include("v_and_v_per_face_cht.jl")
    end
    @testset verbose = true "V&V: enthalpy energy equation" begin
        safe_include("v_and_v_enthalpy.jl")
    end
    @testset verbose = true "V&V: MULES-wired α transport" begin
        safe_include("v_and_v_mules_integration.jl")
    end
    @testset verbose = true "V&V: isoAdvector interface flux" begin
        safe_include("v_and_v_iso_advector.jl")
    end
    @testset verbose = true "V&V: contact angle (static + Cox-Voinov)" begin
        safe_include("v_and_v_contact_angle.jl")
    end
    @testset verbose = true "V&V: over-relaxed non-orthogonal correction" begin
        safe_include("v_and_v_over_relaxed.jl")
    end
    @testset verbose = true "V&V: least-squares gradient" begin
        safe_include("v_and_v_lsq_gradient.jl")
    end
    # Wave 2 (v3.103) — combustion + radiation + Lagrangian + dynamic mesh + cavitation/porous
    @testset verbose = true "V&V: variable Lewis number" begin
        safe_include("v_and_v_variable_lewis.jl")
    end
    @testset verbose = true "V&V: multi-step mechanism" begin
        safe_include("v_and_v_multi_step.jl")
    end
    @testset verbose = true "V&V: FGM tabulated chemistry" begin
        safe_include("v_and_v_fgm.jl")
    end
    @testset verbose = true "V&V: fvDOM scattering" begin
        safe_include("v_and_v_scattering.jl")
    end
    @testset verbose = true "V&V: SN quadratures (S6/S8/S12)" begin
        safe_include("v_and_v_sn_quadratures.jl")
    end
    @testset verbose = true "V&V: WSGGM weighted-sum-of-grey-gases" begin
        safe_include("v_and_v_wsggm.jl")
    end
    @testset verbose = true "V&V: hard-sphere DEM collision" begin
        safe_include("v_and_v_hard_sphere_dem.jl")
    end
    @testset verbose = true "V&V: agglomeration" begin
        safe_include("v_and_v_agglomeration.jl")
    end
    @testset verbose = true "V&V: primary breakup (KH-ACT + LISA)" begin
        safe_include("v_and_v_primary_breakup.jl")
    end
    @testset verbose = true "V&V: injection patterns" begin
        safe_include("v_and_v_injection_patterns.jl")
    end
    @testset verbose = true "V&V: 6-DOF rigid body" begin
        safe_include("v_and_v_six_dof.jl")
    end
    @testset verbose = true "V&V: topoChanger" begin
        safe_include("v_and_v_topo_changer.jl")
    end
    @testset verbose = true "V&V: overset/chimera interpolation" begin
        safe_include("v_and_v_overset.jl")
    end
    @testset verbose = true "V&V: AMI sliding interface" begin
        safe_include("v_and_v_ami.jl")
    end
    @testset verbose = true "V&V: Kunz cavitation" begin
        safe_include("v_and_v_kunz.jl")
    end
    @testset verbose = true "V&V: Schnerr-Sauer cavitation" begin
        safe_include("v_and_v_schnerr_sauer.jl")
    end
    @testset verbose = true "V&V: Merkle cavitation" begin
        safe_include("v_and_v_merkle.jl")
    end
    @testset verbose = true "V&V: Darcy-Forchheimer porous" begin
        safe_include("v_and_v_darcy_forchheimer.jl")
    end
    # Wave 3 (v3.104) — MRF + solid mechanics + FSI + aeroacoustics + PBM
    @testset verbose = true "V&V: MRF single-zone Coriolis+centrifugal" begin
        safe_include("v_and_v_mrf_single_zone.jl")
    end
    @testset verbose = true "V&V: MRF multi-zone" begin
        safe_include("v_and_v_mrf_multi_zone.jl")
    end
    @testset verbose = true "V&V: linear elasticity" begin
        safe_include("v_and_v_linear_elasticity.jl")
    end
    @testset verbose = true "V&V: finite strain (updated-Lagrangian)" begin
        safe_include("v_and_v_finite_strain.jl")
    end
    @testset verbose = true "V&V: Aitken relaxation" begin
        safe_include("v_and_v_aitken.jl")
    end
    @testset verbose = true "V&V: partitioned FSI" begin
        safe_include("v_and_v_partitioned_fsi.jl")
    end
    @testset verbose = true "V&V: FW-H aeroacoustics" begin
        safe_include("v_and_v_fwh.jl")
    end
    @testset verbose = true "V&V: PML sponge zones" begin
        safe_include("v_and_v_pml.jl")
    end
    @testset verbose = true "V&V: QMoM Wheeler inversion" begin
        safe_include("v_and_v_qmom.jl")
    end
    @testset verbose = true "V&V: DQMoM" begin
        safe_include("v_and_v_dqmom.jl")
    end
    @testset verbose = true "V&V: Class Method PBM" begin
        safe_include("v_and_v_class_method.jl")
    end
    # Wave 4 (v3.105) — mesh gen + collocated AMR + adjoint + KA/Enzyme + units/properties
    @testset verbose = true "V&V: octree primitive" begin
        safe_include("v_and_v_octree.jl")
    end
    @testset verbose = true "V&V: SnappyMesher stub" begin
        safe_include("v_and_v_snappy_stub.jl")
    end
    @testset verbose = true "V&V: Gmsh pipeline stub" begin
        safe_include("v_and_v_gmsh_pipeline.jl")
    end
    @testset verbose = true "V&V: ZZ error indicator" begin
        safe_include("v_and_v_zz_indicator.jl")
    end
    @testset verbose = true "V&V: residual error indicator" begin
        safe_include("v_and_v_residual_indicator.jl")
    end
    @testset verbose = true "V&V: collocated refinement/coarsening" begin
        safe_include("v_and_v_collocated_refine.jl")
    end
    @testset verbose = true "V&V: steady adjoint identity" begin
        safe_include("v_and_v_steady_adjoint.jl")
    end
    @testset verbose = true "V&V: transient adjoint stub" begin
        safe_include("v_and_v_transient_adjoint_stub.jl")
    end
    @testset verbose = true "V&V: KA backend CPU path" begin
        safe_include("v_and_v_ka_backend.jl")
    end
    @testset verbose = true "V&V: Enzyme full-solver stub" begin
        safe_include("v_and_v_enzyme_stub.jl")
    end
    @testset verbose = true "V&V: runtime expression BC" begin
        safe_include("v_and_v_expression_bc.jl")
    end
    @testset verbose = true "V&V: Unitful hook" begin
        safe_include("v_and_v_unitful.jl")
    end
    @testset verbose = true "V&V: CoolProp stub" begin
        safe_include("v_and_v_coolprop_stub.jl")
    end
    @testset verbose = true "V&V: PETSc stub" begin
        safe_include("v_and_v_petsc_stub.jl")
    end
    # Wave 5 (v3.106) — true MPI decomposition + Eulerian two-fluid
    @testset verbose = true "V&V: LocalFVMMesh partition view" begin
        safe_include("v_and_v_local_mesh.jl")
    end
    @testset verbose = true "V&V: Metis partition stub" begin
        safe_include("v_and_v_metis_partition_stub.jl")
    end
    @testset verbose = true "V&V: two-fluid drag closures" begin
        safe_include("v_and_v_drag_closures.jl")
    end
    @testset verbose = true "V&V: Eulerian two-fluid (experimental)" begin
        safe_include("v_and_v_two_fluid.jl")
    end
    # v3.1 roadmap — production two-fluid + transient adjoint + snappy + IDDES-full + primary-breakup FSI
    @testset verbose = true "V&V: two-fluid coupled solver (v3.1)" begin
        safe_include("v_and_v_two_fluid_solver.jl")
    end
    @testset verbose = true "V&V: transient adjoint (checkpointed)" begin
        safe_include("v_and_v_transient_adjoint.jl")
    end
    @testset verbose = true "V&V: IDDES full Shur-2008 shielding" begin
        safe_include("v_and_v_iddes_full.jl")
    end
    @testset verbose = true "V&V: primary-breakup FSI coupling" begin
        safe_include("v_and_v_primary_breakup_fsi.jl")
    end
    @testset verbose = true "V&V: STL reader" begin
        safe_include("v_and_v_stl_reader.jl")
    end
    @testset verbose = true "V&V: snappyHexMesh native (castellated + snap)" begin
        safe_include("v_and_v_snappy_native.jl")
    end
    # Grid-convergence study runs three full SIMPLE solves — slower.
    # Gated behind FVM_RUN_VANDV like Ghia.
    if get(ENV, "FVM_RUN_VANDV", "false") == "true"
        @testset verbose = true "V&V: Poiseuille grid convergence" begin
            safe_include("v_and_v_poiseuille_convergence.jl")
        end
    end
    # Slow V&V (full SIMPLE solve, ~1-2 min each) gated behind an env flag.
    if get(ENV, "FVM_RUN_VANDV", "false") == "true"
        @testset verbose = true "V&V: Ghia lid-driven cavity Re=100" begin
            safe_include("v_and_v_ghia_cavity.jl")
        end
    end
    @testset verbose = true "Problem" begin
        safe_include("problem.jl")
    end
    @testset verbose = true "Equations" begin
        safe_include("equations.jl")
    end
    @testset verbose = true "Schemes" begin
        safe_include("schemes.jl")
    end
    @testset verbose = true "Advanced BCs" begin
        safe_include("advanced_bcs.jl")
    end
    @testset verbose = true "Physics Models" begin
        safe_include("physics.jl")
    end
    @testset verbose = true "Hyperbolic Solver" begin
        safe_include("hyperbolic.jl")
    end
    @testset verbose = true "Hyperbolic 2D + HLLC" begin
        safe_include("hyperbolic_2d.jl")
    end
    @testset verbose = true "MHD + HLLD" begin
        safe_include("mhd.jl")
    end
    @testset verbose = true "MHD 2D + CT" begin
        safe_include("mhd_2d.jl")
    end
    @testset verbose = true "Navier-Stokes" begin
        safe_include("navier_stokes.jl")
    end
    @testset verbose = true "SRMHD" begin
        safe_include("srmhd.jl")
    end
    @testset verbose = true "SRMHD 2D" begin
        safe_include("srmhd_2d.jl")
    end
    @testset verbose = true "GRMHD" begin
        safe_include("grmhd.jl")
    end
    @testset verbose = true "GRMHD 2D" begin
        safe_include("grmhd_2d.jl")
    end
    @testset verbose = true "Hyperbolic 3D" begin
        safe_include("hyperbolic_3d.jl")
    end
    @testset verbose = true "MHD 3D" begin
        safe_include("mhd_3d.jl")
    end
    @testset verbose = true "AMR" begin
        safe_include("amr.jl")
    end
    @testset verbose = true "WENO" begin
        safe_include("weno.jl")
    end
    @testset verbose = true "IMEX" begin
        safe_include("imex.jl")
    end
    @testset verbose = true "Unstructured Hyperbolic" begin
        safe_include("unstructured_hyperbolic.jl")
    end
    @testset verbose = true "Multi-Physics Coupling" begin
        safe_include("coupling.jl")
    end
    @testset verbose = true "Performance & Threading" begin
        safe_include("performance.jl")
    end
    @testset verbose = true "Performance Calibration" begin
        safe_include("performance_calibration.jl")
    end
    @testset verbose = true "Advanced Numerics" begin
        safe_include("advanced_numerics.jl")
    end
    @testset verbose = true "Extended Physics" begin
        safe_include("extended_physics.jl")
    end
    @testset verbose = true "Reactive Euler" begin
        safe_include("reactive_euler.jl")
    end
    @testset verbose = true "Incompressible NS" begin
        safe_include("incompressible.jl")
    end
    @testset verbose = true "Incompressible SciML Compliance" begin
        safe_include("incompressible_sciml.jl")
    end
    @testset verbose = true "RANS Turbulence" begin
        safe_include("turbulence_rans.jl")
    end
    @testset verbose = true "Conjugate Heat Transfer" begin
        safe_include("thermal.jl")
    end
    @testset verbose = true "Polyhedral Mesh I/O" begin
        safe_include("mesh_io.jl")
    end
    @testset verbose = true "Post-Processing" begin
        safe_include("postprocessing.jl")
    end
    @testset verbose = true "Linear Solvers" begin
        safe_include("linear_solvers.jl")
    end
    @testset verbose = true "LES Turbulence" begin
        safe_include("turbulence_les.jl")
    end
    @testset verbose = true "Multiphase VOF" begin
        safe_include("multiphase_vof.jl")
    end
    @testset verbose = true "Radiation" begin
        safe_include("radiation.jl")
    end
    @testset verbose = true "Combustion" begin
        safe_include("combustion.jl")
    end
    @testset verbose = true "Lagrangian DPM" begin
        safe_include("lagrangian_dpm.jl")
    end
    @testset verbose = true "Dynamic Mesh" begin
        safe_include("dynamic_mesh.jl")
    end
    @testset verbose = true "Remaining Features" begin
        safe_include("remaining_features.jl")
    end
    @testset verbose = true "README" begin
        safe_include("README.jl")
    end

    @testset verbose = true "Coordinate Systems" begin
        safe_include("test_coordinate_systems.jl")
    end

    @testset verbose = true "Dashboard" begin
        safe_include("test_dashboard.jl")
    end

    @testset verbose = true "I/O" begin
        safe_include("io.jl")
    end

    @testset verbose = true "Remake" begin
        safe_include("test_remake.jl")
    end

    @testset verbose = true "Semidiscrete Core" begin
        safe_include("semidiscrete.jl")
    end

    @testset verbose = true "Semidiscrete MHD" begin
        safe_include("semidiscrete_mhd.jl")
    end

    @testset verbose = true "Semidiscrete AMR" begin
        safe_include("semidiscrete_amr.jl")
    end

    @testset verbose = true "Semidiscrete IMEX" begin
        safe_include("semidiscrete_imex.jl")
    end

    @testset verbose = true "SciML Contract" begin
        safe_include("sciml_contract.jl")
    end

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

    @testset verbose = true "Aqua" begin
        # piracies disabled: Aqua 0.7 traverses Core.TypeName.mt which was
        # removed in Julia 1.12 (FieldError). Re-enable once Aqua >= 0.8.
        Aqua.test_all(
            FiniteVolumeMethod;
            ambiguities = false, project_extras = false, unbound_args = false,
            piracies = false,
        )
        Aqua.test_unbound_args(FiniteVolumeMethod)
        Aqua.test_ambiguities(FiniteVolumeMethod) # don't pick up Base and Core...
    end

    @testset verbose = true "Environment Integrity" begin
        safe_include("environment_integrity.jl")
    end

    @testset verbose = true "SciML Audit" begin
        safe_include("sciml_audit.jl")
    end

    @testset verbose = true "Repository Governance" begin
        safe_include("repository_governance.jl")
    end

    @testset verbose = true "Reproducibility Bundles" begin
        safe_include("reproducibility_bundle.jl")
    end

    @testset verbose = true "Reference Artifacts" begin
        safe_include("reference_artifacts.jl")
    end

    @testset verbose = true "Backend Parity" begin
        safe_include("backend_parity.jl")
    end

    @testset verbose = true "Summary Replay" begin
        safe_include("summary_replay.jl")
    end

    @testset verbose = true "Quality Ledger" begin
        safe_include("quality_ledger.jl")
    end

    @testset verbose = true "Explicit Imports" begin
        safe_include("explicit_imports.jl")
    end
end
