# ============================================================
# Layer 1: Domain / Problem Definitions
# ============================================================
#
# Transitional layer file. As of Stage 3d the parabolic family lives in
# the Parabolic submodule; this layer wires the cell-vertex conditions
# engine, the Parabolic family, and the collocated Phase-0 operators.

include("../vertex_conditions/VertexConditions.jl")
using .VertexConditions

include("../parabolic/Parabolic.jl")
using .Parabolic

# The collocated (OpenFOAM-style) family: operators, incompressible
# SIMPLE/PISO/PIMPLE, multiphase, DPM, dynamic mesh, collocated AMR,
# post-processing, zone models, and nested Collocated.Physics
# (turbulence/thermal/radiation/combustion). Loads after Parabolic:
# its BC handling dispatches on AbstractBoundaryCondition.
include("../collocated/Collocated.jl")
using .Collocated
# Dispatch-fracture guards + qualified-internal passthroughs (Stage-3
# recipe): the flat pressure_based/ family calls unexported
# incompressible/cyclic internals, solid_mechanics uses _face_tag, and
# tests/validation/docs reach the remaining internals as
# FiniteVolumeMethod.<name> — all must resolve to the Collocated
# bindings (temporary over-import, curated in Stage 4).
import .Collocated: DynamicContactAngle, KHACTBreakup, LISABreakup,
    NoMassTransfer, StaticContactAngle, TwoFluidProblem,
    _cyclic_cell_pairs, _extract_component, _face_diffusivity, _face_tag,
    _make_incompressible_workspace, _make_scalar_field,
    _needs_pressure_reference, _particle_reynolds, _pimple_step!,
    _set_component!, _snapshot_old_time!, _velocity_labels,
    _non_ortho_E_magnitude, _zz_indicator_smoothed, add_darcy_forchheimer_source!,
    apply_contact_angle, apply_cyclic_to_equation!,
    assemble_isoadvector_flux!, collect_cyclic_pairs, compute_HbyA_flux,
    couple_primary_breakup_fsi!, cox_voinov_angle, expand_bcs_pressure,
    expand_pressure_bc, expand_velocity_bc, fix_pressure_reference!,
    kunz_cond_rate, kunz_rate, kunz_vap_rate, least_squares_gradient,
    merkle_cond_rate, merkle_rate, merkle_vap_rate, run!,
    schnerr_sauer_bubble_radius, schnerr_sauer_rate, solve_two_fluid,
    two_fluid_mixture_continuity_residual, under_relax_momentum!,
    update!, update_boundary_cyclic!, update_boundary_pressure!,
    update_boundary_velocity!, warn_experimental!,
    # Collocated.Physics internals (passed through Collocated)
    T_from_h, h_from_T, _sym_self_magnitude_sq, _durbin_C_T,
    _wall_projection, patankar_interface_coupling,
    EquilibriumWMLES, IDDES, WSGGMModel, _EDC_FALLBACK_MIXING_RATE,
    _apply_durbin_cap!, _blend, _cell_absorption, _ddes_length_scale,
    _ddes_shielding, _iddes_alpha, _iddes_f_B, _iddes_f_d_tilde,
    _iddes_f_dt, _iddes_f_e, _iddes_r_dl, _iddes_r_dt, _s12_quadrature,
    _s2_quadrature, _s4_quadrature, _s6_quadrature, _s8_quadrature,
    _sa_fv1, _species_index, _sst_F1, _sst_F2, _sym_contract,
    _test_filter, _update_turbulence!, compute_band_emissivity,
    compute_band_weight, enthalpy_bcs_from_temperature,
    enthalpy_field_from_temperature, iddes_blended_length,
    scattering_phase_value, scattering_source_contribution,
    solve_wsggm_radiation, temperature_from_enthalpy!,
    turbulent_viscosity_sa!, wmles_wall_nut, wmles_wall_shear,
    wsggm_effective_absorption
