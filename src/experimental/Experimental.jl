# ============================================================
# Experimental — quarantined research scaffolds (Stage 3h)
# ============================================================
#
# Everything in this module is EXPERIMENTAL: either its
# validation/manifest.toml feature is `experimental` (mpi_parallel;
# pressure_based/adjoint/mesh_generation as smoke-only evidence under
# experimental features) or it has no manifest coverage at all
# (aeroacoustics, population_balance, solid_mechanics, fsi). Nothing
# here is validated for production use. Entry points emit a one-time
# warning per feature via `_experimental_warn`.
#
# Names are temporarily re-exported at the top level (Stage-4 export
# curation removes them from the default surface).

module Experimental

using LinearAlgebra
using Printf: Printf, @sprintf
using StaticArrays: StaticArrays, SVector

using ..Geometry
using ..Numerics
# _dispatch_solve is unexported as of Stage 4c; the pressure_based,
# solid_mechanics, and adjoint solver files call it bare.
using ..Numerics: _dispatch_solve
using ..VertexConditions
using ..Parabolic
using ..Collocated
using ..Hyperbolic

# pressure_based/ calls these unexported incompressible/cyclic internals
# and solid_mechanics/linear_elasticity.jl uses _face_tag — a temporary
# seam into Collocated, curated in Stage 4.
import ..Collocated: compute_HbyA_flux, collect_cyclic_pairs,
    _cyclic_cell_pairs, expand_bcs_pressure, _extract_component,
    _make_scalar_field, _needs_pressure_reference, _velocity_labels,
    update_boundary_velocity!, update_boundary_pressure!,
    update_boundary_cyclic!, apply_cyclic_to_equation!,
    under_relax_momentum!, _set_component!, fix_pressure_reference!,
    _snapshot_old_time!, _face_tag
# mesh_generation/octree.jl adds an is_leaf(::Octree) method to the shared
# generic owned by Hyperbolic (AMRBlock methods) — import so it extends
# instead of shadowing (the amr/semidiscrete_amr dispatch fractures
# otherwise).
import ..Hyperbolic: is_leaf

# --- one-time experimental-use warning, per feature ---
const _EXP_WARNED = Set{Symbol}()

"""
    _experimental_warn(feature::Symbol)

Emit a single warning the first time an experimental `feature`'s entry
point is called in a session. Never called at include/precompile time.
"""
function _experimental_warn(feature::Symbol)
    if !(feature in _EXP_WARNED)
        push!(_EXP_WARNED, feature)
        @warn "FiniteVolumeMethod: `$feature` is an EXPERIMENTAL scaffold — not validated for production use." maxlog = 1
    end
    return nothing
end

export AbstractAdjointAlgorithm, AbstractRheology, AbstractThermoModel,
    AitkenRelaxation, BirdCarreauRheology, BoussinesqThermo, CassonRheology,
    CoolPropFluid, CurleSurface, FSIInterface, FWHObserver, FWHSurface,
    GmshPipeline, HerschelBulkleyRheology, IdealGas, IncompressibleThermo,
    IsotropicElastic, LocalFVMMesh, LocalMeshData, NewtonianRheology, Octree,
    PowerLawRheology, SnappyMesher, SolidDisplacementProblem, SteadyAdjoint,
    SutherlandGas, SutherlandViscosity, TransientAdjoint, auto_remediate!,
    beta_at, build_local_mesh, build_octree, cantilever_tip_deflection,
    count_leaves, cp_at, curle_dipole_pressure, density_at, distribute_mesh,
    extract_local_mesh, fwh_farassat1a, fwh_monopole_pressure,
    halo_exchange!, interface_residual_norm, intersects_sphere,
    is_compressible, partition_mesh_metis, partition_rcb,
    qmom_moment_source_aggregation, qmom_moment_source_breakage,
    qmom_moment_source_growth, qmom_recover_abscissae_weights,
    refine_near_sphere!, run_gmsh_pipeline, small_strain_tensor,
    solve_adjoint, solve_simple_distributed, solve_steady_adjoint,
    solve_transient_adjoint, stress_tensor, subdivide!, update_aitken!,
    verify_adjoint_gradient, viscosity_at

# Include order preserved verbatim from the pre-3h flat order (the
# pressure_based split — thermo first, compressible last — and the
# qmom-before-types quirk are both load-bearing).
include("pressure_based/thermo_models.jl")
include("pressure_based/rheology.jl")
include("pressure_based/coolprop_stub.jl")
include("aeroacoustics/fwh.jl")
include("aeroacoustics/pml.jl")
include("population_balance/qmom.jl")
include("population_balance/types.jl")
include("population_balance/dqmom.jl")
include("population_balance/class_method.jl")
include("solid_mechanics/types.jl")
include("solid_mechanics/linear_elasticity.jl")
include("solid_mechanics/finite_strain.jl")
include("solid_mechanics/solvers.jl")
include("fsi/coupling.jl")
include("fsi/interface.jl")
include("fsi/partitioned.jl")
include("mesh_generation/octree.jl")
include("mesh_generation/stl_reader.jl")
include("mesh_generation/snap.jl")
include("mesh_generation/snappy.jl")
include("mesh_generation/gmsh_pipeline.jl")
include("pressure_based/eos_coupling.jl")
include("pressure_based/compressible_simple.jl")
include("pressure_based/compressible_pimple.jl")
include("adjoint/types.jl")
include("adjoint/steady.jl")
include("adjoint/checkpointing.jl")
include("adjoint/reverse_sweep.jl")
include("adjoint/transient.jl")
include("adjoint/solvers.jl")
include("parallel/stubs.jl")
include("parallel/rcb_partitioner.jl")
include("parallel/local_mesh.jl")
include("parallel/metis_stub.jl")

end # module Experimental
