# ============================================================
# Layer 1: Domain / Problem Definitions
# ============================================================
#
# Transitional layer file that groups the foundational problem,
# mesh, law, and type definitions while preserving the historical
# load order required by the current v2 overhaul.

include("../core/backends.jl")

include("../parabolic/types.jl")
include("../parabolic/mesh/types.jl")
include("../parabolic/mesh/structured.jl")
include("../parabolic/mesh/curvilinear.jl")
include("../parabolic/mesh/unstructured.jl")
include("../parabolic/mesh/fvm_mesh.jl")
include("../parabolic/mesh/io.jl")

# Polyhedral Mesh I/O (Phase 4)
# Depends on mesh types, io.jl (volume_tet, Node3D), fvm_mesh.jl (UnstructuredFVMMesh).
include("../mesh/polyhedral_volumes.jl")
include("../mesh/convert.jl")
include("../mesh/openfoam_io.jl")
include("../mesh/openfoam_writer.jl")
include("../mesh/quality.jl")

include("../parabolic/mesh/partitioning.jl")

include("../schemes/limiters.jl")

include("../parabolic/models.jl")
include("../parabolic/utils.jl")
include("../parabolic/boundary_conditions.jl")
include("../parabolic/gradients.jl")
include("../parabolic/limiters.jl")
include("../parabolic/schemes.jl")
include("../parabolic/compressible_fluxes.jl")
include("../parabolic/turbulence.jl")
include("../parabolic/particles.jl")
include("../parabolic/fsi.jl")
include("../parabolic/kernels.jl")
include("../parabolic/assembly/assembly_1d.jl")
include("../parabolic/assembly/assembly_2d.jl")
include("../parabolic/assembly/assembly_3d.jl")
include("../parabolic/assembly/assembly_cylindrical.jl")
include("../parabolic/assembly/assembly_spherical.jl")
include("../parabolic/assembly/assembly_unstructured.jl")
include("../parabolic/assembly/assembly_curvilinear.jl")
include("../parabolic/assembly/assembly_system.jl")

# Collocated cell-centered operators (Phase 0 — OpenFOAM-style FVM)
# Types must load after UnstructuredFVMMesh and AbstractBoundaryCondition;
# operators load here since they depend on mesh + BC types from Layer 1.
include("../collocated/types.jl")
include("../collocated/interpolation.jl")
include("../collocated/gradient.jl")
include("../collocated/laplacian.jl")
include("../collocated/divergence.jl")
include("../collocated/ddt.jl")

include("../coordinate_systems.jl")
include("../geometry.jl")
include("../conditions.jl")
include("../problem.jl")
