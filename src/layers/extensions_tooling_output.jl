# ============================================================
# Layer 4: Extensions / Tooling / Output
# ============================================================
#
# Dashboard-facing types, output management, checkpointing, and the
# machine-readable capability contract live here.

include("../dashboard_types.jl")
include("../io/utils.jl")
include("../io/manager.jl")
include("../io/diagnostics.jl")
include("../io/vtk.jl")
include("../io/insitu.jl")
include("../io/registry.jl")
include("../io/hdf5.jl")
include("../io/checkpointing.jl")
include("../capabilities.jl")

# Post-Processing (Phase 12)
include("../postprocessing/field_operations.jl")
include("../postprocessing/wall_quantities.jl")
include("../postprocessing/forces.jl")
include("../postprocessing/sampling.jl")
include("../postprocessing/field_statistics.jl")
