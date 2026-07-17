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

# Collocated cell-centered operators (Phase 0 — OpenFOAM-style FVM).
# Load after Parabolic: collocated BC handling dispatches on
# AbstractBoundaryCondition (a Parabolic export).
include("../collocated/types.jl")
include("../collocated/interpolation.jl")
include("../collocated/gradient.jl")
include("../collocated/laplacian.jl")
include("../collocated/divergence.jl")
include("../collocated/ddt.jl")
include("../collocated/cyclic.jl")
