# ============================================================
# Layer 3: SciML Adapters / Accessors
# ============================================================
#
# All canonical SciML problem construction, remake behavior, cache
# layout, and solution-accessor logic is collected here.

# The semidiscrete cache/state-mapping/ODE-construction/contract/accessor
# files moved into the Hyperbolic submodule (Layer 2) in Stage 3e. The
# cross-family glue below stays flat until the sciml/ relocation step.
include("../core/symbolic_indexing.jl")
include("../core/sciml_structures.jl")
include("../remake.jl")
include("../incompressible/solution.jl")
include("../incompressible/sciml_interface.jl")
