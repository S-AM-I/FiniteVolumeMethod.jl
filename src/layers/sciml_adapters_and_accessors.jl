# ============================================================
# Layer 3: SciML Adapters / Accessors
# ============================================================
#
# All canonical SciML problem construction, remake behavior, cache
# layout, and solution-accessor logic is collected here.

include("../core/cache.jl")
include("../core/state_mapping.jl")
include("../core/cfl_callback.jl")
include("../core/callback_merge.jl")
include("../core/ode_construction.jl")
include("../core/split_construction.jl")
include("../core/sciml_contract.jl")
include("../core/results.jl")
include("../core/symbolic_indexing.jl")
include("../core/sciml_structures.jl")
include("../remake.jl")
include("../incompressible/solution.jl")
include("../incompressible/sciml_interface.jl")
