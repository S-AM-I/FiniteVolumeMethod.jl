# ============================================================
# Layer 3: SciML Adapters / Accessors
# ============================================================
#
# All canonical SciML problem construction, remake behavior, cache
# layout, and solution-accessor logic is collected here.

# The semidiscrete cache/state-mapping/ODE-construction/contract/accessor
# files moved into the Hyperbolic submodule (Layer 2) in Stage 3e; the
# incompressible solution/façade moved into Collocated in Stage 3f. The
# cross-family glue below stays flat until the sciml/ relocation step.
# sciml_structures.jl and remake.jl define the SciMLStructures/remake
# methods for IncompressibleProblem — import so they extend against the
# Collocated-owned type.
import .Collocated: IncompressibleProblem
include("../core/symbolic_indexing.jl")
include("../core/sciml_structures.jl")
include("../remake.jl")
