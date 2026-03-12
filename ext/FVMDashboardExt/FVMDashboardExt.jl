module FVMDashboardExt

using FiniteVolumeMethod
using JSON3

using StaticArrays: SVector
using SciMLBase: DiscreteCallback

include("callbacks.jl")
include("export_import.jl")

end # module
