module FVMPETScExt

using FiniteVolumeMethod
using PETSc

# Override the stub constructor so that once PETSc.jl is loaded the
# `PETScLinearSolver` type can be created with options.
function FiniteVolumeMethod.PETScLinearSolver(; options...)
    return FiniteVolumeMethod.PETScLinearSolver(Dict{Symbol, Any}(options), true)
end

end
