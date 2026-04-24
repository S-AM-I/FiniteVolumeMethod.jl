# linear_solvers/petsc_stub.jl — PETSc.jl weak-dep stub.

"""
    PETScLinearSolver(options...)

Marker type activated by `FVMPETScExt` when `PETSc.jl` is loaded.
Without PETSc the constructor errors with a helpful message. Provides
large distributed direct / Krylov solves beyond what `LinearSolve.jl`
natively wraps.
"""
struct PETScLinearSolver
    options::Dict{Symbol, Any}
    _petsc_loaded::Bool
end

function PETScLinearSolver(; options...)
    return error(
        "PETSc.jl required — add `using PETSc` to activate FVMPETScExt",
    )
end
