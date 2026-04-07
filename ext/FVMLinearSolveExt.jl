module FVMLinearSolveExt

using FiniteVolumeMethod
using LinearSolve: KrylovJL_CG, KrylovJL_BICGSTAB, KrylovJL_GMRES

FiniteVolumeMethod._try_krylov_solver(::Val{:cg}) = KrylovJL_CG()
FiniteVolumeMethod._try_krylov_solver(::Val{:bicgstab}) = KrylovJL_BICGSTAB()
FiniteVolumeMethod._try_krylov_solver(::Val{:gmres}) = KrylovJL_GMRES()

end # module
