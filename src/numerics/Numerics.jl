"""
    Numerics

Cross-family numerical kernels: execution backends, equations of state, flux
limiters, gradient reconstruction, MUSCL face reconstruction, unit handling,
kernel-abstraction stubs, and the per-field linear-solver configuration layer.
Depends only on `Geometry` (and external packages) — every solver family builds
on top of this module.
"""
module Numerics

using LinearAlgebra: Diagonal, diag, mul!
import LinearAlgebra
using SparseArrays: SparseMatrixCSC, sparse
using CommonSolve: solve
using DelaunayTriangulation: DelaunayTriangulation, get_neighbours, is_ghost_vertex,
    num_solid_vertices, get_adjacent, get_point, getxy, triangle_vertices,
    each_solid_vertex
using ..Geometry: FVMGeometry, get_triangle_props, _safe_get_triangle_props

include("backends.jl")
include("eos/eos_interface.jl")
include("eos/ideal_gas.jl")
include("eos/stiffened_gas.jl")
include("schemes/limiters.jl")
include("schemes/limiters_1d.jl")
include("schemes/gradients.jl")
include("schemes/muscl.jl")
include("units/units.jl")
include("units/unitful_integration.jl")
include("kernels/types.jl")
include("kernels/ka_stubs.jl")
include("kernels/enzyme_stub.jl")
include("linear_solvers/abstract_operator.jl")
include("linear_solvers/matrix_free.jl")
include("linear_solvers/preconditioners.jl")
include("linear_solvers/solver_config.jl")
include("linear_solvers/petsc_stub.jl")

# Public API (curated in Stage 4)
export AbstractBackend, CPUBackend, CUDASolverBackend,
    to_backend, to_host, supports_backend, backend_summary
export AbstractEOS, IdealGasEOS, StiffenedGasEOS,
    pressure, sound_speed, internal_energy, total_energy
export AbstractLimiter, MinmodLimiter, SuperbeeLimiter, VanLeerLimiter,
    VenkatakrishnanLimiter, BarthJespersenLimiter, KorenLimiter, OspreLimiter,
    minmod, superbee, van_leer, venkatakrishnan, barth_jespersen, koren, ospre,
    apply_limiter, select_limiter
export select_limiter_strategy, compute_slope_ratio_1d, limit_slope_1d
export AbstractGradientMethod, GreenGaussGradient, LeastSquaresGradient,
    reconstruct_gradient, reconstruct_gradient_at_edge,
    reconstruct_gradient_at_point, reconstruct_all_gradients
export MUSCLScheme, MUSCLFluxFunction, muscl_reconstruct_face_value,
    muscl_reconstruct_edge_values, muscl_advective_flux, muscl_diffusive_flux
export strip_units, is_dimensionless, as_si_velocity, as_si_density,
    as_si_viscosity, as_si_temperature, annotate_units, is_unitful
export KernelBackend, KABackend, kernel_backend, per_term_ad
export AbstractLinearOperator, SparseMatrixLinearOperator, MatrixFreeError,
    underlying_matrix, as_linear_operator, MatrixFreeLinearOperator
export FVMSolverConfig, FieldSolverConfig, default_solver_config,
    build_preconditioner, PETScLinearSolver
# Temporary over-exports for the still-flat remainder (Stage 4 curates)
export _dispatch_solve, _cpu_backend_only, _unsupported_backend,
    _solve_with_config, _resolve_solver, _try_krylov_solver,
    _extension_preconditioner, _solve_linear, autodiff_forward_step,
    compute_slope_ratio, limit_gradient

end # module Numerics
