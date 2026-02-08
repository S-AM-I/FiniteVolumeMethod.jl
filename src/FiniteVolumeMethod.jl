module FiniteVolumeMethod

using ChunkSplitters: ChunkSplitters, chunks
using CommonSolve: CommonSolve, solve
using DelaunayTriangulation: DelaunayTriangulation, Triangulation,
    add_ghost_triangles!,
    convert_boundary_points_to_indices,
    delete_ghost_triangles!, each_solid_triangle,
    each_solid_vertex, get_adjacent, get_area,
    get_boundary_edge_map, get_boundary_nodes,
    get_ghost_vertex_map, get_neighbours, get_point,
    getxy, lock_convex_hull!, num_boundary_edges,
    num_solid_triangles, refine!, statistics,
    triangle_vertices, triangulate,
    triangulate_rectangle, unlock_convex_hull!
using LinearAlgebra: LinearAlgebra, norm
using PreallocationTools: PreallocationTools, DiffCache, get_tmp
using SciMLBase: SciMLBase, CallbackSet, DiscreteCallback, LinearProblem,
    MatrixOperator, ODEFunction, ODEProblem, SteadyStateProblem
using SparseArrays: SparseArrays, sparse
using StaticArrays: StaticArrays, SVector
using Base.Threads

include("geometry.jl")
include("conditions.jl")
include("problem.jl")
include("equations/boundary_edge_contributions.jl")
include("equations/control_volumes.jl")
include("equations/dirichlet.jl")
include("equations/individual_flux_contributions.jl")
include("equations/main_equations.jl")
include("equations/shape_functions.jl")
include("equations/source_contributions.jl")
include("equations/triangle_contributions.jl")
include("solve.jl")
include("utils.jl")

# Schemes for higher-order methods
include("schemes/limiters.jl")
include("schemes/gradients.jl")
include("schemes/muscl.jl")

# Advanced boundary conditions
include("conditions/nonlinear.jl")
include("conditions/periodic.jl")
include("conditions/coupled.jl")

include("specific_problems/abstract_templates.jl")
include("specific_problems/advection_diffusion_equation.jl")
include("specific_problems/anisotropic_diffusion.jl")

# Physics models
include("physics/turbulence/k_epsilon.jl")

# Hyperbolic solver framework (cell-centered FVM)
include("mesh/abstract_mesh.jl")
include("mesh/structured_mesh.jl")
include("eos/eos_interface.jl")
include("eos/ideal_gas.jl")
include("hyperbolic/conservation_laws.jl")
include("hyperbolic/euler.jl")
include("hyperbolic/riemann_solvers.jl")
include("hyperbolic/reconstruction.jl")
include("hyperbolic/boundary_conditions_hyp.jl")
include("hyperbolic/hllc_solver.jl")
include("hyperbolic/mhd.jl")
include("hyperbolic/hlld_solver.jl")
include("hyperbolic/hyperbolic_problem.jl")
include("hyperbolic/hyperbolic_solve.jl")
include("hyperbolic/hyperbolic_problem_2d.jl")
include("hyperbolic/boundary_conditions_2d.jl")
include("hyperbolic/hyperbolic_solve_2d.jl")

# Constrained transport for MHD
include("constrained_transport/ct_data.jl")
include("constrained_transport/emf.jl")
include("constrained_transport/ct_update.jl")
include("constrained_transport/divb.jl")

# 2D MHD solver with constrained transport
include("hyperbolic/mhd_solve_2d.jl")

export FVMGeometry,
    FVMProblem,
    FVMSystem,
    SteadyFVMProblem,
    BoundaryConditions,
    InternalConditions,
    Conditions,
    Neumann,
    Dudt,
    Dirichlet,
    Constrained,
    Robin,
    solve,
    compute_flux,
    pl_interpolate,
    # Flux limiters
    AbstractLimiter,
    MinmodLimiter,
    SuperbeeLimiter,
    VanLeerLimiter,
    VenkatakrishnanLimiter,
    BarthJespersenLimiter,
    KorenLimiter,
    OspreLimiter,
    minmod,
    superbee,
    van_leer,
    venkatakrishnan,
    barth_jespersen,
    koren,
    ospre,
    apply_limiter,
    select_limiter,
    # Gradient reconstruction
    AbstractGradientMethod,
    GreenGaussGradient,
    LeastSquaresGradient,
    reconstruct_gradient,
    reconstruct_gradient_at_edge,
    reconstruct_gradient_at_point,
    reconstruct_all_gradients,
    # MUSCL scheme
    MUSCLScheme,
    muscl_reconstruct_face_value,
    muscl_reconstruct_edge_values,
    muscl_advective_flux,
    muscl_diffusive_flux,
    MUSCLFluxFunction,
    create_muscl_problem,
    # Advection-diffusion
    AdvectionDiffusionEquation,
    # Nonlinear BCs
    NonlinearDirichlet,
    NonlinearNeumann,
    NonlinearRobin,
    linearize_bc,
    compute_boundary_gradient,
    evaluate_nonlinear_bc,
    # Periodic BCs
    PeriodicBC,
    PeriodicNodeMapping,
    PeriodicConditions,
    compute_periodic_mapping,
    apply_periodic_constraints!,
    has_periodic_conditions,
    # Coupled multi-field BCs
    CoupledBC,
    CoupledDirichlet,
    CoupledNeumann,
    CoupledRobin,
    CoupledBoundaryConditions,
    evaluate_coupled_bc,
    add_coupled_bc!,
    get_coupled_bc,
    has_coupled_bc,
    get_target_field,
    # Anisotropic diffusion
    AnisotropicDiffusionEquation,
    make_rotation_tensor,
    make_spatially_varying_tensor,
    # Turbulence models
    StandardKEpsilon,
    KappaOmegaSST,
    compute_turbulent_viscosity,
    compute_strain_rate_magnitude,
    compute_production,
    compute_friction_velocity,
    k_wall_value,
    epsilon_wall_value,
    TurbulentWallBC,
    # Mesh abstractions
    AbstractMesh,
    StructuredMesh1D,
    StructuredMesh2D,
    ncells,
    nfaces,
    cell_center,
    cell_volume,
    face_area,
    face_owner,
    face_neighbor,
    ndims_mesh,
    # Equations of state
    AbstractEOS,
    IdealGasEOS,
    pressure,
    sound_speed,
    internal_energy,
    total_energy,
    # Conservation laws
    AbstractConservationLaw,
    EulerEquations,
    IdealMHDEquations,
    fast_magnetosonic_speed,
    slow_magnetosonic_speed,
    nvariables,
    physical_flux,
    max_wave_speed,
    wave_speeds,
    conserved_to_primitive,
    primitive_to_conserved,
    # Riemann solvers
    AbstractRiemannSolver,
    LaxFriedrichsSolver,
    HLLSolver,
    HLLCSolver,
    HLLDSolver,
    solve_riemann,
    # Reconstruction
    CellCenteredMUSCL,
    NoReconstruction,
    reconstruct_interface,
    # Hyperbolic boundary conditions
    AbstractHyperbolicBC,
    TransmissiveBC,
    ReflectiveBC,
    InflowBC,
    PeriodicHyperbolicBC,
    DirichletHyperbolicBC,
    # Hyperbolic problem and solver
    HyperbolicProblem,
    HyperbolicProblem2D,
    solve_hyperbolic,
    compute_dt,
    compute_dt_2d,
    hyperbolic_rhs!,
    hyperbolic_rhs_2d!,
    to_primitive,
    # 2D mesh helpers
    cell_ij,
    cell_idx,
    # Constrained transport
    CTData2D,
    initialize_ct!,
    initialize_ct_from_potential!,
    face_to_cell_B!,
    copy_ct,
    copyto_ct!,
    compute_emf_2d!,
    ct_update!,
    compute_divB,
    max_divB,
    l2_divB

using PrecompileTools: PrecompileTools, @compile_workload, @setup_workload
@setup_workload begin
    @compile_workload begin
        # Compile a non-steady problem
        n = 5
        α = π / 4
        x₁ = [0.0, 1.0]
        y₁ = [0.0, 0.0]
        r₂ = fill(1, n)
        θ₂ = LinRange(0, α, n)
        x₂ = @. r₂ * cos(θ₂)
        y₂ = @. r₂ * sin(θ₂)
        x₃ = [cos(α), 0.0]
        y₃ = [sin(α), 0.0]
        x = [x₁, x₂, x₃]
        y = [y₁, y₂, y₃]
        boundary_nodes, points = convert_boundary_points_to_indices(x, y)
        tri = triangulate(points; boundary_nodes)
        A = get_area(tri)
        refine!(tri)
        mesh = FVMGeometry(tri)
        lower_bc = arc_bc = upper_bc = (x, y, t, u, p) -> zero(u)
        types = (Neumann, Dirichlet, Neumann)
        BCs = BoundaryConditions(mesh, (lower_bc, arc_bc, upper_bc), types)
        f = (x, y) -> 1 - sqrt(x^2 + y^2)
        D = (x, y, t, u, p) -> one(u)
        initial_condition = [
            f(x, y)
                for (x, y) in
                DelaunayTriangulation.DelaunayTriangulation.each_point(tri)
        ]
        final_time = 0.1
        prob = FVMProblem(mesh, BCs; diffusion_function = D, initial_condition, final_time)
        ode_prob = ODEProblem(prob)
        steady_prob = SteadyFVMProblem(prob)
        nl_prob = SteadyStateProblem(steady_prob)

        # Compile a system
        tri = triangulate_rectangle(0, 100, 0, 100, 5, 5, single_boundary = true)
        mesh = FVMGeometry(tri)
        bc_u = (x, y, t, (u, v), p) -> zero(u)
        bc_v = (x, y, t, (u, v), p) -> zero(v)
        BCs_u = BoundaryConditions(mesh, bc_u, Neumann)
        BCs_v = BoundaryConditions(mesh, bc_v, Neumann)
        q_u = (x, y, t, (αu, αv), (βu, βv), (γu, γv), p) -> begin
            u = αu * x + βu * y + γu
            ∇u = (αu, βu)
            ∇v = (αv, βv)
            χu = p.c * u / (1 + u^2)
            _q = χu .* ∇v .- ∇u
            return _q
        end
        q_v = (x, y, t, (αu, αv), (βu, βv), (γu, γv), p) -> begin
            ∇v = (αv, βv)
            _q = -p.D .* ∇v
            return _q
        end
        S_u = (x, y, t, (u, v), p) -> begin
            return u * (1 - u)
        end
        S_v = (x, y, t, (u, v), p) -> begin
            return u - p.a * v
        end
        q_u_parameters = (c = 4.0,)
        q_v_parameters = (D = 1.0,)
        S_v_parameters = (a = 0.1,)
        u_initial_condition = 0.01rand(DelaunayTriangulation.num_solid_vertices(tri))
        v_initial_condition = zeros(DelaunayTriangulation.num_solid_vertices(tri))
        final_time = 1000.0
        u_prob = FVMProblem(
            mesh, BCs_u;
            flux_function = q_u, flux_parameters = q_u_parameters,
            source_function = S_u,
            initial_condition = u_initial_condition, final_time = final_time
        )
        v_prob = FVMProblem(
            mesh, BCs_v;
            flux_function = q_v, flux_parameters = q_v_parameters,
            source_function = S_v, source_parameters = S_v_parameters,
            initial_condition = v_initial_condition, final_time = final_time
        )
        prob = FVMSystem(u_prob, v_prob)
        ode_prob = ODEProblem(prob)
        steady_prob = SteadyFVMProblem(prob)
        nl_prob = SteadyStateProblem(steady_prob)
    end
end
end
