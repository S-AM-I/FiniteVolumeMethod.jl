# muscl_problem.jl — FVMProblem convenience constructor for MUSCL fluxes.
# Lives outside the Numerics submodule because it constructs FVMProblem
# (flat, loads after problem.jl); the pure MUSCL reconstruction kernels
# live in Numerics.

@doc raw"""
    create_muscl_problem(mesh, BCs, ICs=InternalConditions();
                         diffusion_function, diffusion_parameters=nothing,
                         velocity_function=nothing, velocity_parameters=nothing,
                         source_function=(x,y,t,u,p)->zero(u), source_parameters=nothing,
                         initial_condition, initial_time=0.0, final_time,
                         limiter=VanLeerLimiter(), gradient_method=GreenGaussGradient(),
                         kwargs...)

Create an FVMProblem with MUSCL-based flux function.

This is a convenience constructor that sets up a standard FVMProblem
with a MUSCL flux function.

# Arguments
- `mesh::FVMGeometry`: The mesh geometry
- `BCs::BoundaryConditions`: Boundary conditions
- `ICs::InternalConditions`: Internal conditions (optional)

# Keyword Arguments
- `diffusion_function`: D(x, y, t, u, p) -> scalar
- `diffusion_parameters`: Parameters for diffusion function
- `velocity_function`: v(x, y, t, u, p) -> (vx, vy), or nothing for pure diffusion
- `velocity_parameters`: Parameters for velocity function
- `source_function`: S(x, y, t, u, p) -> scalar
- `source_parameters`: Parameters for source function
- `initial_condition`: Initial condition vector
- `initial_time=0.0`: Start time
- `final_time`: End time
- `limiter=VanLeerLimiter()`: Flux limiter to use
- `gradient_method=GreenGaussGradient()`: Gradient reconstruction method
- `kwargs...`: Additional arguments passed to FVMProblem

# Returns
An FVMProblem configured with MUSCL reconstruction.
"""
function create_muscl_problem(
        mesh::FVMGeometry,
        BCs::BoundaryConditions,
        ICs::InternalConditions = InternalConditions();
        diffusion_function,
        diffusion_parameters = nothing,
        velocity_function = nothing,
        velocity_parameters = nothing,
        source_function = (x, y, t, u, p) -> zero(typeof(x)),
        source_parameters = nothing,
        initial_condition,
        initial_time = 0.0,
        final_time,
        limiter::AbstractLimiter = VanLeerLimiter(),
        gradient_method::AbstractGradientMethod = GreenGaussGradient(),
        kwargs...
    )
    scheme = MUSCLScheme(limiter = limiter, gradient_method = gradient_method)

    # Wrap functions to include parameters
    D_wrapped = (x, y, t, u, p) -> diffusion_function(x, y, t, u, diffusion_parameters)
    v_wrapped = isnothing(velocity_function) ? nothing :
        (x, y, t, u, p) -> velocity_function(x, y, t, u, velocity_parameters)

    flux_fn = MUSCLFluxFunction(scheme; diffusion = D_wrapped, velocity = v_wrapped)

    # The flux function signature for FVMProblem is q(x, y, t, α, β, γ, p)
    flux = (x, y, t, α, β, γ, p) -> flux_fn(x, y, t, α, β, γ, p)

    return FVMProblem(
        mesh, BCs, ICs;
        flux_function = flux,
        flux_parameters = nothing,
        source_function = source_function,
        source_parameters = source_parameters,
        initial_condition = initial_condition,
        initial_time = initial_time,
        final_time = final_time,
        kwargs...
    )
end
