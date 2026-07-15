# sciml_bridge.jl — Bridge between parabolic assembly and SciMLBase
#
# Provides helpers to convert assembled matrices (A, b) from the parabolic
# solver into SciMLBase problem types (ODEProblem, LinearProblem), enabling
# use of OrdinaryDiffEq.jl, LinearSolve.jl, and NonlinearSolve.jl.

"""
    parabolic_to_odefunction(A, M, b)

Convert assembled parabolic system matrices into a SciMLBase `ODEFunction`.

The parabolic assembly produces `M du/dt + A u = b`, which can be rewritten as
`M du/dt = b - A u`, or `du/dt = M \\ (b - A u)`.

# Arguments
- `A`: System (stiffness) matrix from `assemble_system`
- `M`: Mass matrix from `assemble_mass_matrix` (pass `I` if none)
- `b`: Source/RHS vector from `assemble_system`

# Returns
An `ODEFunction` suitable for `ODEProblem(f, u0, tspan)`.

# Example
```julia
A, b = assemble_system(Diffusion1D(1.0), mesh, bc_left, bc_right)
M = assemble_mass_matrix(mesh)
f = parabolic_to_odefunction(A, M, b)
prob = ODEProblem(f, u0, (0.0, 1.0))
sol = solve(prob, TRBDF2())  # or any OrdinaryDiffEq algorithm
```
"""
function parabolic_to_odefunction(A, M, b)
    # Precompute factorization of M for efficiency
    M_fact = LinearAlgebra.factorize(M)
    _tmp = similar(b)
    f = ODEFunction(
        function (du, u, p, t)
            # du = M^{-1} (b - A u)
            LinearAlgebra.mul!(_tmp, A, u)
            @. _tmp = b - _tmp
            LinearAlgebra.ldiv!(du, M_fact, _tmp)
            return nothing
        end;
        jac_prototype = -M_fact \ A,
    )
    return f
end

"""
    parabolic_to_odefunction(A, b)

Simplified version assuming mass matrix `M = I`.

# Example
```julia
A, b = assemble_system(Diffusion1D(1.0), mesh, bc_left, bc_right)
f = parabolic_to_odefunction(A, b)
prob = ODEProblem(f, u0, (0.0, 1.0))
sol = solve(prob, Tsit5())
```
"""
function parabolic_to_odefunction(A, b)
    _tmp = similar(b)
    f = ODEFunction(
        function (du, u, p, t)
            # du = b - A u
            LinearAlgebra.mul!(_tmp, A, u)
            @. du = b - _tmp
            return nothing
        end;
        jac_prototype = -A,
    )
    return f
end

"""
    parabolic_to_linearproblem(A, b)

Convert assembled steady-state system into a SciMLBase `LinearProblem`.

The assembled system `A u = b` maps directly to `LinearProblem(A, b)`.

# Example
```julia
A, b = assemble_system(Diffusion1D(1.0), mesh, bc_left, bc_right)
prob = parabolic_to_linearproblem(A, b)

# Using LinearSolve.jl:
using LinearSolve
sol = solve(prob)                    # direct solver
sol = solve(prob, KrylovJL_GMRES()) # true GMRES with Krylov
```
"""
function parabolic_to_linearproblem(A, b)
    return LinearProblem(A, b)
end

"""
    SciMLBase.ODEProblem(model, mesh::AbstractParabolicMesh, bcs...;
                         tspan, u0, source = nothing, transient = false)

Convenience constructor for the structured-parabolic stack: assembles the
`(A, b)` system from `(model, mesh, bcs...)`, takes the mass matrix from
`assemble_mass_matrix(mesh)`, wraps them with `parabolic_to_odefunction`, and
returns an `SciMLBase.ODEProblem` ready for `solve`.

Equivalent (long-form) to:
```julia
A, b = assemble_system(model, mesh, bcs...; source = source, transient = transient)
M = assemble_mass_matrix(mesh)
f = parabolic_to_odefunction(A, M, b)
prob = ODEProblem(f, u0, tspan)
```

Works for any `(model, mesh)` pair `assemble_system` already dispatches on
(1D/2D/3D structured parabolic, cylindrical, spherical). The variadic `bcs...`
matches the existing assembly call shape (1D: `bc_left, bc_right`; 2D/3D:
single tuple of face BCs).

# Example
```julia
mesh = generate_mesh_1d(50, 1.0)
prob = ODEProblem(Diffusion1D(2.0e-7), mesh,
                  ParabolicDirichlet(600.0), ParabolicNeumann(0.0);
                  tspan = (0.0, 10.0), u0 = fill(560.0, length(mesh.cells)))
sol = solve(prob, ImplicitEuler(); adaptive = false, dt = 0.01)
```
"""
function SciMLBase.ODEProblem(
        model::AbstractEquationModel, mesh::AbstractParabolicMesh, bcs...;
        tspan::Tuple,
        u0::AbstractVector,
        source = nothing,
        transient::Bool = false,
    )
    A, b = assemble_system(model, mesh, bcs...; source = source, transient = transient)
    M = assemble_mass_matrix(mesh)
    f = parabolic_to_odefunction(A, M, b)
    return ODEProblem(f, u0, tspan)
end
