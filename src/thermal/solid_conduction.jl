# thermal/solid_conduction.jl — Solid conduction equation
#
# Solves: rho*Cp * dT/dt = div(k * grad(T)) + Q_gen
# For steady state: div(k * grad(T)) = -Q_gen

"""
    assemble_solid_conduction!(
        eq::CollocatedEquation{T},
        solid::SolidThermalProperties{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
        T_old::Union{Nothing, Vector{T}} = nothing,
    )

Assemble the solid conduction equation into `eq`.

For steady state, assembles `div(k * grad(T)) + Q_gen = 0`.
For transient, adds `rho*Cp * (T - T_old) / dt`.

# Arguments
- `eq` — equation (modified in-place)
- `solid` — solid thermal properties
- `mesh` — solid mesh
- `bcs_T` — temperature boundary conditions for the solid
- `dt` — time step (nothing for steady)
- `T_old` — temperature at previous time step (required if transient)
"""
function assemble_solid_conduction!(
        eq::CollocatedEquation{T},
        solid::SolidThermalProperties{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
        T_old::Union{Nothing, Vector{T}} = nothing,
    ) where {Dim, T}
    # Diffusion: div(k * grad(T))
    assemble_laplacian!(eq, solid.k, mesh, bcs_T)

    # Volumetric heat generation source
    nc = length(mesh.cell_volumes)
    if solid.Q_gen != zero(T)
        for c in 1:nc
            eq.b[c] += solid.Q_gen * mesh.cell_volumes[c]
        end
    end

    # Temporal term (transient only)
    if dt !== nothing && T_old !== nothing
        rho_Cp = solid.rho * solid.Cp
        assemble_ddt_euler!(eq, rho_Cp, T_old, mesh, dt)
    end

    return nothing
end

"""
    solve_solid_conduction(
        mesh::UnstructuredFVMMesh{Dim, T},
        solid::SolidThermalProperties{T},
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
        dt = nothing,
        T_old = nothing,
        linear_solver = nothing,
    ) -> CollocatedScalarField{T}

Solve the solid conduction equation and return the temperature field.

For steady state (`dt = nothing`), performs a single linear solve.
For transient, requires `T_old` (previous temperature values).
"""
function solve_solid_conduction(
        mesh::UnstructuredFVMMesh{Dim, T},
        solid::SolidThermalProperties{T},
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
        T_old::Union{Nothing, Vector{T}} = nothing,
        linear_solver = nothing,
    ) where {Dim, T}
    eq = CollocatedEquation(mesh)
    assemble_solid_conduction!(eq, solid, mesh, bcs_T; dt = dt, T_old = T_old)

    lp = to_linear_problem(eq)
    sol = _solve_linear(lp, linear_solver)

    nc = length(mesh.cell_volumes)
    T_field = CollocatedScalarField(:T_solid, mesh)
    for c in 1:nc
        T_field.internal[c] = sol.u[c]
    end

    return T_field
end
