# radiation/p1.jl — P1 radiation equation assembly and solver
#
# Assembles: -div(Gamma * grad(G)) + a * G = 4 * a * sigma * T^4
# where Gamma = 1/(3a). Uses Phase 0 Laplacian + source terms.

"""
    assemble_p1!(
        eq, rad_model, T_field, mesh, bcs_G,
    )

Assemble the P1 radiation equation into `eq`.

The equation `-div(Gamma * grad(G)) + a * G = 4 * a * sigma * T^4` becomes:
- Laplacian with diffusivity `Gamma = 1/(3a)` -> contributes to A (positive diagonal)
- Absorption `a * V` -> added to diagonal
- Emission `4 * a * sigma * T^4 * V` -> added to RHS
"""
function assemble_p1!(
        eq::CollocatedEquation{T},
        rad_model::P1Model{T},
        T_field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_G::Dict{Symbol, <:AbstractBoundaryCondition},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    a = rad_model.a

    # Radiation diffusivity: Gamma = 1/(3a)
    gamma = one(T) / (T(3) * a)

    # Laplacian: -div(Gamma * grad(G)) assembled as positive-definite operator
    assemble_laplacian!(eq, gamma, mesh, bcs_G)

    # Absorption (implicit): a * V on diagonal
    for c in 1:nc
        eq.A[c, c] += a * mesh.cell_volumes[c]
    end

    # Emission (explicit RHS): 4 * a * sigma * T^4 * V
    sigma = T(STEFAN_BOLTZMANN)
    for c in 1:nc
        T_c = max(T_field.internal[c], zero(T))
        eq.b[c] += T(4) * a * sigma * T_c^4 * mesh.cell_volumes[c]
    end

    return nothing
end

"""
    solve_p1_radiation(
        rad_model, T_field, mesh, bcs_G; linear_solver = nothing,
    ) -> CollocatedScalarField{T}

Assemble and solve the P1 radiation equation, returning the G field.
"""
function solve_p1_radiation(
        rad_model::P1Model{T},
        T_field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_G::Dict{Symbol, <:AbstractBoundaryCondition};
        linear_solver = nothing,
        solver_config = nothing,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    eq = CollocatedEquation(mesh)
    assemble_p1!(eq, rad_model, T_field, mesh, bcs_G)

    lp = to_linear_problem(eq)
    sol = _dispatch_solve(lp, linear_solver, solver_config, :G)

    G = CollocatedScalarField(:G, mesh)
    for c in 1:nc
        G.internal[c] = max(sol.u[c], zero(T))
    end

    return G
end

"""
    compute_radiation_source(
        rad_model, G, T_field,
    ) -> Vector{T}

Compute the volumetric radiation source term for the energy equation:
`S_rad[c] = a * G[c] - 4 * a * sigma * T[c]^4`

Positive = net absorption (fluid heats up).
Negative = net emission (fluid cools).

To add to the energy equation (which is scaled by 1/(rho * Cp)):
`eq.b[c] += S_rad[c] * V_c / (rho * Cp)`
"""
function compute_radiation_source(
        rad_model::P1Model{T},
        G::CollocatedScalarField{T},
        T_field::CollocatedScalarField{T},
    ) where {T}
    nc = length(G.internal)
    a = rad_model.a
    sigma = T(STEFAN_BOLTZMANN)
    S_rad = Vector{T}(undef, nc)

    for c in 1:nc
        T_c = max(T_field.internal[c], zero(T))
        S_rad[c] = a * G.internal[c] - T(4) * a * sigma * T_c^4
    end

    return S_rad
end
