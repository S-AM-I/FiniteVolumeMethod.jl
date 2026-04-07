# combustion/species_transport.jl — Species transport equation assembly and solve
#
# Assembles and solves the species mass fraction transport equation:
#   ∂Y_i/∂t + div(phi · Y_i) = div(D_eff_i · grad(Y_i)) + ω_i / ρ
# where D_eff_i = D_i + ν_t / Sc_t is the effective mass diffusivity.

"""
    assemble_species!(
        eq, Y_i, phi, D_eff, mesh, bcs_Yi; dt = nothing,
    )

Assemble the species transport equation for mass fraction `Y_i`.

Adds convection, diffusion, and (optionally) temporal terms to `eq`.
The reaction source `ω_i / ρ` must be added to `eq.b` separately
after calling this function.

# Arguments
- `eq::CollocatedEquation{T}` — equation (modified in-place)
- `Y_i::CollocatedScalarField{T}` — current mass fraction field
- `phi::FaceFluxField{T}` — face volumetric flux
- `D_eff::Union{T, Vector{T}}` — effective mass diffusivity (scalar or per-cell)
- `mesh::UnstructuredFVMMesh` — mesh
- `bcs_Yi::Dict{Symbol, <:AbstractBoundaryCondition}` — BCs for this species
- `dt` — time step (`nothing` for steady state)
"""
function assemble_species!(
        eq::CollocatedEquation{T},
        Y_i::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        D_eff::Union{T, Vector{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_Yi::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
    ) where {Dim, T}
    # Convection: div(phi · Y_i)
    assemble_convection!(eq, phi, mesh, bcs_Yi)

    # Diffusion: div(D_eff · grad(Y_i))
    assemble_laplacian!(eq, D_eff, mesh, bcs_Yi)

    # Temporal term (if transient)
    if dt !== nothing
        assemble_ddt_euler!(eq, one(T), Y_i.internal, mesh, dt)
    end

    return nothing
end

"""
    solve_species!(
        species_state, phi, combustion_props, reaction_rates,
        nu_t, density, mesh, bcs_species; dt, linear_solver,
    )

Solve the species transport equations for all species.

For each species `i`:
1. Compute effective diffusivity `D_eff_i = D_i + ν_t / Sc_t`
2. Assemble convection + diffusion + temporal into a `CollocatedEquation`
3. Add reaction source `ω_i / ρ × V_c` to the RHS
4. Solve the linear system
5. Clip `Y_i` to `[0, 1]`

# Arguments
- `species_state::SpeciesState{NS, T}` — species fields (modified in-place)
- `phi::FaceFluxField{T}` — face volumetric flux
- `combustion_props::CombustionProperties{NS, T}` — thermochemical properties
- `reaction_rates::NTuple{NS, Vector{T}}` — per-species per-cell reaction rates [kg/(m³·s)]
- `nu_t::Union{Nothing, Vector{T}}` — turbulent viscosity (or `nothing`)
- `density::T` — fluid density
- `mesh::UnstructuredFVMMesh` — mesh
- `bcs_species::Dict{Symbol, Dict{Symbol, <:AbstractBoundaryCondition}}` — BCs keyed by species name
- `dt` — time step (`nothing` for steady state)
- `linear_solver` — linear solver algorithm (or `nothing` for default)
"""
function solve_species!(
        species_state::SpeciesState{NS, T},
        phi::FaceFluxField{T},
        combustion_props::CombustionProperties{NS, T},
        reaction_rates::NTuple{NS, Vector{T}},
        nu_t::Union{Nothing, Vector{T}},
        density::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_species::Dict{Symbol, <:Any};
        dt::Union{Nothing, T} = nothing,
        linear_solver = nothing,
        solver_config = nothing,
    ) where {Dim, T, NS}
    nc = length(mesh.cell_volumes)

    for i in 1:NS
        name_i = combustion_props.species_names[i]
        Y_i = species_state.Y[i]

        # Effective mass diffusivity: D_i + nu_t / Sc_t
        D_lam = combustion_props.diffusivities[i]
        D_eff = if nu_t === nothing
            D_lam
        else
            D_vec = Vector{T}(undef, nc)
            Sc_t_val = combustion_props.Sc_t
            for c in 1:nc
                D_vec[c] = D_lam + nu_t[c] / Sc_t_val
            end
            D_vec
        end

        # Get BCs for this species
        bcs_Yi = bcs_species[name_i]

        # Assemble equation
        eq = CollocatedEquation(mesh)
        assemble_species!(eq, Y_i, phi, D_eff, mesh, bcs_Yi; dt = dt)

        # Add reaction source: ω_i / ρ * V_c
        omega_i = reaction_rates[i]
        for c in 1:nc
            eq.b[c] += omega_i[c] / density * mesh.cell_volumes[c]
        end

        # Solve
        sol = _dispatch_solve(to_linear_problem(eq), linear_solver, solver_config, name_i)

        # Update field with clipping to [0, 1]
        for c in 1:nc
            species_state.Y[i].internal[c] = clamp(sol.u[c], zero(T), one(T))
        end
    end

    return nothing
end
