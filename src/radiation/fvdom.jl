# radiation/fvdom.jl — Finite Volume Discrete Ordinates Method (fvDOM)
#
# Solves the radiative transfer equation per discrete ordinate direction:
#   div(I_i * s_i) + a * I_i = a * sigma * T^4 / pi
# where I_i is the radiative intensity in direction s_i, a is the
# absorption coefficient.  The incident radiation G is reconstructed
# as the weighted sum of all directional intensities.

"""
    FvDOMModel{Dim, T} <: AbstractRadiationModel

Finite Volume Discrete Ordinates Method for radiation.

Solves one transport equation per solid-angle direction, then
reconstructs the incident radiation field as
`G = sum_i w_i * I_i`.

# Fields
- `a::T` --- absorption coefficient [1/m]
- `directions::Vector{SVector{Dim, T}}` --- discrete ordinate directions
- `weights::Vector{T}` --- quadrature weights
"""
struct FvDOMModel{Dim, T} <: AbstractRadiationModel
    a::T
    directions::Vector{SVector{Dim, T}}
    weights::Vector{T}
end

"""
    FvDOMModel(; a = 0.1, Dim = 2)

Construct an fvDOM radiation model with S2 level-symmetric quadrature.

For 2D, S2 produces 4 directions at +/-45 degrees with equal weights pi/2.
For 3D, S2 produces 8 directions (octant corners of the unit sphere) with
equal weights pi/2.
"""
function FvDOMModel(; a::Real = 0.1, Dim::Int = 2)
    T = typeof(Float64(a))
    directions, weights = _s2_quadrature(Val(Dim), T)
    return FvDOMModel{Dim, T}(T(a), directions, weights)
end

"""Generate S2 quadrature directions and weights for 2D (4 directions)."""
function _s2_quadrature(::Val{2}, ::Type{T}) where {T}
    inv_sqrt2 = one(T) / sqrt(T(2))
    dirs = SVector{2, T}[
        SVector{2, T}(inv_sqrt2, inv_sqrt2),
        SVector{2, T}(-inv_sqrt2, inv_sqrt2),
        SVector{2, T}(-inv_sqrt2, -inv_sqrt2),
        SVector{2, T}(inv_sqrt2, -inv_sqrt2),
    ]
    w = fill(T(pi) / 2, 4)
    return dirs, w
end

"""Generate S2 quadrature directions and weights for 3D (8 directions)."""
function _s2_quadrature(::Val{3}, ::Type{T}) where {T}
    inv_sqrt3 = one(T) / sqrt(T(3))
    dirs = SVector{3, T}[]
    for sx in (one(T), -one(T))
        for sy in (one(T), -one(T))
            for sz in (one(T), -one(T))
                push!(dirs, SVector{3, T}(sx * inv_sqrt3, sy * inv_sqrt3, sz * inv_sqrt3))
            end
        end
    end
    w = fill(T(pi) / 2, 8)
    return dirs, w
end

"""
    solve_fvdom_radiation(
        model, T_field, mesh, bcs_G;
        linear_solver = nothing, solver_config = nothing,
    ) -> RadiationState{T}

Solve the fvDOM radiation equations for all ordinate directions and
return a [`RadiationState`](@ref) with the incident radiation field G.

For each direction `s_i` with weight `w_i`, solves:
    div(I_i * s_i) + a * I_i = a * sigma * T^4 / pi

The incident radiation is `G = sum_i w_i * I_i`.

# Arguments
- `model::FvDOMModel{Dim, T}` --- fvDOM model with directions and weights
- `T_field::CollocatedScalarField{T}` --- temperature field [K]
- `mesh::UnstructuredFVMMesh{Dim, T}` --- mesh
- `bcs_G::Dict{Symbol, <:AbstractBoundaryCondition}` --- BCs applied to each
  intensity equation (typically `ParabolicDirichlet(sigma*T_wall^4/pi)`)
- `linear_solver` --- optional linear solver algorithm
- `solver_config` --- optional solver configuration
"""
function solve_fvdom_radiation(
        model::FvDOMModel{Dim, T},
        T_field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_G::Dict{Symbol, <:AbstractBoundaryCondition};
        linear_solver = nothing,
        solver_config = nothing,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    a = model.a
    sigma = T(STEFAN_BOLTZMANN)
    n_dirs = length(model.directions)

    # Accumulator for G = sum_i w_i * I_i
    G_values = zeros(T, nc)

    for idx in 1:n_dirs
        s_i = model.directions[idx]
        w_i = model.weights[idx]

        eq = CollocatedEquation(mesh)

        # Build a face flux field for the convection in direction s_i:
        #   flux_f = dot(s_i, S_f)  where S_f is the face area vector
        dir_flux = FaceFluxField(:fvdom_flux, mesh)
        for f in 1:nf
            S_f = face_normal_area(mesh, f)
            dir_flux.values[f] = dot(s_i, S_f)
        end

        # Convection: div(I_i * s_i) assembled implicitly
        assemble_convection!(eq, dir_flux, mesh, bcs_G)

        # Absorption: a * V_c on diagonal (implicit)
        for c in 1:nc
            eq.A[c, c] += a * mesh.cell_volumes[c]
        end

        # Emission source: a * sigma * T^4 / pi * V_c (explicit RHS)
        for c in 1:nc
            T_c = max(T_field.internal[c], zero(T))
            eq.b[c] += a * sigma * T_c^4 / T(pi) * mesh.cell_volumes[c]
        end

        # Solve for I_i
        lp = to_linear_problem(eq)
        sol = _dispatch_solve(lp, linear_solver, solver_config, :fvdom)

        # Accumulate G
        for c in 1:nc
            I_c = max(sol.u[c], zero(T))
            G_values[c] += w_i * I_c
        end
    end

    # Build RadiationState with computed G
    G = CollocatedScalarField(:G, mesh)
    for c in 1:nc
        G.internal[c] = G_values[c]
    end

    return RadiationState{T}(G)
end

"""
    compute_radiation_source(
        model::FvDOMModel{Dim, T}, G, T_field,
    ) -> Vector{T}

Compute the volumetric radiation source term for the energy equation
using fvDOM results.  Same formula as the P1 model:

    S_rad[c] = a * G[c] - 4 * a * sigma * T[c]^4
"""
function compute_radiation_source(
        model::FvDOMModel{Dim, T},
        G::CollocatedScalarField{T},
        T_field::CollocatedScalarField{T},
    ) where {Dim, T}
    nc = length(G.internal)
    a = model.a
    sigma = T(STEFAN_BOLTZMANN)
    S_rad = Vector{T}(undef, nc)

    for c in 1:nc
        T_c = max(T_field.internal[c], zero(T))
        S_rad[c] = a * G.internal[c] - T(4) * a * sigma * T_c^4
    end

    return S_rad
end
