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
struct FvDOMModel{Dim, T, A <: Union{T, Vector{T}}} <: AbstractRadiationModel
    a::A
    directions::Vector{SVector{Dim, T}}
    weights::Vector{T}
end

"""
    FvDOMModel(; a = 0.1, Dim = 2, order = :S2)

Construct an fvDOM radiation model.

`a` may be scalar (uniform) or `Vector` (per-cell).
`order` selects the quadrature: `:S2` (4/8 dirs) or `:S4` (12/24 dirs).
"""
function FvDOMModel(; a = 0.1, Dim::Int = 2, order::Symbol = :S2)
    if a isa AbstractVector
        T = eltype(a)
        directions, weights = _fvdom_quadrature(Val(Dim), T, order)
        return FvDOMModel{Dim, T, Vector{T}}(Vector{T}(a), directions, weights)
    else
        T = typeof(Float64(a))
        directions, weights = _fvdom_quadrature(Val(Dim), T, order)
        return FvDOMModel{Dim, T, T}(T(a), directions, weights)
    end
end

"""Dispatch to the appropriate quadrature set."""
function _fvdom_quadrature(dim_val, ::Type{T}, order::Symbol) where {T}
    if order === :S2
        return _s2_quadrature(dim_val, T)
    elseif order === :S4
        return _s4_quadrature(dim_val, T)
    else
        error("Unknown fvDOM quadrature order :$order. Supported: :S2, :S4")
    end
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

# ── S4 quadrature ─────────────────────────────────────────────────

"""Generate S4 level-symmetric quadrature for 2D (12 directions).

S4 in 2D uses 3 directions per quadrant at polar angles corresponding
to the S4 level-symmetric ordinates (Carlson & Lathrop, 1968).
"""
function _s4_quadrature(::Val{2}, ::Type{T}) where {T}
    # S4 2D ordinates: 3 per quadrant = 12 total
    # Level-symmetric S4: mu values are roots of P_4
    mu1 = T(0.2958759)
    mu2 = T(0.9082483)
    # Remaining direction cosine from normalization: eta = sqrt(1 - mu^2)
    dirs = SVector{2, T}[]
    weights = T[]
    # Weights: w1 for edge ordinates, w2 for mid ordinates
    w1 = T(pi) / 6  # weight for each direction in 2D S4
    w2 = T(pi) / 3
    for (sx, sy) in ((1, 1), (-1, 1), (-1, -1), (1, -1))
        s = T(sx); t = T(sy)
        push!(dirs, SVector{2, T}(s * mu1, t * sqrt(one(T) - mu1^2)))
        push!(weights, w1)
        push!(dirs, SVector{2, T}(s * mu2, t * sqrt(one(T) - mu2^2)))
        push!(weights, w1)
        push!(dirs, SVector{2, T}(s * sqrt(T(0.5)), t * sqrt(T(0.5))))
        push!(weights, w2)
    end
    return dirs, weights
end

"""Generate S4 level-symmetric quadrature for 3D (24 directions).

S4 in 3D uses 3 directions per octant = 24 total.
"""
function _s4_quadrature(::Val{3}, ::Type{T}) where {T}
    # S4 3D level-symmetric ordinates (Carlson & Lathrop)
    mu1 = T(0.2958759)
    mu2 = T(0.9082483)
    # Weight per ordinate (total solid angle = 4pi, 24 directions)
    w = T(4) * T(pi) / T(24)

    dirs = SVector{3, T}[]
    weights = T[]
    # Generate 3 permutations per octant × 8 octants = 24
    for sx in (one(T), -one(T))
        for sy in (one(T), -one(T))
            for sz in (one(T), -one(T))
                push!(dirs, SVector{3, T}(sx * mu1, sy * mu1, sz * mu2))
                push!(weights, w)
                push!(dirs, SVector{3, T}(sx * mu1, sy * mu2, sz * mu1))
                push!(weights, w)
                push!(dirs, SVector{3, T}(sx * mu2, sy * mu1, sz * mu1))
                push!(weights, w)
            end
        end
    end
    return dirs, weights
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

        # Absorption: a_c * V_c on diagonal (implicit)
        for c in 1:nc
            a_c = _cell_absorption(a, c)
            eq.A[c, c] += a_c * mesh.cell_volumes[c]
        end

        # Emission source: a_c * sigma * T^4 / pi * V_c (explicit RHS)
        for c in 1:nc
            a_c = _cell_absorption(a, c)
            T_c = max(T_field.internal[c], zero(T))
            eq.b[c] += a_c * sigma * T_c^4 / T(pi) * mesh.cell_volumes[c]
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
        a_c = _cell_absorption(a, c)
        T_c = max(T_field.internal[c], zero(T))
        S_rad[c] = a_c * G.internal[c] - T(4) * a_c * sigma * T_c^4
    end

    return S_rad
end
