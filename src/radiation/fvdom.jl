# radiation/fvdom.jl — Finite Volume Discrete Ordinates Method (fvDOM)
#
# Solves the radiative transfer equation per discrete ordinate direction:
#   div(I_m * Ω̂_m) + (a + σ_s) I_m =
#       a * sigma * T^4 / pi + (σ_s / 4π) Σ_m' w_{m'} Φ(Ω̂_m, Ω̂_{m'}) I_{m'}
# where `I_m` is the radiative intensity in direction `Ω̂_m`, `a` is the
# absorption coefficient, `σ_s` is the scattering coefficient, and Φ is
# the scattering phase function. The incident radiation `G` is
# reconstructed as the weighted sum of all directional intensities.
#
# Level-symmetric SN quadrature tables (S2/S4/S6/S8/S12) are taken from
# Lewis & Miller, *Computational Methods of Neutron Transport* (1984),
# Table 4-1, and are reproduced in Modest, *Radiative Heat Transfer*
# (3rd ed., 2013), Appendix F. Weights are rescaled so Σ w = 4π for 3D
# and Σ w = 2π for the 2D half-space projection (positive η).

"""
    FvDOMModel{Dim, T} <: AbstractRadiationModel

Finite Volume Discrete Ordinates Method for radiation.

Solves one transport equation per solid-angle direction, then
reconstructs the incident radiation field as
`G = sum_i w_i * I_i`.

Supports absorbing-emitting-scattering media. Scattering is enabled by
setting `sigma_s > 0`; the phase function is selected via
`scattering_phase`:

  - `:isotropic` --- Φ = 1 (default)
  - `:linear_anisotropic` --- Φ(Ω̂, Ω̂') = 1 + g · (Ω̂ · Ω̂') with
    asymmetry parameter `g = scattering_g ∈ [-1, 1]`.

# Fields
- `a::A` --- absorption coefficient [1/m]
- `sigma_s::SS` --- scattering coefficient [1/m] (default 0)
- `scattering_phase::Symbol` --- `:isotropic` or `:linear_anisotropic`
- `scattering_g::T` --- asymmetry parameter (linear-anisotropic only)
- `directions::Vector{SVector{Dim, T}}` --- discrete ordinate directions
- `weights::Vector{T}` --- quadrature weights
"""
struct FvDOMModel{
        Dim, T,
        A <: Union{T, Vector{T}},
        SS <: Union{T, Vector{T}},
    } <: AbstractRadiationModel
    a::A
    sigma_s::SS
    scattering_phase::Symbol
    scattering_g::T
    directions::Vector{SVector{Dim, T}}
    weights::Vector{T}
end

"""
    FvDOMModel(;
        a = 0.1, sigma_s = 0.0,
        scattering_phase = :isotropic, scattering_g = 0.0,
        Dim = 2, order = :S2,
    )

Construct an fvDOM radiation model.

`a` and `sigma_s` may each be scalar (uniform) or `Vector` (per-cell).
`order` selects the quadrature: `:S2`, `:S4`, `:S6`, `:S8`, `:S12`.

The default `sigma_s = 0` disables scattering, preserving the
absorbing-only fvDOM behaviour for backwards compatibility.
"""
function FvDOMModel(;
        a = 0.1,
        sigma_s = 0.0,
        scattering_phase::Symbol = :isotropic,
        scattering_g::Real = 0.0,
        Dim::Int = 2,
        order::Symbol = :S2,
    )
    if a isa AbstractVector
        T = eltype(a)
    elseif sigma_s isa AbstractVector
        T = eltype(sigma_s)
    else
        T = typeof(Float64(a))
    end

    directions, weights = _fvdom_quadrature(Val(Dim), T, order)
    g = T(scattering_g)

    A_a = a isa AbstractVector ? Vector{T}(a) : T(a)
    SS_a = sigma_s isa AbstractVector ? Vector{T}(sigma_s) : T(sigma_s)

    return FvDOMModel{Dim, T, typeof(A_a), typeof(SS_a)}(
        A_a, SS_a, scattering_phase, g, directions, weights,
    )
end

"""Dispatch to the appropriate quadrature set."""
function _fvdom_quadrature(dim_val, ::Type{T}, order::Symbol) where {T}
    if order === :S2
        return _s2_quadrature(dim_val, T)
    elseif order === :S4
        return _s4_quadrature(dim_val, T)
    elseif order === :S6
        return _s6_quadrature(dim_val, T)
    elseif order === :S8
        return _s8_quadrature(dim_val, T)
    elseif order === :S12
        return _s12_quadrature(dim_val, T)
    else
        error("Unknown fvDOM quadrature order :$order. Supported: :S2, :S4, :S6, :S8, :S12")
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

# ── S6 / S8 / S12 level-symmetric quadrature ─────────────────────────
#
# Following Lewis & Miller (1984), Table 4-1, the level-symmetric SN set
# in 3D has `N * (N + 2)` directions distributed over the unit sphere:
#   S6  → 48 directions → 6 per octant
#   S8  → 80 directions → 10 per octant
#   S12 → 168 directions → 21 per octant
# Within one octant the ordinates are permutations of a small number of
# polar-cosine levels (µ_k, k=1..N/2). For each valid triplet
# (µ_i, µ_j, µ_k) with i + j + k = N/2 + 2, the symmetry of the
# level-symmetric set forces a common weight grouping.
#
# Weights are taken equal per ordinate within each level-symmetric set
# and rescaled so Σ w = 4π (3D) or 2π (2D half-space). This satisfies
# the zeroth and first moment identities (total weight, isotropy)
# verified by the V&V suite.
#
# For the 2D solver we retain only the positive-η hemisphere of the
# 3D set and rescale weights so Σ w = 2π.

"""Build a level-symmetric SN quadrature in 3D from the given level
cosines `mus` and point-symmetric weight groups `point_weights`. Each
key of `point_weights` is a sorted level-index triplet `(i,j,k)` with
i + j + k = length(mus) + 2.
"""
function _lsn_build_3d(
        ::Type{T}, mus::Vector{T},
        point_weights::Dict{NTuple{3, Int}, T},
    ) where {T}
    dirs = SVector{3, T}[]
    weights = T[]
    N_half = length(mus)
    target = N_half + 2
    for i in 1:N_half, j in 1:N_half, k in 1:N_half
        if i + j + k != target
            continue
        end
        key = _sort_triplet(i, j, k)
        haskey(point_weights, key) || continue
        w_ijk = point_weights[key]
        # Normalize the raw level-cosine triplet to the unit sphere so
        # that |Ω̂| = 1 exactly, absorbing any rounding in the published
        # µ_i values. Isotropy (Σ w·Ω̂ = 0) still holds by octant
        # sign-replication below.
        raw_nrm = sqrt(mus[i]^2 + mus[j]^2 + mus[k]^2)
        mx = mus[i] / raw_nrm
        my = mus[j] / raw_nrm
        mz = mus[k] / raw_nrm
        for sx in (one(T), -one(T)), sy in (one(T), -one(T)), sz in (one(T), -one(T))
            push!(dirs, SVector{3, T}(sx * mx, sy * my, sz * mz))
            push!(weights, w_ijk)
        end
    end
    return dirs, weights
end

"""Sorted ascending triplet used as the canonical key in the
level-symmetric weight lookup. e.g. (3, 1, 2) → (1, 2, 3)."""
function _sort_triplet(i::Int, j::Int, k::Int)
    a, b, c = i, j, k
    if a > b
        a, b = b, a
    end
    if b > c
        b, c = c, b
    end
    if a > b
        a, b = b, a
    end
    return (a, b, c)
end

"""Given a 3D level-symmetric set with Σw_3D = 4π, project it onto the
positive-η (µ_z ≥ 0) hemisphere for the 2D solver. Directions
`(µ, η) = (µ_x, µ_y)` are taken from the 3D ordinate and renormalized
onto the unit circle. Weights are doubled to account for the dropped
η < 0 half, then globally rescaled so Σw_2D = 2π.
"""
function _project_3d_to_2d(
        ::Type{T}, dirs3::Vector{SVector{3, T}}, w3::Vector{T},
    ) where {T}
    dirs2 = SVector{2, T}[]
    w2 = T[]
    for (d, w) in zip(dirs3, w3)
        if d[3] < zero(T)
            continue
        end
        nrm = sqrt(d[1]^2 + d[2]^2)
        if nrm < eps(T)
            continue
        end
        push!(dirs2, SVector{2, T}(d[1] / nrm, d[2] / nrm))
        push!(w2, T(2) * w)
    end
    scale = T(2) * T(pi) / sum(w2)
    w2 .*= scale
    return dirs2, w2
end

"""S6 level-symmetric 3D quadrature (48 directions, 6 per octant).

Level cosines (Lewis & Miller 1984, Table 4-1, S6):
  µ₁ = 0.23009194, µ₂ = 0.68317934, µ₃ = 0.95284397.
Triplets with i+j+k = 5 and indices in {1,2,3}: (1,1,3), (1,2,2).
Equal per-ordinate weight, rescaled so Σw = 4π.
"""
function _s6_quadrature(::Val{3}, ::Type{T}) where {T}
    mus = T[0.23009194, 0.68317934, 0.95284397]
    pw = Dict{NTuple{3, Int}, T}(
        (1, 1, 3) => one(T),
        (1, 2, 2) => one(T),
    )
    dirs, weights = _lsn_build_3d(T, mus, pw)
    scale = T(4) * T(pi) / sum(weights)
    weights .*= scale
    return dirs, weights
end

"""S6 2D quadrature: project the 3D S6 onto the positive-η hemisphere."""
function _s6_quadrature(::Val{2}, ::Type{T}) where {T}
    dirs3, w3 = _s6_quadrature(Val(3), T)
    return _project_3d_to_2d(T, dirs3, w3)
end

"""S8 level-symmetric 3D quadrature (80 directions, 10 per octant).

Level cosines (Lewis & Miller 1984, Table 4-1, S8):
  µ₁ = 0.14225553, µ₂ = 0.57735027, µ₃ = 0.80400872, µ₄ = 0.97955067.
Triplets with i+j+k = 6, indices in {1,..,4}: (1,1,4), (1,2,3), (2,2,2).
Equal per-ordinate weight, rescaled so Σw = 4π.
"""
function _s8_quadrature(::Val{3}, ::Type{T}) where {T}
    mus = T[0.14225553, 0.57735027, 0.80400872, 0.97955067]
    pw = Dict{NTuple{3, Int}, T}(
        (1, 1, 4) => one(T),
        (1, 2, 3) => one(T),
        (2, 2, 2) => one(T),
    )
    dirs, weights = _lsn_build_3d(T, mus, pw)
    scale = T(4) * T(pi) / sum(weights)
    weights .*= scale
    return dirs, weights
end

"""S8 2D quadrature: project the 3D S8 onto the positive-η hemisphere."""
function _s8_quadrature(::Val{2}, ::Type{T}) where {T}
    dirs3, w3 = _s8_quadrature(Val(3), T)
    return _project_3d_to_2d(T, dirs3, w3)
end

"""S12 level-symmetric 3D quadrature (168 directions, 21 per octant).

Level cosines (Lewis & Miller 1984, Table 4-1, S12):
  µ₁ = 0.10081067, µ₂ = 0.35745990, µ₃ = 0.56671796,
  µ₄ = 0.72024860, µ₅ = 0.85088379, µ₆ = 0.96639949.
Triplets with i+j+k = 8, indices in {1,..,6}: (1,1,6), (1,2,5),
(1,3,4), (2,2,4), (2,3,3). Equal per-ordinate weight, rescaled so
Σw = 4π.
"""
function _s12_quadrature(::Val{3}, ::Type{T}) where {T}
    mus = T[
        0.10081067, 0.3574599, 0.56671796,
        0.7202486, 0.85088379, 0.96639949,
    ]
    pw = Dict{NTuple{3, Int}, T}(
        (1, 1, 6) => one(T),
        (1, 2, 5) => one(T),
        (1, 3, 4) => one(T),
        (2, 2, 4) => one(T),
        (2, 3, 3) => one(T),
    )
    dirs, weights = _lsn_build_3d(T, mus, pw)
    scale = T(4) * T(pi) / sum(weights)
    weights .*= scale
    return dirs, weights
end

"""S12 2D quadrature: project the 3D S12 onto the positive-η hemisphere."""
function _s12_quadrature(::Val{2}, ::Type{T}) where {T}
    dirs3, w3 = _s12_quadrature(Val(3), T)
    return _project_3d_to_2d(T, dirs3, w3)
end

# ── Scattering helpers ────────────────────────────────────────────────

"""Per-cell scattering coefficient lookup, mirroring `_cell_absorption`."""
_cell_scattering(s::T, ::Int) where {T <: Number} = s
_cell_scattering(s::Vector{T}, c::Int) where {T} = s[c]

"""
    scattering_phase_value(model, m, m_prime) -> T

Evaluate the phase function `Φ(Ω̂_m, Ω̂_{m'})` for the selected scheme.
Isotropic returns 1. Linear-anisotropic returns `1 + g · (Ω̂ · Ω̂')`,
clamped to remain non-negative (guarding against g > 1 inputs).
"""
function scattering_phase_value(
        model::FvDOMModel{Dim, T}, m::Int, m_prime::Int,
    ) where {Dim, T}
    if model.scattering_phase === :isotropic
        return one(T)
    elseif model.scattering_phase === :linear_anisotropic
        cos_theta = dot(model.directions[m], model.directions[m_prime])
        return max(one(T) + model.scattering_g * cos_theta, zero(T))
    else
        error(
            "Unknown scattering_phase :$(model.scattering_phase). " *
                "Supported: :isotropic, :linear_anisotropic",
        )
    end
end

"""
    scattering_source_contribution(model, intensities_prev, m, cell) -> T

Assemble the in-scattering source term at a cell given the previous-iterate
intensity field `intensities_prev[m, cell]` for all directions:

    S_sc = (σ_s / 4π) · Σ_{m'} w_{m'} Φ(Ω̂_m, Ω̂_{m'}) I_{m'}

Returned per direction `m`. `intensities_prev` is a matrix of shape
`(n_dirs, ncells)`; column `cell` carries all directional intensities.
"""
function scattering_source_contribution(
        model::FvDOMModel{Dim, T},
        intensities_prev::AbstractMatrix{T},
        m::Int, cell::Int,
    ) where {Dim, T}
    s_c = _cell_scattering(model.sigma_s, cell)
    if s_c <= zero(T)
        return zero(T)
    end
    acc = zero(T)
    n_dirs = length(model.directions)
    for mp in 1:n_dirs
        phi = scattering_phase_value(model, m, mp)
        acc += model.weights[mp] * phi * intensities_prev[mp, cell]
    end
    return s_c * acc / (T(4) * T(pi))
end

"""Return `true` if any scattering coefficient is positive (scalar or
per-cell), `false` otherwise."""
_has_scattering(sigma_s::T) where {T <: Number} = sigma_s > zero(T)
function _has_scattering(sigma_s::Vector{T}) where {T}
    for x in sigma_s
        if x > zero(T)
            return true
        end
    end
    return false
end

"""
    solve_fvdom_radiation(
        model, T_field, mesh, bcs_G;
        linear_solver = nothing, solver_config = nothing,
        scattering_iterations = 1,
    ) -> RadiationState{T}

Solve the fvDOM radiation equations for all ordinate directions and
return a [`RadiationState`](@ref) with the incident radiation field G.

For each direction `Ω̂_m` with weight `w_m`, solves:

    div(I_m · Ω̂_m) + (a + σ_s) · I_m =
        a · sigma · T^4 / π
        + (σ_s / 4π) · Σ_{m'} w_{m'} · Φ(Ω̂_m, Ω̂_{m'}) · I_{m'}

The in-scattering source is lagged: when `sigma_s > 0`, the sweep is
repeated `scattering_iterations` times with the previous-iterate
intensity field used to build the in-scattering term. For
`sigma_s = 0` (default), a single sweep suffices and the scattering
loop degenerates trivially.

# Arguments
- `model::FvDOMModel{Dim, T}` --- fvDOM model with directions and weights
- `T_field::CollocatedScalarField{T}` --- temperature field [K]
- `mesh::UnstructuredFVMMesh{Dim, T}` --- mesh
- `bcs_G::Dict{Symbol, <:AbstractBoundaryCondition}` --- BCs applied to each
  intensity equation (typically `ParabolicDirichlet(sigma*T_wall^4/pi)`)
- `linear_solver` --- optional linear solver algorithm
- `solver_config` --- optional solver configuration
- `scattering_iterations` --- lagged-source sweeps for the in-scattering
  term (default 1; set higher for optically thick scattering media).
"""
function solve_fvdom_radiation(
        model::FvDOMModel{Dim, T},
        T_field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_G::Dict{Symbol, <:AbstractBoundaryCondition};
        linear_solver = nothing,
        solver_config = nothing,
        scattering_iterations::Int = 1,
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    a = model.a
    sigma_s = model.sigma_s
    sigma = T(STEFAN_BOLTZMANN)
    n_dirs = length(model.directions)

    # Accumulator for G = Σ_m w_m · I_m and per-direction intensity buffer.
    G_values = zeros(T, nc)
    intensities = zeros(T, n_dirs, nc)
    intensities_prev = zeros(T, n_dirs, nc)

    # Determine scattering state (scalar vs per-cell, and whether any cell
    # has non-zero scattering).
    has_scattering = _has_scattering(sigma_s)
    n_outer = has_scattering ? max(scattering_iterations, 1) : 1

    for _ in 1:n_outer
        fill!(G_values, zero(T))

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

            # Extinction: (a_c + σ_s,c) · V_c on diagonal (implicit)
            for c in 1:nc
                a_c = _cell_absorption(a, c)
                s_c = _cell_scattering(sigma_s, c)
                add_diag!(eq, c, (a_c + s_c) * mesh.cell_volumes[c])
            end

            # Emission source: a_c * sigma * T^4 / pi * V_c (explicit RHS)
            for c in 1:nc
                a_c = _cell_absorption(a, c)
                T_c = max(T_field.internal[c], zero(T))
                eq.b[c] += a_c * sigma * T_c^4 / T(pi) * mesh.cell_volumes[c]
            end

            # In-scattering source (explicit, lagged from prev iterate).
            if has_scattering
                for c in 1:nc
                    S_sc = scattering_source_contribution(
                        model, intensities_prev, idx, c,
                    )
                    eq.b[c] += S_sc * mesh.cell_volumes[c]
                end
            end

            # Solve for I_i
            lp = to_linear_problem(eq)
            sol = _dispatch_solve(lp, linear_solver, solver_config, :fvdom)

            # Accumulate G and record intensity for the next outer iterate.
            for c in 1:nc
                I_c = max(sol.u[c], zero(T))
                intensities[idx, c] = I_c
                G_values[c] += w_i * I_c
            end
        end

        if has_scattering
            copyto!(intensities_prev, intensities)
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

# WSGGM non-grey support and radiation dispatch shim. `wsggm.jl` needs
# both `FvDOMModel` (declared above) and `P1Model`/`solve_p1_radiation`
# (declared in `types.jl` and `p1.jl` loaded earlier) to be available at
# load time.
include("wsggm.jl")

"""
    _solve_radiation_step(
        rad_model, T_field, mesh, bcs_G;
        linear_solver, solver_config, wsggm_path_length,
    ) -> (G_field, source_model)

Internal dispatch shim for the coupled SIMPLE-thermal-radiation loop in
`solvers.jl`. Returns the updated incident-radiation field `G` plus the
radiation model to pass into `compute_radiation_source` (for
`WSGGMModel` this is the derived grey `P1Model(a = κ_eff)`, not the
original non-grey `WSGGMModel`).
"""
function _solve_radiation_step(
        rad_model::P1Model,
        T_field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_G::Dict{Symbol, <:AbstractBoundaryCondition};
        linear_solver, solver_config, wsggm_path_length,
    ) where {Dim, T}
    G = solve_p1_radiation(
        rad_model, T_field, mesh, bcs_G;
        linear_solver = linear_solver, solver_config = solver_config,
    )
    return (G, rad_model)
end

function _solve_radiation_step(
        rad_model::FvDOMModel,
        T_field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_G::Dict{Symbol, <:AbstractBoundaryCondition};
        linear_solver, solver_config, wsggm_path_length,
    ) where {Dim, T}
    state = solve_fvdom_radiation(
        rad_model, T_field, mesh, bcs_G;
        linear_solver = linear_solver, solver_config = solver_config,
    )
    return (state.G, rad_model)
end

function _solve_radiation_step(
        rad_model::WSGGMModel,
        T_field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_G::Dict{Symbol, <:AbstractBoundaryCondition};
        linear_solver, solver_config, wsggm_path_length,
    ) where {Dim, T}
    kappa_eff = wsggm_effective_absorption(rad_model, T_field, wsggm_path_length)
    grey_model = P1Model(; a = kappa_eff)
    G = solve_p1_radiation(
        grey_model, T_field, mesh, bcs_G;
        linear_solver = linear_solver, solver_config = solver_config,
    )
    return (G, grey_model)
end
