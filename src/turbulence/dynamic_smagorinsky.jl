# turbulence/dynamic_smagorinsky.jl — Dynamic Smagorinsky SGS model
#
# Computes the Smagorinsky constant Cs dynamically from the Germano
# identity using a test filter (volume-weighted neighbor average).

"""
    DynamicSmagorinsky{T} <: AbstractLESModel

Dynamic Smagorinsky SGS model with Germano identity.

The Smagorinsky constant Cs is computed dynamically each time step
using a test filter (volume-weighted average over face-connected
neighbors). This makes the model self-calibrating.

# Fields
- `delta::Vector{T}` — grid filter width per cell
- `test_filter_ratio::T` — test filter / grid filter ratio (default 2.0)
"""
struct DynamicSmagorinsky{T} <: AbstractLESModel
    delta::Vector{T}
    test_filter_ratio::T
end

"""
    DynamicSmagorinsky(mesh; test_filter_ratio = 2.0)
"""
function DynamicSmagorinsky(
        mesh::UnstructuredFVMMesh{Dim, T};
        test_filter_ratio::Real = 2.0,
    ) where {Dim, T}
    delta = compute_filter_width(mesh)
    return DynamicSmagorinsky{T}(delta, T(test_filter_ratio))
end

"""
    _test_filter(values, mesh) -> Vector

Volume-weighted average of `values` over each cell and its face-connected neighbors.
"""
function _test_filter(
        values::Vector{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    filtered = zeros(T, nc)
    weights = zeros(T, nc)

    # Self-contribution
    for c in 1:nc
        filtered[c] += values[c] * mesh.cell_volumes[c]
        weights[c] += mesh.cell_volumes[c]
    end

    # Neighbor contributions via faces
    for f in 1:nf
        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            filtered[P] += values[N] * mesh.cell_volumes[N]
            weights[P] += mesh.cell_volumes[N]
            filtered[N] += values[P] * mesh.cell_volumes[P]
            weights[N] += mesh.cell_volumes[P]
        end
    end

    for c in 1:nc
        filtered[c] /= max(weights[c], eps(T))
    end

    return filtered
end

function turbulent_viscosity!(
        nu_t::Vector{T},
        model::DynamicSmagorinsky{T},
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    alpha = model.test_filter_ratio

    # Strain rate at grid level
    S_mag = compute_strain_rate(U, mesh)

    # |S| * S_ij approximation: use |S|² as the contraction magnitude
    S_sq = [S_mag[c]^2 for c in 1:nc]

    # Test-filtered quantities
    S_mag_filtered = _test_filter(S_mag, mesh)
    S_sq_filtered = _test_filter(S_sq, mesh)

    # Compute dynamic Cs² per cell via simplified Germano-Lilly
    # M = 2Δ²(α² |S̃|² - |S|²) (simplified scalar version)
    # L = test_filter(|S|²) - test_filter(|S|)²  (scalar Leonard stress proxy)
    for c in 1:nc
        delta_c = model.delta[c]
        M = T(2) * delta_c^2 * (alpha^2 * S_mag_filtered[c]^2 - S_sq[c])

        L = S_sq_filtered[c] - S_mag_filtered[c]^2

        if abs(M) > eps(T)
            Cs_sq = max(L / M, zero(T))  # clip negative for stability
        else
            Cs_sq = T(0.01)  # fallback
        end

        # Cap Cs to prevent excessive values
        Cs_sq = min(Cs_sq, T(0.04))  # Cs < 0.2

        nu_t[c] = Cs_sq * delta_c^2 * S_mag[c]
    end

    return nothing
end
