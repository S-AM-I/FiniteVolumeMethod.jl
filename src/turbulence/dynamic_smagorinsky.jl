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

    # ── 1. Compute velocity gradients and strain rate tensor ────────
    grad_U = Vector{Vector{SVector{Dim, T}}}(undef, Dim)
    for d in 1:Dim
        u_d_field = CollocatedScalarField(Symbol(:U, d), mesh; value = zero(T))
        for c in 1:nc
            u_d_field.internal[c] = U.internal[c][d]
        end
        for (i, f) in enumerate(u_d_field.boundary_face_indices)
            bi = findfirst(==(f), U.boundary_face_indices)
            if bi !== nothing
                u_d_field.boundary[i] = U.boundary[bi][d]
            end
        end
        grad_U[d] = gradient(u_d_field, mesh)
    end

    # Strain rate magnitude
    S_mag = Vector{T}(undef, nc)
    for c in 1:nc
        S_sq = _strain_rate_squared(Val(Dim), grad_U, c)
        S_mag[c] = sqrt(max(S_sq, zero(T)))
    end

    # ── 2. Test-filter velocity components ──────────────────────────
    U_comp = [T[U.internal[c][d] for c in 1:nc] for d in 1:Dim]
    U_filtered = [_test_filter(U_comp[d], mesh) for d in 1:Dim]

    # ── 3. Leonard stress tensor L_ij (symmetric, via contraction) ──
    # L_ij = test_filter(u_i * u_j) - test_filter(u_i) * test_filter(u_j)
    # We only need the contraction L_ij * M_ij, computed per cell.

    # Pre-compute u_i * u_j products and test-filter them
    # For symmetric tensor: store as flat array of unique components
    n_sym = Dim == 2 ? 3 : 6  # xx, yy, xy [, zz, xz, yz]
    uiuj = [Vector{T}(undef, nc) for _ in 1:n_sym]
    _fill_uiuj!(uiuj, U_comp, nc, Val(Dim))
    uiuj_filtered = [_test_filter(uiuj[k], mesh) for k in 1:n_sym]

    # ── 4. Test-filtered strain rate ────────────────────────────────
    S_mag_filtered = _test_filter(S_mag, mesh)

    # ── 5. Germano-Lilly with full tensor contraction ──────────────
    for c in 1:nc
        delta_c = model.delta[c]

        # Leonard stress components
        L = _leonard_components(
            uiuj_filtered, U_filtered, c, nc, Val(Dim),
        )

        # Grid-level S_ij components
        S_grid = _strain_components(grad_U, c, Val(Dim))

        # Test-filtered S_ij: approximate from filtered velocity gradients
        # For efficiency, use |S̃| and directional proportions from grid level
        S_filt = S_grid .* (S_mag_filtered[c] / max(S_mag[c], eps(T)))

        # M_ij = 2Δ²(α² |S̃| S̃_ij - |S| S_ij)
        M = ntuple(length(S_grid)) do k
            T(2) * delta_c^2 * (
                alpha^2 * S_mag_filtered[c] * S_filt[k] -
                S_mag[c] * S_grid[k]
            )
        end

        # Contractions: L_ij M_ij and M_ij M_ij
        LM = _sym_contract(L, M, Val(Dim))
        MM = _sym_contract(M, M, Val(Dim))

        if MM > eps(T) * T(100)
            Cs_sq = max(LM / MM, zero(T))
        else
            Cs_sq = T(0.01)
        end

        Cs_sq = min(Cs_sq, T(0.04))  # cap Cs < 0.2
        nu_t[c] = Cs_sq * delta_c^2 * S_mag[c]
    end

    return nothing
end

# ── Tensor helpers for Germano identity ────────────────────────────

"""Fill u_i*u_j product arrays (symmetric components only)."""
function _fill_uiuj!(uiuj, U_comp, nc, ::Val{2})
    for c in 1:nc
        uiuj[1][c] = U_comp[1][c] * U_comp[1][c]  # uu
        uiuj[2][c] = U_comp[2][c] * U_comp[2][c]  # vv
        uiuj[3][c] = U_comp[1][c] * U_comp[2][c]  # uv
    end
end

function _fill_uiuj!(uiuj, U_comp, nc, ::Val{3})
    for c in 1:nc
        uiuj[1][c] = U_comp[1][c] * U_comp[1][c]  # uu
        uiuj[2][c] = U_comp[2][c] * U_comp[2][c]  # vv
        uiuj[3][c] = U_comp[1][c] * U_comp[2][c]  # uv
        uiuj[4][c] = U_comp[3][c] * U_comp[3][c]  # ww
        uiuj[5][c] = U_comp[1][c] * U_comp[3][c]  # uw
        uiuj[6][c] = U_comp[2][c] * U_comp[3][c]  # vw
    end
end

"""Compute Leonard stress components at cell c."""
function _leonard_components(uiuj_filt, U_filt, c, nc, ::Val{2})
    return (
        uiuj_filt[1][c] - U_filt[1][c] * U_filt[1][c],  # L_xx
        uiuj_filt[2][c] - U_filt[2][c] * U_filt[2][c],  # L_yy
        uiuj_filt[3][c] - U_filt[1][c] * U_filt[2][c],  # L_xy
    )
end

function _leonard_components(uiuj_filt, U_filt, c, nc, ::Val{3})
    return (
        uiuj_filt[1][c] - U_filt[1][c] * U_filt[1][c],  # L_xx
        uiuj_filt[2][c] - U_filt[2][c] * U_filt[2][c],  # L_yy
        uiuj_filt[3][c] - U_filt[1][c] * U_filt[2][c],  # L_xy
        uiuj_filt[4][c] - U_filt[3][c] * U_filt[3][c],  # L_zz
        uiuj_filt[5][c] - U_filt[1][c] * U_filt[3][c],  # L_xz
        uiuj_filt[6][c] - U_filt[2][c] * U_filt[3][c],  # L_yz
    )
end

"""Extract strain rate components at cell c: (S_xx, S_yy, S_xy[, S_zz, S_xz, S_yz])."""
function _strain_components(grad_U, c, ::Val{2})
    dudx = grad_U[1][c][1]; dudy = grad_U[1][c][2]
    dvdx = grad_U[2][c][1]; dvdy = grad_U[2][c][2]
    S_xx = dudx
    S_yy = dvdy
    S_xy = typeof(dudx)(0.5) * (dudy + dvdx)
    return (S_xx, S_yy, S_xy)
end

function _strain_components(grad_U, c, ::Val{3})
    dudx = grad_U[1][c][1]; dudy = grad_U[1][c][2]; dudz = grad_U[1][c][3]
    dvdx = grad_U[2][c][1]; dvdy = grad_U[2][c][2]; dvdz = grad_U[2][c][3]
    dwdx = grad_U[3][c][1]; dwdy = grad_U[3][c][2]; dwdz = grad_U[3][c][3]
    h = typeof(dudx)(0.5)
    return (dudx, dvdy, h * (dudy + dvdx), dwdz, h * (dudz + dwdx), h * (dvdz + dwdy))
end

"""Symmetric tensor double contraction: A_ij B_ij with off-diag counted twice."""
function _sym_contract(A, B, ::Val{2})
    return A[1] * B[1] + A[2] * B[2] + typeof(A[1])(2) * A[3] * B[3]
end

function _sym_contract(A, B, ::Val{3})
    return A[1] * B[1] + A[2] * B[2] + A[4] * B[4] +
           typeof(A[1])(2) * (A[3] * B[3] + A[5] * B[5] + A[6] * B[6])
end
