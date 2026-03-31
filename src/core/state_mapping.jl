# ============================================================
# State Mapping: flat SciML state <-> ghost-padded arrays
# ============================================================
#
# The ODE state is a flat Vector{FT} containing only interior cell
# values (plus face-B for MHD/CT). Ghost cells live in the cache
# and are filled during RHS evaluation via boundary conditions.
#
# reinterpret(SVector{N,FT}, u) provides zero-copy SVector views.
# SciML requires AbstractVector{<:Number} for error estimators
# and adaptive stepping, hence the flat Vector{FT} representation.

# ============================================================
# 1D State Mapping
# ============================================================

"""
    unfold_to_padded!(cache::HyperbolicCache1D{N,FT}, u) where {N,FT}

Copy interior cell data from flat SciML state `u` into the ghost-padded
array `cache.padded_U`. Ghost cells are NOT filled here; that happens
in the RHS via `apply_boundary_conditions!`.
"""
function unfold_to_padded!(cache::HyperbolicCache1D{N, FT}, u::AbstractVector) where {N, FT}
    ng = cache.ng
    nc = cache.nc
    u_sv = reinterpret(SVector{N, FT}, u)
    @inbounds for i in 1:nc
        cache.padded_U[i + ng] = u_sv[i]
    end
    return nothing
end

"""
    fold_from_padded!(du, cache::HyperbolicCache1D{N,FT}) where {N,FT}

Copy interior cell RHS data from `cache.padded_dU` back into the flat
SciML derivative vector `du`.
"""
function fold_from_padded!(du::AbstractVector, cache::HyperbolicCache1D{N, FT}) where {N, FT}
    ng = cache.ng
    nc = cache.nc
    du_sv = reinterpret(SVector{N, FT}, du)
    @inbounds for i in 1:nc
        du_sv[i] = cache.padded_dU[i + ng]
    end
    return nothing
end

# ============================================================
# 2D State Mapping
# ============================================================

function unfold_to_padded!(cache::HyperbolicCache2D{N, FT}, u::AbstractVector) where {N, FT}
    ng = cache.ng
    nx, ny = cache.nx, cache.ny
    u_sv = reinterpret(SVector{N, FT}, u)
    @inbounds for iy in 1:ny, ix in 1:nx
        flat_idx = (iy - 1) * nx + ix
        cache.padded_U[ix + ng, iy + ng] = u_sv[flat_idx]
    end
    return nothing
end

function fold_from_padded!(du::AbstractVector, cache::HyperbolicCache2D{N, FT}) where {N, FT}
    ng = cache.ng
    nx, ny = cache.nx, cache.ny
    du_sv = reinterpret(SVector{N, FT}, du)
    @inbounds for iy in 1:ny, ix in 1:nx
        flat_idx = (iy - 1) * nx + ix
        du_sv[flat_idx] = cache.padded_dU[ix + ng, iy + ng]
    end
    return nothing
end

# ============================================================
# 3D State Mapping
# ============================================================

function unfold_to_padded!(cache::HyperbolicCache3D{N, FT}, u::AbstractVector) where {N, FT}
    ng = cache.ng
    nx, ny, nz = cache.nx, cache.ny, cache.nz
    u_sv = reinterpret(SVector{N, FT}, u)
    @inbounds for iz in 1:nz, iy in 1:ny, ix in 1:nx
        flat_idx = ((iz - 1) * ny + (iy - 1)) * nx + ix
        cache.padded_U[ix + ng, iy + ng, iz + ng] = u_sv[flat_idx]
    end
    return nothing
end

function fold_from_padded!(du::AbstractVector, cache::HyperbolicCache3D{N, FT}) where {N, FT}
    ng = cache.ng
    nx, ny, nz = cache.nx, cache.ny, cache.nz
    du_sv = reinterpret(SVector{N, FT}, du)
    @inbounds for iz in 1:nz, iy in 1:ny, ix in 1:nx
        flat_idx = ((iz - 1) * ny + (iy - 1)) * nx + ix
        du_sv[flat_idx] = cache.padded_dU[ix + ng, iy + ng, iz + ng]
    end
    return nothing
end

# ============================================================
# Unstructured State Mapping
# ============================================================

function unfold_to_padded!(cache::UnstructuredCache{N, FT}, u::AbstractVector) where {N, FT}
    u_sv = reinterpret(SVector{N, FT}, u)
    @inbounds for i in 1:(cache.ntri)
        cache.U[i] = u_sv[i]
    end
    return nothing
end

function fold_from_padded!(du::AbstractVector, cache::UnstructuredCache{N, FT}) where {N, FT}
    du_sv = reinterpret(SVector{N, FT}, du)
    @inbounds for i in 1:(cache.ntri)
        du_sv[i] = cache.dU[i]
    end
    return nothing
end

# ============================================================
# MHD/CT Augmented State Mapping (2D)
# ============================================================

"""
    unfold_mhd_augmented!(cache::MHDCTCache2D{N,FT}, u) where {N,FT}

Unfold the augmented MHD state vector into the ghost-padded cell array.
Only the cell-centered conserved variables are copied; face-B is
extracted separately when needed (e.g. in stage_limiter!).

State layout: [cell_conserved (nx*ny*N) | Bx_face ((nx+1)*ny) | By_face (nx*(ny+1))]
"""
function unfold_mhd_augmented!(cache::MHDCTCache2D{N, FT}, u::AbstractVector) where {N, FT}
    ng = cache.ng
    nx, ny = cache.nx, cache.ny
    # Only unfold cell-centered conserved variables
    u_sv = reinterpret(SVector{N, FT}, @view u[1:(cache.n_cell_vars)])
    @inbounds for iy in 1:ny, ix in 1:nx
        flat_idx = (iy - 1) * nx + ix
        cache.padded_U[ix + ng, iy + ng] = u_sv[flat_idx]
    end
    return nothing
end

function unfold_mhd_augmented!(cache::GRMHDCTCache2D{N, FT}, u::AbstractVector) where {N, FT}
    ng = cache.ng
    nx, ny = cache.nx, cache.ny
    u_sv = reinterpret(SVector{N, FT}, @view u[1:(cache.n_cell_vars)])
    @inbounds for iy in 1:ny, ix in 1:nx
        flat_idx = (iy - 1) * nx + ix
        cache.padded_U[ix + ng, iy + ng] = u_sv[flat_idx]
    end
    return nothing
end

"""
    fold_mhd_augmented!(du, cache::MHDCTCache2D{N,FT}) where {N,FT}

Fold cell-centered dU and face-B time derivatives into the flat du vector.

For the cell-centered part, copies `cache.padded_dU` interior values.
For the face-B part, computes dBx/dt and dBy/dt from the EMF stored in `cache.emf_z`.
"""
function fold_mhd_augmented!(du::AbstractVector, cache::MHDCTCache2D{N, FT}) where {N, FT}
    ng = cache.ng
    nx, ny = cache.nx, cache.ny

    # Fold cell-centered dU
    du_sv = reinterpret(SVector{N, FT}, @view du[1:(cache.n_cell_vars)])
    @inbounds for iy in 1:ny, ix in 1:nx
        flat_idx = (iy - 1) * nx + ix
        du_sv[flat_idx] = cache.padded_dU[ix + ng, iy + ng]
    end

    dx = cache.prob.mesh.dx
    dy = cache.prob.mesh.dy
    emf = cache.emf_z

    # dBx_face/dt = -1/dy * (emf_z[i,j+1] - emf_z[i,j])
    bx_offset = cache.n_cell_vars
    @inbounds for j in 1:ny, i in 1:(nx + 1)
        bx_idx = bx_offset + (j - 1) * (nx + 1) + i
        du[bx_idx] = -(emf[i, j + 1] - emf[i, j]) / dy
    end

    # dBy_face/dt = +1/dx * (emf_z[i+1,j] - emf_z[i,j])
    by_offset = cache.n_cell_vars + cache.n_bx_face
    @inbounds for j in 1:(ny + 1), i in 1:nx
        by_idx = by_offset + (j - 1) * nx + i
        du[by_idx] = (emf[i + 1, j] - emf[i, j]) / dx
    end

    return nothing
end

function fold_mhd_augmented!(du::AbstractVector, cache::GRMHDCTCache2D{N, FT}) where {N, FT}
    ng = cache.ng
    nx, ny = cache.nx, cache.ny

    du_sv = reinterpret(SVector{N, FT}, @view du[1:(cache.n_cell_vars)])
    @inbounds for iy in 1:ny, ix in 1:nx
        flat_idx = (iy - 1) * nx + ix
        du_sv[flat_idx] = cache.padded_dU[ix + ng, iy + ng]
    end

    dx = cache.prob.mesh.dx
    dy = cache.prob.mesh.dy
    emf = cache.emf_z

    bx_offset = cache.n_cell_vars
    @inbounds for j in 1:ny, i in 1:(nx + 1)
        bx_idx = bx_offset + (j - 1) * (nx + 1) + i
        du[bx_idx] = -(emf[i, j + 1] - emf[i, j]) / dy
    end

    by_offset = cache.n_cell_vars + cache.n_bx_face
    @inbounds for j in 1:(ny + 1), i in 1:nx
        by_idx = by_offset + (j - 1) * nx + i
        du[by_idx] = (emf[i + 1, j] - emf[i, j]) / dx
    end

    return nothing
end

function unfold_mhd_augmented!(cache::MHDCTCache3D{N, FT}, u::AbstractVector) where {N, FT}
    ng = cache.ng
    nx, ny, nz = cache.nx, cache.ny, cache.nz
    u_sv = reinterpret(SVector{N, FT}, @view u[1:(cache.n_cell_vars)])
    @inbounds for iz in 1:nz, iy in 1:ny, ix in 1:nx
        flat_idx = ((iz - 1) * ny + (iy - 1)) * nx + ix
        cache.padded_U[ix + ng, iy + ng, iz + ng] = u_sv[flat_idx]
    end
    return nothing
end

function fold_mhd_augmented!(du::AbstractVector, cache::MHDCTCache3D{N, FT}) where {N, FT}
    ng = cache.ng
    nx, ny, nz = cache.nx, cache.ny, cache.nz

    du_sv = reinterpret(SVector{N, FT}, @view du[1:(cache.n_cell_vars)])
    @inbounds for iz in 1:nz, iy in 1:ny, ix in 1:nx
        flat_idx = ((iz - 1) * ny + (iy - 1)) * nx + ix
        du_sv[flat_idx] = cache.padded_dU[ix + ng, iy + ng, iz + ng]
    end

    dx = cache.prob.mesh.dx
    dy = cache.prob.mesh.dy
    dz = cache.prob.mesh.dz
    Ex = cache.ct.emf_x
    Ey = cache.ct.emf_y
    Ez = cache.ct.emf_z

    bx_offset = cache.n_cell_vars
    @inbounds for iz in 1:nz, iy in 1:ny, ix in 1:(nx + 1)
        bx_idx = bx_offset + ((iz - 1) * ny + (iy - 1)) * (nx + 1) + ix
        du[bx_idx] = -(Ez[ix, iy + 1, iz] - Ez[ix, iy, iz]) / dy
        du[bx_idx] += (Ey[ix, iy, iz + 1] - Ey[ix, iy, iz]) / dz
    end

    by_offset = cache.n_cell_vars + cache.n_bx_face
    @inbounds for iz in 1:nz, iy in 1:(ny + 1), ix in 1:nx
        by_idx = by_offset + ((iz - 1) * (ny + 1) + (iy - 1)) * nx + ix
        du[by_idx] = (Ez[ix + 1, iy, iz] - Ez[ix, iy, iz]) / dx
        du[by_idx] -= (Ex[ix, iy, iz + 1] - Ex[ix, iy, iz]) / dz
    end

    bz_offset = cache.n_cell_vars + cache.n_bx_face + cache.n_by_face
    @inbounds for iz in 1:(nz + 1), iy in 1:ny, ix in 1:nx
        bz_idx = bz_offset + ((iz - 1) * ny + (iy - 1)) * nx + ix
        du[bz_idx] = -(Ey[ix + 1, iy, iz] - Ey[ix, iy, iz]) / dx
        du[bz_idx] += (Ex[ix, iy + 1, iz] - Ex[ix, iy, iz]) / dy
    end

    return nothing
end

# ============================================================
# AMR State Mapping
# ============================================================

"""Copy flat ODE state into per-block ghost-padded arrays for AMR RHS evaluation."""
function unfold_amr!(cache::AMRCache{N, FT}, u::AbstractVector) where {N, FT}
    for (idx, bid) in enumerate(cache.block_ids)
        offset = cache.block_offsets[idx]
        block = cache.grid.blocks[bid]
        nx, ny = block.dims[1], block.dims[2]
        ncells_block = nx * ny
        u_block = reinterpret(SVector{N, FT}, @view u[(offset * N + 1):((offset + ncells_block) * N)])
        pad = cache.per_block_padded[bid]
        @inbounds for iy in 1:ny, ix in 1:nx
            flat_idx = (iy - 1) * nx + ix
            pad[ix + 2, iy + 2] = u_block[flat_idx]
        end
        # Zero-gradient ghost cells
        _fill_amr_ghost_2d!(pad, nx, ny)
    end
    return nothing
end

"""Copy per-block RHS results back into the flat ODE derivative vector."""
function fold_amr!(du::AbstractVector, cache::AMRCache{N, FT}) where {N, FT}
    for (idx, bid) in enumerate(cache.block_ids)
        offset = cache.block_offsets[idx]
        block = cache.grid.blocks[bid]
        nx, ny = block.dims[1], block.dims[2]
        ncells_block = nx * ny
        du_block = reinterpret(SVector{N, FT}, @view du[(offset * N + 1):((offset + ncells_block) * N)])
        du_pad = cache.per_block_dU[bid]
        @inbounds for iy in 1:ny, ix in 1:nx
            flat_idx = (iy - 1) * nx + ix
            du_block[flat_idx] = du_pad[ix + 2, iy + 2]
        end
    end
    return nothing
end

function _fill_amr_ghost_2d!(U_pad, nx, ny)
    for j in 1:(ny + 4)
        U_pad[2, j] = U_pad[3, j]
        U_pad[1, j] = U_pad[3, j]
        U_pad[nx + 3, j] = U_pad[nx + 2, j]
        U_pad[nx + 4, j] = U_pad[nx + 2, j]
    end
    for i in 1:(nx + 4)
        U_pad[i, 2] = U_pad[i, 3]
        U_pad[i, 1] = U_pad[i, 3]
        U_pad[i, ny + 3] = U_pad[i, ny + 2]
        U_pad[i, ny + 4] = U_pad[i, ny + 2]
    end
    return nothing
end

# ============================================================
# Initial State Construction
# ============================================================

"""
    initial_state_flat(prob, cache) -> Vector{FT}

Create the flat initial state vector from the problem's initial condition.
"""
function initial_state_flat(prob::HyperbolicProblem, cache::HyperbolicCache1D{N, FT}) where {N, FT}
    nc = cache.nc
    u0 = Vector{FT}(undef, nc * N)
    u0_sv = reinterpret(SVector{N, FT}, u0)
    for i in 1:nc
        x = cell_center(prob.mesh, i)
        w = prob.initial_condition(x)
        u0_sv[i] = primitive_to_conserved(prob.law, w)
    end
    return u0
end

function initial_state_flat(prob::HyperbolicProblem2D, cache::HyperbolicCache2D{N, FT}) where {N, FT}
    nx, ny = cache.nx, cache.ny
    mesh = prob.mesh
    u0 = Vector{FT}(undef, nx * ny * N)
    u0_sv = reinterpret(SVector{N, FT}, u0)
    for iy in 1:ny, ix in 1:nx
        x, y = cell_center(mesh, cell_idx(mesh, ix, iy))
        w = prob.initial_condition(x, y)
        flat_idx = (iy - 1) * nx + ix
        u0_sv[flat_idx] = primitive_to_conserved(prob.law, w)
    end
    return u0
end

function initial_state_flat(prob::HyperbolicProblem3D, cache::HyperbolicCache3D{N, FT}) where {N, FT}
    nx, ny, nz = cache.nx, cache.ny, cache.nz
    mesh = prob.mesh
    u0 = Vector{FT}(undef, nx * ny * nz * N)
    u0_sv = reinterpret(SVector{N, FT}, u0)
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        x, y, z = cell_center(mesh, cell_idx_3d(mesh, ix, iy, iz))
        w = prob.initial_condition(x, y, z)
        flat_idx = ((iz - 1) * ny + (iy - 1)) * nx + ix
        u0_sv[flat_idx] = primitive_to_conserved(prob.law, w)
    end
    return u0
end

function initial_state_flat(prob::UnstructuredHyperbolicProblem, cache::UnstructuredCache{N, FT}) where {N, FT}
    mesh = prob.mesh
    ntri = cache.ntri
    u0 = Vector{FT}(undef, ntri * N)
    u0_sv = reinterpret(SVector{N, FT}, u0)
    for i in 1:ntri
        x, y = mesh.tri_centroids[i]
        w = prob.initial_condition(x, y)
        u0_sv[i] = primitive_to_conserved(prob.law, w)
    end
    return u0
end

"""
    initial_mhd_augmented_state(prob, cache; vector_potential=nothing) -> Vector{FT}

Create the augmented initial state for MHD/CT problems.
Layout: [cell_conserved | Bx_face | By_face]
"""
function initial_mhd_augmented_state(
        prob::HyperbolicProblem2D, cache::MHDCTCache2D{N, FT};
        vector_potential = nothing
    ) where {N, FT}
    nx, ny = cache.nx, cache.ny
    mesh = prob.mesh

    total_len = cache.n_cell_vars + cache.n_bx_face + cache.n_by_face
    u0 = zeros(FT, total_len)

    # Fill cell-centered conserved variables
    u0_sv = reinterpret(SVector{N, FT}, @view u0[1:(cache.n_cell_vars)])
    for iy in 1:ny, ix in 1:nx
        x, y = cell_center(mesh, cell_idx(mesh, ix, iy))
        w = prob.initial_condition(x, y)
        flat_idx = (iy - 1) * nx + ix
        u0_sv[flat_idx] = primitive_to_conserved(prob.law, w)
    end

    # Fill face-centered B
    dx, dy = mesh.dx, mesh.dy
    bx_offset = cache.n_cell_vars
    by_offset = cache.n_cell_vars + cache.n_bx_face

    if vector_potential !== nothing
        # From vector potential Az: Bx = dAz/dy, By = -dAz/dx
        for j in 1:ny, i in 1:(nx + 1)
            x = mesh.xmin + (i - 1) * dx
            y_bottom = mesh.ymin + (j - 1) * dy
            y_top = mesh.ymin + j * dy
            bx_idx = bx_offset + (j - 1) * (nx + 1) + i
            u0[bx_idx] = (vector_potential(x, y_top) - vector_potential(x, y_bottom)) / dy
        end
        for j in 1:(ny + 1), i in 1:nx
            x_left = mesh.xmin + (i - 1) * dx
            x_right = mesh.xmin + i * dx
            y = mesh.ymin + (j - 1) * dy
            by_idx = by_offset + (j - 1) * nx + i
            u0[by_idx] = -(vector_potential(x_right, y) - vector_potential(x_left, y)) / dx
        end
    else
        # From initial condition face values
        for j in 1:ny, i in 1:(nx + 1)
            x_face = mesh.xmin + (i - 1) * dx
            y_face = mesh.ymin + (j - 0.5) * dy
            w = prob.initial_condition(x_face, y_face)
            bx_idx = bx_offset + (j - 1) * (nx + 1) + i
            u0[bx_idx] = w[6]
        end
        for j in 1:(ny + 1), i in 1:nx
            x_face = mesh.xmin + (i - 0.5) * dx
            y_face = mesh.ymin + (j - 1) * dy
            w = prob.initial_condition(x_face, y_face)
            by_idx = by_offset + (j - 1) * nx + i
            u0[by_idx] = w[7]
        end
    end

    # Sync cell-centered B from face values
    _sync_cell_B_from_faces!(u0, cache)

    return u0
end

function initial_mhd_augmented_state(
        prob::HyperbolicProblem2D{<:GRMHDEquations{2}}, cache::GRMHDCTCache2D{N, FT};
        vector_potential = nothing
    ) where {N, FT}
    # Delegate to the MHD version by creating a temporary wrapper
    # The layout is identical
    temp_cache = MHDCTCache2D{N, FT, typeof(prob)}(
        prob, cache.padded_U, cache.padded_dU, cache.nx, cache.ny, cache.ng,
        cache.Fx_all, cache.Fy_all, cache.emf_z,
        cache.n_cell_vars, cache.n_bx_face, cache.n_by_face
    )
    return initial_mhd_augmented_state(prob, temp_cache; vector_potential = vector_potential)
end

function initial_mhd_augmented_state(
        prob::HyperbolicProblem3D{<:IdealMHDEquations{3}}, cache::MHDCTCache3D{N, FT};
        vector_potential_x = nothing,
        vector_potential_y = nothing,
        vector_potential_z = nothing,
    ) where {N, FT}
    nx, ny, nz = cache.nx, cache.ny, cache.nz
    mesh = prob.mesh

    total_len = cache.n_cell_vars + cache.n_bx_face + cache.n_by_face + cache.n_bz_face
    u0 = zeros(FT, total_len)

    u0_sv = reinterpret(SVector{N, FT}, @view u0[1:(cache.n_cell_vars)])
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        x, y, z = cell_center(mesh, cell_idx_3d(mesh, ix, iy, iz))
        w = prob.initial_condition(x, y, z)
        flat_idx = ((iz - 1) * ny + (iy - 1)) * nx + ix
        u0_sv[flat_idx] = primitive_to_conserved(prob.law, w)
    end

    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    bx_offset = cache.n_cell_vars
    by_offset = cache.n_cell_vars + cache.n_bx_face
    bz_offset = cache.n_cell_vars + cache.n_bx_face + cache.n_by_face
    has_vector_potential =
        vector_potential_x !== nothing &&
        vector_potential_y !== nothing &&
        vector_potential_z !== nothing

    if has_vector_potential
        for iz in 1:nz, iy in 1:ny, ix in 1:(nx + 1)
            x = mesh.xmin + (ix - 1) * dx
            y_lo = mesh.ymin + (iy - 1) * dy
            y_hi = mesh.ymin + iy * dy
            z_lo = mesh.zmin + (iz - 1) * dz
            z_hi = mesh.zmin + iz * dz
            y_mid = 0.5 * (y_lo + y_hi)
            z_mid = 0.5 * (z_lo + z_hi)
            bx_idx = bx_offset + ((iz - 1) * ny + (iy - 1)) * (nx + 1) + ix
            dAz_dy = (vector_potential_z(x, y_hi, z_mid) - vector_potential_z(x, y_lo, z_mid)) / dy
            dAy_dz = (vector_potential_y(x, y_mid, z_hi) - vector_potential_y(x, y_mid, z_lo)) / dz
            u0[bx_idx] = dAz_dy - dAy_dz
        end

        for iz in 1:nz, iy in 1:(ny + 1), ix in 1:nx
            x_lo = mesh.xmin + (ix - 1) * dx
            x_hi = mesh.xmin + ix * dx
            y = mesh.ymin + (iy - 1) * dy
            z_lo = mesh.zmin + (iz - 1) * dz
            z_hi = mesh.zmin + iz * dz
            x_mid = 0.5 * (x_lo + x_hi)
            z_mid = 0.5 * (z_lo + z_hi)
            by_idx = by_offset + ((iz - 1) * (ny + 1) + (iy - 1)) * nx + ix
            dAx_dz = (vector_potential_x(x_mid, y, z_hi) - vector_potential_x(x_mid, y, z_lo)) / dz
            dAz_dx = (vector_potential_z(x_hi, y, z_mid) - vector_potential_z(x_lo, y, z_mid)) / dx
            u0[by_idx] = dAx_dz - dAz_dx
        end

        for iz in 1:(nz + 1), iy in 1:ny, ix in 1:nx
            x_lo = mesh.xmin + (ix - 1) * dx
            x_hi = mesh.xmin + ix * dx
            y_lo = mesh.ymin + (iy - 1) * dy
            y_hi = mesh.ymin + iy * dy
            z = mesh.zmin + (iz - 1) * dz
            x_mid = 0.5 * (x_lo + x_hi)
            y_mid = 0.5 * (y_lo + y_hi)
            bz_idx = bz_offset + ((iz - 1) * ny + (iy - 1)) * nx + ix
            dAy_dx = (vector_potential_y(x_hi, y_mid, z) - vector_potential_y(x_lo, y_mid, z)) / dx
            dAx_dy = (vector_potential_x(x_mid, y_hi, z) - vector_potential_x(x_mid, y_lo, z)) / dy
            u0[bz_idx] = dAy_dx - dAx_dy
        end
    else
        for iz in 1:nz, iy in 1:ny, ix in 1:(nx + 1)
            x_face = mesh.xmin + (ix - 1) * dx
            y_face = mesh.ymin + (iy - 0.5) * dy
            z_face = mesh.zmin + (iz - 0.5) * dz
            w = prob.initial_condition(x_face, y_face, z_face)
            bx_idx = bx_offset + ((iz - 1) * ny + (iy - 1)) * (nx + 1) + ix
            u0[bx_idx] = w[6]
        end

        for iz in 1:nz, iy in 1:(ny + 1), ix in 1:nx
            x_face = mesh.xmin + (ix - 0.5) * dx
            y_face = mesh.ymin + (iy - 1) * dy
            z_face = mesh.zmin + (iz - 0.5) * dz
            w = prob.initial_condition(x_face, y_face, z_face)
            by_idx = by_offset + ((iz - 1) * (ny + 1) + (iy - 1)) * nx + ix
            u0[by_idx] = w[7]
        end

        for iz in 1:(nz + 1), iy in 1:ny, ix in 1:nx
            x_face = mesh.xmin + (ix - 0.5) * dx
            y_face = mesh.ymin + (iy - 0.5) * dy
            z_face = mesh.zmin + (iz - 1) * dz
            w = prob.initial_condition(x_face, y_face, z_face)
            bz_idx = bz_offset + ((iz - 1) * ny + (iy - 1)) * nx + ix
            u0[bz_idx] = w[8]
        end
    end

    _sync_cell_B_from_faces!(u0, cache)

    return u0
end

"""
    _sync_cell_B_from_faces!(u, cache)

Overwrite cell-centered Bx (index 6) and By (index 7) in the flat state
with averages of adjacent face-centered B values.
"""
function _sync_cell_B_from_faces!(u::AbstractVector{FT}, cache) where {FT}
    nx, ny = cache.nx, cache.ny
    N_var = div(cache.n_cell_vars, nx * ny)  # nvariables
    bx_offset = cache.n_cell_vars
    by_offset = cache.n_cell_vars + cache.n_bx_face

    for iy in 1:ny, ix in 1:nx
        # Bx_face[ix, iy] and Bx_face[ix+1, iy]
        bx_left_idx = bx_offset + (iy - 1) * (nx + 1) + ix
        bx_right_idx = bx_offset + (iy - 1) * (nx + 1) + (ix + 1)
        Bx_cell = 0.5 * (u[bx_left_idx] + u[bx_right_idx])

        # By_face[ix, iy] and By_face[ix, iy+1]
        by_bottom_idx = by_offset + (iy - 1) * nx + ix
        by_top_idx = by_offset + iy * nx + ix
        By_cell = 0.5 * (u[by_bottom_idx] + u[by_top_idx])

        # Overwrite Bx (component 6) and By (component 7) in cell conserved vars
        cell_base = ((iy - 1) * nx + (ix - 1)) * N_var
        u[cell_base + 6] = Bx_cell
        u[cell_base + 7] = By_cell
    end
    return nothing
end

function _sync_cell_B_from_faces!(u::AbstractVector{FT}, cache::MHDCTCache3D) where {FT}
    nx, ny, nz = cache.nx, cache.ny, cache.nz
    N_var = div(cache.n_cell_vars, nx * ny * nz)
    bx_offset = cache.n_cell_vars
    by_offset = cache.n_cell_vars + cache.n_bx_face
    bz_offset = cache.n_cell_vars + cache.n_bx_face + cache.n_by_face

    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        bx_left_idx = bx_offset + ((iz - 1) * ny + (iy - 1)) * (nx + 1) + ix
        bx_right_idx = bx_offset + ((iz - 1) * ny + (iy - 1)) * (nx + 1) + (ix + 1)
        bx_cell = 0.5 * (u[bx_left_idx] + u[bx_right_idx])

        by_lower_idx = by_offset + ((iz - 1) * (ny + 1) + (iy - 1)) * nx + ix
        by_upper_idx = by_offset + ((iz - 1) * (ny + 1) + iy) * nx + ix
        by_cell = 0.5 * (u[by_lower_idx] + u[by_upper_idx])

        bz_back_idx = bz_offset + ((iz - 1) * ny + (iy - 1)) * nx + ix
        bz_front_idx = bz_offset + (iz * ny + (iy - 1)) * nx + ix
        bz_cell = 0.5 * (u[bz_back_idx] + u[bz_front_idx])

        cell_base = (((iz - 1) * ny + (iy - 1)) * nx + (ix - 1)) * N_var
        u[cell_base + 6] = bx_cell
        u[cell_base + 7] = by_cell
        u[cell_base + 8] = bz_cell
    end

    return nothing
end

"""
    flatten_amr_state(cache::AMRCache{N,FT}) -> Vector{FT}

Flatten all active AMR block interiors into a single state vector.
"""
function flatten_amr_state(cache::AMRCache{N, FT}) where {N, FT}
    u0 = Vector{FT}(undef, cache.total_cells * N)
    for (idx, bid) in enumerate(cache.block_ids)
        offset = cache.block_offsets[idx]
        block = cache.grid.blocks[bid]
        nx, ny = block.dims[1], block.dims[2]
        ncells_block = nx * ny
        u0_block = reinterpret(SVector{N, FT}, @view u0[(offset * N + 1):((offset + ncells_block) * N)])
        @inbounds for iy in 1:ny, ix in 1:nx
            flat_idx = (iy - 1) * nx + ix
            u0_block[flat_idx] = block.U[ix, iy]
        end
    end
    return u0
end
