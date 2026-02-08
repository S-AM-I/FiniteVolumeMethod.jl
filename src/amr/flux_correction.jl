# ============================================================
# Berger-Colella Flux Correction at Fine/Coarse Interfaces
# ============================================================
#
# At the interface between a fine and coarse block, the numerical
# fluxes computed on the two grids are inconsistent. The coarse
# flux is computed using the coarse solution, while the fine
# fluxes are computed using the fine solution.
#
# The Berger-Colella (1989) refluxing procedure corrects the
# coarse solution at the interface:
#
#   U_coarse += dt/dx * (sum(F_fine) / n_fine - F_coarse) * face_area
#
# where n_fine is the number of fine faces covering one coarse face
# (2 in 2D, 4 in 3D).
#
# This ensures global conservation across refinement level interfaces.

"""
    FluxRegister{N, FT, Dim}

Accumulates flux corrections at fine/coarse boundaries.

# Fields
- `coarse_flux::Dict{Symbol, Array{SVector{N,FT}, Dim-1}}`:
  Coarse flux at each face, keyed by direction (:left, :right, etc.).
- `fine_flux_sum::Dict{Symbol, Array{SVector{N,FT}, Dim-1}}`:
  Sum of fine fluxes at each coarse face location.
- `n_fine::Int`: Number of fine faces per coarse face (2 in 2D, 4 in 3D).
"""
struct FluxRegister{N, FT}
    coarse_flux::Dict{Symbol, Vector{SVector{N, FT}}}
    fine_flux_sum::Dict{Symbol, Vector{SVector{N, FT}}}
    n_fine::Int
end

"""
    FluxRegister(N, FT, n_faces, n_fine) -> FluxRegister

Create a zero-initialized flux register.
"""
function FluxRegister(::Val{N}, ::Type{FT}, n_faces::Int, n_fine::Int) where {N, FT}
    zero_state = zero(SVector{N, FT})
    coarse_flux = Dict{Symbol, Vector{SVector{N, FT}}}()
    fine_flux_sum = Dict{Symbol, Vector{SVector{N, FT}}}()
    for face in (:left, :right, :bottom, :top, :front, :back)
        coarse_flux[face] = fill(zero_state, n_faces)
        fine_flux_sum[face] = fill(zero_state, n_faces)
    end
    return FluxRegister{N, FT}(coarse_flux, fine_flux_sum, n_fine)
end

"""
    reset_flux_register!(reg::FluxRegister)

Zero out all stored fluxes.
"""
function reset_flux_register!(reg::FluxRegister{N, FT}) where {N, FT}
    zero_state = zero(SVector{N, FT})
    for (_, v) in reg.coarse_flux
        fill!(v, zero_state)
    end
    for (_, v) in reg.fine_flux_sum
        fill!(v, zero_state)
    end
    return nothing
end

"""
    accumulate_fine_flux!(reg, face, fine_face_idx, coarse_face_idx, flux)

Add a fine flux contribution at a coarse face location.
"""
function accumulate_fine_flux!(reg::FluxRegister, face::Symbol,
        coarse_face_idx::Int, flux::SVector)
    reg.fine_flux_sum[face][coarse_face_idx] = reg.fine_flux_sum[face][coarse_face_idx] + flux
    return nothing
end

"""
    store_coarse_flux!(reg, face, coarse_face_idx, flux)

Store the coarse flux at a face location.
"""
function store_coarse_flux!(reg::FluxRegister, face::Symbol,
        coarse_face_idx::Int, flux::SVector)
    reg.coarse_flux[face][coarse_face_idx] = flux
    return nothing
end

"""
    apply_flux_correction!(U_coarse, reg, dt, dx, face, law, nx, ny)

Apply the Berger-Colella flux correction to the coarse solution at the interface.

The correction is:
  dU = dt/dx * (F_fine_avg - F_coarse)
where F_fine_avg = sum(F_fine) / n_fine.

This is added to the coarse cells adjacent to the interface.
"""
function apply_flux_correction_2d!(
        U::AbstractMatrix, reg::FluxRegister{N, FT},
        dt, dx, dy, nx::Int, ny::Int, face::Symbol) where {N, FT}
    n_inv = FT(1) / FT(reg.n_fine)

    if face == :left
        for j in 1:ny
            F_coarse = reg.coarse_flux[:left][j]
            F_fine_avg = reg.fine_flux_sum[:left][j] * n_inv
            correction = (F_fine_avg - F_coarse)
            # Left boundary: flux enters cell (1, j), padded at (3, j+2)
            U[3, j + 2] = U[3, j + 2] + (dt / dx) * correction
        end
    elseif face == :right
        for j in 1:ny
            F_coarse = reg.coarse_flux[:right][j]
            F_fine_avg = reg.fine_flux_sum[:right][j] * n_inv
            correction = (F_fine_avg - F_coarse)
            # Right boundary: flux leaves cell (nx, j), padded at (nx+2, j+2)
            U[nx + 2, j + 2] = U[nx + 2, j + 2] - (dt / dx) * correction
        end
    elseif face == :bottom
        for i in 1:nx
            F_coarse = reg.coarse_flux[:bottom][i]
            F_fine_avg = reg.fine_flux_sum[:bottom][i] * n_inv
            correction = (F_fine_avg - F_coarse)
            U[i + 2, 3] = U[i + 2, 3] + (dt / dy) * correction
        end
    elseif face == :top
        for i in 1:nx
            F_coarse = reg.coarse_flux[:top][i]
            F_fine_avg = reg.fine_flux_sum[:top][i] * n_inv
            correction = (F_fine_avg - F_coarse)
            U[i + 2, ny + 2] = U[i + 2, ny + 2] - (dt / dy) * correction
        end
    end

    return nothing
end

function apply_flux_correction_3d!(
        U::AbstractArray{T, 3}, reg::FluxRegister{N, FT},
        dt, dx, dy, dz, nx::Int, ny::Int, nz::Int, face::Symbol) where {T, N, FT}
    n_inv = FT(1) / FT(reg.n_fine)

    if face == :left
        idx = 0
        for k in 1:nz, j in 1:ny
            idx += 1
            F_coarse = reg.coarse_flux[:left][idx]
            F_fine_avg = reg.fine_flux_sum[:left][idx] * n_inv
            correction = (F_fine_avg - F_coarse)
            U[3, j + 2, k + 2] = U[3, j + 2, k + 2] + (dt / dx) * correction
        end
    elseif face == :right
        idx = 0
        for k in 1:nz, j in 1:ny
            idx += 1
            F_coarse = reg.coarse_flux[:right][idx]
            F_fine_avg = reg.fine_flux_sum[:right][idx] * n_inv
            correction = (F_fine_avg - F_coarse)
            U[nx + 2, j + 2, k + 2] = U[nx + 2, j + 2, k + 2] - (dt / dx) * correction
        end
    elseif face == :bottom
        idx = 0
        for k in 1:nz, i in 1:nx
            idx += 1
            F_coarse = reg.coarse_flux[:bottom][idx]
            F_fine_avg = reg.fine_flux_sum[:bottom][idx] * n_inv
            correction = (F_fine_avg - F_coarse)
            U[i + 2, 3, k + 2] = U[i + 2, 3, k + 2] + (dt / dy) * correction
        end
    elseif face == :top
        idx = 0
        for k in 1:nz, i in 1:nx
            idx += 1
            F_coarse = reg.coarse_flux[:top][idx]
            F_fine_avg = reg.fine_flux_sum[:top][idx] * n_inv
            correction = (F_fine_avg - F_coarse)
            U[i + 2, ny + 2, k + 2] = U[i + 2, ny + 2, k + 2] - (dt / dy) * correction
        end
    elseif face == :front
        idx = 0
        for j in 1:ny, i in 1:nx
            idx += 1
            F_coarse = reg.coarse_flux[:front][idx]
            F_fine_avg = reg.fine_flux_sum[:front][idx] * n_inv
            correction = (F_fine_avg - F_coarse)
            U[i + 2, j + 2, 3] = U[i + 2, j + 2, 3] + (dt / dz) * correction
        end
    elseif face == :back
        idx = 0
        for j in 1:ny, i in 1:nx
            idx += 1
            F_coarse = reg.coarse_flux[:back][idx]
            F_fine_avg = reg.fine_flux_sum[:back][idx] * n_inv
            correction = (F_fine_avg - F_coarse)
            U[i + 2, j + 2, nz + 2] = U[i + 2, j + 2, nz + 2] - (dt / dz) * correction
        end
    end

    return nothing
end
