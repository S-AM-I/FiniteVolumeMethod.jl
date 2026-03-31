# Higher-order numerical schemes
# Migrated from Simu.jl SimuFVM/schemes.jl

using .ParabolicLimiters: limit_slope_1d

"""
    muscl_reconstruction_1d(phi, i, direction, limiter_type)

MUSCL (Monotone Upstream-centered Scheme for Conservation Laws) reconstruction.
Returns reconstructed value at face.
direction can be :left or :right.
limiter_type can be :minmod, :superbee, :van_leer, or :venkatakrishnan.
"""
function muscl_reconstruction_1d(phi, i::Int, direction::Symbol, limiter_type::Symbol = :minmod)
    nx = length(phi)

    return if direction == :left
        if i == 1
            slope = limit_slope_1d(phi, 1, :left, limiter_type)
            return phi[1] - 0.5 * slope
        elseif i == 2
            return 0.5 * (phi[1] + phi[2])
        else
            slope = limit_slope_1d(phi, i - 1, :right, limiter_type)
            return phi[i - 1] + 0.5 * slope
        end
    else # direction == :right
        if i >= nx
            slope = limit_slope_1d(phi, nx, :right, limiter_type)
            return phi[nx] + 0.5 * slope
        elseif i == nx - 1
            return 0.5 * (phi[nx - 1] + phi[nx])
        else
            slope = limit_slope_1d(phi, i, :right, limiter_type)
            return phi[i] + 0.5 * slope
        end
    end
end

"""
    quick_reconstruction_1d(phi, i, direction, mesh=nothing)

QUICK (Quadratic Upstream Interpolation for Convective Kinematics) reconstruction.
Returns reconstructed value at face using quadratic interpolation.
"""
function quick_reconstruction_1d(phi, i::Int, direction::Symbol, mesh = nothing)
    nx = length(phi)

    return if direction == :left
        if i == 1
            if nx >= 3
                return 1.875 * phi[1] - 1.25 * phi[2] + 0.375 * phi[3]
            elseif nx == 2
                return 1.5 * phi[1] - 0.5 * phi[2]
            else
                return phi[1]
            end
        elseif i == 2
            if mesh !== nothing && mesh isa Mesh1D
                dx1 = mesh.cells[2].center - mesh.cells[1].center
                x_face = (mesh.cells[1].center + mesh.cells[2].center) / 2.0
                w1 = (mesh.cells[2].center - x_face) / dx1
                w2 = (x_face - mesh.cells[1].center) / dx1
                return w1 * phi[1] + w2 * phi[2]
            else
                return 0.5 * (phi[1] + phi[2])
            end
        else
            if mesh !== nothing && mesh isa Mesh1D
                x_i2 = mesh.cells[i - 2].center
                x_i1 = mesh.cells[i - 1].center
                x_i = mesh.cells[i].center
                x_face = (x_i1 + x_i) / 2.0

                L2 = ((x_face - x_i1) * (x_face - x_i)) / ((x_i2 - x_i1) * (x_i2 - x_i))
                L1 = ((x_face - x_i2) * (x_face - x_i)) / ((x_i1 - x_i2) * (x_i1 - x_i))
                L0 = ((x_face - x_i2) * (x_face - x_i1)) / ((x_i - x_i2) * (x_i - x_i1))

                return L2 * phi[i - 2] + L1 * phi[i - 1] + L0 * phi[i]
            else
                return -0.125 * phi[i - 2] + 0.75 * phi[i - 1] + 0.375 * phi[i]
            end
        end
    else # direction == :right
        if i == nx
            if nx >= 3
                return 1.875 * phi[nx] - 1.25 * phi[nx - 1] + 0.375 * phi[nx - 2]
            elseif nx == 2
                return 1.5 * phi[nx] - 0.5 * phi[nx - 1]
            else
                return phi[nx]
            end
        elseif i == nx - 1
            if mesh !== nothing && mesh isa Mesh1D
                dx1 = mesh.cells[nx].center - mesh.cells[nx - 1].center
                x_face = (mesh.cells[nx - 1].center + mesh.cells[nx].center) / 2.0
                w1 = (mesh.cells[nx].center - x_face) / dx1
                w2 = (x_face - mesh.cells[nx - 1].center) / dx1
                return w1 * phi[nx - 1] + w2 * phi[nx]
            else
                return 0.5 * (phi[nx - 1] + phi[nx])
            end
        else
            if mesh !== nothing && mesh isa Mesh1D
                x_i = mesh.cells[i].center
                x_i1 = mesh.cells[i + 1].center
                x_i2 = mesh.cells[i + 2].center
                x_face = (x_i + x_i1) / 2.0

                L0 = ((x_face - x_i1) * (x_face - x_i2)) / ((x_i - x_i1) * (x_i - x_i2))
                L1 = ((x_face - x_i) * (x_face - x_i2)) / ((x_i1 - x_i) * (x_i1 - x_i2))
                L2 = ((x_face - x_i) * (x_face - x_i1)) / ((x_i2 - x_i) * (x_i2 - x_i1))

                return L0 * phi[i] + L1 * phi[i + 1] + L2 * phi[i + 2]
            else
                return 0.375 * phi[i] + 0.75 * phi[i + 1] - 0.125 * phi[i + 2]
            end
        end
    end
end

"""
    second_order_diffusion_flux_1d(diffusion, mesh, phi, i, direction)

Compute second-order accurate diffusion flux at face.
Uses gradient reconstruction for second-order accuracy.
"""
function second_order_diffusion_flux_1d(diffusion::Union{Diffusion1D, VariableDiffusion1D}, mesh::Mesh1D, phi, i::Int, direction::Symbol)
    gamma = get_diffusion_coefficient(diffusion, mesh, i)

    return if direction == :left
        if i == 1
            dx = mesh.cells[i].center - mesh.nodes[1].x
            return gamma * (phi[i] - 0.0) / dx
        else
            grad = reconstruct_gradient_green_gauss(mesh, phi, i - 1)
            return gamma * grad
        end
    else # direction == :right
        if i == length(mesh.cells)
            dx = mesh.nodes[end].x - mesh.cells[i].center
            return gamma * (0.0 - phi[i]) / dx
        else
            grad = reconstruct_gradient_green_gauss(mesh, phi, i)
            return gamma * grad
        end
    end
end

"""
    muscl_advection_flux_1d(advection, mesh, phi, i, direction, limiter_type)

Compute MUSCL advection flux at face.
"""
function muscl_advection_flux_1d(advection::Union{Advection1D, VariableAdvection1D}, mesh::Mesh1D, phi, i::Int, direction::Symbol, limiter_type::Symbol = :minmod)
    v = get_velocity(advection, mesh, i, direction)

    return if direction == :left
        phi_L = muscl_reconstruction_1d(phi, i, :left, limiter_type)
        if i > 1
            phi_R = muscl_reconstruction_1d(phi, i - 1, :right, limiter_type)
        else
            phi_R = phi[i]
        end

        if v >= 0
            return v * phi_L
        else
            return v * phi_R
        end
    else # direction == :right
        phi_L = muscl_reconstruction_1d(phi, i, :right, limiter_type)
        if i < length(phi)
            phi_R = muscl_reconstruction_1d(phi, i + 1, :left, limiter_type)
        else
            phi_R = phi[i]
        end

        if v >= 0
            return v * phi_L
        else
            return v * phi_R
        end
    end
end

"""
    quick_advection_flux_1d(advection, mesh, phi, i, direction)

Compute QUICK advection flux at face.
"""
function quick_advection_flux_1d(advection::Union{Advection1D, VariableAdvection1D}, mesh::Mesh1D, phi, i::Int, direction::Symbol)
    v = get_velocity(advection, mesh, i, direction)

    return if direction == :left
        phi_face = quick_reconstruction_1d(phi, i, :left, mesh)
        if v >= 0
            return v * phi_face
        else
            if i > 1
                phi_face = quick_reconstruction_1d(phi, i - 1, :right, mesh)
            else
                phi_face = phi[i]
            end
            return v * phi_face
        end
    else # direction == :right
        phi_face = quick_reconstruction_1d(phi, i, :right, mesh)
        if v >= 0
            return v * phi_face
        else
            if i < length(phi)
                phi_face = quick_reconstruction_1d(phi, i + 1, :left, mesh)
            else
                phi_face = phi[i]
            end
            return v * phi_face
        end
    end
end

"""
    weno5_reconstruction_1d(phi, i, direction)

WENO5 reconstruction at face.
direction: :left (face i-1/2), :right (face i+1/2).
Returns reconstructed value.
"""
function weno5_reconstruction_1d(phi, i::Int, direction::Symbol)
    nx = length(phi)
    epsilon = 1.0e-6

    idx = (direction == :left) ? i - 1 : i

    if idx < 3 || idx > nx - 2
        return quick_reconstruction_1d(phi, idx, :right)
    end

    v1 = phi[idx - 2]
    v2 = phi[idx - 1]
    v3 = phi[idx]
    v4 = phi[idx + 1]
    v5 = phi[idx + 2]

    # Smoothness indicators
    beta0 = 13.0 / 12.0 * (v1 - 2.0 * v2 + v3)^2 + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3)^2
    beta1 = 13.0 / 12.0 * (v2 - 2.0 * v3 + v4)^2 + 0.25 * (v2 - v4)^2
    beta2 = 13.0 / 12.0 * (v3 - 2.0 * v4 + v5)^2 + 0.25 * (3.0 * v3 - 4.0 * v4 + v5)^2

    # Weights
    d0 = 0.1; d1 = 0.6; d2 = 0.3

    alpha0 = d0 / (epsilon + beta0)^2
    alpha1 = d1 / (epsilon + beta1)^2
    alpha2 = d2 / (epsilon + beta2)^2

    sum_alpha = alpha0 + alpha1 + alpha2
    w0 = alpha0 / sum_alpha
    w1 = alpha1 / sum_alpha
    w2 = alpha2 / sum_alpha

    # Candidate stencils
    q0 = 1.0 / 3.0 * v1 - 7.0 / 6.0 * v2 + 11.0 / 6.0 * v3
    q1 = -1.0 / 6.0 * v2 + 5.0 / 6.0 * v3 + 1.0 / 3.0 * v4
    q2 = 1.0 / 3.0 * v3 + 5.0 / 6.0 * v4 - 1.0 / 6.0 * v5

    return w0 * q0 + w1 * q1 + w2 * q2
end

"""
    weno5_advection_flux_1d(advection, mesh, phi, i, direction)

Compute WENO5 advection flux at face.
"""
function weno5_advection_flux_1d(advection::Union{Advection1D, VariableAdvection1D}, mesh::Mesh1D, phi, i::Int, direction::Symbol)
    v = get_velocity(advection, mesh, i, direction)

    u_minus = weno5_reconstruction_1d(phi, i, direction)

    val = 0.0

    if v >= 0
        val = u_minus
    else
        val = weno5_reconstruction_right_biased(phi, i, direction)
    end

    return v * val
end

"""WENO5 reconstruction of `phi` at cell `i` using a right-biased stencil in the given `direction`."""
function weno5_reconstruction_right_biased(phi, i::Int, direction::Symbol)
    nx = length(phi)
    epsilon = 1.0e-6

    idx = (direction == :left) ? i : i + 1

    if idx < 3 || idx > nx - 2
        if idx <= nx
            return quick_reconstruction_1d(phi, idx, :left)
        else
            return phi[nx]
        end
    end

    v1 = phi[idx - 2]
    v2 = phi[idx - 1]
    v3 = phi[idx]
    v4 = phi[idx + 1]
    v5 = phi[idx + 2]

    # Mirror: u_plus(v1..v5) = u_minus(v5..v1)
    rv1, rv2, rv3, rv4, rv5 = v5, v4, v3, v2, v1

    rbeta0 = 13.0 / 12.0 * (rv1 - 2.0 * rv2 + rv3)^2 + 0.25 * (rv1 - 4.0 * rv2 + 3.0 * rv3)^2
    rbeta1 = 13.0 / 12.0 * (rv2 - 2.0 * rv3 + rv4)^2 + 0.25 * (rv2 - rv4)^2
    rbeta2 = 13.0 / 12.0 * (rv3 - 2.0 * rv4 + rv5)^2 + 0.25 * (3.0 * rv3 - 4.0 * rv4 + rv5)^2

    d0 = 0.1; d1 = 0.6; d2 = 0.3

    alpha0 = d0 / (epsilon + rbeta0)^2
    alpha1 = d1 / (epsilon + rbeta1)^2
    alpha2 = d2 / (epsilon + rbeta2)^2

    sum_alpha = alpha0 + alpha1 + alpha2
    w0 = alpha0 / sum_alpha
    w1 = alpha1 / sum_alpha
    w2 = alpha2 / sum_alpha

    q0 = 1.0 / 3.0 * rv1 - 7.0 / 6.0 * rv2 + 11.0 / 6.0 * rv3
    q1 = -1.0 / 6.0 * rv2 + 5.0 / 6.0 * rv3 + 1.0 / 3.0 * rv4
    q2 = 1.0 / 3.0 * rv3 + 5.0 / 6.0 * rv4 - 1.0 / 6.0 * rv5

    return w0 * q0 + w1 * q1 + w2 * q2
end

# Higher-order schemes for 2D

"""
    muscl_reconstruction_2d(phi, mesh, i, j, direction, limiter_type)

MUSCL reconstruction for 2D advection.
direction can be :left, :right, :bottom, or :top.
"""
function muscl_reconstruction_2d(phi, mesh::Mesh2D, i::Int, j::Int, direction::Symbol, limiter_type::Symbol = :minmod)
    ny = mesh.ny
    k = (i - 1) * ny + j

    return if direction == :left || direction == :right
        if direction == :left
            if i <= 1
                return phi[k]
            elseif i == 2
                k_left = (i - 2) * ny + j
                return 0.5 * (phi[k_left] + phi[k])
            else
                phi_x = [phi[(ii - 1) * ny + j] for ii in max(1, i - 2):min(mesh.nx, i + 1)]
                idx_local = i - max(1, i - 2) + 1
                return muscl_reconstruction_1d(phi_x, idx_local, :left, limiter_type)
            end
        else # :right
            if i >= mesh.nx
                return phi[k]
            elseif i == mesh.nx - 1
                k_right = i * ny + j
                return 0.5 * (phi[k] + phi[k_right])
            else
                phi_x = [phi[(ii - 1) * ny + j] for ii in max(1, i - 1):min(mesh.nx, i + 2)]
                idx_local = i - max(1, i - 1) + 1
                return muscl_reconstruction_1d(phi_x, idx_local, :right, limiter_type)
            end
        end
    else # :bottom || :top
        if direction == :bottom
            if j <= 1
                return phi[k]
            elseif j == 2
                k_bottom = (i - 1) * ny + (j - 1)
                return 0.5 * (phi[k_bottom] + phi[k])
            else
                phi_y = [phi[(i - 1) * ny + jj] for jj in max(1, j - 2):min(ny, j + 1)]
                idx_local = j - max(1, j - 2) + 1
                return muscl_reconstruction_1d(phi_y, idx_local, :left, limiter_type)
            end
        else # :top
            if j >= ny
                return phi[k]
            elseif j == ny - 1
                k_top = (i - 1) * ny + (j + 1)
                return 0.5 * (phi[k] + phi[k_top])
            else
                phi_y = [phi[(i - 1) * ny + jj] for jj in max(1, j - 1):min(ny, j + 2)]
                idx_local = j - max(1, j - 1) + 1
                return muscl_reconstruction_1d(phi_y, idx_local, :right, limiter_type)
            end
        end
    end
end

"""
    quick_reconstruction_2d(phi, mesh, i, j, direction)

QUICK reconstruction for 2D advection.
Uses direction-split approach: reconstructs in one direction at a time.
"""
function quick_reconstruction_2d(phi, mesh::Mesh2D, i::Int, j::Int, direction::Symbol)
    ny = mesh.ny

    if direction == :left || direction == :right
        phi_x = [phi[(ii - 1) * ny + j] for ii in 1:mesh.nx]
        return quick_reconstruction_1d(phi_x, i, direction, mesh)
    else # :bottom || :top
        phi_y = [phi[(i - 1) * ny + jj] for jj in 1:ny]
        dir_map = direction == :bottom ? :left : :right
        return quick_reconstruction_1d(phi_y, j, dir_map, mesh)
    end
end

"""
    muscl_advection_flux_2d(advection, mesh, phi, i, j, direction, limiter_type)

Compute MUSCL advection flux at face for 2D problem.
"""
function muscl_advection_flux_2d(advection::Union{Advection2D, VariableAdvection2D}, mesh::Mesh2D, phi, i::Int, j::Int, direction::Symbol, limiter_type::Symbol = :minmod)
    if direction == :left || direction == :right
        v = advection.vx
        phi_face = muscl_reconstruction_2d(phi, mesh, i, j, direction, limiter_type)
        return v * phi_face
    else # :bottom || :top
        v = advection.vy
        phi_face = muscl_reconstruction_2d(phi, mesh, i, j, direction, limiter_type)
        return v * phi_face
    end
end

"""
    quick_advection_flux_2d(advection, mesh, phi, i, j, direction)

Compute QUICK advection flux at face for 2D problem.
"""
function quick_advection_flux_2d(advection::Union{Advection2D, VariableAdvection2D}, mesh::Mesh2D, phi, i::Int, j::Int, direction::Symbol)
    if direction == :left || direction == :right
        v = advection.vx
        phi_face = quick_reconstruction_2d(phi, mesh, i, j, direction)
        return v * phi_face
    else # :bottom || :top
        v = advection.vy
        phi_face = quick_reconstruction_2d(phi, mesh, i, j, direction)
        return v * phi_face
    end
end
