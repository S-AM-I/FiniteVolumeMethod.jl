# Helper functions for variable coefficients and utilities
# Migrated from Simu.jl SimuFVM/utils.jl

"""
    get_diffusion_coefficient(diffusion, mesh, i)

Get diffusion coefficient at cell i for 1D mesh.
"""
function get_diffusion_coefficient(diffusion::Diffusion1D, mesh::Mesh1D, i)
    return diffusion.gamma
end

"""
    get_diffusion_coefficient(diffusion, mesh, i)

Get diffusion coefficient for VariableDiffusion1D.
Evaluates the function at the cell center or returns the array value.
"""
function get_diffusion_coefficient(diffusion::VariableDiffusion1D, mesh::Mesh1D, i)
    if diffusion.gamma isa Function
        return diffusion.gamma(mesh.cells[i].center)
    else
        return diffusion.gamma[i]
    end
end

"""
    get_diffusion_coefficient_at_face(diffusion, mesh, i, side)

Get diffusion coefficient at face between cells for 1D mesh.
Uses harmonic mean for interface between cells with different coefficients.
"""
function get_diffusion_coefficient_at_face(diffusion::Diffusion1D, mesh::Mesh1D, i, side)
    return diffusion.gamma
end

"""
    get_diffusion_coefficient_at_face(diffusion, mesh, i, side)

Get diffusion coefficient at face for VariableDiffusion1D.
Uses harmonic mean of cell coefficients at internal faces.
"""
function get_diffusion_coefficient_at_face(diffusion::VariableDiffusion1D, mesh::Mesh1D, i, side)
    return if side == :left
        if i == 1
            return get_diffusion_coefficient(diffusion, mesh, i)
        else
            gamma_L = get_diffusion_coefficient(diffusion, mesh, i - 1)
            gamma_R = get_diffusion_coefficient(diffusion, mesh, i)
            if iszero(gamma_L) || iszero(gamma_R)
                return zero(gamma_L)
            end
            return 2 * gamma_L * gamma_R / (gamma_L + gamma_R)
        end
    else # side == :right
        if i == length(mesh.cells)
            return get_diffusion_coefficient(diffusion, mesh, i)
        else
            gamma_L = get_diffusion_coefficient(diffusion, mesh, i)
            gamma_R = get_diffusion_coefficient(diffusion, mesh, i + 1)
            if iszero(gamma_L) || iszero(gamma_R)
                return zero(gamma_L)
            end
            return 2 * gamma_L * gamma_R / (gamma_L + gamma_R)
        end
    end
end

"""
    get_diffusion_coefficient(diffusion, mesh, i, j)

Get diffusion coefficient at cell (i,j) for 2D mesh.
"""
function get_diffusion_coefficient(diffusion::Diffusion2D, mesh::Union{Mesh2D, CurvilinearMesh2D}, i, j)
    return diffusion.gamma
end

function get_diffusion_coefficient(diffusion::VariableDiffusion2D, mesh::Mesh2D, i, j)
    if diffusion.gamma isa Function
        cell = mesh.cells[(i - 1) * mesh.ny + j]
        return diffusion.gamma(cell.center[1], cell.center[2])
    else
        return diffusion.gamma[i, j]
    end
end

function get_diffusion_coefficient(diffusion::VariableDiffusion2D, mesh::CurvilinearMesh2D, i, j)
    if diffusion.gamma isa Function
        cell_center = get_cell_center(mesh, i, j)
        return diffusion.gamma(cell_center[1], cell_center[2])
    else
        return diffusion.gamma[i, j]
    end
end

"""
    get_velocity(advection, mesh, i, side)

Get velocity at face between cells for 1D mesh.
"""
function get_velocity(advection::Advection1D, mesh::Mesh1D, i, side)
    return advection.v
end

function get_velocity(advection::VariableAdvection1D, mesh::Mesh1D, i, side)
    if advection.v isa Function
        if side == :left
            if i == 1
                return advection.v(mesh.cells[i].center)
            else
                x_face = (mesh.cells[i - 1].center + mesh.cells[i].center) / 2.0
                return advection.v(x_face)
            end
        else # side == :right
            if i == length(mesh.cells)
                return advection.v(mesh.cells[i].center)
            else
                x_face = (mesh.cells[i].center + mesh.cells[i + 1].center) / 2.0
                return advection.v(x_face)
            end
        end
    else
        return advection.v[i]
    end
end

"""
    get_velocity(advection, mesh, i, j, direction)

Get velocity component at face for 2D mesh.
"""
function get_velocity(advection::Advection2D, mesh::Mesh2D, i, j, direction)
    if direction == :x
        return advection.vx
    else
        return advection.vy
    end
end

function get_velocity(advection::VariableAdvection2D, mesh::Mesh2D, i, j, direction)
    return if direction == :x
        if advection.vx isa Function
            cell = mesh.cells[(i - 1) * mesh.ny + j]
            return advection.vx(cell.center[1], cell.center[2])
        else
            return advection.vx[i, j]
        end
    else
        if advection.vy isa Function
            cell = mesh.cells[(i - 1) * mesh.ny + j]
            return advection.vy(cell.center[1], cell.center[2])
        else
            return advection.vy[i, j]
        end
    end
end

"""
    get_diffusion_coefficient(diffusion, mesh, i, j, k)

Get diffusion coefficient at cell (i,j,k) for 3D mesh.
"""
function get_diffusion_coefficient(diffusion::Diffusion3D, mesh::Mesh3D, i, j, k)
    return diffusion.gamma
end

function get_diffusion_coefficient(diffusion::AnisotropicDiffusion1D, mesh::Mesh1D, i)
    return diffusion.D
end

function get_diffusion_coefficient(diffusion::AnisotropicDiffusion2D, mesh::Mesh2D, i, j)
    return (diffusion.D[1, 1] + diffusion.D[2, 2]) / 2.0
end

"""
    get_anisotropic_flux_2d(diffusion, mesh, phi, i, j, direction, grad_x, grad_y)

Compute anisotropic diffusion flux in 2D using tensor-vector product.
"""
function get_anisotropic_flux_2d(diffusion::AnisotropicDiffusion2D, mesh::Mesh2D, phi, i, j, direction::Symbol, grad_x::Float64, grad_y::Float64)
    D = diffusion.D
    if direction == :x
        return -(D[1, 1] * grad_x + D[1, 2] * grad_y)
    else # direction == :y
        return -(D[2, 1] * grad_x + D[2, 2] * grad_y)
    end
end

"""
    get_anisotropic_flux_3d(diffusion, mesh, phi, i, j, k, direction, grad_x, grad_y, grad_z)

Compute anisotropic diffusion flux in 3D using tensor-vector product.
"""
function get_anisotropic_flux_3d(diffusion::AnisotropicDiffusion3D, mesh::Mesh3D, phi, i, j, k, direction::Symbol, grad_x::Float64, grad_y::Float64, grad_z::Float64)
    if diffusion.D isa Array{Float64, 5} && size(diffusion.D, 1) == 3 && size(diffusion.D, 2) == 3
        D = diffusion.D[:, :, i, j, k]
    elseif diffusion.D isa Array{Float64, 2}
        D = diffusion.D
    else
        D = reshape(diffusion.D[1:9], 3, 3)
    end

    if direction == :x
        return -(D[1, 1] * grad_x + D[1, 2] * grad_y + D[1, 3] * grad_z)
    elseif direction == :y
        return -(D[2, 1] * grad_x + D[2, 2] * grad_y + D[2, 3] * grad_z)
    else # direction == :z
        return -(D[3, 1] * grad_x + D[3, 2] * grad_y + D[3, 3] * grad_z)
    end
end

function get_diffusion_coefficient(diffusion::AnisotropicDiffusion3D, mesh::Mesh3D, i, j, k)
    if diffusion.D isa Array{Float64, 5} && size(diffusion.D, 1) == 3 && size(diffusion.D, 2) == 3
        return (diffusion.D[1, 1, i, j, k] + diffusion.D[2, 2, i, j, k] + diffusion.D[3, 3, i, j, k]) / 3.0
    elseif diffusion.D isa Array{Float64, 2}
        return (diffusion.D[1, 1] + diffusion.D[2, 2] + diffusion.D[3, 3]) / 3.0
    else
        return 1.0
    end
end

function get_diffusion_coefficient(diffusion::VariableDiffusion3D, mesh::Mesh3D, i, j, k)
    if diffusion.gamma isa Function
        idx = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
        cell = mesh.cells[idx]
        return diffusion.gamma(cell.center[1], cell.center[2], cell.center[3])
    else
        return diffusion.gamma[i, j, k]
    end
end

"""
    get_velocity(advection, mesh, i, j, k, direction)

Get velocity component at face for 3D mesh.
"""
function get_velocity(advection::Advection3D, mesh::Mesh3D, i, j, k, direction)
    if direction == :x
        return advection.vx
    elseif direction == :y
        return advection.vy
    else
        return advection.vz
    end
end

function get_velocity(advection::VariableAdvection3D, mesh::Mesh3D, i, j, k, direction)
    return if direction == :x
        if advection.vx isa Function
            idx = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
            cell = mesh.cells[idx]
            return advection.vx(cell.center[1], cell.center[2], cell.center[3])
        else
            return advection.vx[i, j, k]
        end
    elseif direction == :y
        if advection.vy isa Function
            idx = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
            cell = mesh.cells[idx]
            return advection.vy(cell.center[1], cell.center[2], cell.center[3])
        else
            return advection.vy[i, j, k]
        end
    else # direction == :z
        if advection.vz isa Function
            idx = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
            cell = mesh.cells[idx]
            return advection.vz(cell.center[1], cell.center[2], cell.center[3])
        else
            return advection.vz[i, j, k]
        end
    end
end

# --- Helper functions for 2D mesh cell dimensions ---

function get_cell_dx(mesh::Mesh2D, i, j)
    k = (i - 1) * mesh.ny + j
    cell = mesh.cells[k]
    if length(cell.nodes) >= 2
        x_coords = [n.x for n in cell.nodes]
        return maximum(x_coords) - minimum(x_coords)
    else
        return mesh.Lx / mesh.nx
    end
end

function get_cell_dy(mesh::Mesh2D, i, j)
    k = (i - 1) * mesh.ny + j
    cell = mesh.cells[k]
    if length(cell.nodes) >= 2
        y_coords = [n.y for n in cell.nodes]
        return maximum(y_coords) - minimum(y_coords)
    else
        return mesh.Ly / mesh.ny
    end
end

function get_face_dx(mesh::Mesh2D, i, j, side)
    k = (i - 1) * mesh.ny + j
    cell = mesh.cells[k]
    return if side == :left
        if i == 1
            return cell.center[1] - 0.0
        else
            k_left = (i - 2) * mesh.ny + j
            cell_left = mesh.cells[k_left]
            return cell.center[1] - cell_left.center[1]
        end
    else # side == :right
        if i == mesh.nx
            return mesh.Lx - cell.center[1]
        else
            k_right = i * mesh.ny + j
            cell_right = mesh.cells[k_right]
            return cell_right.center[1] - cell.center[1]
        end
    end
end

function get_face_dy(mesh::Mesh2D, i, j, side)
    k = (i - 1) * mesh.ny + j
    cell = mesh.cells[k]
    return if side == :bottom
        if j == 1
            return cell.center[2] - 0.0
        else
            k_bottom = (i - 1) * mesh.ny + (j - 1)
            cell_bottom = mesh.cells[k_bottom]
            return cell.center[2] - cell_bottom.center[2]
        end
    else # side == :top
        if j == mesh.ny
            return mesh.Ly - cell.center[2]
        else
            k_top = (i - 1) * mesh.ny + (j + 1)
            cell_top = mesh.cells[k_top]
            return cell_top.center[2] - cell.center[2]
        end
    end
end

# --- Helper functions for 3D mesh cell dimensions ---

function get_cell_dx(mesh::Mesh3D, i, j, k)
    idx = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
    cell = mesh.cells[idx]
    if length(cell.nodes) >= 2
        x_coords = [n.x for n in cell.nodes]
        return maximum(x_coords) - minimum(x_coords)
    else
        return mesh.Lx / mesh.nx
    end
end

function get_cell_dy(mesh::Mesh3D, i, j, k)
    idx = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
    cell = mesh.cells[idx]
    if length(cell.nodes) >= 2
        y_coords = [n.y for n in cell.nodes]
        return maximum(y_coords) - minimum(y_coords)
    else
        return mesh.Ly / mesh.ny
    end
end

function get_cell_dz(mesh::Mesh3D, i, j, k)
    idx = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
    cell = mesh.cells[idx]
    if length(cell.nodes) >= 2
        z_coords = [n.z for n in cell.nodes]
        return maximum(z_coords) - minimum(z_coords)
    else
        return mesh.Lz / mesh.nz
    end
end

function get_face_dx(mesh::Mesh3D, i, j, k, side)
    idx = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
    cell = mesh.cells[idx]
    return if side == :left
        if i == 1
            return cell.center[1] - 0.0
        else
            idx_left = (i - 2) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
            cell_left = mesh.cells[idx_left]
            return cell.center[1] - cell_left.center[1]
        end
    else # side == :right
        if i == mesh.nx
            return mesh.Lx - cell.center[1]
        else
            idx_right = i * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
            cell_right = mesh.cells[idx_right]
            return cell_right.center[1] - cell.center[1]
        end
    end
end

function get_face_dy(mesh::Mesh3D, i, j, k, side)
    idx = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
    cell = mesh.cells[idx]
    return if side == :bottom
        if j == 1
            return cell.center[2] - 0.0
        else
            idx_bottom = (i - 1) * mesh.ny * mesh.nz + (j - 2) * mesh.nz + k
            cell_bottom = mesh.cells[idx_bottom]
            return cell.center[2] - cell_bottom.center[2]
        end
    else # side == :top
        if j == mesh.ny
            return mesh.Ly - cell.center[2]
        else
            idx_top = (i - 1) * mesh.ny * mesh.nz + j * mesh.nz + k
            cell_top = mesh.cells[idx_top]
            return cell_top.center[2] - cell.center[2]
        end
    end
end

function get_face_dz(mesh::Mesh3D, i, j, k, side)
    idx = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + k
    cell = mesh.cells[idx]
    return if side == :front
        if k == 1
            return cell.center[3] - 0.0
        else
            idx_front = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + (k - 1)
            cell_front = mesh.cells[idx_front]
            return cell.center[3] - cell_front.center[3]
        end
    else # side == :back
        if k == mesh.nz
            return mesh.Lz - cell.center[3]
        else
            idx_back = (i - 1) * mesh.ny * mesh.nz + (j - 1) * mesh.nz + (k + 1)
            cell_back = mesh.cells[idx_back]
            return cell_back.center[3] - cell.center[3]
        end
    end
end

"""
    get_diffusion_coefficient_at_face_3d(diffusion, mesh, i, j, k, side)

Get diffusion coefficient at face for 3D mesh.
"""
function get_diffusion_coefficient_at_face_3d(diffusion::Diffusion3D, mesh::Mesh3D, i, j, k, side)
    return diffusion.gamma
end

function get_diffusion_coefficient_at_face_3d(diffusion::VariableDiffusion3D, mesh::Mesh3D, i, j, k, side)
    gamma_P = get_diffusion_coefficient(diffusion, mesh, i, j, k)
    if side == :left
        i == 1 && return gamma_P
        gamma_W = get_diffusion_coefficient(diffusion, mesh, i - 1, j, k)
        (iszero(gamma_P) || iszero(gamma_W)) && return zero(gamma_P)
        return 2 * gamma_P * gamma_W / (gamma_P + gamma_W)
    elseif side == :right
        i == mesh.nx && return gamma_P
        gamma_E = get_diffusion_coefficient(diffusion, mesh, i + 1, j, k)
        (iszero(gamma_P) || iszero(gamma_E)) && return zero(gamma_P)
        return 2 * gamma_P * gamma_E / (gamma_P + gamma_E)
    elseif side == :bottom
        j == 1 && return gamma_P
        gamma_S = get_diffusion_coefficient(diffusion, mesh, i, j - 1, k)
        (iszero(gamma_P) || iszero(gamma_S)) && return zero(gamma_P)
        return 2 * gamma_P * gamma_S / (gamma_P + gamma_S)
    elseif side == :top
        j == mesh.ny && return gamma_P
        gamma_N = get_diffusion_coefficient(diffusion, mesh, i, j + 1, k)
        (iszero(gamma_P) || iszero(gamma_N)) && return zero(gamma_P)
        return 2 * gamma_P * gamma_N / (gamma_P + gamma_N)
    elseif side == :front
        k == 1 && return gamma_P
        gamma_F = get_diffusion_coefficient(diffusion, mesh, i, j, k - 1)
        (iszero(gamma_P) || iszero(gamma_F)) && return zero(gamma_P)
        return 2 * gamma_P * gamma_F / (gamma_P + gamma_F)
    else # side == :back
        k == mesh.nz && return gamma_P
        gamma_B = get_diffusion_coefficient(diffusion, mesh, i, j, k + 1)
        (iszero(gamma_P) || iszero(gamma_B)) && return zero(gamma_P)
        return 2 * gamma_P * gamma_B / (gamma_P + gamma_B)
    end
end

# --- CFL and stable dt calculations ---

function calculate_cfl(mesh::Mesh1D, velocity::Float64, dt::Float64)
    dx_min = minimum([mesh.cells[i + 1].center - mesh.cells[i].center for i in 1:(length(mesh.cells) - 1)])
    return abs(velocity) * dt / dx_min
end

function calculate_cfl(mesh::Mesh2D, vx::Float64, vy::Float64, dt::Float64)
    dx_min = mesh.Lx / mesh.nx
    dy_min = mesh.Ly / mesh.ny
    cfl_x = abs(vx) * dt / dx_min
    cfl_y = abs(vy) * dt / dy_min
    return max(cfl_x, cfl_y)
end

function calculate_cfl(mesh::Mesh3D, vx::Float64, vy::Float64, vz::Float64, dt::Float64)
    dx_min = mesh.Lx / mesh.nx
    dy_min = mesh.Ly / mesh.ny
    dz_min = mesh.Lz / mesh.nz
    return max(abs(vx) * dt / dx_min, abs(vy) * dt / dy_min, abs(vz) * dt / dz_min)
end

function assemble_mass_matrix(mesh::Mesh1D)
    nx = length(mesh.cells)
    M = SparseArrays.spzeros(nx, nx)
    for i in 1:nx
        M[i, i] = mesh.cells[i].volume
    end
    return M
end

function assemble_mass_matrix(mesh::Mesh2D)
    nx = mesh.nx
    ny = mesh.ny
    M = SparseArrays.spzeros(nx * ny, nx * ny)
    for i in 1:nx
        for j in 1:ny
            k = (i - 1) * ny + j
            M[k, k] = mesh.cells[k].volume
        end
    end
    return M
end

function assemble_mass_matrix(mesh::Mesh3D)
    nx = mesh.nx
    ny = mesh.ny
    nz = mesh.nz
    M = SparseArrays.spzeros(nx * ny * nz, nx * ny * nz)
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz
                idx = (i - 1) * ny * nz + (j - 1) * nz + k
                M[idx, idx] = mesh.cells[idx].volume
            end
        end
    end
    return M
end

function calculate_stable_dt_diffusion(mesh::Mesh1D, gamma::Float64; cfl_target = 0.5)
    dx_min = minimum([mesh.cells[i + 1].center - mesh.cells[i].center for i in 1:(length(mesh.cells) - 1)])
    return cfl_target * dx_min^2 / (2 * gamma)
end

function calculate_stable_dt_diffusion(mesh::Mesh2D, gamma::Float64; cfl_target = 0.5)
    dx_min = mesh.Lx / mesh.nx
    dy_min = mesh.Ly / mesh.ny
    return cfl_target * min(dx_min^2, dy_min^2) / (2 * gamma)
end

function calculate_stable_dt_diffusion(mesh::Mesh3D, gamma::Float64; cfl_target = 0.5)
    dx_min = mesh.Lx / mesh.nx
    dy_min = mesh.Ly / mesh.ny
    dz_min = mesh.Lz / mesh.nz
    return cfl_target * min(dx_min^2, dy_min^2, dz_min^2) / (2 * gamma)
end

function calculate_stable_dt_advection(mesh::Mesh1D, velocity::Float64; cfl_target = 0.5)
    dx_min = minimum([mesh.cells[i + 1].center - mesh.cells[i].center for i in 1:(length(mesh.cells) - 1)])
    return cfl_target * dx_min / abs(velocity)
end

function calculate_stable_dt_advection(mesh::Mesh2D, vx::Float64, vy::Float64; cfl_target = 0.5)
    dx_min = mesh.Lx / mesh.nx
    dy_min = mesh.Ly / mesh.ny
    return min(cfl_target * dx_min / abs(vx), cfl_target * dy_min / abs(vy))
end

function calculate_stable_dt_advection(mesh::Mesh3D, vx::Float64, vy::Float64, vz::Float64; cfl_target = 0.5)
    dx_min = mesh.Lx / mesh.nx
    dy_min = mesh.Ly / mesh.ny
    dz_min = mesh.Lz / mesh.nz
    return min(cfl_target * dx_min / abs(vx), cfl_target * dy_min / abs(vy), cfl_target * dz_min / abs(vz))
end

function calculate_stable_dt_advection_diffusion(mesh::Mesh1D, velocity::Float64, gamma::Float64; cfl_target = 0.5)
    return min(
        calculate_stable_dt_advection(mesh, velocity; cfl_target = cfl_target),
        calculate_stable_dt_diffusion(mesh, gamma; cfl_target = cfl_target)
    )
end

function calculate_stable_dt_advection_diffusion(mesh::Mesh2D, vx::Float64, vy::Float64, gamma::Float64; cfl_target = 0.5)
    return min(
        calculate_stable_dt_advection(mesh, vx, vy; cfl_target = cfl_target),
        calculate_stable_dt_diffusion(mesh, gamma; cfl_target = cfl_target)
    )
end

function calculate_stable_dt_advection_diffusion(mesh::Mesh3D, vx::Float64, vy::Float64, vz::Float64, gamma::Float64; cfl_target = 0.5)
    return min(
        calculate_stable_dt_advection(mesh, vx, vy, vz; cfl_target = cfl_target),
        calculate_stable_dt_diffusion(mesh, gamma; cfl_target = cfl_target)
    )
end

function recommend_time_step(mesh::Mesh1D, model::Diffusion1D; cfl_target = 0.5)
    return calculate_stable_dt_diffusion(mesh, model.gamma; cfl_target = cfl_target)
end

function recommend_time_step(mesh::Mesh1D, model::Advection1D; cfl_target = 0.5)
    return calculate_stable_dt_advection(mesh, model.v; cfl_target = cfl_target)
end

function recommend_time_step(mesh::Mesh1D, model::AdvectionDiffusion1D; cfl_target = 0.5)
    return calculate_stable_dt_advection_diffusion(mesh, model.advection.v, model.diffusion.gamma; cfl_target = cfl_target)
end

function recommend_time_step(mesh::Mesh2D, model::Diffusion2D; cfl_target = 0.5)
    return calculate_stable_dt_diffusion(mesh, model.gamma; cfl_target = cfl_target)
end

function recommend_time_step(mesh::Mesh2D, model::Advection2D; cfl_target = 0.5)
    return calculate_stable_dt_advection(mesh, model.vx, model.vy; cfl_target = cfl_target)
end

function recommend_time_step(mesh::Mesh2D, model::AdvectionDiffusion2D; cfl_target = 0.5)
    return calculate_stable_dt_advection_diffusion(mesh, model.advection.vx, model.advection.vy, model.diffusion.gamma; cfl_target = cfl_target)
end

function recommend_time_step(mesh::Mesh3D, model::Diffusion3D; cfl_target = 0.5)
    return calculate_stable_dt_diffusion(mesh, model.gamma; cfl_target = cfl_target)
end

function recommend_time_step(mesh::Mesh3D, model::Advection3D; cfl_target = 0.5)
    return calculate_stable_dt_advection(mesh, model.vx, model.vy, model.vz; cfl_target = cfl_target)
end

function recommend_time_step(mesh::Mesh3D, model::AdvectionDiffusion3D; cfl_target = 0.5)
    return calculate_stable_dt_advection_diffusion(mesh, model.advection.vx, model.advection.vy, model.advection.vz, model.diffusion.gamma; cfl_target = cfl_target)
end

# --- Time-Dependent Boundary Conditions ---

"""
    TimeDependentDirichlet(f)

Time-dependent Dirichlet boundary condition.  `f(t)` returns the prescribed
boundary value at time `t`.  Evaluated via [`evaluate_bc`](@ref) each time the
linear system is assembled.
"""
struct TimeDependentDirichlet <: AbstractBoundaryCondition
    f::Function  # f(t) -> Float64
end

"""
    TimeDependentNeumann(f)

Time-dependent Neumann boundary condition.  `f(t)` returns the prescribed
normal flux at time `t`.
"""
struct TimeDependentNeumann <: AbstractBoundaryCondition
    f::Function  # f(t) -> Float64
end

"""
    TimeDependentRobin(a, b, c)

Time-dependent Robin boundary condition.  The callable coefficients `a(t)`,
`b(t)`, `c(t)` define the relation `a(t) u + b(t) du/dn = c(t)` at each
time level.
"""
struct TimeDependentRobin <: AbstractBoundaryCondition
    a::Function  # a(t) -> Float64
    b::Function  # b(t) -> Float64
    c::Function  # c(t) -> Float64
end

function evaluate_bc(bc::TimeDependentDirichlet, t::Float64)
    return ParabolicDirichlet(bc.f(t))
end

function evaluate_bc(bc::TimeDependentNeumann, t::Float64)
    return ParabolicNeumann(bc.f(t))
end

function evaluate_bc(bc::TimeDependentRobin, t::Float64)
    return ParabolicRobin(bc.a(t), bc.b(t), bc.c(t))
end

# For backward compatibility, constant BCs evaluate to themselves
evaluate_bc(bc::ParabolicDirichlet, t::Float64) = bc
evaluate_bc(bc::ParabolicNeumann, t::Float64) = bc
evaluate_bc(bc::ParabolicRobin, t::Float64) = bc

"""
    add_entry!(I, J, V, i, j, val)

Helper to add entry to sparse matrix coordinate lists.
"""
function add_entry!(I, J, V, i::Int, j::Int, val::Float64)
    push!(I, i)
    push!(J, j)
    return push!(V, val)
end

"""
    apply_source_term!(A, b, source, mesh)

Apply source term to system matrix A and RHS vector b.
"""
function apply_source_term!(A, b, source, mesh)
    nx = length(mesh.cells)
    return if source isa ConstantSource
        for i in 1:nx
            b[i] += source.value * mesh.cells[i].volume
        end
    elseif source isa SpatialSource
        for i in 1:nx
            if source.values isa Vector
                b[i] += source.values[i] * mesh.cells[i].volume
            elseif source.values isa Matrix && mesh isa Mesh2D
                b[i] += source.values[i] * mesh.cells[i].volume
            end
        end
    elseif source isa FunctionSource
        for i in 1:nx
            center = mesh.cells[i].center
            val = 0.0
            if center isa Number
                val = source.f(center)
            elseif length(center) == 2
                val = source.f(center[1], center[2])
            elseif length(center) == 3
                val = source.f(center[1], center[2], center[3])
            end
            b[i] += val * mesh.cells[i].volume
        end
    elseif source isa LinearizedSource
        for i in 1:nx
            vol = mesh.cells[i].volume
            sc_val = source.sc isa Vector ? source.sc[i] : (source.sc isa Number ? source.sc : 0.0)
            b[i] += sc_val * vol
            sp_val = source.sp isa Vector ? source.sp[i] : (source.sp isa Number ? source.sp : 0.0)
            A[i, i] -= sp_val * vol
        end
    end
end

# --- Field Helpers ---

"""
    number_of_cells(mesh)

Generic helper to get cell count.
"""
number_of_cells(mesh::Mesh1D) = length(mesh.cells)
number_of_cells(mesh::Mesh2D) = mesh.nx * mesh.ny
number_of_cells(mesh::Mesh3D) = mesh.nx * mesh.ny * mesh.nz
number_of_cells(mesh) = hasproperty(mesh, :cells) ? length(mesh.cells) : error("Unknown mesh type for cell counting")
