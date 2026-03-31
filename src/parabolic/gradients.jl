# Gradient reconstruction methods for higher-order schemes
# Migrated from Simu.jl SimuFVM/gradients.jl

function reconstruct_gradient_green_gauss(mesh::Mesh1D, phi, i)
    nx = length(mesh.cells)
    return if i == 1
        if nx >= 3
            dx = mesh.cells[2].center - mesh.cells[1].center
            if nx >= 3 && (mesh.cells[3].center - mesh.cells[2].center) ≈ dx
                return (-3.0 * phi[1] + 4.0 * phi[2] - phi[3]) / (2.0 * dx)
            else
                return (phi[2] - phi[1]) / dx
            end
        else
            dx = mesh.cells[2].center - mesh.cells[1].center
            return (phi[2] - phi[1]) / dx
        end
    elseif i == nx
        if nx >= 3
            dx = mesh.cells[end].center - mesh.cells[end - 1].center
            if nx >= 3 && (mesh.cells[end - 1].center - mesh.cells[end - 2].center) ≈ dx
                return (3.0 * phi[end] - 4.0 * phi[end - 1] + phi[end - 2]) / (2.0 * dx)
            else
                return (phi[end] - phi[end - 1]) / dx
            end
        else
            dx = mesh.cells[end].center - mesh.cells[end - 1].center
            return (phi[end] - phi[end - 1]) / dx
        end
    else
        dx_left = mesh.cells[i].center - mesh.cells[i - 1].center
        dx_right = mesh.cells[i + 1].center - mesh.cells[i].center
        if abs(dx_left - dx_right) < 1.0e-10
            return (phi[i + 1] - phi[i - 1]) / (dx_left + dx_right)
        else
            grad_left = (phi[i] - phi[i - 1]) / dx_left
            grad_right = (phi[i + 1] - phi[i]) / dx_right
            w_left = 1.0 / dx_left; w_right = 1.0 / dx_right
            return (w_left * grad_left + w_right * grad_right) / (w_left + w_right)
        end
    end
end

"""Reconstruct the gradient of `phi` at cell `i` using weighted least squares (1D)."""
function reconstruct_gradient_least_squares_1d(mesh::Mesh1D, phi, i)
    nx = length(mesh.cells)
    neighbors = Int[]; distances = Float64[]
    if i > 1
        push!(neighbors, i - 1); push!(distances, mesh.cells[i].center - mesh.cells[i - 1].center)
    end
    if i < nx
        push!(neighbors, i + 1); push!(distances, mesh.cells[i + 1].center - mesh.cells[i].center)
    end
    if length(neighbors) == 0
        return 0.0
    elseif length(neighbors) == 1
        return (phi[neighbors[1]] - phi[i]) / distances[1]
    else
        grad_L = (phi[i] - phi[neighbors[1]]) / distances[1]
        grad_R = (phi[neighbors[2]] - phi[i]) / distances[2]
        w_L = 1.0 / distances[1]; w_R = 1.0 / distances[2]
        return (w_L * grad_L + w_R * grad_R) / (w_L + w_R)
    end
end

"""Reconstruct the gradient of `phi` at cell `(i, j)` using the Green-Gauss theorem (2D structured)."""
function reconstruct_gradient_green_gauss_2d(mesh::Mesh2D, phi, i, j)
    k = (i - 1) * mesh.ny + j
    dx = get_cell_dx(mesh, i, j); dy = get_cell_dy(mesh, i, j)
    grad_x = 0.0; grad_y = 0.0
    if i == 1
        k_right = i * mesh.ny + j; grad_x = (phi[k_right] - phi[k]) / dx
    elseif i == mesh.nx
        k_left = (i - 2) * mesh.ny + j; grad_x = (phi[k] - phi[k_left]) / dx
    else
        k_left = (i - 2) * mesh.ny + j; k_right = i * mesh.ny + j
        dx_center = mesh.cells[k_right].center[1] - mesh.cells[k_left].center[1]
        grad_x = (phi[k_right] - phi[k_left]) / dx_center
    end
    if j == 1
        k_top = (i - 1) * mesh.ny + (j + 1); grad_y = (phi[k_top] - phi[k]) / dy
    elseif j == mesh.ny
        k_bottom = (i - 1) * mesh.ny + (j - 1); grad_y = (phi[k] - phi[k_bottom]) / dy
    else
        k_bottom = (i - 1) * mesh.ny + (j - 1); k_top = (i - 1) * mesh.ny + (j + 1)
        dy_center = mesh.cells[k_top].center[2] - mesh.cells[k_bottom].center[2]
        grad_y = (phi[k_top] - phi[k_bottom]) / dy_center
    end
    return (grad_x, grad_y)
end

"""Reconstruct the gradient of `phi` at cell `(i, j)` using weighted least squares (2D structured)."""
function reconstruct_gradient_least_squares_2d(mesh::Mesh2D, phi, i, j; weighted = true)
    k = (i - 1) * mesh.ny + j
    cell = mesh.cells[k]; x0, y0 = cell.center
    neighbors = Tuple{Int, Float64, Float64, Float64}[]
    for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ni, nj = i + di, j + dj
        (ni < 1 || ni > mesh.nx || nj < 1 || nj > mesh.ny) && continue
        nk = (ni - 1) * mesh.ny + nj; nc = mesh.cells[nk]
        ddx = nc.center[1] - x0; ddy = nc.center[2] - y0
        dist = sqrt(ddx^2 + ddy^2)
        w = weighted ? 1.0 / (dist^2 + 1.0e-12) : 1.0
        push!(neighbors, (nk, ddx, ddy, w))
    end
    length(neighbors) < 2 && return length(neighbors) == 1 ? let (idx, ddx, ddy, _) = neighbors[1]
            dist = sqrt(ddx^2 + ddy^2); dphi = phi[idx] - phi[k]; (dphi * ddx / dist^2, dphi * ddy / dist^2)
    end : (0.0, 0.0)
    ATA_xx = 0.0; ATA_xy = 0.0; ATA_yy = 0.0; ATb_x = 0.0; ATb_y = 0.0
    for (idx, ddx, ddy, w) in neighbors
        dphi = phi[idx] - phi[k]
        ATA_xx += w * ddx * ddx; ATA_xy += w * ddx * ddy; ATA_yy += w * ddy * ddy
        ATb_x += w * ddx * dphi; ATb_y += w * ddy * dphi
    end
    det = ATA_xx * ATA_yy - ATA_xy * ATA_xy
    abs(det) < 1.0e-12 && return let (idx, ddx, ddy, _) = neighbors[1]
        dist = sqrt(ddx^2 + ddy^2); dphi = phi[idx] - phi[k]; (dphi * ddx / dist^2, dphi * ddy / dist^2)
    end
    return ((ATA_yy * ATb_x - ATA_xy * ATb_y) / det, (ATA_xx * ATb_y - ATA_xy * ATb_x) / det)
end

function reconstruct_gradient_green_gauss_2d(mesh::UnstructuredMesh2D, phi, cell_idx; bcs = nothing)
    cell = mesh.cells[cell_idx]
    grad_x = 0.0; grad_y = 0.0
    for f_idx in cell.faces
        face = mesh.faces[f_idx]; phi_f = 0.0
        if face.neighbor > 0
            c_owner = mesh.cells[face.owner].center; c_neighbor = mesh.cells[face.neighbor].center; c_face = face.center
            d_owner = norm(c_face .- c_owner); d_neighbor = norm(c_face .- c_neighbor)
            w = d_owner / (d_owner + d_neighbor)
            phi_f = (1.0 - w) * phi[face.owner] + w * phi[face.neighbor]
        else
            if bcs !== nothing && haskey(bcs, f_idx)
                bc = bcs[f_idx]
                phi_f = bc isa ParabolicDirichlet ? bc.value : phi[face.owner]
            else
                phi_f = phi[face.owner]
            end
        end
        nx = face.normal[1]; ny = face.normal[2]
        if face.neighbor == cell_idx
            nx = -nx; ny = -ny
        end
        grad_x += phi_f * nx * face.area; grad_y += phi_f * ny * face.area
    end
    grad_x /= cell.volume; grad_y /= cell.volume
    return (grad_x, grad_y)
end

"""Reconstruct the gradient of `phi` at `cell_idx` using the Green-Gauss theorem (3D unstructured)."""
function reconstruct_gradient_green_gauss_3d(mesh::UnstructuredMesh3D, phi, cell_idx; bcs = nothing)
    cell = mesh.cells[cell_idx]
    grad_x = 0.0; grad_y = 0.0; grad_z = 0.0
    for f_idx in cell.faces
        face = mesh.faces[f_idx]; phi_f = 0.0
        if face.neighbor > 0
            c_owner = mesh.cells[face.owner].center; c_neighbor = mesh.cells[face.neighbor].center; c_face = face.center
            d_owner = norm(c_face .- c_owner); d_neighbor = norm(c_face .- c_neighbor)
            w = d_owner / (d_owner + d_neighbor)
            phi_f = (1.0 - w) * phi[face.owner] + w * phi[face.neighbor]
        else
            if bcs !== nothing && haskey(bcs, f_idx)
                bc = bcs[f_idx]
                phi_f = bc isa ParabolicDirichlet ? bc.value : phi[face.owner]
            else
                phi_f = phi[face.owner]
            end
        end
        nx, ny, nz = face.normal
        if face.neighbor == cell_idx
            nx, ny, nz = -nx, -ny, -nz
        end
        grad_x += phi_f * nx * face.area; grad_y += phi_f * ny * face.area; grad_z += phi_f * nz * face.area
    end
    grad_x /= cell.volume; grad_y /= cell.volume; grad_z /= cell.volume
    return (grad_x, grad_y, grad_z)
end

function reconstruct_gradient_at_boundary_2d(mesh::Mesh2D, phi, i, j, boundary_side::Symbol)
    k = (i - 1) * mesh.ny + j; grad_x = 0.0; grad_y = 0.0
    if boundary_side == :left
        if i < mesh.nx
            k_right = i * mesh.ny + j; grad_x = (phi[k_right] - phi[k]) / (mesh.cells[k_right].center[1] - mesh.cells[k].center[1])
        end
    elseif boundary_side == :right
        if i > 1
            k_left = (i - 2) * mesh.ny + j; grad_x = (phi[k] - phi[k_left]) / (mesh.cells[k].center[1] - mesh.cells[k_left].center[1])
        end
    elseif boundary_side == :bottom
        if j < mesh.ny
            k_top = (i - 1) * mesh.ny + (j + 1); grad_y = (phi[k_top] - phi[k]) / (mesh.cells[k_top].center[2] - mesh.cells[k].center[2])
        end
    else # :top
        if j > 1
            k_bottom = (i - 1) * mesh.ny + (j - 1); grad_y = (phi[k] - phi[k_bottom]) / (mesh.cells[k].center[2] - mesh.cells[k_bottom].center[2])
        end
    end
    # Cross-direction gradient
    if boundary_side == :left || boundary_side == :right
        if j > 1 && j < mesh.ny
            k_bottom = (i - 1) * mesh.ny + (j - 1); k_top = (i - 1) * mesh.ny + (j + 1)
            grad_y = (phi[k_top] - phi[k_bottom]) / (mesh.cells[k_top].center[2] - mesh.cells[k_bottom].center[2])
        elseif j > 1
            k_bottom = (i - 1) * mesh.ny + (j - 1); grad_y = (phi[k] - phi[k_bottom]) / (mesh.cells[k].center[2] - mesh.cells[k_bottom].center[2])
        elseif j < mesh.ny
            k_top = (i - 1) * mesh.ny + (j + 1); grad_y = (phi[k_top] - phi[k]) / (mesh.cells[k_top].center[2] - mesh.cells[k].center[2])
        end
    else
        if i > 1 && i < mesh.nx
            k_left = (i - 2) * mesh.ny + j; k_right = i * mesh.ny + j
            grad_x = (phi[k_right] - phi[k_left]) / (mesh.cells[k_right].center[1] - mesh.cells[k_left].center[1])
        elseif i > 1
            k_left = (i - 2) * mesh.ny + j; grad_x = (phi[k] - phi[k_left]) / (mesh.cells[k].center[1] - mesh.cells[k_left].center[1])
        elseif i < mesh.nx
            k_right = i * mesh.ny + j; grad_x = (phi[k_right] - phi[k]) / (mesh.cells[k_right].center[1] - mesh.cells[k].center[1])
        end
    end
    return (grad_x, grad_y)
end
