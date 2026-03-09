# Assembly routines for 2D problems
# Migrated from Simu.jl SimuFVM/assembly/assembly_2d.jl

function assemble_system(advection::Union{Advection2D, VariableAdvection2D}, mesh::Mesh2D, bcs; transient = false, source = nothing)
    nx = mesh.nx; ny = mesh.ny; n_cells = nx * ny
    b = zeros(n_cells)
    max_tid = 1024
    I_chunks = [Int[] for _ in 1:max_tid]; J_chunks = [Int[] for _ in 1:max_tid]; V_chunks = [Float64[] for _ in 1:max_tid]
    bc_left, bc_right, bc_bottom, bc_top = bcs

    Threads.@threads :static for k in 1:n_cells
        tid = Threads.threadid()
        tid > max_tid && error("Thread ID $tid exceeds maximum supported $max_tid")
        I_local = I_chunks[tid]; J_local = J_chunks[tid]; V_local = V_chunks[tid]
        j = mod(k - 1, ny) + 1; i = div(k - 1, ny) + 1
        dx = get_cell_dx(mesh, i, j); dy = get_cell_dy(mesh, i, j)
        dx_left = get_face_dx(mesh, i, j, :left); dx_right = get_face_dx(mesh, i, j, :right)
        dy_bottom = get_face_dy(mesh, i, j, :bottom); dy_top = get_face_dy(mesh, i, j, :top)
        vx_w = get_velocity(advection, mesh, i, j, :x); vx_e = get_velocity(advection, mesh, i, j, :x)
        vy_s = get_velocity(advection, mesh, i, j, :y); vy_n = get_velocity(advection, mesh, i, j, :y)
        if advection isa VariableAdvection2D
            if i > 1
                vx_w = (get_velocity(advection, mesh, i - 1, j, :x) + get_velocity(advection, mesh, i, j, :x)) / 2.0
            end
            if i < nx
                vx_e = (get_velocity(advection, mesh, i, j, :x) + get_velocity(advection, mesh, i + 1, j, :x)) / 2.0
            end
            if j > 1
                vy_s = (get_velocity(advection, mesh, i, j - 1, :y) + get_velocity(advection, mesh, i, j, :y)) / 2.0
            end
            if j < ny
                vy_n = (get_velocity(advection, mesh, i, j, :y) + get_velocity(advection, mesh, i, j + 1, :y)) / 2.0
            end
        end
        if i == 1
            handle_advection_boundary_condition_2d!(I_local, J_local, V_local, b, advection, mesh, k, bc_left, :left, transient, dx_left, dy, vx_w)
        else
            k_w = k - ny; if vx_w >= 0
                add_entry!(I_local, J_local, V_local, k, k_w, -vx_w * dy)
            else
                add_entry!(I_local, J_local, V_local, k, k, abs(vx_w) * dy)
            end
        end
        if i == nx
            handle_advection_boundary_condition_2d!(I_local, J_local, V_local, b, advection, mesh, k, bc_right, :right, transient, dx_right, dy, vx_e)
        else
            k_e = k + ny; if vx_e >= 0
                add_entry!(I_local, J_local, V_local, k, k, vx_e * dy)
            else
                add_entry!(I_local, J_local, V_local, k, k_e, -abs(vx_e) * dy)
            end
        end
        if j == 1
            handle_advection_boundary_condition_2d!(I_local, J_local, V_local, b, advection, mesh, k, bc_bottom, :bottom, transient, dx, dy_bottom, vy_s)
        else
            k_s = k - 1; if vy_s >= 0
                add_entry!(I_local, J_local, V_local, k, k_s, -vy_s * dx)
            else
                add_entry!(I_local, J_local, V_local, k, k, abs(vy_s) * dx)
            end
        end
        if j == ny
            handle_advection_boundary_condition_2d!(I_local, J_local, V_local, b, advection, mesh, k, bc_top, :top, transient, dx, dy_top, vy_n)
        else
            k_n = k + 1; if vy_n >= 0
                add_entry!(I_local, J_local, V_local, k, k, vy_n * dx)
            else
                add_entry!(I_local, J_local, V_local, k, k_n, -abs(vy_n) * dx)
            end
        end
        if source !== nothing
            b[k] += evaluate_source(source, mesh, i, j) * mesh.cells[k].volume
        end
    end
    I = reduce(vcat, I_chunks); J = reduce(vcat, J_chunks); V = reduce(vcat, V_chunks)
    A = sparse(I, J, V, n_cells, n_cells)
    return A, b
end

function assemble_system(diffusion::Union{Diffusion2D, VariableDiffusion2D}, mesh::Mesh2D, bcs; transient = false, source = nothing)
    nx = mesh.nx; ny = mesh.ny; n_cells = nx * ny
    b = zeros(n_cells)
    max_tid = 1024
    I_chunks = [Int[] for _ in 1:max_tid]; J_chunks = [Int[] for _ in 1:max_tid]; V_chunks = [Float64[] for _ in 1:max_tid]
    bc_left, bc_right, bc_bottom, bc_top = bcs

    Threads.@threads :static for k in 1:n_cells
        tid = Threads.threadid()
        tid > max_tid && error("Thread ID $tid exceeds maximum supported $max_tid")
        I_local = I_chunks[tid]; J_local = J_chunks[tid]; V_local = V_chunks[tid]
        j = mod(k - 1, ny) + 1; i = div(k - 1, ny) + 1
        dx = get_cell_dx(mesh, i, j); dy = get_cell_dy(mesh, i, j)
        dx_left = get_face_dx(mesh, i, j, :left); dx_right = get_face_dx(mesh, i, j, :right)
        dy_bottom = get_face_dy(mesh, i, j, :bottom); dy_top = get_face_dy(mesh, i, j, :top)

        if i == 1
            handle_boundary_condition_2d!(I_local, J_local, V_local, b, diffusion, mesh, k, bc_left, :left, transient, dx_left, dy)
        else
            neighbor_k = k - ny; gamma_face = get_diffusion_coefficient_at_face_2d(diffusion, mesh, i, j, :left); flux_coeff = gamma_face * dy / dx_left; add_entry!(I_local, J_local, V_local, k, k, flux_coeff); add_entry!(I_local, J_local, V_local, k, neighbor_k, -flux_coeff)
        end
        if i == nx
            handle_boundary_condition_2d!(I_local, J_local, V_local, b, diffusion, mesh, k, bc_right, :right, transient, dx_right, dy)
        else
            neighbor_k = k + ny; gamma_face = get_diffusion_coefficient_at_face_2d(diffusion, mesh, i, j, :right); flux_coeff = gamma_face * dy / dx_right; add_entry!(I_local, J_local, V_local, k, k, flux_coeff); add_entry!(I_local, J_local, V_local, k, neighbor_k, -flux_coeff)
        end
        if j == 1
            handle_boundary_condition_2d!(I_local, J_local, V_local, b, diffusion, mesh, k, bc_bottom, :bottom, transient, dx, dy_bottom)
        else
            neighbor_k = k - 1; gamma_face = get_diffusion_coefficient_at_face_2d(diffusion, mesh, i, j, :bottom); flux_coeff = gamma_face * dx / dy_bottom; add_entry!(I_local, J_local, V_local, k, k, flux_coeff); add_entry!(I_local, J_local, V_local, k, neighbor_k, -flux_coeff)
        end
        if j == ny
            handle_boundary_condition_2d!(I_local, J_local, V_local, b, diffusion, mesh, k, bc_top, :top, transient, dx, dy_top)
        else
            neighbor_k = k + 1; gamma_face = get_diffusion_coefficient_at_face_2d(diffusion, mesh, i, j, :top); flux_coeff = gamma_face * dx / dy_top; add_entry!(I_local, J_local, V_local, k, k, flux_coeff); add_entry!(I_local, J_local, V_local, k, neighbor_k, -flux_coeff)
        end
        if source !== nothing
            b[k] += evaluate_source(source, mesh, i, j) * mesh.cells[k].volume
        end
    end
    I = reduce(vcat, I_chunks); J = reduce(vcat, J_chunks); V = reduce(vcat, V_chunks)
    A = sparse(I, J, V, n_cells, n_cells)
    return A, b
end

function assemble_system(diffusion::AnisotropicDiffusion2D, mesh::Mesh2D, bcs; transient = false, source = nothing)
    nx = mesh.nx; ny = mesh.ny; n_cells = nx * ny
    b = zeros(n_cells)
    max_tid = 1024
    I_chunks = [Int[] for _ in 1:max_tid]; J_chunks = [Int[] for _ in 1:max_tid]; V_chunks = [Float64[] for _ in 1:max_tid]
    bc_left, bc_right, bc_bottom, bc_top = bcs
    D = diffusion.D

    Threads.@threads :static for k in 1:n_cells
        tid = Threads.threadid()
        tid > max_tid && error("Thread ID $tid exceeds maximum supported $max_tid")
        I_local = I_chunks[tid]; J_local = J_chunks[tid]; V_local = V_chunks[tid]
        j = mod(k - 1, ny) + 1; i = div(k - 1, ny) + 1
        dx = get_cell_dx(mesh, i, j); dy = get_cell_dy(mesh, i, j)
        dx_left = get_face_dx(mesh, i, j, :left); dx_right = get_face_dx(mesh, i, j, :right)
        dy_bottom = get_face_dy(mesh, i, j, :bottom); dy_top = get_face_dy(mesh, i, j, :top)

        if i == 1
            handle_boundary_condition_2d!(I_local, J_local, V_local, b, diffusion, mesh, k, bc_left, :left, transient, dx_left, dy)
        else
            neighbor_k = k - ny
            flux_coeff_main = D[1, 1] * dy / dx_left
            add_entry!(I_local, J_local, V_local, k, k, flux_coeff_main); add_entry!(I_local, J_local, V_local, k, neighbor_k, -flux_coeff_main)
            if abs(D[1, 2]) > 1.0e-12 && j > 1 && j < ny
                cross_coeff = D[1, 2] * dy * 0.25 / dx_left
                add_entry!(I_local, J_local, V_local, k, k + 1, -cross_coeff); add_entry!(I_local, J_local, V_local, k, k - 1, cross_coeff)
                add_entry!(I_local, J_local, V_local, k, neighbor_k + 1, cross_coeff); add_entry!(I_local, J_local, V_local, k, neighbor_k - 1, -cross_coeff)
            end
        end
        if i == nx
            handle_boundary_condition_2d!(I_local, J_local, V_local, b, diffusion, mesh, k, bc_right, :right, transient, dx_right, dy)
        else
            neighbor_k = k + ny
            flux_coeff_main = D[1, 1] * dy / dx_right
            add_entry!(I_local, J_local, V_local, k, k, flux_coeff_main); add_entry!(I_local, J_local, V_local, k, neighbor_k, -flux_coeff_main)
            if abs(D[1, 2]) > 1.0e-12 && j > 1 && j < ny
                cross_coeff = D[1, 2] * dy * 0.25 / dx_right
                add_entry!(I_local, J_local, V_local, k, k + 1, -cross_coeff); add_entry!(I_local, J_local, V_local, k, k - 1, cross_coeff)
                add_entry!(I_local, J_local, V_local, k, neighbor_k + 1, cross_coeff); add_entry!(I_local, J_local, V_local, k, neighbor_k - 1, -cross_coeff)
            end
        end
        if j == 1
            handle_boundary_condition_2d!(I_local, J_local, V_local, b, diffusion, mesh, k, bc_bottom, :bottom, transient, dx, dy_bottom)
        else
            neighbor_k = k - 1
            flux_coeff_main = D[2, 2] * dx / dy_bottom
            add_entry!(I_local, J_local, V_local, k, k, flux_coeff_main); add_entry!(I_local, J_local, V_local, k, neighbor_k, -flux_coeff_main)
            if abs(D[2, 1]) > 1.0e-12 && i > 1 && i < nx
                cross_coeff = D[2, 1] * dx * 0.25 / dy_bottom
                add_entry!(I_local, J_local, V_local, k, k - ny, -cross_coeff); add_entry!(I_local, J_local, V_local, k, k + ny, cross_coeff)
                add_entry!(I_local, J_local, V_local, k, neighbor_k - ny, cross_coeff); add_entry!(I_local, J_local, V_local, k, neighbor_k + ny, -cross_coeff)
            end
        end
        if j == ny
            handle_boundary_condition_2d!(I_local, J_local, V_local, b, diffusion, mesh, k, bc_top, :top, transient, dx, dy_top)
        else
            neighbor_k = k + 1
            flux_coeff_main = D[2, 2] * dx / dy_top
            add_entry!(I_local, J_local, V_local, k, k, flux_coeff_main); add_entry!(I_local, J_local, V_local, k, neighbor_k, -flux_coeff_main)
            if abs(D[2, 1]) > 1.0e-12 && i > 1 && i < nx
                cross_coeff = D[2, 1] * dx * 0.25 / dy_top
                add_entry!(I_local, J_local, V_local, k, k - ny, -cross_coeff); add_entry!(I_local, J_local, V_local, k, k + ny, cross_coeff)
                add_entry!(I_local, J_local, V_local, k, neighbor_k - ny, cross_coeff); add_entry!(I_local, J_local, V_local, k, neighbor_k + ny, -cross_coeff)
            end
        end
        if source !== nothing
            b[k] += evaluate_source(source, mesh, i, j) * mesh.cells[k].volume
        end
    end
    I = reduce(vcat, I_chunks); J = reduce(vcat, J_chunks); V = reduce(vcat, V_chunks)
    A = sparse(I, J, V, n_cells, n_cells)
    return A, b
end

function assemble_system(model::Union{AdvectionDiffusion2D, VariableAdvectionDiffusion2D}, mesh::Mesh2D, bcs; transient = false, source = nothing)
    nx = mesh.nx; ny = mesh.ny; n_cells = nx * ny
    b = zeros(n_cells)
    max_tid = 1024
    I_chunks = [Int[] for _ in 1:max_tid]; J_chunks = [Int[] for _ in 1:max_tid]; V_chunks = [Float64[] for _ in 1:max_tid]
    bc_left, bc_right, bc_bottom, bc_top = bcs
    advection = model.advection; diffusion = model.diffusion

    Threads.@threads :static for k in 1:n_cells
        tid = Threads.threadid()
        tid > max_tid && error("Thread ID $tid exceeds maximum supported $max_tid")
        I_local = I_chunks[tid]; J_local = J_chunks[tid]; V_local = V_chunks[tid]
        j = mod(k - 1, ny) + 1; i = div(k - 1, ny) + 1
        dx = get_cell_dx(mesh, i, j); dy = get_cell_dy(mesh, i, j)
        dx_left = get_face_dx(mesh, i, j, :left); dx_right = get_face_dx(mesh, i, j, :right)
        dy_bottom = get_face_dy(mesh, i, j, :bottom); dy_top = get_face_dy(mesh, i, j, :top)
        vx_w, vx_e, vy_s, vy_n = 0.0, 0.0, 0.0, 0.0
        if advection isa VariableAdvection2D
            vx_w = i > 1 ? (get_velocity(advection, mesh, i - 1, j, :x) + get_velocity(advection, mesh, i, j, :x)) / 2.0 : get_velocity(advection, mesh, i, j, :x)
            vx_e = i < nx ? (get_velocity(advection, mesh, i, j, :x) + get_velocity(advection, mesh, i + 1, j, :x)) / 2.0 : get_velocity(advection, mesh, i, j, :x)
            vy_s = j > 1 ? (get_velocity(advection, mesh, i, j - 1, :y) + get_velocity(advection, mesh, i, j, :y)) / 2.0 : get_velocity(advection, mesh, i, j, :y)
            vy_n = j < ny ? (get_velocity(advection, mesh, i, j, :y) + get_velocity(advection, mesh, i, j + 1, :y)) / 2.0 : get_velocity(advection, mesh, i, j, :y)
        else
            vx_w = advection.vx; vx_e = advection.vx; vy_s = advection.vy; vy_n = advection.vy
        end
        F_w = vx_w * dy; F_e = vx_e * dy; F_s = vy_s * dx; F_n = vy_n * dx
        gamma_w = get_diffusion_coefficient_at_face_2d(diffusion, mesh, i, j, :left)
        gamma_e = get_diffusion_coefficient_at_face_2d(diffusion, mesh, i, j, :right)
        gamma_s = get_diffusion_coefficient_at_face_2d(diffusion, mesh, i, j, :bottom)
        gamma_n = get_diffusion_coefficient_at_face_2d(diffusion, mesh, i, j, :top)
        D_w = gamma_w * dy / dx_left; D_e = gamma_e * dy / dx_right
        D_s = gamma_s * dx / dy_bottom; D_n = gamma_n * dx / dy_top

        if i == 1
            handle_advection_diffusion_boundary_condition_2d!(I_local, J_local, V_local, b, model, mesh, k, bc_left, :left, transient, dx_left, dy, vx_w)
        else
            k_w = k - ny; a_W_nb = D_w + max(F_w, 0); a_P_nb = D_w + max(-F_w, 0); add_entry!(I_local, J_local, V_local, k, k, a_P_nb); add_entry!(I_local, J_local, V_local, k, k_w, -a_W_nb)
        end
        if i == nx
            handle_advection_diffusion_boundary_condition_2d!(I_local, J_local, V_local, b, model, mesh, k, bc_right, :right, transient, dx_right, dy, vx_e)
        else
            k_e = k + ny; a_E_nb = D_e + max(-F_e, 0); a_P_nb = D_e + max(F_e, 0); add_entry!(I_local, J_local, V_local, k, k, a_P_nb); add_entry!(I_local, J_local, V_local, k, k_e, -a_E_nb)
        end
        if j == 1
            handle_advection_diffusion_boundary_condition_2d!(I_local, J_local, V_local, b, model, mesh, k, bc_bottom, :bottom, transient, dx, dy_bottom, vy_s)
        else
            k_s = k - 1; a_S_nb = D_s + max(F_s, 0); a_P_nb = D_s + max(-F_s, 0); add_entry!(I_local, J_local, V_local, k, k, a_P_nb); add_entry!(I_local, J_local, V_local, k, k_s, -a_S_nb)
        end
        if j == ny
            handle_advection_diffusion_boundary_condition_2d!(I_local, J_local, V_local, b, model, mesh, k, bc_top, :top, transient, dx, dy_top, vy_n)
        else
            k_n = k + 1; a_N_nb = D_n + max(-F_n, 0); a_P_nb = D_n + max(F_n, 0); add_entry!(I_local, J_local, V_local, k, k, a_P_nb); add_entry!(I_local, J_local, V_local, k, k_n, -a_N_nb)
        end
        if source !== nothing
            b[k] += evaluate_source(source, mesh, i, j) * mesh.cells[k].volume
        end
    end
    I = reduce(vcat, I_chunks); J = reduce(vcat, J_chunks); V = reduce(vcat, V_chunks)
    A = sparse(I, J, V, n_cells, n_cells)
    return A, b
end

function get_diffusion_coefficient_at_face_2d(diffusion::Diffusion2D, mesh::Union{Mesh2D, CurvilinearMesh2D}, i, j, side)
    return diffusion.gamma
end

function get_diffusion_coefficient_at_face_2d(diffusion::VariableDiffusion2D, mesh::Union{Mesh2D, CurvilinearMesh2D}, i, j, side)
    gamma_P = get_diffusion_coefficient(diffusion, mesh, i, j)
    if side == :left
        if i == 1
            return gamma_P
        end
        gamma_W = get_diffusion_coefficient(diffusion, mesh, i - 1, j)
        return (gamma_P == 0 || gamma_W == 0) ? 0 : 2 * gamma_P * gamma_W / (gamma_P + gamma_W)
    elseif side == :right
        if i == mesh.nx
            return gamma_P
        end
        gamma_E = get_diffusion_coefficient(diffusion, mesh, i + 1, j)
        return (gamma_P == 0 || gamma_E == 0) ? 0 : 2 * gamma_P * gamma_E / (gamma_P + gamma_E)
    elseif side == :bottom
        if j == 1
            return gamma_P
        end
        gamma_S = get_diffusion_coefficient(diffusion, mesh, i, j - 1)
        return (gamma_P == 0 || gamma_S == 0) ? 0 : 2 * gamma_P * gamma_S / (gamma_P + gamma_S)
    else
        if j == mesh.ny
            return gamma_P
        end
        gamma_N = get_diffusion_coefficient(diffusion, mesh, i, j + 1)
        return (gamma_P == 0 || gamma_N == 0) ? 0 : 2 * gamma_P * gamma_N / (gamma_P + gamma_N)
    end
end
