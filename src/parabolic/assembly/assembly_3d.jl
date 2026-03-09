# Assembly routines for 3D problems
# Migrated from Simu.jl SimuFVM/assembly/assembly_3d.jl

function assemble_system(advection::Union{Advection3D, VariableAdvection3D}, mesh::Mesh3D, bcs; transient = false, source = nothing)
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz; n_cells = nx * ny * nz
    b = zeros(n_cells)
    max_tid = 1024
    I_chunks = [Int[] for _ in 1:max_tid]; J_chunks = [Int[] for _ in 1:max_tid]; V_chunks = [Float64[] for _ in 1:max_tid]
    bc_left, bc_right, bc_bottom, bc_top, bc_front, bc_back = bcs

    Threads.@threads :static for idx in 1:n_cells
        tid = Threads.threadid(); tid > max_tid && error("Thread ID $tid exceeds maximum supported $max_tid")
        I_local = I_chunks[tid]; J_local = J_chunks[tid]; V_local = V_chunks[tid]
        idx0 = idx - 1; k = (idx0 % nz) + 1; j = (div(idx0, nz) % ny) + 1; i = div(idx0, ny * nz) + 1
        dx = get_cell_dx(mesh, i, j, k); dy = get_cell_dy(mesh, i, j, k); dz = get_cell_dz(mesh, i, j, k)
        dx_left = get_face_dx(mesh, i, j, k, :left); dx_right = get_face_dx(mesh, i, j, k, :right)
        dy_bottom = get_face_dy(mesh, i, j, k, :bottom); dy_top = get_face_dy(mesh, i, j, k, :top)
        dz_front = get_face_dz(mesh, i, j, k, :front); dz_back = get_face_dz(mesh, i, j, k, :back)
        if advection isa Advection3D
            vx_w = advection.vx; vx_e = advection.vx; vy_s = advection.vy; vy_n = advection.vy; vz_f = advection.vz; vz_b = advection.vz
        else
            vx_w = i > 1 ? (get_velocity(advection, mesh, i - 1, j, k, :x) + get_velocity(advection, mesh, i, j, k, :x)) / 2.0 : get_velocity(advection, mesh, i, j, k, :x)
            vx_e = i < nx ? (get_velocity(advection, mesh, i, j, k, :x) + get_velocity(advection, mesh, i + 1, j, k, :x)) / 2.0 : get_velocity(advection, mesh, i, j, k, :x)
            vy_s = j > 1 ? (get_velocity(advection, mesh, i, j - 1, k, :y) + get_velocity(advection, mesh, i, j, k, :y)) / 2.0 : get_velocity(advection, mesh, i, j, k, :y)
            vy_n = j < ny ? (get_velocity(advection, mesh, i, j, k, :y) + get_velocity(advection, mesh, i, j + 1, k, :y)) / 2.0 : get_velocity(advection, mesh, i, j, k, :y)
            vz_f = k > 1 ? (get_velocity(advection, mesh, i, j, k - 1, :z) + get_velocity(advection, mesh, i, j, k, :z)) / 2.0 : get_velocity(advection, mesh, i, j, k, :z)
            vz_b = k < nz ? (get_velocity(advection, mesh, i, j, k, :z) + get_velocity(advection, mesh, i, j, k + 1, :z)) / 2.0 : get_velocity(advection, mesh, i, j, k, :z)
        end
        if i == 1
            handle_advection_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, advection, mesh, idx, bc_left, :left, transient, dx_left, dy, dz, vx_w)
        else
            idx_w = idx - ny * nz; if vx_w >= 0
                add_entry!(I_local, J_local, V_local, idx, idx_w, -vx_w * dy * dz)
            else
                add_entry!(I_local, J_local, V_local, idx, idx, abs(vx_w) * dy * dz)
            end
        end
        if i == nx
            handle_advection_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, advection, mesh, idx, bc_right, :right, transient, dx_right, dy, dz, vx_e)
        else
            idx_e = idx + ny * nz; if vx_e >= 0
                add_entry!(I_local, J_local, V_local, idx, idx, vx_e * dy * dz)
            else
                add_entry!(I_local, J_local, V_local, idx, idx_e, -abs(vx_e) * dy * dz)
            end
        end
        if j == 1
            handle_advection_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, advection, mesh, idx, bc_bottom, :bottom, transient, dx, dy_bottom, dz, vy_s)
        else
            idx_s = idx - nz; if vy_s >= 0
                add_entry!(I_local, J_local, V_local, idx, idx_s, -vy_s * dx * dz)
            else
                add_entry!(I_local, J_local, V_local, idx, idx, abs(vy_s) * dx * dz)
            end
        end
        if j == ny
            handle_advection_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, advection, mesh, idx, bc_top, :top, transient, dx, dy_top, dz, vy_n)
        else
            idx_n = idx + nz; if vy_n >= 0
                add_entry!(I_local, J_local, V_local, idx, idx, vy_n * dx * dz)
            else
                add_entry!(I_local, J_local, V_local, idx, idx_n, -abs(vy_n) * dx * dz)
            end
        end
        if k == 1
            handle_advection_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, advection, mesh, idx, bc_front, :front, transient, dx, dy, dz_front, vz_f)
        else
            idx_f = idx - 1; if vz_f >= 0
                add_entry!(I_local, J_local, V_local, idx, idx_f, -vz_f * dx * dy)
            else
                add_entry!(I_local, J_local, V_local, idx, idx, abs(vz_f) * dx * dy)
            end
        end
        if k == nz
            handle_advection_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, advection, mesh, idx, bc_back, :back, transient, dx, dy, dz_back, vz_b)
        else
            idx_b = idx + 1; if vz_b >= 0
                add_entry!(I_local, J_local, V_local, idx, idx, vz_b * dx * dy)
            else
                add_entry!(I_local, J_local, V_local, idx, idx_b, -abs(vz_b) * dx * dy)
            end
        end
        if source !== nothing
            b[idx] += evaluate_source(source, mesh, i, j, k) * mesh.cells[idx].volume
        end
    end
    A = sparse(reduce(vcat, I_chunks), reduce(vcat, J_chunks), reduce(vcat, V_chunks), n_cells, n_cells)
    return A, b
end

function assemble_system(diffusion::Union{Diffusion3D, VariableDiffusion3D}, mesh::Mesh3D, bcs; transient = false, source = nothing)
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz; n_cells = nx * ny * nz
    b = zeros(n_cells)
    max_tid = 1024
    I_chunks = [Int[] for _ in 1:max_tid]; J_chunks = [Int[] for _ in 1:max_tid]; V_chunks = [Float64[] for _ in 1:max_tid]
    bc_left, bc_right, bc_bottom, bc_top, bc_front, bc_back = bcs

    Threads.@threads :static for idx in 1:n_cells
        tid = Threads.threadid(); tid > max_tid && error("Thread ID $tid exceeds maximum supported $max_tid")
        I_local = I_chunks[tid]; J_local = J_chunks[tid]; V_local = V_chunks[tid]
        idx0 = idx - 1; k = (idx0 % nz) + 1; j = (div(idx0, nz) % ny) + 1; i = div(idx0, ny * nz) + 1
        dx_left = get_face_dx(mesh, i, j, k, :left); dx_right = get_face_dx(mesh, i, j, k, :right)
        dy_bottom = get_face_dy(mesh, i, j, k, :bottom); dy_top = get_face_dy(mesh, i, j, k, :top)
        dz_front = get_face_dz(mesh, i, j, k, :front); dz_back = get_face_dz(mesh, i, j, k, :back)
        dx = get_cell_dx(mesh, i, j, k); dy = get_cell_dy(mesh, i, j, k); dz = get_cell_dz(mesh, i, j, k)
        if i == 1
            handle_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, diffusion, mesh, idx, bc_left, :left, transient, dx, dy, dz)
        else
            ni = idx - ny * nz; gf = get_diffusion_coefficient_at_face_3d(diffusion, mesh, i, j, k, :left); fc = gf * dy * dz / dx_left; add_entry!(I_local, J_local, V_local, idx, idx, fc); add_entry!(I_local, J_local, V_local, idx, ni, -fc)
        end
        if i == nx
            handle_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, diffusion, mesh, idx, bc_right, :right, transient, dx, dy, dz)
        else
            ni = idx + ny * nz; gf = get_diffusion_coefficient_at_face_3d(diffusion, mesh, i, j, k, :right); fc = gf * dy * dz / dx_right; add_entry!(I_local, J_local, V_local, idx, idx, fc); add_entry!(I_local, J_local, V_local, idx, ni, -fc)
        end
        if j == 1
            handle_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, diffusion, mesh, idx, bc_bottom, :bottom, transient, dx, dy, dz)
        else
            ni = idx - nz; gf = get_diffusion_coefficient_at_face_3d(diffusion, mesh, i, j, k, :bottom); fc = gf * dx * dz / dy_bottom; add_entry!(I_local, J_local, V_local, idx, idx, fc); add_entry!(I_local, J_local, V_local, idx, ni, -fc)
        end
        if j == ny
            handle_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, diffusion, mesh, idx, bc_top, :top, transient, dx, dy, dz)
        else
            ni = idx + nz; gf = get_diffusion_coefficient_at_face_3d(diffusion, mesh, i, j, k, :top); fc = gf * dx * dz / dy_top; add_entry!(I_local, J_local, V_local, idx, idx, fc); add_entry!(I_local, J_local, V_local, idx, ni, -fc)
        end
        if k == 1
            handle_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, diffusion, mesh, idx, bc_front, :front, transient, dx, dy, dz)
        else
            ni = idx - 1; gf = get_diffusion_coefficient_at_face_3d(diffusion, mesh, i, j, k, :front); fc = gf * dx * dy / dz_front; add_entry!(I_local, J_local, V_local, idx, idx, fc); add_entry!(I_local, J_local, V_local, idx, ni, -fc)
        end
        if k == nz
            handle_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, diffusion, mesh, idx, bc_back, :back, transient, dx, dy, dz)
        else
            ni = idx + 1; gf = get_diffusion_coefficient_at_face_3d(diffusion, mesh, i, j, k, :back); fc = gf * dx * dy / dz_back; add_entry!(I_local, J_local, V_local, idx, idx, fc); add_entry!(I_local, J_local, V_local, idx, ni, -fc)
        end
        if source !== nothing
            b[idx] += evaluate_source(source, mesh, i, j, k) * mesh.cells[idx].volume
        end
    end
    A = sparse(reduce(vcat, I_chunks), reduce(vcat, J_chunks), reduce(vcat, V_chunks), n_cells, n_cells)
    return A, b
end

function assemble_system(diffusion::AnisotropicDiffusion3D, mesh::Mesh3D, bcs; transient = false, source = nothing)
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz; n_cells = nx * ny * nz
    b = zeros(n_cells)
    max_tid = 1024
    I_chunks = [Int[] for _ in 1:max_tid]; J_chunks = [Int[] for _ in 1:max_tid]; V_chunks = [Float64[] for _ in 1:max_tid]
    bc_left, bc_right, bc_bottom, bc_top, bc_front, bc_back = bcs

    Threads.@threads :static for idx in 1:n_cells
        tid = Threads.threadid(); tid > max_tid && error("Thread ID $tid exceeds maximum supported $max_tid")
        I_local = I_chunks[tid]; J_local = J_chunks[tid]; V_local = V_chunks[tid]
        idx0 = idx - 1; k = (idx0 % nz) + 1; j = (div(idx0, nz) % ny) + 1; i = div(idx0, ny * nz) + 1
        D = (diffusion.D isa Array{Float64, 5}) ? diffusion.D[:, :, i, j, k] : diffusion.D
        dx, dy, dz = get_cell_dx(mesh, i, j, k), get_cell_dy(mesh, i, j, k), get_cell_dz(mesh, i, j, k)
        dx_l, dx_r = get_face_dx(mesh, i, j, k, :left), get_face_dx(mesh, i, j, k, :right)
        dy_b, dy_t = get_face_dy(mesh, i, j, k, :bottom), get_face_dy(mesh, i, j, k, :top)
        dz_f, dz_bk = get_face_dz(mesh, i, j, k, :front), get_face_dz(mesh, i, j, k, :back)
        if i == 1
            handle_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, diffusion, mesh, idx, bc_left, :left, transient, dx, dy, dz)
        else
            fc = D[1, 1] * dy * dz / dx_l; add_entry!(I_local, J_local, V_local, idx, idx, fc); add_entry!(I_local, J_local, V_local, idx, idx - ny * nz, -fc)
        end
        if i == nx
            handle_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, diffusion, mesh, idx, bc_right, :right, transient, dx, dy, dz)
        else
            fc = D[1, 1] * dy * dz / dx_r; add_entry!(I_local, J_local, V_local, idx, idx, fc); add_entry!(I_local, J_local, V_local, idx, idx + ny * nz, -fc)
        end
        if j == 1
            handle_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, diffusion, mesh, idx, bc_bottom, :bottom, transient, dx, dy, dz)
        else
            fc = D[2, 2] * dx * dz / dy_b; add_entry!(I_local, J_local, V_local, idx, idx, fc); add_entry!(I_local, J_local, V_local, idx, idx - nz, -fc)
        end
        if j == ny
            handle_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, diffusion, mesh, idx, bc_top, :top, transient, dx, dy, dz)
        else
            fc = D[2, 2] * dx * dz / dy_t; add_entry!(I_local, J_local, V_local, idx, idx, fc); add_entry!(I_local, J_local, V_local, idx, idx + nz, -fc)
        end
        if k == 1
            handle_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, diffusion, mesh, idx, bc_front, :front, transient, dx, dy, dz)
        else
            fc = D[3, 3] * dx * dy / dz_f; add_entry!(I_local, J_local, V_local, idx, idx, fc); add_entry!(I_local, J_local, V_local, idx, idx - 1, -fc)
        end
        if k == nz
            handle_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, diffusion, mesh, idx, bc_back, :back, transient, dx, dy, dz)
        else
            fc = D[3, 3] * dx * dy / dz_bk; add_entry!(I_local, J_local, V_local, idx, idx, fc); add_entry!(I_local, J_local, V_local, idx, idx + 1, -fc)
        end
        if source !== nothing
            b[idx] += evaluate_source(source, mesh, i, j, k) * mesh.cells[idx].volume
        end
    end
    A = sparse(reduce(vcat, I_chunks), reduce(vcat, J_chunks), reduce(vcat, V_chunks), n_cells, n_cells)
    return A, b
end

function assemble_system(model::AdvectionDiffusion3D, mesh::Mesh3D, bcs; transient = false, source = nothing)
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz; n_cells = nx * ny * nz
    b = zeros(n_cells)
    max_tid = 1024
    I_chunks = [Int[] for _ in 1:max_tid]; J_chunks = [Int[] for _ in 1:max_tid]; V_chunks = [Float64[] for _ in 1:max_tid]
    bc_l, bc_r, bc_bt, bc_tp, bc_fr, bc_bk = bcs
    diff, adv = model.diffusion, model.advection
    gamma, vx, vy, vz = diff.gamma, adv.vx, adv.vy, adv.vz

    Threads.@threads :static for idx in 1:n_cells
        tid = Threads.threadid(); tid > max_tid && error("Thread ID $tid exceeds maximum supported $max_tid")
        I_local = I_chunks[tid]; J_local = J_chunks[tid]; V_local = V_chunks[tid]
        idx0 = idx - 1; k = (idx0 % nz) + 1; j = (div(idx0, nz) % ny) + 1; i = div(idx0, ny * nz) + 1
        dx, dy, dz = get_cell_dx(mesh, i, j, k), get_cell_dy(mesh, i, j, k), get_cell_dz(mesh, i, j, k)
        dx_l, dx_r = get_face_dx(mesh, i, j, k, :left), get_face_dx(mesh, i, j, k, :right)
        dy_b, dy_t = get_face_dy(mesh, i, j, k, :bottom), get_face_dy(mesh, i, j, k, :top)
        dz_f, dz_bk = get_face_dz(mesh, i, j, k, :front), get_face_dz(mesh, i, j, k, :back)
        F_w, F_e = vx * dy * dz, vx * dy * dz; F_s, F_n = vy * dx * dz, vy * dx * dz; F_f, F_b = vz * dx * dy, vz * dx * dy
        D_w, D_e = gamma * dy * dz / dx_l, gamma * dy * dz / dx_r; D_s, D_n = gamma * dx * dz / dy_b, gamma * dx * dz / dy_t; D_f, D_b = gamma * dx * dy / dz_f, gamma * dx * dy / dz_bk
        if i == 1
            handle_advection_diffusion_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, model, mesh, idx, bc_l, :left, transient, dx, dy, dz, vx)
        else
            a_W_nb = D_w + max(F_w, 0); a_P_nb = D_w + max(-F_w, 0); add_entry!(I_local, J_local, V_local, idx, idx, a_P_nb); add_entry!(I_local, J_local, V_local, idx, idx - ny * nz, -a_W_nb)
        end
        if i == nx
            handle_advection_diffusion_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, model, mesh, idx, bc_r, :right, transient, dx, dy, dz, vx)
        else
            a_E_nb = D_e + max(-F_e, 0); a_P_nb = D_e + max(F_e, 0); add_entry!(I_local, J_local, V_local, idx, idx, a_P_nb); add_entry!(I_local, J_local, V_local, idx, idx + ny * nz, -a_E_nb)
        end
        if j == 1
            handle_advection_diffusion_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, model, mesh, idx, bc_bt, :bottom, transient, dx, dy, dz, vy)
        else
            a_S_nb = D_s + max(F_s, 0); a_P_nb = D_s + max(-F_s, 0); add_entry!(I_local, J_local, V_local, idx, idx, a_P_nb); add_entry!(I_local, J_local, V_local, idx, idx - nz, -a_S_nb)
        end
        if j == ny
            handle_advection_diffusion_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, model, mesh, idx, bc_tp, :top, transient, dx, dy, dz, vy)
        else
            a_N_nb = D_n + max(-F_n, 0); a_P_nb = D_n + max(F_n, 0); add_entry!(I_local, J_local, V_local, idx, idx, a_P_nb); add_entry!(I_local, J_local, V_local, idx, idx + nz, -a_N_nb)
        end
        if k == 1
            handle_advection_diffusion_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, model, mesh, idx, bc_fr, :front, transient, dx, dy, dz, vz)
        else
            a_F_nb = D_f + max(F_f, 0); a_P_nb = D_f + max(-F_f, 0); add_entry!(I_local, J_local, V_local, idx, idx, a_P_nb); add_entry!(I_local, J_local, V_local, idx, idx - 1, -a_F_nb)
        end
        if k == nz
            handle_advection_diffusion_boundary_condition_3d_triplet!(I_local, J_local, V_local, b, model, mesh, idx, bc_bk, :back, transient, dx, dy, dz, vz)
        else
            a_B_nb = D_b + max(-F_b, 0); a_P_nb = D_b + max(F_b, 0); add_entry!(I_local, J_local, V_local, idx, idx, a_P_nb); add_entry!(I_local, J_local, V_local, idx, idx + 1, -a_B_nb)
        end
        if source !== nothing
            b[idx] += evaluate_source(source, mesh, i, j, k) * mesh.cells[idx].volume
        end
    end
    A = sparse(reduce(vcat, I_chunks), reduce(vcat, J_chunks), reduce(vcat, V_chunks), n_cells, n_cells)
    return A, b
end
