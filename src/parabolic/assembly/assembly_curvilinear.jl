# assembly_curvilinear.jl - Flux assembly for curvilinear meshes
# Migrated from Simu.jl SimuFVM/assembly/assembly_curvilinear.jl
# Note: SimuGeometry dependency removed; CurvilinearMesh2D is now in parabolic/mesh/curvilinear.jl

"""
    assemble_system(diffusion, mesh::CurvilinearMesh2D, bcs; transient=false, source=nothing)

Assemble system matrix and RHS for 2D diffusion on a curvilinear mesh.
Uses geometric metrics (area, normal, distance) computed from node coordinates.
Approximates flux using primary gradient along the line connecting cell centers.
"""
function assemble_system(diffusion::Union{Diffusion2D, VariableDiffusion2D}, mesh::CurvilinearMesh2D, bcs; transient = false, source = nothing)
    nx = mesh.nx
    ny = mesh.ny
    n_cells = nx * ny

    b = zeros(n_cells)

    max_tid = 1024
    I_chunks = [Int[] for _ in 1:max_tid]
    J_chunks = [Int[] for _ in 1:max_tid]
    V_chunks = [Float64[] for _ in 1:max_tid]

    bc_left, bc_right, bc_bottom, bc_top = bcs

    Threads.@threads :static for k in 1:n_cells
        tid = Threads.threadid()
        if tid > max_tid
            error("Thread ID $tid exceeds maximum supported $max_tid")
        end
        I_local = I_chunks[tid]
        J_local = J_chunks[tid]
        V_local = V_chunks[tid]

        # Recover i, j (Column-major: k = (i-1)*ny + j)
        j = mod(k - 1, ny) + 1
        i = div(k - 1, ny) + 1

        cell_center = get_cell_center(mesh, i, j)

        # --- West Face (Left) ---
        center_w, normal_w, area_w = get_face_geo(mesh, i, j, :left)
        dist_w = norm(center_w - cell_center) # Distance to face

        if i == 1
            # Boundary
            handle_boundary_condition_2d!(I_local, J_local, V_local, b, diffusion, mesh, k, bc_left, :left, transient, dist_w, area_w)
        else
            neighbor_k = k - ny
            neighbor_center = get_cell_center(mesh, i - 1, j)
            d_PN = norm(cell_center - neighbor_center)

            vec_PN = neighbor_center - cell_center
            e_PN = vec_PN / d_PN
            ortho_corr = dot(e_PN, normal_w) # Should be ~1.0

            gamma_face = get_diffusion_coefficient_at_face_2d(diffusion, mesh, i, j, :left)

            # Total coeff
            flux_coeff = gamma_face * area_w * ortho_corr / d_PN

            add_entry!(I_local, J_local, V_local, k, k, flux_coeff)
            add_entry!(I_local, J_local, V_local, k, neighbor_k, -flux_coeff)
        end

        # --- East Face (Right) ---
        center_e, normal_e, area_e = get_face_geo(mesh, i, j, :right)
        dist_e = norm(center_e - cell_center)

        if i == nx
            handle_boundary_condition_2d!(I_local, J_local, V_local, b, diffusion, mesh, k, bc_right, :right, transient, dist_e, area_e)
        else
            neighbor_k = k + ny
            neighbor_center = get_cell_center(mesh, i + 1, j)
            d_PN = norm(neighbor_center - cell_center)
            vec_PN = neighbor_center - cell_center
            e_PN = vec_PN / d_PN
            ortho_corr = dot(e_PN, normal_e)

            gamma_face = get_diffusion_coefficient_at_face_2d(diffusion, mesh, i, j, :right)
            flux_coeff = gamma_face * area_e * ortho_corr / d_PN

            add_entry!(I_local, J_local, V_local, k, k, flux_coeff)
            add_entry!(I_local, J_local, V_local, k, neighbor_k, -flux_coeff)
        end

        # --- South Face (Bottom) ---
        center_s, normal_s, area_s = get_face_geo(mesh, i, j, :bottom)
        dist_s = norm(center_s - cell_center)

        if j == 1
            handle_boundary_condition_2d!(I_local, J_local, V_local, b, diffusion, mesh, k, bc_bottom, :bottom, transient, area_s, dist_s)
        else
            neighbor_k = k - 1
            neighbor_center = get_cell_center(mesh, i, j - 1)
            d_PN = norm(neighbor_center - cell_center)
            vec_PN = neighbor_center - cell_center
            e_PN = vec_PN / d_PN
            ortho_corr = dot(e_PN, normal_s)

            gamma_face = get_diffusion_coefficient_at_face_2d(diffusion, mesh, i, j, :bottom)
            flux_coeff = gamma_face * area_s * ortho_corr / d_PN

            add_entry!(I_local, J_local, V_local, k, k, flux_coeff)
            add_entry!(I_local, J_local, V_local, k, neighbor_k, -flux_coeff)
        end

        # --- North Face (Top) ---
        center_n, normal_n, area_n = get_face_geo(mesh, i, j, :top)
        dist_n = norm(center_n - cell_center)

        if j == ny
            handle_boundary_condition_2d!(I_local, J_local, V_local, b, diffusion, mesh, k, bc_top, :top, transient, area_n, dist_n)
        else
            neighbor_k = k + 1
            neighbor_center = get_cell_center(mesh, i, j + 1)
            d_PN = norm(neighbor_center - cell_center)
            vec_PN = neighbor_center - cell_center
            e_PN = vec_PN / d_PN
            ortho_corr = dot(e_PN, normal_n)

            gamma_face = get_diffusion_coefficient_at_face_2d(diffusion, mesh, i, j, :top)
            flux_coeff = gamma_face * area_n * ortho_corr / d_PN

            add_entry!(I_local, J_local, V_local, k, k, flux_coeff)
            add_entry!(I_local, J_local, V_local, k, neighbor_k, -flux_coeff)
        end

        # Source term
        if source !== nothing
            # Note: accessing mesh.cells[k].volume. CurvilinearMesh2D constructor populates this.
            b[k] += evaluate_source(source, mesh, i, j) * mesh.cells[k].volume
        end
    end

    I = reduce(vcat, I_chunks)
    J = reduce(vcat, J_chunks)
    V = reduce(vcat, V_chunks)
    A = sparse(I, J, V, n_cells, n_cells)
    return A, b
end
