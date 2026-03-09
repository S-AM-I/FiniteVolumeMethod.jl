# Diagnostics — migrated from Simu.jl SimuIO
# Volume integrals, conservation summaries, boundary flux computation, and CSV output.

"""
    volume_integral(mesh, vals)

Volume-weighted integral of a field on a mesh.

Supports `StructuredFVMMesh`, `CurvilinearFVMMesh`, `UnstructuredFVMMesh` (via `cell_volumes`),
and `Mesh1D`/`Mesh2D`/`Mesh3D` (via per-cell `volume` property).
"""
function volume_integral(mesh, vals)
    if mesh isa StructuredFVMMesh || mesh isa CurvilinearFVMMesh || mesh isa UnstructuredFVMMesh
        return sum(vec(vals) .* vec(mesh.cell_volumes))
    elseif hasproperty(mesh, :cell_volumes)
        return sum(vec(vals) .* vec(mesh.cell_volumes))
    elseif hasproperty(mesh, :cells)
        cells = getproperty(mesh, :cells)
        if !isempty(cells) && hasproperty(first(cells), :volume)
            volumes = [getproperty(c, :volume) for c in cells]
            if length(volumes) == length(vals)
                return sum(vec(vals) .* vec(volumes))
            end
        end
        return sum(vals)
    else
        # Fallback for meshes without cell_volumes property
        return sum(vals)
    end
end

"""
    conservation_summary(mesh, vals)

Quick conservation summary (min/max/total) for a field.
"""
function conservation_summary(mesh, vals)
    total = volume_integral(mesh, vals)
    return (; total, min = minimum(vals), max = maximum(vals))
end

"""
    boundary_fluxes(f!, u, t; p=nothing)

Compute boundary flux tallies and operator splits by setting task-local storage
before evaluating the RHS function `f!(du, u, p, t)`.

Returns `(flux_dict, du_snapshot, operator_splits)`.
"""
function boundary_fluxes(f!, u, t; p = nothing)
    task_local_storage()[:boundary_fluxes] = Dict{Any, Float64}()
    task_local_storage()[:operator_splits] = Dict{Symbol, Dict{Symbol, Float64}}()
    du = similar(u)
    f!(du, u, p, t)
    fluxes = deepcopy(get(task_local_storage(), :boundary_fluxes, Dict{Any, Float64}()))
    splits = deepcopy(get(task_local_storage(), :operator_splits, Dict{Symbol, Dict{Symbol, Float64}}()))
    return fluxes, du, splits
end

"""
    boundary_fluxes(mesh::Mesh1D, field_values, model, bc_left, bc_right; field_name=:phi)

Compute boundary fluxes for a 1D field on a mesh.
"""
function boundary_fluxes(mesh::Mesh1D, field_values, model, bc_left, bc_right; field_name = :phi)
    fluxes = Dict{Tuple{Symbol, Symbol}, Float64}()

    # Determine model type and get coefficients
    gamma = nothing
    v = nothing
    if hasproperty(model, :gamma)
        gamma = model.gamma
    elseif hasproperty(model, :diffusion)
        gamma = model.diffusion.gamma
    end
    if hasproperty(model, :v)
        v = model.v
    elseif hasproperty(model, :advection)
        v = model.advection.v
    end

    # Compute left boundary flux
    if length(field_values) > 0
        if bc_left isa ParabolicDirichlet
            # Diffusive Flux = -gamma * (phi[1] - bc.value) / dx_half
            dx_half = mesh.cells[1].center - mesh.nodes[1].x
            flux_diff = (gamma !== nothing) ? -gamma * (field_values[1] - bc_left.value) / dx_half : 0.0

            # Advective Flux = v * phi_face
            # Upwind: if v >= 0 (inflow), phi_face = bc.value. if v < 0 (outflow), phi_face = phi[1]
            flux_adv = 0.0
            if v !== nothing
                if v >= 0
                    flux_adv = v * bc_left.value
                else
                    flux_adv = v * field_values[1]
                end
            end

            # Outward flux (normal -1)
            fluxes[(field_name, :left)] = -(flux_diff + flux_adv)
        elseif bc_left isa ParabolicNeumann
            fluxes[(field_name, :left)] = bc_left.value
            if v !== nothing
                if v < 0 # outflow
                    fluxes[(field_name, :left)] += abs(v) * field_values[1]
                end
            end
        elseif bc_left isa ParabolicRobin
            dx_half = mesh.cells[1].center - mesh.nodes[1].x
            phi_bc = (gamma !== nothing) ? bc_left.c / (bc_left.a + bc_left.b * gamma / dx_half) : bc_left.c / bc_left.a
            flux_diff = (gamma !== nothing) ? -gamma * (field_values[1] - phi_bc) / dx_half : 0.0

            flux_adv = 0.0
            if v !== nothing
                if v >= 0
                    flux_adv = v * phi_bc
                else
                    flux_adv = v * field_values[1]
                end
            end
            fluxes[(field_name, :left)] = -(flux_diff + flux_adv)
        end

        # Compute right boundary flux
        if bc_right isa ParabolicDirichlet
            dx_half = mesh.nodes[end].x - mesh.cells[end].center
            flux_diff = (gamma !== nothing) ? -gamma * (bc_right.value - field_values[end]) / dx_half : 0.0

            flux_adv = 0.0
            if v !== nothing
                if v >= 0 # outflow
                    flux_adv = v * field_values[end]
                else # inflow
                    flux_adv = v * bc_right.value
                end
            end

            # Outward flux (normal +1)
            fluxes[(field_name, :right)] = (flux_diff + flux_adv)
        elseif bc_right isa ParabolicNeumann
            fluxes[(field_name, :right)] = bc_right.value
            if v !== nothing
                if v >= 0 # outflow
                    fluxes[(field_name, :right)] += v * field_values[end]
                end
            end
        elseif bc_right isa ParabolicRobin
            dx_half = mesh.nodes[end].x - mesh.cells[end].center
            phi_bc = (gamma !== nothing) ? bc_right.c / (bc_right.a + bc_right.b * gamma / dx_half) : bc_right.c / bc_right.a
            flux_diff = (gamma !== nothing) ? -gamma * (phi_bc - field_values[end]) / dx_half : 0.0

            flux_adv = 0.0
            if v !== nothing
                if v >= 0
                    flux_adv = v * field_values[end]
                else
                    flux_adv = v * phi_bc
                end
            end
            fluxes[(field_name, :right)] = (flux_diff + flux_adv)
        end
    end

    return fluxes
end

"""
    boundary_fluxes(mesh::Mesh2D, field_values, model, bcs; field_name=:phi)

Compute boundary fluxes for a 2D field on a mesh.
"""
function boundary_fluxes(mesh::Mesh2D, field_values, model, bcs; field_name = :phi)
    fluxes = Dict{Tuple{Symbol, Symbol}, Float64}()
    bc_left, bc_right, bc_bottom, bc_top = bcs

    nx = mesh.nx
    ny = mesh.ny
    dx = mesh.Lx / nx
    dy = mesh.Ly / ny

    if field_values isa AbstractVector
        field_2d = reshape(field_values, ny, nx)'
    else
        field_2d = field_values
    end

    # Determine model type and get coefficients
    gamma = nothing
    vx = nothing
    vy = nothing
    if hasproperty(model, :gamma)
        gamma = model.gamma
    elseif hasproperty(model, :diffusion)
        gamma = model.diffusion.gamma
    end
    if hasproperty(model, :vx)
        vx = model.vx
        vy = model.vy
    elseif hasproperty(model, :advection)
        vx = model.advection.vx
        vy = model.advection.vy
    end

    # Left boundary (x = 0)
    if bc_left isa ParabolicDirichlet
        flux_sum = 0.0
        for j in 1:ny
            dx_half = dx / 2
            if gamma !== nothing
                flux_diff = -gamma * (field_2d[1, j] - bc_left.value) / dx_half * dy
            else
                flux_diff = 0.0
            end
            if vx !== nothing
                if vx >= 0
                    flux_adv = vx * bc_left.value * dy
                else
                    flux_adv = vx * field_2d[1, j] * dy
                end
            else
                flux_adv = 0.0
            end
            flux_sum += flux_diff + flux_adv
        end
        fluxes[(field_name, :left)] = -flux_sum
    elseif bc_left isa ParabolicNeumann
        if gamma !== nothing
            fluxes[(field_name, :left)] = bc_left.value * mesh.Ly
        else
            fluxes[(field_name, :left)] = 0.0
        end
    elseif bc_left isa ParabolicRobin
        flux_sum = 0.0
        for j in 1:ny
            dx_half = dx / 2
            if gamma !== nothing
                phi_bc = bc_left.c / (bc_left.a + bc_left.b * gamma / dx_half)
                flux_diff = -gamma * (field_2d[1, j] - phi_bc) / dx_half * dy
            else
                flux_diff = 0.0
            end
            if vx !== nothing
                phi_bc = bc_left.c / (bc_left.a + bc_left.b * abs(vx))
                flux_adv = vx * phi_bc * dy
            else
                flux_adv = 0.0
            end
            flux_sum += flux_diff + flux_adv
        end
        fluxes[(field_name, :left)] = -flux_sum
    end

    # Right boundary (x = Lx)
    if bc_right isa ParabolicDirichlet
        flux_sum = 0.0
        for j in 1:ny
            dx_half = dx / 2
            if gamma !== nothing
                flux_diff = -gamma * (bc_right.value - field_2d[nx, j]) / dx_half * dy
            else
                flux_diff = 0.0
            end
            if vx !== nothing
                if vx >= 0
                    flux_adv = vx * field_2d[nx, j] * dy
                else
                    flux_adv = vx * bc_right.value * dy
                end
            end
            flux_sum += flux_diff + flux_adv
        end
        fluxes[(field_name, :right)] = flux_sum
    elseif bc_right isa ParabolicNeumann
        if gamma !== nothing
            fluxes[(field_name, :right)] = bc_right.value * mesh.Ly
        else
            fluxes[(field_name, :right)] = 0.0
        end
    elseif bc_right isa ParabolicRobin
        flux_sum = 0.0
        for j in 1:ny
            dx_half = dx / 2
            if gamma !== nothing
                phi_bc = bc_right.c / (bc_right.a + bc_right.b * gamma / dx_half)
                flux_diff = -gamma * (phi_bc - field_2d[nx, j]) / dx_half * dy
            else
                flux_diff = 0.0
            end
            if vx !== nothing
                phi_bc = bc_right.c / (bc_right.a + bc_right.b * abs(vx))
                flux_adv = vx * phi_bc * dy
            end
            flux_sum += flux_diff + flux_adv
        end
        fluxes[(field_name, :right)] = flux_sum
    end

    # Bottom boundary (y = 0)
    if bc_bottom isa ParabolicDirichlet
        flux_sum = 0.0
        for i in 1:nx
            dy_half = dy / 2
            if gamma !== nothing
                flux_diff = -gamma * (field_2d[i, 1] - bc_bottom.value) / dy_half * dx
            else
                flux_diff = 0.0
            end
            if vy !== nothing
                if vy >= 0
                    flux_adv = vy * bc_bottom.value * dx
                else
                    flux_adv = vy * field_2d[i, 1] * dx
                end
            else
                flux_adv = 0.0
            end
            flux_sum += flux_diff + flux_adv
        end
        fluxes[(field_name, :bottom)] = -flux_sum
    elseif bc_bottom isa ParabolicNeumann
        if gamma !== nothing
            fluxes[(field_name, :bottom)] = bc_bottom.value * mesh.Lx
        else
            fluxes[(field_name, :bottom)] = 0.0
        end
    elseif bc_bottom isa ParabolicRobin
        flux_sum = 0.0
        for i in 1:nx
            dy_half = dy / 2
            if gamma !== nothing
                phi_bc = bc_bottom.c / (bc_bottom.a + bc_bottom.b * gamma / dy_half)
                flux_diff = -gamma * (field_2d[i, 1] - phi_bc) / dy_half * dx
            else
                flux_diff = 0.0
            end
            if vy !== nothing
                phi_bc = bc_bottom.c / (bc_bottom.a + bc_bottom.b * abs(vy))
                flux_adv = vy * phi_bc * dx
            else
                flux_adv = 0.0
            end
            flux_sum += flux_diff + flux_adv
        end
        fluxes[(field_name, :bottom)] = -flux_sum
    end

    # Top boundary (y = Ly)
    if bc_top isa ParabolicDirichlet
        flux_sum = 0.0
        for i in 1:nx
            dy_half = dy / 2
            if gamma !== nothing
                flux_diff = -gamma * (bc_top.value - field_2d[i, ny]) / dy_half * dx
            else
                flux_diff = 0.0
            end
            if vy !== nothing
                if vy >= 0
                    flux_adv = vy * field_2d[i, ny] * dx
                else
                    flux_adv = vy * bc_top.value * dx
                end
            else
                flux_adv = 0.0
            end
            flux_sum += flux_diff + flux_adv
        end
        fluxes[(field_name, :top)] = flux_sum
    elseif bc_top isa ParabolicNeumann
        if gamma !== nothing
            fluxes[(field_name, :top)] = bc_top.value * mesh.Lx
        else
            fluxes[(field_name, :top)] = 0.0
        end
    elseif bc_top isa ParabolicRobin
        flux_sum = 0.0
        for i in 1:nx
            dy_half = dy / 2
            if gamma !== nothing
                phi_bc = bc_top.c / (bc_top.a + bc_top.b * gamma / dy_half)
                flux_diff = -gamma * (phi_bc - field_2d[i, ny]) / dy_half * dx
            else
                flux_diff = 0.0
            end
            if vy !== nothing
                phi_bc = bc_top.c / (bc_top.a + bc_top.b * abs(vy))
                flux_adv = vy * phi_bc * dx
            else
                flux_adv = 0.0
            end
            flux_sum += flux_diff + flux_adv
        end
        fluxes[(field_name, :top)] = flux_sum
    end

    return fluxes
end

"""
    boundary_fluxes(mesh::Mesh3D, field_values, model, bcs; field_name=:phi)

Calculate boundary fluxes for a 3D problem.
"""
function boundary_fluxes(mesh::Mesh3D, field_values, model, bcs; field_name = :phi)
    bc_left, bc_right, bc_bottom, bc_top, bc_front, bc_back = bcs
    fluxes = Dict{Tuple{Symbol, Symbol}, Float64}()

    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz

    # Helper for linear indexing: (k varies fastest, then j, then i)
    get_idx(i, j, k) = (i - 1) * ny * nz + (j - 1) * nz + k

    # Determine model type and get coefficients
    _get_gamma(model, _mesh, _i, _j, _k) = begin
        if model isa Diffusion3D || model isa VariableDiffusion3D
            return hasproperty(model, :gamma) ? model.gamma : 0.0
        elseif model isa AdvectionDiffusion3D
            if hasproperty(model.diffusion, :gamma) && model.diffusion.gamma isa Number
                return model.diffusion.gamma
            else
                return 0.0
            end
        end
        return 0.0
    end

    _get_vel(model, comp) = begin
        if model isa AdvectionDiffusion3D
            adv = model.advection
            if comp == :x && hasproperty(adv, :vx) && adv.vx isa Number
                return adv.vx
            elseif comp == :y && hasproperty(adv, :vy) && adv.vy isa Number
                return adv.vy
            elseif comp == :z && hasproperty(adv, :vz) && adv.vz isa Number
                return adv.vz
            end
        end
        return 0.0
    end

    # Compute dx/dy/dz from mesh
    dx_val = mesh.Lx / nx
    dy_val = mesh.Ly / ny
    dz_val = mesh.Lz / nz

    # --- Left Boundary (x = 0, i = 1) ---
    flux_sum = 0.0
    for j in 1:ny, k in 1:nz
        idx = get_idx(1, j, k)
        area = dy_val * dz_val
        gamma = _get_gamma(model, mesh, 1, j, k)
        vx = _get_vel(model, :x)

        flux_diff = 0.0
        if bc_left isa ParabolicDirichlet
            flux_diff = -gamma * (field_values[idx] - bc_left.value) / (dx_val / 2) * area
        elseif bc_left isa ParabolicNeumann
            flux_diff = bc_left.value * area
        end

        flux_adv = 0.0
        if vx != 0.0
            val = (bc_left isa ParabolicDirichlet && vx >= 0) ? bc_left.value : field_values[idx]
            if bc_left isa ParabolicDirichlet || (bc_left isa ParabolicNeumann && vx < 0)
                flux_adv = vx * val * area
            end
        end
        flux_sum += flux_diff + flux_adv
    end
    fluxes[(field_name, :left)] = -flux_sum # Outward normal is -x

    # --- Right Boundary (x = Lx, i = nx) ---
    flux_sum = 0.0
    for j in 1:ny, k in 1:nz
        idx = get_idx(nx, j, k)
        area = dy_val * dz_val
        gamma = _get_gamma(model, mesh, nx, j, k)
        vx = _get_vel(model, :x)

        flux_diff = 0.0
        if bc_right isa ParabolicDirichlet
            flux_diff = -gamma * (bc_right.value - field_values[idx]) / (dx_val / 2) * area
        elseif bc_right isa ParabolicNeumann
            flux_diff = bc_right.value * area
        end

        flux_adv = 0.0
        if vx != 0.0
            val = (bc_right isa ParabolicDirichlet && vx < 0) ? bc_right.value : field_values[idx]
            if bc_right isa ParabolicDirichlet || (bc_right isa ParabolicNeumann && vx >= 0)
                flux_adv = vx * val * area
            end
        end
        flux_sum += flux_diff + flux_adv
    end
    fluxes[(field_name, :right)] = flux_sum # Outward normal is +x

    # --- Bottom Boundary (y = 0, j = 1) ---
    flux_sum = 0.0
    for i in 1:nx, k in 1:nz
        idx = get_idx(i, 1, k)
        area = dx_val * dz_val
        gamma = _get_gamma(model, mesh, i, 1, k)
        vy = _get_vel(model, :y)

        flux_diff = 0.0
        if bc_bottom isa ParabolicDirichlet
            flux_diff = -gamma * (field_values[idx] - bc_bottom.value) / (dy_val / 2) * area
        elseif bc_bottom isa ParabolicNeumann
            flux_diff = bc_bottom.value * area
        end

        flux_adv = 0.0
        if vy != 0.0
            val = (bc_bottom isa ParabolicDirichlet && vy >= 0) ? bc_bottom.value : field_values[idx]
            if bc_bottom isa ParabolicDirichlet || (bc_bottom isa ParabolicNeumann && vy < 0)
                flux_adv = vy * val * area
            end
        end
        flux_sum += flux_diff + flux_adv
    end
    fluxes[(field_name, :bottom)] = -flux_sum # Outward normal is -y

    # --- Top Boundary (y = Ly, j = ny) ---
    flux_sum = 0.0
    for i in 1:nx, k in 1:nz
        idx = get_idx(i, ny, k)
        area = dx_val * dz_val
        gamma = _get_gamma(model, mesh, i, ny, k)
        vy = _get_vel(model, :y)

        flux_diff = 0.0
        if bc_top isa ParabolicDirichlet
            flux_diff = -gamma * (bc_top.value - field_values[idx]) / (dy_val / 2) * area
        elseif bc_top isa ParabolicNeumann
            flux_diff = bc_top.value * area
        end

        flux_adv = 0.0
        if vy != 0.0
            val = (bc_top isa ParabolicDirichlet && vy < 0) ? bc_top.value : field_values[idx]
            if bc_top isa ParabolicDirichlet || (bc_top isa ParabolicNeumann && vy >= 0)
                flux_adv = vy * val * area
            end
        end
        flux_sum += flux_diff + flux_adv
    end
    fluxes[(field_name, :top)] = flux_sum # Outward normal is +y

    # --- Front Boundary (z = 0, k = 1) ---
    flux_sum = 0.0
    for i in 1:nx, j in 1:ny
        idx = get_idx(i, j, 1)
        area = dx_val * dy_val
        gamma = _get_gamma(model, mesh, i, j, 1)
        vz = _get_vel(model, :z)

        flux_diff = 0.0
        if bc_front isa ParabolicDirichlet
            flux_diff = -gamma * (field_values[idx] - bc_front.value) / (dz_val / 2) * area
        elseif bc_front isa ParabolicNeumann
            flux_diff = bc_front.value * area
        end

        flux_adv = 0.0
        if vz != 0.0
            val = (bc_front isa ParabolicDirichlet && vz >= 0) ? bc_front.value : field_values[idx]
            if bc_front isa ParabolicDirichlet || (bc_front isa ParabolicNeumann && vz < 0)
                flux_adv = vz * val * area
            end
        end
        flux_sum += flux_diff + flux_adv
    end
    fluxes[(field_name, :front)] = -flux_sum # Outward normal is -z

    # --- Back Boundary (z = Lz, k = nz) ---
    flux_sum = 0.0
    for i in 1:nx, j in 1:ny
        idx = get_idx(i, j, nz)
        area = dx_val * dy_val
        gamma = _get_gamma(model, mesh, i, j, nz)
        vz = _get_vel(model, :z)

        flux_diff = 0.0
        if bc_back isa ParabolicDirichlet
            flux_diff = -gamma * (bc_back.value - field_values[idx]) / (dz_val / 2) * area
        elseif bc_back isa ParabolicNeumann
            flux_diff = bc_back.value * area
        end

        flux_adv = 0.0
        if vz != 0.0
            val = (bc_back isa ParabolicDirichlet && vz < 0) ? bc_back.value : field_values[idx]
            if bc_back isa ParabolicDirichlet || (bc_back isa ParabolicNeumann && vz >= 0)
                flux_adv = vz * val * area
            end
        end
        flux_sum += flux_diff + flux_adv
    end
    fluxes[(field_name, :back)] = flux_sum # Outward normal is +z

    return fluxes
end

"""
    flux_inout(fluxes::Dict)

Separate inflow (<0) and outflow (>0) totals per field from a boundary fluxes dictionary.
"""
function flux_inout(fluxes::Dict)
    summary = Dict{Symbol, Dict{Symbol, Float64}}()
    for (k, v) in fluxes
        if k isa Tuple && length(k) == 2
            field = k[1]
            fld = get!(summary, field) do
                Dict(:inflow => 0.0, :outflow => 0.0)
            end
            if v >= 0
                fld[:outflow] += v
            else
                fld[:inflow] += v
            end
            summary[field] = fld
        end
    end
    return summary
end

"""
    write_boundary_flux_csv(dir, filename, fluxes)

Write boundary flux table to CSV.
"""
function write_boundary_flux_csv(dir::AbstractString, filename::AbstractString, fluxes::Dict)
    path = joinpath(dir, filename)
    open(path, "w") do io
        println(io, "field,region,flux,in_or_out")
        for (k, v) in fluxes
            if k isa Tuple && length(k) >= 2
                field, region = k[1], k[2]
            else
                field, region = Symbol("unknown"), string(k)
            end
            tag = v >= 0 ? "outflow" : "inflow"
            println(io, "$(field),$(region),$(v),$(tag)")
        end
    end
    return path
end

"""
    write_operator_splits_csv(dir, filename, splits)

Write operator split totals to CSV.
"""
function write_operator_splits_csv(dir::AbstractString, filename::AbstractString, splits::Dict)
    path = joinpath(dir, filename)
    open(path, "w") do io
        println(io, "field,volume,flux,flux_advection,flux_diffusion,source_reaction,source_generic,bc,total")
        for (field, d) in splits
            if d isa Dict
                vol = get(d, :volume, 0.0)
                flux = get(d, :flux, 0.0)
                flux_adv = get(d, :flux_advection, 0.0)
                flux_diff = get(d, :flux_diffusion, 0.0)
                src_rxn = get(d, :source_reaction, 0.0)
                src_gen = get(d, :source_generic, 0.0)
                bc = get(d, :bc, 0.0)
                total = get(d, :total, 0.0)
                println(io, "$(field),$(vol),$(flux),$(flux_adv),$(flux_diff),$(src_rxn),$(src_gen),$(bc),$(total)")
            end
        end
    end
    return path
end
