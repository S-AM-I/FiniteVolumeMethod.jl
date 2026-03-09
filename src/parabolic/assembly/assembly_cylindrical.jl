# Assembly routines for cylindrical coordinates
# Migrated from Simu.jl SimuFVM/assembly/assembly_cylindrical.jl

"""
    assemble_system(model::CylindricalDiffusion1D, mesh::Mesh1D, bc_left, bc_right; source=nothing, transient=false)

Assemble the system matrix and RHS for 1D cylindrical diffusion (radial only).
The radial Laplacian is (1/r) d/dr (r * gamma * dphi/dr).
In FVM form: integrate gamma grad(phi) . n dA.
For 1D radial: (gamma * r * dphi/dr)|_out * 2pi - (gamma * r * dphi/dr)|_in * 2pi = Source * Volume.
"""
function assemble_system(model::CylindricalDiffusion1D, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition; source = nothing, transient = false)
    nx = length(mesh.cells)
    A = SparseArrays.spzeros(nx, nx)
    b = zeros(nx)
    gamma = model.gamma

    for i in 1:nx
        r_in = mesh.nodes[i].x
        r_out = mesh.nodes[i + 1].x
        dr = r_out - r_in

        # Cell volume: V = pi * (r_out^2 - r_in^2)
        volume = pi * (r_out^2 - r_in^2)

        # Left face (r_in)
        if i == 1
            # Boundary condition at r_min
            # Area at r_in: 2pi * r_in
            area_in = 2 * pi * r_in
            handle_cylindrical_boundary_condition!(A, b, model, mesh, i, bc_left, :left, area_in, transient)
        else
            # Internal face
            r_face = r_in
            area_face = 2 * pi * r_face
            dr_face = mesh.cells[i].center - mesh.cells[i - 1].center
            flux_coeff = gamma * area_face / dr_face
            A[i, i] += flux_coeff
            A[i, i - 1] -= flux_coeff
        end

        # Right face (r_out)
        if i == nx
            # Boundary condition at r_max
            # Area at r_out: 2pi * r_out
            area_out = 2 * pi * r_out
            handle_cylindrical_boundary_condition!(A, b, model, mesh, i, bc_right, :right, area_out, transient)
        else
            # Internal face
            r_face = r_out
            area_face = 2 * pi * r_face
            dr_face = mesh.cells[i + 1].center - mesh.cells[i].center
            flux_coeff = gamma * area_face / dr_face
            A[i, i] += flux_coeff
            A[i, i + 1] -= flux_coeff
        end

        # Source term
        if source !== nothing
            b[i] += evaluate_source(source, mesh, i) * volume
        end
    end

    return A, b
end

"""
    assemble_system(model::CylindricalAdvection1D, mesh::Mesh1D, bc_left, bc_right; source=nothing, transient=false)

Assemble system for 1D cylindrical advection (radial).
Flux = v * phi * Area. Area = 2 * pi * r.
"""
function assemble_system(model::CylindricalAdvection1D, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition; source = nothing, transient = false)
    nx = length(mesh.cells)
    A = SparseArrays.spzeros(nx, nx)
    b = zeros(nx)
    v = model.v

    for i in 1:nx
        r_in = mesh.nodes[i].x
        r_out = mesh.nodes[i + 1].x
        volume = pi * (r_out^2 - r_in^2)

        # Left face (r_in)
        area_in = 2 * pi * r_in
        if i == 1
            handle_cylindrical_advection_bc!(A, b, model, mesh, i, bc_left, :left, area_in, transient)
        else
            # Upwind
            if v >= 0
                # Flow from left (i-1) -> i
                flux = v * area_in
                A[i, i - 1] -= flux
            else
                # Flow from i -> left (i-1)
                flux = abs(v) * area_in
                A[i, i] += flux
            end
        end

        # Right face (r_out)
        area_out = 2 * pi * r_out
        if i == nx
            handle_cylindrical_advection_bc!(A, b, model, mesh, i, bc_right, :right, area_out, transient)
        else
            # Upwind
            if v >= 0
                # Flow from i -> right (i+1)
                flux = v * area_out
                A[i, i] += flux
            else
                # Flow from right (i+1) -> i
                flux = abs(v) * area_out
                A[i, i + 1] -= flux
            end
        end

        if source !== nothing
            b[i] += evaluate_source(source, mesh, i) * volume
        end
    end

    return A, b
end

"""
    assemble_system(model::CylindricalAdvectionDiffusion1D, mesh::Mesh1D, bc_left, bc_right; source=nothing, transient=false)

Assemble system for 1D cylindrical advection-diffusion.
"""
function assemble_system(model::CylindricalAdvectionDiffusion1D, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition; source = nothing, transient = false)
    nx = length(mesh.cells)
    A = SparseArrays.spzeros(nx, nx)
    b = zeros(nx)

    v = model.advection.v
    gamma = model.diffusion.gamma

    for i in 1:nx
        r_in = mesh.nodes[i].x
        r_out = mesh.nodes[i + 1].x
        volume = pi * (r_out^2 - r_in^2)

        # Left face
        area_in = 2 * pi * r_in
        if i == 1
            handle_cylindrical_advection_diffusion_bc!(A, b, model, mesh, i, bc_left, :left, area_in, transient)
        else
            dr = mesh.cells[i].center - mesh.cells[i - 1].center
            # Diffusion
            diff_flux_coeff = gamma * area_in / dr
            A[i, i] += diff_flux_coeff
            A[i, i - 1] -= diff_flux_coeff

            # Advection
            if v >= 0
                adv_flux = v * area_in
                A[i, i - 1] -= adv_flux
            else
                adv_flux = abs(v) * area_in
                A[i, i] += adv_flux
            end
        end

        # Right face
        area_out = 2 * pi * r_out
        if i == nx
            handle_cylindrical_advection_diffusion_bc!(A, b, model, mesh, i, bc_right, :right, area_out, transient)
        else
            dr = mesh.cells[i + 1].center - mesh.cells[i].center
            # Diffusion
            diff_flux_coeff = gamma * area_out / dr
            A[i, i] += diff_flux_coeff
            A[i, i + 1] -= diff_flux_coeff

            # Advection
            if v >= 0
                adv_flux = v * area_out
                A[i, i] += adv_flux
            else
                adv_flux = abs(v) * area_out
                A[i, i + 1] -= adv_flux
            end
        end

        if source !== nothing
            b[i] += evaluate_source(source, mesh, i) * volume
        end
    end

    return A, b
end

"""
    assemble_system(model::CylindricalDiffusion2D, mesh::Mesh2D, bcs; source=nothing, transient=false)

Assemble the system matrix and RHS for 2D cylindrical diffusion (axisymmetric r-z).
The Laplacian is (1/r) d/dr (r * gamma * dphi/dr) + d/dz (gamma * dphi/dz).
x-coordinate is r, y-coordinate is z.
"""
function assemble_system(model::CylindricalDiffusion2D, mesh::Mesh2D, bcs; source = nothing, transient = false)
    nx = mesh.nx
    ny = mesh.ny
    A = SparseArrays.spzeros(nx * ny, nx * ny)
    b = zeros(nx * ny)
    gamma = model.gamma

    bc_left, bc_right, bc_bottom, bc_top = bcs

    for i in 1:nx
        # Radial positions
        dx = mesh.Lx / nx
        dy = mesh.Ly / ny
        r_in = (i - 1) * dx
        r_out = i * dx

        # Areas
        area_r_in = 2 * pi * r_in * dy
        area_r_out = 2 * pi * r_out * dy
        # Volume: pi * (r_out^2 - r_in^2) * dy
        volume = pi * (r_out^2 - r_in^2) * dy

        # Area for z-faces: pi * (r_out^2 - r_in^2)
        area_z = pi * (r_out^2 - r_in^2)

        for j in 1:ny
            k = (i - 1) * ny + j

            # --- Radial fluxes (x-direction) ---
            # West face (r_in)
            if i == 1
                handle_cylindrical_boundary_condition_2d!(A, b, model, mesh, k, bc_left, :left, area_r_in, dx, transient)
            else
                k_w = k - ny
                dr = dx # uniform mesh assumption for now
                flux_coeff = gamma * area_r_in / dr
                A[k, k] += flux_coeff
                A[k, k_w] -= flux_coeff
            end

            # East face (r_out)
            if i == nx
                handle_cylindrical_boundary_condition_2d!(A, b, model, mesh, k, bc_right, :right, area_r_out, dx, transient)
            else
                k_e = k + ny
                dr = dx
                flux_coeff = gamma * area_r_out / dr
                A[k, k] += flux_coeff
                A[k, k_e] -= flux_coeff
            end

            # --- Axial fluxes (z-direction, y in Mesh2D) ---
            # South face (z_bottom)
            if j == 1
                handle_cylindrical_boundary_condition_2d!(A, b, model, mesh, k, bc_bottom, :bottom, area_z, dy, transient)
            else
                k_s = k - 1
                dz = dy
                flux_coeff = gamma * area_z / dz
                A[k, k] += flux_coeff
                A[k, k_s] -= flux_coeff
            end

            # North face (z_top)
            if j == ny
                handle_cylindrical_boundary_condition_2d!(A, b, model, mesh, k, bc_top, :top, area_z, dy, transient)
            else
                k_n = k + 1
                dz = dy
                flux_coeff = gamma * area_z / dz
                A[k, k] += flux_coeff
                A[k, k_n] -= flux_coeff
            end

            # Source term
            if source !== nothing
                b[k] += evaluate_source(source, mesh, i, j) * volume
            end
        end
    end

    return A, b
end

"""
    assemble_system(model::CylindricalAdvection2D, mesh::Mesh2D, bcs; source=nothing, transient=false)

Assemble system for 2D cylindrical advection (r-z).
"""
function assemble_system(model::CylindricalAdvection2D, mesh::Mesh2D, bcs; source = nothing, transient = false)
    nx = mesh.nx
    ny = mesh.ny
    A = SparseArrays.spzeros(nx * ny, nx * ny)
    b = zeros(nx * ny)

    vr = model.vr
    vz = model.vz
    bc_left, bc_right, bc_bottom, bc_top = bcs

    dx = mesh.Lx / nx
    dy = mesh.Ly / ny

    for i in 1:nx
        r_in = (i - 1) * dx
        r_out = i * dx

        area_r_in = 2 * pi * r_in * dy
        area_r_out = 2 * pi * r_out * dy
        area_z = pi * (r_out^2 - r_in^2)
        volume = area_z * dy

        for j in 1:ny
            k = (i - 1) * ny + j

            # --- Radial Advection (x-direction) ---
            # West face (r_in)
            if i == 1
                handle_cylindrical_advection_bc_2d!(A, b, model, mesh, k, bc_left, :left, area_r_in, dx, transient)
            else
                k_w = k - ny
                if vr >= 0
                    A[k, k_w] -= vr * area_r_in
                else
                    A[k, k] += abs(vr) * area_r_in
                end
            end

            # East face (r_out)
            if i == nx
                handle_cylindrical_advection_bc_2d!(A, b, model, mesh, k, bc_right, :right, area_r_out, dx, transient)
            else
                k_e = k + ny
                if vr >= 0
                    A[k, k] += vr * area_r_out
                else
                    A[k, k_e] -= abs(vr) * area_r_out
                end
            end

            # --- Axial Advection (y-direction) ---
            # South face (z_bottom)
            if j == 1
                handle_cylindrical_advection_bc_2d!(A, b, model, mesh, k, bc_bottom, :bottom, area_z, dy, transient)
            else
                k_s = k - 1
                if vz >= 0
                    A[k, k_s] -= vz * area_z
                else
                    A[k, k] += abs(vz) * area_z
                end
            end

            # North face (z_top)
            if j == ny
                handle_cylindrical_advection_bc_2d!(A, b, model, mesh, k, bc_top, :top, area_z, dy, transient)
            else
                k_n = k + 1
                if vz >= 0
                    A[k, k] += vz * area_z
                else
                    A[k, k_n] -= abs(vz) * area_z
                end
            end

            if source !== nothing
                b[k] += evaluate_source(source, mesh, i, j) * volume
            end
        end
    end

    return A, b
end

"""
    assemble_system(model::CylindricalAdvectionDiffusion2D, mesh::Mesh2D, bcs; source=nothing, transient=false)

Assemble system for 2D cylindrical advection-diffusion.
"""
function assemble_system(model::CylindricalAdvectionDiffusion2D, mesh::Mesh2D, bcs; source = nothing, transient = false)
    nx = mesh.nx
    ny = mesh.ny
    A = SparseArrays.spzeros(nx * ny, nx * ny)
    b = zeros(nx * ny)

    vr = model.advection.vr
    vz = model.advection.vz
    gamma = model.diffusion.gamma

    bc_left, bc_right, bc_bottom, bc_top = bcs

    dx = mesh.Lx / nx
    dy = mesh.Ly / ny

    for i in 1:nx
        r_in = (i - 1) * dx
        r_out = i * dx

        area_r_in = 2 * pi * r_in * dy
        area_r_out = 2 * pi * r_out * dy
        area_z = pi * (r_out^2 - r_in^2)
        volume = area_z * dy

        for j in 1:ny
            k = (i - 1) * ny + j

            # --- Radial (x-direction) ---
            # West face
            if i == 1
                handle_cylindrical_advection_diffusion_bc_2d!(A, b, model, mesh, k, bc_left, :left, area_r_in, dx, transient)
            else
                k_w = k - ny
                dr = dx
                diff_flux = gamma * area_r_in / dr
                A[k, k] += diff_flux
                A[k, k_w] -= diff_flux

                if vr >= 0
                    A[k, k_w] -= vr * area_r_in
                else
                    A[k, k] += abs(vr) * area_r_in
                end
            end

            # East face
            if i == nx
                handle_cylindrical_advection_diffusion_bc_2d!(A, b, model, mesh, k, bc_right, :right, area_r_out, dx, transient)
            else
                k_e = k + ny
                dr = dx
                diff_flux = gamma * area_r_out / dr
                A[k, k] += diff_flux
                A[k, k_e] -= diff_flux

                if vr >= 0
                    A[k, k] += vr * area_r_out
                else
                    A[k, k_e] -= abs(vr) * area_r_out
                end
            end

            # --- Axial (y-direction) ---
            # South face
            if j == 1
                handle_cylindrical_advection_diffusion_bc_2d!(A, b, model, mesh, k, bc_bottom, :bottom, area_z, dy, transient)
            else
                k_s = k - 1
                dz = dy
                diff_flux = gamma * area_z / dz
                A[k, k] += diff_flux
                A[k, k_s] -= diff_flux

                if vz >= 0
                    A[k, k_s] -= vz * area_z
                else
                    A[k, k] += abs(vz) * area_z
                end
            end

            # North face
            if j == ny
                handle_cylindrical_advection_diffusion_bc_2d!(A, b, model, mesh, k, bc_top, :top, area_z, dy, transient)
            else
                k_n = k + 1
                dz = dy
                diff_flux = gamma * area_z / dz
                A[k, k] += diff_flux
                A[k, k_n] -= diff_flux

                if vz >= 0
                    A[k, k] += vz * area_z
                else
                    A[k, k_n] -= abs(vz) * area_z
                end
            end

            if source !== nothing
                b[k] += evaluate_source(source, mesh, i, j) * volume
            end
        end
    end

    return A, b
end

"""
    assemble_mass_matrix(mesh::Mesh1D, model::CylindricalDiffusion1D)

Assemble the mass matrix for 1D cylindrical coordinates.
"""
function assemble_mass_matrix(mesh::Mesh1D, model::CylindricalDiffusion1D)
    nx = length(mesh.cells)
    M = SparseArrays.spzeros(nx, nx)
    for i in 1:nx
        r_in = mesh.nodes[i].x
        r_out = mesh.nodes[i + 1].x
        volume = pi * (r_out^2 - r_in^2)
        M[i, i] = volume
    end
    return M
end

"""
    assemble_mass_matrix(mesh::Mesh2D, model::CylindricalDiffusion2D)

Assemble the mass matrix for 2D cylindrical coordinates (axisymmetric r-z).
"""
function assemble_mass_matrix(mesh::Mesh2D, model::CylindricalDiffusion2D)
    nx = mesh.nx
    ny = mesh.ny
    M = SparseArrays.spzeros(nx * ny, nx * ny)
    dx = mesh.Lx / nx
    dy = mesh.Ly / ny
    for i in 1:nx
        r_in = (i - 1) * dx
        r_out = i * dx
        volume = pi * (r_out^2 - r_in^2) * dy
        for j in 1:ny
            k = (i - 1) * ny + j
            M[k, k] = volume
        end
    end
    return M
end

# --- Boundary Condition Handlers for Cylindrical Coordinates ---

function handle_cylindrical_boundary_condition!(A, b, model::CylindricalDiffusion1D, mesh, i, bc::ParabolicDirichlet, side, area, transient)
    dx = mesh.cells[i].center - (side == :left ? mesh.nodes[i].x : mesh.nodes[i + 1].x)
    flux_coeff = model.gamma * area / abs(dx)
    A[i, i] += flux_coeff
    return b[i] += flux_coeff * bc.value
end

function handle_cylindrical_boundary_condition!(A, b, model::CylindricalDiffusion1D, mesh, i, bc::ParabolicNeumann, side, area, transient)
    # Neumann BC: flux = -gamma * dphi/dn.
    # Total flux = flux * area.
    # In b vector, we add the inward flux.
    return if side == :left
        b[i] -= bc.value * area
    else
        b[i] += bc.value * area
    end
end

function handle_cylindrical_boundary_condition!(A, b, model::CylindricalDiffusion1D, mesh, i, bc::ParabolicRobin, side, area, transient)
    dx = abs(mesh.cells[i].center - (side == :left ? mesh.nodes[i].x : mesh.nodes[i + 1].x))
    gamma = model.gamma

    denominator = bc.a * dx + bc.b * gamma
    flux_coeff = gamma * bc.a * area / denominator
    A[i, i] += flux_coeff
    return if side == :left
        b[i] += gamma * bc.c * area / denominator
    else
        b[i] -= gamma * bc.c * area / denominator
    end
end

function handle_cylindrical_boundary_condition_2d!(A, b, model::CylindricalDiffusion2D, mesh, k, bc::ParabolicDirichlet, side, area, dr_or_dz, transient)
    # Distance from cell center to boundary is roughly half cell size
    dist = dr_or_dz / 2.0
    flux_coeff = model.gamma * area / dist
    A[k, k] += flux_coeff
    return b[k] += flux_coeff * bc.value
end

function handle_cylindrical_boundary_condition_2d!(A, b, model::CylindricalDiffusion2D, mesh, k, bc::ParabolicNeumann, side, area, dr_or_dz, transient)
    return if side == :left || side == :bottom
        b[k] -= bc.value * area
    else
        b[k] += bc.value * area
    end
end

function handle_cylindrical_boundary_condition_2d!(A, b, model::CylindricalDiffusion2D, mesh, k, bc::ParabolicRobin, side, area, dr_or_dz, transient)
    dist = dr_or_dz / 2.0
    gamma = model.gamma
    denominator = bc.a * dist + bc.b * gamma
    flux_coeff = gamma * bc.a * area / denominator
    A[k, k] += flux_coeff
    return if side == :left || side == :bottom
        b[k] += gamma * bc.c * area / denominator
    else
        b[k] -= gamma * bc.c * area / denominator
    end
end

# --- Cylindrical Advection Boundary Condition Handlers ---

function handle_cylindrical_advection_bc!(A, b, model::CylindricalAdvection1D, mesh, i, bc::ParabolicDirichlet, side, area, transient)
    v = model.v
    return if side == :left
        if v >= 0
            # Inlet: flux = v * phi_bc * area
            b[i] += v * bc.value * area
        else
            # Outlet: flux = |v| * phi_i * area
            A[i, i] += abs(v) * area
        end
    else # side == :right
        if v >= 0
            # Outlet
            A[i, i] += v * area
        else
            # Inlet
            b[i] += abs(v) * bc.value * area
        end
    end
end

function handle_cylindrical_advection_bc!(A, b, model::CylindricalAdvection1D, mesh, i, bc::ParabolicNeumann, side, area, transient)
    v = model.v
    return if side == :left
        if v >= 0
            b[i] += bc.value * area
        else
            A[i, i] += abs(v) * area
        end
    else # side == :right
        if v >= 0
            A[i, i] += v * area
        else
            b[i] += bc.value * area
        end
    end
end

function handle_cylindrical_advection_bc!(A, b, model::CylindricalAdvection1D, mesh, i, bc::OutflowBC, side, area, transient)
    v = model.v
    return if side == :left
        if v < 0
            A[i, i] += abs(v) * area
        end
    else
        if v >= 0
            A[i, i] += v * area
        end
    end
end

function handle_cylindrical_advection_bc_2d!(A, b, model::CylindricalAdvection2D, mesh, k, bc::ParabolicDirichlet, side, area, dr_or_dz, transient)
    if side == :left || side == :right
        v = model.vr
    else
        v = model.vz
    end

    return if side == :left || side == :bottom
        if v >= 0
            b[k] += v * bc.value * area
        else
            A[k, k] += abs(v) * area
        end
    else # right or top
        if v >= 0
            A[k, k] += v * area
        else
            b[k] += abs(v) * bc.value * area
        end
    end
end

function handle_cylindrical_advection_bc_2d!(A, b, model::CylindricalAdvection2D, mesh, k, bc::OutflowBC, side, area, dr_or_dz, transient)
    if side == :left || side == :right
        v = model.vr
    else
        v = model.vz
    end

    is_outlet = (side == :left && v < 0) || (side == :right && v >= 0) ||
        (side == :bottom && v < 0) || (side == :top && v >= 0)

    return if is_outlet
        A[k, k] += abs(v) * area
    end
end

function handle_cylindrical_advection_bc_2d!(A, b, model::CylindricalAdvection2D, mesh, k, bc::ParabolicNeumann, side, area, dr_or_dz, transient)
    if side == :left || side == :right
        v = model.vr
    else
        v = model.vz
    end

    return if side == :left || side == :bottom
        if v >= 0
            b[k] += bc.value * area
        else
            A[k, k] += abs(v) * area
        end
    else # right or top
        if v >= 0
            A[k, k] += v * area
        else
            b[k] += bc.value * area
        end
    end
end

# --- Cylindrical Advection-Diffusion Boundary Condition Handlers ---

function handle_cylindrical_advection_diffusion_bc!(A, b, model::CylindricalAdvectionDiffusion1D, mesh, i, bc::ParabolicDirichlet, side, area, transient)
    v = model.advection.v
    gamma = model.diffusion.gamma
    dx = abs(mesh.cells[i].center - (side == :left ? mesh.nodes[i].x : mesh.nodes[i + 1].x))

    diff_flux_coeff = gamma * area / dx
    A[i, i] += diff_flux_coeff
    b[i] += diff_flux_coeff * bc.value

    return if side == :left
        if v >= 0
            b[i] += v * bc.value * area
        else
            A[i, i] += abs(v) * area
        end
    else
        if v >= 0
            A[i, i] += v * area
        else
            b[i] += abs(v) * bc.value * area
        end
    end
end

function handle_cylindrical_advection_diffusion_bc!(A, b, model::CylindricalAdvectionDiffusion1D, mesh, i, bc::ParabolicNeumann, side, area, transient)
    v = model.advection.v
    if side == :left
        b[i] -= bc.value * area
    else
        b[i] += bc.value * area
    end

    return if side == :left
        if v < 0 # Outlet
            A[i, i] += abs(v) * area
        end
    else
        if v >= 0 # Outlet
            A[i, i] += v * area
        end
    end
end

function handle_cylindrical_advection_diffusion_bc_2d!(A, b, model::CylindricalAdvectionDiffusion2D, mesh, k, bc::ParabolicDirichlet, side, area, dr_or_dz, transient)
    if side == :left || side == :right
        v = model.advection.vr
    else
        v = model.advection.vz
    end
    gamma = model.diffusion.gamma
    dist = dr_or_dz / 2.0

    diff_flux_coeff = gamma * area / dist
    A[k, k] += diff_flux_coeff
    b[k] += diff_flux_coeff * bc.value

    return if side == :left || side == :bottom
        if v >= 0
            b[k] += v * bc.value * area
        else
            A[k, k] += abs(v) * area
        end
    else
        if v >= 0
            A[k, k] += v * area
        else
            b[k] += abs(v) * bc.value * area
        end
    end
end

# --- Mass Matrix Overloads for Advection Models ---

function assemble_mass_matrix(mesh::Mesh1D, model::CylindricalAdvection1D)
    return assemble_mass_matrix(mesh, CylindricalDiffusion1D(0.0))
end

function assemble_mass_matrix(mesh::Mesh1D, model::CylindricalAdvectionDiffusion1D)
    return assemble_mass_matrix(mesh, CylindricalDiffusion1D(0.0))
end

function assemble_mass_matrix(mesh::Mesh2D, model::CylindricalAdvection2D)
    return assemble_mass_matrix(mesh, CylindricalDiffusion2D(0.0))
end

function assemble_mass_matrix(mesh::Mesh2D, model::CylindricalAdvectionDiffusion2D)
    return assemble_mass_matrix(mesh, CylindricalDiffusion2D(0.0))
end
