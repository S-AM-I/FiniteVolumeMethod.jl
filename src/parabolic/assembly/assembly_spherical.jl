# Assembly routines for spherical coordinates
# Migrated from Simu.jl SimuFVM/assembly/assembly_spherical.jl

"""
    assemble_system(model::SphericalDiffusion1D, mesh::Mesh1D, bc_left, bc_right; source=nothing, transient=false)

Assemble the system matrix and RHS for 1D spherical diffusion (radial only).
The radial Laplacian is (1/r^2) d/dr (r^2 * gamma * dphi/dr).
Volume of shell: 4/3 * pi * (r_out^3 - r_in^3).
Area: 4pi * r^2.
"""
function assemble_system(model::SphericalDiffusion1D, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition; source = nothing, transient = false)
    nx = length(mesh.cells)
    A = SparseArrays.spzeros(nx, nx)
    b = zeros(nx)
    gamma = model.gamma

    for i in 1:nx
        r_in = mesh.nodes[i].x
        r_out = mesh.nodes[i + 1].x
        dr = r_out - r_in

        # Cell volume: V = 4/3 * pi * (r_out^3 - r_in^3)
        volume = (4.0 / 3.0) * pi * (r_out^3 - r_in^3)

        # Left face (r_in)
        if i == 1
            # Boundary condition at r_min
            # Area at r_in: 4pi * r_in^2
            area_in = 4 * pi * r_in^2
            handle_spherical_boundary_condition!(A, b, model, mesh, i, bc_left, :left, area_in, transient)
        else
            # Internal face
            r_face = r_in
            area_face = 4 * pi * r_face^2
            dr_face = mesh.cells[i].center - mesh.cells[i - 1].center
            flux_coeff = gamma * area_face / dr_face
            A[i, i] += flux_coeff
            A[i, i - 1] -= flux_coeff
        end

        # Right face (r_out)
        if i == nx
            # Boundary condition at r_max
            # Area at r_out: 4pi * r_out^2
            area_out = 4 * pi * r_out^2
            handle_spherical_boundary_condition!(A, b, model, mesh, i, bc_right, :right, area_out, transient)
        else
            # Internal face
            r_face = r_out
            area_face = 4 * pi * r_face^2
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
    assemble_system(model::SphericalAdvection1D, mesh::Mesh1D, bc_left, bc_right; source=nothing, transient=false)

Assemble system for 1D spherical advection (radial).
Flux = v * phi * Area. Area = 4 * pi * r^2.
"""
function assemble_system(model::SphericalAdvection1D, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition; source = nothing, transient = false)
    nx = length(mesh.cells)
    A = SparseArrays.spzeros(nx, nx)
    b = zeros(nx)
    v = model.v

    for i in 1:nx
        r_in = mesh.nodes[i].x
        r_out = mesh.nodes[i + 1].x

        # Cell volume: V = 4/3 * pi * (r_out^3 - r_in^3)
        volume = (4.0 / 3.0) * pi * (r_out^3 - r_in^3)

        # Left face (r_in)
        area_in = 4 * pi * r_in^2
        if i == 1
            handle_spherical_advection_bc!(A, b, model, mesh, i, bc_left, :left, area_in, transient)
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
        area_out = 4 * pi * r_out^2
        if i == nx
            handle_spherical_advection_bc!(A, b, model, mesh, i, bc_right, :right, area_out, transient)
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
    assemble_system(model::SphericalAdvectionDiffusion1D, mesh::Mesh1D, bc_left, bc_right; source=nothing, transient=false)

Assemble system for 1D spherical advection-diffusion.
"""
function assemble_system(model::SphericalAdvectionDiffusion1D, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition; source = nothing, transient = false)
    nx = length(mesh.cells)
    A = SparseArrays.spzeros(nx, nx)
    b = zeros(nx)

    v = model.advection.v
    gamma = model.diffusion.gamma

    for i in 1:nx
        r_in = mesh.nodes[i].x
        r_out = mesh.nodes[i + 1].x
        volume = (4.0 / 3.0) * pi * (r_out^3 - r_in^3)

        # Left face
        area_in = 4 * pi * r_in^2
        if i == 1
            handle_spherical_advection_diffusion_bc!(A, b, model, mesh, i, bc_left, :left, area_in, transient)
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
        area_out = 4 * pi * r_out^2
        if i == nx
            handle_spherical_advection_diffusion_bc!(A, b, model, mesh, i, bc_right, :right, area_out, transient)
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
    assemble_mass_matrix(mesh::Mesh1D, model::SphericalDiffusion1D)

Assemble the mass matrix for 1D spherical coordinates.
"""
function assemble_mass_matrix(mesh::Mesh1D, model::SphericalDiffusion1D)
    nx = length(mesh.cells)
    M = SparseArrays.spzeros(nx, nx)
    for i in 1:nx
        r_in = mesh.nodes[i].x
        r_out = mesh.nodes[i + 1].x
        volume = (4.0 / 3.0) * pi * (r_out^3 - r_in^3)
        M[i, i] = volume
    end
    return M
end

function handle_spherical_boundary_condition!(A, b, model::SphericalDiffusion1D, mesh, i, bc::DirichletBC, side, area, transient)
    dx = abs(mesh.cells[i].center - (side == :left ? mesh.nodes[i].x : mesh.nodes[i + 1].x))
    return if dx > 1.0e-12 && area > 1.0e-12
        flux_coeff = model.gamma * area / dx
        A[i, i] += flux_coeff
        b[i] += flux_coeff * bc.value
    end
end

function handle_spherical_boundary_condition!(A, b, model::SphericalDiffusion1D, mesh, i, bc::NeumannBC, side, area, transient)
    return if side == :left
        b[i] -= bc.value * area
    else
        b[i] += bc.value * area
    end
end

function handle_spherical_boundary_condition!(A, b, model::SphericalDiffusion1D, mesh, i, bc::RobinBC, side, area, transient)
    dx = abs(mesh.cells[i].center - (side == :left ? mesh.nodes[i].x : mesh.nodes[i + 1].x))
    gamma = model.gamma

    denominator = bc.a * dx + bc.b * gamma
    return if abs(denominator) > 1.0e-12
        flux_coeff = gamma * bc.a * area / denominator
        A[i, i] += flux_coeff
        if side == :left
            b[i] += gamma * bc.c * area / denominator
        else
            b[i] -= gamma * bc.c * area / denominator
        end
    end
end

# --- Spherical Advection Boundary Condition Handlers ---

function handle_spherical_advection_bc!(A, b, model::SphericalAdvection1D, mesh, i, bc::DirichletBC, side, area, transient)
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

function handle_spherical_advection_bc!(A, b, model::SphericalAdvection1D, mesh, i, bc::NeumannBC, side, area, transient)
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

function handle_spherical_advection_bc!(A, b, model::SphericalAdvection1D, mesh, i, bc::OutflowBC, side, area, transient)
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

# --- Spherical Advection-Diffusion Boundary Condition Handlers ---

function handle_spherical_advection_diffusion_bc!(A, b, model::SphericalAdvectionDiffusion1D, mesh, i, bc::DirichletBC, side, area, transient)
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

function handle_spherical_advection_diffusion_bc!(A, b, model::SphericalAdvectionDiffusion1D, mesh, i, bc::NeumannBC, side, area, transient)
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

# --- Mass Matrix Overloads ---

function assemble_mass_matrix(mesh::Mesh1D, model::SphericalAdvection1D)
    return assemble_mass_matrix(mesh, SphericalDiffusion1D(0.0))
end

function assemble_mass_matrix(mesh::Mesh1D, model::SphericalAdvectionDiffusion1D)
    return assemble_mass_matrix(mesh, SphericalDiffusion1D(0.0))
end
