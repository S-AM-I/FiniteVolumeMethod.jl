# Assembly routines for cylindrical coordinates
# Migrated from Simu.jl SimuFVM/assembly/assembly_cylindrical.jl

# 2D node access. `generate_mesh_2d` builds nodes via
# `[Node2D(x, y) for i in 0:nx, j in 0:ny]` then `vec(...)`, which is column-
# major: nodes[i, j] -> mesh.nodes[i + (j-1)*(nx+1)]. Reading geometry from
# nodes (rather than computing from index + Lx/nx) supports r_inner > 0 and
# non-uniform meshes.
@inline _node2d(mesh::Mesh2D, i::Int, j::Int) = mesh.nodes[i + (j - 1) * (mesh.nx + 1)]

# Cell-local γ lookup so the BC handlers can serve both the constant-γ and
# variable-γ models without branching.
@inline _gamma_at_cell(model::CylindricalDiffusion1D, mesh::Mesh1D, i::Int) = model.gamma
@inline _gamma_at_cell(model::VariableCylindricalDiffusion1D, mesh::Mesh1D, i::Int) =
    get_diffusion_coefficient(model, mesh, i)
@inline _gamma_at_cell(model::CylindricalDiffusion2D, mesh::Mesh2D, k::Int) = model.gamma
@inline function _gamma_at_cell(model::VariableCylindricalDiffusion2D, mesh::Mesh2D, k::Int)
    j = mod(k - 1, mesh.ny) + 1
    i = div(k - 1, mesh.ny) + 1
    return get_diffusion_coefficient(model, mesh, i, j)
end

# Face γ via harmonic mean of adjacent cell γ values (matches the Cartesian
# `VariableDiffusion*` pattern in utils.jl).
@inline _gamma_at_face_1d(model::CylindricalDiffusion1D, mesh::Mesh1D, i, side) = model.gamma
@inline function _gamma_at_face_1d(model::VariableCylindricalDiffusion1D, mesh::Mesh1D, i, side)
    if side == :left
        i == 1 && return get_diffusion_coefficient(model, mesh, i)
        return _harmonic_mean(get_diffusion_coefficient(model, mesh, i - 1),
                              get_diffusion_coefficient(model, mesh, i))
    else
        i == length(mesh.cells) && return get_diffusion_coefficient(model, mesh, i)
        return _harmonic_mean(get_diffusion_coefficient(model, mesh, i),
                              get_diffusion_coefficient(model, mesh, i + 1))
    end
end

@inline _gamma_at_face_2d(model::CylindricalDiffusion2D, mesh::Mesh2D, i, j, side) = model.gamma
@inline function _gamma_at_face_2d(model::VariableCylindricalDiffusion2D, mesh::Mesh2D, i, j, side)
    nx = mesh.nx; ny = mesh.ny
    γP = get_diffusion_coefficient(model, mesh, i, j)
    if side == :left
        i == 1 && return γP
        return _harmonic_mean(get_diffusion_coefficient(model, mesh, i - 1, j), γP)
    elseif side == :right
        i == nx && return γP
        return _harmonic_mean(γP, get_diffusion_coefficient(model, mesh, i + 1, j))
    elseif side == :bottom
        j == 1 && return γP
        return _harmonic_mean(get_diffusion_coefficient(model, mesh, i, j - 1), γP)
    else # :top
        j == ny && return γP
        return _harmonic_mean(γP, get_diffusion_coefficient(model, mesh, i, j + 1))
    end
end

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
    assemble_system(model::VariableCylindricalDiffusion1D, mesh::Mesh1D, bc_left, bc_right; source=nothing, transient=false)

1D cylindrical diffusion with spatially varying `gamma(r)`. Face γ uses the
harmonic mean of the two adjacent cell γ values; BCs are dispatched through
the same handlers as the constant-coefficient case via the
`_CylDiffusion1D` Union.
"""
function assemble_system(model::VariableCylindricalDiffusion1D, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition; source = nothing, transient = false)
    nx = length(mesh.cells)
    A = SparseArrays.spzeros(nx, nx)
    b = zeros(nx)

    for i in 1:nx
        r_in = mesh.nodes[i].x
        r_out = mesh.nodes[i + 1].x
        volume = pi * (r_out^2 - r_in^2)

        if i == 1
            area_in = 2 * pi * r_in
            handle_cylindrical_boundary_condition!(A, b, model, mesh, i, bc_left, :left, area_in, transient)
        else
            area_face = 2 * pi * r_in
            dr_face = mesh.cells[i].center - mesh.cells[i - 1].center
            γ = _gamma_at_face_1d(model, mesh, i, :left)
            flux_coeff = γ * area_face / dr_face
            A[i, i] += flux_coeff
            A[i, i - 1] -= flux_coeff
        end

        if i == nx
            area_out = 2 * pi * r_out
            handle_cylindrical_boundary_condition!(A, b, model, mesh, i, bc_right, :right, area_out, transient)
        else
            area_face = 2 * pi * r_out
            dr_face = mesh.cells[i + 1].center - mesh.cells[i].center
            γ = _gamma_at_face_1d(model, mesh, i, :right)
            flux_coeff = γ * area_face / dr_face
            A[i, i] += flux_coeff
            A[i, i + 1] -= flux_coeff
        end

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
        for j in 1:ny
            k = (i - 1) * ny + j

            r_in  = _node2d(mesh, i,     j    ).x
            r_out = _node2d(mesh, i + 1, j    ).x
            z_lo  = _node2d(mesh, i,     j    ).y
            z_hi  = _node2d(mesh, i,     j + 1).y
            dr = r_out - r_in
            dz = z_hi - z_lo

            area_r_in  = 2 * pi * r_in  * dz
            area_r_out = 2 * pi * r_out * dz
            area_z     = pi * (r_out^2 - r_in^2)
            volume     = area_z * dz

            # --- Radial fluxes (x-direction) ---
            if i == 1
                handle_cylindrical_boundary_condition_2d!(A, b, model, mesh, k, bc_left, :left, area_r_in, dr, transient)
            else
                k_w = k - ny
                dr_face = mesh.cells[k].center[1] - mesh.cells[k_w].center[1]
                flux_coeff = gamma * area_r_in / dr_face
                A[k, k] += flux_coeff
                A[k, k_w] -= flux_coeff
            end

            if i == nx
                handle_cylindrical_boundary_condition_2d!(A, b, model, mesh, k, bc_right, :right, area_r_out, dr, transient)
            else
                k_e = k + ny
                dr_face = mesh.cells[k_e].center[1] - mesh.cells[k].center[1]
                flux_coeff = gamma * area_r_out / dr_face
                A[k, k] += flux_coeff
                A[k, k_e] -= flux_coeff
            end

            # --- Axial fluxes (y-direction = z) ---
            if j == 1
                handle_cylindrical_boundary_condition_2d!(A, b, model, mesh, k, bc_bottom, :bottom, area_z, dz, transient)
            else
                k_s = k - 1
                dz_face = mesh.cells[k].center[2] - mesh.cells[k_s].center[2]
                flux_coeff = gamma * area_z / dz_face
                A[k, k] += flux_coeff
                A[k, k_s] -= flux_coeff
            end

            if j == ny
                handle_cylindrical_boundary_condition_2d!(A, b, model, mesh, k, bc_top, :top, area_z, dz, transient)
            else
                k_n = k + 1
                dz_face = mesh.cells[k_n].center[2] - mesh.cells[k].center[2]
                flux_coeff = gamma * area_z / dz_face
                A[k, k] += flux_coeff
                A[k, k_n] -= flux_coeff
            end

            if source !== nothing
                b[k] += evaluate_source(source, mesh, i, j) * volume
            end
        end
    end

    return A, b
end

"""
    assemble_system(model::VariableCylindricalDiffusion2D, mesh::Mesh2D, bcs; source=nothing, transient=false)

2D axisymmetric diffusion with spatially varying `gamma(r, z)`. Face γ uses
the harmonic mean of the two adjacent cell γ values; BCs are dispatched
through the same handlers as the constant-coefficient case via the
`_CylDiffusion2D` Union.
"""
function assemble_system(model::VariableCylindricalDiffusion2D, mesh::Mesh2D, bcs; source = nothing, transient = false)
    nx = mesh.nx
    ny = mesh.ny
    A = SparseArrays.spzeros(nx * ny, nx * ny)
    b = zeros(nx * ny)

    bc_left, bc_right, bc_bottom, bc_top = bcs

    for i in 1:nx
        for j in 1:ny
            k = (i - 1) * ny + j

            r_in  = _node2d(mesh, i,     j    ).x
            r_out = _node2d(mesh, i + 1, j    ).x
            z_lo  = _node2d(mesh, i,     j    ).y
            z_hi  = _node2d(mesh, i,     j + 1).y
            dr = r_out - r_in
            dz = z_hi - z_lo

            area_r_in  = 2 * pi * r_in  * dz
            area_r_out = 2 * pi * r_out * dz
            area_z     = pi * (r_out^2 - r_in^2)
            volume     = area_z * dz

            if i == 1
                handle_cylindrical_boundary_condition_2d!(A, b, model, mesh, k, bc_left, :left, area_r_in, dr, transient)
            else
                k_w = k - ny
                dr_face = mesh.cells[k].center[1] - mesh.cells[k_w].center[1]
                γ = _gamma_at_face_2d(model, mesh, i, j, :left)
                flux_coeff = γ * area_r_in / dr_face
                A[k, k] += flux_coeff
                A[k, k_w] -= flux_coeff
            end

            if i == nx
                handle_cylindrical_boundary_condition_2d!(A, b, model, mesh, k, bc_right, :right, area_r_out, dr, transient)
            else
                k_e = k + ny
                dr_face = mesh.cells[k_e].center[1] - mesh.cells[k].center[1]
                γ = _gamma_at_face_2d(model, mesh, i, j, :right)
                flux_coeff = γ * area_r_out / dr_face
                A[k, k] += flux_coeff
                A[k, k_e] -= flux_coeff
            end

            if j == 1
                handle_cylindrical_boundary_condition_2d!(A, b, model, mesh, k, bc_bottom, :bottom, area_z, dz, transient)
            else
                k_s = k - 1
                dz_face = mesh.cells[k].center[2] - mesh.cells[k_s].center[2]
                γ = _gamma_at_face_2d(model, mesh, i, j, :bottom)
                flux_coeff = γ * area_z / dz_face
                A[k, k] += flux_coeff
                A[k, k_s] -= flux_coeff
            end

            if j == ny
                handle_cylindrical_boundary_condition_2d!(A, b, model, mesh, k, bc_top, :top, area_z, dz, transient)
            else
                k_n = k + 1
                dz_face = mesh.cells[k_n].center[2] - mesh.cells[k].center[2]
                γ = _gamma_at_face_2d(model, mesh, i, j, :top)
                flux_coeff = γ * area_z / dz_face
                A[k, k] += flux_coeff
                A[k, k_n] -= flux_coeff
            end

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

    for i in 1:nx
        for j in 1:ny
            k = (i - 1) * ny + j

            r_in  = _node2d(mesh, i,     j    ).x
            r_out = _node2d(mesh, i + 1, j    ).x
            z_lo  = _node2d(mesh, i,     j    ).y
            z_hi  = _node2d(mesh, i,     j + 1).y
            dr = r_out - r_in
            dz = z_hi - z_lo

            area_r_in  = 2 * pi * r_in  * dz
            area_r_out = 2 * pi * r_out * dz
            area_z     = pi * (r_out^2 - r_in^2)
            volume     = area_z * dz

            # --- Radial Advection (x-direction) ---
            if i == 1
                handle_cylindrical_advection_bc_2d!(A, b, model, mesh, k, bc_left, :left, area_r_in, dr, transient)
            else
                k_w = k - ny
                if vr >= 0
                    A[k, k_w] -= vr * area_r_in
                else
                    A[k, k] += abs(vr) * area_r_in
                end
            end

            if i == nx
                handle_cylindrical_advection_bc_2d!(A, b, model, mesh, k, bc_right, :right, area_r_out, dr, transient)
            else
                k_e = k + ny
                if vr >= 0
                    A[k, k] += vr * area_r_out
                else
                    A[k, k_e] -= abs(vr) * area_r_out
                end
            end

            # --- Axial Advection (y-direction) ---
            if j == 1
                handle_cylindrical_advection_bc_2d!(A, b, model, mesh, k, bc_bottom, :bottom, area_z, dz, transient)
            else
                k_s = k - 1
                if vz >= 0
                    A[k, k_s] -= vz * area_z
                else
                    A[k, k] += abs(vz) * area_z
                end
            end

            if j == ny
                handle_cylindrical_advection_bc_2d!(A, b, model, mesh, k, bc_top, :top, area_z, dz, transient)
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

    for i in 1:nx
        for j in 1:ny
            k = (i - 1) * ny + j

            r_in  = _node2d(mesh, i,     j    ).x
            r_out = _node2d(mesh, i + 1, j    ).x
            z_lo  = _node2d(mesh, i,     j    ).y
            z_hi  = _node2d(mesh, i,     j + 1).y
            dr = r_out - r_in
            dz = z_hi - z_lo

            area_r_in  = 2 * pi * r_in  * dz
            area_r_out = 2 * pi * r_out * dz
            area_z     = pi * (r_out^2 - r_in^2)
            volume     = area_z * dz

            # --- Radial (x-direction) ---
            if i == 1
                handle_cylindrical_advection_diffusion_bc_2d!(A, b, model, mesh, k, bc_left, :left, area_r_in, dr, transient)
            else
                k_w = k - ny
                dr_face = mesh.cells[k].center[1] - mesh.cells[k_w].center[1]
                diff_flux = gamma * area_r_in / dr_face
                A[k, k] += diff_flux
                A[k, k_w] -= diff_flux

                if vr >= 0
                    A[k, k_w] -= vr * area_r_in
                else
                    A[k, k] += abs(vr) * area_r_in
                end
            end

            if i == nx
                handle_cylindrical_advection_diffusion_bc_2d!(A, b, model, mesh, k, bc_right, :right, area_r_out, dr, transient)
            else
                k_e = k + ny
                dr_face = mesh.cells[k_e].center[1] - mesh.cells[k].center[1]
                diff_flux = gamma * area_r_out / dr_face
                A[k, k] += diff_flux
                A[k, k_e] -= diff_flux

                if vr >= 0
                    A[k, k] += vr * area_r_out
                else
                    A[k, k_e] -= abs(vr) * area_r_out
                end
            end

            # --- Axial (y-direction) ---
            if j == 1
                handle_cylindrical_advection_diffusion_bc_2d!(A, b, model, mesh, k, bc_bottom, :bottom, area_z, dz, transient)
            else
                k_s = k - 1
                dz_face = mesh.cells[k].center[2] - mesh.cells[k_s].center[2]
                diff_flux = gamma * area_z / dz_face
                A[k, k] += diff_flux
                A[k, k_s] -= diff_flux

                if vz >= 0
                    A[k, k_s] -= vz * area_z
                else
                    A[k, k] += abs(vz) * area_z
                end
            end

            if j == ny
                handle_cylindrical_advection_diffusion_bc_2d!(A, b, model, mesh, k, bc_top, :top, area_z, dz, transient)
            else
                k_n = k + 1
                dz_face = mesh.cells[k_n].center[2] - mesh.cells[k].center[2]
                diff_flux = gamma * area_z / dz_face
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
    for i in 1:nx
        for j in 1:ny
            k = (i - 1) * ny + j
            r_in  = _node2d(mesh, i,     j    ).x
            r_out = _node2d(mesh, i + 1, j    ).x
            z_lo  = _node2d(mesh, i,     j    ).y
            z_hi  = _node2d(mesh, i,     j + 1).y
            dz = z_hi - z_lo
            M[k, k] = pi * (r_out^2 - r_in^2) * dz
        end
    end
    return M
end

# --- Boundary Condition Handlers for Cylindrical Coordinates ---

const _CylDiffusion1D = Union{CylindricalDiffusion1D, VariableCylindricalDiffusion1D}

function handle_cylindrical_boundary_condition!(A, b, model::_CylDiffusion1D, mesh, i, bc::ParabolicDirichlet, side, area, transient)
    dx = mesh.cells[i].center - (side == :left ? mesh.nodes[i].x : mesh.nodes[i + 1].x)
    γ = _gamma_at_cell(model, mesh, i)
    flux_coeff = γ * area / abs(dx)
    A[i, i] += flux_coeff
    return b[i] += flux_coeff * bc.value
end

function handle_cylindrical_boundary_condition!(A, b, model::_CylDiffusion1D, mesh, i, bc::ParabolicNeumann, side, area, transient)
    # Neumann BC: flux = -gamma * dphi/dn.
    # Total flux = flux * area.
    # In b vector, we add the inward flux.
    return if side == :left
        b[i] -= bc.value * area
    else
        b[i] += bc.value * area
    end
end

function handle_cylindrical_boundary_condition!(A, b, model::_CylDiffusion1D, mesh, i, bc::ParabolicRobin, side, area, transient)
    dx = abs(mesh.cells[i].center - (side == :left ? mesh.nodes[i].x : mesh.nodes[i + 1].x))
    γ = _gamma_at_cell(model, mesh, i)

    denominator = bc.a * dx + bc.b * γ
    flux_coeff = γ * bc.a * area / denominator
    A[i, i] += flux_coeff
    return b[i] += γ * bc.c * area / denominator
end

const _CylDiffusion2D = Union{CylindricalDiffusion2D, VariableCylindricalDiffusion2D}

function handle_cylindrical_boundary_condition_2d!(A, b, model::_CylDiffusion2D, mesh, k, bc::ParabolicDirichlet, side, area, dr_or_dz, transient)
    dist = dr_or_dz / 2.0
    γ = _gamma_at_cell(model, mesh, k)
    flux_coeff = γ * area / dist
    A[k, k] += flux_coeff
    return b[k] += flux_coeff * bc.value
end

function handle_cylindrical_boundary_condition_2d!(A, b, model::_CylDiffusion2D, mesh, k, bc::ParabolicNeumann, side, area, dr_or_dz, transient)
    return if side == :left || side == :bottom
        b[k] -= bc.value * area
    else
        b[k] += bc.value * area
    end
end

function handle_cylindrical_boundary_condition_2d!(A, b, model::_CylDiffusion2D, mesh, k, bc::ParabolicRobin, side, area, dr_or_dz, transient)
    dist = dr_or_dz / 2.0
    γ = _gamma_at_cell(model, mesh, k)
    denominator = bc.a * dist + bc.b * γ
    flux_coeff = γ * bc.a * area / denominator
    A[k, k] += flux_coeff
    return b[k] += γ * bc.c * area / denominator
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
