# Assembly routines for 1D problems
# Migrated from Simu.jl SimuFVM/assembly/assembly_1d.jl
# PeriodicBC -> StructuredPeriodicBC

"""
    calculate_flux(diffusion, phi_L, phi_R, dx)

Calculate the diffusive flux between two cells.
"""
function calculate_flux(diffusion::Diffusion1D, phi_L, phi_R, dx)
    return diffusion.gamma * (phi_R - phi_L) / dx
end

function calculate_flux(diffusion::Diffusion1D, phi_L2, phi_L, phi_R, phi_R2, dx_L, dx, dx_R)
    return diffusion.gamma * (phi_R - phi_L) / dx
end

function calculate_flux(advection::Advection1D, phi_L, phi_R, v_face)
    if v_face >= 0
        return v_face * phi_L
    else
        return v_face * phi_R
    end
end

function calculate_flux(advection::Union{Advection1D, VariableAdvection1D}, mesh::Mesh1D, phi, i::Int, direction::Symbol)
    v = get_velocity(advection, mesh, i, direction)

    if advection.scheme == :muscl
        limiter = :minmod
        return muscl_advection_flux_1d(advection, mesh, phi, i, direction, limiter)
    elseif advection.scheme == :quick
        return quick_advection_flux_1d(advection, mesh, phi, i, direction)
    else
        # Default: upwind
        if direction == :left
            phi_L = i > 1 ? phi[i - 1] : phi[i]
            phi_R = phi[i]
            if v >= 0
                return v * phi_L
            else
                return v * phi_R
            end
        else # direction == :right
            phi_L = phi[i]
            phi_R = i < length(phi) ? phi[i + 1] : phi[i]
            if v >= 0
                return v * phi_L
            else
                return v * phi_R
            end
        end
    end
end

"""
    assemble_system(diffusion, mesh, bc_left, bc_right; periodic=nothing, transient=false, source=nothing)

Assemble the global system matrix and source vector for a 1D diffusion problem.
"""
function assemble_system(diffusion::Union{Diffusion1D, VariableDiffusion1D}, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition; periodic = nothing, transient = false, source = nothing)
    nx = length(mesh.cells)
    A = SparseArrays.spzeros(nx, nx)
    b = zeros(nx)

    use_periodic = false
    if periodic !== nothing && periodic isa StructuredPeriodicBC
        use_periodic = (periodic.pair == (:left, :right) || periodic.pair == (:right, :left))
    end

    for i in 1:nx
        # Left face
        if i == 1
            if use_periodic
                dx_left = mesh.cells[1].center - mesh.nodes[1].x
                dx_right_periodic = mesh.nodes[end].x - mesh.cells[nx].center
                gamma_face = get_diffusion_coefficient_at_face(diffusion, mesh, 1, :left)
                dx_periodic = (dx_left + dx_right_periodic) / 2.0
                flux_coeff = gamma_face / dx_periodic
                A[1, 1] += flux_coeff
                A[1, nx] -= flux_coeff
                if abs(periodic.shift) > 1.0e-12
                    b[1] += flux_coeff * periodic.shift
                end
            else
                handle_boundary_condition!(A, b, diffusion, mesh, i, bc_left, :left, transient)
            end
        else
            dx = mesh.cells[i].center - mesh.cells[i - 1].center
            gamma_face = get_diffusion_coefficient_at_face(diffusion, mesh, i, :left)
            flux_coeff = gamma_face / dx
            A[i, i] += flux_coeff
            A[i, i - 1] -= flux_coeff
        end

        # Right face
        if i == nx
            if use_periodic
                dx_right = mesh.nodes[end].x - mesh.cells[nx].center
                dx_left_periodic = mesh.cells[1].center - mesh.nodes[1].x
                gamma_face = get_diffusion_coefficient_at_face(diffusion, mesh, nx, :right)
                dx_periodic = (dx_right + dx_left_periodic) / 2.0
                flux_coeff = gamma_face / dx_periodic
                A[nx, nx] += flux_coeff
                A[nx, 1] -= flux_coeff
                if abs(periodic.shift) > 1.0e-12
                    b[nx] -= flux_coeff * periodic.shift
                end
            else
                handle_boundary_condition!(A, b, diffusion, mesh, i, bc_right, :right, transient)
            end
        else
            dx = mesh.cells[i + 1].center - mesh.cells[i].center
            gamma_face = get_diffusion_coefficient_at_face(diffusion, mesh, i, :right)
            flux_coeff = gamma_face / dx
            A[i, i] += flux_coeff
            A[i, i + 1] -= flux_coeff
        end
    end

    if source !== nothing
        apply_source_term!(A, b, source, mesh)
    end

    return A, b
end

"""
    assemble_system(diffusion, mesh, bc_left, bc_right; transient=false, source=nothing)

Assemble system for 1D anisotropic diffusion.
"""
function assemble_system(diffusion::AnisotropicDiffusion1D, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition; transient = false, source = nothing)
    nx = length(mesh.cells)
    A = SparseArrays.spzeros(nx, nx)
    b = zeros(nx)

    gamma = diffusion.D

    for i in 1:nx
        if i == 1
            handle_boundary_condition!(A, b, diffusion, mesh, i, bc_left, :left, transient)
        else
            dx = mesh.cells[i].center - mesh.cells[i - 1].center
            flux_coeff = gamma / dx
            A[i, i] += flux_coeff
            A[i, i - 1] -= flux_coeff
        end

        if i == nx
            handle_boundary_condition!(A, b, diffusion, mesh, i, bc_right, :right, transient)
        else
            dx = mesh.cells[i + 1].center - mesh.cells[i].center
            flux_coeff = gamma / dx
            A[i, i] += flux_coeff
            A[i, i + 1] -= flux_coeff
        end
    end

    if source !== nothing
        apply_source_term!(A, b, source, mesh)
    end

    return A, b
end

function assemble_system_upwind(advection::Union{Advection1D, VariableAdvection1D}, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition; periodic = nothing, transient = false, source = nothing, stabilization = 1.0e-10)
    nx = length(mesh.cells)
    A = SparseArrays.spzeros(nx, nx)
    b = zeros(nx)

    use_periodic = false
    if periodic !== nothing && periodic isa StructuredPeriodicBC
        use_periodic = (periodic.pair == (:left, :right) || periodic.pair == (:right, :left))
    end

    for i in 1:nx
        if i == 1
            if use_periodic
                v_face = get_velocity(advection, mesh, 1, :left)
                if v_face >= 0
                    A[1, nx] -= v_face
                    if abs(periodic.shift) > 1.0e-12
                        b[1] += v_face * periodic.shift
                    end
                else
                    A[1, 1] += abs(v_face)
                end
            else
                handle_advection_boundary_condition!(A, b, advection, mesh, i, bc_left, :left, transient)
            end
        else
            v_face = get_velocity(advection, mesh, i, :left)
            if v_face >= 0
                A[i, i - 1] -= v_face
            else
                A[i, i] += abs(v_face)
            end
        end

        if i == nx
            if use_periodic
                v_face = get_velocity(advection, mesh, nx, :right)
                if v_face >= 0
                    A[nx, nx] += v_face
                else
                    A[nx, 1] -= abs(v_face)
                    if abs(periodic.shift) > 1.0e-12
                        b[nx] -= abs(v_face) * periodic.shift
                    end
                end
            else
                handle_advection_boundary_condition!(A, b, advection, mesh, i, bc_right, :right, transient)
            end
        else
            v_face = get_velocity(advection, mesh, i, :right)
            if v_face >= 0
                A[i, i] += v_face
            else
                A[i, i + 1] -= abs(v_face)
            end
        end

        if !transient && abs(A[i, i]) < stabilization
            A[i, i] += stabilization
        end
    end

    if source !== nothing
        apply_source_term!(A, b, source, mesh)
    end

    return A, b
end

function assemble_system_higher_order(advection::Union{Advection1D, VariableAdvection1D}, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition, phi; periodic = nothing, transient = false, source = nothing, stabilization = 1.0e-10)
    nx = length(mesh.cells)
    A = SparseArrays.spzeros(nx, nx)
    b = zeros(nx)

    use_periodic = false
    if periodic !== nothing && periodic isa StructuredPeriodicBC
        use_periodic = (periodic.pair == (:left, :right) || periodic.pair == (:right, :left))
    end

    for i in 1:nx
        if i == 1
            if use_periodic
                v_face = get_velocity(advection, mesh, 1, :left)
                if v_face >= 0
                    A[1, nx] -= v_face
                    if abs(periodic.shift) > 1.0e-12
                        b[1] += v_face * periodic.shift
                    end
                else
                    A[1, 1] += abs(v_face)
                end
            else
                handle_advection_boundary_condition!(A, b, advection, mesh, i, bc_left, :left, transient)
                v_face = get_velocity(advection, mesh, i, :left)
                if (advection.scheme == :muscl || advection.scheme == :quick) && v_face < 0
                    flux_higher = calculate_flux(advection, mesh, phi, i, :left)
                    flux_upwind = v_face * phi[i]
                    flux_correction = flux_higher - flux_upwind
                    b[i] += flux_correction
                end
            end
        else
            v_face = get_velocity(advection, mesh, i, :left)
            if advection.scheme == :muscl || advection.scheme == :quick
                flux_higher = calculate_flux(advection, mesh, phi, i, :left)
                if v_face >= 0
                    flux_upwind = v_face * phi[i - 1]
                    A[i, i - 1] -= v_face
                else
                    flux_upwind = v_face * phi[i]
                    A[i, i] += abs(v_face)
                end
                flux_correction = flux_higher - flux_upwind
                b[i] += flux_correction
            else
                if v_face >= 0
                    A[i, i - 1] -= v_face
                else
                    A[i, i] += abs(v_face)
                end
            end
        end

        if i == nx
            if use_periodic
                v_face = get_velocity(advection, mesh, nx, :right)
                if v_face >= 0
                    A[nx, nx] += v_face
                else
                    A[nx, 1] -= abs(v_face)
                    if abs(periodic.shift) > 1.0e-12
                        b[nx] -= abs(v_face) * periodic.shift
                    end
                end
            else
                handle_advection_boundary_condition!(A, b, advection, mesh, i, bc_right, :right, transient)
                v_face = get_velocity(advection, mesh, i, :right)
                if (advection.scheme == :muscl || advection.scheme == :quick) && v_face > 0
                    flux_higher = calculate_flux(advection, mesh, phi, i, :right)
                    flux_upwind = v_face * phi[i]
                    flux_correction = flux_higher - flux_upwind
                    b[i] -= flux_correction
                end
            end
        else
            v_face = get_velocity(advection, mesh, i, :right)
            if advection.scheme == :muscl || advection.scheme == :quick
                flux_higher = calculate_flux(advection, mesh, phi, i, :right)
                if v_face >= 0
                    flux_upwind = v_face * phi[i]
                    A[i, i] += v_face
                else
                    flux_upwind = v_face * phi[i + 1]
                    A[i, i + 1] -= abs(v_face)
                end
                flux_correction = flux_higher - flux_upwind
                b[i] -= flux_correction
            else
                if v_face >= 0
                    A[i, i] += v_face
                else
                    A[i, i + 1] -= abs(v_face)
                end
            end
        end

        if !transient && abs(A[i, i]) < stabilization
            A[i, i] += stabilization
        end
    end

    if source !== nothing
        apply_source_term!(A, b, source, mesh)
    end

    return A, b
end

function assemble_system_iterative(advection::Union{Advection1D, VariableAdvection1D}, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition; periodic = nothing, max_iter = 10, tol = 1.0e-6, transient = false, source = nothing, stabilization = 1.0e-10, verbose = false)
    if transient || (advection.scheme != :muscl && advection.scheme != :quick)
        return assemble_system_upwind(advection, mesh, bc_left, bc_right; periodic = periodic, transient = transient, source = source, stabilization = stabilization)
    end

    A, b = assemble_system_upwind(advection, mesh, bc_left, bc_right; periodic = periodic, transient = transient, source = source, stabilization = stabilization)
    phi = A \ b

    A_new = A
    b_new = b

    for iter in 1:max_iter
        A_new, b_new = assemble_system_higher_order(advection, mesh, bc_left, bc_right, phi; periodic = periodic, transient = transient, source = source, stabilization = stabilization)
        phi_new = A_new \ b_new

        change_norm = norm(phi_new - phi)
        if change_norm < tol
            return A_new, b_new
        end
        phi = phi_new
    end
    return A_new, b_new
end

function assemble_system(advection::Union{Advection1D, VariableAdvection1D}, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition; periodic = nothing, transient = false, source = nothing, stabilization = 1.0e-10, max_iter = 10, tol = 1.0e-6, verbose = false)
    if !transient && (advection.scheme == :muscl || advection.scheme == :quick)
        return assemble_system_iterative(advection, mesh, bc_left, bc_right; periodic = periodic, max_iter = max_iter, tol = tol, transient = transient, source = source, stabilization = stabilization, verbose = verbose)
    else
        return assemble_system_upwind(advection, mesh, bc_left, bc_right; periodic = periodic, transient = transient, source = source, stabilization = stabilization)
    end
end

function assemble_system(model::AdvectionDiffusion1D, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition; transient = false, source = nothing)
    nx = length(mesh.cells)
    A = SparseArrays.spzeros(nx, nx)
    b = zeros(nx)

    advection = model.advection
    diffusion = model.diffusion
    v = advection.v
    gamma = diffusion.gamma

    for i in 1:nx
        if i == 1
            handle_advection_diffusion_boundary_condition!(A, b, model, mesh, i, bc_left, :left, transient)
        else
            dx = mesh.cells[i].center - mesh.cells[i - 1].center
            D = gamma / dx
            A[i, i] += D
            A[i, i - 1] -= D

            if v >= 0
                F = v; A[i, i - 1] -= F
            else
                F = abs(v); A[i, i] += F
            end
        end

        if i == nx
            handle_advection_diffusion_boundary_condition!(A, b, model, mesh, i, bc_right, :right, transient)
        else
            dx = mesh.cells[i + 1].center - mesh.cells[i].center
            D = gamma / dx
            A[i, i] += D
            A[i, i + 1] -= D

            if v >= 0
                F = v; A[i, i] += F
            else
                F = abs(v); A[i, i + 1] -= F
            end
        end
    end

    if source !== nothing
        apply_source_term!(A, b, source, mesh)
    end

    return A, b
end

function assemble_system(model::VariableAdvectionDiffusion1D, mesh::Mesh1D, bc_left::AbstractBoundaryCondition, bc_right::AbstractBoundaryCondition; transient = false, source = nothing)
    nx = length(mesh.cells)
    A = SparseArrays.spzeros(nx, nx)
    b = zeros(nx)

    advection = model.advection
    diffusion = model.diffusion

    for i in 1:nx
        if i == 1
            handle_advection_diffusion_boundary_condition!(A, b, model, mesh, i, bc_left, :left, transient)
        else
            dx = mesh.cells[i].center - mesh.cells[i - 1].center
            gamma_face = get_diffusion_coefficient_at_face(diffusion, mesh, i, :left)
            v_face = get_velocity(advection, mesh, i, :left)

            D = gamma_face / dx
            A[i, i] += D
            A[i, i - 1] -= D

            if v_face >= 0
                F = v_face; A[i, i - 1] -= F
            else
                F = abs(v_face); A[i, i] += F
            end
        end

        if i == nx
            handle_advection_diffusion_boundary_condition!(A, b, model, mesh, i, bc_right, :right, transient)
        else
            dx = mesh.cells[i + 1].center - mesh.cells[i].center
            gamma_face = get_diffusion_coefficient_at_face(diffusion, mesh, i, :right)
            v_face = get_velocity(advection, mesh, i, :right)

            D = gamma_face / dx
            A[i, i] += D
            A[i, i + 1] -= D

            if v_face >= 0
                F = v_face; A[i, i] += F
            else
                F = abs(v_face); A[i, i + 1] -= F
            end
        end
    end

    if source !== nothing
        apply_source_term!(A, b, source, mesh)
    end

    return A, b
end
