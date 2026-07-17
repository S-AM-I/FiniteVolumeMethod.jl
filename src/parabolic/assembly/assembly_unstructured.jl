# Assembly routines for unstructured meshes
# Migrated from Simu.jl SimuFVM/assembly/assembly_unstructured.jl

"""
    assemble_system(model::Union{Diffusion2D, Diffusion3D}, mesh::Union{UnstructuredMesh2D, UnstructuredMesh3D}, bcs::Dict; source=nothing, transient=false)

Assemble system for diffusion on an unstructured mesh (2D or 3D).
`bcs` is a Dictionary mapping boundary face indices (integers) to AbstractBoundaryCondition objects.
Faces not in `bcs` are treated as insulated (Neumann(0)).
"""
function assemble_system(model::Union{Diffusion2D, Diffusion3D}, mesh::Union{UnstructuredMesh2D, UnstructuredMesh3D}, bcs::Dict; source = nothing, transient = false)
    nx = length(mesh.cells)
    A = SparseArrays.spzeros(nx, nx)
    b = zeros(nx)
    gamma = model.gamma

    # Iterate over all faces
    for (f_idx, face) in enumerate(mesh.faces)
        owner = face.owner
        neighbor = face.neighbor

        # Calculate distance between centroids
        if neighbor > 0
            # Internal face
            c_owner = mesh.cells[owner].center
            c_neighbor = mesh.cells[neighbor].center

            # Distance approximation (valid for orthogonal-like meshes)
            d_vec = c_neighbor .- c_owner
            dist = norm(d_vec)

            # Flux coefficient: Gamma * Area / Distance
            # Steepest Descent approximation (stable)
            flux_coeff = gamma * face.area / dist

            # Flux from Owner to Neighbor: F = coeff * (phi_N - phi_O)
            # Owner equation: ... - Flux ... = Source
            # Flux leaving owner: coeff * (phi_O - phi_N)
            # Diagonal (phi_O): +coeff
            # Off-diagonal (phi_N): -coeff

            A[owner, owner] += flux_coeff
            A[owner, neighbor] -= flux_coeff

            # Neighbor sees flux entering (opposite normal): -F
            # Neighbor equation: Flux leaving neighbor = coeff * (phi_N - phi_O)
            # Diagonal (phi_N): +coeff
            # Off-diagonal (phi_O): -coeff

            A[neighbor, neighbor] += flux_coeff
            A[neighbor, owner] -= flux_coeff

        else
            # Boundary face
            # owner is the cell. neighbor is 0.
            # Check for BC
            if haskey(bcs, f_idx)
                bc = bcs[f_idx]
                c_owner = mesh.cells[owner].center
                c_face = face.center

                # Distance from cell center to face center
                dist = norm(c_face .- c_owner)

                # Handle BC
                handle_unstructured_bc!(A, b, model, bc, owner, f_idx, face.area, dist)
            else
                # Default: Zero flux (Neumann 0)
                # No contribution to A or b
            end
        end
    end

    # Source terms
    if source !== nothing
        apply_source_term!(A, b, source, mesh)
    end

    return A, b
end

function handle_unstructured_bc!(A, b, model::Union{Diffusion2D, Diffusion3D}, bc::DirichletBC, cell_idx, f_idx, area, dist)
    # Flux leaving cell: coeff * (phi_Cell - phi_BC)
    # A[c,c] += coeff
    # RHS += coeff * phi_BC

    flux_coeff = model.gamma * area / dist
    A[cell_idx, cell_idx] += flux_coeff
    return b[cell_idx] += flux_coeff * bc.value
end

function handle_unstructured_bc!(A, b, model::Union{Diffusion2D, Diffusion3D}, bc::NeumannBC, cell_idx, f_idx, area, dist)
    # Prescribed Outward Flux: q = bc.value
    # Flux_Out = bc.value * area
    # b -= bc.value * area

    return b[cell_idx] -= bc.value * area
end

"""
    assemble_deferred_correction(model::Diffusion2D, mesh::UnstructuredMesh2D, bcs::Dict, phi::Vector{Float64})

Calculate the deferred correction source term for non-orthogonal meshes.
Returns a vector `b_corr` to be added to the RHS of the linear system.
"""
function assemble_deferred_correction(model::Diffusion2D, mesh::UnstructuredMesh2D, bcs::Dict, phi::Vector{Float64})
    nx = length(mesh.cells)
    b_corr = zeros(nx)
    gamma = model.gamma

    # 1. Compute gradients at all cells
    gradients = Vector{Tuple{Float64, Float64}}(undef, nx)
    for i in 1:nx
        gradients[i] = reconstruct_gradient_green_gauss_2d(mesh, phi, i; bcs = bcs)
    end

    # 2. Iterate faces and compute non-orthogonal flux correction
    for (f_idx, face) in enumerate(mesh.faces)
        owner = face.owner
        neighbor = face.neighbor

        if neighbor > 0
            # Internal face
            c_owner = mesh.cells[owner].center
            c_neighbor = mesh.cells[neighbor].center

            d_vec = c_neighbor .- c_owner
            d_mag_sq = dot(d_vec, d_vec)
            dist = sqrt(d_mag_sq)

            # Face area vector S = n * A
            S = face.normal * face.area

            # Orthogonal decomposition consistent with assemble_system:
            # Steepest Descent: E_impl = A * d_hat
            d_hat = d_vec / dist
            E_impl = face.area * d_hat
            T = S - E_impl

            # Interpolate gradient to face
            grad_owner = gradients[owner]
            grad_neighbor = gradients[neighbor]

            # Simple average for gradient interpolation
            grad_f_x = 0.5 * (grad_owner[1] + grad_neighbor[1])
            grad_f_y = 0.5 * (grad_owner[2] + grad_neighbor[2])
            grad_f = [grad_f_x, grad_f_y]

            # Non-orthogonal flux correction: -Gamma * grad_f . T
            flux_corr = -gamma * dot(grad_f, T)

            # Add to RHS (Deferred Correction)
            b_corr[owner] -= flux_corr
            b_corr[neighbor] += flux_corr
        else
            # Boundary face
            if haskey(bcs, f_idx)
                bc = bcs[f_idx]
                if bc isa DirichletBC
                    c_owner = mesh.cells[owner].center
                    c_face = face.center

                    d_vec = c_face .- c_owner
                    dist = norm(d_vec)
                    d_hat = d_vec / dist

                    # S points out of domain (owner -> face)
                    S = face.normal * face.area

                    # Implicit term models flux along d_vec
                    E_impl = face.area * d_hat
                    T = S - E_impl

                    # Gradient at face: use cell gradient
                    grad_owner = gradients[owner]
                    grad_f = [grad_owner[1], grad_owner[2]]

                    flux_corr = -gamma * dot(grad_f, T)

                    # Subtract from Owner RHS
                    b_corr[owner] -= flux_corr
                end
            end
        end
    end

    return b_corr
end

"""
    assemble_deferred_correction(model::Diffusion3D, mesh::UnstructuredMesh3D, bcs::Dict, phi::Vector{Float64})

Calculate the deferred correction source term for non-orthogonal 3D meshes.
"""
function assemble_deferred_correction(model::Diffusion3D, mesh::UnstructuredMesh3D, bcs::Dict, phi::Vector{Float64})
    nx = length(mesh.cells)
    b_corr = zeros(nx)
    gamma = model.gamma

    # 1. Compute gradients at all cells
    gradients = Vector{Tuple{Float64, Float64, Float64}}(undef, nx)
    for i in 1:nx
        gradients[i] = reconstruct_gradient_green_gauss_3d(mesh, phi, i; bcs = bcs)
    end

    # 2. Iterate faces
    for (f_idx, face) in enumerate(mesh.faces)
        owner = face.owner
        neighbor = face.neighbor

        if neighbor > 0
            # Internal face
            c_owner = mesh.cells[owner].center
            c_neighbor = mesh.cells[neighbor].center

            d_vec = c_neighbor .- c_owner
            dist = norm(d_vec)
            d_hat = d_vec / dist

            # Surface vector S = n * A
            S = face.normal * face.area

            # Decomposition: S = E + T
            # E = A * d_hat
            E_impl = face.area * d_hat
            T = S - E_impl

            # Interpolate gradient
            grad_owner = gradients[owner]
            grad_neighbor = gradients[neighbor]

            grad_f_x = 0.5 * (grad_owner[1] + grad_neighbor[1])
            grad_f_y = 0.5 * (grad_owner[2] + grad_neighbor[2])
            grad_f_z = 0.5 * (grad_owner[3] + grad_neighbor[3])
            grad_f = [grad_f_x, grad_f_y, grad_f_z]

            # Correction flux = -Gamma * (grad . T)
            flux_corr = -gamma * dot(grad_f, T)

            b_corr[owner] -= flux_corr
            b_corr[neighbor] += flux_corr

        else
            # Boundary face
            if haskey(bcs, f_idx)
                bc = bcs[f_idx]
                if bc isa DirichletBC
                    c_owner = mesh.cells[owner].center
                    c_face = face.center
                    d_vec = c_face .- c_owner
                    dist = norm(d_vec)
                    d_hat = d_vec / dist

                    S = face.normal * face.area
                    E_impl = face.area * d_hat
                    T = S - E_impl

                    grad_owner = gradients[owner]
                    grad_f = [grad_owner[1], grad_owner[2], grad_owner[3]]

                    flux_corr = -gamma * dot(grad_f, T)
                    b_corr[owner] -= flux_corr
                end
            end
        end
    end

    return b_corr
end
