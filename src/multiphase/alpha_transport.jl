# multiphase/alpha_transport.jl — Volume fraction transport equation
#
# Assembles the alpha advection equation with optional interface
# compression term for maintaining interface sharpness.

"""
    compute_compression_flux(
        alpha, phi, mesh; C_alpha = 1.0,
    ) -> Vector{T}

Compute the interface compression flux per face.

`phi_c_f = C_alpha · |phi_f| · (n_interface · S_f) / |S_f|`

where `n_interface = ∇α/|∇α|` is the interface normal direction.
"""
function compute_compression_flux(
        alpha::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T};
        C_alpha::T = one(T),
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    phi_c = zeros(T, nf)

    # Compute interface normal (gradient of alpha)
    grad_alpha = gradient(alpha, mesh)

    for f in 1:nf
        if is_internal_face(mesh, f)
            P = owner(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)

            # Interpolate interface normal to face
            n_P = grad_alpha[P]
            n_N = grad_alpha[N]
            n_f = w * n_P + (one(T) - w) * n_N
            n_mag = norm(n_f)

            if n_mag > eps(T)
                n_hat = n_f / n_mag
                S_f = face_normal_area(mesh, f)
                S_mag = mesh.face_areas[f]

                # Compression flux: aligned with interface normal
                phi_c[f] = C_alpha * abs(phi.values[f]) * dot(n_hat, S_f) / max(S_mag, eps(T))
            end
        end
    end

    return phi_c
end

"""
    assemble_alpha!(
        eq, alpha, phi, mesh, bcs_alpha;
        dt, C_alpha = 1.0,
    )

Assemble the volume fraction transport equation with interface compression.

The equation is:
`∂α/∂t + div(phi · α) + div(phi_c · α(1-α)) = 0`

The standard convection `div(phi · α)` is assembled implicitly.
The compression term `div(phi_c · α(1-α))` is added explicitly to the RHS.
"""
function assemble_alpha!(
        eq::CollocatedEquation{T},
        alpha::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_alpha::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::T,
        C_alpha::T = one(T),
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)

    # Standard convection: div(phi · alpha)
    assemble_convection!(eq, phi, mesh, bcs_alpha)

    # Temporal: ddt(alpha)
    assemble_ddt_euler!(eq, one(T), alpha.internal, mesh, dt)

    # Interface compression (explicit source)
    if C_alpha > zero(T)
        phi_c = compute_compression_flux(alpha, phi, mesh; C_alpha = C_alpha)

        for f in 1:nf
            if is_internal_face(mesh, f)
                P = owner(mesh, f)
                N = neighbour(mesh, f)
                w = face_weight(mesh, f)

                # Interpolate alpha to face
                alpha_f = w * alpha.internal[P] + (one(T) - w) * alpha.internal[N]
                compression = phi_c[f] * alpha_f * (one(T) - alpha_f)

                eq.b[P] -= compression
                eq.b[N] += compression
            end
        end
    end

    return nothing
end
