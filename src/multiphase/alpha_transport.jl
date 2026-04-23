# multiphase/alpha_transport.jl — Volume fraction transport equation
#
# Assembles the α advection equation with optional interface
# compression. Two boundedness paths are supported:
#
#   * `use_mules = true` (default, v3.92+): build explicit first-order
#     (`phi_upwind`) and compressive high-order (`phi_high`) α-fluxes,
#     pass them to `mules_limit_flux!` to obtain a Zalesak-FCT-limited
#     face-flux that is bounded by construction, and assemble that
#     flux as an explicit RHS source against a pure-temporal LHS.
#     `clip_alpha!` remains a post-solve safety net.
#
#   * `use_mules = false` (legacy): original implicit assembly using
#     `assemble_convection!` on the underlying velocity flux `phi`
#     plus an explicit compression source. Boundedness is enforced
#     exclusively by the hard-clip + global redistribution in
#     `clip_alpha!`.
#
# References: Weller (2006); Rusche (2002) thesis; Zalesak (1979).

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
        dt, C_alpha = 1.0, use_mules = true,
    )

Assemble the volume fraction transport equation.

The equation is `∂α/∂t + div(phi · α) + div(phi_c · α(1-α)) = 0`.

When `use_mules = true` (default) the boundedness-preserving path is
taken: both a first-order upwind α-flux and a high-order compressive
α-flux are built, MULES produces a Zalesak-FCT-limited face-flux, and
the solve reduces to the trivial identity `(V/dt) α^{n+1} = (V/dt) α^n - Σ F^α_limited`.

When `use_mules = false` the legacy implicit assembly is used — useful
for diagnostics and regression against the pre-v3.92 behaviour.
"""
function assemble_alpha!(
        eq::CollocatedEquation{T},
        alpha::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_alpha::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::T,
        C_alpha::T = one(T),
        use_mules::Bool = true,
    ) where {Dim, T}
    if use_mules
        return _assemble_alpha_mules!(eq, alpha, phi, mesh; dt = dt, C_alpha = C_alpha)
    else
        return _assemble_alpha_legacy!(
            eq, alpha, phi, mesh, bcs_alpha; dt = dt, C_alpha = C_alpha,
        )
    end
end

# ── Legacy (implicit + explicit compression, hard-clip safety) ───────

function _assemble_alpha_legacy!(
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

# ── MULES path (explicit, bounded by construction) ───────────────────

function _assemble_alpha_mules!(
        eq::CollocatedEquation{T},
        alpha::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T};
        dt::T,
        C_alpha::T = one(T),
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Temporal: (V/dt) α^{n+1} = (V/dt) α^n - Σ F^α_limited, assembled
    # as a pure diagonal system.
    for c in 1:nc
        coeff = mesh.cell_volumes[c] / dt
        add_diag!(eq, c, coeff)
        eq.b[c] += coeff * alpha.internal[c]
    end

    # Build upwind (first-order) and high-order (compressive) α-face
    # fluxes in volumetric units (same convention as `phi`).
    phi_up = FaceFluxField(:phi_alpha_up, mesh; value = zero(T))
    phi_hi = FaceFluxField(:phi_alpha_hi, mesh; value = zero(T))
    phi_c = C_alpha > zero(T) ?
        compute_compression_flux(alpha, phi, mesh; C_alpha = C_alpha) :
        zeros(T, nf)

    @inbounds for f in 1:nf
        F_f = phi.values[f]
        P = owner(mesh, f)

        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)
            a_P = alpha.internal[P]
            a_N = alpha.internal[N]
            # First-order upwind α-flux.
            alpha_up = F_f >= zero(T) ? a_P : a_N
            phi_up.values[f] = F_f * alpha_up
            # Linear-interpolated α plus compressive sharpening term.
            alpha_lin = w * a_P + (one(T) - w) * a_N
            alpha_comp = alpha_lin * (one(T) - alpha_lin)
            phi_hi.values[f] = F_f * alpha_lin + phi_c[f] * alpha_comp
        else
            # Boundary face — keep upwind from owner; compression term
            # is zero (compute_compression_flux is zero on boundaries).
            phi_up.values[f] = F_f * alpha.internal[P]
            phi_hi.values[f] = F_f * alpha.internal[P]
        end
    end

    # MULES-limit the anti-diffusive flux.
    phi_lim = FaceFluxField(:phi_alpha_lim, mesh; value = zero(T))
    mules_limit_flux!(phi_lim, alpha, phi_up, phi_hi, mesh, dt)

    # Subtract the divergence of the limited α-flux from RHS.
    @inbounds for f in 1:nf
        F_f_alpha = phi_lim.values[f]
        P = owner(mesh, f)
        eq.b[P] -= F_f_alpha
        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            eq.b[N] += F_f_alpha
        end
    end

    return nothing
end
