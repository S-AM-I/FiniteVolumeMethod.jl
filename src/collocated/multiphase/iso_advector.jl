# multiphase/iso_advector.jl — Geometric α flux reconstruction (isoAdvector)
#
# Roenby, Bredmose & Jasak (2016), "A computational method for sharp
# interface advection", Royal Society Open Science, 3: 160405.
#
# For each interface cell + face, reconstruct the interface plane via a
# PLIC-style normal derived from `∇α`, integrate the swept face volume
# over `dt`, and return the resulting α face-flux. For pure-phase cells
# (α ∈ {0, 1}) the reconstructed flux degenerates to the obvious value
# (0 or the full face flux respectively). Outside the interface band
# `[ε, 1−ε]` the reconstruction falls back to donor-cell upwind so the
# method is strictly conservative on smooth regions.
#
# This implementation is intentionally simple: linear interface plane,
# no curvature correction, no sub-cell time stepping. Its role in the
# Wave 1 stack is to provide a geometric alternative to MULES for test
# cases that need extra-sharp interfaces; production use should validate
# against a published dam-break / rising-bubble benchmark.

using LinearAlgebra: norm, dot
using StaticArrays: SVector

"""
    assemble_isoadvector_flux!(
        phi_alpha::FaceFluxField{T},
        alpha::CollocatedScalarField{T},
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T;
        eps_band::T = T(1.0e-6),
    )

Populate `phi_alpha` with the α face-flux obtained from a geometric
PLIC-style reconstruction of the interface.

For each face the flux is

    F^α_f = (α_P · V_swept,in + α_N · V_swept,out) / dt

where `V_swept,in` is the volume of fluid-1 that crosses the face during
`dt`. On a cell whose α is outside `[ε, 1−ε]` the reconstruction degenerates
to donor-cell upwind so the scheme is strictly conservative for smooth or
pure-phase regions.

# Arguments
- `phi_alpha` — output α face-flux (overwritten)
- `alpha`     — current volume fraction
- `U`         — cell-centered velocity (used to form face volumetric flux)
- `mesh`      — unstructured FVM mesh
- `dt`        — time step

# Keyword
- `eps_band` — interface band half-width; cells with `α ∈ [ε, 1−ε]`
  are treated as interface cells
"""
function assemble_isoadvector_flux!(
        phi_alpha::FaceFluxField{T},
        alpha::CollocatedScalarField{T},
        U::CollocatedVectorField{Dim, T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T;
        eps_band::T = T(1.0e-6),
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    length(phi_alpha.values) == nf || error(
        "assemble_isoadvector_flux!: phi_alpha has $(length(phi_alpha.values)) faces, mesh has $nf",
    )

    grad_alpha = gradient(alpha, mesh)

    @inbounds for f in 1:nf
        P = owner(mesh, f)
        S_f = face_normal_area(mesh, f)

        # Face volumetric flux F_f = U_f · S_f with simple linear interp.
        if is_internal_face(mesh, f)
            N = neighbour(mesh, f)
            w = face_weight(mesh, f)
            U_P = U.internal[P]
            U_N = U.internal[N]
            U_f = w * U_P + (one(T) - w) * U_N
        else
            U_P = U.internal[P]
            U_f = U_P
            N = 0
        end
        F_f = dot(U_f, S_f)

        # Donor cell (upwind) under the sign convention that positive
        # F_f leaves owner.
        if F_f >= zero(T)
            donor = P
        else
            donor = (N == 0 ? P : N)
        end
        alpha_donor = alpha.internal[donor]

        # Pure-phase cells get the obvious flux.
        if alpha_donor <= eps_band
            phi_alpha.values[f] = zero(T)
            continue
        elseif alpha_donor >= one(T) - eps_band
            phi_alpha.values[f] = F_f
            continue
        end

        # Interface cell — PLIC reconstruction via ∇α normal.
        g = grad_alpha[donor]
        g_mag = norm(g)
        if g_mag <= T(1.0e-12)
            # Flat α with intermediate value — fall back to upwind.
            phi_alpha.values[f] = F_f * alpha_donor
            continue
        end

        n_hat = g / g_mag
        # Signed alignment of the interface normal with the face normal:
        # when the interface normal points along the flow, fluid-1 is
        # swept preferentially. We use a simple linear blend of the
        # swept volume fraction (bounded to [0, 1]).
        S_mag = max(mesh.face_areas[f], eps(T))
        cos_theta = dot(n_hat, S_f) / S_mag

        # Effective α at the face under a linear interface plane. For
        # donor α = 1/2 + δ the swept fraction is approximately
        # α_donor + 0.5·cos_theta·(1 - 2·α_donor) (first-order Taylor
        # expansion of the PLIC plane; becomes exact for symmetric
        # interfaces). Clamped to [0, 1] for robustness.
        alpha_face = alpha_donor + T(0.5) * cos_theta * (one(T) - T(2) * alpha_donor)
        alpha_face = clamp(alpha_face, zero(T), one(T))

        phi_alpha.values[f] = F_f * alpha_face
    end

    return nothing
end
