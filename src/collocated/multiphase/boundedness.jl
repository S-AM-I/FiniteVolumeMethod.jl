# multiphase/boundedness.jl — Boundedness limiter for volume fraction
#
# Two complementary tools live here:
#   1. `clip_alpha!` — a conservative post-solve safety net that clips α
#      to [0, 1] and redistributes the clipped amount globally.
#   2. `mules_limit_flux!` — Stage 5b MULES flux limiter (Zalesak FCT)
#      that should be called DURING alpha transport to enforce
#      boundedness without losing interface sharpness. Applied before
#      clip_alpha so the latter becomes a no-op in practice.
#
# MULES references: Weller (2006), OpenFOAM technical report;
# Rusche (2002), PhD thesis, Imperial College. Clean-room implementation
# from the algorithm description.

"""
    clip_alpha!(alpha, mesh)

Clip volume fraction to [0, 1] bounds with conservative redistribution.

After clipping, any global excess or deficit is distributed proportionally
among cells that are not at the bounds, preserving total α·V.
"""
function clip_alpha!(
        alpha::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)

    # Total alpha*volume before clipping
    total_before = zero(T)
    for c in 1:nc
        total_before += alpha.internal[c] * mesh.cell_volumes[c]
    end

    # Clip
    for c in 1:nc
        alpha.internal[c] = clamp(alpha.internal[c], zero(T), one(T))
    end

    # Total after clipping
    total_after = zero(T)
    for c in 1:nc
        total_after += alpha.internal[c] * mesh.cell_volumes[c]
    end

    # Redistribute difference proportionally to maintain conservation
    diff = total_before - total_after
    if abs(diff) > eps(T) * abs(total_before)
        # Find cells that can absorb the correction
        total_correctable_volume = zero(T)
        for c in 1:nc
            a = alpha.internal[c]
            if diff > 0 && a < one(T)
                total_correctable_volume += mesh.cell_volumes[c]
            elseif diff < 0 && a > zero(T)
                total_correctable_volume += mesh.cell_volumes[c]
            end
        end

        if total_correctable_volume > eps(T)
            correction = diff / total_correctable_volume
            for c in 1:nc
                a = alpha.internal[c]
                if diff > 0 && a < one(T)
                    alpha.internal[c] = min(a + correction, one(T))
                elseif diff < 0 && a > zero(T)
                    alpha.internal[c] = max(a + correction, zero(T))
                end
            end
        end
    end

    return nothing
end

# ── MULES flux limiter (Stage 5b) ────────────────────────────────────

"""
    mules_limit_flux!(
        limited_flux::FaceFluxField{T},
        alpha::CollocatedScalarField{T},
        phi_upwind::FaceFluxField{T},
        phi_high::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T;
        alpha_max::T = one(T),
        alpha_min::T = zero(T),
    )

Apply MULES (Multidimensional Universal Limiter with Explicit Solution)
to construct a bounded anti-diffusive flux for the α transport equation.
Given:

- `phi_upwind[f]`  — monotone first-order upwind flux (bounded).
- `phi_high[f]`    — high-order compressive flux (sharp interface but
                     can overshoot/undershoot).

Writes `limited_flux[f] = phi_upwind[f] + λ_f * (phi_high[f] - phi_upwind[f])`
where `λ_f ∈ [0, 1]` is the Zalesak FCT limiter, chosen per face so that
advancing α with the limited flux keeps every cell's α in
`[alpha_min, alpha_max]` after one explicit Euler step of size `dt`.

Algorithm (per Zalesak 1979 / Rusche 2002):

1. Compute anti-diffusive flux `F_ad = phi_high - phi_upwind` per face.
2. For each cell, compute max incoming and outgoing anti-diffusive flux
   budgets, and the headroom from the upwind-stable α to the bounds.
3. Per-face limiter = min of the cell-side headroom ratios clamped to
   `[0, 1]`.

The resulting limited flux is monotone (never creates overshoots) while
retaining as much anti-diffusion as possible to keep the interface
sharp.

Arguments align with the existing `FaceFluxField` convention: positive
face flux points owner → neighbour.
"""
function mules_limit_flux!(
        limited_flux::FaceFluxField{T},
        alpha::CollocatedScalarField{T},
        phi_upwind::FaceFluxField{T},
        phi_high::FaceFluxField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        dt::T;
        alpha_max::T = one(T),
        alpha_min::T = zero(T),
    ) where {Dim, T}
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    # Step 1: upwind-stable α after a single Euler step with only
    # phi_upwind. Boundedness of phi_upwind's contribution is implied by
    # the first-order upwind construction.
    alpha_td = Vector{T}(undef, nc)
    for c in 1:nc
        alpha_td[c] = alpha.internal[c]
    end
    @inbounds for f in 1:nf
        F_up = phi_upwind.values[f]
        P = mesh.face_cells[1, f]
        N = mesh.face_cells[2, f]
        # Convection updates: -(F φ) leaves owner, enters neighbour.
        # For upwind, φ_f = φ_upwind(F) — but for the α-advance we only
        # need the net face mass transfer.
        flux_contrib = F_up * dt
        alpha_td[P] -= flux_contrib / mesh.cell_volumes[P]
        if N != 0
            alpha_td[N] += flux_contrib / mesh.cell_volumes[N]
        end
    end

    # Step 2: per-cell headroom to the bounds.
    p_plus = Vector{T}(undef, nc)
    p_minus = Vector{T}(undef, nc)
    q_plus = Vector{T}(undef, nc)
    q_minus = Vector{T}(undef, nc)
    for c in 1:nc
        V_c = mesh.cell_volumes[c]
        # Maximum α_increase that keeps α_td[c] ≤ alpha_max.
        q_plus[c] = max((alpha_max - alpha_td[c]) * V_c / dt, zero(T))
        # Maximum α_decrease that keeps α_td[c] ≥ alpha_min.
        q_minus[c] = max((alpha_td[c] - alpha_min) * V_c / dt, zero(T))
        p_plus[c] = zero(T)
        p_minus[c] = zero(T)
    end

    # Step 3: sum incoming / outgoing anti-diffusive fluxes per cell.
    @inbounds for f in 1:nf
        F_ad = phi_high.values[f] - phi_upwind.values[f]
        P = mesh.face_cells[1, f]
        N = mesh.face_cells[2, f]
        # Sign convention: positive F_ad leaves P, enters N.
        # For P: outgoing if F_ad > 0 (|F_ad| contributes to p_minus),
        #        incoming if F_ad < 0 (contributes to p_plus).
        if F_ad >= zero(T)
            p_minus[P] += F_ad
            if N != 0
                p_plus[N] += F_ad
            end
        else
            p_plus[P] += -F_ad
            if N != 0
                p_minus[N] += -F_ad
            end
        end
    end

    # Step 4: cell-side limiter ratios.
    r_plus = Vector{T}(undef, nc)
    r_minus = Vector{T}(undef, nc)
    for c in 1:nc
        r_plus[c] = p_plus[c] > zero(T) ? min(one(T), q_plus[c] / p_plus[c]) : one(T)
        r_minus[c] = p_minus[c] > zero(T) ? min(one(T), q_minus[c] / p_minus[c]) : one(T)
    end

    # Step 5: per-face limiter λ_f = min of source/destination limiter
    # appropriate to the sign of F_ad.
    @inbounds for f in 1:nf
        F_up = phi_upwind.values[f]
        F_ad = phi_high.values[f] - F_up
        P = mesh.face_cells[1, f]
        N = mesh.face_cells[2, f]
        lambda_f = if F_ad >= zero(T)
            N == 0 ? r_minus[P] : min(r_minus[P], r_plus[N])
        else
            N == 0 ? r_plus[P] : min(r_plus[P], r_minus[N])
        end
        limited_flux.values[f] = F_up + lambda_f * F_ad
    end

    return nothing
end
