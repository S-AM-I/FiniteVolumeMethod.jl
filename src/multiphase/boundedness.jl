# multiphase/boundedness.jl — Boundedness limiter for volume fraction
#
# Clips alpha to [0, 1] and redistributes the error to maintain
# global conservation of the volume fraction field.

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
