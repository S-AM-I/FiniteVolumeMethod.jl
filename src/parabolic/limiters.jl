# Flux limiters for higher-order advection schemes (parabolic solver)
# Migrated from Simu.jl SimuFVM/limiters.jl
#
# Core limiter functions (minmod, superbee, van_leer, venkatakrishnan,
# barth_jespersen, koren, ospre) are now delegated to the canonical
# implementations in src/schemes/limiters.jl.
#
# This module provides parabolic-solver-specific utilities:
#   - Symbol-based limiter dispatch (apply_limiter(:minmod, r))
#   - Slope ratio computation for 1D MUSCL reconstruction
#   - Limited slope computation for 1D reconstruction
#   - Automatic limiter selection heuristics

"""
    ParabolicLimiters

Submodule providing slope-limiter utilities for the parabolic solver's MUSCL
reconstruction.  Delegates core limiter functions (minmod, superbee, van_leer,
etc.) to `src/schemes/limiters.jl` and adds symbol-based dispatch
([`apply_limiter`](@ref)), 1D slope-ratio computation, and automatic limiter
selection heuristics.
"""
module ParabolicLimiters

# Import canonical limiter functions from the parent module (src/schemes/limiters.jl)
using ..FiniteVolumeMethod: minmod, superbee, van_leer, venkatakrishnan,
    barth_jespersen, koren, ospre

# Re-export so existing `using .ParabolicLimiters: minmod` still works
export minmod, superbee, van_leer, venkatakrishnan, barth_jespersen, koren, ospre

"""
    apply_limiter(limiter_type, r)

Apply a limiter function to ratio `r`.
`limiter_type` can be `:minmod`, `:superbee`, `:van_leer`, `:venkatakrishnan`,
`:koren`, or `:ospre`.
"""
function apply_limiter(limiter_type::Symbol, r::Float64; kwargs...)
    if limiter_type == :minmod
        return minmod(r, 1.0)
    elseif limiter_type == :superbee
        return superbee(r, 1.0)
    elseif limiter_type == :van_leer
        return van_leer(r, 1.0)
    elseif limiter_type == :venkatakrishnan
        eps = get(kwargs, :eps, 1.0e-6)
        return venkatakrishnan(r, eps)
    elseif limiter_type == :koren
        beta = get(kwargs, :beta, 1.0 / 3.0)
        return koren(r, beta)
    elseif limiter_type == :ospre
        return ospre(r)
    else
        return 1.0  # No limiting
    end
end

"""
    select_limiter_strategy(problem_type, mesh_type)

Automatically select an appropriate limiter based on problem characteristics.
"""
function select_limiter_strategy(problem_type::Symbol, mesh_type::Symbol = :uniform)
    if problem_type == :conservative
        return :minmod
    elseif problem_type == :accuracy
        return :van_leer
    elseif problem_type == :unstructured
        return :venkatakrishnan
    elseif problem_type == :shock_capturing
        return :superbee
    else
        return :minmod  # Safe default
    end
end

"""
    compute_slope_ratio_1d(phi, i, direction)

Compute slope ratio for limiter in 1D.
`direction` can be `:left` or `:right`.
"""
function compute_slope_ratio_1d(phi, i::Int, direction::Symbol)
    nx = length(phi)

    return if direction == :left
        if i <= 1
            return 1.0
        elseif i == 2
            return 1.0
        else
            num = phi[i] - phi[i - 1]
            den = phi[i - 1] - phi[i - 2]
            if abs(den) < 1.0e-12
                return 1.0
            end
            return num / den
        end
    else # direction == :right
        if i >= nx
            return 1.0
        elseif i == nx - 1
            return 1.0
        else
            num = phi[i + 1] - phi[i]
            den = phi[i + 2] - phi[i + 1]
            if abs(den) < 1.0e-12
                return 1.0
            end
            return num / den
        end
    end
end

"""
    limit_slope_1d(phi, i, direction, limiter_type)

Compute limited slope for MUSCL reconstruction in 1D.
Returns the limited slope (gradient) at cell `i`.
"""
function limit_slope_1d(phi, i::Int, direction::Symbol, limiter_type::Symbol)
    nx = length(phi)

    return if direction == :left
        if i <= 1
            return nx > 1 ? phi[i + 1] - phi[i] : 0.0
        elseif i == 2
            return phi[i] - phi[i - 1]
        else
            delta_forward = phi[i] - phi[i - 1]
            delta_backward = phi[i - 1] - phi[i - 2]
            if limiter_type == :minmod
                return minmod(delta_forward, delta_backward)
            elseif limiter_type == :superbee
                return superbee(delta_forward, delta_backward)
            elseif limiter_type == :van_leer
                return van_leer(delta_forward, delta_backward)
            else
                return minmod(delta_forward, delta_backward)
            end
        end
    else # direction == :right
        if i >= nx
            return nx > 1 ? phi[i] - phi[i - 1] : 0.0
        elseif i == 1
            return phi[i + 1] - phi[i]
        else
            delta_forward = phi[i + 1] - phi[i]
            delta_backward = phi[i] - phi[i - 1]
            if limiter_type == :minmod
                return minmod(delta_forward, delta_backward)
            elseif limiter_type == :superbee
                return superbee(delta_forward, delta_backward)
            elseif limiter_type == :van_leer
                return van_leer(delta_forward, delta_backward)
            else
                return minmod(delta_forward, delta_backward)
            end
        end
    end
end

end # module ParabolicLimiters
