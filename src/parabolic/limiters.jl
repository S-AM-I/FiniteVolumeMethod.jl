# Flux limiters for higher-order advection schemes
# Migrated from Simu.jl SimuFVM/limiters.jl
# Wrapped in module ParabolicLimiters to avoid name collisions with existing FVM.jl limiter exports

module ParabolicLimiters

"""
    minmod(a, b)

Minmod limiter: returns 0 if a and b have opposite signs, otherwise returns the one with smaller magnitude.
"""
function minmod(a::Float64, b::Float64)
    if a * b <= 0.0
        return 0.0
    else
        return abs(a) < abs(b) ? a : b
    end
end

"""
    superbee(a, b)

Superbee limiter: returns the maximum of minmod(2a, b) and minmod(a, 2b).
"""
function superbee(a::Float64, b::Float64)
    if a * b <= 0.0
        return 0.0
    else
        return max(minmod(2.0 * a, b), minmod(a, 2.0 * b))
    end
end

"""
    van_leer(a, b)

Van Leer limiter: harmonic mean of a and b.
"""
function van_leer(a::Float64, b::Float64)
    if a * b <= 0.0
        return 0.0
    else
        return 2.0 * a * b / (a + b)
    end
end

"""
    venkatakrishnan(r, eps=1e-6)

Venkatakrishnan limiter for unstructured meshes.
r is the ratio of consecutive gradients.
"""
function venkatakrishnan(r::Float64; eps = 1.0e-6)
    if r <= 0.0
        return 0.0
    else
        numerator = (r^2 + 2.0 * r) / (r^2 + r + 2.0)
        return numerator
    end
end

"""
    barth_jespersen(phi, phi_min, phi_max, phi_face)

Barth-Jespersen limiter for preserving local extrema.
Returns limiting factor in [0, 1] to ensure phi_face is between phi_min and phi_max.
"""
function barth_jespersen(phi_center::Float64, phi_min::Float64, phi_max::Float64, phi_face::Float64)
    if abs(phi_face - phi_center) < 1.0e-12
        return 1.0
    end

    return if phi_face > phi_center
        # Need to limit from above
        if phi_max - phi_center > 1.0e-12
            return min(1.0, (phi_max - phi_center) / (phi_face - phi_center))
        else
            return 0.0
        end
    else
        # Need to limit from below
        if phi_center - phi_min > 1.0e-12
            return min(1.0, (phi_center - phi_min) / (phi_center - phi_face))
        else
            return 0.0
        end
    end
end

"""
    koren(r, beta=1.0/3.0)

Koren limiter: smooth limiter that is third-order accurate.
r is the ratio of consecutive gradients.
beta controls the limiter behavior (typically 1/3).
"""
function koren(r::Float64; beta = 1.0 / 3.0)
    if r <= 0.0
        return 0.0
    else
        return max(0.0, min(2.0 * r, min((1.0 + 2.0 * r) / 3.0, 2.0)))
    end
end

"""
    ospre(r)

OSPRE (Optimized Second-order Polynomial for Recovery Enhancement) limiter.
Compromise between accuracy and monotonicity.
"""
function ospre(r::Float64)
    if r <= 0.0
        return 0.0
    else
        return 1.5 * (r^2 + r) / (r^2 + r + 1.0)
    end
end

"""
    apply_limiter(limiter_type, r)

Apply a limiter function to ratio r.
limiter_type can be :minmod, :superbee, :van_leer, or :venkatakrishnan.
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
        return venkatakrishnan(r; eps = eps)
    elseif limiter_type == :koren
        beta = get(kwargs, :beta, 1.0 / 3.0)
        return koren(r; beta = beta)
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
direction can be :left or :right.
r = (phi[i] - phi[neighbor]) / (phi[neighbor] - phi[far_neighbor])
"""
function compute_slope_ratio_1d(phi, i::Int, direction::Symbol)
    nx = length(phi)

    return if direction == :left
        if i <= 1
            return 1.0
        elseif i == 2
            return 1.0  # Can't compute ratio with only one neighbor
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
Returns the limited slope (gradient) at cell i.
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
