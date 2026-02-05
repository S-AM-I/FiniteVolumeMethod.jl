@doc raw"""
    Flux Limiters Module

This module provides flux limiters for higher-order finite volume methods.
Flux limiters are used to ensure TVD (Total Variation Diminishing) or monotonicity
properties while maintaining higher-order accuracy in smooth regions.

The limiters operate on consecutive slope ratios and return a limiting factor.
"""

"""
    AbstractLimiter

Abstract type for flux limiters.
"""
abstract type AbstractLimiter end

"""
    MinmodLimiter <: AbstractLimiter

The minmod limiter - most diffusive, guaranteed TVD.
"""
struct MinmodLimiter <: AbstractLimiter end

"""
    SuperbeeLimiter <: AbstractLimiter

The superbee limiter - least diffusive TVD limiter, can be oscillatory.
"""
struct SuperbeeLimiter <: AbstractLimiter end

"""
    VanLeerLimiter <: AbstractLimiter

The van Leer limiter - smooth, good balance between accuracy and stability.
"""
struct VanLeerLimiter <: AbstractLimiter end

"""
    VenkatakrishnanLimiter <: AbstractLimiter

The Venkatakrishnan limiter - designed for unstructured meshes, smooth and differentiable.
"""
struct VenkatakrishnanLimiter <: AbstractLimiter end

"""
    BarthJespersenLimiter <: AbstractLimiter

The Barth-Jespersen limiter - preserves local extrema, commonly used for unstructured meshes.
"""
struct BarthJespersenLimiter <: AbstractLimiter end

"""
    KorenLimiter <: AbstractLimiter

The Koren limiter - third-order accurate for smooth solutions.
"""
struct KorenLimiter <: AbstractLimiter end

"""
    OspreLimiter <: AbstractLimiter

The OSPRE limiter - optimized second-order polynomial, smooth and symmetric.
"""
struct OspreLimiter <: AbstractLimiter end

@doc raw"""
    minmod(a::T, b::T) where {T<:Real} -> T

Compute the minmod function of two values.

```math
\text{minmod}(a, b) = \begin{cases}
\min(a, b) & \text{if } a > 0 \text{ and } b > 0 \\
\max(a, b) & \text{if } a < 0 \text{ and } b < 0 \\
0 & \text{otherwise}
\end{cases}
```

This is the most diffusive TVD limiter, providing maximum stability
at the cost of accuracy.

# Arguments
- `a::T`: First value
- `b::T`: Second value

# Returns
The minmod of `a` and `b`.
"""
@inline function minmod(a::T, b::T) where {T<:Real}
    if a > zero(T) && b > zero(T)
        return min(a, b)
    elseif a < zero(T) && b < zero(T)
        return max(a, b)
    else
        return zero(T)
    end
end

@doc raw"""
    superbee(a::T, b::T) where {T<:Real} -> T

Compute the superbee limiter of two values.

```math
\text{superbee}(a, b) = \max\left(\text{minmod}(2a, b), \text{minmod}(a, 2b)\right)
```

This is the least diffusive TVD limiter. It can produce sharper
discontinuities but may be slightly oscillatory near extrema.

# Arguments
- `a::T`: First value
- `b::T`: Second value

# Returns
The superbee limiter value.
"""
@inline function superbee(a::T, b::T) where {T<:Real}
    return max(minmod(2a, b), minmod(a, 2b))
end

@doc raw"""
    van_leer(a::T, b::T) where {T<:Real} -> T

Compute the van Leer limiter of two values.

```math
\text{vanLeer}(a, b) = \begin{cases}
\frac{2ab}{a + b} & \text{if } ab > 0 \\
0 & \text{otherwise}
\end{cases}
```

This limiter provides a good balance between accuracy and stability,
using a harmonic mean formulation.

# Arguments
- `a::T`: First value
- `b::T`: Second value

# Returns
The van Leer limiter value.
"""
@inline function van_leer(a::T, b::T) where {T<:Real}
    ab = a * b
    if ab > zero(T)
        return 2ab / (a + b)
    else
        return zero(T)
    end
end

@doc raw"""
    venkatakrishnan(r::T, ε::T=T(1e-6)) where {T<:Real} -> T

Compute the Venkatakrishnan limiter for a slope ratio.

```math
\phi(r) = \frac{r^2 + 2r}{r^2 + r + 2}
```

This limiter is smooth and differentiable, making it well-suited for
implicit methods and unstructured meshes. The parameter `ε` provides
a smoothing factor for very small values.

# Arguments
- `r::T`: Slope ratio
- `ε::T=1e-6`: Smoothing parameter (default: 1e-6)

# Returns
The Venkatakrishnan limiter value in [0, 1].
"""
@inline function venkatakrishnan(r::T, ε::T=T(1e-6)) where {T<:Real}
    if r < ε
        return zero(T)
    end
    r² = r * r
    return (r² + 2r) / (r² + r + 2)
end

@doc raw"""
    barth_jespersen(φ_center::T, φ_min::T, φ_max::T, φ_face::T) where {T<:Real} -> T

Compute the Barth-Jespersen limiter for unstructured mesh reconstruction.

```math
\phi = \begin{cases}
\min\left(1, \frac{\phi_{\max} - \phi_C}{\phi_f - \phi_C}\right) & \text{if } \phi_f > \phi_C \\
\min\left(1, \frac{\phi_{\min} - \phi_C}{\phi_f - \phi_C}\right) & \text{if } \phi_f < \phi_C \\
1 & \text{if } \phi_f = \phi_C
\end{cases}
```

This limiter preserves local extrema and is commonly used for
gradient reconstruction on unstructured meshes.

# Arguments
- `φ_center::T`: Cell center value
- `φ_min::T`: Minimum value in neighborhood
- `φ_max::T`: Maximum value in neighborhood
- `φ_face::T`: Reconstructed face value

# Returns
The limiting factor in [0, 1].
"""
@inline function barth_jespersen(φ_center::T, φ_min::T, φ_max::T, φ_face::T) where {T<:Real}
    Δ = φ_face - φ_center
    if abs(Δ) < eps(T)
        return one(T)
    elseif Δ > zero(T)
        return min(one(T), (φ_max - φ_center) / Δ)
    else
        return min(one(T), (φ_min - φ_center) / Δ)
    end
end

@doc raw"""
    koren(r::T, β::T=T(1//3)) where {T<:Real} -> T

Compute the Koren limiter for a slope ratio.

```math
\phi(r) = \max\left(0, \min\left(2r, \min\left(\frac{1 + 2r}{3}, 2\right)\right)\right)
```

This limiter provides third-order accuracy for smooth solutions
while maintaining TVD properties.

# Arguments
- `r::T`: Slope ratio
- `β::T=1/3`: Parameter controlling the scheme order (default: 1/3 for third-order)

# Returns
The Koren limiter value.
"""
@inline function koren(r::T, β::T=T(1//3)) where {T<:Real}
    return max(zero(T), min(2r, min((one(T) + 2r) / 3, T(2))))
end

@doc raw"""
    ospre(r::T) where {T<:Real} -> T

Compute the OSPRE (Optimized Second-order Polynomial Ratio for Edges) limiter.

```math
\phi(r) = \frac{1.5(r^2 + r)}{r^2 + r + 1}
```

This limiter provides a smooth, symmetric limiting function that is
optimized for second-order accuracy.

# Arguments
- `r::T`: Slope ratio

# Returns
The OSPRE limiter value.
"""
@inline function ospre(r::T) where {T<:Real}
    r² = r * r
    denom = r² + r + one(T)
    if denom < eps(T)
        return zero(T)
    end
    return T(1.5) * (r² + r) / denom
end

"""
    apply_limiter(::MinmodLimiter, a::T, b::T) where {T<:Real}

Apply the minmod limiter.
"""
@inline apply_limiter(::MinmodLimiter, a::T, b::T) where {T<:Real} = minmod(a, b)

"""
    apply_limiter(::SuperbeeLimiter, a::T, b::T) where {T<:Real}

Apply the superbee limiter.
"""
@inline apply_limiter(::SuperbeeLimiter, a::T, b::T) where {T<:Real} = superbee(a, b)

"""
    apply_limiter(::VanLeerLimiter, a::T, b::T) where {T<:Real}

Apply the van Leer limiter.
"""
@inline apply_limiter(::VanLeerLimiter, a::T, b::T) where {T<:Real} = van_leer(a, b)

"""
    apply_limiter(::VenkatakrishnanLimiter, r::T) where {T<:Real}

Apply the Venkatakrishnan limiter.
"""
@inline apply_limiter(::VenkatakrishnanLimiter, r::T) where {T<:Real} = venkatakrishnan(r)

"""
    apply_limiter(::KorenLimiter, r::T) where {T<:Real}

Apply the Koren limiter.
"""
@inline apply_limiter(::KorenLimiter, r::T) where {T<:Real} = koren(r)

"""
    apply_limiter(::OspreLimiter, r::T) where {T<:Real}

Apply the OSPRE limiter.
"""
@inline apply_limiter(::OspreLimiter, r::T) where {T<:Real} = ospre(r)

@doc raw"""
    compute_slope_ratio(φ_L::T, φ_C::T, φ_R::T) where {T<:Real} -> T

Compute the slope ratio for TVD limiters.

```math
r = \frac{\phi_C - \phi_L}{\phi_R - \phi_C}
```

# Arguments
- `φ_L::T`: Left/upstream value
- `φ_C::T`: Center value
- `φ_R::T`: Right/downstream value

# Returns
The slope ratio, or a large value if the denominator is near zero.
"""
@inline function compute_slope_ratio(φ_L::T, φ_C::T, φ_R::T) where {T<:Real}
    Δ_down = φ_R - φ_C
    Δ_up = φ_C - φ_L
    if abs(Δ_down) < eps(T)
        return sign(Δ_up) * T(1e10)
    end
    return Δ_up / Δ_down
end

@doc raw"""
    select_limiter(problem_type::Symbol, mesh_type::Symbol=:structured) -> AbstractLimiter

Select an appropriate limiter based on problem characteristics.

# Arguments
- `problem_type::Symbol`: One of `:conservative`, `:accuracy`, `:shock_capturing`
- `mesh_type::Symbol=:structured`: One of `:structured`, `:unstructured`

# Returns
An appropriate limiter instance.
"""
function select_limiter(problem_type::Symbol, mesh_type::Symbol=:structured)
    if mesh_type == :unstructured
        return VenkatakrishnanLimiter()
    end

    if problem_type == :conservative
        return MinmodLimiter()
    elseif problem_type == :accuracy
        return VanLeerLimiter()
    elseif problem_type == :shock_capturing
        return SuperbeeLimiter()
    else
        return VanLeerLimiter()
    end
end
