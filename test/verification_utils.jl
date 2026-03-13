"""
    Verification Utilities for FiniteVolumeMethod.jl

Reusable functions for convergence studies, error computation, and verification
assertions. Used by scripts in `docs/src/literate_verification/` and
`test/` benchmark tests.

References:
- Roache (1998), "Verification and Validation in Computational Science and Engineering"
- Toro (2009), "Riemann Solvers and Numerical Methods for Fluid Dynamics", 3rd ed.
"""

using Test

# ============================================================
# Error Norms
# ============================================================

"""
    l1_error(numerical, exact, dx)

Compute the grid-weighted L1 error: `sum(|numerical - exact| * dx) / sum(dx)`.
For uniform grids, pass a scalar `dx`.
"""
function l1_error(numerical::AbstractVector, exact::AbstractVector, dx)
    n = length(numerical)
    @assert length(exact) == n
    if dx isa Number
        return sum(abs(numerical[i] - exact[i]) for i in 1:n) / n
    else
        @assert length(dx) == n
        return sum(abs(numerical[i] - exact[i]) * dx[i] for i in 1:n) / sum(dx)
    end
end

"""
    l2_error(numerical, exact, dx)

Compute the grid-weighted L2 error: `sqrt(sum((numerical - exact)^2 * dx) / sum(dx))`.
For uniform grids, pass a scalar `dx`.
"""
function l2_error(numerical::AbstractVector, exact::AbstractVector, dx)
    n = length(numerical)
    @assert length(exact) == n
    if dx isa Number
        return sqrt(sum((numerical[i] - exact[i])^2 for i in 1:n) / n)
    else
        @assert length(dx) == n
        return sqrt(sum((numerical[i] - exact[i])^2 * dx[i] for i in 1:n) / sum(dx))
    end
end

"""
    linf_error(numerical, exact)

Compute the L∞ (max absolute) error.
"""
function linf_error(numerical::AbstractVector, exact::AbstractVector)
    @assert length(numerical) == length(exact)
    return maximum(abs(numerical[i] - exact[i]) for i in eachindex(numerical))
end

"""
    all_errors(numerical, exact, dx) -> (L1, L2, Linf)

Compute L1, L2, and L∞ errors in a single pass.
"""
function all_errors(numerical::AbstractVector, exact::AbstractVector, dx)
    return (
        l1_error(numerical, exact, dx),
        l2_error(numerical, exact, dx),
        linf_error(numerical, exact),
    )
end

# ============================================================
# Convergence Rates
# ============================================================

"""
    convergence_rates(errors) -> Vector{Float64}

Compute log2 convergence rates from a sequence of errors at successively
refined grids (each doubling the resolution): `log2(e[i] / e[i+1])`.
Returns a vector of length `length(errors) - 1`.
"""
function convergence_rates(errors::AbstractVector)
    return [log2(errors[i] / errors[i + 1]) for i in 1:(length(errors) - 1)]
end

"""
    convergence_rates(errors, h) -> Vector{Float64}

Compute convergence rates from errors and grid spacings using the log-ratio
formula: `log(e[i]/e[i+1]) / log(h[i]/h[i+1])`. Works for non-uniform
refinement ratios.
"""
function convergence_rates(errors::AbstractVector, h::AbstractVector)
    @assert length(errors) == length(h)
    return [
        log(errors[i] / errors[i + 1]) / log(h[i] / h[i + 1])
            for i in 1:(length(errors) - 1)
    ]
end

# ============================================================
# Assertions
# ============================================================

"""
    assert_convergence_order(rates, expected; tol=0.3, label="")

Assert that all convergence rates exceed `expected - tol`. Uses `@test` so
failures are reported in the standard test framework.
"""
function assert_convergence_order(rates, expected; tol = 0.3, label = "")
    for (i, r) in enumerate(rates)
        @test r > expected - tol
    end
    return
end

"""
    assert_monotone_convergence(errors; label="")

Assert that errors decrease monotonically with grid refinement (each error
is smaller than the previous).
"""
function assert_monotone_convergence(errors; label = "")
    for i in 1:(length(errors) - 1)
        @test errors[i] > errors[i + 1]
    end
    return
end

# ============================================================
# Reference Data I/O
# ============================================================

"""
    load_reference_data(filename) -> Dict

Load reference benchmark data from a JSON file in `test/reference_data/`.
Returns a Dict with the parsed contents.
"""
function load_reference_data(filename)
    path = joinpath(@__DIR__, "reference_data", filename)
    return JSON3.read(read(path, String), Dict)
end

# ============================================================
# Grid Convergence Index (ASME V&V 20-2009)
# ============================================================

"""
    grid_convergence_index(e1, e2, e3, r; safety_factor=1.25)

Compute the ASME V&V 20-2009 Grid Convergence Index from three error norms
on successively coarsened grids with constant refinement ratio `r`.

Arguments:
- `e1`: error on finest grid
- `e2`: error on medium grid
- `e3`: error on coarsest grid
- `r`: grid refinement ratio (e.g. 2.0 for doubling)
- `safety_factor`: GCI safety factor (default 1.25 for 3+ grids)

Returns a named tuple `(p, gci_fine, gci_coarse, asymptotic_ratio)`:
- `p`: observed order of convergence via Richardson extrapolation
- `gci_fine`: fine-grid GCI (uncertainty band on finest solution)
- `gci_coarse`: coarse-grid GCI (uncertainty band on medium solution)
- `asymptotic_ratio`: should be ≈ 1.0 when in asymptotic convergence range

Reference: Roache (1998), ASME V&V 20-2009 Standard.
"""
function grid_convergence_index(e1, e2, e3, r; safety_factor = 1.25)
    p = log(e3 / e2) / log(r)
    gci_fine = safety_factor * abs(e2 - e1) / (r^p - 1)
    gci_coarse = safety_factor * abs(e3 - e2) / (r^p - 1)
    asymptotic_ratio = gci_coarse / (r^p * gci_fine)
    return (; p, gci_fine, gci_coarse, asymptotic_ratio)
end

"""
    assert_gci_asymptotic(ratio; tol=0.1)

Assert that the GCI asymptotic ratio is within `tol` of 1.0, indicating
the solution is in the asymptotic convergence range.
"""
function assert_gci_asymptotic(ratio; tol = 0.1)
    @test abs(ratio - 1.0) < tol
    return
end

"""
    assert_conservation(initial, final; rtol=1e-10, labels=nothing)

Assert that conserved quantities have not drifted beyond `rtol` relative
tolerance between `initial` and `final` values. Both arguments should be
vectors of the same length. Optional `labels` provides names for each
quantity in failure messages.
"""
function assert_conservation(initial, final; rtol = 1.0e-10, labels = nothing)
    @assert length(initial) == length(final)
    for i in eachindex(initial)
        rel_err = abs(final[i] - initial[i]) / (abs(initial[i]) + eps(Float64))
        @test rel_err < rtol
    end
    return
end
