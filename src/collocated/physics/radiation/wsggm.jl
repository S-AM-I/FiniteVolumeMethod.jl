# radiation/wsggm.jl — Weighted-Sum-of-Grey-Gases coupling
#
# The `WSGGMModel` type itself is defined in `types.jl`. This file adds
# spatial-domain helpers that use a `WSGGMModel` to produce per-cell
# effective absorption coefficients or total emissivities for CO2/H2O
# participating mixtures, and a high-level solver wrapper that plugs a
# WSGGM-derived P1 or fvDOM solve into the thermal radiation loop.
#
# Primary reference:
#   Smith, T. F., Shen, Z. F., Friedman, J. N. (1982), "Evaluation of
#   Coefficients for the Weighted Sum of Gray Gases Model",
#   J. Heat Transfer 104, 602–608.

"""
    wsggm_total_emissivity(model, T_field, path) -> Vector{T}

Per-cell total emissivity using the WSGGM expansion:
`ε(T_c, L) = Σ_i a_i(T_c) · (1 − exp(−κ_i · L))`.

`path` may be a scalar characteristic length [m] or a per-cell vector
(e.g. derived from mesh-specific mean beam lengths).
"""
function wsggm_total_emissivity(
        model::WSGGMModel{NB, T},
        T_field::CollocatedScalarField{T},
        path,
    ) where {NB, T}
    nc = length(T_field.internal)
    eps = Vector{T}(undef, nc)
    for c in 1:nc
        L_c = path isa AbstractVector ? T(path[c]) : T(path)
        eps[c] = compute_band_emissivity(model, T_field.internal[c], L_c)
    end
    return eps
end

"""
    wsggm_effective_absorption(model, T_field, path) -> Vector{T}

Per-cell effective grey absorption coefficient obtained from the WSGGM
total emissivity via `κ_eff = -ln(1 - ε) / L`, where `ε = ε(T_c, L)`.
Returns a per-cell `Vector{T}` suitable for feeding into a grey
`P1Model(; a = κ_eff)` or `FvDOMModel(; a = κ_eff)` for each outer
SIMPLE iteration.

A tiny floor of `1e-20` guards against the zero-emissivity corner case
(cold window-band-only mixtures) and a ceiling of `1 − 1e-12` guards
against log-of-zero when `ε → 1` at very long paths.
"""
function wsggm_effective_absorption(
        model::WSGGMModel{NB, T},
        T_field::CollocatedScalarField{T},
        path,
    ) where {NB, T}
    nc = length(T_field.internal)
    kappa_eff = Vector{T}(undef, nc)
    for c in 1:nc
        L_c = path isa AbstractVector ? T(path[c]) : T(path)
        eps_c = compute_band_emissivity(model, T_field.internal[c], L_c)
        eps_c = clamp(eps_c, T(1.0e-20), one(T) - T(1.0e-12))
        kappa_eff[c] = -log(one(T) - eps_c) / max(L_c, T(1.0e-20))
    end
    return kappa_eff
end

"""
    solve_wsggm_radiation(
        wsggm_model, T_field, mesh, bcs_G;
        path_length, grey_solver = :p1,
        linear_solver = nothing, solver_config = nothing,
    ) -> CollocatedScalarField{T}

Solve the non-grey radiative transfer problem using the WSGGM expansion
by constructing an effective grey absorption coefficient field and
dispatching to the underlying grey solver (P1 or fvDOM).

`grey_solver = :p1` (default) solves a P1 diffusion problem with
`P1Model(; a = κ_eff)`. `grey_solver = :fvdom` uses an S4 fvDOM set.
"""
function solve_wsggm_radiation(
        wsggm_model::WSGGMModel{NB, T},
        T_field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_G::Dict{Symbol, <:AbstractBoundaryCondition};
        path_length,
        grey_solver::Symbol = :p1,
        linear_solver = nothing,
        solver_config = nothing,
    ) where {NB, T, Dim}
    kappa_eff = wsggm_effective_absorption(wsggm_model, T_field, path_length)

    if grey_solver === :p1
        model = P1Model(; a = kappa_eff)
        return solve_p1_radiation(
            model, T_field, mesh, bcs_G;
            linear_solver = linear_solver, solver_config = solver_config,
        )
    elseif grey_solver === :fvdom
        model = FvDOMModel(; a = kappa_eff, Dim = Dim, order = :S4)
        rad_state = solve_fvdom_radiation(
            model, T_field, mesh, bcs_G;
            linear_solver = linear_solver, solver_config = solver_config,
        )
        return rad_state.G
    else
        error("Unknown grey_solver :$grey_solver. Supported: :p1, :fvdom")
    end
end
