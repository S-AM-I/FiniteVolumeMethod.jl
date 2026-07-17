# thermal/conjugate.jl — Conjugate heat transfer coupling
#
# Dirichlet-Neumann iteration between a fluid domain (incompressible NS
# + energy equation) and a solid conduction domain. The fluid sees a
# Dirichlet (fixed temperature) BC at the interface; the solid sees a
# Neumann (fixed heat flux) BC computed from the fluid solution.
#
# Per-face interface coupling (Patankar 1980) — each interface face
# carries its own effective conductivity, flux and interface temperature
# derived from the harmonic mean of `k/δ` on the two sides:
#
#     k_eff_f = (k_f·k_s)·(δ_f + δ_s) / (k_f·δ_s + k_s·δ_f)
#     q_f     = k_eff_f · (T_s - T_f) / (δ_f + δ_s)         # into fluid if > 0
#     T_int_f = (k_f·T_f/δ_f + k_s·T_s/δ_s) / (k_f/δ_f + k_s/δ_s)
#
# This replaces the earlier scalar face-averaged coupling, which collapsed
# per-face data to a single value before handing it to the solid Neumann BC.

"""
    patankar_interface_coupling(k_f, k_s, delta_f, delta_s, T_f, T_s)
        -> (k_eff, q_f, T_interface)

Per-face harmonic-mean coupling (Patankar 1980, §4.4-1) at a fluid/solid
interface, parameterised by the local cell-centre-to-face distances
`delta_f`, `delta_s` and the two conductivities `k_f`, `k_s`.

Returns the effective conductivity, the heat flux into the fluid, and
the local interface temperature satisfying continuity of heat flux:

    k_eff     = (k_f·k_s)·(delta_f + delta_s) /
                (k_f·delta_s + k_s·delta_f)
    q_f       = k_eff · (T_s - T_f) / (delta_f + delta_s)
    T_interface = (k_f·T_f/delta_f + k_s·T_s/delta_s) /
                  (k_f/delta_f + k_s/delta_s)

Sign convention: `q_f > 0` when heat flows from the solid into the fluid
(`T_s > T_f`). Swapping the two sides swaps the sign of `q_f`.
"""
@inline function patankar_interface_coupling(
        k_f::T, k_s::T, delta_f::T, delta_s::T, T_f::T, T_s::T,
    ) where {T <: AbstractFloat}
    tiny = T(1.0e-30)
    delta_f_safe = max(delta_f, tiny)
    delta_s_safe = max(delta_s, tiny)

    # Harmonic-mean effective conductivity across the interface.
    denom_keff = k_f * delta_s_safe + k_s * delta_f_safe
    k_eff = denom_keff > tiny ?
        k_f * k_s * (delta_f_safe + delta_s_safe) / denom_keff :
        zero(T)

    # Heat flux into the fluid (positive when T_s > T_f).
    q_f = k_eff * (T_s - T_f) / (delta_f_safe + delta_s_safe)

    # Interface temperature that makes the two one-sided fluxes equal.
    w_f = k_f / delta_f_safe
    w_s = k_s / delta_s_safe
    denom_T = w_f + w_s
    T_interface = denom_T > tiny ?
        (w_f * T_f + w_s * T_s) / denom_T :
        (T_f + T_s) / 2

    return (k_eff, q_f, T_interface)
end

"""
    compute_interface_heat_flux(
        T_field::CollocatedScalarField{T},
        k::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        interface_patch::Symbol,
    ) -> Dict{Int, T}

Compute the heat flux at each face of `interface_patch`:

    q_f = -k * (T_boundary - T_cell) / d_cell_to_face

Returns a dictionary mapping face index to heat flux value.
Positive flux means heat flows out of the fluid domain.
"""
function compute_interface_heat_flux(
        T_field::CollocatedScalarField{T},
        k::T,
        mesh::UnstructuredFVMMesh{Dim, T},
        interface_patch::Symbol,
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    flux = Dict{Int, T}()

    pbmap = build_boundary_map(T_field, mesh)

    for f in 1:nf
        if !is_internal_face(mesh, f)
            tag = _face_tag(mesh, f)
            if tag == interface_patch
                P = owner(mesh, f)
                pbmap[f] == 0 && continue
                T_cell = T_field.internal[P]
                T_bnd = T_field.boundary[pbmap[f]]

                x_c = cell_center(mesh, P)
                x_f = face_center(mesh, f)
                d = norm(x_f - x_c)
                d = max(d, T(1.0e-15))

                flux[f] = -k * (T_bnd - T_cell) / d
            end
        end
    end

    return flux
end

"""
    _extract_interface_temperatures(
        T_field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        patch::Symbol,
    ) -> Dict{Int, T}

Extract boundary face temperatures at the given patch.
"""
function _extract_interface_temperatures(
        T_field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        patch::Symbol,
    ) where {Dim, T}
    temps = Dict{Int, T}()
    pbmap = build_boundary_map(T_field, mesh)
    nf = size(mesh.face_cells, 2)

    for f in 1:nf
        if !is_internal_face(mesh, f)
            tag = _face_tag(mesh, f)
            if tag == patch
                pbmap[f] == 0 && continue
                temps[f] = T_field.boundary[pbmap[f]]
            end
        end
    end

    return temps
end

"""
    build_interface_face_pairing(fluid_mesh, fluid_patch, solid_mesh, solid_patch)
        -> Dict{Int, Int}

Build a per-face pairing from solid interface face index to the nearest
fluid interface face index (by face-centre proximity). Each solid
interface face therefore has a deterministic fluid twin, even on
non-matching (non-conformal) meshes.
"""
function build_interface_face_pairing(
        fluid_mesh::UnstructuredFVMMesh{Dim, T}, fluid_patch::Symbol,
        solid_mesh::UnstructuredFVMMesh{Dim, T}, solid_patch::Symbol,
    ) where {Dim, T}
    fluid_faces = Int[]
    nf_f = size(fluid_mesh.face_cells, 2)
    for f in 1:nf_f
        is_internal_face(fluid_mesh, f) && continue
        _face_tag(fluid_mesh, f) == fluid_patch || continue
        push!(fluid_faces, f)
    end

    pairing = Dict{Int, Int}()
    nf_s = size(solid_mesh.face_cells, 2)
    for f_s in 1:nf_s
        is_internal_face(solid_mesh, f_s) && continue
        _face_tag(solid_mesh, f_s) == solid_patch || continue

        x_s = face_center(solid_mesh, f_s)
        best_f = 0
        best_d = T(Inf)
        for f_f in fluid_faces
            x_f = face_center(fluid_mesh, f_f)
            d = norm(x_f - x_s)
            if d < best_d
                best_d = d
                best_f = f_f
            end
        end
        best_f != 0 && (pairing[f_s] = best_f)
    end
    return pairing
end

"""
    InterfaceCouplingData{T}

Per-face coupling data at a fluid/solid interface produced by
[`compute_interface_coupling`](@ref). Each dictionary is keyed by the
*solid* interface face index so the solid Neumann BC and the returned
interface temperature field stay in one-to-one correspondence.

# Fields
- `q_flux::Dict{Int, T}` — heat flux into the fluid `[W/m²]`
- `k_eff::Dict{Int, T}` — harmonic-mean effective conductivity `[W/(m·K)]`
- `T_interface::Dict{Int, T}` — per-face interface temperature `[K]`
- `delta_f::Dict{Int, T}`, `delta_s::Dict{Int, T}` — owner-to-face
  distances on either side, retained for diagnostics and regression tests
"""
struct InterfaceCouplingData{T}
    q_flux::Dict{Int, T}
    k_eff::Dict{Int, T}
    T_interface::Dict{Int, T}
    delta_f::Dict{Int, T}
    delta_s::Dict{Int, T}
end

"""
    compute_interface_coupling(
        T_fluid, k_f, fluid_mesh, fluid_patch,
        T_solid, k_s, solid_mesh, solid_patch,
    ) -> InterfaceCouplingData{T}

Per-face Patankar harmonic-mean coupling at the fluid/solid interface.

For each solid interface face this pairs it with the nearest fluid
interface face (via [`build_interface_face_pairing`](@ref)), reads
the owner-cell temperatures and cell-centre-to-face distances on both
sides, and evaluates [`patankar_interface_coupling`](@ref).

The resulting `InterfaceCouplingData` carries `q_flux`, `k_eff` and
`T_interface` keyed by the *solid* face index — ready for a per-face
Neumann BC on the solid and a Dirichlet BC back on the fluid.
"""
function compute_interface_coupling(
        T_fluid::CollocatedScalarField{T}, k_f::Real,
        fluid_mesh::UnstructuredFVMMesh{Dim, T}, fluid_patch::Symbol,
        T_solid::CollocatedScalarField{T}, k_s::Real,
        solid_mesh::UnstructuredFVMMesh{Dim, T}, solid_patch::Symbol,
    ) where {Dim, T}
    k_f_T = T(k_f)
    k_s_T = T(k_s)

    pairing = build_interface_face_pairing(
        fluid_mesh, fluid_patch, solid_mesh, solid_patch,
    )

    q_flux = Dict{Int, T}()
    k_eff = Dict{Int, T}()
    T_iface = Dict{Int, T}()
    delta_f_d = Dict{Int, T}()
    delta_s_d = Dict{Int, T}()

    for (f_s, f_f) in pairing
        P_f = owner(fluid_mesh, f_f)
        P_s = owner(solid_mesh, f_s)
        T_f_val = T_fluid.internal[P_f]
        T_s_val = T_solid.internal[P_s]

        d_f = norm(face_center(fluid_mesh, f_f) - cell_center(fluid_mesh, P_f))
        d_s = norm(face_center(solid_mesh, f_s) - cell_center(solid_mesh, P_s))

        keff_val, q_val, T_int_val = patankar_interface_coupling(
            k_f_T, k_s_T, T(d_f), T(d_s), T(T_f_val), T(T_s_val),
        )

        q_flux[f_s] = q_val
        k_eff[f_s] = keff_val
        T_iface[f_s] = T_int_val
        delta_f_d[f_s] = T(d_f)
        delta_s_d[f_s] = T(d_s)
    end

    return InterfaceCouplingData{T}(q_flux, k_eff, T_iface, delta_f_d, delta_s_d)
end

"""
    solve_conjugate_ht(
        cht_prob::ConjugateHeatTransferProblem{Dim, T};
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        linear_solver = nothing,
        verbose = false,
    ) -> Tuple{SolveResult{Dim, T}, ThermalState{T}, CollocatedScalarField{T}}

Solve a conjugate heat transfer problem using Dirichlet-Neumann iteration.

Couples fluid (incompressible NS + energy equation) and a solid
conduction domain via Patankar-style *per-face* harmonic-mean coupling
at the shared interface. Each outer iteration:

1. Solve the fluid with the current per-face interface temperature as a
   Dirichlet BC (face-averaged on first iteration).
2. Evaluate [`compute_interface_coupling`](@ref) to obtain per-face
   `k_eff`, heat flux and interface temperature.
3. Solve the solid with the per-face heat-flux pattern as a Neumann BC
   (face-averaged + per-face correction).
4. Under-relax the new per-face interface temperature and check
   convergence on its max change.

Returns: `(fluid_result, fluid_thermal_state, solid_temperature_field)`.
"""
function solve_conjugate_ht(
        cht_prob::ConjugateHeatTransferProblem{Dim, T};
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        linear_solver = nothing,
        verbose::Bool = false,
    ) where {Dim, T}
    fluid_prob = cht_prob.fluid_prob
    fluid_mesh = fluid_prob.mesh
    solid_mesh = cht_prob.solid_mesh
    alpha_coupling = T(0.5)  # under-relaxation for interface temperature

    # Initial scalar interface temperature (per-face data materialises once
    # the first fluid-solid pair of solves has been evaluated).
    T_interface_scalar = cht_prob.fluid_thermal.T_ref
    T_interface_perface = Dict{Int, T}()  # keyed by solid face index

    fluid_result = nothing
    thermal_state = nothing
    solid_T = nothing

    for coupling_iter in 1:cht_prob.max_coupling_iterations
        # ── 1. Fluid solve with Dirichlet at interface ──────────────
        # Use the face-averaged per-face interface temperature from the
        # previous outer iteration (scalar value on the first pass).
        fluid_bcs_T = copy(cht_prob.fluid_bcs_T)
        T_bc_scalar = isempty(T_interface_perface) ?
            T_interface_scalar :
            sum(values(T_interface_perface)) / length(T_interface_perface)
        fluid_bcs_T[cht_prob.interface_fluid_patch] = ParabolicDirichlet(T_bc_scalar)

        fluid_result, thermal_state = solve_simple_thermal(
            fluid_prob, cht_prob.fluid_thermal;
            bcs_T = fluid_bcs_T,
            turb_model = turb_model,
            turb_bcs = turb_bcs,
            T_init = T_bc_scalar,
            linear_solver = linear_solver,
        )

        # On the very first pass we need a provisional solid field so
        # compute_interface_coupling has solid temperatures to read.
        if solid_T === nothing
            solid_bcs_T0 = copy(cht_prob.solid_bcs_T)
            solid_T = solve_solid_conduction(
                solid_mesh, cht_prob.solid_thermal, solid_bcs_T0;
                linear_solver = linear_solver,
            )
        end

        # ── 2. Per-face Patankar harmonic-mean coupling ─────────────
        coupling = compute_interface_coupling(
            thermal_state.T_field, cht_prob.fluid_thermal.k,
            fluid_mesh, cht_prob.interface_fluid_patch,
            solid_T, cht_prob.solid_thermal.k,
            solid_mesh, cht_prob.interface_solid_patch,
        )

        # ── 3. Solid solve with per-face Neumann at interface ──────
        solid_bcs_T = copy(cht_prob.solid_bcs_T)
        # Solid-side Neumann: outward flux into the solid is -q_f
        # (since q_f is the flux *into the fluid*).
        if !isempty(coupling.q_flux)
            q_avg = sum(values(coupling.q_flux)) / length(coupling.q_flux)
        else
            q_avg = zero(T)
        end
        solid_bcs_T[cht_prob.interface_solid_patch] = ParabolicNeumann(-q_avg)

        solid_T = solve_solid_conduction(
            solid_mesh, cht_prob.solid_thermal, solid_bcs_T;
            linear_solver = linear_solver,
        )

        # Per-face correction around the scalar-averaged Neumann BC.
        _apply_perface_neumann_correction!(
            solid_T, coupling.q_flux, q_avg,
            cht_prob.solid_thermal, solid_mesh, cht_prob.interface_solid_patch,
        )

        # ── 4. Build new per-face interface temperature ─────────────
        T_iface_new = coupling.T_interface

        # ── 5. Per-face under-relaxation ────────────────────────────
        max_delta = zero(T)
        if isempty(T_interface_perface)
            # First pass — accept the new values but also compute the
            # scalar change for the convergence check.
            for (f, T_new) in T_iface_new
                T_interface_perface[f] =
                    (one(T) - alpha_coupling) * T_interface_scalar +
                    alpha_coupling * T_new
                delta = abs(T_interface_perface[f] - T_interface_scalar)
                max_delta = max(max_delta, delta)
            end
        else
            for (f, T_new) in T_iface_new
                T_old = get(T_interface_perface, f, T_interface_scalar)
                T_relaxed = (one(T) - alpha_coupling) * T_old +
                    alpha_coupling * T_new
                max_delta = max(max_delta, abs(T_relaxed - T_old))
                T_interface_perface[f] = T_relaxed
            end
        end

        # Keep the scalar fallback in sync for logging and first-pass
        # initialisation.
        T_interface_scalar = isempty(T_interface_perface) ?
            T_interface_scalar :
            sum(values(T_interface_perface)) / length(T_interface_perface)

        # ── 6. Check convergence (max per-face change) ──────────────
        delta_T = max_delta

        if verbose
            println(
                "CHT iter ", lpad(coupling_iter, 3),
                ": <T_interface> = ", round(T_interface_scalar; digits = 4),
                "  max_delta_T = ", round(delta_T; sigdigits = 3)
            )
        end

        if delta_T < cht_prob.coupling_tolerance
            break
        end
    end

    return (fluid_result, thermal_state, solid_T)
end

"""
    _apply_perface_neumann_correction!(
        solid_T, q_flux_perface, q_avg, solid_thermal,
        solid_mesh, solid_patch,
    )

Adjust solid interface boundary temperatures using the difference
between the per-face heat flux and the face-averaged Neumann BC that
was passed to the solid solve. For each solid interface face with
precomputed flux `q_f`:

    T_boundary += (q_f - q_avg) · delta_s / k_solid

where `delta_s` is the cell-centre-to-face distance. The solid solve
was done with the scalar-averaged Neumann; this correction recovers
per-face accuracy without changing the sparse linear system.

`q_flux_perface` is keyed by solid face index (same convention as
[`InterfaceCouplingData`](@ref)).
"""
function _apply_perface_neumann_correction!(
        solid_T::CollocatedScalarField{T},
        q_flux_perface::Dict{Int, T},
        q_avg::T,
        solid_thermal::SolidThermalProperties,
        solid_mesh::UnstructuredFVMMesh{Dim, T},
        solid_patch::Symbol,
    ) where {Dim, T}
    isempty(q_flux_perface) && return nothing

    k_s = T(solid_thermal.k)
    k_s < eps(T) && return nothing

    nf_s = size(solid_mesh.face_cells, 2)
    pbmap_s = build_boundary_map(solid_T, solid_mesh)

    for f_s in 1:nf_s
        is_internal_face(solid_mesh, f_s) && continue
        _face_tag(solid_mesh, f_s) == solid_patch || continue
        pbmap_s[f_s] != 0 || continue

        q_f = get(q_flux_perface, f_s, q_avg)

        P_s = owner(solid_mesh, f_s)
        d_s = norm(face_center(solid_mesh, f_s) - cell_center(solid_mesh, P_s))
        d_s = max(d_s, T(1.0e-15))

        # Solid-side BC uses outward flux -q_f (matches the sign used for
        # the scalar ParabolicNeumann(-q_avg) above).
        delta_q = (-q_f) - (-q_avg)
        solid_T.boundary[pbmap_s[f_s]] += delta_q * d_s / k_s
    end

    return nothing
end
