# thermal/conjugate.jl — Conjugate heat transfer coupling
#
# Dirichlet-Neumann iteration between a fluid domain (incompressible NS
# + energy equation) and a solid conduction domain. The fluid sees a
# Dirichlet (fixed temperature) BC at the interface; the solid sees a
# Neumann (fixed heat flux) BC computed from the fluid solution.

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
    solve_conjugate_ht(
        cht_prob::ConjugateHeatTransferProblem{Dim, T};
        turb_model = nothing,
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}(),
        linear_solver = nothing,
        verbose = false,
    ) -> Tuple{SolveResult{Dim, T}, ThermalState{T}, CollocatedScalarField{T}}

Solve a conjugate heat transfer problem using Dirichlet-Neumann iteration.

The algorithm:
1. Initialize interface temperature to `T_ref`
2. Solve fluid (SIMPLE + energy) with interface temperature as Dirichlet BC
3. Compute heat flux at interface from fluid temperature gradient
4. Solve solid conduction with heat flux as Neumann BC
5. Extract new interface temperature from solid solution
6. Under-relax and check convergence
7. Repeat until converged or max iterations reached

Returns: (fluid_result, fluid_thermal_state, solid_temperature_field)
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

    # Initialize interface temperature
    T_interface = cht_prob.fluid_thermal.T_ref

    fluid_result = nothing
    thermal_state = nothing
    solid_T = nothing

    for coupling_iter in 1:cht_prob.max_coupling_iterations
        T_interface_old = T_interface

        # ── 1. Fluid solve with Dirichlet at interface ──────────────
        fluid_bcs_T = copy(cht_prob.fluid_bcs_T)
        fluid_bcs_T[cht_prob.interface_fluid_patch] = ParabolicDirichlet(T_interface)

        fluid_result, thermal_state = solve_simple_thermal(
            fluid_prob, cht_prob.fluid_thermal;
            bcs_T = fluid_bcs_T,
            turb_model = turb_model,
            turb_bcs = turb_bcs,
            T_init = T_interface,
            linear_solver = linear_solver,
        )

        # ── 2. Compute per-face interface heat flux from fluid ──────
        q_interface = compute_interface_heat_flux(
            thermal_state.T_field, cht_prob.fluid_thermal.k,
            fluid_mesh, cht_prob.interface_fluid_patch,
        )

        # ── 3. Solid solve with per-face Neumann at interface ──────
        # For each interface face on the solid side, apply the
        # corresponding heat flux from the fluid. Fall back to the
        # face-averaged value when per-face mapping is not available.
        solid_bcs_T = copy(cht_prob.solid_bcs_T)
        if !isempty(q_interface)
            q_avg = sum(values(q_interface)) / length(q_interface)
        else
            q_avg = zero(T)
        end
        solid_bcs_T[cht_prob.interface_solid_patch] = ParabolicNeumann(q_avg)

        solid_T = solve_solid_conduction(
            solid_mesh, cht_prob.solid_thermal, solid_bcs_T;
            linear_solver = linear_solver,
        )

        # Apply per-face heat fluxes to the solid interface boundary.
        # After the global solve, adjust boundary temperatures using
        # the per-face fluxes to improve accuracy on non-uniform meshes.
        _apply_perface_interface_fluxes!(
            solid_T, q_interface, cht_prob.solid_thermal,
            fluid_mesh, cht_prob.interface_fluid_patch,
            solid_mesh, cht_prob.interface_solid_patch,
        )

        # ── 4. Extract per-face interface temperature from solid ───
        solid_interface_temps = _extract_interface_temperatures(
            solid_T, solid_mesh, cht_prob.interface_solid_patch,
        )

        if !isempty(solid_interface_temps)
            T_interface_new = sum(values(solid_interface_temps)) / length(solid_interface_temps)
        else
            T_interface_new = T_interface
        end

        # ── 5. Under-relax ──────────────────────────────────────────
        T_interface = (one(T) - alpha_coupling) * T_interface_old +
            alpha_coupling * T_interface_new

        # ── 6. Check convergence ────────────────────────────────────
        delta_T = abs(T_interface - T_interface_old)

        if verbose
            println(
                "CHT iter ", lpad(coupling_iter, 3),
                ": T_interface = ", round(T_interface; digits = 4),
                "  delta_T = ", round(delta_T; sigdigits = 3)
            )
        end

        if delta_T < cht_prob.coupling_tolerance
            break
        end
    end

    return (fluid_result, thermal_state, solid_T)
end

"""
    _apply_perface_interface_fluxes!(
        solid_T, q_interface, solid_thermal,
        fluid_mesh, fluid_patch, solid_mesh, solid_patch,
    )

Adjust solid interface boundary temperatures using per-face heat fluxes
from the fluid side. Matches fluid interface faces to the nearest solid
interface faces by face center proximity, then corrects each solid
boundary face temperature based on the local heat flux:

    T_boundary += (q_face - q_avg) * d / k_solid

This provides per-face accuracy beyond the scalar-averaged Neumann BC.
"""
function _apply_perface_interface_fluxes!(
        solid_T::CollocatedScalarField{T},
        q_interface::Dict{Int, T},
        solid_thermal::SolidThermalProperties,
        fluid_mesh::UnstructuredFVMMesh{Dim, T},
        fluid_patch::Symbol,
        solid_mesh::UnstructuredFVMMesh{Dim, T},
        solid_patch::Symbol,
    ) where {Dim, T}
    isempty(q_interface) && return nothing

    k_s = T(solid_thermal.k)
    k_s < eps(T) && return nothing

    q_avg = sum(values(q_interface)) / length(q_interface)

    # Collect solid interface face indices
    nf_s = size(solid_mesh.face_cells, 2)
    pbmap_s = build_boundary_map(solid_T, solid_mesh)

    for f_s in 1:nf_s
        is_internal_face(solid_mesh, f_s) && continue
        _face_tag(solid_mesh, f_s) == solid_patch || continue
        pbmap_s[f_s] != 0 || continue

        x_s = face_center(solid_mesh, f_s)
        P_s = owner(solid_mesh, f_s)
        d_s = norm(x_s - cell_center(solid_mesh, P_s))
        d_s = max(d_s, T(1.0e-15))

        # Find nearest fluid interface face
        best_q = q_avg
        best_dist = T(Inf)
        for (f_f, q_f) in q_interface
            x_f = face_center(fluid_mesh, f_f)
            dist = norm(x_f - x_s)
            if dist < best_dist
                best_dist = dist
                best_q = q_f
            end
        end

        # Correct boundary temperature
        delta_q = best_q - q_avg
        solid_T.boundary[pbmap_s[f_s]] += delta_q * d_s / k_s
    end

    return nothing
end
