# thermal/enthalpy_equation.jl — Enthalpy-form energy equation
#
# Assembles the specific-enthalpy transport equation:
#
#   ∂(ρ·h) / ∂t + ∇·(ρ·U·h) = ∇·(k/Cp · ∇h) + ρ·(D·p/Dt)
#                            + Φ_visc + q_rad + Q_gen
#
# Dividing by constant ρ gives the same algebraic structure as the
# temperature form but with diffusivity `k/(ρ·Cp)` applied to `h`.
#
# For constant `Cp` the transformation `h = Cp·(T - T_ref)` makes the
# h-form and T-form equivalent up to a linear shift (proven by
# `test/v_and_v_enthalpy.jl`). Separating them cleanly here is the
# foundation for high-Mach / variable-Cp extensions.

"""
    h_from_T(T_val, T_ref, Cp) -> h

Specific enthalpy relative to the reference state `(T_ref, 0)` for a
constant-Cp fluid:

    h = Cp · (T - T_ref)

This is the single point of truth used by the solvers and V&V tests to
switch between temperature and enthalpy representations.
"""
@inline h_from_T(T_val, T_ref, Cp) = Cp * (T_val - T_ref)

"""
    T_from_h(h_val, T_ref, Cp) -> T

Inverse of [`h_from_T`](@ref) for constant `Cp`:

    T = T_ref + h / Cp
"""
@inline T_from_h(h_val, T_ref, Cp) = T_ref + h_val / Cp

"""
    enthalpy_field_from_temperature(T_field, T_ref, Cp) -> CollocatedScalarField

Build a `CollocatedScalarField{:h}` from a temperature field using
[`h_from_T`](@ref). Internal and boundary storage are both populated so
the resulting field can be used directly as the state vector of the
enthalpy equation assembly.
"""
function enthalpy_field_from_temperature(
        T_field::CollocatedScalarField{T}, T_ref::Real, Cp::Real,
    ) where {T}
    h_field = CollocatedScalarField{T, typeof(T_field.internal)}(
        :h,
        similar(T_field.internal),
        similar(T_field.boundary),
        copy(T_field.boundary_face_indices),
    )
    T_ref_T = T(T_ref)
    Cp_T = T(Cp)
    for c in eachindex(h_field.internal)
        h_field.internal[c] = h_from_T(T_field.internal[c], T_ref_T, Cp_T)
    end
    for b in eachindex(h_field.boundary)
        h_field.boundary[b] = h_from_T(T_field.boundary[b], T_ref_T, Cp_T)
    end
    return h_field
end

"""
    temperature_from_enthalpy!(T_field, h_field, T_ref, Cp)

Write `T = T_ref + h / Cp` into `T_field.internal` and `.boundary`
from a pre-computed enthalpy field.
"""
function temperature_from_enthalpy!(
        T_field::CollocatedScalarField{T}, h_field::CollocatedScalarField{T},
        T_ref::Real, Cp::Real,
    ) where {T}
    T_ref_T = T(T_ref)
    Cp_T = T(Cp)
    for c in eachindex(T_field.internal)
        T_field.internal[c] = T_from_h(h_field.internal[c], T_ref_T, Cp_T)
    end
    for b in eachindex(T_field.boundary)
        T_field.boundary[b] = T_from_h(h_field.boundary[b], T_ref_T, Cp_T)
    end
    return T_field
end

"""
    enthalpy_bcs_from_temperature(bcs_T, T_ref, Cp) -> Dict{Symbol, AbstractBoundaryCondition}

Translate a temperature-BC dictionary into the equivalent enthalpy-BC
dictionary via `h = Cp·(T - T_ref)`:

- `DirichletBC(T_val)`      → `DirichletBC(h_from_T(T_val))`
- `NeumannBC(q/Cp_scale)`   → `NeumannBC(q * Cp)` where `q`
  was originally supplied as a temperature gradient (∂T/∂n)
- `RobinBC(a, b, c)`        → `RobinBC(a, b, c)` with
  `a, c` rescaled to enthalpy units

The Neumann and Robin cases assume the BC value was expressed as a
temperature gradient; for a true heat-flux Neumann in `[W/m²]` the user
should scale the coefficient by `1/Cp` (i.e. supply the value directly).
"""
function enthalpy_bcs_from_temperature(
        bcs_T::Dict{Symbol, <:AbstractBoundaryCondition}, T_ref::Real, Cp::Real,
    )
    out = Dict{Symbol, AbstractBoundaryCondition}()
    for (tag, bc) in bcs_T
        if bc isa DirichletBC
            out[tag] = DirichletBC(h_from_T(bc.value, T_ref, Cp))
        elseif bc isa NeumannBC
            # ∂T/∂n = g  ⇒  ∂h/∂n = Cp · g
            out[tag] = NeumannBC(Cp * bc.value)
        elseif bc isa RobinBC
            # a·T + b·∂T/∂n = c  ⇒
            # (a/Cp)·h + b·∂h/∂n = c - a·T_ref     (with h = Cp·(T-T_ref))
            a_h = bc.a / Cp
            c_h = bc.c - bc.a * T_ref
            out[tag] = RobinBC(a_h, bc.b, c_h)
        else
            out[tag] = bc
        end
    end
    return out
end

"""
    assemble_enthalpy!(
        eq::CollocatedEquation{T},
        h_field::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        alpha_h::Union{T, Vector{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_h::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
    )

Assemble the enthalpy transport equation into `eq`.

After dividing by constant `ρ` the equation has the same algebraic
structure as the temperature form:
- Convection uses the volumetric face flux `phi` directly
- Diffusion uses the enthalpy diffusivity `alpha_h = k / (ρ · Cp)`
  (identical to the thermal diffusivity `alpha_eff` in the T-form for
  constant `Cp`)
- Temporal term has unit coefficient

# Arguments
- `eq` — equation (modified in-place)
- `h_field` — current enthalpy field (for the temporal term)
- `phi` — face volumetric flux from the flow solver
- `alpha_h` — enthalpy diffusivity: scalar or per-cell vector
- `mesh` — unstructured FVM mesh
- `bcs_h` — enthalpy boundary conditions (use
  [`enthalpy_bcs_from_temperature`](@ref) to convert from T-BCs)
- `dt` — time step (`nothing` for steady state)
"""
function assemble_enthalpy!(
        eq::CollocatedEquation{T},
        h_field::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        alpha_h::Union{T, Vector{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_h::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
    ) where {Dim, T}
    # Convection: div(phi · h)
    assemble_convection!(eq, phi, mesh, bcs_h)

    # Diffusion: div(alpha_h · grad(h))
    assemble_laplacian!(eq, alpha_h, mesh, bcs_h)

    # Temporal term (if transient)
    if dt !== nothing
        assemble_ddt_euler!(eq, one(T), h_field.internal, mesh, dt)
    end

    return nothing
end

"""
    compute_alpha_h(k_eff, rho, Cp) -> Vector

Enthalpy diffusivity `alpha_h = k_eff / (ρ · Cp)`. This is numerically
identical to `compute_alpha_eff` for constant `Cp` and is kept as a
named helper so the enthalpy code path stays self-documenting.
"""
function compute_alpha_h(k_eff::Vector{T}, rho::T, Cp::T) where {T}
    rho_Cp = rho * Cp
    alpha = Vector{T}(undef, length(k_eff))
    for c in eachindex(k_eff)
        alpha[c] = k_eff[c] / rho_Cp
    end
    return alpha
end

"""
    solve_enthalpy_equation(
        h_field, phi, alpha_h, mesh, bcs_h;
        dt = nothing, linear_solver = nothing, solver_config = nothing,
    ) -> CollocatedScalarField

Assemble and solve one enthalpy equation step, writing the solution
back into `h_field.internal` and returning the updated field. The
caller is responsible for updating `h_field.boundary` via the BC
application (typically done by the solver wrappers).
"""
function solve_enthalpy_equation(
        h_field::CollocatedScalarField{T},
        phi::FaceFluxField{T},
        alpha_h::Union{T, Vector{T}},
        mesh::UnstructuredFVMMesh{Dim, T},
        bcs_h::Dict{Symbol, <:AbstractBoundaryCondition};
        dt::Union{Nothing, T} = nothing,
        linear_solver = nothing,
        solver_config = nothing,
    ) where {Dim, T}
    eq = CollocatedEquation(mesh)
    assemble_enthalpy!(eq, h_field, phi, alpha_h, mesh, bcs_h; dt = dt)

    sol = _dispatch_solve(to_linear_problem(eq), linear_solver, solver_config, :h)
    nc = length(mesh.cell_volumes)
    for c in 1:nc
        h_field.internal[c] = sol.u[c]
    end

    return h_field
end
