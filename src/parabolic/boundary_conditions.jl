# Boundary condition assembly for the parabolic FVM solver.
#
# Each BC type modifies the linear system  A u = b  assembled by the
# cell-centred finite volume discretization of  ∂u/∂t + ∇·q = S:
#
#   Dirichlet  u = g       →  ghost-node flux  γ/Δx · (g - u_P)  added to row P
#   Neumann    ∂u/∂n = g   →  prescribed flux  g  added to RHS of row P
#   Robin      �� u + β ∂u/∂n = g  →  combined diagonal + RHS modification
#
# These types live inside the `Parabolic` submodule, so they need no name
# prefix to stay clear of the hyperbolic solver's boundary conditions.

# ==============================================================================
# 1. Advanced Boundary Condition Types
# ==============================================================================

"""
    InterfaceBC(; value_continuity=true, flux_continuity=true, jump_value=0.0, interface_tag=:default)

Interface boundary condition for multi-domain coupling.  Enforces value and/or
flux continuity across an internal interface, with an optional prescribed jump.
"""
struct InterfaceBC <: AbstractBoundaryCondition
    value_continuity::Bool
    flux_continuity::Bool
    jump_value::Float64
    interface_tag::Symbol
end
InterfaceBC(; value_continuity = true, flux_continuity = true, jump_value = 0.0, interface_tag = :default) =
    InterfaceBC(value_continuity, flux_continuity, jump_value, interface_tag)

"""
    StructuredPeriodicBC(pair::Tuple{Symbol,Symbol}, shift=0.0)

Periodic boundary condition for the structured parabolic meshes, identifying
two opposite boundary faces by name (e.g. `(:left, :right)`).  An optional
constant `shift` is added when mapping values across the period.

Not interchangeable with `VertexConditions.PeriodicBC`, which pairs *segment
indices* on an unstructured triangulation: same physical idea, different mesh
representation.
"""
struct StructuredPeriodicBC <: AbstractBoundaryCondition
    pair::Tuple{Symbol, Symbol}
    shift::Float64
end
StructuredPeriodicBC(pair::Tuple{Symbol, Symbol}) = StructuredPeriodicBC(pair, 0.0)
StructuredPeriodicBC(left::Symbol, right::Symbol) = StructuredPeriodicBC((left, right), 0.0)

"""
    OutflowBC(; type=:zero_gradient, pressure=0.0, backflow_prevention=true)

Outflow (open) boundary condition.  Applies a zero-gradient extrapolation by
default so that convected quantities leave the domain without reflection.
`backflow_prevention` adds upwind dissipation when the local velocity points
inward, preventing spurious inflow at outflow faces.
"""
struct OutflowBC <: AbstractBoundaryCondition
    type::Symbol
    pressure::Float64
    backflow_prevention::Bool
end
OutflowBC(; type = :zero_gradient, pressure = 0.0, backflow_prevention = true) =
    OutflowBC(type, pressure, backflow_prevention)

struct BoundaryRegion
    tag::Symbol
    coordinates::Union{Nothing, Function}
    description::String
end
BoundaryRegion(tag::Symbol) = BoundaryRegion(tag, nothing, "")

struct RegionBC <: AbstractBoundaryCondition
    bc::AbstractBoundaryCondition
    region::BoundaryRegion
end

"""
    TurbulentWall(; roughness=0.0)

Wall boundary condition for turbulence-model equations.  Applies standard
wall-function treatment; `roughness` (in mesh length units) activates the
rough-wall log-law when positive.
"""
struct TurbulentWall <: AbstractBoundaryCondition
    roughness::Float64
    TurbulentWall(; roughness = 0.0) = new(roughness)
end

# ==============================================================================
# 2. Matrix-based Handlers (1D)
# ==============================================================================
#
# Each handler modifies the global system  A u = b  for the boundary cell i.
# The discrete diffusion flux at a boundary face is  q_f = γ (u_b - u_i) / Δx,
# which contributes +γ/Δx to A[i,i] (diagonal) and +γ/Δx · g to b[i] (RHS).

function handle_boundary_condition!(A, b, diffusion, mesh::Mesh1D, i, bc::DirichletBC, side, transient)
    # Dirichlet u = g:  flux = γ/Δx · (g - u_i), so A[i,i] += γ/Δx, b[i] += γ/Δx · g
    dx = (side == :left) ? mesh.cells[i].center - mesh.nodes[1].x : mesh.nodes[end].x - mesh.cells[i].center
    gamma = get_diffusion_coefficient(diffusion, mesh, i)
    flux_coeff = gamma / dx
    A[i, i] += flux_coeff
    return b[i] += flux_coeff * bc.value
end

function handle_boundary_condition!(A, b, diffusion, mesh::Mesh1D, i, bc::NeumannBC, side, transient)
    # Neumann ∂u/∂n = g:  prescribed flux enters RHS directly (sign convention: outward normal)
    return b[i] += (side == :left ? -1 : 1) * bc.value
end

function handle_boundary_condition!(A, b, diffusion, mesh::Mesh1D, i, bc::RobinBC, side, transient)
    # Robin  a·u + b·∂u/∂n = c:  eliminate ghost value using the BC to get
    # a combined diagonal and RHS contribution.
    dx = (side == :left) ? mesh.cells[i].center - mesh.nodes[1].x : mesh.nodes[end].x - mesh.cells[i].center
    gamma = get_diffusion_coefficient(diffusion, mesh, i)
    denominator = bc.a * dx + bc.b * gamma
    flux_coeff = gamma * bc.a / denominator
    A[i, i] += flux_coeff
    return b[i] += (side == :left ? 1 : -1) * gamma * bc.c / denominator
end

function handle_advection_boundary_condition!(A, b, advection, mesh::Mesh1D, i, bc::DirichletBC, side, transient)
    v = get_velocity(advection, mesh, i, side)
    return if side == :left
        if v >= 0
            b[i] += v * bc.value
        else
            A[i, i] += abs(v)
        end
    else # side == :right
        if v >= 0
            A[i, i] += v
        else
            b[i] += abs(v) * bc.value
        end
    end
end

function handle_advection_boundary_condition!(A, b, advection, mesh::Mesh1D, i, bc::NeumannBC, side, transient)
    v = get_velocity(advection, mesh, i, side)
    return if (side == :left && v >= 0) || (side == :right && v < 0)
        b[i] += bc.value
    else
        A[i, i] += abs(v)
    end
end

function handle_advection_boundary_condition!(A, b, advection, mesh::Mesh1D, i, bc::RobinBC, side, transient)
    v = get_velocity(advection, mesh, i, side)
    return if (side == :left && v >= 0) || (side == :right && v < 0)
        phi_bc = bc.c / (bc.a + bc.b * abs(v))
        b[i] += abs(v) * phi_bc
    else
        A[i, i] += abs(v)
    end
end

function handle_advection_boundary_condition!(A, b, advection, mesh::Mesh1D, i, bc::OutflowBC, side, transient)
    v = get_velocity(advection, mesh, i, side)
    return if (side == :left && v < 0) || (side == :right && v > 0)
        A[i, i] += abs(v)
    end
end

function handle_advection_diffusion_boundary_condition!(A, b, model, mesh::Mesh1D, i, bc::DirichletBC, side, transient)
    v = get_velocity(model.advection, mesh, i, side)
    gamma = get_diffusion_coefficient(model.diffusion, mesh, i)
    dx = (side == :left) ? mesh.cells[i].center - mesh.nodes[1].x : mesh.nodes[end].x - mesh.cells[i].center
    D = gamma / dx
    F = abs(v)
    return if (side == :left && v >= 0) || (side == :right && v < 0)
        A[i, i] += D
        b[i] += (D + F) * bc.value
    else
        A[i, i] += D + F
        b[i] += D * bc.value
    end
end

# ==============================================================================
# 3. Triplet-based Handlers (2D/3D)
# ==============================================================================

# --- 3D Triplet Handlers ---

function handle_boundary_condition_3d_triplet!(Is, Js, Vs, b, diffusion, mesh::Mesh3D, idx, bc::DirichletBC, side, transient, dx, dy, dz)
    gamma = get_diffusion_coefficient(diffusion, mesh, div(idx - 1, mesh.ny * mesh.nz) + 1, div(mod(idx - 1, mesh.ny * mesh.nz), mesh.nz) + 1, mod(idx - 1, mesh.nz) + 1)
    dist = (side == :left || side == :right) ? dx : ((side == :bottom || side == :top) ? dy : dz)
    area = (side == :left || side == :right) ? dy * dz : ((side == :bottom || side == :top) ? dx * dz : dx * dy)
    flux_coeff = gamma * area / dist
    add_entry!(Is, Js, Vs, idx, idx, flux_coeff)
    return b[idx] += flux_coeff * bc.value
end

function handle_boundary_condition_3d_triplet!(Is, Js, Vs, b, diffusion, mesh::Mesh3D, idx, bc::NeumannBC, side, transient, dx, dy, dz)
    area = (side == :left || side == :right) ? dy * dz : (side == :bottom || side == :top ? dx * dz : dx * dy)
    return b[idx] += (side == :left || side == :bottom || side == :front ? -1 : 1) * bc.value * area
end

function handle_boundary_condition_3d_triplet!(Is, Js, Vs, b, diffusion, mesh::Mesh3D, idx, bc::RobinBC, side, transient, dx, dy, dz)
    gamma = get_diffusion_coefficient(diffusion, mesh, div(idx - 1, mesh.ny * mesh.nz) + 1, div(mod(idx - 1, mesh.ny * mesh.nz), mesh.nz) + 1, mod(idx - 1, mesh.nz) + 1)
    dist = (side == :left || side == :right) ? dx : ((side == :bottom || side == :top) ? dy : dz)
    area = (side == :left || side == :right) ? dy * dz : ((side == :bottom || side == :top) ? dx * dz : dx * dy)
    denominator = bc.a * dist + bc.b * gamma
    flux_coeff = gamma * bc.a * area / denominator
    add_entry!(Is, Js, Vs, idx, idx, flux_coeff)
    return b[idx] += (side == :left || side == :bottom || side == :front ? 1 : -1) * gamma * bc.c * area / denominator
end

function handle_advection_boundary_condition_3d_triplet!(Is, Js, Vs, b, advection, mesh::Mesh3D, idx, bc::DirichletBC, side, transient, dx, dy, dz, v_face)
    area = (side == :left || side == :right) ? dy * dz : ((side == :bottom || side == :top) ? dx * dz : dx * dy)
    return if (side == :left && v_face >= 0) || (side == :bottom && v_face >= 0) || (side == :front && v_face >= 0)
        b[idx] += v_face * area * bc.value
    elseif (side == :right && v_face < 0) || (side == :top && v_face < 0) || (side == :back && v_face < 0)
        b[idx] += abs(v_face) * area * bc.value
    else
        add_entry!(Is, Js, Vs, idx, idx, abs(v_face) * area)
    end
end

function handle_advection_boundary_condition_3d_triplet!(Is, Js, Vs, b, advection, mesh::Mesh3D, idx, bc::NeumannBC, side, transient, dx, dy, dz, v_face)
    area = (side == :left || side == :right) ? dy * dz : ((side == :bottom || side == :top) ? dx * dz : dx * dy)
    return if (side == :left && v_face >= 0) || (side == :bottom && v_face >= 0) || (side == :front && v_face >= 0)
        b[idx] += bc.value * area
    elseif (side == :right && v_face < 0) || (side == :top && v_face < 0) || (side == :back && v_face < 0)
        b[idx] += bc.value * area
    else
        add_entry!(Is, Js, Vs, idx, idx, abs(v_face) * area)
    end
end

function handle_advection_boundary_condition_3d_triplet!(Is, Js, Vs, b, advection, mesh::Mesh3D, idx, bc::OutflowBC, side, transient, dx, dy, dz, v_face)
    return if (side == :left && v_face < 0) || (side == :right && v_face > 0) ||
            (side == :bottom && v_face < 0) || (side == :top && v_face > 0) ||
            (side == :front && v_face < 0) || (side == :back && v_face > 0)
        area = (side == :left || side == :right) ? dy * dz : ((side == :bottom || side == :top) ? dx * dz : dx * dy)
        add_entry!(Is, Js, Vs, idx, idx, abs(v_face) * area)
    end
end

function handle_advection_diffusion_boundary_condition_3d_triplet!(Is, Js, Vs, b, model, mesh::Mesh3D, idx, bc::DirichletBC, side, transient, dx, dy, dz, v_face)
    gamma = model.diffusion.gamma
    dist = (side == :left || side == :right) ? dx : ((side == :bottom || side == :top) ? dy : dz)
    area = (side == :left || side == :right) ? dy * dz : ((side == :bottom || side == :top) ? dx * dz : dx * dy)
    D = gamma * area / dist
    F = abs(v_face) * area
    return if (side == :left && v_face >= 0) || (side == :right && v_face < 0) || (side == :bottom && v_face >= 0) || (side == :top && v_face < 0) || (side == :front && v_face >= 0) || (side == :back && v_face < 0)
        add_entry!(Is, Js, Vs, idx, idx, D)
        b[idx] += (D + F) * bc.value
    else
        add_entry!(Is, Js, Vs, idx, idx, D + F)
        b[idx] += D * bc.value
    end
end

function handle_advection_diffusion_boundary_condition_3d_triplet!(Is, Js, Vs, b, model, mesh::Mesh3D, idx, bc::OutflowBC, side, transient, dx, dy, dz, v_face)
    return if (side == :left && v_face < 0) || (side == :right && v_face > 0) || (side == :bottom && v_face < 0) || (side == :top && v_face > 0) || (side == :front && v_face < 0) || (side == :back && v_face > 0)
        area = (side == :left || side == :right) ? dy * dz : ((side == :bottom || side == :top) ? dx * dz : dx * dy)
        add_entry!(Is, Js, Vs, idx, idx, abs(v_face) * area)
    end
end

# --- 2D Triplet Handlers ---

function handle_boundary_condition_2d!(I, J, V, b, diffusion, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::DirichletBC, side, transient, dx, dy)
    ny = mesh.ny; i = div(k - 1, ny) + 1; j = mod(k - 1, ny) + 1
    gamma = get_diffusion_coefficient(diffusion, mesh, i, j)
    dist = (side == :left || side == :right) ? dx : dy
    area = (side == :left || side == :right) ? dy : dx
    flux_coeff = gamma * area / dist
    add_entry!(I, J, V, k, k, flux_coeff)
    return b[k] += flux_coeff * bc.value
end

function handle_boundary_condition_2d!(I, J, V, b, diffusion, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::NeumannBC, side, transient, dx, dy)
    area = (side == :left || side == :right) ? dy : dx
    return b[k] += (side == :left || side == :bottom ? -1 : 1) * bc.value * area
end

function handle_boundary_condition_2d!(I, J, V, b, diffusion, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::RobinBC, side, transient, dx, dy)
    ny = mesh.ny; i = div(k - 1, ny) + 1; j = mod(k - 1, ny) + 1
    gamma = get_diffusion_coefficient(diffusion, mesh, i, j)
    dist = (side == :left || side == :right) ? dx : dy
    area = (side == :left || side == :right) ? dy : dx
    denominator = bc.a * dist + bc.b * gamma
    flux_coeff = gamma * bc.a * area / denominator
    add_entry!(I, J, V, k, k, flux_coeff)
    return b[k] += (side == :left || side == :bottom ? 1 : -1) * gamma * bc.c * area / denominator
end

function handle_advection_boundary_condition_2d!(I, J, V, b, advection, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::DirichletBC, side, transient, dx, dy, v_face)
    area = (side == :left || side == :right) ? dy : dx
    return if (side == :left && v_face >= 0) || (side == :bottom && v_face >= 0)
        b[k] += v_face * area * bc.value
    elseif (side == :right && v_face < 0) || (side == :top && v_face < 0)
        b[k] += abs(v_face) * area * bc.value
    else
        add_entry!(I, J, V, k, k, abs(v_face) * area)
    end
end

function handle_advection_boundary_condition_2d!(I, J, V, b, advection, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::OutflowBC, side, transient, dx, dy, v_face)
    return if (side == :left && v_face < 0) || (side == :right && v_face > 0) || (side == :bottom && v_face < 0) || (side == :top && v_face > 0)
        area = (side == :left || side == :right) ? dy : dx
        add_entry!(I, J, V, k, k, abs(v_face) * area)
    end
end

function handle_advection_boundary_condition_2d!(I, J, V, b, advection, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::NeumannBC, side, transient, dx, dy, v_face)
    area = (side == :left || side == :right) ? dy : dx
    return if (side == :left && v_face >= 0) || (side == :bottom && v_face >= 0)
        b[k] += bc.value * area
    else
        add_entry!(I, J, V, k, k, abs(v_face) * area)
    end
end

function handle_advection_boundary_condition_2d!(I, J, V, b, advection, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::RobinBC, side, transient, dx, dy, v_face)
    area = (side == :left || side == :right) ? dy : dx
    return if (side == :left && v_face >= 0) || (side == :bottom && v_face >= 0)
        phi_bc = bc.c / (bc.a + bc.b * v_face)
        b[k] += v_face * area * phi_bc
    else
        add_entry!(I, J, V, k, k, abs(v_face) * area)
    end
end

function handle_advection_diffusion_boundary_condition_2d!(I, J, V, b, model, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::DirichletBC, side, transient, dx, dy, v_face)
    ny = mesh.ny; i = div(k - 1, ny) + 1; j = mod(k - 1, ny) + 1
    gamma = get_diffusion_coefficient(model.diffusion, mesh, i, j)
    dist = (side == :left || side == :right) ? dx : dy
    area = (side == :left || side == :right) ? dy : dx
    D = gamma * area / dist
    F = abs(v_face) * area
    return if (side == :left && v_face >= 0) || (side == :right && v_face < 0) || (side == :bottom && v_face >= 0) || (side == :top && v_face < 0)
        add_entry!(I, J, V, k, k, D)
        b[k] += (D + F) * bc.value
    else
        add_entry!(I, J, V, k, k, D + F)
        b[k] += D * bc.value
    end
end

function handle_advection_diffusion_boundary_condition_2d!(I, J, V, b, model, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::NeumannBC, side, transient, dx, dy, v_face)
    area = (side == :left || side == :right) ? dy : dx
    b[k] += (side == :left || side == :bottom ? -1 : 1) * bc.value * area
    return if (side == :left && v_face < 0) || (side == :right && v_face >= 0) || (side == :bottom && v_face < 0) || (side == :top && v_face >= 0)
        add_entry!(I, J, V, k, k, abs(v_face) * area)
    end
end

function handle_advection_diffusion_boundary_condition_2d!(I, J, V, b, model, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::RobinBC, side, transient, dx, dy, v_face)
    ny = mesh.ny; i = div(k - 1, ny) + 1; j = mod(k - 1, ny) + 1
    gamma = get_diffusion_coefficient(model.diffusion, mesh, i, j)
    dist = (side == :left || side == :right) ? dx : dy
    area = (side == :left || side == :right) ? dy : dx
    denominator = bc.a * dist + bc.b * gamma
    flux_coeff = gamma * bc.a * area / denominator
    add_entry!(I, J, V, k, k, flux_coeff)
    b[k] += (side == :left || side == :bottom ? 1 : -1) * gamma * bc.c * area / denominator
    return if (side == :left && v_face < 0) || (side == :right && v_face >= 0) || (side == :bottom && v_face < 0) || (side == :top && v_face >= 0)
        add_entry!(I, J, V, k, k, abs(v_face) * area)
    elseif (side == :left && v_face >= 0) || (side == :bottom && v_face >= 0)
        phi_bc = bc.c / (bc.a + bc.b * abs(v_face))
        b[k] += abs(v_face) * area * phi_bc
    end
end

function handle_advection_diffusion_boundary_condition_2d!(I, J, V, b, model, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::OutflowBC, side, transient, dx, dy, v_face)
    return if (side == :left && v_face < 0) || (side == :right && v_face > 0) || (side == :bottom && v_face < 0) || (side == :top && v_face > 0)
        area = (side == :left || side == :right) ? dy : dx
        add_entry!(I, J, V, k, k, abs(v_face) * area)
    end
end

# ==============================================================================
# 4. Utilities
# ==============================================================================

"""
    UnsupportedBCError(bc, context)

Thrown when a routine is asked to evaluate or apply a boundary condition that
it does not know how to handle. `bc` is the offending BC instance; `context`
is a short string identifying the caller (e.g. `"evaluate_bc(bc, solution, mesh, cell_idx, side, t)"`).
"""
struct UnsupportedBCError <: Exception
    bc::Any
    context::String
end

function Base.showerror(io::IO, e::UnsupportedBCError)
    print(io, "UnsupportedBCError: no ", e.context, " implementation for ")
    print(io, typeof(e.bc))
    print(io, ". ")
    print(io, "Either (a) add a method dispatching on this concrete type, ")
    print(io, "or (b) use one of the BC types with implemented evaluators: ")
    print(io, "DirichletBC, NeumannBC, TimeDependentDirichlet.")
    return
end

function evaluate_bc(bc::AbstractBoundaryCondition, solution, mesh, cell_idx, side, t = 0.0)
    if bc isa DirichletBC
        return bc.value
    elseif bc isa NeumannBC
        return bc.value
    elseif bc isa TimeDependentDirichlet
        return evaluate_bc(bc, t).value
    else
        throw(UnsupportedBCError(bc, "evaluate_bc(bc, solution, mesh, cell_idx, side, t)"))
    end
end

# --- Advanced BC implementations ---

function apply_interface_bc!(A, b, model, mesh, bc::InterfaceBC, cell_left_idx, cell_right_idx)
    if bc.value_continuity
        weight = 1.0e12
        A[cell_left_idx, cell_left_idx] += weight
        A[cell_left_idx, cell_right_idx] -= weight
        b[cell_left_idx] += weight * bc.jump_value
    end
    return if bc.flux_continuity
        if mesh isa Mesh1D
            if model isa Union{Diffusion1D, VariableDiffusion1D}
                gamma_left = get_diffusion_coefficient_at_face(model, mesh, cell_left_idx, :right)
                gamma_right = get_diffusion_coefficient_at_face(model, mesh, cell_right_idx, :left)
                if cell_left_idx < length(mesh.cells)
                    dx_left = mesh.cells[cell_left_idx + 1].center - mesh.cells[cell_left_idx].center
                else
                    dx_left = mesh.nodes[end].x - mesh.cells[cell_left_idx].center
                end
                if cell_right_idx > 1
                    dx_right = mesh.cells[cell_right_idx].center - mesh.cells[cell_right_idx - 1].center
                else
                    dx_right = mesh.cells[cell_right_idx].center - mesh.nodes[1].x
                end
                dx_interface = (dx_left + dx_right) / 2.0
                if gamma_left > 0 && gamma_right > 0
                    gamma_eff = 2.0 * gamma_left * gamma_right / (gamma_left + gamma_right)
                else
                    gamma_eff = 0.0
                end
                flux_coeff = gamma_eff / dx_interface
                A[cell_left_idx, cell_left_idx] += flux_coeff
                A[cell_left_idx, cell_right_idx] -= flux_coeff
                A[cell_right_idx, cell_right_idx] += flux_coeff
                A[cell_right_idx, cell_left_idx] -= flux_coeff
            end
        end
    end
end
