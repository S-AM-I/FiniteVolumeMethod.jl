# Boundary condition assembly for the parabolic FVM solver.
#
# Each BC type modifies the linear system  A u = b  assembled by the
# cell-centred finite volume discretization of  ∂u/∂t + ∇·q = S:
#
#   Dirichlet  u = g       →  ghost-node flux  γ/Δx · (g - u_P)  added to row P
#   Neumann    ∂u/∂n = g   →  prescribed flux  g  added to RHS of row P
#   Robin      �� u + β ∂u/∂n = g  →  combined diagonal + RHS modification
#
# Types are prefixed "Parabolic" to avoid name collisions with the
# hyperbolic solver's boundary condition enums.

# ==============================================================================
# 1. Advanced Boundary Condition Types
# ==============================================================================

struct InterfaceBC <: AbstractBoundaryCondition
    value_continuity::Bool
    flux_continuity::Bool
    jump_value::Float64
    interface_tag::Symbol
end
InterfaceBC(; value_continuity = true, flux_continuity = true, jump_value = 0.0, interface_tag = :default) =
    InterfaceBC(value_continuity, flux_continuity, jump_value, interface_tag)

struct ParabolicPeriodicBC <: AbstractBoundaryCondition
    pair::Tuple{Symbol, Symbol}
    shift::Float64
end
ParabolicPeriodicBC(pair::Tuple{Symbol, Symbol}) = ParabolicPeriodicBC(pair, 0.0)
ParabolicPeriodicBC(left::Symbol, right::Symbol) = ParabolicPeriodicBC((left, right), 0.0)

struct ParabolicNonlinearDirichlet <: AbstractBoundaryCondition
    f::Function
end
struct ParabolicNonlinearNeumann <: AbstractBoundaryCondition
    f::Function
end
struct ParabolicNonlinearRobin <: AbstractBoundaryCondition
    f::Function
end

struct ParabolicCoupledBC <: AbstractBoundaryCondition
    fields::Vector{Symbol}
    coefficients::Vector{Float64}
    value::Float64
end

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

struct ParabolicTurbulentWall <: AbstractBoundaryCondition
    roughness::Float64
    ParabolicTurbulentWall(; roughness = 0.0) = new(roughness)
end

# ==============================================================================
# 2. Matrix-based Handlers (1D)
# ==============================================================================
#
# Each handler modifies the global system  A u = b  for the boundary cell i.
# The discrete diffusion flux at a boundary face is  q_f = γ (u_b - u_i) / Δx,
# which contributes +γ/Δx to A[i,i] (diagonal) and +γ/Δx · g to b[i] (RHS).

function handle_boundary_condition!(A, b, diffusion, mesh::Mesh1D, i, bc::ParabolicDirichlet, side, transient)
    # Dirichlet u = g:  flux = γ/Δx · (g - u_i), so A[i,i] += γ/Δx, b[i] += γ/Δx · g
    dx = (side == :left) ? mesh.cells[i].center - mesh.nodes[1].x : mesh.nodes[end].x - mesh.cells[i].center
    gamma = get_diffusion_coefficient(diffusion, mesh, i)
    flux_coeff = gamma / dx
    A[i, i] += flux_coeff
    return b[i] += flux_coeff * bc.value
end

function handle_boundary_condition!(A, b, diffusion, mesh::Mesh1D, i, bc::ParabolicNeumann, side, transient)
    # Neumann ∂u/∂n = g:  prescribed flux enters RHS directly (sign convention: outward normal)
    return b[i] += (side == :left ? -1 : 1) * bc.value
end

function handle_boundary_condition!(A, b, diffusion, mesh::Mesh1D, i, bc::ParabolicRobin, side, transient)
    # Robin  a·u + b·∂u/∂n = c:  eliminate ghost value using the BC to get
    # a combined diagonal and RHS contribution.
    dx = (side == :left) ? mesh.cells[i].center - mesh.nodes[1].x : mesh.nodes[end].x - mesh.cells[i].center
    gamma = get_diffusion_coefficient(diffusion, mesh, i)
    denominator = bc.a * dx + bc.b * gamma
    flux_coeff = gamma * bc.a / denominator
    A[i, i] += flux_coeff
    return b[i] += (side == :left ? 1 : -1) * gamma * bc.c / denominator
end

function handle_advection_boundary_condition!(A, b, advection, mesh::Mesh1D, i, bc::ParabolicDirichlet, side, transient)
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

function handle_advection_boundary_condition!(A, b, advection, mesh::Mesh1D, i, bc::ParabolicNeumann, side, transient)
    v = get_velocity(advection, mesh, i, side)
    return if (side == :left && v >= 0) || (side == :right && v < 0)
        b[i] += bc.value
    else
        A[i, i] += abs(v)
    end
end

function handle_advection_boundary_condition!(A, b, advection, mesh::Mesh1D, i, bc::ParabolicRobin, side, transient)
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

function handle_advection_diffusion_boundary_condition!(A, b, model, mesh::Mesh1D, i, bc::ParabolicDirichlet, side, transient)
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

function handle_boundary_condition_3d_triplet!(Is, Js, Vs, b, diffusion, mesh::Mesh3D, idx, bc::ParabolicDirichlet, side, transient, dx, dy, dz)
    gamma = get_diffusion_coefficient(diffusion, mesh, div(idx - 1, mesh.ny * mesh.nz) + 1, div(mod(idx - 1, mesh.ny * mesh.nz), mesh.nz) + 1, mod(idx - 1, mesh.nz) + 1)
    dist = (side == :left || side == :right) ? dx : ((side == :bottom || side == :top) ? dy : dz)
    area = (side == :left || side == :right) ? dy * dz : ((side == :bottom || side == :top) ? dx * dz : dx * dy)
    flux_coeff = gamma * area / dist
    add_entry!(Is, Js, Vs, idx, idx, flux_coeff)
    return b[idx] += flux_coeff * bc.value
end

function handle_boundary_condition_3d_triplet!(Is, Js, Vs, b, diffusion, mesh::Mesh3D, idx, bc::ParabolicNeumann, side, transient, dx, dy, dz)
    area = (side == :left || side == :right) ? dy * dz : (side == :bottom || side == :top ? dx * dz : dx * dy)
    return b[idx] += (side == :left || side == :bottom || side == :front ? -1 : 1) * bc.value * area
end

function handle_boundary_condition_3d_triplet!(Is, Js, Vs, b, diffusion, mesh::Mesh3D, idx, bc::ParabolicRobin, side, transient, dx, dy, dz)
    gamma = get_diffusion_coefficient(diffusion, mesh, div(idx - 1, mesh.ny * mesh.nz) + 1, div(mod(idx - 1, mesh.ny * mesh.nz), mesh.nz) + 1, mod(idx - 1, mesh.nz) + 1)
    dist = (side == :left || side == :right) ? dx : ((side == :bottom || side == :top) ? dy : dz)
    area = (side == :left || side == :right) ? dy * dz : ((side == :bottom || side == :top) ? dx * dz : dx * dy)
    denominator = bc.a * dist + bc.b * gamma
    flux_coeff = gamma * bc.a * area / denominator
    add_entry!(Is, Js, Vs, idx, idx, flux_coeff)
    return b[idx] += (side == :left || side == :bottom || side == :front ? 1 : -1) * gamma * bc.c * area / denominator
end

function handle_advection_boundary_condition_3d_triplet!(Is, Js, Vs, b, advection, mesh::Mesh3D, idx, bc::ParabolicDirichlet, side, transient, dx, dy, dz, v_face)
    area = (side == :left || side == :right) ? dy * dz : ((side == :bottom || side == :top) ? dx * dz : dx * dy)
    return if (side == :left && v_face >= 0) || (side == :bottom && v_face >= 0) || (side == :front && v_face >= 0)
        b[idx] += v_face * area * bc.value
    elseif (side == :right && v_face < 0) || (side == :top && v_face < 0) || (side == :back && v_face < 0)
        b[idx] += abs(v_face) * area * bc.value
    else
        add_entry!(Is, Js, Vs, idx, idx, abs(v_face) * area)
    end
end

function handle_advection_boundary_condition_3d_triplet!(Is, Js, Vs, b, advection, mesh::Mesh3D, idx, bc::ParabolicNeumann, side, transient, dx, dy, dz, v_face)
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

function handle_advection_diffusion_boundary_condition_3d_triplet!(Is, Js, Vs, b, model, mesh::Mesh3D, idx, bc::ParabolicDirichlet, side, transient, dx, dy, dz, v_face)
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

function handle_boundary_condition_2d!(I, J, V, b, diffusion, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::ParabolicDirichlet, side, transient, dx, dy)
    ny = mesh.ny; i = div(k - 1, ny) + 1; j = mod(k - 1, ny) + 1
    gamma = get_diffusion_coefficient(diffusion, mesh, i, j)
    dist = (side == :left || side == :right) ? dx : dy
    area = (side == :left || side == :right) ? dy : dx
    flux_coeff = gamma * area / dist
    add_entry!(I, J, V, k, k, flux_coeff)
    return b[k] += flux_coeff * bc.value
end

function handle_boundary_condition_2d!(I, J, V, b, diffusion, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::ParabolicNeumann, side, transient, dx, dy)
    area = (side == :left || side == :right) ? dy : dx
    return b[k] += (side == :left || side == :bottom ? -1 : 1) * bc.value * area
end

function handle_boundary_condition_2d!(I, J, V, b, diffusion, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::ParabolicRobin, side, transient, dx, dy)
    ny = mesh.ny; i = div(k - 1, ny) + 1; j = mod(k - 1, ny) + 1
    gamma = get_diffusion_coefficient(diffusion, mesh, i, j)
    dist = (side == :left || side == :right) ? dx : dy
    area = (side == :left || side == :right) ? dy : dx
    denominator = bc.a * dist + bc.b * gamma
    flux_coeff = gamma * bc.a * area / denominator
    add_entry!(I, J, V, k, k, flux_coeff)
    return b[k] += (side == :left || side == :bottom ? 1 : -1) * gamma * bc.c * area / denominator
end

function handle_advection_boundary_condition_2d!(I, J, V, b, advection, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::ParabolicDirichlet, side, transient, dx, dy, v_face)
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

function handle_advection_boundary_condition_2d!(I, J, V, b, advection, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::ParabolicNeumann, side, transient, dx, dy, v_face)
    area = (side == :left || side == :right) ? dy : dx
    return if (side == :left && v_face >= 0) || (side == :bottom && v_face >= 0)
        b[k] += bc.value * area
    else
        add_entry!(I, J, V, k, k, abs(v_face) * area)
    end
end

function handle_advection_boundary_condition_2d!(I, J, V, b, advection, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::ParabolicRobin, side, transient, dx, dy, v_face)
    area = (side == :left || side == :right) ? dy : dx
    return if (side == :left && v_face >= 0) || (side == :bottom && v_face >= 0)
        phi_bc = bc.c / (bc.a + bc.b * v_face)
        b[k] += v_face * area * phi_bc
    else
        add_entry!(I, J, V, k, k, abs(v_face) * area)
    end
end

function handle_advection_diffusion_boundary_condition_2d!(I, J, V, b, model, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::ParabolicDirichlet, side, transient, dx, dy, v_face)
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

function handle_advection_diffusion_boundary_condition_2d!(I, J, V, b, model, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::ParabolicNeumann, side, transient, dx, dy, v_face)
    area = (side == :left || side == :right) ? dy : dx
    b[k] += (side == :left || side == :bottom ? -1 : 1) * bc.value * area
    return if (side == :left && v_face < 0) || (side == :right && v_face >= 0) || (side == :bottom && v_face < 0) || (side == :top && v_face >= 0)
        add_entry!(I, J, V, k, k, abs(v_face) * area)
    end
end

function handle_advection_diffusion_boundary_condition_2d!(I, J, V, b, model, mesh::Union{Mesh2D, CurvilinearMesh2D}, k, bc::ParabolicRobin, side, transient, dx, dy, v_face)
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

function evaluate_bc(bc::AbstractBoundaryCondition, solution, mesh, cell_idx, side, t = 0.0)
    if bc isa ParabolicDirichlet
        return bc.value
    elseif bc isa ParabolicNeumann
        return bc.value
    elseif bc isa TimeDependentDirichlet
        return evaluate_bc(bc, t).value
    else
        error("BC evaluation not implemented for $(typeof(bc))")
    end
end

# --- Advanced BC implementations ---

function apply_periodic_bc!(A, b, mesh::Mesh1D, bc::ParabolicPeriodicBC)
    nx = length(mesh.cells)
    return if bc.pair == (:left, :right) || bc.pair == (:right, :left)
        A[1, nx] += 1.0; A[1, 1] -= 1.0; b[1] += bc.shift
        A[nx, 1] += 1.0; A[nx, nx] -= 1.0; b[nx] -= bc.shift
    end
end

function apply_periodic_bc!(A, b, mesh::Mesh2D, bc::ParabolicPeriodicBC)
    nx, ny = mesh.nx, mesh.ny
    return if bc.pair == (:left, :right) || bc.pair == (:right, :left)
        for j in 1:ny
            k_left = j; k_right = (nx - 1) * ny + j
            A[k_left, k_right] += 1.0; A[k_left, k_left] -= 1.0; b[k_left] += bc.shift
            A[k_right, k_left] += 1.0; A[k_right, k_right] -= 1.0; b[k_right] -= bc.shift
        end
    elseif bc.pair == (:bottom, :top) || bc.pair == (:top, :bottom)
        for i in 1:nx
            k_bottom = (i - 1) * ny + 1; k_top = (i - 1) * ny + ny
            A[k_bottom, k_top] += 1.0; A[k_bottom, k_bottom] -= 1.0; b[k_bottom] += bc.shift
            A[k_top, k_bottom] += 1.0; A[k_top, k_top] -= 1.0; b[k_top] -= bc.shift
        end
    end
end

function apply_periodic_bc!(A, b, mesh::Mesh3D, bc::ParabolicPeriodicBC)
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    return if bc.pair == (:left, :right) || bc.pair == (:right, :left)
        for j in 1:ny, k in 1:nz
            idx_left = (j - 1) * nz + k; idx_right = (nx - 1) * ny * nz + (j - 1) * nz + k
            A[idx_left, idx_right] += 1.0; A[idx_left, idx_left] -= 1.0; b[idx_left] += bc.shift
            A[idx_right, idx_left] += 1.0; A[idx_right, idx_right] -= 1.0; b[idx_right] -= bc.shift
        end
    elseif bc.pair == (:bottom, :top) || bc.pair == (:top, :bottom)
        for i in 1:nx, k in 1:nz
            idx_bottom = (i - 1) * ny * nz + k; idx_top = (i - 1) * ny * nz + (ny - 1) * nz + k
            A[idx_bottom, idx_top] += 1.0; A[idx_bottom, idx_bottom] -= 1.0; b[idx_bottom] += bc.shift
            A[idx_top, idx_bottom] += 1.0; A[idx_top, idx_top] -= 1.0; b[idx_top] -= bc.shift
        end
    elseif bc.pair == (:front, :back) || bc.pair == (:back, :front)
        for i in 1:nx, j in 1:ny
            idx_front = (i - 1) * ny * nz + (j - 1) * nz + 1; idx_back = (i - 1) * ny * nz + (j - 1) * nz + nz
            A[idx_front, idx_back] += 1.0; A[idx_front, idx_front] -= 1.0; b[idx_front] += bc.shift
            A[idx_back, idx_front] += 1.0; A[idx_back, idx_back] -= 1.0; b[idx_back] -= bc.shift
        end
    end
end

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

function linearize_nonlinear_bc(bc::ParabolicNonlinearDirichlet, phi::Float64, grad_phi::Float64, x::Float64, t::Float64)
    f_val = bc.f(phi, grad_phi, x, t)
    eps = 1.0e-6
    f_pert = bc.f(phi + eps, grad_phi, x, t)
    df_dphi = (f_pert - f_val) / eps
    if abs(df_dphi) > 1.0e-12
        return ParabolicDirichlet(phi - f_val / df_dphi)
    else
        return ParabolicDirichlet(phi)
    end
end

function linearize_nonlinear_bc(bc::ParabolicNonlinearNeumann, phi::Float64, grad_phi::Float64, x::Float64, t::Float64)
    flux_val = bc.f(phi, grad_phi, x, t)
    return ParabolicNeumann(flux_val)
end

function linearize_nonlinear_bc(bc::ParabolicNonlinearRobin, phi::Float64, grad_phi::Float64, x::Float64, t::Float64)
    a, b, c = bc.f(phi, grad_phi, x, t)
    return ParabolicRobin(a, b, c)
end
