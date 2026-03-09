# fsi.jl - Fluid-Structure Interaction
# Migrated from Simu.jl SimuFVM/fsi.jl

abstract type AbstractStructuralModel end

"""
    SpringMassSystem

Simple structure where each boundary node is attached to a spring-mass-damper.
m x'' + c x' + k x = F_fluid
"""
mutable struct SpringMassSystem <: AbstractStructuralModel
    mass::Float64
    damping::Float64
    stiffness::Float64

    # State: displacement and velocity for each node on interface
    displacements::Vector{Float64} # Normal displacement (y-direction)
    velocities::Vector{Float64}

    # Interface nodes
    node_indices::Vector{Int}
    initial_positions::Vector{Vector{Float64}}
end

function SpringMassSystem(m, c, k, node_indices, mesh)
    n = length(node_indices)
    disps = zeros(n)
    vels = zeros(n)

    initial_pos = Vector{Vector{Float64}}(undef, n)
    for (i, idx) in enumerate(node_indices)
        node = mesh.nodes[idx]
        initial_pos[i] = [node.x, node.y]
    end

    return SpringMassSystem(m, c, k, disps, vels, node_indices, initial_pos)
end

"""
    update_structure!(struct_model, forces, dt)

Update structural state using explicit Euler.
`forces`: Vector of scalar forces (pressure * area * normal_projection) at each node.
"""
function update_structure!(sys::SpringMassSystem, forces::Vector{Float64}, dt::Float64)
    n = length(sys.node_indices)

    for i in 1:n
        F_net = forces[i] - sys.stiffness * sys.displacements[i] - sys.damping * sys.velocities[i]
        a = F_net / sys.mass
        sys.velocities[i] += a * dt
        sys.displacements[i] += sys.velocities[i] * dt
    end
    return
end

"""
    deform_mesh!(mesh, sys::SpringMassSystem)

Update mesh node coordinates based on structural displacement.
Uses in-place modification of mutable Node2D.
"""
function deform_mesh!(mesh::UnstructuredMesh2D, sys::SpringMassSystem)
    for (i, node_idx) in enumerate(sys.node_indices)
        x0, y0 = sys.initial_positions[i]
        dy = sys.displacements[i]

        node = mesh.nodes[node_idx]
        node.x = x0
        node.y = y0 + dy
    end

    return update_mesh_geometry!(mesh)
end

"""
    update_mesh_geometry!(mesh)

Recompute geometric properties (centers, volumes, normals) after node movement.
"""
function update_mesh_geometry!(mesh::UnstructuredMesh2D)
    # 1. Update Faces
    for face in mesh.faces
        n1 = face.nodes[1]
        n2 = face.nodes[2]

        # Recompute center
        face.center = [(n1.x + n2.x) / 2.0, (n1.y + n2.y) / 2.0]

        # Recompute area (length)
        dx = n2.x - n1.x
        dy = n2.y - n1.y
        face.area = sqrt(dx^2 + dy^2)

        # Recompute normal
        new_normal = [dy, -dx]
        len = norm(new_normal)
        if len > 0
            new_normal /= len
        end

        if dot(new_normal, face.normal) < 0
            new_normal = -new_normal
        end

        face.normal = new_normal
    end

    # 2. Update Cells
    for cell in mesh.cells
        nodes = cell.nodes
        n = length(nodes)
        cx, cy, area = 0.0, 0.0, 0.0

        # Signed area
        for i in 1:n
            j = i == n ? 1 : i + 1
            xi, yi = nodes[i].x, nodes[i].y
            xj, yj = nodes[j].x, nodes[j].y
            factor = (xi * yj - xj * yi)
            area += factor
            cx += (xi + xj) * factor
            cy += (yi + yj) * factor
        end

        area *= 0.5
        cell.volume = abs(area)
        if area != 0
            factor = 1.0 / (6.0 * area)
            cell.center = [cx * factor, cy * factor]
        end
    end
    return
end
