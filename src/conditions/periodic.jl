@doc raw"""
    Periodic Boundary Conditions Module

This module provides support for periodic boundary conditions on unstructured
triangular meshes. Periodic BCs connect two boundary segments such that the
solution on one segment equals the solution on the paired segment (optionally
with a constant shift).

## Mathematical Formulation

For a periodic pair (Γ₁, Γ₂), the condition enforces:

```math
u(\vb x_1) = u(\vb x_2) + \text{shift}
```

where $\vb x_1 \in \Gamma_1$ and $\vb x_2 \in \Gamma_2$ are corresponding points.

## Usage

1. Define matching boundary segments in the mesh
2. Create a `PeriodicBC` specifying the segment indices to pair
3. The module automatically matches nodes by position

## Notes

On unstructured meshes, nodes are matched by finding the closest corresponding
node on the paired boundary after applying the coordinate transformation.
"""

"""
    PeriodicBC

Periodic boundary condition that connects two boundary segments.

# Fields
- `segment_pair`: Tuple of boundary segment indices `(seg1, seg2)` to connect
- `shift`: Constant shift applied: `u(seg1) = u(seg2) + shift`
- `direction`: Direction of periodicity `:x`, `:y`, or `:custom`
- `transform`: Custom coordinate transform function (for non-axis-aligned periodicity)

# Constructors

    PeriodicBC(seg1::Int, seg2::Int; shift=0.0, direction=:x)
    PeriodicBC(segment_pair::Tuple{Int,Int}; shift=0.0, direction=:x)

# Example

```julia
# Connect boundary segment 1 (left) to segment 3 (right) with x-periodicity
periodic_bc = PeriodicBC(1, 3; direction=:x)

# With a phase shift
periodic_bc_shifted = PeriodicBC(1, 3; shift=1.0, direction=:x)
```
"""
struct PeriodicBC{T<:Real, F}
    segment_pair::Tuple{Int, Int}
    shift::T
    direction::Symbol
    transform::F
end

# Default constructors
function PeriodicBC(seg1::Int, seg2::Int; shift::Real=0.0, direction::Symbol=:x, transform=nothing)
    T = typeof(float(shift))
    if transform === nothing
        transform = identity_transform
    end
    return PeriodicBC{T, typeof(transform)}((seg1, seg2), T(shift), direction, transform)
end

function PeriodicBC(segment_pair::Tuple{Int, Int}; kwargs...)
    return PeriodicBC(segment_pair[1], segment_pair[2]; kwargs...)
end

# Identity transform (no coordinate modification)
identity_transform(x, y) = (x, y)

"""
    PeriodicNodeMapping

Stores the mapping between nodes on paired periodic boundaries.

# Fields
- `node_pairs`: Vector of `(node1, node2)` pairs that should have equal values
- `segment_pair`: The boundary segment indices
- `shift`: The constant shift value
"""
struct PeriodicNodeMapping{T<:Real}
    node_pairs::Vector{Tuple{Int, Int}}
    segment_pair::Tuple{Int, Int}
    shift::T
end

@doc raw"""
    compute_periodic_mapping(mesh::FVMGeometry, bc::PeriodicBC; tol=1e-10)

Compute the node-to-node mapping for a periodic boundary condition.

For each node on segment 1, finds the corresponding node on segment 2
based on position matching after applying the periodicity transform.

# Arguments
- `mesh`: The FVM geometry
- `bc`: The periodic BC specification
- `tol`: Position matching tolerance

# Returns
A `PeriodicNodeMapping` containing the paired nodes.

# Algorithm
1. Extract nodes on each boundary segment
2. For each node on segment 1, find the node on segment 2 with matching
   coordinates (after projecting out the periodic direction)
3. Return the list of matched pairs
"""
function compute_periodic_mapping(mesh::FVMGeometry, bc::PeriodicBC; tol=1e-10)
    tri = mesh.triangulation
    seg1, seg2 = bc.segment_pair

    # Get boundary nodes for each segment
    nodes1 = get_segment_nodes(mesh, seg1)
    nodes2 = get_segment_nodes(mesh, seg2)

    # Get positions
    positions1 = [(getxy(get_point(mesh, n))..., n) for n in nodes1]
    positions2 = [(getxy(get_point(mesh, n))..., n) for n in nodes2]

    # Match nodes based on direction
    node_pairs = Tuple{Int, Int}[]

    for (x1, y1, n1) in positions1
        # Find matching node on segment 2
        best_dist = Inf
        best_n2 = -1

        for (x2, y2, n2) in positions2
            # Compute distance in the non-periodic direction(s)
            dist = compute_periodic_distance(x1, y1, x2, y2, bc.direction)

            if dist < best_dist
                best_dist = dist
                best_n2 = n2
            end
        end

        if best_dist < tol && best_n2 > 0
            push!(node_pairs, (n1, best_n2))
        end
    end

    return PeriodicNodeMapping(node_pairs, bc.segment_pair, bc.shift)
end

"""
    compute_periodic_distance(x1, y1, x2, y2, direction::Symbol)

Compute the distance between two points in the non-periodic direction(s).

For `:x` periodicity, compares y-coordinates.
For `:y` periodicity, compares x-coordinates.
For `:custom`, compares both (requires exact match after transform).
"""
function compute_periodic_distance(x1, y1, x2, y2, direction::Symbol)
    if direction == :x
        # x is periodic, match by y
        return abs(y1 - y2)
    elseif direction == :y
        # y is periodic, match by x
        return abs(x1 - x2)
    else
        # Custom: need exact match
        return sqrt((x1 - x2)^2 + (y1 - y2)^2)
    end
end

"""
    get_segment_nodes(mesh::FVMGeometry, segment_index::Int)

Get all nodes on a given boundary segment.

# Returns
Vector of node indices on the specified boundary segment.
"""
function get_segment_nodes(mesh::FVMGeometry, segment_index::Int)
    tri = mesh.triangulation
    nodes = Int[]

    # Get boundary nodes for this segment
    bn_map = DelaunayTriangulation.get_ghost_vertex_map(tri)

    for (ghost_vertex, seg_idx) in bn_map
        if -ghost_vertex == segment_index
            # Found the segment
            bn = DelaunayTriangulation.get_boundary_nodes(tri, seg_idx)
            nedges = DelaunayTriangulation.num_boundary_edges(bn)

            for i in 1:nedges
                u = DelaunayTriangulation.get_boundary_nodes(bn, i)
                if !(u in nodes)
                    push!(nodes, u)
                end
            end
            # Add last node
            v = DelaunayTriangulation.get_boundary_nodes(bn, nedges + 1)
            if !(v in nodes)
                push!(nodes, v)
            end
            break
        end
    end

    return nodes
end

@doc raw"""
    apply_periodic_constraints!(u, mapping::PeriodicNodeMapping)

Apply periodic boundary constraints to a solution vector.

Enforces `u[n1] = u[n2] + shift` for all paired nodes.

# Arguments
- `u`: Solution vector (modified in-place)
- `mapping`: The periodic node mapping

# Notes
This function averages the values and applies the shift, ensuring
consistency across the periodic boundary.
"""
function apply_periodic_constraints!(u, mapping::PeriodicNodeMapping)
    for (n1, n2) in mapping.node_pairs
        # Average and apply shift symmetrically
        avg = (u[n1] + u[n2] - mapping.shift) / 2
        u[n1] = avg + mapping.shift / 2
        u[n2] = avg - mapping.shift / 2
    end
    return u
end

@doc raw"""
    build_periodic_constraint_matrix(n_nodes::Int, mapping::PeriodicNodeMapping)

Build sparse matrices that enforce periodic constraints via Lagrange multipliers
or direct elimination.

Returns matrices `C` and `d` such that the constraint `C*u = d` enforces periodicity.

# Returns
- `C`: Sparse constraint matrix
- `d`: Constraint RHS vector
"""
function build_periodic_constraint_matrix(n_nodes::Int, mapping::PeriodicNodeMapping)
    n_constraints = length(mapping.node_pairs)
    T = typeof(mapping.shift)

    # Build sparse constraint matrix
    I_idx = Int[]
    J_idx = Int[]
    V = T[]
    d = zeros(T, n_constraints)

    for (k, (n1, n2)) in enumerate(mapping.node_pairs)
        # Constraint: u[n1] - u[n2] = shift
        push!(I_idx, k)
        push!(J_idx, n1)
        push!(V, one(T))

        push!(I_idx, k)
        push!(J_idx, n2)
        push!(V, -one(T))

        d[k] = mapping.shift
    end

    C = sparse(I_idx, J_idx, V, n_constraints, n_nodes)
    return C, d
end

"""
    PeriodicConditions{M}

Container for multiple periodic boundary condition specifications.

# Fields
- `mappings`: Vector of `PeriodicNodeMapping` for each periodic BC pair
"""
struct PeriodicConditions{T<:Real}
    mappings::Vector{PeriodicNodeMapping{T}}
end

"""
    PeriodicConditions(mesh::FVMGeometry, bcs::Vector{PeriodicBC}; tol=1e-10)

Construct periodic conditions from a list of periodic BC specifications.
"""
function PeriodicConditions(mesh::FVMGeometry, bcs::Vector{<:PeriodicBC}; tol=1e-10)
    mappings = [compute_periodic_mapping(mesh, bc; tol=tol) for bc in bcs]
    T = isempty(mappings) ? Float64 : typeof(first(mappings).shift)
    return PeriodicConditions{T}(mappings)
end

"""
    apply_periodic_constraints!(u, pc::PeriodicConditions)

Apply all periodic constraints to a solution vector.
"""
function apply_periodic_constraints!(u, pc::PeriodicConditions)
    for mapping in pc.mappings
        apply_periodic_constraints!(u, mapping)
    end
    return u
end

"""
    has_periodic_conditions(pc::PeriodicConditions)

Check if there are any periodic conditions.
"""
has_periodic_conditions(pc::PeriodicConditions) = !isempty(pc.mappings)
