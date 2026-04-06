# mesh/polyhedral_volumes.jl — Correct volume computations for polyhedral cells
#
# Computes volumes of prisms and pyramids by decomposing into tetrahedra
# and summing. Uses existing volume_tet from parabolic/mesh/io.jl.
# volume_hex is fixed in-place in io.jl.

using LinearAlgebra: dot, cross

"""
    volume_prism(nodes::Vector{Node3D}) -> Float64

Compute the volume of a triangular prism (wedge) from its 6 vertices
by decomposing into 3 tetrahedra.

Node numbering follows Gmsh convention:
```
  Triangular faces: (0,1,2) bottom, (3,4,5) top
  Quad faces connect corresponding edges
```
"""
function volume_prism(nodes::Vector{Node3D})
    length(nodes) == 6 || error("Prism requires exactly 6 nodes, got $(length(nodes))")
    n = nodes
    # Decompose into 3 tetrahedra
    V = volume_tet([n[1], n[2], n[3], n[4]]) +
        volume_tet([n[2], n[3], n[4], n[6]]) +
        volume_tet([n[2], n[4], n[5], n[6]])
    return V
end

"""
    volume_pyramid(nodes::Vector{Node3D}) -> Float64

Compute the volume of a pyramid from its 5 vertices (4-node base + apex)
by decomposing into 2 tetrahedra.

Node numbering: nodes 1-4 = base quad, node 5 = apex.
"""
function volume_pyramid(nodes::Vector{Node3D})
    length(nodes) == 5 || error("Pyramid requires exactly 5 nodes, got $(length(nodes))")
    n = nodes
    # Split base quad into 2 triangles, each forms a tet with the apex
    V = volume_tet([n[1], n[2], n[3], n[5]]) +
        volume_tet([n[1], n[3], n[4], n[5]])
    return V
end
