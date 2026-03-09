# Parabolic Mesh Types - Migrated from Simu.jl SimuGeometry/structured.jl
# Concrete struct definitions for structured 1D/2D/3D meshes.
# Mesh types use AbstractParabolicMesh instead of Simu.jl's AbstractMesh.

# --- Concrete types for a 1D mesh ---

"""
    Node1D(x)

Computational node in 1D at position `x`.
"""
mutable struct Node1D <: AbstractNode
    x::Float64
end

"""
    Cell1D(nodes, center, volume)

Finite volume cell in 1D.
"""
struct Cell1D <: AbstractCell
    nodes::Vector{Node1D}
    center::Float64
    volume::Float64
end

"""
    Face1D(nodes, normal, area)

Face (boundary) between cells in 1D.
"""
struct Face1D <: AbstractFace
    nodes::Vector{Node1D}
    normal::Float64
    area::Float64
end

"""
    Mesh1D(nodes, cells, faces)

Structured 1D computational mesh.
"""
struct Mesh1D <: AbstractParabolicMesh
    nodes::Vector{Node1D}
    cells::Vector{Cell1D}
    faces::Vector{Face1D}
end

# --- Concrete types for a 2D mesh ---

"""
    Node2D(x, y)

Computational node in 2D at position `(x, y)`.
"""
mutable struct Node2D <: AbstractNode
    x::Float64
    y::Float64
end

"""
    Cell2D(nodes, center, volume)

Finite volume cell in 2D.
"""
struct Cell2D <: AbstractCell
    nodes::Vector{Node2D}
    center::Vector{Float64}
    volume::Float64
end

"""
    Face2D(nodes, normal, area)

Face between cells in 2D.
"""
struct Face2D <: AbstractFace
    nodes::Vector{Node2D}
    normal::Vector{Float64}
    area::Float64
end

"""
    Mesh2D(nodes, cells, faces, nx, ny, Lx, Ly)

Structured 2D computational mesh.
"""
struct Mesh2D <: AbstractParabolicMesh
    nodes::Vector{Node2D}
    cells::Vector{Cell2D}
    faces::Vector{Face2D}
    nx::Int
    ny::Int
    Lx::Float64
    Ly::Float64
end

# --- Concrete types for a 3D mesh ---

"""
    Node3D(x, y, z)

Computational node in 3D at position `(x, y, z)`.
"""
mutable struct Node3D <: AbstractNode
    x::Float64
    y::Float64
    z::Float64
end

"""
    Cell3D(nodes, center, volume)

Finite volume cell in 3D.
"""
struct Cell3D <: AbstractCell
    nodes::Vector{Node3D}
    center::Vector{Float64}
    volume::Float64
end

"""
    Face3D(nodes, normal, area)

Face between cells in 3D.
"""
struct Face3D <: AbstractFace
    nodes::Vector{Node3D}
    normal::Vector{Float64}
    area::Float64
end

"""
    Mesh3D(nodes, cells, faces, nx, ny, nz, Lx, Ly, Lz)

Structured 3D computational mesh.
"""
struct Mesh3D <: AbstractParabolicMesh
    nodes::Vector{Node3D}
    cells::Vector{Cell3D}
    faces::Vector{Face3D}
    nx::Int
    ny::Int
    nz::Int
    Lx::Float64
    Ly::Float64
    Lz::Float64
end
