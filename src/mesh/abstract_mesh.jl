"""
    AbstractMesh{Dim}

Abstract supertype for all mesh types in the hyperbolic solver framework.
`Dim` is the spatial dimension (1 or 2).
"""
abstract type AbstractMesh{Dim} end

"""
    ndims_mesh(mesh::AbstractMesh{Dim}) -> Int

Return the spatial dimension of the mesh.
"""
ndims_mesh(::AbstractMesh{Dim}) where {Dim} = Dim

"""
    ncells(mesh::AbstractMesh)

Return the number of cells in the mesh.
"""
function ncells end

"""
    nfaces(mesh::AbstractMesh)

Return the number of internal faces in the mesh.
"""
function nfaces end

"""
    cell_center(mesh::AbstractMesh, i::Int)

Return the coordinates of the center of cell `i`.
"""
function cell_center end

"""
    cell_volume(mesh::AbstractMesh, i::Int)

Return the volume (or area in 2D, length in 1D) of cell `i`.
"""
function cell_volume end

"""
    face_normal(mesh::AbstractMesh, f::Int)

Return the outward-pointing normal vector of face `f` (pointing from owner to neighbor).
"""
function face_normal end

"""
    face_area(mesh::AbstractMesh, f::Int)

Return the area (or length in 2D, 1.0 in 1D) of face `f`.
"""
function face_area end

"""
    face_owner(mesh::AbstractMesh, f::Int)

Return the index of the cell that owns face `f` (the "left" cell).
"""
function face_owner end

"""
    face_neighbor(mesh::AbstractMesh, f::Int)

Return the index of the neighbor cell across face `f` (the "right" cell).
"""
function face_neighbor end
