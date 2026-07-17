# ============================================================
# Geometry — meshes, coordinate systems, cell-vertex geometry
# ============================================================
#
# First real submodule of the v4 reorganization (Stage 3a). Owns every mesh
# type in the package (parabolic structured/curvilinear/unstructured, the
# FVM wrappers, the hyperbolic structured meshes, and the Delaunay
# cell-vertex `FVMGeometry`), the unified mesh supertype `AbstractFVMesh`,
# coordinate systems, and mesh I/O (Gmsh, OpenFOAM polyMesh, PLY/VTK).
#
# Include order is load-bearing: abstract roots first, `coordinate_systems`
# before `fvm_geometry` (struct-definition-time reference), and
# `generic_interface` after every concrete mesh it touches.

module Geometry

using LinearAlgebra: LinearAlgebra, norm, cross, dot, det
using Printf: Printf, @printf, @sprintf
using StaticArrays: SVector
using DelaunayTriangulation: DelaunayTriangulation, Triangulation, statistics,
    each_solid_triangle, triangle_vertices, get_point, getxy,
    num_solid_triangles, get_adjacent, get_boundary_edge_map

include("abstract_mesh.jl")
include("parabolic_abstract_types.jl")
include("parabolic_mesh_types.jl")
include("parabolic_structured.jl")
include("parabolic_curvilinear.jl")
include("parabolic_unstructured.jl")
include("fvm_mesh.jl")
include("parabolic_mesh_io.jl")
include("polyhedral_volumes.jl")
include("convert.jl")
include("openfoam_io.jl")
include("openfoam_writer.jl")
include("quality.jl")
include("gmsh_reader.jl")
include("partitioning.jl")
include("coordinate_systems.jl")
include("fvm_geometry.jl")
include("generic_interface.jl")
include("structured_mesh.jl")
include("structured_mesh_3d.jl")
include("unstructured_hyperbolic_mesh.jl")

export
    # Unified mesh hierarchy
    AbstractFVMesh,
    AbstractFiniteVolumeMesh,
    AbstractMesh,
    AbstractFVMMesh,
    AbstractParabolicMesh,
    AbstractGeometry,
    AbstractGeometryComponent,
    AbstractNode,
    AbstractCell,
    AbstractFace,
    AbstractFVMBoundaryCondition,
    CellType,
    CT_Tetrahedron,
    CT_Hexahedron,
    CT_Prism,
    CT_Pyramid,
    CT_Polyhedron,
    # Parabolic structured meshes
    Node1D,
    Cell1D,
    Face1D,
    Mesh1D,
    Node2D,
    Cell2D,
    Face2D,
    Mesh2D,
    Node3D,
    Cell3D,
    Face3D,
    Mesh3D,
    generate_mesh_1d,
    generate_mesh_1d_nonuniform,
    generate_mesh_2d,
    generate_mesh_2d_nonuniform,
    generate_mesh_3d,
    generate_mesh_3d_nonuniform,
    # Curvilinear meshes
    CurvilinearMesh2D,
    CurvilinearMesh3D,
    get_cell_center,
    get_face_geo,
    # Parabolic unstructured meshes
    UnstructuredFace2D,
    UnstructuredCell2D,
    UnstructuredMesh2D,
    UnstructuredFace3D,
    UnstructuredCell3D,
    UnstructuredMesh3D,
    convert_to_unstructured,
    check_mesh_quality,
    refine_uniform,
    # FVM mesh wrappers + builders
    dim_of,
    n_cells,
    n_faces,
    StructuredFVMMesh,
    CurvilinearFVMMesh,
    UnstructuredFVMMesh,
    validate_mesh,
    is_internal_face,
    owner,
    neighbour,
    face_center,
    face_normal_area,
    face_weight,
    owner_neighbour_distance,
    build_structured_mesh3d,
    build_axisymmetric_rz_mesh,
    structured_boundary_tags,
    build_curvilinear_mesh,
    polygon_area,
    parse_ply,
    parse_vtk,
    tag_unstructured_faces_by_bounds,
    build_unstructured_from_polygons,
    load_unstructured_mesh,
    # Mesh I/O
    read_gmsh,
    volume_tet,
    volume_hex,
    build_faces_from_cells,
    get_cell_faces,
    write_vtk_unstructured,
    read_openfoam_polymesh,
    write_openfoam_field,
    convert_to_fvm_mesh,
    volume_prism,
    volume_pyramid,
    MeshQualityReport,
    print_mesh_quality,
    # Partitioning
    PartitionedMesh,
    partition_mesh_rcb,
    recursive_bisection,
    extract_submesh,
    # Coordinate systems
    AbstractCoordinateSystem,
    Cartesian,
    Cylindrical,
    Spherical,
    geometric_volume_weight,
    geometric_flux_weight,
    # Cell-vertex geometry (DelaunayTriangulation-based)
    FVMGeometry,
    TriangleProperties,
    get_triangle_props,
    _safe_get_triangle_props,
    get_volume,
    # Hyperbolic structured meshes + interface verbs
    StructuredMesh1D,
    StructuredMesh2D,
    StructuredMesh3D,
    UnstructuredHyperbolicMesh,
    ncells,
    nfaces,
    nedges,
    ndims_mesh,
    cell_center,
    cell_volume,
    face_area,
    face_normal,
    face_owner,
    face_neighbor,
    face_position,
    cell_ij,
    cell_idx,
    cell_ijk,
    cell_idx_3d

end # module Geometry
