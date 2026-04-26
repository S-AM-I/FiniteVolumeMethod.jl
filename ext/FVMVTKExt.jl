module FVMVTKExt

using FiniteVolumeMethod
using WriteVTK: WriteVTK

"""
    write_structured_vtk_3d(path, mesh::StructuredMesh3D, data::AbstractArray{<:Real,3}; label="value")

Write a 3D structured grid and scalar cell data to a VTK file using WriteVTK.jl.

`path` is the base filename (without extension). `data` should have dimensions
matching `(mesh.nx, mesh.ny, mesh.nz)`.

Returns the full path of the written file.
"""
function FiniteVolumeMethod.write_structured_vtk_3d(
        path::AbstractString,
        mesh::FiniteVolumeMethod.StructuredMesh3D,
        data::AbstractArray{<:Real, 3};
        label::AbstractString = "value",
    )
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    size(data) == (nx, ny, nz) || error("data dimensions $(size(data)) must match mesh ($nx, $ny, $nz)")

    # Build node coordinates (nx+1, ny+1, nz+1 nodes for nx*ny*nz cells)
    xs = range(mesh.xmin; stop = mesh.xmax, length = nx + 1)
    ys = range(mesh.ymin; stop = mesh.ymax, length = ny + 1)
    zs = range(mesh.zmin; stop = mesh.zmax, length = nz + 1)

    vtk_grid(path, xs, ys, zs) do vtk
        vtk[label, VTKCellData()] = data
    end

    return path
end

end # module
