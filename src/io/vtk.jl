# VTK output — migrated from Simu.jl SimuIO
# Provides VTK file writing for 1D polyline data and 3D structured grids.
#
# DEPRECATION NOTE: These functions write minimal ASCII VTK only.
# For production use, prefer WriteVTK.jl which provides:
#   - Binary and XML VTK formats (much smaller files, faster I/O)
#   - Structured, unstructured, multiblock, and time-series VTK
#   - Proper cell/point data with multiple fields
#   - ParaView-compatible .vtu/.vts/.pvd output
#
# Example migration:
#   using WriteVTK
#   vtk_grid("output", xcoords, ycoords, zcoords) do vtk
#       vtk["temperature"] = data
#   end

"""
    write_line_vtk(path, xcoords, scalars; label="value")

Write a simple 1D VTK polyline with scalar point data.
`path` is the full file path for the output .vtk file.

!!! warning "Deprecated"
    Use `WriteVTK.jl` for production VTK output.
"""
function write_line_vtk(path::AbstractString, xcoords::AbstractVector{<:Real}, scalars::AbstractVector{<:Real}; label::AbstractString = "value")
    length(xcoords) == length(scalars) || error("xcoords and scalars length mismatch")
    npts = length(xcoords)
    open(path, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, "Line data")
        println(io, "ASCII")
        println(io, "DATASET POLYDATA")
        println(io, "POINTS $npts float")
        for xi in xcoords
            println(io, "$(xi) 0 0")
        end
        nlines = npts - 1
        println(io, "LINES $nlines $(nlines * 3)")
        for i in 0:(nlines - 1)
            println(io, "2 $i $(i + 1)")
        end
        println(io, "POINT_DATA $npts")
        println(io, "SCALARS $label float 1")
        println(io, "LOOKUP_TABLE default")
        for val in scalars
            println(io, val)
        end
    end
    return path
end

"""
    write_line_vtk(path, mesh::Mesh1D, data, label="value")

Write a 1D mesh and scalar cell data to VTK polyline format.
Uses cell center coordinates from the mesh.

!!! warning "Deprecated"
    Use `WriteVTK.jl` for production VTK output.
"""
function write_line_vtk(path::AbstractString, mesh::Mesh1D, data::AbstractVector{<:Real}, label::AbstractString = "value")
    xcoords = [c.center for c in mesh.cells]
    return write_line_vtk(path, xcoords, data; label = label)
end

"""
    write_structured_vtk_3d()

Placeholder for 3D structured VTK output.

!!! warning "Deprecated"
    Use `WriteVTK.jl` for production VTK output:
    ```julia
    using WriteVTK
    vtk_grid("output", x, y, z) do vtk
        vtk["field"] = data
    end
    ```
"""
function write_structured_vtk_3d()
    # Implementation pending — use WriteVTK.jl instead
    return nothing
end
