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
    write_structured_vtk_3d(path, x, y, z, fields; field_type = :cell)

Write a 3D structured rectilinear grid with scalar fields to ASCII VTK
format (.vtk legacy format, RECTILINEAR_GRID).

# Arguments
- `path::AbstractString` — output file path (`.vtk` extension)
- `x::AbstractVector` — x-coordinates of grid lines (length `nx + 1` for cell data)
- `y::AbstractVector` — y-coordinates of grid lines (length `ny + 1`)
- `z::AbstractVector` — z-coordinates of grid lines (length `nz + 1`)
- `fields::Dict{String, AbstractVector}` — named scalar fields (cell or point data)
- `field_type` — `:cell` (cell-centered, default) or `:point`

!!! note "Production use"
    For binary/XML VTK, use `WriteVTK.jl` instead. This function writes
    minimal ASCII VTK for quick visualization.
"""
function write_structured_vtk_3d(
        path::AbstractString,
        x::AbstractVector{<:Real},
        y::AbstractVector{<:Real},
        z::AbstractVector{<:Real},
        fields::Dict{String, <:AbstractVector{<:Real}};
        field_type::Symbol = :cell,
    )
    nx = length(x)
    ny = length(y)
    nz = length(z)

    if field_type === :cell
        n_data = (nx - 1) * (ny - 1) * (nz - 1)
    else
        n_data = nx * ny * nz
    end

    for (name, data) in fields
        length(data) == n_data || error(
            "Field '$name' has $(length(data)) values, expected $n_data for field_type=:$field_type"
        )
    end

    open(path, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, "3D structured grid output")
        println(io, "ASCII")
        println(io, "DATASET RECTILINEAR_GRID")
        println(io, "DIMENSIONS $nx $ny $nz")

        println(io, "X_COORDINATES $nx float")
        for xi in x
            println(io, xi)
        end

        println(io, "Y_COORDINATES $ny float")
        for yi in y
            println(io, yi)
        end

        println(io, "Z_COORDINATES $nz float")
        for zi in z
            println(io, zi)
        end

        if field_type === :cell
            println(io, "CELL_DATA $n_data")
        else
            println(io, "POINT_DATA $n_data")
        end

        for (name, data) in fields
            println(io, "SCALARS $name float 1")
            println(io, "LOOKUP_TABLE default")
            for val in data
                println(io, val)
            end
        end
    end

    return path
end
