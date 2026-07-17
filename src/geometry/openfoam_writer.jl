# mesh/openfoam_writer.jl — OpenFOAM field writer
#
# Writes cell-centered field data in OpenFOAM ASCII format.  Designed for
# round-trip workflows where a mesh was read with `read_openfoam_polymesh`
# and simulation results need to be exported back into an OpenFOAM case
# directory for post-processing with paraFoam or similar tools.

# -- FoamFile header boilerplate -------------------------------------------

"""
    _write_foam_header(io, class, object)

Write a standard FoamFile header block to `io`.
"""
function _write_foam_header(io::IO, class::AbstractString, object::AbstractString)
    println(io, "FoamFile")
    println(io, "{")
    println(io, "    version     2.0;")
    println(io, "    format      ascii;")
    println(io, "    class       $class;")
    println(io, "    object      $object;")
    println(io, "}")
    println(io)
    return nothing
end

# -- Scalar field writer ---------------------------------------------------

"""
    write_openfoam_field(values, field_name, case_dir, time_dir; class = "volScalarField")

Write a cell-centered scalar field in OpenFOAM ASCII format.

Creates `case_dir/time_dir/field_name` with a FoamFile header followed by
the value list.  Use `time_dir = "0"` for initial conditions or a numeric
string like `"0.5"` for intermediate time directories.

# Arguments
- `values::Vector{T}` — one value per cell
- `field_name::AbstractString` — field name (e.g. `"p"`, `"T"`, `"k"`)
- `case_dir::AbstractString` — path to the OpenFOAM case directory
- `time_dir::AbstractString` — time directory name inside `case_dir`
- `class` — OpenFOAM class string (default `"volScalarField"`)
"""
function write_openfoam_field(
        values::Vector{T},
        field_name::AbstractString,
        case_dir::AbstractString,
        time_dir::AbstractString;
        class::AbstractString = "volScalarField",
    ) where {T}
    dir = joinpath(case_dir, time_dir)
    mkpath(dir)
    path = joinpath(dir, field_name)
    open(path, "w") do io
        _write_foam_header(io, class, field_name)
        println(io, length(values))
        println(io, "(")
        for v in values
            println(io, v)
        end
        println(io, ")")
    end
    return nothing
end

# -- Vector field writer ---------------------------------------------------

"""
    write_openfoam_field(values, field_name, case_dir, time_dir; class = "volVectorField")

Write a cell-centered vector field in OpenFOAM ASCII format.

Each entry is written as `(x y z)`.

# Arguments
- `values::Vector{SVector{3, T}}` — one 3-vector per cell
- `field_name::AbstractString` — field name (e.g. `"U"`)
- `case_dir::AbstractString` — path to the OpenFOAM case directory
- `time_dir::AbstractString` — time directory name inside `case_dir`
- `class` — OpenFOAM class string (default `"volVectorField"`)
"""
function write_openfoam_field(
        values::Vector{SVector{3, T}},
        field_name::AbstractString,
        case_dir::AbstractString,
        time_dir::AbstractString;
        class::AbstractString = "volVectorField",
    ) where {T}
    dir = joinpath(case_dir, time_dir)
    mkpath(dir)
    path = joinpath(dir, field_name)
    open(path, "w") do io
        _write_foam_header(io, class, field_name)
        println(io, length(values))
        println(io, "(")
        for v in values
            println(io, "(", v[1], " ", v[2], " ", v[3], ")")
        end
        println(io, ")")
    end
    return nothing
end
