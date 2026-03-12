module FVMHdf5Ext

using FiniteVolumeMethod
using HDF5

"""
    write_solution_hdf5(filename, solution, mesh; fields=Dict(), metadata=Dict())

Write simulation solution data to an HDF5 file.

# Arguments
- `filename::AbstractString`: Output file path.
- `solution`: Solution array (any dimension).
- `mesh`: Mesh object providing domain information.
- `fields::Dict{String,<:AbstractArray}`: Additional named fields to store.
- `metadata::Dict{String,Any}`: Scalar metadata (stored as HDF5 attributes).
"""
function FiniteVolumeMethod.write_solution_hdf5(
        filename::AbstractString,
        solution,
        mesh;
        fields::Dict{String} = Dict{String, Any}(),
        metadata::Dict{String} = Dict{String, Any}(),
    )
    HDF5.h5open(filename, "w") do fid
        fid["solution"] = collect(solution)

        if !isempty(fields)
            g = HDF5.create_group(fid, "fields")
            for (name, data) in fields
                g[name] = collect(data)
            end
        end

        if !isempty(metadata)
            for (key, val) in metadata
                HDF5.attributes(fid)[key] = val
            end
        end
    end
    return filename
end

"""
    read_solution_hdf5(filename; load_fields=true) -> Dict{String,Any}

Read simulation data from an HDF5 file.

Returns a dictionary with keys `"solution"`, `"fields"` (if present),
and `"metadata"`.
"""
function FiniteVolumeMethod.read_solution_hdf5(
        filename::AbstractString;
        load_fields::Bool = true,
    )
    result = Dict{String, Any}()
    HDF5.h5open(filename, "r") do fid
        result["solution"] = read(fid["solution"])

        if load_fields && haskey(fid, "fields")
            fields = Dict{String, Any}()
            g = fid["fields"]
            for name in keys(g)
                fields[name] = read(g[name])
            end
            result["fields"] = fields
        end

        attrs = HDF5.attributes(fid)
        meta = Dict{String, Any}()
        for key in keys(attrs)
            meta[key] = read(attrs[key])
        end
        result["metadata"] = meta
    end
    return result
end

end # module
