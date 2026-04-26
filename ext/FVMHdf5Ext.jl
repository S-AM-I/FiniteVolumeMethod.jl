module FVMHdf5Ext

using FiniteVolumeMethod
using HDF5: HDF5

function _hdf5_metadata(mesh, metadata)
    normalized = FiniteVolumeMethod.stringify_keys(Dict{Any, Any}(metadata))
    return merge(
        Dict{String, Any}(
            "solution_format_version" => 1,
            "mesh_type" => string(nameof(typeof(mesh))),
            "solution_writer" => "FiniteVolumeMethod.FVMHdf5Ext",
        ),
        normalized,
    )
end

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
        fields::AbstractDict = Dict{String, Any}(),
        metadata::AbstractDict = Dict{String, Any}(),
    )
    normalized_fields = FiniteVolumeMethod.stringify_keys(Dict{Any, Any}(fields))
    normalized_metadata = _hdf5_metadata(mesh, metadata)

    HDF5.h5open(filename, "w") do fid
        fid["solution"] = collect(solution)

        if !isempty(normalized_fields)
            g = HDF5.create_group(fid, "fields")
            for name in sort!(collect(keys(normalized_fields)); by = string)
                data = normalized_fields[name]
                g[name] = collect(data)
            end
        end

        if !isempty(normalized_metadata)
            for key in sort!(collect(keys(normalized_metadata)); by = string)
                val = normalized_metadata[key]
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
        result["metadata"] = FiniteVolumeMethod.stringify_keys(meta)
    end
    return result
end

end # module
