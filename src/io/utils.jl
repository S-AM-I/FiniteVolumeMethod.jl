# I/O utilities — migrated from Simu.jl SimuIO
# Provides output directory management, CSV writing, TOML metadata,
# formatted printing, and file-name helpers.

using DelimitedFiles
using TOML
using Printf

"""
    ensure_output_dirs(base)

Ensure output subdirectories exist; returns NamedTuple with base/data/plots/reports/vtk paths.
"""
function ensure_output_dirs(base::AbstractString)
    dirs = (
        base = base,
        data = joinpath(base, "data"),
        plots = joinpath(base, "plots"),
        reports = joinpath(base, "reports"),
        vtk = joinpath(base, "vtk"),
    )
    for d in (dirs.data, dirs.plots, dirs.reports, dirs.vtk)
        isdir(d) || mkpath(d)
    end
    return dirs
end

"""
    write_csv(dir, filename, cols)

Write columns to CSV at dir/filename; returns path.
"""
function write_csv(dir::AbstractString, filename::AbstractString, cols)
    path = joinpath(dir, filename)
    writedlm(path, cols, ',')
    return path
end

"""
    stringify_keys(x)

Recursively stringify dict keys for TOML writing.
"""
function stringify_keys(x)
    if x isa Dict
        out = Dict{String, Any}()
        for k in sort!(collect(keys(x)); by = key -> string(key))
            v = x[k]
            out[string(k)] = stringify_keys(v)
        end
        return out
    elseif x isa AbstractVector
        return [stringify_keys(v) for v in x]
    else
        return x
    end
end

"""
    write_metadata_toml(dir, filename, meta)

Write metadata Dict to TOML at reports folder; returns path.
"""
function write_metadata_toml(dir::AbstractString, filename::AbstractString, meta::Dict)
    path = joinpath(dir, filename)
    meta_str = stringify_keys(meta)
    open(path, "w") do io
        TOML.print(io, meta_str)
    end
    return path
end

# --- Formatted Printing & File Utilities ---

function print_scientific(value::Real, precision::Int = 3)
    format_str = "%.$(precision)e"
    return Printf.format(Printf.Format(format_str), value)
end

function print_with_units(value::Real, unit::String, precision::Int = 3)
    formatted = Printf.format(Printf.Format("%.$(precision)f"), value)
    return "$(formatted) $(unit)"
end

function print_table_header(headers::Vector{String}, widths::Vector{Int})
    header_line = join([rpad(h, w) for (h, w) in zip(headers, widths)], " | ")
    separator = join([repeat("-", w) for w in widths], "-+-")
    println(header_line)
    return println(separator)
end

function print_table_row(values::Vector, widths::Vector{Int}, formats::Vector{String} = String[])
    if isempty(formats)
        formats = fill("s", length(values))
    end
    row_data = String[]
    for (val, width, fmt) in zip(values, widths, formats)
        if fmt == "f"
            push!(row_data, rpad(Printf.@sprintf("%.3f", val), width))
        elseif fmt == "e"
            push!(row_data, rpad(Printf.@sprintf("%.2e", val), width))
        elseif fmt == "d"
            push!(row_data, rpad(string(val), width))
        else
            push!(row_data, rpad(string(val), width))
        end
    end
    return println(join(row_data, " | "))
end

function print_progress(current::Int, total::Int, operation::String = "Processing")
    percentage = round(100 * current / total, digits = 1)
    Printf.@printf("\r%s: %d/%d (%.1f%%)", operation, current, total, percentage)
    return if current == total
        println()
    end
end

function ensure_extension(filename::String, ext::String)
    ext = startswith(ext, ".") ? ext : "." * ext
    return endswith(filename, ext) ? filename : filename * ext
end

function safe_filename(name::String)
    # Replace common illegal filename characters with underscores
    safe = replace(name, r"[\<\>:\"\/\\\|\?\*]" => "_")
    return replace(safe, r"\s+" => "_")
end
