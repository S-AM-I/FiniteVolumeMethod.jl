# In-situ monitoring — migrated from Simu.jl SimuIO InSitu module
# Provides probe sampling and integral monitoring without full I/O.

"""Abstract supertype for in-situ monitoring probes and diagnostics."""
abstract type AbstractMonitor end

"""
    Probe(location::Vector{Float64}, field_name::String, cell_index::Int)

A point probe that records the value of a field at a specific location.
"""
struct Probe <: AbstractMonitor
    location::Vector{Float64}
    field_name::String
    cell_index::Int  # Cached cell index
end

"""
    Probe(mesh, location::Vector{Float64}, field_name::String)

Constructor that locates the cell containing the probe.
"""
function Probe(mesh, location::Vector{Float64}, field_name::String)
    idx = find_cell_containing(mesh, location)
    if idx == 0
        @warn "Probe at $location is outside the mesh."
    end
    return Probe(location, field_name, idx)
end

"""
    IntegralMonitor(field_name::String; region=:volume)

Monitor that computes the integral of a field over the domain (volume) or boundary.
"""
struct IntegralMonitor <: AbstractMonitor
    field_name::String
    region::Symbol  # :volume, or boundary name (not fully impl)
end

IntegralMonitor(field_name::String; region::Symbol = :volume) =
    IntegralMonitor(field_name, region)

# --- Implementation ---

"""
    find_cell_containing(mesh::UnstructuredMesh3D, p::Vector{Float64})

Find the cell index containing point p. Returns 0 if not found.
"""
function find_cell_containing(mesh::UnstructuredMesh3D, p::Vector{Float64})
    # Simple bounding-box linear search
    for (i, c) in enumerate(mesh.cells)
        min_c = [Inf, Inf, Inf]
        max_c = [-Inf, -Inf, -Inf]
        for n in c.nodes
            min_c = min.(min_c, [n.x, n.y, n.z])
            max_c = max.(max_c, [n.x, n.y, n.z])
        end

        if all(p .>= min_c) && all(p .<= max_c)
            return i
        end
    end
    return 0
end

function find_cell_containing(mesh::Mesh1D, p::Vector{Float64})
    x = p[1]
    if x < 0 || x > mesh.cells[end].nodes[2].x
        return 0
    end
    for (i, c) in enumerate(mesh.cells)
        if x >= c.nodes[1].x && x <= c.nodes[2].x
            return i
        end
    end
    return 0
end

"""
    sample_probe(probe::Probe, mesh, u)

Get the value at the probe.
`u` is the field vector (cell-centered).
"""
function sample_probe(probe::Probe, mesh, u::AbstractVector)
    if probe.cell_index > 0 && probe.cell_index <= length(u)
        # 0th order interpolation (nearest cell center)
        return u[probe.cell_index]
    else
        return NaN
    end
end

"""
    compute_integral(monitor::IntegralMonitor, mesh, u)

Compute integral of u dV over the domain.
"""
function compute_integral(monitor::IntegralMonitor, mesh, u::AbstractVector)
    if monitor.region == :volume
        total = 0.0
        if hasproperty(mesh, :cells)
            for i in 1:length(mesh.cells)
                total += u[i] * mesh.cells[i].volume
            end
        elseif hasproperty(mesh, :cell_volumes)
            for i in eachindex(u)
                total += u[i] * mesh.cell_volumes[i]
            end
        else
            total = sum(u)
        end
        return total
    else
        # Boundary integral: sum(u[owner_cell] * face_area) over
        # boundary faces matching the patch name in monitor.region.
        patch = monitor.region
        total = 0.0
        if hasproperty(mesh, :face_cells) && hasproperty(mesh, :face_areas)
            nf = size(mesh.face_cells, 2)
            for f in 1:nf
                # Boundary face: neighbour == 0
                mesh.face_cells[2, f] == 0 || continue
                tag = if hasproperty(mesh, :face_tags) && mesh.face_tags !== nothing
                    mesh.face_tags[f]
                else
                    :boundary
                end
                tag === patch || continue
                P = mesh.face_cells[1, f]
                if P >= 1 && P <= length(u)
                    total += u[P] * mesh.face_areas[f]
                end
            end
        elseif hasproperty(mesh, :cells)
            # Legacy structured mesh: sum boundary face contributions
            # from cells on the boundary matching the patch name
            for (i, c) in enumerate(mesh.cells)
                for face in c.faces
                    if face.is_boundary
                        total += u[i] * face.area
                    end
                end
            end
        end
        return total
    end
end
