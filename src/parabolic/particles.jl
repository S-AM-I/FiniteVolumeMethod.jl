# particles.jl - Lagrangian Particle Tracking
# Migrated from Simu.jl SimuFVM/particles.jl

abstract type AbstractParticle end

mutable struct LagrangianParticle{N, T} <: AbstractParticle
    position::SVector{N, T}
    velocity::SVector{N, T}
    cell_index::Int # Cache for efficient tracking
    id::Int
    active::Bool
    properties::Dict{Symbol, Any}
end

struct ParticleTracker{N, T}
    particles::Vector{LagrangianParticle{N, T}}
    next_id::Ref{Int}
end

function ParticleTracker{N, T}() where {N, T}
    return ParticleTracker(Vector{LagrangianParticle{N, T}}(), Ref(1))
end

"""
    inject_particles!(tracker, positions; initial_velocity=zeros)

Inject particles at specified positions.
"""
function inject_particles!(tracker::ParticleTracker{N, T}, positions::Vector{SVector{N, T}}; initial_velocity = zeros(SVector{N, T})) where {N, T}
    for pos in positions
        id = tracker.next_id[]
        tracker.next_id[] += 1
        p = LagrangianParticle(pos, initial_velocity, 0, id, true, Dict{Symbol, Any}())
        push!(tracker.particles, p)
    end
    return tracker
end

"""
    find_cell_index(mesh, point, start_index=0)

Find the cell containing the point.
Returns cell index or 0 if not found.
Uses a simple linear search for now, or neighbor walk if start_index provided.
"""
function find_cell_index(mesh::UnstructuredMesh2D, point::SVector{2, Float64}, start_index::Int = 0)
    if start_index > 0 && start_index <= length(mesh.cells)
        if is_point_in_cell(mesh, start_index, point)
            return start_index
        end
    end

    for (i, cell) in enumerate(mesh.cells)
        if is_point_in_cell(mesh, i, point)
            return i
        end
    end

    return 0 # Outside domain
end

"""
    is_point_in_cell(mesh, cell_idx, point)

Check if 2D point is inside the cell (assumes convex polygon).
"""
function is_point_in_cell(mesh::UnstructuredMesh2D, cell_idx::Int, point::SVector{2, Float64})
    cell = mesh.cells[cell_idx]

    for f_idx in cell.faces
        face = mesh.faces[f_idx]

        n = SVector{2}(face.normal[1], face.normal[2])
        if face.neighbor == cell_idx
            n = -n
        elseif face.owner != cell_idx
            error("Cell connectivity error")
        end

        c_face = SVector{2}(face.center[1], face.center[2])

        d = point - c_face

        if dot(d, n) > 1.0e-10
            return false
        end
    end
    return true
end

"""
    advect_particles!(tracker, mesh, velocity_field, dt)

Advect particles using Forward Euler and update their cell cache.
`velocity_field`: Function or Interpolator v(x) -> vector.
"""
function advect_particles!(tracker::ParticleTracker{2, Float64}, mesh, velocity_field, dt::Float64)
    for p in tracker.particles
        if !p.active
            continue
        end

        v_part = SVector(0.0, 0.0)

        if p.cell_index == 0
            p.cell_index = find_cell_index(mesh, p.position)
        end

        if p.cell_index > 0
            if velocity_field isa Function
                v_part = SVector{2}(velocity_field(p.position[1], p.position[2]))
            elseif velocity_field isa AbstractArray
                error("Array velocity field not yet supported, wrap in function.")
            end
        else
            p.active = false
            continue
        end

        # Advect
        p.position += v_part * dt
        p.velocity = v_part

        # Update Cell Index (tracking)
        if !is_point_in_cell(mesh, p.cell_index, p.position)
            new_idx = find_cell_index(mesh, p.position, p.cell_index)
            if new_idx == 0
                p.active = false
            else
                p.cell_index = new_idx
            end
        end
    end
    return
end
