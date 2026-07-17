# mesh/quality.jl — Mesh quality metrics (OpenFOAM checkMesh equivalent)
#
# Computes non-orthogonality, skewness, and aspect ratio for
# UnstructuredFVMMesh. Reports per-face/per-cell values and summary stats.

using LinearAlgebra: norm, dot
using Printf: @sprintf

"""
    MeshQualityReport{T}

Results of a mesh quality check.

# Fields
- `non_orthogonality::Vector{T}` -- angle (degrees) per internal face
- `skewness::Vector{T}` -- dimensionless skewness per internal face
- `aspect_ratio::Vector{T}` -- dimensionless aspect ratio per cell
- `max_non_orthogonality::T`, `avg_non_orthogonality::T`
- `max_skewness::T`, `avg_skewness::T`
- `max_aspect_ratio::T`
"""
struct MeshQualityReport{T}
    non_orthogonality::Vector{T}
    skewness::Vector{T}
    aspect_ratio::Vector{T}
    max_non_orthogonality::T
    avg_non_orthogonality::T
    max_skewness::T
    avg_skewness::T
    max_aspect_ratio::T
end

"""
    check_mesh_quality(mesh::UnstructuredFVMMesh{Dim, T}) -> MeshQualityReport{T}

Compute mesh quality metrics: non-orthogonality, skewness, and aspect ratio.
"""
function check_mesh_quality(
        mesh::UnstructuredFVMMesh{Dim, T},
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)
    nc = length(mesh.cell_volumes)

    # Collect internal face indices
    internal_faces = Int[]
    for f in 1:nf
        if is_internal_face(mesh, f)
            push!(internal_faces, f)
        end
    end

    n_internal = length(internal_faces)

    # Non-orthogonality: angle between face normal and cell-center vector
    non_ortho = Vector{T}(undef, n_internal)
    for (i, f) in enumerate(internal_faces)
        S_f = face_normal_area(mesh, f)
        d_vec, d_mag = owner_neighbour_distance(mesh, f)

        if d_mag > zero(T)
            cos_theta = abs(dot(S_f, d_vec)) / (norm(S_f) * d_mag)
            cos_theta = clamp(cos_theta, zero(T), one(T))
            non_ortho[i] = acos(cos_theta) * T(180) / T(pi)
        else
            non_ortho[i] = zero(T)
        end
    end

    # Skewness: offset of face center from cell-center line intersection
    skewness = Vector{T}(undef, n_internal)
    for (i, f) in enumerate(internal_faces)
        x_f = face_center(mesh, f)
        P = owner(mesh, f)
        N = neighbour(mesh, f)
        x_P = cell_center(mesh, P)
        x_N = cell_center(mesh, N)

        # Intersection point: project face center onto P-N line
        d_vec = x_N - x_P
        d_mag_sq = dot(d_vec, d_vec)
        if d_mag_sq > zero(T)
            t = dot(x_f - x_P, d_vec) / d_mag_sq
            x_intersection = x_P + t * d_vec
            offset = norm(x_f - x_intersection)
            face_scale = sqrt(mesh.face_areas[f])
            skewness[i] = face_scale > zero(T) ? offset / face_scale : zero(T)
        else
            skewness[i] = zero(T)
        end
    end

    # Aspect ratio: approximate from face areas and volume
    aspect_ratio = Vector{T}(undef, nc)
    if mesh.cell_faces !== nothing
        for c in 1:nc
            faces_c = mesh.cell_faces[c]
            if !isempty(faces_c)
                max_area = maximum(mesh.face_areas[f] for f in faces_c)
                V = mesh.cell_volumes[c]
                aspect_ratio[c] = V > zero(T) ? max_area / V^(T(2) / T(3)) : one(T)
            else
                aspect_ratio[c] = one(T)
            end
        end
    else
        fill!(aspect_ratio, one(T))
    end

    # Summary statistics
    max_no = n_internal > 0 ? maximum(non_ortho) : zero(T)
    avg_no = n_internal > 0 ? sum(non_ortho) / n_internal : zero(T)
    max_sk = n_internal > 0 ? maximum(skewness) : zero(T)
    avg_sk = n_internal > 0 ? sum(skewness) / n_internal : zero(T)
    max_ar = nc > 0 ? maximum(aspect_ratio) : one(T)

    return MeshQualityReport{T}(
        non_ortho, skewness, aspect_ratio,
        max_no, avg_no, max_sk, avg_sk, max_ar,
    )
end

"""
    print_mesh_quality(report::MeshQualityReport; io::IO = stdout)

Print an OpenFOAM-style mesh quality summary.
"""
function print_mesh_quality(report::MeshQualityReport{T}; io::IO = stdout) where {T}
    println(io, "Mesh Quality Report")
    println(io, "===================")
    println(
        io, "Non-orthogonality: max = ", @sprintf("%.1f°", report.max_non_orthogonality),
        ", avg = ", @sprintf("%.1f°", report.avg_non_orthogonality)
    )
    println(
        io, "Skewness:          max = ", @sprintf("%.4f", report.max_skewness),
        ", avg = ", @sprintf("%.4f", report.avg_skewness)
    )
    println(io, "Aspect ratio:      max = ", @sprintf("%.2f", report.max_aspect_ratio))

    status = "OK"
    if report.max_non_orthogonality > T(85)
        status = "ERROR (non-orthogonality > 85°)"
    elseif report.max_non_orthogonality > T(70)
        status = "WARNING (non-orthogonality > 70°)"
    elseif report.max_skewness > T(0.85)
        status = "WARNING (skewness > 0.85)"
    end
    println(io, "Status: ", status)

    return nothing
end
