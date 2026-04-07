# collocated/cyclic.jl — Cyclic (periodic) boundary condition assembly
#
# Provides face matching and equation modification for true cyclic
# periodicity on collocated FVM meshes.  Replaces the placeholder
# Neumann(0) expansion with proper cross-boundary cell coupling.

"""
    match_cyclic_faces(
        mesh::UnstructuredFVMMesh{Dim, T},
        patch1::Symbol,
        patch2::Symbol,
    ) -> Vector{Tuple{Int, Int}}

Match boundary faces between two cyclic patches by proximity of face
centers.  The periodic offset is computed as the difference between
the mean face center of each patch.  After removing this offset, each
face on `patch1` is matched to its nearest face on `patch2`.

Returns a vector of `(face_on_patch1, face_on_patch2)` index pairs.

# Arguments
- `mesh` --- `UnstructuredFVMMesh`
- `patch1` --- name of the first cyclic patch (e.g. `:left`)
- `patch2` --- name of the partner cyclic patch (e.g. `:right`)
"""
function match_cyclic_faces(
        mesh::UnstructuredFVMMesh{Dim, T},
        patch1::Symbol,
        patch2::Symbol,
    ) where {Dim, T}
    nf = size(mesh.face_cells, 2)

    # Collect boundary faces for each patch
    faces1 = Int[]
    faces2 = Int[]
    for f in 1:nf
        mesh.face_cells[2, f] == 0 || continue
        tag = _face_tag(mesh, f)
        tag === patch1 && push!(faces1, f)
        tag === patch2 && push!(faces2, f)
    end

    isempty(faces1) && error("No boundary faces found for patch :$patch1")
    isempty(faces2) && error("No boundary faces found for patch :$patch2")
    length(faces1) != length(faces2) && error(
        "Patch :$patch1 has $(length(faces1)) faces but :$patch2 has $(length(faces2))"
    )

    # Compute mean face centers for each patch
    mean1 = zero(SVector{Dim, T})
    for f in faces1
        mean1 = mean1 + face_center(mesh, f)
    end
    mean1 = mean1 / T(length(faces1))

    mean2 = zero(SVector{Dim, T})
    for f in faces2
        mean2 = mean2 + face_center(mesh, f)
    end
    mean2 = mean2 / T(length(faces2))

    # Periodic offset: patch2_center - patch1_center
    offset = mean2 - mean1

    # Match faces: for each face on patch1, find nearest on patch2
    # after shifting patch1 face centers by the offset
    matched = Vector{Tuple{Int, Int}}(undef, length(faces1))
    used = falses(length(faces2))

    for (i, f1) in enumerate(faces1)
        x1_shifted = face_center(mesh, f1) + offset
        best_j = 0
        best_dist = T(Inf)
        for (j, f2) in enumerate(faces2)
            used[j] && continue
            d = norm(x1_shifted - face_center(mesh, f2))
            if d < best_dist
                best_dist = d
                best_j = j
            end
        end
        best_j == 0 && error("Failed to match face $f1 on patch :$patch1")
        used[best_j] = true
        matched[i] = (f1, faces2[best_j])
    end

    return matched
end

"""
    apply_cyclic_bc!(
        eq::CollocatedEquation{T},
        field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        face_pairs::Vector{Tuple{Int, Int}},
    )

Modify the assembled equation to enforce cyclic (periodic) boundary
conditions.  For each matched pair `(f1, f2)`:

- The boundary face value of `f1` is set to the internal value of `f2`'s
  owner cell, and vice versa.
- This couples the owner cells across the periodic boundary by adding
  off-diagonal entries and modifying the RHS.

The coupling uses a simple Dirichlet-like treatment where the boundary
face value equals the partner's owner cell value.

# Arguments
- `eq` --- assembled equation (A and b modified in-place)
- `field` --- current field values (used for explicit contributions)
- `mesh` --- `UnstructuredFVMMesh`
- `face_pairs` --- matched face pairs from [`match_cyclic_faces`](@ref)
"""
function apply_cyclic_bc!(
        eq::CollocatedEquation{T},
        field::CollocatedScalarField{T},
        mesh::UnstructuredFVMMesh{Dim, T},
        face_pairs::Vector{Tuple{Int, Int}},
    ) where {Dim, T}
    for (f1, f2) in face_pairs
        c1 = owner(mesh, f1)
        c2 = owner(mesh, f2)

        # Compute a diffusion-like coupling coefficient between the
        # partner cells based on the face geometry
        A_f1 = mesh.face_areas[f1]
        A_f2 = mesh.face_areas[f2]
        A_avg = (A_f1 + A_f2) / 2

        # Distance: cell center to face center (each side)
        d1 = norm(face_center(mesh, f1) - cell_center(mesh, c1))
        d2 = norm(face_center(mesh, f2) - cell_center(mesh, c2))
        d_total = max(d1 + d2, T(1.0e-20))

        flux_coeff = A_avg / d_total

        # Couple c1 to c2: add diffusion-like cross terms
        # phi_f1 = phi_c2 and phi_f2 = phi_c1
        eq.A[c1, c1] += flux_coeff
        eq.A[c1, c2] -= flux_coeff
        eq.A[c2, c2] += flux_coeff
        eq.A[c2, c1] -= flux_coeff
    end

    return nothing
end
