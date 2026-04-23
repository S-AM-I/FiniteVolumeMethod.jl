# parallel/local_mesh.jl — Extract a per-rank submesh from a global mesh
#
# Stage 2c: given a `cell_to_rank` partition, build an `UnstructuredFVMMesh`
# containing only the cells owned by `my_rank` plus one halo layer of
# off-rank neighbours. Cells are re-indexed local 1..n_local with owned
# cells first, halo cells last.
#
# Returns the local mesh plus bookkeeping arrays (local↔global maps,
# rank-of-halo-cell) that the distributed solver + halo exchange need.

"""
    LocalMeshData{Dim, T}

Packaged output of `extract_local_mesh`: a per-rank `UnstructuredFVMMesh`
holding only owned + halo cells, plus the index maps needed to drive MPI
communication.

# Fields
- `mesh::UnstructuredFVMMesh{Dim, T}` — the local submesh. Cells 1..n_owned
  are owned; cells n_owned+1..n_local are halo (neighbour-of-owned cells
  residing on other ranks). Face ordering preserves the order seen in the
  global mesh for consistent assembly.
- `n_owned::Int` — number of owned cells on this rank.
- `n_local::Int` — total local cells (owned + halo).
- `local_to_global::Vector{Int}` — length-`n_local`; global cell index for each local.
- `global_to_local::Dict{Int, Int}` — inverse map; `0` if a global cell isn't in this rank's local mesh.
- `halo_owner_rank::Vector{Int}` — length-`n_local`; the rank that owns each cell (== my_rank for owned, other ranks for halo).
- `local_to_global_face::Vector{Int}` — length = n_local_faces; global face index for each local face (for debugging / diagnostics).
"""
struct LocalMeshData{Dim, T}
    mesh::UnstructuredFVMMesh{Dim, T}
    n_owned::Int
    n_local::Int
    local_to_global::Vector{Int}
    global_to_local::Dict{Int, Int}
    halo_owner_rank::Vector{Int}
    local_to_global_face::Vector{Int}
end

"""
    extract_local_mesh(
        global_mesh::UnstructuredFVMMesh{Dim, T},
        cell_to_rank::AbstractVector{Int},
        my_rank::Int,
    ) -> LocalMeshData{Dim, T}

Build the per-rank submesh for `my_rank`, given the global mesh and a
per-cell rank assignment. The submesh includes owned cells plus one halo
layer of off-rank neighbours (across internal faces). Boundary faces
attached to owned cells are preserved.
"""
function extract_local_mesh(
        global_mesh::UnstructuredFVMMesh{Dim, T},
        cell_to_rank::AbstractVector{Int},
        my_rank::Int,
    ) where {Dim, T}
    nc_g = length(global_mesh.cell_volumes)
    nf_g = size(global_mesh.face_cells, 2)
    length(cell_to_rank) == nc_g ||
        error("cell_to_rank length $(length(cell_to_rank)) ≠ ncells $nc_g")

    # Phase 1: identify owned + halo cells (global indices).
    owned_global = Int[]
    halo_global_set = Set{Int}()
    @inbounds for c in 1:nc_g
        if cell_to_rank[c] == my_rank
            push!(owned_global, c)
        end
    end
    @inbounds for f in 1:nf_g
        P = global_mesh.face_cells[1, f]
        N = global_mesh.face_cells[2, f]
        N == 0 && continue  # boundary face
        P_owned = cell_to_rank[P] == my_rank
        N_owned = cell_to_rank[N] == my_rank
        if P_owned && !N_owned
            push!(halo_global_set, N)
        elseif !P_owned && N_owned
            push!(halo_global_set, P)
        end
    end
    halo_global = sort!(collect(halo_global_set))

    n_owned = length(owned_global)
    n_local = n_owned + length(halo_global)

    # Phase 2: build local↔global maps.
    local_to_global = Vector{Int}(undef, n_local)
    halo_owner_rank = Vector{Int}(undef, n_local)
    @inbounds for (i, g) in pairs(owned_global)
        local_to_global[i] = g
        halo_owner_rank[i] = my_rank
    end
    @inbounds for (i, g) in pairs(halo_global)
        local_to_global[n_owned + i] = g
        halo_owner_rank[n_owned + i] = cell_to_rank[g]
    end
    global_to_local = Dict{Int, Int}()
    sizehint!(global_to_local, n_local)
    @inbounds for i in 1:n_local
        global_to_local[local_to_global[i]] = i
    end

    # Phase 3: collect local faces. A face is local if at least one of its
    # incident cells is owned — which guarantees the other cell is either
    # owned or halo (by construction of halo_global_set above) or is
    # boundary.
    keep_face = Bool[false for _ in 1:nf_g]
    @inbounds for f in 1:nf_g
        P = global_mesh.face_cells[1, f]
        N = global_mesh.face_cells[2, f]
        P_owned = cell_to_rank[P] == my_rank
        N_owned = N == 0 ? false : cell_to_rank[N] == my_rank
        keep_face[f] = P_owned || N_owned
    end
    local_to_global_face = [f for f in 1:nf_g if keep_face[f]]
    n_local_faces = length(local_to_global_face)

    # Phase 4: build the local UnstructuredFVMMesh. Copy per-cell and
    # per-face data, re-indexing cells to local indices.
    cell_centers = Matrix{T}(undef, Dim, n_local)
    cell_volumes = Vector{T}(undef, n_local)
    @inbounds for i in 1:n_local
        g = local_to_global[i]
        for d in 1:Dim
            cell_centers[d, i] = global_mesh.cell_centers[d, g]
        end
        cell_volumes[i] = global_mesh.cell_volumes[g]
    end

    face_cells = zeros(Int, 2, n_local_faces)
    face_centers = Matrix{T}(undef, Dim, n_local_faces)
    face_areas = Vector{T}(undef, n_local_faces)
    face_normals = Matrix{T}(undef, Dim, n_local_faces)
    face_tags = global_mesh.face_tags === nothing ?
        nothing : Vector{Symbol}(undef, n_local_faces)

    @inbounds for (local_f, global_f) in pairs(local_to_global_face)
        Pg = global_mesh.face_cells[1, global_f]
        Ng = global_mesh.face_cells[2, global_f]
        face_cells[1, local_f] = global_to_local[Pg]
        # If N is 0 (boundary face), keep 0. If N is non-zero but
        # not-owned-and-not-halo, that means the face belongs to
        # another rank entirely; such faces are excluded by keep_face.
        face_cells[2, local_f] =
            Ng == 0 ? 0 :
            get(global_to_local, Ng, 0)
        for d in 1:Dim
            face_centers[d, local_f] = global_mesh.face_centers[d, global_f]
            face_normals[d, local_f] = global_mesh.face_normals[d, global_f]
        end
        face_areas[local_f] = global_mesh.face_areas[global_f]
        if face_tags !== nothing
            face_tags[local_f] = global_mesh.face_tags[global_f]
        end
    end

    # Phase 5: rebuild `cell_faces` adjacency for the local mesh.
    cell_faces = [Int[] for _ in 1:n_local]
    @inbounds for local_f in 1:n_local_faces
        P = face_cells[1, local_f]
        N = face_cells[2, local_f]
        push!(cell_faces[P], local_f)
        if N != 0
            push!(cell_faces[N], local_f)
        end
    end

    local_mesh = UnstructuredFVMMesh{Dim, T}(
        cell_centers,
        cell_volumes,
        face_cells,
        face_centers,
        face_areas,
        face_normals,
        face_tags,
        nothing,  # face_velocity — reconstructed per-iteration by the solver
        cell_faces,
    )

    return LocalMeshData{Dim, T}(
        local_mesh, n_owned, n_local,
        local_to_global, global_to_local, halo_owner_rank,
        local_to_global_face,
    )
end
