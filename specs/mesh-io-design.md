---
date: 2026-04-06
---

# Phase 4: Polyhedral Mesh I/O

**Status**: Design
**Depends on**: Nothing (independent, but enables Phases 0-3 on real meshes)

## Goal

Enable the collocated solver stack (Phases 0-3) to run on real-world meshes by adding an OpenFOAM polyMesh reader, a converter from the existing Gmsh-produced `UnstructuredMesh3D` to `UnstructuredFVMMesh`, a correct hexahedral volume computation (fixing the placeholder), and mesh quality metrics.

## Scope

**In scope:**
1. OpenFOAM `constant/polyMesh/` reader → `UnstructuredFVMMesh{3, Float64}`
2. `UnstructuredMesh3D` → `UnstructuredFVMMesh{3, Float64}` converter (bridges existing `read_gmsh`)
3. Fix `volume_hex` (currently returns 1.0 placeholder) + add `volume_prism`, `volume_pyramid`
4. Mesh quality metrics: non-orthogonality, skewness, aspect ratio (OpenFOAM `checkMesh` equivalent)

**Deferred:**
- OpenFOAM polyMesh writer (Phase 4b)
- Gmsh v4 format reader (Phase 4b — existing v2.2 reader works)
- blockMesh equivalent (users can use Gmsh or OpenFOAM blockMesh externally)

## Architecture

### File Layout

New files in `src/mesh/`:

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `openfoam_io.jl` | Read OpenFOAM `constant/polyMesh/` directory | ~250 |
| `convert.jl` | `UnstructuredMesh3D` → `UnstructuredFVMMesh` converter | ~100 |
| `polyhedral_volumes.jl` | Correct volume_hex, volume_prism, volume_pyramid via tet decomposition | ~80 |
| `quality.jl` | Non-orthogonality, skewness, aspect ratio metrics | ~120 |

Wired into Layer 1 (`domain_problem_definitions.jl`) after existing mesh includes, since these are mesh-level utilities with no solver dependencies.

## OpenFOAM polyMesh Reader

### Format

OpenFOAM stores meshes in `constant/polyMesh/` with these files:
- `points` — vertex coordinates `(x y z)`, 0-indexed
- `faces` — face-to-vertex connectivity `N(v0 v1 ... vN-1)`, 0-indexed
- `owner` — owner cell index per face (length = nfaces)
- `neighbour` — neighbour cell index for internal faces only (length = n_internal_faces)
- `boundary` — patch definitions with `nFaces` and `startFace`

Files may be plain ASCII or gzip-compressed (`.gz`). The FoamFile header contains metadata (version, format, class, object). Data follows after the header in parenthesized lists.

### Reader Design

```julia
function read_openfoam_polymesh(
    case_dir::AbstractString;
    mesh_dir::AbstractString = "constant/polyMesh",
) -> UnstructuredFVMMesh{3, Float64}
```

The reader:
1. Parses each file, skipping the FoamFile header and comments
2. Reads the count `N` then the parenthesized data block `( ... )`
3. Handles both plain and gzip-compressed files transparently
4. Converts 0-indexed OpenFOAM indices to 1-indexed Julia

**Internal parsing functions:**
```julia
_read_openfoam_points(path) -> Matrix{Float64}  # 3 x npoints
_read_openfoam_faces(path) -> Vector{Vector{Int}}  # face-to-vertex, 1-indexed
_read_openfoam_labels(path) -> Vector{Int}  # owner or neighbour, 1-indexed
_read_openfoam_boundary(path) -> Vector{NamedTuple{(:name, :type, :nFaces, :startFace)}}
```

**Mesh construction from OpenFOAM data:**
1. Build `face_cells` from owner + neighbour: `[owner[f]; neighbour[f]]` for internal, `[owner[f]; 0]` for boundary
2. Compute face centers, areas, normals from face vertex lists
3. Compute cell centers as average of incident face centers
4. Compute cell volumes via face-based divergence theorem: `V = (1/3) * sum_f (x_f · S_f)`
5. Build face_tags from boundary patch definitions
6. Build cell_faces connectivity (invert face_cells)

**Face geometry from vertices:**
```julia
function _compute_face_geometry(points, face_vertices)
    # For triangular faces: direct cross product
    # For polygonal faces: decompose into triangles from centroid
    # Returns: (center::SVector{3}, area::Float64, normal::SVector{3})
end
```

**Cell volume via Gauss divergence theorem:**
```
V_c = (1/3) * sum_f sign_f * (x_f · S_f)
```
where `sign_f = +1` if cell is owner, `-1` if neighbour. This works for arbitrary polyhedra without tet decomposition.

### Gzip Handling

OpenFOAM files may be gzip-compressed. Detection:
```julia
function _open_foam_file(path)
    # Try path as-is, then path.gz
    # Use CodecZlib.GzipDecompressorStream for .gz files
    # Return IO stream
end
```

**No new dependencies**: Use `GzipDecompressorStream` from CodecZlib.jl only if the file is actually gzipped. Since CodecZlib is a lightweight stdlib-like package, add it as a direct dependency (not a weak dep). Alternatively, only support uncompressed files initially and document that users should `gunzip` compressed meshes. **Decision: support uncompressed only initially.** Users can decompress with `gunzip constant/polyMesh/*` before reading. This avoids a new dependency.

## UnstructuredMesh3D → UnstructuredFVMMesh Converter

### Purpose

The existing `read_gmsh` returns `UnstructuredMesh3D` (object-oriented, node-based). The collocated solver needs `UnstructuredFVMMesh` (matrix-oriented, face-based). This converter bridges them.

### Design

```julia
function convert_to_fvm_mesh(
    mesh::UnstructuredMesh3D;
    tag_boundary::Bool = true,
) -> UnstructuredFVMMesh{3, Float64}
```

Extracts:
- `cell_centers` from `mesh.cells[c].center`
- `cell_volumes` from `mesh.cells[c].volume`
- `face_cells` from `mesh.faces[f].owner`, `mesh.faces[f].neighbor`
- `face_centers` from `mesh.faces[f].center`
- `face_areas` from `mesh.faces[f].area`
- `face_normals` from `mesh.faces[f].normal` (normalized)
- `cell_faces` from `mesh.cells[c].faces`
- `face_tags` via `tag_unstructured_faces_by_bounds` if `tag_boundary=true`

Also add a 2D version:
```julia
function convert_to_fvm_mesh(
    mesh::UnstructuredMesh2D;
    tag_boundary::Bool = true,
) -> UnstructuredFVMMesh{2, Float64}
```

### Note on `convert_to_unstructured`

The function name `convert_to_unstructured` is already exported (from mesh partitioning). To avoid confusion, the new function is named `convert_to_fvm_mesh`.

## Volume Computations

### Fix volume_hex

The current `volume_hex` in `src/parabolic/mesh/io.jl` returns `1.0` (placeholder). Replace with tet decomposition:

```julia
function volume_hex(nodes::Vector{Node3D})
    # Decompose hex into 5 or 6 tetrahedra and sum volumes
    # Standard decomposition: split hex into 5 tets
    # Uses existing volume_tet for each sub-tet
end
```

### Add volume_prism, volume_pyramid

Currently prism and pyramid volumes are hardcoded to 1.0 in `read_gmsh`. Add:

```julia
function volume_prism(nodes::Vector{Node3D})
    # Decompose prism into 3 tetrahedra
end

function volume_pyramid(nodes::Vector{Node3D})
    # Decompose pyramid into 2 tetrahedra
end
```

### Integration

Update `read_gmsh` in `io.jl` to use the corrected volume functions instead of `1.0` placeholders.

## Mesh Quality Metrics

### Design

```julia
struct MeshQualityReport{T}
    non_orthogonality::Vector{T}  # per face, degrees
    skewness::Vector{T}           # per face, dimensionless [0,1]
    aspect_ratio::Vector{T}       # per cell, dimensionless ≥ 1
    max_non_orthogonality::T
    avg_non_orthogonality::T
    max_skewness::T
    avg_skewness::T
    max_aspect_ratio::T
end

function check_mesh_quality(
    mesh::UnstructuredFVMMesh{Dim, T},
) -> MeshQualityReport{T}
```

### Non-orthogonality (per internal face)

Angle between face normal `S_f` and cell-center vector `d = x_N - x_P`:
```
θ = acos(|S_f · d| / (|S_f| * |d|))
```
Ideal: 0°. Warning > 70°. Error > 85°.

### Skewness (per internal face)

Distance from face center to the intersection of the cell-center vector with the face plane, normalized by face size:
```
skewness = |x_f - x_intersection| / |face_diagonal|
```
Ideal: 0. Warning > 0.85.

### Aspect Ratio (per cell)

Ratio of longest to shortest cell dimension. For cells with known face areas and volume:
```
AR = max_face_area / (V^(2/3))  (approximate)
```
Exact computation requires bounding box or PCA of face centers.

### Summary Output

```julia
function print_mesh_quality(report::MeshQualityReport; io::IO = stdout)
    # Prints OpenFOAM-style checkMesh summary:
    # Mesh Quality Report
    # ===================
    # Non-orthogonality: max = 45.2°, avg = 12.3°
    # Skewness:          max = 0.32,  avg = 0.08
    # Aspect ratio:      max = 3.2
    # Status: OK (all metrics within acceptable limits)
end
```

## Export List

```julia
# OpenFOAM I/O
export read_openfoam_polymesh

# Mesh conversion
export convert_to_fvm_mesh

# Volume computations (volume_tet already exported)
export volume_prism, volume_pyramid
# volume_hex already exported but fix the implementation

# Quality
export MeshQualityReport, check_mesh_quality, print_mesh_quality
```

## Validation

- **OpenFOAM reader round-trip**: Read a simple OpenFOAM hex mesh (manually constructed in test), verify cell count, face count, boundary patches, and cell volumes match expected values.
- **Gmsh converter**: Read a Gmsh .msh file with `read_gmsh`, convert to `UnstructuredFVMMesh`, verify `validate_mesh` passes and cell volumes are positive.
- **Volume accuracy**: Compare computed tet/hex/prism/pyramid volumes against analytical values for unit cells.
- **Quality metrics**: Verify orthogonal Cartesian mesh has 0° non-orthogonality and 0 skewness.
