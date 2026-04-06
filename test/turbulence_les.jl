using FiniteVolumeMethod
using Test
using LinearAlgebra
using StaticArrays

# ── Mesh builder (copied from incompressible.jl — safe_include isolates modules) ──

"""
    build_cartesian_unstructured_mesh(nx, ny, Lx, Ly)

Create an `UnstructuredFVMMesh{2, Float64}` from a regular Cartesian grid
with `nx x ny` cells on domain `[0, Lx] x [0, Ly]`.

Face normals follow the convention:
- Internal vertical faces (between i and i+1): normal `(1, 0)`, area `dy`
- Internal horizontal faces (between j and j+1): normal `(0, 1)`, area `dx`
- Boundary faces point outward from the domain

Boundary face tags: `:left`, `:right`, `:bottom`, `:top`.
Internal face tags: `:internal`.
"""
function build_cartesian_unstructured_mesh(
        nx::Int, ny::Int, Lx::Float64, Ly::Float64,
    )
    dx = Lx / nx
    dy = Ly / ny
    ncells = nx * ny

    # Cell indexing: (i, j) -> (j-1)*nx + i,  i in 1:nx, j in 1:ny
    cell_idx(i, j) = (j - 1) * nx + i

    # Cell centers and volumes
    cell_centers = zeros(Float64, 2, ncells)
    cell_volumes = zeros(Float64, ncells)
    for j in 1:ny, i in 1:nx
        c = cell_idx(i, j)
        cell_centers[1, c] = (i - 0.5) * dx
        cell_centers[2, c] = (j - 0.5) * dy
        cell_volumes[c] = dx * dy
    end

    # Build faces: internal vertical, internal horizontal, then boundary
    face_cells_list = Tuple{Int, Int}[]    # (owner, neighbour)
    face_normals_list = Tuple{Float64, Float64}[]
    face_areas_list = Float64[]
    face_centers_list = Tuple{Float64, Float64}[]
    face_tags_list = Symbol[]

    # Internal vertical faces: between (i, j) and (i+1, j)
    for j in 1:ny, i in 1:(nx - 1)
        P = cell_idx(i, j)
        N = cell_idx(i + 1, j)
        push!(face_cells_list, (P, N))
        push!(face_normals_list, (1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (i * dx, (j - 0.5) * dy))
        push!(face_tags_list, :internal)
    end

    # Internal horizontal faces: between (i, j) and (i, j+1)
    for j in 1:(ny - 1), i in 1:nx
        P = cell_idx(i, j)
        N = cell_idx(i, j + 1)
        push!(face_cells_list, (P, N))
        push!(face_normals_list, (0.0, 1.0))
        push!(face_areas_list, dx)
        push!(face_centers_list, ((i - 0.5) * dx, j * dy))
        push!(face_tags_list, :internal)
    end

    # Boundary: left (i=1, normal=-x)
    for j in 1:ny
        P = cell_idx(1, j)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (-1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (0.0, (j - 0.5) * dy))
        push!(face_tags_list, :left)
    end

    # Boundary: right (i=nx, normal=+x)
    for j in 1:ny
        P = cell_idx(nx, j)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (Lx, (j - 0.5) * dy))
        push!(face_tags_list, :right)
    end

    # Boundary: bottom (j=1, normal=-y)
    for i in 1:nx
        P = cell_idx(i, 1)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (0.0, -1.0))
        push!(face_areas_list, dx)
        push!(face_centers_list, ((i - 0.5) * dx, 0.0))
        push!(face_tags_list, :bottom)
    end

    # Boundary: top (j=ny, normal=+y)
    for i in 1:nx
        P = cell_idx(i, ny)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (0.0, 1.0))
        push!(face_areas_list, dx)
        push!(face_centers_list, ((i - 0.5) * dx, Ly))
        push!(face_tags_list, :top)
    end

    nfaces = length(face_cells_list)

    # Pack into matrices
    face_cells = zeros(Int, 2, nfaces)
    face_normals = zeros(Float64, 2, nfaces)
    face_areas = zeros(Float64, nfaces)
    face_centers = zeros(Float64, 2, nfaces)
    for f in 1:nfaces
        face_cells[1, f] = face_cells_list[f][1]
        face_cells[2, f] = face_cells_list[f][2]
        face_normals[1, f] = face_normals_list[f][1]
        face_normals[2, f] = face_normals_list[f][2]
        face_areas[f] = face_areas_list[f]
        face_centers[1, f] = face_centers_list[f][1]
        face_centers[2, f] = face_centers_list[f][2]
    end

    # Build cell_faces connectivity
    cell_faces = [Int[] for _ in 1:ncells]
    for f in 1:nfaces
        P = face_cells[1, f]
        push!(cell_faces[P], f)
        N = face_cells[2, f]
        if N != 0
            push!(cell_faces[N], f)
        end
    end

    return FiniteVolumeMethod.UnstructuredFVMMesh{2, Float64}(
        cell_centers,
        cell_volumes,
        face_cells,
        face_centers,
        face_areas,
        face_normals,
        face_tags_list,
        nothing,       # face_velocity
        cell_faces,
    )
end

# ── Helper: set up a shear velocity field U = (y, 0) on mesh ────────
function make_shear_field(mesh)
    nc = length(mesh.cell_volumes)
    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        y_c = mesh.cell_centers[2, c]
        U.internal[c] = SVector(y_c, 0.0)
    end
    for (i, f) in enumerate(U.boundary_face_indices)
        y_f = mesh.face_centers[2, f]
        U.boundary[i] = SVector(y_f, 0.0)
    end
    return U
end

# ── Helper: set up a uniform velocity field U = (1, 0) on mesh ──────
function make_uniform_field(mesh)
    U = CollocatedVectorField(:U, mesh; value = SVector(1.0, 0.0))
    for (i, _) in enumerate(U.boundary_face_indices)
        U.boundary[i] = SVector(1.0, 0.0)
    end
    return U
end

# ── Tests ─────────────────────────────────────────────────────────────

@testset "LES & Hybrid Turbulence" begin

    # ── 1. compute_filter_width ──────────────────────────────────────
    @testset "compute_filter_width" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        delta = compute_filter_width(mesh)
        nc = length(mesh.cell_volumes)

        @test length(delta) == nc
        # For 2D: Δ = V^(1/2) = (dx * dy)^(1/2)
        # dx = dy = 0.25, cell_volume = 0.0625, Δ = 0.0625^(1/2) = 0.25
        expected = 0.0625^(1 / 2)
        for c in 1:nc
            @test delta[c] ≈ expected atol = 1.0e-14
        end
    end

    # ── 2. LESTurbulenceState construction ───────────────────────────
    @testset "LESTurbulenceState construction" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        ts = LESTurbulenceState(mesh)
        @test ts isa LESTurbulenceState{Float64}
        @test length(ts.nu_t) == nc
        @test all(==(0.0), ts.nu_t)
    end

    # ── 3. Smagorinsky construction ──────────────────────────────────
    @testset "Smagorinsky construction" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        smag = Smagorinsky(mesh)
        @test smag isa Smagorinsky{Float64}
        @test smag.Cs == 0.1
        @test length(smag.delta) == nc

        smag2 = Smagorinsky(mesh; Cs = 0.15)
        @test smag2.Cs == 0.15
    end

    # ── 4. Smagorinsky on shear flow ─────────────────────────────────
    @testset "Smagorinsky shear flow" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        smag = Smagorinsky(mesh)
        U = make_shear_field(mesh)
        nu_t = zeros(Float64, nc)

        turbulent_viscosity!(nu_t, smag, U, mesh)

        @test all(isfinite, nu_t)
        @test all(>(0), nu_t)
    end

    # ── 5. Smagorinsky on uniform flow ───────────────────────────────
    @testset "Smagorinsky uniform flow" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        smag = Smagorinsky(mesh)
        U = make_uniform_field(mesh)
        nu_t = zeros(Float64, nc)

        turbulent_viscosity!(nu_t, smag, U, mesh)

        @test all(isfinite, nu_t)
        # Uniform flow has zero strain rate -> zero SGS viscosity
        for c in 1:nc
            @test nu_t[c] ≈ 0.0 atol = 1.0e-10
        end
    end

    # ── 6. WALE construction ─────────────────────────────────────────
    @testset "WALE construction" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        wale = WALE(mesh)
        @test wale isa WALE{Float64}
        @test wale.Cw == 0.325
        @test length(wale.delta) == nc

        wale2 = WALE(mesh; Cw = 0.5)
        @test wale2.Cw == 0.5
    end

    # ── 7. WALE on shear flow ────────────────────────────────────────
    @testset "WALE shear flow" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        wale = WALE(mesh)
        U = make_shear_field(mesh)
        nu_t = zeros(Float64, nc)

        turbulent_viscosity!(nu_t, wale, U, mesh)

        @test all(isfinite, nu_t)
        @test all(>=(0), nu_t)
    end

    # ── 8. DynamicSmagorinsky construction ───────────────────────────
    @testset "DynamicSmagorinsky construction" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        ds = DynamicSmagorinsky(mesh)
        @test ds isa DynamicSmagorinsky{Float64}
        @test ds.test_filter_ratio == 2.0
        @test length(ds.delta) == nc

        ds2 = DynamicSmagorinsky(mesh; test_filter_ratio = 3.0)
        @test ds2.test_filter_ratio == 3.0
    end

    # ── 9. DynamicSmagorinsky viscosity on shear flow ────────────────
    @testset "DynamicSmagorinsky shear flow" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        ds = DynamicSmagorinsky(mesh)
        U = make_shear_field(mesh)
        nu_t = zeros(Float64, nc)

        turbulent_viscosity!(nu_t, ds, U, mesh)

        @test all(isfinite, nu_t)
        @test all(>=(0), nu_t)
    end

    # ── 10. _test_filter on constant field ───────────────────────────
    @testset "_test_filter constant field" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        constant_val = 3.14
        values = fill(constant_val, nc)
        filtered = FiniteVolumeMethod._test_filter(values, mesh)

        @test length(filtered) == nc
        for c in 1:nc
            @test filtered[c] ≈ constant_val atol = 1.0e-12
        end
    end

    # ── 11. DDES construction ────────────────────────────────────────
    @testset "DDES construction" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        sa = SpalartAllmaras(mesh, [:bottom, :top])
        ddes = DDES(sa, mesh, [:bottom, :top])

        @test ddes isa DDES{SpalartAllmaras{Float64}, Float64}
        @test ddes.C_DES == 0.65
        @test ddes.base_model === sa
        @test length(ddes.delta) == nc
        @test length(ddes.d_wall) == nc
        @test all(>(0), ddes.d_wall)
        @test all(isfinite, ddes.d_wall)
    end

    # ── 12. n_turbulence_fields ──────────────────────────────────────
    @testset "n_turbulence_fields" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)

        smag = Smagorinsky(mesh)
        wale = WALE(mesh)
        ds = DynamicSmagorinsky(mesh)

        @test n_turbulence_fields(smag) == 0
        @test n_turbulence_fields(wale) == 0
        @test n_turbulence_fields(ds) == 0

        @test turbulence_field_names(smag) == ()
        @test turbulence_field_names(wale) == ()
        @test turbulence_field_names(ds) == ()

        # DDES delegates to base model (SA has 1 field)
        sa = SpalartAllmaras(mesh, [:bottom, :top])
        ddes = DDES(sa, mesh, [:bottom, :top])
        @test n_turbulence_fields(ddes) == 1
        @test turbulence_field_names(ddes) == (:nu_tilde,)
    end

    # ── 13. solve_turbulence! no-op for LES ──────────────────────────
    @testset "solve_turbulence! no-op" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        smag = Smagorinsky(mesh)
        ts = LESTurbulenceState(mesh)
        U = make_uniform_field(mesh)
        phi = FaceFluxField(:phi, mesh; value = 0.0)
        bcs_turb = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}()

        result = solve_turbulence!(ts, smag, U, phi, 1.0e-3, mesh, bcs_turb)
        @test result === nothing
        # nu_t should remain zeros (no-op, not viscosity computation)
        @test all(==(0.0), ts.nu_t)
    end

    # ── 14. _update_turbulence! LES path ─────────────────────────────
    @testset "_update_turbulence! LES path" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)

        smag = Smagorinsky(mesh)
        ts = LESTurbulenceState(mesh)
        state = IncompressibleState(mesh)

        # Set shear flow on the incompressible state
        for c in 1:nc
            y_c = mesh.cell_centers[2, c]
            state.U.internal[c] = SVector(y_c, 0.0)
        end
        for (i, f) in enumerate(state.U.boundary_face_indices)
            y_f = mesh.face_centers[2, f]
            state.U.boundary[i] = SVector(y_f, 0.0)
        end

        bcs = Dict{Symbol, AbstractBoundaryCondition}(
            :left => FixedVelocityBC((0.0, 0.0)),
            :right => FixedVelocityBC((0.0, 0.0)),
            :bottom => NoSlipWallBC(),
            :top => FixedVelocityBC((1.0, 0.0)),
        )
        algo = SIMPLE()
        prob = IncompressibleProblem(mesh, bcs, algo; nu = 0.01)
        turb_bcs = Dict{Symbol, Dict{Symbol, AbstractBoundaryCondition}}()

        # Before update, nu_t should be zero
        @test all(==(0.0), ts.nu_t)

        FiniteVolumeMethod._update_turbulence!(ts, smag, state, prob, mesh, turb_bcs)

        # After update, nu_t should be positive (shear flow)
        @test all(isfinite, ts.nu_t)
        @test all(>(0), ts.nu_t)
    end

end
