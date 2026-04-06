using FiniteVolumeMethod
using Test
using LinearAlgebra
using StaticArrays

# ── Mesh builder (copied from incompressible.jl; safe_include isolation) ──

function build_cartesian_unstructured_mesh(
        nx::Int, ny::Int, Lx::Float64, Ly::Float64,
    )
    dx = Lx / nx
    dy = Ly / ny
    ncells = nx * ny

    cell_idx(i, j) = (j - 1) * nx + i

    cell_centers = zeros(Float64, 2, ncells)
    cell_volumes = zeros(Float64, ncells)
    for j in 1:ny, i in 1:nx
        c = cell_idx(i, j)
        cell_centers[1, c] = (i - 0.5) * dx
        cell_centers[2, c] = (j - 0.5) * dy
        cell_volumes[c] = dx * dy
    end

    face_cells_list = Tuple{Int, Int}[]
    face_normals_list = Tuple{Float64, Float64}[]
    face_areas_list = Float64[]
    face_centers_list = Tuple{Float64, Float64}[]
    face_tags_list = Symbol[]

    # Internal vertical faces
    for j in 1:ny, i in 1:(nx - 1)
        P = cell_idx(i, j)
        N = cell_idx(i + 1, j)
        push!(face_cells_list, (P, N))
        push!(face_normals_list, (1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (i * dx, (j - 0.5) * dy))
        push!(face_tags_list, :internal)
    end

    # Internal horizontal faces
    for j in 1:(ny - 1), i in 1:nx
        P = cell_idx(i, j)
        N = cell_idx(i, j + 1)
        push!(face_cells_list, (P, N))
        push!(face_normals_list, (0.0, 1.0))
        push!(face_areas_list, dx)
        push!(face_centers_list, ((i - 0.5) * dx, j * dy))
        push!(face_tags_list, :internal)
    end

    # Boundary: left
    for j in 1:ny
        P = cell_idx(1, j)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (-1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (0.0, (j - 0.5) * dy))
        push!(face_tags_list, :left)
    end

    # Boundary: right
    for j in 1:ny
        P = cell_idx(nx, j)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (1.0, 0.0))
        push!(face_areas_list, dy)
        push!(face_centers_list, (Lx, (j - 0.5) * dy))
        push!(face_tags_list, :right)
    end

    # Boundary: bottom
    for i in 1:nx
        P = cell_idx(i, 1)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (0.0, -1.0))
        push!(face_areas_list, dx)
        push!(face_centers_list, ((i - 0.5) * dx, 0.0))
        push!(face_tags_list, :bottom)
    end

    # Boundary: top
    for i in 1:nx
        P = cell_idx(i, ny)
        push!(face_cells_list, (P, 0))
        push!(face_normals_list, (0.0, 1.0))
        push!(face_areas_list, dx)
        push!(face_centers_list, ((i - 0.5) * dx, Ly))
        push!(face_tags_list, :top)
    end

    nfaces = length(face_cells_list)

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

# ── Tests ─────────────────────────────────────────────────────────────

@testset "Post-Processing" begin

    # ── 1. Vorticity of rigid rotation ────────────────────────────────
    @testset "Vorticity of rigid rotation" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        U = CollocatedVectorField(:U, mesh)

        # Rigid-body rotation: u = -y, v = x  (omega_z = dv/dx - du/dy = 1 - (-1) = 2)
        for c in 1:nc
            x_c = mesh.cell_centers[1, c]
            y_c = mesh.cell_centers[2, c]
            U.internal[c] = SVector(-y_c, x_c)
        end
        # Set boundary values consistent with rotation field
        for (i, f) in enumerate(U.boundary_face_indices)
            x_f = mesh.face_centers[1, f]
            y_f = mesh.face_centers[2, f]
            U.boundary[i] = SVector(-y_f, x_f)
        end

        omega = compute_vorticity(U, mesh)
        @test length(omega) == nc

        # Interior cells (not touching boundary) should have vorticity ~ 2.0
        # On a 4x4 mesh, interior cells are (i, j) with i in 2:3, j in 2:3
        cell_idx(i, j) = (j - 1) * 4 + i
        for j in 2:3, i in 2:3
            c = cell_idx(i, j)
            @test omega[c] ≈ 2.0 atol = 0.3
        end
    end

    # ── 2. Q-criterion of uniform flow ────────────────────────────────
    @testset "Q-criterion of uniform flow" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        U = CollocatedVectorField(:U, mesh)

        # Uniform flow: U = (1, 0) everywhere
        for c in 1:nc
            U.internal[c] = SVector(1.0, 0.0)
        end
        for i in eachindex(U.boundary)
            U.boundary[i] = SVector(1.0, 0.0)
        end

        Q = compute_q_criterion(U, mesh)
        @test length(Q) == nc
        for c in 1:nc
            @test Q[c] ≈ 0.0 atol = 1.0e-10
        end
    end

    # ── 3. Enstrophy positive for shear flow ──────────────────────────
    @testset "Enstrophy positive for shear flow" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        U = CollocatedVectorField(:U, mesh)

        # Linear shear: U = (y, 0), vorticity omega_z = dv/dx - du/dy = 0 - 1 = -1
        for c in 1:nc
            y_c = mesh.cell_centers[2, c]
            U.internal[c] = SVector(y_c, 0.0)
        end
        for (i, f) in enumerate(U.boundary_face_indices)
            y_f = mesh.face_centers[2, f]
            U.boundary[i] = SVector(y_f, 0.0)
        end

        ens = compute_enstrophy(U, mesh)
        @test length(ens) == nc
        # Enstrophy = omega^2, should be > 0 for non-zero shear
        @test all(e -> e >= 0.0, ens)
        # At least some cells should have positive enstrophy
        @test any(e -> e > 0.0, ens)
    end

    # ── 4. Courant number ─────────────────────────────────────────────
    @testset "Courant number" begin
        nx, ny = 4, 4
        mesh = build_cartesian_unstructured_mesh(nx, ny, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        nf = size(mesh.face_cells, 2)

        phi = FaceFluxField(:phi, mesh)
        phi_val = 0.1
        for f in 1:nf
            phi.values[f] = phi_val
        end

        dt = 0.01
        Co = compute_courant_number(phi, mesh, dt)
        @test length(Co) == nc

        # For a uniform 4x4 mesh: dx = dy = 0.25, cell_volume = 0.0625
        # Each interior cell has 4 faces, each with |phi| = 0.1
        # Co = dt * sum|phi_f| / (2 * V) = 0.01 * 4*0.1 / (2*0.0625) = 0.032
        # Corner cells have 3 faces, edge cells have 3-4 faces — check interior
        cell_idx(i, j) = (j - 1) * nx + i
        for j in 2:3, i in 2:3
            c = cell_idx(i, j)
            expected = dt * 4 * phi_val / (2 * 0.0625)
            @test Co[c] ≈ expected atol = 1.0e-12
        end
    end

    # ── 5. Wall shear stress ──────────────────────────────────────────
    @testset "Wall shear stress" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        U = CollocatedVectorField(:U, mesh)

        # Channel-like flow: U = (1.0, 0) everywhere
        for c in 1:nc
            U.internal[c] = SVector(1.0, 0.0)
        end
        for i in eachindex(U.boundary)
            U.boundary[i] = SVector(0.0, 0.0)  # no-slip walls
        end

        nu = 0.01
        tau = compute_wall_shear_stress(U, nu, mesh, :bottom)
        @test length(tau) > 0

        # Wall shear at bottom should be in the x-direction (tangential)
        for t in tau
            @test norm(t) > 0.0
            # The tangential component should dominate (wall normal is y-direction)
            @test abs(t[1]) > 0.0  # x-component is non-zero
        end
    end

    # ── 6. y+ finite and non-negative ─────────────────────────────────
    @testset "y+ finite and non-negative" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        U = CollocatedVectorField(:U, mesh)

        for c in 1:nc
            U.internal[c] = SVector(1.0, 0.0)
        end
        for i in eachindex(U.boundary)
            U.boundary[i] = SVector(0.0, 0.0)
        end

        nu = 0.01
        yp = compute_y_plus(U, nu, mesh, :bottom)
        @test length(yp) > 0
        for y in yp
            @test y >= 0.0
            @test isfinite(y)
        end
    end

    # ── 7. Wall heat flux sign ────────────────────────────────────────
    @testset "Wall heat flux sign" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        T_field = CollocatedScalarField(:T, mesh; value = 300.0)

        # Set boundary temperature at bottom patch to 400 (hot wall)
        for (i, f) in enumerate(T_field.boundary_face_indices)
            tag = mesh.face_tags[f]
            if tag == :bottom
                T_field.boundary[i] = 400.0
            else
                T_field.boundary[i] = 300.0
            end
        end

        k = 1.0
        q = compute_wall_heat_flux(T_field, k, mesh, :bottom)
        @test length(q) > 0

        # q = -k * (T_wall - T_cell) / d = -1.0 * (400 - 300) / d < 0
        for qi in q
            @test qi < 0.0
        end
    end

    # ── 8. Nusselt number finite ──────────────────────────────────────
    @testset "Nusselt number finite" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        T_field = CollocatedScalarField(:T, mesh; value = 300.0)

        for (i, f) in enumerate(T_field.boundary_face_indices)
            tag = mesh.face_tags[f]
            if tag == :bottom
                T_field.boundary[i] = 400.0
            else
                T_field.boundary[i] = 300.0
            end
        end

        k = 1.0
        Nu = compute_nusselt_number(
            T_field, k, mesh, :bottom;
            T_ref = 300.0, L_ref = 1.0,
        )
        @test length(Nu) > 0
        for n in Nu
            @test isfinite(n)
            @test n >= 0.0
        end
    end

    # ── 9. Pressure force ─────────────────────────────────────────────
    @testset "Pressure force" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        nf = size(mesh.face_cells, 2)

        p_field = CollocatedScalarField(:p, mesh; value = 10.0)
        # Set all boundary pressures to the same constant
        for i in eachindex(p_field.boundary)
            p_field.boundary[i] = 10.0
        end

        U = CollocatedVectorField(:U, mesh)
        # Zero velocity so viscous force is zero
        for c in 1:nc
            U.internal[c] = SVector(0.0, 0.0)
        end
        for i in eachindex(U.boundary)
            U.boundary[i] = SVector(0.0, 0.0)
        end

        nu = 0.01
        forces = compute_forces(p_field, U, nu, mesh, :right)

        # On :right patch, faces have normal (1,0) and area dy = 0.25
        # Total face area vector = ny * dy * (1,0) = 4 * 0.25 * (1,0) = (1,0)
        # F_pressure = -sum p_f * S_f = -10.0 * (1, 0)
        @test forces.pressure[1] ≈ -10.0 atol = 1.0e-10
        @test forces.pressure[2] ≈ 0.0 atol = 1.0e-10

        # Viscous force should be zero (no velocity)
        @test norm(forces.viscous) ≈ 0.0 atol = 1.0e-10
    end

    # ── 10. Force coefficients arithmetic ─────────────────────────────
    @testset "Force coefficients arithmetic" begin
        F_p = SVector(2.0, 0.5)
        F_v = SVector(0.5, 0.1)
        rho_ref = 1.0
        U_ref = 10.0
        A_ref = 1.0

        coeffs = force_coefficients(
            F_p, F_v;
            rho_ref = rho_ref,
            U_ref = U_ref,
            A_ref = A_ref,
        )

        # q = 0.5 * rho * U^2 = 0.5 * 1.0 * 100 = 50
        # qA = 50 * 1.0 = 50
        # F_total = (2.5, 0.6)
        # Cd = F_total . (1,0) / qA = 2.5 / 50 = 0.05
        # Cl = F_total . (0,1) / qA = 0.6 / 50 = 0.012
        @test coeffs.Cd ≈ 0.05 atol = 1.0e-12
        @test coeffs.Cl ≈ 0.012 atol = 1.0e-12
        @test coeffs.Cd_pressure ≈ 2.0 / 50.0 atol = 1.0e-12
        @test coeffs.Cd_viscous ≈ 0.5 / 50.0 atol = 1.0e-12
    end

    # ── 11. sample_line ───────────────────────────────────────────────
    @testset "sample_line linear field" begin
        mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        T_field = CollocatedScalarField(:T, mesh)

        # Linear field: T(x, y) = x
        for c in 1:nc
            T_field.internal[c] = mesh.cell_centers[1, c]
        end
        for (i, f) in enumerate(T_field.boundary_face_indices)
            T_field.boundary[i] = mesh.face_centers[1, f]
        end

        p1 = SVector(0.0, 0.5)
        p2 = SVector(1.0, 0.5)
        n_pts = 10
        result = sample_line(T_field, mesh, p1, p2, n_pts)

        @test length(result.positions) == n_pts
        @test length(result.distances) == n_pts
        @test length(result.values) == n_pts

        # Distances should be monotonically increasing
        for i in 2:n_pts
            @test result.distances[i] > result.distances[i - 1]
        end

        # Values should be approximately linear in x (nearest-cell interpolation
        # produces a staircase, but overall trend should be increasing)
        @test result.values[end] > result.values[1]

        # Check that the first and last values are roughly at the extremes
        @test result.values[1] < 0.2  # near x=0
        @test result.values[end] > 0.8  # near x=1
    end

    # ── 12. sample_field_at_point ─────────────────────────────────────
    @testset "sample_field_at_point exact" begin
        mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        T_field = CollocatedScalarField(:T, mesh)

        # Set each cell to a distinct value
        for c in 1:nc
            T_field.internal[c] = Float64(c)
        end

        # Sample at each cell center — should return exactly that cell's value
        for c in 1:nc
            pt = SVector(mesh.cell_centers[1, c], mesh.cell_centers[2, c])
            val = sample_field_at_point(T_field, mesh, pt)
            @test val == Float64(c)
        end
    end
end
