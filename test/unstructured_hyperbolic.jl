using FiniteVolumeMethod
using Test
using StaticArrays
using DelaunayTriangulation
using LinearAlgebra

# ============================================================
# Helper: Create a refined square mesh
# ============================================================
function make_square_mesh(; max_area = 0.01)
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    boundary_nodes = [[1, 2], [2, 3], [3, 4], [4, 1]]
    tri = triangulate(points; boundary_nodes)
    refine!(tri; max_area = max_area)
    return UnstructuredHyperbolicMesh(tri)
end

function make_centered_square_mesh(; xmin = -0.5, xmax = 0.5, ymin = -0.5, ymax = 0.5, max_area = 0.01)
    points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    boundary_nodes = [[1, 2], [2, 3], [3, 4], [4, 1]]
    tri = triangulate(points; boundary_nodes)
    refine!(tri; max_area = max_area)
    return UnstructuredHyperbolicMesh(tri)
end

# ============================================================
# Mesh Construction Tests
# ============================================================
@testset "UnstructuredHyperbolicMesh Construction" begin
    umesh = make_square_mesh(max_area = 0.05)

    @test umesh.ntri > 0
    @test umesh.n_interior_edges > 0
    @test umesh.n_boundary_edges > 0
    @test length(umesh.tri_verts) == umesh.ntri
    @test length(umesh.tri_areas) == umesh.ntri
    @test length(umesh.tri_centroids) == umesh.ntri

    n_total = umesh.n_interior_edges + umesh.n_boundary_edges
    @test length(umesh.edge_nx) == n_total
    @test length(umesh.edge_ny) == n_total
    @test length(umesh.edge_lengths) == n_total

    # All areas positive
    @test all(umesh.tri_areas .> 0)

    # All edge lengths positive
    @test all(umesh.edge_lengths .> 0)

    # Normals are unit vectors
    for e in 1:n_total
        n_mag = sqrt(umesh.edge_nx[e]^2 + umesh.edge_ny[e]^2)
        @test n_mag ≈ 1.0 atol = 1.0e-14
    end

    # Euler characteristic: V - E + F = 1 for planar triangulation with boundary
    n_verts = length(umesh.vertex_coords)
    euler = n_verts - n_total + umesh.ntri
    @test euler == 1

    # Centroids are inside domain [0,1]^2
    for (x, y) in umesh.tri_centroids
        @test 0.0 <= x <= 1.0
        @test 0.0 <= y <= 1.0
    end

    # Edge left/right consistency
    for e in 1:(umesh.n_interior_edges)
        @test 1 <= umesh.edge_left[e] <= umesh.ntri
        @test 1 <= umesh.edge_right[e] <= umesh.ntri
        @test umesh.edge_left[e] != umesh.edge_right[e]
    end

    # Boundary edges have right = 0
    offset = umesh.n_interior_edges
    for e in 1:(umesh.n_boundary_edges)
        @test umesh.edge_right[offset + e] == 0
        @test 1 <= umesh.edge_left[offset + e] <= umesh.ntri
        @test umesh.edge_bnd_segment[offset + e] > 0
    end

    # Total area should sum to 1.0 (unit square)
    @test sum(umesh.tri_areas) ≈ 1.0 atol = 1.0e-12
end

@testset "UnstructuredHyperbolicMesh Refined" begin
    umesh_coarse = make_square_mesh(max_area = 0.1)
    umesh_fine = make_square_mesh(max_area = 0.01)

    @test umesh_fine.ntri > umesh_coarse.ntri
    @test sum(umesh_fine.tri_areas) ≈ 1.0 atol = 1.0e-12
    @test sum(umesh_coarse.tri_areas) ≈ 1.0 atol = 1.0e-12
end

# ============================================================
# Rotation Tests
# ============================================================
@testset "State Rotation" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    # Test rotation for x-aligned normal (nx=1, ny=0)
    w = SVector(1.0, 2.0, 3.0, 4.0)
    w_rot = rotate_to_normal(law, w, 1.0, 0.0)
    @test w_rot == SVector(1.0, 2.0, 3.0, 4.0)  # No change

    # Test rotation for y-aligned normal (nx=0, ny=1)
    w_rot_y = rotate_to_normal(law, w, 0.0, 1.0)
    @test w_rot_y[1] == 1.0  # density unchanged
    @test w_rot_y[2] ≈ 3.0   # vn = vy
    @test w_rot_y[3] ≈ -2.0  # vt = -vx
    @test w_rot_y[4] == 4.0  # pressure unchanged

    # Test rotation + inverse rotation = identity
    for angle in [0.0, π / 6, π / 4, π / 3, π / 2, π, 3π / 2]
        nx, ny = cos(angle), sin(angle)
        w_test = SVector(1.5, 0.3, -0.2, 2.0)
        w_rot_test = rotate_to_normal(law, w_test, nx, ny)
        F_rot = SVector(0.1, 0.2, 0.3, 0.4)
        F_phys = rotate_flux_from_normal(law, F_rot, nx, ny)
        # The flux rotation should give back the same when re-rotated
        F_re_rot = rotate_to_normal(law, F_phys, nx, ny)
        @test F_re_rot ≈ F_rot atol = 1.0e-14
    end
end

@testset "MHD State Rotation" begin
    eos = IdealGasEOS(5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)

    w = SVector(1.0, 2.0, 3.0, 0.5, 4.0, 1.0, -1.0, 0.3)

    # x-aligned normal
    w_rot = rotate_to_normal(law, w, 1.0, 0.0)
    @test w_rot == w

    # y-aligned normal
    w_rot_y = rotate_to_normal(law, w, 0.0, 1.0)
    @test w_rot_y[2] ≈ 3.0   # vn = vy
    @test w_rot_y[3] ≈ -2.0  # vt = -vx
    @test w_rot_y[6] ≈ -1.0  # Bn = By
    @test w_rot_y[7] ≈ -1.0  # Bt = -Bx

    # Rotation + inverse = identity for flux
    for angle in [π / 6, π / 3, 2π / 3]
        nx, ny = cos(angle), sin(angle)
        F_rot = SVector(0.1, 0.2, 0.3, 0.05, 0.4, 0.15, -0.1, 0.02)
        F_phys = rotate_flux_from_normal(law, F_rot, nx, ny)
        F_back = rotate_to_normal(law, F_phys, nx, ny)
        @test F_back ≈ F_rot atol = 1.0e-14
    end
end

# ============================================================
# Boundary Ghost State Tests
# ============================================================
@testset "Boundary Ghost States" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)
    w = SVector(1.0, 0.5, -0.3, 2.0)
    nx, ny = 1.0, 0.0

    # Transmissive: ghost = interior
    wg = boundary_ghost_state(TransmissiveBC(), law, w, nx, ny)
    @test wg == w

    # Dirichlet: ghost = prescribed state
    w_bc = SVector(0.125, 0.0, 0.0, 0.1)
    wg_d = boundary_ghost_state(DirichletHyperbolicBC(w_bc), law, w, nx, ny)
    @test wg_d == w_bc

    # Reflective: normal velocity flipped
    wg_r = boundary_ghost_state(ReflectiveBC(), law, w, nx, ny)
    @test wg_r[1] == w[1]     # density unchanged
    @test wg_r[2] ≈ -w[2]     # vx flipped (nx=1)
    @test wg_r[3] ≈ w[3]      # vy unchanged
    @test wg_r[4] == w[4]     # pressure unchanged

    # Reflective with angled normal
    nx2, ny2 = 1 / sqrt(2), 1 / sqrt(2)
    wg_r2 = boundary_ghost_state(ReflectiveBC(), law, w, nx2, ny2)
    vn = w[2] * nx2 + w[3] * ny2
    @test wg_r2[2] ≈ w[2] - 2 * vn * nx2 atol = 1.0e-14
    @test wg_r2[3] ≈ w[3] - 2 * vn * ny2 atol = 1.0e-14
end

# ============================================================
# Uniform Flow Preservation
# ============================================================
@testset "Uniform Flow Preservation" begin
    umesh = make_square_mesh(max_area = 0.02)
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    # Uniform state should be preserved exactly
    w_uniform = SVector(1.0, 0.1, -0.2, 2.5)
    ic(x, y) = w_uniform

    prob = UnstructuredHyperbolicProblem(
        law, umesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), ic;
        final_time = 0.1, cfl = 0.3
    )
    centroids, U, t = solve_hyperbolic(prob)

    for i in eachindex(U)
        w = conserved_to_primitive(law, U[i])
        @test w[1] ≈ w_uniform[1] atol = 1.0e-12
        @test w[2] ≈ w_uniform[2] atol = 1.0e-12
        @test w[3] ≈ w_uniform[3] atol = 1.0e-12
        @test w[4] ≈ w_uniform[4] atol = 1.0e-12
    end
end

# ============================================================
# 2D Sod Shock Tube (x-direction)
# ============================================================
@testset "2D Sod Shock Tube (x-direction)" begin
    umesh = make_square_mesh(max_area = 0.005)
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    wL = SVector(1.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.1)
    ic(x, y) = x < 0.5 ? wL : wR

    prob = UnstructuredHyperbolicProblem(
        law, umesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), ic;
        final_time = 0.2, cfl = 0.3
    )
    centroids, U, t = solve_hyperbolic(prob)
    @test t ≈ 0.2

    rho = [conserved_to_primitive(law, U[i])[1] for i in eachindex(U)]
    P = [conserved_to_primitive(law, U[i])[4] for i in eachindex(U)]

    # All positive
    @test all(rho .> 0)
    @test all(P .> 0)
    @test all(isfinite, rho)

    # Shock should have propagated: not all cells at initial values
    @test maximum(rho) < 1.0 + 0.01  # Left state preserved roughly
    @test minimum(rho) > 0.125 - 0.01  # Right state preserved roughly
end

# ============================================================
# Conservation with Periodic-like Setup
# ============================================================
@testset "Mass Conservation (Transmissive)" begin
    umesh = make_centered_square_mesh(max_area = 0.01)
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    # Smooth IC that doesn't reach the boundary
    ic(x, y) = SVector(
        1.0 + 0.5 * exp(-100 * (x^2 + y^2)),
        0.0, 0.0, 1.0
    )

    prob = UnstructuredHyperbolicProblem(
        law, umesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), ic;
        final_time = 0.01, cfl = 0.3
    )
    centroids, U, t = solve_hyperbolic(prob)

    # Initial mass
    mass_0 = sum(
        primitive_to_conserved(law, ic(centroids[i]...))[1] * umesh.tri_areas[i]
            for i in 1:(umesh.ntri)
    )

    # Final mass
    mass_f = sum(U[i][1] * umesh.tri_areas[i] for i in 1:(umesh.ntri))

    # Mass should be approximately conserved (small time, waves don't reach boundary)
    @test mass_f ≈ mass_0 rtol = 0.02
end

# ============================================================
# Solver Options
# ============================================================
@testset "Forward Euler vs SSP-RK3" begin
    umesh = make_square_mesh(max_area = 0.02)
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    wL = SVector(1.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.1)
    ic(x, y) = x < 0.5 ? wL : wR

    prob = UnstructuredHyperbolicProblem(
        law, umesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), ic;
        final_time = 0.05, cfl = 0.3
    )

    _, U_euler, t_euler = solve_hyperbolic(prob; method = :euler)
    _, U_rk3, t_rk3 = solve_hyperbolic(prob; method = :ssprk3)

    @test t_euler ≈ 0.05
    @test t_rk3 ≈ 0.05

    # Both should produce positive densities
    rho_euler = [conserved_to_primitive(law, U_euler[i])[1] for i in eachindex(U_euler)]
    rho_rk3 = [conserved_to_primitive(law, U_rk3[i])[1] for i in eachindex(U_rk3)]
    @test all(rho_euler .> 0)
    @test all(rho_rk3 .> 0)
end

# ============================================================
# All Riemann Solvers
# ============================================================
@testset "Riemann Solvers on Unstructured Mesh" begin
    umesh = make_square_mesh(max_area = 0.02)
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    wL = SVector(1.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.1)
    ic(x, y) = x < 0.5 ? wL : wR

    for solver in [LaxFriedrichsSolver(), HLLSolver(), HLLCSolver()]
        prob = UnstructuredHyperbolicProblem(
            law, umesh, solver, NoReconstruction(),
            TransmissiveBC(), ic;
            final_time = 0.05, cfl = 0.3
        )
        centroids, U, t = solve_hyperbolic(prob)
        rho = [conserved_to_primitive(law, U[i])[1] for i in eachindex(U)]
        @test all(rho .> 0)
        @test all(isfinite, rho)
    end
end

# ============================================================
# Reflective BC — Symmetric Explosion
# ============================================================
@testset "Reflective BC Sedov-like" begin
    umesh = make_centered_square_mesh(max_area = 0.01)
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    function sedov_ic(x, y)
        r = sqrt(x^2 + y^2)
        P = r < 0.05 ? 10.0 : 0.01
        return SVector(1.0, 0.0, 0.0, P)
    end

    prob = UnstructuredHyperbolicProblem(
        law, umesh, HLLSolver(), NoReconstruction(),
        ReflectiveBC(), sedov_ic;
        final_time = 0.05, cfl = 0.25
    )
    centroids, U, t = solve_hyperbolic(prob)

    rho = [conserved_to_primitive(law, U[i])[1] for i in eachindex(U)]
    P = [conserved_to_primitive(law, U[i])[4] for i in eachindex(U)]
    @test all(rho .> 0)
    @test all(P .> 0)
    @test t ≈ 0.05
end

# ============================================================
# Dirichlet BC
# ============================================================
@testset "Dirichlet BC" begin
    umesh = make_square_mesh(max_area = 0.02)
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    wL = SVector(1.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.1)
    ic(x, y) = x < 0.5 ? wL : wR

    prob = UnstructuredHyperbolicProblem(
        law, umesh, HLLSolver(), NoReconstruction(),
        DirichletHyperbolicBC(SVector(0.5, 0.0, 0.0, 0.5)), ic;
        final_time = 0.05, cfl = 0.3
    )
    centroids, U, t = solve_hyperbolic(prob)

    rho = [conserved_to_primitive(law, U[i])[1] for i in eachindex(U)]
    @test all(rho .> 0)
    @test all(isfinite, rho)
end

# ============================================================
# Different Gamma
# ============================================================
@testset "Different Gamma (5/3)" begin
    umesh = make_square_mesh(max_area = 0.02)
    eos = IdealGasEOS(5.0 / 3.0)
    law = EulerEquations{2}(eos)

    wL = SVector(1.0, 0.0, 0.0, 1.0)
    wR = SVector(0.125, 0.0, 0.0, 0.1)
    ic(x, y) = x < 0.5 ? wL : wR

    prob = UnstructuredHyperbolicProblem(
        law, umesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), ic;
        final_time = 0.1, cfl = 0.3
    )
    centroids, U, t = solve_hyperbolic(prob)

    rho = [conserved_to_primitive(law, U[i])[1] for i in eachindex(U)]
    @test all(rho .> 0)
    @test all(isfinite, rho)
end

# ============================================================
# MHD on Unstructured Mesh
# ============================================================
@testset "MHD on Unstructured Mesh" begin
    umesh = make_square_mesh(max_area = 0.02)
    eos = IdealGasEOS(5.0 / 3.0)
    law = IdealMHDEquations{2}(eos)

    # Brio-Wu-like IC in x
    wL = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0)
    wR = SVector(0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0)
    ic(x, y) = x < 0.5 ? wL : wR

    prob = UnstructuredHyperbolicProblem(
        law, umesh, HLLSolver(), NoReconstruction(),
        TransmissiveBC(), ic;
        final_time = 0.05, cfl = 0.25
    )
    centroids, U, t = solve_hyperbolic(prob)

    rho = [conserved_to_primitive(law, U[i])[1] for i in eachindex(U)]
    P = [conserved_to_primitive(law, U[i])[5] for i in eachindex(U)]
    @test all(rho .> 0)
    @test all(P .> 0)
    @test all(isfinite, rho)
end

# ============================================================
# Convergence: Acoustic Wave
# ============================================================
@testset "Acoustic Wave Convergence" begin
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    rho0, P0 = 1.0, 1.0
    c0 = sqrt(1.4 * P0 / rho0)
    A = 1.0e-4  # small amplitude
    k = 2π

    # 1D acoustic wave in x with periodic-like BC (transmissive, short time)
    ic(x, y) = SVector(rho0 + A * sin(k * x), A * c0 / rho0 * sin(k * x), 0.0, P0 + A * 1.4 * sin(k * x))

    errors = Float64[]
    areas = [0.02, 0.005]

    for max_a in areas
        umesh = make_square_mesh(max_area = max_a)
        prob = UnstructuredHyperbolicProblem(
            law, umesh, HLLSolver(), NoReconstruction(),
            TransmissiveBC(), ic;
            final_time = 0.01, cfl = 0.3
        )
        centroids, U, t = solve_hyperbolic(prob)

        # Compute L1 error in density vs IC (short time, wave barely moves)
        err = 0.0
        total_area = 0.0
        for i in eachindex(U)
            x, y = centroids[i]
            rho_exact = rho0 + A * sin(k * (x - c0 * t))
            rho_num = conserved_to_primitive(law, U[i])[1]
            err += abs(rho_num - rho_exact) * umesh.tri_areas[i]
            total_area += umesh.tri_areas[i]
        end
        push!(errors, err / total_area)
    end

    # Error should decrease with refinement
    @test errors[2] < errors[1]
end

# ============================================================
# Per-Segment Boundary Conditions
# ============================================================
@testset "Per-Segment BCs" begin
    umesh = make_square_mesh(max_area = 0.02)
    eos = IdealGasEOS(1.4)
    law = EulerEquations{2}(eos)

    ic(x, y) = SVector(1.0, 0.0, 0.0, 1.0)

    # Different BCs per segment (4 segments for square)
    bcs = Dict(
        1 => ReflectiveBC(),
        2 => TransmissiveBC(),
        3 => ReflectiveBC(),
        4 => TransmissiveBC()
    )

    prob = UnstructuredHyperbolicProblem(
        law, umesh, HLLSolver(), NoReconstruction(),
        bcs, TransmissiveBC(), ic;
        final_time = 0.01, cfl = 0.3
    )
    centroids, U, t = solve_hyperbolic(prob)

    rho = [conserved_to_primitive(law, U[i])[1] for i in eachindex(U)]
    @test all(rho .> 0)
    @test all(isfinite, rho)
end
