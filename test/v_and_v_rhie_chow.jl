# test/v_and_v_rhie_chow.jl — Rhie-Chow interpolation V&V (v3.6)
#
# Completes the Phase 0 operator V&V suite. The three previous V&V
# files (v_and_v_laplacian_mms, v_and_v_operator_mms) verified the
# Laplacian, gradient, and divergence. This file verifies
# `rhie_chow_correction!` — the momentum-interpolation that
# suppresses checkerboard pressure oscillations on collocated meshes.
#
# Three correctness invariants (Rhie & Chow 1983; Jasak 1996 Ch. 5):
#
#   1. LINEAR-PRESSURE identity: if p is an affine function of (x, y),
#      the compact face-normal gradient equals the interpolated
#      cell-center gradient, so the Rhie-Chow correction is zero —
#      the face flux equals the plain linear-interpolation of U·S.
#
#   2. UNIFORM-U sanity: for constant U (and the resulting analytical
#      pressure field), the corrected face flux equals U·S everywhere.
#
#   3. CHECKERBOARD suppression: a pressure field with the pattern
#      p[c] = (-1)^(i+j) should produce a non-trivial correction that
#      DAMPS the face flux relative to the naive U·S — this is the
#      Rhie-Chow design goal.

using FiniteVolumeMethod
using LinearAlgebra: norm, dot
using LinearSolve
using StaticArrays: SVector
using Test

include("TestHelpers.jl")

@testset "V&V: Rhie-Chow identity on linear pressure field" begin
    # Affine pressure field: p(x, y) = 2.0 + 3x - 1.5y. Compact
    # face-normal gradient equals interpolated cell-center gradient
    # to machine precision → Rhie-Chow correction is zero. Corrected
    # flux must equal U·S up to floating-point noise.
    N = 20
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    U = CollocatedVectorField(:U, mesh)
    p = CollocatedScalarField(:p, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        # Uniform velocity
        U.internal[c] = SVector(1.0, 0.5)
        p.internal[c] = 2.0 + 3.0 * x - 1.5 * y
    end
    # Boundary values from exact field
    for (i, f) in enumerate(U.boundary_face_indices)
        x = mesh.face_centers[1, f]
        y = mesh.face_centers[2, f]
        U.boundary[i] = SVector(1.0, 0.5)
    end
    for (i, f) in enumerate(p.boundary_face_indices)
        x = mesh.face_centers[1, f]
        y = mesh.face_centers[2, f]
        p.boundary[i] = 2.0 + 3.0 * x - 1.5 * y
    end

    phi = FaceFluxField(:phi, mesh; value = 0.0)
    # Dummy momentum-diagonal A_P; choose so D_f = V/A_P is bounded.
    A_P = fill(1.0, nc)

    rhie_chow_correction!(phi, U, p, A_P, mesh)

    # Naive face flux: linear-interpolated U · S_f at each internal
    # face; boundary faces use boundary value.
    phi_naive = zeros(nf)
    for f in 1:nf
        Af = mesh.face_areas[f]
        nhat = SVector(mesh.face_normals[1, f], mesh.face_normals[2, f])
        if mesh.face_cells[2, f] == 0
            # Boundary face
            phi_naive[f] = Af * (
                U.boundary[findfirst(==(f), U.boundary_face_indices)][1] * nhat[1] +
                    U.boundary[findfirst(==(f), U.boundary_face_indices)][2] * nhat[2]
            )
        else
            # Internal: linear interpolation in cell centers
            P = mesh.face_cells[1, f]
            Nn = mesh.face_cells[2, f]
            U_f = 0.5 * (U.internal[P] + U.internal[Nn])
            phi_naive[f] = Af * (U_f[1] * nhat[1] + U_f[2] * nhat[2])
        end
    end

    # Interior-face comparison (skip boundary faces where Rhie-Chow
    # doesn't apply a correction anyway).
    max_diff = 0.0
    for f in 1:nf
        mesh.face_cells[2, f] == 0 && continue
        max_diff = max(max_diff, abs(phi.values[f] - phi_naive[f]))
    end
    @test max_diff < 1.0e-10   # machine-precision match
end

@testset "V&V: Rhie-Chow suppresses checkerboard pressure mode" begin
    # Construct a pressure field with a checkerboard pattern that the
    # INTERPOLATED gradient doesn't see but the compact face-normal
    # gradient does. Rhie-Chow must produce a non-zero correction on
    # those faces.
    N = 10
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    U = CollocatedVectorField(:U, mesh)
    p = CollocatedScalarField(:p, mesh)
    for c in 1:nc
        # Uniform velocity (so the naive U·S_f captures everything)
        U.internal[c] = SVector(1.0, 0.0)
    end
    # Recover (i, j) indices: cell c = (j-1)*N + i for 1 ≤ i, j ≤ N.
    for c in 1:nc
        j = div(c - 1, N) + 1
        i = c - (j - 1) * N
        p.internal[c] = (-1.0)^(i + j)   # ±1 checkerboard
    end
    for (k, f) in enumerate(U.boundary_face_indices)
        U.boundary[k] = SVector(1.0, 0.0)
    end
    for (k, f) in enumerate(p.boundary_face_indices)
        p.boundary[k] = 0.0
    end

    phi_rc = FaceFluxField(:phi_rc, mesh; value = 0.0)
    phi_naive = FaceFluxField(:phi_naive, mesh; value = 0.0)
    A_P = fill(1.0, nc)

    rhie_chow_correction!(phi_rc, U, p, A_P, mesh)
    compute_face_flux!(phi_naive, U, mesh)    # linear-interp only, no correction

    # Internal-face differences — there MUST be a non-zero correction on
    # the faces where the checkerboard appears.
    max_diff = 0.0
    for f in 1:nf
        mesh.face_cells[2, f] == 0 && continue
        max_diff = max(max_diff, abs(phi_rc.values[f] - phi_naive.values[f]))
    end
    # Rhie-Chow is doing its job: face flux DEVIATES from the naive
    # interpolation to suppress the pressure checkerboard.
    @test max_diff > 1.0e-3
end

@testset "V&V: Rhie-Chow preserves face flux for constant-pressure field" begin
    # Constant pressure → grad_p_compact = grad_p_interp = 0 at every
    # face, so Rhie-Chow correction is zero. Face flux equals naive U·S.
    N = 15
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    U = CollocatedVectorField(:U, mesh)
    p = CollocatedScalarField(:p, mesh; value = 1.234)   # constant
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(sin(π * x), cos(π * y))  # smooth but varying
    end
    for (k, f) in enumerate(U.boundary_face_indices)
        x = mesh.face_centers[1, f]
        y = mesh.face_centers[2, f]
        U.boundary[k] = SVector(sin(π * x), cos(π * y))
    end

    phi_rc = FaceFluxField(:phi_rc, mesh; value = 0.0)
    phi_naive = FaceFluxField(:phi_naive, mesh; value = 0.0)
    A_P = fill(1.0, nc)

    rhie_chow_correction!(phi_rc, U, p, A_P, mesh)
    compute_face_flux!(phi_naive, U, mesh)

    max_diff = 0.0
    for f in 1:nf
        mesh.face_cells[2, f] == 0 && continue
        max_diff = max(max_diff, abs(phi_rc.values[f] - phi_naive.values[f]))
    end
    @test max_diff < 1.0e-10
end

@testset "V&V: Rhie-Chow face coefficient is the harmonic pressure-Laplacian D_f" begin
    # Finding: the corrected flux can only be divergence-consistent with
    # the pressure equation if the D_f used by `rhie_chow_correction!` is
    # IDENTICAL to the face diffusivity of the pressure Laplacian
    # (harmonic mean via `_face_diffusivity`).  This identity test
    # reconstructs the expected flux with the harmonic D_f and compares —
    # with the previous arithmetic interpolation it fails whenever A_P
    # varies across a face.
    N = 10
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    U = CollocatedVectorField(:U, mesh)
    p = CollocatedScalarField(:p, mesh)
    A_P = Vector{Float64}(undef, nc)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(0.3 * sin(π * x), -0.2 * cos(π * y))
        p.internal[c] = sin(2π * x) * cos(π * y)
        A_P[c] = exp(2.0 * sin(2π * x) * sin(2π * y))  # strong variation
    end
    for (k, f) in enumerate(p.boundary_face_indices)
        p.boundary[k] = p.internal[FiniteVolumeMethod.owner(mesh, f)]
    end

    phi = FaceFluxField(:phi, mesh; value = 0.0)
    rhie_chow_correction!(phi, U, p, A_P, mesh)

    D_vec = [mesh.cell_volumes[c] / A_P[c] for c in 1:nc]
    grad_p = gradient(p, mesh)

    n_varying = 0
    for f in 1:nf
        FiniteVolumeMethod.is_internal_face(mesh, f) || continue
        P = FiniteVolumeMethod.owner(mesh, f)
        Nc = FiniteVolumeMethod.neighbour(mesh, f)
        w = FiniteVolumeMethod.face_weight(mesh, f)
        S_f = FiniteVolumeMethod.face_normal_area(mesh, f)
        U_f = w * U.internal[P] + (1.0 - w) * U.internal[Nc]
        d_vec, d_mag = FiniteVolumeMethod.owner_neighbour_distance(mesh, f)
        compact = (p.internal[Nc] - p.internal[P]) / d_mag * mesh.face_areas[f]
        interp = dot(w * grad_p[P] + (1.0 - w) * grad_p[Nc], S_f)
        D_f = FiniteVolumeMethod._face_diffusivity(D_vec, mesh, f)
        expected = dot(U_f, S_f) - D_f * (compact - interp)
        @test phi.values[f] ≈ expected atol = 1.0e-13
        if abs(D_vec[P] - D_vec[Nc]) > 1.0e-3
            n_varying += 1
        end
    end
    # The mesh must actually exercise varying A_P across faces
    @test n_varying > 50
end

@testset "V&V: projection with varying A_P reduces flux imbalance" begin
    # End-to-end pressure projection with a strongly varying momentum
    # diagonal: solving the pressure equation and applying the
    # velocity/flux corrections must substantially reduce the raw cell
    # flux imbalance of the H/A field.
    mesh = build_cartesian_unstructured_mesh(12, 12, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(), :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(), :top => NoSlipWallBC(),
    )
    prob = IncompressibleProblem(mesh, bcs, SIMPLE(); nu = 0.01)
    state = IncompressibleState(mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        state.A_P[c] = exp(3.0 * sin(2π * x) * sin(2π * y))
        state.H_U[c] = state.A_P[c] * 0.1 * SVector(sin(π * x), sin(π * y))
    end
    FiniteVolumeMethod.update_boundary_velocity!(state, bcs, mesh)

    phi_HbyA = FiniteVolumeMethod.compute_HbyA_flux(state, mesh)
    imb0 = zeros(nc)
    for f in 1:nf
        P = FiniteVolumeMethod.owner(mesh, f)
        Nc = FiniteVolumeMethod.neighbour(mesh, f)
        imb0[P] += phi_HbyA[f]
        Nc != 0 && (imb0[Nc] -= phi_HbyA[f])
    end
    r_pre = sum(abs, imb0)
    @test r_pre > 0.1  # a genuinely non-solenoidal starting field

    p_eq = CollocatedEquation(mesh)
    FiniteVolumeMethod.assemble_pressure!(p_eq, state, prob)
    FiniteVolumeMethod.fix_pressure_reference!(p_eq, 1, 0.0)
    p_sol = solve(to_linear_problem(p_eq))
    state.p.internal .= p_sol.u
    FiniteVolumeMethod.update_boundary_pressure!(state, bcs, mesh)
    FiniteVolumeMethod.correct_velocity!(state, mesh)
    FiniteVolumeMethod.update_boundary_velocity!(state, bcs, mesh)
    FiniteVolumeMethod.correct_fluxes!(state, mesh)

    r_post = FiniteVolumeMethod.continuity_residual(state, mesh; normalize = false)
    # One projection with the CONSISTENT harmonic D_f reduces the raw
    # imbalance decisively (observed ratio ~0.34 for ~400x A_P variation;
    # the remaining part is the Rhie-Chow gradient-smoothing term, which
    # subsequent correctors remove).
    @test r_post < 0.5 * r_pre
end
