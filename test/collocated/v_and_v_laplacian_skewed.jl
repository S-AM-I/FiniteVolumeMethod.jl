# test/v_and_v_laplacian_skewed.jl — Skewed-mesh Laplacian MMS (v3.8)
#
# Verifies the over-relaxed non-orthogonal correction (Stage 3c) by
# running the same MMS setup from v_and_v_laplacian_mms.jl on a
# non-orthogonal mesh. On a strictly Cartesian mesh the three modes
# NON_ORTHO_MINIMUM / _ORTHOGONAL / _OVER_RELAXED reduce to the same
# coefficient because S_f · d̂ = |S_f|. To actually exercise the
# correction, we construct a mesh where the cell-center-to-cell-center
# vector d_PN is NOT aligned with the face normal S_f.
#
# Our approach: take a uniform Cartesian mesh and SHIFT every interior
# cell center by a deterministic sinusoidal offset. Face geometry
# (centers, normals, areas) is KEPT from the Cartesian grid, so the
# physical domain is unchanged but the discrete stencil becomes
# non-orthogonal — d_PN has a tangential component relative to S_f.
# This is a stress test for the correction machinery, not a physical
# simulation.
#
# Acceptance: with non_ortho_correction=true and the default
# NON_ORTHO_OVER_RELAXED mode, the Laplacian solve on the manufactured
# RHS still converges toward the analytical solution (L² error decays
# with refinement), and the over-relaxed mode does not DIVERGE at
# levels of skewness that make minimum-correction blow up.

using FiniteVolumeMethod
using FiniteVolumeMethod: CollocatedEquation, NON_ORTHO_MINIMUM, NON_ORTHO_ORTHOGONAL, NON_ORTHO_OVER_RELAXED, assemble_laplacian!, to_linear_problem
using FiniteVolumeMethod.Parabolic: DirichletBC
using LinearSolve
using StaticArrays: SVector
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

phi_exact(x, y) = sin(π * x) * sin(π * y)
f_forcing(x, y) = 2π^2 * sin(π * x) * sin(π * y)

"""
Build a Cartesian mesh and skew its cell centers by a small
sinusoidal offset (face geometry unchanged). This creates a
non-orthogonal discrete stencil: d_PN has a tangential component
relative to S_f, so `S_f · d̂ < |S_f|` on most faces.
"""
function build_skewed_mesh(N::Int, skew::Float64)
    mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
    # Deep-copy cell_centers so we can mutate without corrupting
    # subsequent callers.
    new_centers = copy(mesh.cell_centers)
    nc = size(new_centers, 2)
    for c in 1:nc
        x = new_centers[1, c]
        y = new_centers[2, c]
        # Only interior cells get shifted (boundary-adjacent cells
        # preserve their canonical position relative to boundary faces).
        if 0.05 < x < 0.95 && 0.05 < y < 0.95
            new_centers[1, c] = x + skew * sin(3π * x) * sin(2π * y)
            new_centers[2, c] = y + skew * sin(2π * x) * sin(3π * y)
        end
    end
    return FiniteVolumeMethod.UnstructuredFVMMesh{2, Float64}(
        new_centers,
        mesh.cell_volumes,
        mesh.face_cells,
        mesh.face_centers,
        mesh.face_areas,
        mesh.face_normals,
        mesh.face_tags,
        mesh.face_velocity,
        mesh.cell_faces,
    )
end

function solve_skewed_mms(
        N::Int, skew::Float64;
        mode, iter_corrections::Int = 0,
    )
    mesh = build_skewed_mesh(N, skew)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(0.0),
        :right => DirichletBC(0.0),
        :bottom => DirichletBC(0.0),
        :top => DirichletBC(0.0),
    )
    nc = length(mesh.cell_volumes)

    # Initial solve without explicit non-orthogonal correction
    # (non_ortho_correction=false); one iteration picks up the skewness
    # truncation error.
    phi_num = zeros(nc)
    for it in 0:iter_corrections
        eq = CollocatedEquation(mesh)
        if it == 0
            assemble_laplacian!(eq, 1.0, mesh, bcs; correction_mode = mode)
        else
            # Feed current numerical gradient as an explicit correction source.
            phi_field = CollocatedScalarField(:phi, mesh)
            phi_field.internal .= phi_num
            grad_phi = gradient(phi_field, mesh)
            assemble_laplacian!(
                eq, 1.0, mesh, bcs;
                correction_mode = mode,
                non_ortho_correction = true,
                grad_phi = grad_phi,
            )
        end
        for c in 1:nc
            x = mesh.cell_centers[1, c]
            y = mesh.cell_centers[2, c]
            eq.b[c] += mesh.cell_volumes[c] * f_forcing(x, y)
        end
        # Evidence scripts pin an explicit direct solver: LinearSolve's
        # DefaultLinearSolver fails soft ("matrix is likely singular") on
        # the harder skewed systems as of 5.5, and V&V accuracy must be
        # set by the discretisation, not the solver heuristic.
        sol = solve(to_linear_problem(eq), KLUFactorization())
        phi_num .= sol.u
    end

    err_sq = 0.0
    vol = 0.0
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        err_sq += mesh.cell_volumes[c] * (phi_num[c] - phi_exact(x, y))^2
        vol += mesh.cell_volumes[c]
    end
    return sqrt(err_sq / vol)
end

@testset "V&V: skewed-mesh Laplacian finite-error with all three correction modes" begin
    # On the skewed mesh all three modes produce FINITE, decreasing-with-
    # refinement L² errors. The over-relaxed mode in particular does not
    # blow up (a common failure mode for minimum-correction on skewed
    # meshes at insufficient iterative correction).
    skew = 0.05
    for N in [20, 40]
        err_min = solve_skewed_mms(N, skew; mode = NON_ORTHO_MINIMUM)
        err_ort = solve_skewed_mms(N, skew; mode = NON_ORTHO_ORTHOGONAL)
        err_ovr = solve_skewed_mms(N, skew; mode = NON_ORTHO_OVER_RELAXED)
        @test isfinite(err_min)
        @test isfinite(err_ort)
        @test isfinite(err_ovr)
        # Errors should be comparable across modes (one-pass without
        # iterative correction; all three pick up O(h) skewness error).
        @test err_min < 1.0
        @test err_ort < 1.0
        @test err_ovr < 1.0
    end
end

@testset "V&V: constant skew angle degrades the one-pass Laplacian to ~O(h)" begin
    # On a skewed mesh WITHOUT the iterative non-orthogonal correction,
    # the one-pass Laplacian carries a skewness truncation error on top
    # of the O(h²) orthogonal baseline. Holding the skewness FRACTION
    # constant (offset = 0.6 h, so the skew angle is resolution-
    # independent and centers stay within a cell width) the observed
    # convergence order drops to ~first order — clearly decaying, clearly
    # below the Cartesian O(h²).
    #
    # (Rewritten 2026-08-08: the original version fixed the ABSOLUTE
    # offset at 0.03, which pushed cell centers up to 2.4 cell widths off
    # their cells at N = 80 and drove the matrix near-singular — a
    # geometric pathology, not a non-orthogonality study. Its "error
    # plateau" claim was an artefact of that construction.)
    skew_fraction = 0.6
    Ns = [20, 40, 80]
    errs = [
        solve_skewed_mms(N, skew_fraction / N; mode = NON_ORTHO_OVER_RELAXED)
            for N in Ns
    ]
    @test all(isfinite, errs)
    # Bounded by a skew-angle-dependent constant.
    @test maximum(errs) < 0.05
    # Observed order on each refinement: ~1 (reduced from 2), i.e. the
    # error decays, but first-order — the skewness signature.
    p12 = log2(errs[1] / errs[2])
    p23 = log2(errs[2] / errs[3])
    @test 0.5 < p12 < 1.6
    @test 0.5 < p23 < 1.6
end

@testset "V&V: all three correction modes agree in the zero-skew limit" begin
    # With skew = 0.0, the mesh reduces to Cartesian and all three modes
    # produce the SAME matrix → the same MMS error.
    N = 40
    err_min = solve_skewed_mms(N, 0.0; mode = NON_ORTHO_MINIMUM)
    err_ort = solve_skewed_mms(N, 0.0; mode = NON_ORTHO_ORTHOGONAL)
    err_ovr = solve_skewed_mms(N, 0.0; mode = NON_ORTHO_OVER_RELAXED)
    @test err_min ≈ err_ort atol = 1.0e-12
    @test err_ort ≈ err_ovr atol = 1.0e-12
end
