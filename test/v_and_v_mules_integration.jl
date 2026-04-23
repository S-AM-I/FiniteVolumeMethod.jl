# test/v_and_v_mules_integration.jl — MULES wired into α-transport (v3.92)
#
# Complements `v_and_v_mules.jl` which tests the flux limiter primitive
# in isolation. Here we verify that the primitive is correctly wired
# into `assemble_alpha!` with `use_mules = true` (the new default):
#
#   1. Boundedness: after one step from α = 0.5 with an aggressive
#      compressive flux, every cell's α lies in [0, 1].
#   2. Mass conservation: total α·V is preserved to 1e-12 in a closed box.
#   3. Regression vs legacy: use_mules=true and use_mules=false agree
#      within 5% on smooth initial data (no boundedness challenge).
#   4. High-compression: a sinusoidal α wave remains in [0, 1] across
#      10 explicit steps under MULES.
#
# Shipped alongside the Wave-1 MULES default switch.

using FiniteVolumeMethod
using LinearAlgebra
using SparseArrays
using StaticArrays
using Test

include("TestHelpers.jl")

const _SparseMatrixCSC = SparseArrays.SparseMatrixCSC

# Apply a single explicit step of the α-transport assembly with MULES on
# or off. Because the MULES path assembles a pure diagonal system, the
# "solve" is a cheap elementwise divide — we do it directly so the test
# does not depend on LinearSolve.jl. The legacy path is fully implicit
# and needs a real solve.
function _step_alpha_explicit!(
        alpha::CollocatedScalarField{Float64},
        phi::FaceFluxField{Float64},
        mesh,
        dt::Float64;
        C_alpha::Float64 = 1.0,
        use_mules::Bool = true,
    )
    bcs = Dict{Symbol, FiniteVolumeMethod.AbstractBoundaryCondition}(
        :left => FiniteVolumeMethod.ParabolicNeumann(0.0),
        :right => FiniteVolumeMethod.ParabolicNeumann(0.0),
        :top => FiniteVolumeMethod.ParabolicNeumann(0.0),
        :bottom => FiniteVolumeMethod.ParabolicNeumann(0.0),
    )
    eq = FiniteVolumeMethod.CollocatedEquation(mesh)
    FiniteVolumeMethod.assemble_alpha!(
        eq, alpha, phi, mesh, bcs;
        dt = dt, C_alpha = C_alpha, use_mules = use_mules,
    )
    # Direct elementwise solve (diagonal A on the MULES path).
    lp = FiniteVolumeMethod.to_linear_problem(eq)
    A = lp.A
    b = lp.b
    nc = length(b)
    x = Vector{Float64}(undef, nc)
    if use_mules
        for c in 1:nc
            x[c] = b[c] / A[c, c]
        end
    else
        x .= A \ Vector(b)
    end
    for c in 1:nc
        alpha.internal[c] = x[c]
    end
    return nothing
end

@testset "V&V: MULES integration — boundedness after one α-step (α=0.5 + aggressive φ_c)" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    alpha = CollocatedScalarField(:alpha, mesh; value = 0.5)
    phi = FaceFluxField(:phi, mesh; value = 0.0)
    # Aggressive, spatially varying velocity flux — the compression
    # term `|phi| · α(1-α)` would violate boundedness without MULES.
    for f in 1:nf
        phi.values[f] = 0.5 * sin(3.0 * f)
    end

    _step_alpha_explicit!(alpha, phi, mesh, 1.0e-2; C_alpha = 2.0, use_mules = true)

    for c in 1:nc
        @test -1.0e-10 <= alpha.internal[c] <= 1.0 + 1.0e-10
    end
end

@testset "V&V: MULES integration — closed-box α·V conservation" begin
    mesh = build_cartesian_unstructured_mesh(10, 10, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    alpha = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        alpha.internal[c] = 0.3 + 0.4 * mesh.cell_centers[1, c]
    end

    # Closed box: boundary faces have zero volumetric flux; internal
    # fluxes are divergence-free (linearly varying x-flux cancels).
    phi = FaceFluxField(:phi, mesh; value = 0.0)
    for f in 1:nf
        if is_internal_face(mesh, f)
            # Divergence-free field in a Cartesian mesh: constant U_x
            # dotted with S_f is constant on vertical faces, zero on
            # horizontal ones → net divergence zero per cell.
            S = face_normal_area(mesh, f)
            phi.values[f] = 0.1 * S[1]
        end
    end

    total_before = sum(alpha.internal[c] * mesh.cell_volumes[c] for c in 1:nc)
    _step_alpha_explicit!(alpha, phi, mesh, 1.0e-3; C_alpha = 1.0, use_mules = true)
    total_after = sum(alpha.internal[c] * mesh.cell_volumes[c] for c in 1:nc)

    @test isapprox(total_after, total_before; rtol = 1.0e-12, atol = 1.0e-12)
end

@testset "V&V: MULES integration — legacy vs MULES on smooth field (≤5 %)" begin
    # On a smooth α field with a gentle velocity flux, MULES should
    # not substantially alter the solution vs the legacy implicit path.
    mesh = build_cartesian_unstructured_mesh(12, 12, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    alpha_m = CollocatedScalarField(:alpha, mesh)
    alpha_l = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        val = 0.5 + 0.3 * sin(2.0 * π * x)
        alpha_m.internal[c] = val
        alpha_l.internal[c] = val
    end

    phi = FaceFluxField(:phi, mesh; value = 0.0)
    for f in 1:nf
        if is_internal_face(mesh, f)
            S = face_normal_area(mesh, f)
            phi.values[f] = 0.05 * S[1]
        end
    end

    _step_alpha_explicit!(
        alpha_m, phi, mesh, 5.0e-3;
        C_alpha = 0.0, use_mules = true,
    )
    _step_alpha_explicit!(
        alpha_l, phi, mesh, 5.0e-3;
        C_alpha = 0.0, use_mules = false,
    )

    max_diff = maximum(abs, alpha_m.internal .- alpha_l.internal)
    @test max_diff < 0.05
end

@testset "V&V: MULES integration — sinusoidal α stays bounded across 10 steps" begin
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    nf = size(mesh.face_cells, 2)

    alpha = CollocatedScalarField(:alpha, mesh)
    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        alpha.internal[c] = 0.5 + 0.49 * sin(2.0 * π * x) * cos(2.0 * π * y)
    end

    phi = FaceFluxField(:phi, mesh; value = 0.0)
    for f in 1:nf
        if is_internal_face(mesh, f)
            S = face_normal_area(mesh, f)
            phi.values[f] = 0.02 * S[1]
        end
    end

    dt = 1.0e-3
    for _ in 1:10
        _step_alpha_explicit!(alpha, phi, mesh, dt; C_alpha = 1.5, use_mules = true)
    end

    for c in 1:nc
        @test -1.0e-9 <= alpha.internal[c] <= 1.0 + 1.0e-9
    end
end
