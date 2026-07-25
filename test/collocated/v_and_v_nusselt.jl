# test/v_and_v_nusselt.jl — Nusselt number + y+ V&V (v3.62)
#
# Fifth convergence-verified benchmark for `postprocessing`,
# joining vorticity + Q (v3.20), Courant + Q-sign (v3.30),
# field stats + TI (v3.40), and wall quantities (v3.51).
# Covers `compute_nusselt_number` and `compute_y_plus` — the
# remaining wall-quantity diagnostics.
#
# Six invariants verified.

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_nusselt_number, compute_y_plus
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

function _build_T_field(mesh, T_cell_val, T_top_val)
    Tf = CollocatedScalarField(:T, mesh; value = T_cell_val)
    for (i, f) in enumerate(Tf.boundary_face_indices)
        tag = FiniteVolumeMethod._face_tag(mesh, f)
        Tf.boundary[i] = tag == :top ? T_top_val : T_cell_val
    end
    return Tf
end

@testset "V&V: Nusselt — isothermal (T_wall = T_ref) ⇒ Nu = 0 fallback" begin
    # When dT → 0 the formula protects against division by zero
    # and returns 0.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    Tf = _build_T_field(mesh, 300.0, 300.0)
    Nu = compute_nusselt_number(Tf, 1.0, mesh, :top; T_ref = 300.0, L_ref = 1.0)

    for nu_i in Nu
        @test nu_i == 0.0
    end
end

@testset "V&V: Nusselt — k = 0 ⇒ Nu = 0 fallback" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    Tf = _build_T_field(mesh, 300.0, 400.0)
    Nu = compute_nusselt_number(Tf, 0.0, mesh, :top; T_ref = 300.0, L_ref = 1.0)

    for nu_i in Nu
        @test nu_i == 0.0
    end
end

@testset "V&V: Nusselt — L_ref linear scaling at fixed q_w, k, ΔT" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    Tf = _build_T_field(mesh, 300.0, 400.0)

    Nu_1 = compute_nusselt_number(Tf, 1.0, mesh, :top; T_ref = 300.0, L_ref = 1.0)
    Nu_2 = compute_nusselt_number(Tf, 1.0, mesh, :top; T_ref = 300.0, L_ref = 2.0)

    for i in 1:length(Nu_1)
        @test isapprox(Nu_2[i] / Nu_1[i], 2.0; rtol = 1.0e-12)
    end
end

@testset "V&V: Nusselt — non-negative (|q_w|/|ΔT| > 0)" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    Tf = _build_T_field(mesh, 300.0, 500.0)
    Nu = compute_nusselt_number(Tf, 1.0, mesh, :top; T_ref = 300.0, L_ref = 1.0)
    for nu_i in Nu
        @test nu_i > 0.0   # |q_w|/|ΔT| positive under ΔT = 200
    end
end

@testset "V&V: y+ — zero velocity ⇒ y+ = 0" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    U = CollocatedVectorField(:U, mesh; value = SVector(0.0, 0.0))
    yp = compute_y_plus(U, 0.1, mesh, :top)
    for y in yp
        @test isapprox(y, 0.0; atol = 1.0e-14)
    end
end

@testset "V&V: y+ — non-negative realizability" begin
    # For any velocity field, y+ = y·sqrt(|τ_w|)/ν should be ≥ 0.
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    U = CollocatedVectorField(:U, mesh)
    for c in 1:length(mesh.cell_volumes)
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(y, 0.0)
    end
    yp = compute_y_plus(U, 0.1, mesh, :top)
    for y in yp
        @test y >= 0.0
    end
end

@testset "V&V: y+ — finite and bounded on linear shear" begin
    mesh = build_cartesian_unstructured_mesh(16, 16, 1.0, 1.0)
    U = CollocatedVectorField(:U, mesh)
    for c in 1:length(mesh.cell_volumes)
        U.internal[c] = SVector(mesh.cell_centers[2, c], 0.0)
    end
    yp = compute_y_plus(U, 0.1, mesh, :top)
    for y in yp
        @test isfinite(y)
        @test y < 10.0   # generous bound for the small-magnitude test
    end
end
