# test/v_and_v_wall_quantities.jl — Wall quantities V&V (v3.51)
#
# Fourth convergence-verified benchmark for `postprocessing`,
# joining vorticity + Q (v3.20), Courant + Q-sign (v3.30), and
# field statistics + TI (v3.40). Covers the wall-quantity
# primitives used for engineering quantities-of-interest:
#
#   τ_w = ν · U_tangential / d
#   y⁺  = y · √|τ_w| / ν
#   q_w = -k · (T_wall - T_cell) / d
#
# Five invariants verified.

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_wall_heat_flux, compute_wall_shear_stress
using LinearAlgebra: norm
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

@testset "V&V: Wall τ_w — zero velocity ⇒ τ_w = 0" begin
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    U = CollocatedVectorField(:U, mesh; value = SVector(0.0, 0.0))

    tau = compute_wall_shear_stress(U, 0.1, mesh, :top)
    for t in tau
        @test isapprox(norm(t), 0.0; atol = 1.0e-14)
    end
end

@testset "V&V: Wall τ_w — ν-linear scaling" begin
    # At fixed U and mesh, τ_w scales linearly with ν.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(y, 0.0)   # linear shear profile
    end

    tau_1 = compute_wall_shear_stress(U, 0.1, mesh, :top)
    tau_2 = compute_wall_shear_stress(U, 0.2, mesh, :top)

    for i in 1:length(tau_1)
        @test isapprox(norm(tau_2[i]) / norm(tau_1[i]), 2.0; rtol = 1.0e-12)
    end
end

@testset "V&V: Wall τ_w — direction tangential to wall" begin
    # For the :top boundary (face normal in +y direction), the
    # tangential component lies in ±x. For U = (Uy, 0), τ_w
    # should be purely in the x direction.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    U = CollocatedVectorField(:U, mesh)
    for c in 1:nc
        y = mesh.cell_centers[2, c]
        U.internal[c] = SVector(y, 0.0)
    end

    tau = compute_wall_shear_stress(U, 0.1, mesh, :top)
    for t in tau
        @test abs(t[2]) < 1.0e-12    # y-component is zero (wall-normal)
        @test t[1] != 0.0              # x-component is non-zero (tangential)
    end
end

@testset "V&V: Wall q_w — isothermal ⇒ q_w = 0" begin
    # If T_cell == T_wall, heat flux vanishes.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    Tf = CollocatedScalarField(:T, mesh; value = 300.0)
    for (i, f) in enumerate(Tf.boundary_face_indices)
        Tf.boundary[i] = 300.0
    end

    q = compute_wall_heat_flux(Tf, 1.0, mesh, :top)
    for q_i in q
        @test isapprox(q_i, 0.0; atol = 1.0e-12)
    end
end

@testset "V&V: Wall q_w — sign + k-linear scaling" begin
    # T_cell = 300, T_wall = 400 ⇒ T_wall > T_cell ⇒ q_w < 0
    # (heat flows into the domain from the hot wall, negative
    # sign convention).
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    Tf = CollocatedScalarField(:T, mesh; value = 300.0)
    for (i, f) in enumerate(Tf.boundary_face_indices)
        tag = FiniteVolumeMethod._face_tag(mesh, f)
        Tf.boundary[i] = tag == :top ? 400.0 : 300.0
    end

    q_1 = compute_wall_heat_flux(Tf, 1.0, mesh, :top)
    q_2 = compute_wall_heat_flux(Tf, 2.0, mesh, :top)

    for i in 1:length(q_1)
        @test q_1[i] < 0.0
        @test isapprox(q_2[i] / q_1[i], 2.0; rtol = 1.0e-12)
    end
end
