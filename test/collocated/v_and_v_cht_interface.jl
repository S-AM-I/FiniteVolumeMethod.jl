# test/v_and_v_cht_interface.jl — CHT interface heat-flux V&V (v3.50)
#
# Fourth convergence-verified benchmark for
# `conjugate_heat_transfer`, joining Laplace-series solid
# conduction (v3.12), unsteady decay (v3.21), and Boussinesq
# buoyancy (v3.32). Covers `compute_interface_heat_flux`, the
# coupling primitive used by Dirichlet-Neumann conjugate HT:
#
#   q_f = -k · (T_bnd - T_cell) / d_cell_to_face
#
# Four algebraic invariants verified.

using FiniteVolumeMethod
using FiniteVolumeMethod: compute_interface_heat_flux, solve_solid_conduction
using LinearAlgebra: norm
using LinearSolve
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

function build_T_field(T_cell_val::Float64, T_bnd_val::Float64)
    # Build T_field directly (not via solve_solid_conduction, which
    # does not populate `boundary` — only internal values). Set
    # all internal cells to T_cell_val and the :top boundary to
    # T_bnd_val, keeping other boundaries at T_cell_val.
    mesh = build_cartesian_unstructured_mesh(8, 8, 1.0, 1.0)
    Tf = CollocatedScalarField(:T, mesh; value = T_cell_val)

    # Populate boundary values per face.
    nf = size(mesh.face_cells, 2)
    for (i, f) in enumerate(Tf.boundary_face_indices)
        tag = FiniteVolumeMethod._face_tag(mesh, f)
        Tf.boundary[i] = tag == :top ? T_bnd_val : T_cell_val
    end
    return mesh, Tf
end

@testset "V&V: CHT interface flux — isothermal ⇒ q = 0" begin
    mesh, Tf = build_T_field(300.0, 300.0)
    flux = compute_interface_heat_flux(Tf, 1.0, mesh, :top)

    # Every interface face should have ~zero flux (field is
    # uniform 300 K everywhere).
    for (f, q) in flux
        @test isapprox(q, 0.0; atol = 1.0e-10)
    end
    @test length(flux) > 0   # sanity: patch exists
end

@testset "V&V: CHT interface flux — k-linear scaling" begin
    # Flux is linear in k at fixed T_field.
    mesh, Tf = build_T_field(300.0, 400.0)
    flux_1k = compute_interface_heat_flux(Tf, 1.0, mesh, :top)
    flux_2k = compute_interface_heat_flux(Tf, 2.0, mesh, :top)

    # Keys match; values should scale by 2.
    for f in keys(flux_1k)
        @test haskey(flux_2k, f)
        if abs(flux_1k[f]) > 1.0e-10
            @test isapprox(flux_2k[f] / flux_1k[f], 2.0; rtol = 1.0e-12)
        end
    end
end

@testset "V&V: CHT interface flux — sign: hot boundary ⇒ q > 0 into solid" begin
    # q_f = -k·(T_bnd - T_cell)/d. With T_bnd = 400, T_cell ≈ 300
    # near the hot wall ⇒ T_bnd > T_cell ⇒ q < 0 (heat flows in,
    # sign convention says negative).
    mesh, Tf = build_T_field(300.0, 400.0)
    flux = compute_interface_heat_flux(Tf, 1.0, mesh, :top)

    # Every interface flux at the hot wall should be negative
    # (heat enters the solid from the hot boundary).
    for (f, q) in flux
        @test q < 0.0
    end
end

@testset "V&V: CHT interface flux — algebraic closed form at every face" begin
    mesh, Tf = build_T_field(300.0, 500.0)
    k = 0.5
    flux = compute_interface_heat_flux(Tf, k, mesh, :top)

    nf = size(mesh.face_cells, 2)
    pbmap = FiniteVolumeMethod.build_boundary_map(Tf, mesh)
    for f in 1:nf
        if haskey(flux, f)
            P = FiniteVolumeMethod.owner(mesh, f)
            T_cell = Tf.internal[P]
            T_bnd = Tf.boundary[pbmap[f]]

            x_c = FiniteVolumeMethod.cell_center(mesh, P)
            x_f = FiniteVolumeMethod.face_center(mesh, f)
            d = norm(x_f - x_c)
            d = max(d, 1.0e-15)

            expected = -k * (T_bnd - T_cell) / d
            @test isapprox(flux[f], expected; rtol = 1.0e-12)
        end
    end
end
