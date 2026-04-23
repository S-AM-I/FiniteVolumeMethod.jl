# test/v_and_v_ddt.jl — temporal ddt assembly primitives V&V (v3.99)

using FiniteVolumeMethod
using Test

include("TestHelpers.jl")

const _ddt_euler = FiniteVolumeMethod.assemble_ddt_euler!
const _ddt_bdf2 = FiniteVolumeMethod.assemble_ddt_bdf2!
const _ddt_cn = FiniteVolumeMethod.assemble_ddt_crank_nicolson!
const _ddt = FiniteVolumeMethod.assemble_ddt!
const _TE = FiniteVolumeMethod.TIME_EULER
const _TB = FiniteVolumeMethod.TIME_BDF2
const _TCN = FiniteVolumeMethod.TIME_CRANK_NICOLSON

@testset "V&V: assemble_ddt_euler! — diagonal = ρ·V/Δt" begin
    # Backward Euler contributes ρV/Δt to A[c,c]; no off-diagonal.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    eq = CollocatedEquation(mesh)
    phi_old = zeros(nc)
    rho = 1.2
    dt = 0.01
    _ddt_euler(eq, rho, phi_old, mesh, dt)
    for c in 1:nc
        expected = rho * mesh.cell_volumes[c] / dt
        @test eq.A[c, c] ≈ expected rtol = 1.0e-14
    end
end

@testset "V&V: assemble_ddt_euler! — RHS = ρ·V·φ_old/Δt" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    eq = CollocatedEquation(mesh)
    phi_old = [Float64(c) for c in 1:nc]
    rho = 1.5
    dt = 0.1
    _ddt_euler(eq, rho, phi_old, mesh, dt)
    for c in 1:nc
        expected = rho * mesh.cell_volumes[c] / dt * phi_old[c]
        @test eq.b[c] ≈ expected rtol = 1.0e-14
    end
end

@testset "V&V: assemble_ddt_euler! — Δt inverse scaling" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    phi_old = ones(nc)
    eq1 = CollocatedEquation(mesh)
    _ddt_euler(eq1, 1.0, phi_old, mesh, 0.1)
    eq2 = CollocatedEquation(mesh)
    _ddt_euler(eq2, 1.0, phi_old, mesh, 0.05)
    # Halving dt doubles diagonal + RHS.
    for c in 1:nc
        @test eq2.A[c, c] ≈ 2.0 * eq1.A[c, c] rtol = 1.0e-14
        @test eq2.b[c] ≈ 2.0 * eq1.b[c] rtol = 1.0e-14
    end
end

@testset "V&V: assemble_ddt_euler! — scalar vs vector density identical" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    phi_old = zeros(nc)
    rho_scalar = 1.3
    rho_vector = fill(1.3, nc)
    eq_s = CollocatedEquation(mesh)
    _ddt_euler(eq_s, rho_scalar, phi_old, mesh, 0.1)
    eq_v = CollocatedEquation(mesh)
    _ddt_euler(eq_v, rho_vector, phi_old, mesh, 0.1)
    for c in 1:nc
        @test eq_s.A[c, c] == eq_v.A[c, c]
        @test eq_s.b[c] == eq_v.b[c]
    end
end

@testset "V&V: assemble_ddt_bdf2! — diagonal = 3·ρV/(2Δt)" begin
    # BDF2: coefficient on A[c,c] is (ρV/(2Δt))·3.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    eq = CollocatedEquation(mesh)
    phi_old = zeros(nc)
    phi_old_old = zeros(nc)
    rho = 1.0
    dt = 0.1
    _ddt_bdf2(eq, rho, phi_old, phi_old_old, mesh, dt)
    for c in 1:nc
        coeff = rho * mesh.cell_volumes[c] / (2 * dt)
        @test eq.A[c, c] ≈ 3 * coeff rtol = 1.0e-14
    end
end

@testset "V&V: assemble_ddt_bdf2! — RHS = (4·φ_old - φ_old_old)·coeff" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    eq = CollocatedEquation(mesh)
    phi_old = [2.0 * c for c in 1:nc]
    phi_old_old = [1.0 * c for c in 1:nc]
    rho = 1.0
    dt = 0.1
    _ddt_bdf2(eq, rho, phi_old, phi_old_old, mesh, dt)
    for c in 1:nc
        coeff = rho * mesh.cell_volumes[c] / (2 * dt)
        expected = 4 * coeff * phi_old[c] - coeff * phi_old_old[c]
        @test eq.b[c] ≈ expected rtol = 1.0e-14
    end
end

@testset "V&V: assemble_ddt_bdf2! — φ_old = φ_old_old recovers steady ddt" begin
    # When both history levels agree, BDF2 ddt on a constant field
    # collapses to ρV·(3 - 4 + 1)·φ / (2Δt) · (on RHS) = 0 (on RHS net).
    # Actually the RHS = coeff·(4·φ - φ) = 3·coeff·φ which cancels the
    # diagonal 3·coeff·φ at steady state.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    eq = CollocatedEquation(mesh)
    phi_steady = fill(5.0, nc)
    _ddt_bdf2(eq, 1.0, phi_steady, phi_steady, mesh, 0.1)
    for c in 1:nc
        coeff = mesh.cell_volumes[c] / 0.2
        @test eq.b[c] ≈ 3 * coeff * 5.0 rtol = 1.0e-14
        @test eq.A[c, c] ≈ 3 * coeff rtol = 1.0e-14
    end
end

@testset "V&V: assemble_ddt_crank_nicolson! — matches Euler diagonal" begin
    # Crank-Nicolson ddt-part is identical to Euler.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    phi_old = rand(nc)
    eq_e = CollocatedEquation(mesh)
    _ddt_euler(eq_e, 1.0, phi_old, mesh, 0.1)
    eq_cn = CollocatedEquation(mesh)
    _ddt_cn(eq_cn, 1.0, phi_old, mesh, 0.1)
    for c in 1:nc
        @test eq_cn.A[c, c] == eq_e.A[c, c]
        @test eq_cn.b[c] == eq_e.b[c]
    end
end

@testset "V&V: assemble_ddt! — dispatches to Euler on TIME_EULER" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    phi_old = rand(nc)
    eq_manual = CollocatedEquation(mesh)
    _ddt_euler(eq_manual, 1.0, phi_old, mesh, 0.1)
    eq_auto = CollocatedEquation(mesh)
    _ddt(eq_auto, 1.0, phi_old, mesh, 0.1; scheme = _TE)
    for c in 1:nc
        @test eq_auto.A[c, c] == eq_manual.A[c, c]
        @test eq_auto.b[c] == eq_manual.b[c]
    end
end

@testset "V&V: assemble_ddt! — dispatches to BDF2 on TIME_BDF2" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    phi_old = rand(nc)
    phi_old_old = rand(nc)
    eq_manual = CollocatedEquation(mesh)
    _ddt_bdf2(eq_manual, 1.0, phi_old, phi_old_old, mesh, 0.1)
    eq_auto = CollocatedEquation(mesh)
    _ddt(
        eq_auto, 1.0, phi_old, mesh, 0.1;
        scheme = _TB, phi_old_old = phi_old_old,
    )
    for c in 1:nc
        @test eq_auto.A[c, c] == eq_manual.A[c, c]
        @test eq_auto.b[c] == eq_manual.b[c]
    end
end

@testset "V&V: assemble_ddt! — BDF2 without phi_old_old errors" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    nc = length(mesh.cell_volumes)
    phi_old = zeros(nc)
    eq = CollocatedEquation(mesh)
    @test_throws ErrorException _ddt(
        eq, 1.0, phi_old, mesh, 0.1; scheme = _TB,
    )
end

@testset "V&V: TimeScheme enum values distinct" begin
    # Three distinct enum variants.
    @test _TE != _TB
    @test _TE != _TCN
    @test _TB != _TCN
end
