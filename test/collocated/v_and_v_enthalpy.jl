# test/v_and_v_enthalpy.jl — Enthalpy-form energy equation V&V
#
# The enthalpy form `∂(ρh)/∂t + ∇·(ρUh) = ∇·(k/Cp · ∇h)` should agree
# with the temperature form for constant Cp (up to the linear shift
# h = Cp·(T - T_ref)). This file exercises four invariants:
#
#   1. Round-trip: h = h_from_T(T); T = T_from_h(h) must return the
#      identity and conserve the boundary-face indices.
#   2. BC translation: Dirichlet, Neumann and Robin BCs translate
#      consistently and invert back to the T-values exactly.
#   3. Constant-Cp equivalence: pure conduction (zero velocity) produces
#      identical steady-state T fields under the h-form and T-form.
#   4. Adiabatic no-drift: initial uniform field + insulated walls + zero
#      Q_gen ⇒ ∂h/∂t ≡ 0, so ∫ρh dV is conserved and `solve` leaves the
#      field untouched.
#
# All tests run on a small Cartesian mesh; the algebraic checks dominate.

using FiniteVolumeMethod
using FiniteVolumeMethod: shift, solve_incompressible_thermal
using FiniteVolumeMethod.Parabolic: DirichletBC, NeumannBC, RobinBC
using LinearSolve
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

# Shorthand references to the internal helpers under test.
const h_from_T = FiniteVolumeMethod.h_from_T
const T_from_h = FiniteVolumeMethod.T_from_h
const enthalpy_field_from_temperature =
    FiniteVolumeMethod.enthalpy_field_from_temperature
const temperature_from_enthalpy! = FiniteVolumeMethod.temperature_from_enthalpy!
const enthalpy_bcs_from_temperature =
    FiniteVolumeMethod.enthalpy_bcs_from_temperature

@testset "V&V: enthalpy — scalar round-trip" begin
    T_ref = 300.0
    Cp = 1005.0
    for T_val in (250.0, 300.0, 350.5, 800.25)
        h = h_from_T(T_val, T_ref, Cp)
        T_back = T_from_h(h, T_ref, Cp)
        @test isapprox(T_back, T_val; rtol = 1.0e-14)
    end

    # h(T_ref) must be exactly zero — enthalpy datum.
    @test h_from_T(T_ref, T_ref, Cp) == 0.0
end

@testset "V&V: enthalpy — field round-trip" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    T_ref = 300.0
    Cp = 1005.0

    T_field = CollocatedScalarField(:T, mesh; value = 400.0)
    for c in eachindex(T_field.internal)
        T_field.internal[c] = 300.0 + c
    end
    for b in eachindex(T_field.boundary)
        T_field.boundary[b] = 400.0 + b
    end

    h = enthalpy_field_from_temperature(T_field, T_ref, Cp)
    @test length(h.internal) == length(T_field.internal)
    @test length(h.boundary) == length(T_field.boundary)
    @test h.boundary_face_indices == T_field.boundary_face_indices

    # Each cell: h == Cp · (T - T_ref)
    for c in eachindex(T_field.internal)
        @test isapprox(h.internal[c], Cp * (T_field.internal[c] - T_ref); rtol = 1.0e-14)
    end

    # Round-trip back to T
    T_back = CollocatedScalarField(:T, mesh; value = 0.0)
    temperature_from_enthalpy!(T_back, h, T_ref, Cp)
    @test all(isapprox.(T_back.internal, T_field.internal; rtol = 1.0e-14))
    @test all(isapprox.(T_back.boundary, T_field.boundary; rtol = 1.0e-14))
end

@testset "V&V: enthalpy — BC translation round-trip" begin
    T_ref = 310.0
    Cp = 1200.0
    bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(400.0),
        :right => NeumannBC(5.0),         # ∂T/∂n = 5 K/m
        :top => RobinBC(2.0, 0.5, 830.0), # 2T + 0.5·∂T/∂n = 830
    )
    bcs_h = enthalpy_bcs_from_temperature(bcs_T, T_ref, Cp)

    # Dirichlet: value transforms via h = Cp·(T - T_ref)
    bc_left = bcs_h[:left]
    @test bc_left isa DirichletBC
    @test isapprox(bc_left.value, Cp * (400.0 - T_ref); rtol = 1.0e-14)

    # Neumann: ∂h/∂n = Cp · ∂T/∂n
    bc_right = bcs_h[:right]
    @test bc_right isa NeumannBC
    @test isapprox(bc_right.value, Cp * 5.0; rtol = 1.0e-14)

    # Robin: substitute h = Cp·(T - T_ref) into a·T + b·∂T/∂n = c
    # ⇒ (a/Cp)·h + b·∂h/∂n = c - a·T_ref
    bc_top = bcs_h[:top]
    @test bc_top isa RobinBC
    @test isapprox(bc_top.a, 2.0 / Cp; rtol = 1.0e-14)
    @test isapprox(bc_top.b, 0.5; rtol = 1.0e-14)
    @test isapprox(bc_top.c, 830.0 - 2.0 * T_ref; rtol = 1.0e-14)
end

@testset "V&V: enthalpy — constant-Cp equivalence (pure conduction)" begin
    # Pure conduction on a square mesh with fixed T on two sides and
    # insulated on the other two. Solve the T-form and the h-form
    # separately — the recovered T field must match to machine
    # tolerance (constant-Cp equivalence).
    mesh = build_cartesian_unstructured_mesh(6, 6, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(), :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(), :top => NoSlipWallBC(),
    )
    algo = SIMPLE(; max_iterations = 1, tolerance = 1.0)  # one-shot
    prob = SteadyIncompressibleProblem(mesh, bcs, algo; nu = 1.0, density = 1.0)

    T_ref = 300.0
    Cp = 1000.0
    props_T = FluidThermalProperties{2}(;
        Cp = Cp, k = 0.5, beta = 0.0, T_ref = T_ref, use_enthalpy = false,
    )
    props_h = FluidThermalProperties{2}(;
        Cp = Cp, k = 0.5, beta = 0.0, T_ref = T_ref, use_enthalpy = true,
    )

    bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(400.0),
        :right => DirichletBC(300.0),
        :bottom => NeumannBC(0.0),
        :top => NeumannBC(0.0),
    )

    _, state_T = solve_simple_thermal(prob, props_T; bcs_T = bcs_T, T_init = 350.0)
    _, state_h = solve_simple_thermal(prob, props_h; bcs_T = bcs_T, T_init = 350.0)

    @test all(isfinite, state_T.T_field.internal)
    @test all(isfinite, state_h.T_field.internal)
    # Constant-Cp equivalence: the enthalpy form must yield the same
    # T field as the temperature form to within linear-solver tolerance.
    for c in eachindex(state_T.T_field.internal)
        @test isapprox(
            state_h.T_field.internal[c], state_T.T_field.internal[c];
            rtol = 1.0e-6, atol = 1.0e-6,
        )
    end
end

@testset "V&V: enthalpy — adiabatic closed box (no drift)" begin
    # Uniform initial field + fully insulated walls + zero velocity
    # ⇒ ∂h/∂t = 0 ⇒ enthalpy field stays flat and ∫ρh dV is conserved.
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(), :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(), :top => NoSlipWallBC(),
    )
    algo = PISO(; n_correctors = 1)
    prob = IncompressibleProblem(mesh, bcs, algo; nu = 1.0, density = 1.0)

    T_ref = 300.0
    Cp = 1000.0
    rho = 1.0
    props_h = FluidThermalProperties{2}(;
        Cp = Cp, k = 0.5, beta = 0.0, T_ref = T_ref, use_enthalpy = true,
    )
    bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NeumannBC(0.0), :right => NeumannBC(0.0),
        :bottom => NeumannBC(0.0), :top => NeumannBC(0.0),
    )

    T0 = 350.0
    _, state = solve_incompressible_thermal(
        prob, props_h, (0.0, 0.02), 0.01;
        bcs_T = bcs_T, T_init = T0,
    )

    # Every cell should still be at T0 (adiabatic, isotropic, no source).
    for c in eachindex(state.T_field.internal)
        @test isapprox(state.T_field.internal[c], T0; atol = 1.0e-8)
    end

    # Total enthalpy ∫ρh dV = ρ·Cp·(T - T_ref)·V_total is conserved.
    V_total = sum(mesh.cell_volumes)
    H_total = sum(
        rho * Cp * (state.T_field.internal[c] - T_ref) * mesh.cell_volumes[c]
            for c in eachindex(state.T_field.internal)
    )
    H_expected = rho * Cp * (T0 - T_ref) * V_total
    @test isapprox(H_total, H_expected; rtol = 1.0e-8)
end

@testset "V&V: enthalpy — dilatation-work placeholder vanishes at constant p" begin
    # The enthalpy equation in its textbook form carries a
    # ρ·Dp/Dt dilatation term that vanishes identically for
    # incompressible flow (constant density + zero p-gradient work on
    # the mean fluid). This test confirms that assembling + solving the
    # enthalpy equation in steady state with a uniform pressure field
    # produces no spurious residual contribution from that term — i.e.
    # the T-form and h-form still agree even though we have not
    # activated any explicit dilatation source.
    mesh = build_cartesian_unstructured_mesh(5, 5, 1.0, 1.0)
    bcs = Dict{Symbol, AbstractBoundaryCondition}(
        :left => NoSlipWallBC(), :right => NoSlipWallBC(),
        :bottom => NoSlipWallBC(), :top => NoSlipWallBC(),
    )
    algo = SIMPLE(; max_iterations = 1, tolerance = 1.0)
    prob = SteadyIncompressibleProblem(mesh, bcs, algo; nu = 1.0, density = 1.0)

    T_ref = 295.0
    Cp = 1100.0
    props_T = FluidThermalProperties{2}(;
        Cp = Cp, k = 0.6, T_ref = T_ref, use_enthalpy = false,
    )
    props_h = FluidThermalProperties{2}(;
        Cp = Cp, k = 0.6, T_ref = T_ref, use_enthalpy = true,
    )

    bcs_T = Dict{Symbol, AbstractBoundaryCondition}(
        :left => DirichletBC(310.0),
        :right => DirichletBC(310.0),
        :bottom => DirichletBC(310.0),
        :top => DirichletBC(330.0),
    )

    _, state_T = solve_simple_thermal(prob, props_T; bcs_T = bcs_T, T_init = 310.0)
    _, state_h = solve_simple_thermal(prob, props_h; bcs_T = bcs_T, T_init = 310.0)

    for c in eachindex(state_T.T_field.internal)
        @test isapprox(
            state_h.T_field.internal[c], state_T.T_field.internal[c];
            rtol = 1.0e-6, atol = 1.0e-6,
        )
    end
end
