# test/v_and_v_two_fluid.jl — Eulerian two-fluid primitive V&V (v3.0 Wave 5)
#
# Primitive-type and invariant coverage for the Wave-5 two-fluid
# module:
#   * `TwoFluidProperties` keyword round-trip + validation.
#   * `TwoFluidState` constructor sets `α_l + α_g = 1` exactly.
#   * `enforce_volume_fraction_sum!` restores the closure relation
#     after perturbation.
#   * Density ratio `ρ_l / ρ_g > 1` for typical water/air defaults.
#   * `TwoFluidSolver` emits the expected one-shot experimental
#     `@warn`.
#
# Full coupled-solver verification is deferred to v3.1.

using FiniteVolumeMethod
using FiniteVolumeMethod: AbstractDragClosure, TwoFluidState, density_ratio, enforce_volume_fraction_sum!, interphase_drag
using StaticArrays
using Logging
using Test

include("TestHelpers.jl")

const _WAVE5_SRC = joinpath(@__DIR__, "..", "src", "collocated", "multiphase")
isdefined(Main, :AbstractDragClosure) || include(joinpath(_WAVE5_SRC, "drag_closures.jl"))
isdefined(Main, :TwoFluidProperties) || include(joinpath(_WAVE5_SRC, "two_fluid.jl"))

@testset "V&V: TwoFluidProperties — keyword round-trip" begin
    props = TwoFluidProperties(;
        rho_l = 998.0, rho_g = 1.2,
        mu_l = 1.0e-3, mu_g = 1.81e-5,
        sigma = 0.072, d_b = 2.0e-3, C_D = 1.0,
    )
    @test props isa TwoFluidProperties{Float64}
    @test props.rho_l == 998.0
    @test props.rho_g == 1.2
    @test props.mu_l == 1.0e-3
    @test props.mu_g == 1.81e-5
    @test props.sigma == 0.072
    @test props.d_b == 2.0e-3
    @test props.C_D == 1.0

    # Defaults
    default = TwoFluidProperties()
    @test default.rho_l == 1000.0
    @test default.rho_g == 1.225
    @test default.mu_l == 1.0e-3
    @test default.mu_g == 1.8e-5
    @test default.sigma == 0.072
    @test default.d_b == 1.0e-3
    @test default.C_D == 1.0
end

@testset "V&V: TwoFluidProperties — invariants & guards" begin
    # Density ratio ρ_l / ρ_g > 1 for water/air defaults (~816).
    props = TwoFluidProperties()
    @test density_ratio(props) > 1.0
    @test isapprox(density_ratio(props), 1000.0 / 1.225; rtol = 1.0e-12)

    # Positivity guards.
    @test_throws ArgumentError TwoFluidProperties(; rho_l = -1.0)
    @test_throws ArgumentError TwoFluidProperties(; rho_g = 0.0)
    @test_throws ArgumentError TwoFluidProperties(; mu_l = -1.0e-3)
    @test_throws ArgumentError TwoFluidProperties(; mu_g = 0.0)
    @test_throws ArgumentError TwoFluidProperties(; d_b = 0.0)
end

@testset "V&V: TwoFluidState — α_l + α_g = 1 on construction" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)

    for alpha_g_init in (0.0, 0.3, 0.5, 0.75, 1.0)
        state = TwoFluidState(mesh; alpha_g_init = alpha_g_init)
        @test all(
            isapprox.(
                state.alpha_g.internal .+ state.alpha_l.internal,
                1.0; atol = 1.0e-14,
            )
        )
        @test all(
            isapprox.(
                state.alpha_g.boundary .+ state.alpha_l.boundary,
                1.0; atol = 1.0e-14,
            )
        )
        @test all(state.alpha_g.internal .== alpha_g_init)
        @test all(state.alpha_l.internal .== 1.0 - alpha_g_init)
    end

    @test_throws ArgumentError TwoFluidState(mesh; alpha_g_init = -0.1)
    @test_throws ArgumentError TwoFluidState(mesh; alpha_g_init = 1.01)
end

@testset "V&V: TwoFluidState — enforce_volume_fraction_sum! restores invariant" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    state = TwoFluidState(mesh; alpha_g_init = 0.3)

    # Perturb α_g in several cells; α_l is now stale.
    state.alpha_g.internal[1] = 0.7
    state.alpha_g.internal[5] = 0.5
    # α_l is still 0.7 everywhere, so cells 1 and 5 now violate
    # closure by 0.4 and 0.2 respectively.
    max_dev = enforce_volume_fraction_sum!(state)
    @test max_dev >= 0.4 - 1.0e-12

    @test all(
        isapprox.(
            state.alpha_g.internal .+ state.alpha_l.internal,
            1.0; atol = 1.0e-14,
        )
    )
    @test state.alpha_l.internal[1] == 1.0 - 0.7
    @test state.alpha_l.internal[5] == 1.0 - 0.5
end

@testset "V&V: TwoFluidSolver — production path (v3.1)" begin
    # v3.0 emitted a one-shot "deferred to v3.1" @warn. v3.1 ships the
    # production-hardened solver; `warn_experimental!` is now a no-op.
    buffer = IOBuffer()
    logger = ConsoleLogger(buffer, Logging.Warn)
    with_logger(logger) do
        FiniteVolumeMethod.warn_experimental!(TwoFluidSolver())
    end
    captured = String(take!(buffer))
    @test !occursin("deferred to v3.1", captured)
end

@testset "V&V: interphase_drag wrapper — zero slip, nominal slip" begin
    props = TwoFluidProperties(;
        rho_l = 1000.0, rho_g = 1.2,
        mu_l = 1.0e-3, mu_g = 1.81e-5,
        sigma = 0.072, d_b = 1.0e-3, C_D = 1.0,
    )

    U_l = SVector(0.0, 0.0)
    U_g = SVector(0.0, 0.0)
    F_zero = interphase_drag(props, U_l, U_g, 0.1)
    @test F_zero == SVector(0.0, 0.0)

    # Finite slip: F_D = (3/4)·C_D·ρ_l·α_g·|U_rel|·U_rel/d_b.
    U_g_nz = SVector(0.01, 0.0)
    alpha_g = 0.05
    F = interphase_drag(props, U_l, U_g_nz, alpha_g)
    # Verify direction is aligned with U_rel and magnitude positive.
    @test F[1] > 0.0
    @test F[2] == 0.0

    # C_D scaling multiplier: doubling C_D in props doubles the force.
    props2 = TwoFluidProperties(;
        rho_l = 1000.0, rho_g = 1.2,
        mu_l = 1.0e-3, mu_g = 1.81e-5,
        sigma = 0.072, d_b = 1.0e-3, C_D = 2.0,
    )
    F2 = interphase_drag(props2, U_l, U_g_nz, alpha_g)
    @test isapprox(F2[1], 2.0 * F[1]; rtol = 1.0e-12)
end
