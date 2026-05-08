using Test
using FiniteVolumeMethod
using JET

# JET type-stability audit on FVM hot paths.
#
# Scoped via target_modules to FiniteVolumeMethod so upstream
# DelaunayTriangulation / SciMLBase / OrdinaryDiffEq noise doesn't
# drown out our findings. Replace these targets with concrete solver
# hot paths as Phase D-1's hot-kernel ranking lands.

const _FVM_TARGETS = (FiniteVolumeMethod,)

# Pinned baselines from julia 1.12.6, package version 3.111.0. Bump
# down as fixes land; treat any increase above (baseline + headroom)
# as a regression to investigate.
const _JET_BASELINES = Dict(
    "IdealGasEOS"            => 0,
    "pressure(IdealGasEOS)"  => 0,
)
const _JET_HEADROOM = 5

@testset "JET — type-stability audit" begin
    @testset "constructor — IdealGasEOS" begin
        # JET requires a function call (not a top-level statement), so
        # we report on the constructor invocation rather than `using`.
        IdealGasEOS(1.4)  # warm-up
        report = JET.@report_opt target_modules = _FVM_TARGETS IdealGasEOS(1.4)
        n = length(JET.get_reports(report))
        @info "JET — IdealGasEOS" optimisation_warnings = n baseline = _JET_BASELINES["IdealGasEOS"]
        @test n <= _JET_BASELINES["IdealGasEOS"] + _JET_HEADROOM
    end

    @testset "thermodynamic call — pressure(IdealGasEOS, ρ, ε)" begin
        eos = IdealGasEOS(1.4)
        FiniteVolumeMethod.pressure(eos, 1.0, 1.0)
        report = JET.@report_opt target_modules = _FVM_TARGETS FiniteVolumeMethod.pressure(eos, 1.0, 1.0)
        n = length(JET.get_reports(report))
        @info "JET — pressure(IdealGasEOS)" optimisation_warnings = n baseline = _JET_BASELINES["pressure(IdealGasEOS)"]
        @test n <= _JET_BASELINES["pressure(IdealGasEOS)"] + _JET_HEADROOM
    end
end
