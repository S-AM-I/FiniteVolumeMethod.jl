using Test
using FiniteVolumeMethod
using JET

# JET type-stability audit on FVM hot paths.
#
# Scoped via target_modules to FiniteVolumeMethod so upstream
# DelaunayTriangulation / SciMLBase / OrdinaryDiffEq noise doesn't drown
# out our findings.
#
# This test is INFORMATIONAL — it always passes but emits the count of
# @report_opt issues per hot path via @info. Promote to a gate by
# replacing each '@test true' with a numeric baseline once counts
# stabilize. See .github/workflows/jet.yml for the manual-trigger
# CI lane.

const _FVM_TARGETS = (FiniteVolumeMethod,)

@testset "JET — type-stability audit (informational)" begin
    # Smoke test: just exercise package load to surface anything that
    # erupts on import / first-call. Replace with concrete solver hot
    # paths once we know which kernels are publication-grade enough to
    # attach a regression contract to (PLAN.md Phase D-1 hot-kernel
    # ranking).
    @testset "package load" begin
        report = JET.@report_opt target_modules = _FVM_TARGETS using FiniteVolumeMethod
        n = length(JET.get_reports(report))
        @info "JET — FVM load" optimisation_warnings = n
        @test true
    end
end
