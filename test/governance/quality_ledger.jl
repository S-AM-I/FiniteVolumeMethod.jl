using Dates
using TOML
using Test

const QUALITY_LEDGER_PATH = joinpath(@__DIR__, "..", "QUALITY_LEDGER.toml")

ledger = TOML.parsefile(QUALITY_LEDGER_PATH)
allowlists = get(ledger, "allowlists", Any[])

@testset "Quality ledger schema" begin
    @test get(ledger, "schema_version", 0) == 1
    # The Aqua.test_unbound_args exception was retired 2026-08-07 (the
    # NTuple{Dim, T} BC constructors now bind all parameters); the ledger
    # is expected to stay empty until a new exception is justified.
    @test length(allowlists) == 0
    for allowlist in allowlists
        @test haskey(allowlist, "id")
        @test haskey(allowlist, "kind")
        @test haskey(allowlist, "owner")
        @test haskey(allowlist, "scope")
        @test haskey(allowlist, "reason")
        @test haskey(allowlist, "expiry_review")
        @test allowlist["kind"] == "quality_exception"
        @test Date(allowlist["expiry_review"]) >= Date(now())
    end
end
