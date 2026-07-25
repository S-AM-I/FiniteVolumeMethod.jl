using Dates
using TOML
using Test

const QUALITY_LEDGER_PATH = joinpath(@__DIR__, "..", "QUALITY_LEDGER.toml")

ledger = TOML.parsefile(QUALITY_LEDGER_PATH)
allowlists = get(ledger, "allowlists", Any[])

@testset "Quality ledger schema" begin
    @test get(ledger, "schema_version", 0) == 1
    @test length(allowlists) == 1
    for allowlist in allowlists
        @test haskey(allowlist, "id")
        @test haskey(allowlist, "kind")
        @test haskey(allowlist, "owner")
        @test haskey(allowlist, "scope")
        @test haskey(allowlist, "reason")
        @test haskey(allowlist, "expiry_review")
        @test allowlist["kind"] == "quality_exception"
        @test allowlist["scope"] == "Aqua.test_unbound_args"
        @test Date(allowlist["expiry_review"]) >= Date(now())
    end
end
