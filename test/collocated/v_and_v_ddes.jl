# test/v_and_v_ddes.jl — DDES algebraic helpers V&V (v3.78)

using FiniteVolumeMethod
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

const _ddes_shield = FiniteVolumeMethod._ddes_shielding
const _ddes_ls = FiniteVolumeMethod._ddes_length_scale

@testset "V&V: DDES — shielding f_d in [0, 1]" begin
    for nu in (1.0e-6, 1.0e-5, 1.0e-4)
        for nu_t in (0.0, 1.0e-4, 1.0e-2)
            for d in (1.0e-4, 1.0e-2, 1.0e-1)
                for S in (1.0e-2, 1.0, 100.0)
                    f_d = _ddes_shield(nu, nu_t, d, S, 0.41)
                    @test 0.0 <= f_d <= 1.0
                end
            end
        end
    end
end

@testset "V&V: DDES — shielding near wall (d → 0) ⇒ f_d → 0 (RANS mode)" begin
    # Very small wall-distance d pushes r_d to infinity; tanh((8 r_d)^3)
    # saturates at 1; so f_d = 1 - 1 = 0 (RANS mode near walls).
    f_d_small = _ddes_shield(1.0e-5, 1.0e-3, 1.0e-8, 1.0, 0.41)
    @test isapprox(f_d_small, 0.0; atol = 1.0e-10)
end

@testset "V&V: DDES — shielding far from wall (d → ∞) ⇒ f_d → 1 (LES mode)" begin
    # Large d makes r_d tiny; tanh((8 r_d)^3) ≈ 0; f_d ≈ 1.
    f_d_large = _ddes_shield(1.0e-5, 1.0e-3, 100.0, 1.0, 0.41)
    @test isapprox(f_d_large, 1.0; rtol = 1.0e-2)
end

@testset "V&V: DDES — length-scale identity l_DDES = l_RANS − f_d·max(0, l_RANS−l_LES)" begin
    # Closed-form match.
    for (l_r, l_l, f_d) in (
            (1.0, 0.5, 0.0), (1.0, 0.5, 0.5), (1.0, 0.5, 1.0),
            (0.5, 1.0, 0.3),   # l_LES > l_RANS ⇒ max(0, -0.5) = 0
            (2.0, 0.5, 0.7),
        )
        l_ddes = _ddes_ls(l_r, l_l, f_d)
        expected = l_r - f_d * max(0.0, l_r - l_l)
        @test isapprox(l_ddes, expected; rtol = 1.0e-14)
    end
end

@testset "V&V: DDES — length-scale RANS-mode reduces to l_RANS" begin
    # f_d = 0 ⇒ l_DDES = l_RANS independent of l_LES.
    for (l_r, l_l) in ((1.0, 0.5), (1.0, 2.0), (0.5, 0.5))
        @test _ddes_ls(l_r, l_l, 0.0) == l_r
    end
end

@testset "V&V: DDES — length-scale LES-mode caps at l_LES" begin
    # f_d = 1 and l_RANS > l_LES ⇒ l_DDES = l_LES (LES mode active).
    for (l_r, l_l) in ((1.0, 0.5), (2.0, 0.3), (3.0, 0.1))
        l_ddes = _ddes_ls(l_r, l_l, 1.0)
        @test isapprox(l_ddes, l_l; rtol = 1.0e-14)
    end
end

@testset "V&V: DDES — constructor + field count match base model" begin
    mesh = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    base = SpalartAllmaras(mesh, Symbol[])
    ddes = DDES(base, mesh, Symbol[:bottom])

    @test FiniteVolumeMethod.n_turbulence_fields(ddes) ==
        FiniteVolumeMethod.n_turbulence_fields(base)
    @test FiniteVolumeMethod.turbulence_field_names(ddes) ==
        FiniteVolumeMethod.turbulence_field_names(base)
end
