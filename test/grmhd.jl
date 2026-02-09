using FiniteVolumeMethod
using Test
using StaticArrays
using LinearAlgebra

# ============================================================
# Metric Tests
# ============================================================
@testset "Metrics" begin
    @testset "MinkowskiMetric" begin
        m = MinkowskiMetric{2}()
        @test lapse(m, 1.0, 2.0) == 1.0
        @test shift(m, 1.0, 2.0) == SVector(0.0, 0.0)
        @test spatial_metric(m, 1.0, 2.0) == StaticArrays.SMatrix{2, 2}(1.0, 0.0, 0.0, 1.0)
        @test sqrt_gamma(m, 1.0, 2.0) == 1.0
        @test inv_spatial_metric(m, 1.0, 2.0) == StaticArrays.SMatrix{2, 2}(1.0, 0.0, 0.0, 1.0)
    end

    @testset "SchwarzschildMetric" begin
        M = 1.0
        m = SchwarzschildMetric(M)
        # Far from BH: should approach Minkowski
        x_far, y_far = 100.0, 0.0
        @test lapse(m, x_far, y_far) ≈ 1.0 atol = 0.02
        β = shift(m, x_far, y_far)
        @test norm(β) < 0.02
        @test sqrt_gamma(m, x_far, y_far) ≈ 1.0 atol = 0.02

        # At r=2M (horizon): lapse should be < 1
        x_hor, y_hor = 2.0, 0.0
        α = lapse(m, x_hor, y_hor)
        @test α < 1.0
        @test α > 0.0

        # Symmetry: metric at (r,0) should equal metric at (-r,0) etc.
        g1 = spatial_metric(m, 3.0, 0.0)
        g2 = spatial_metric(m, -3.0, 0.0)
        @test g1[1, 1] ≈ g2[1, 1]
        @test g1[2, 2] ≈ g2[2, 2]
    end

    @testset "KerrMetric" begin
        M = 1.0
        a = 0.5
        m = KerrMetric(M, a)
        # Far from BH
        @test lapse(m, 100.0, 0.0) ≈ 1.0 atol = 0.02
        β = shift(m, 100.0, 0.0)
        @test norm(β) < 0.02

        # a=0 should match Schwarzschild
        m_kerr0 = KerrMetric(M, 0.0)
        m_sch = SchwarzschildMetric(M)
        for (x, y) in [(5.0, 0.0), (3.0, 4.0), (0.0, 6.0)]
            @test lapse(m_kerr0, x, y) ≈ lapse(m_sch, x, y) atol = 1.0e-10
        end
    end

    @testset "MetricData2D" begin
        m = MinkowskiMetric{2}()
        mesh = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 10, 10)
        md = FiniteVolumeMethod.precompute_metric(m, mesh)
        @test all(md.alpha .≈ 1.0)
        @test all(md.beta_x .≈ 0.0)
        @test all(md.beta_y .≈ 0.0)
        @test all(md.gamma_xx .≈ 1.0)
        @test all(md.gamma_yy .≈ 1.0)
        @test all(md.sqrtg .≈ 1.0)
    end
end

# ============================================================
# GRMHDEquations Type Tests
# ============================================================
@testset "GRMHDEquations Type" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = MinkowskiMetric{2}()

    law = GRMHDEquations{2}(eos, metric)
    @test nvariables(law) == 8
    @test law.eos === eos
    @test law.metric === metric
end

# ============================================================
# Minkowski Limit: GRMHD should match SRMHD
# ============================================================
@testset "Minkowski Limit" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = MinkowskiMetric{2}()
    law_gr = GRMHDEquations{2}(eos, metric)
    law_sr = SRMHDEquations{2}(eos)

    @testset "Primitive↔Conserved roundtrip" begin
        states = [
            SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0),
            SVector(1.0, 0.3, 0.2, 0.0, 2.0, 0.5, 0.8, 0.3),
            SVector(0.5, -0.4, 0.1, 0.2, 3.0, 1.0, -0.5, 0.5),
        ]
        for w in states
            u_gr = primitive_to_conserved(law_gr, w)
            u_sr = primitive_to_conserved(law_sr, w)
            # In Minkowski with √γ=1, GRMHD conserved = SRMHD conserved
            @test u_gr ≈ u_sr atol = 1.0e-12

            w_gr = conserved_to_primitive(law_gr, u_gr)
            @test w_gr ≈ w atol = 1.0e-9
        end
    end

    @testset "Source terms vanish in Minkowski" begin
        w = SVector(1.0, 0.3, 0.2, 0.0, 2.0, 0.5, 0.8, 0.3)
        u = primitive_to_conserved(law_gr, w)
        # Source terms should be zero in flat spacetime (all metric derivatives vanish)
        mesh_src = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 4, 4)
        md_src = FiniteVolumeMethod.precompute_metric(metric, mesh_src)
        src = FiniteVolumeMethod.grmhd_source_terms(law_gr, w, u, md_src, mesh_src, 2, 2)
        @test all(abs.(src) .< 1.0e-14)
    end
end

# ============================================================
# Con2Prim with Metric
# ============================================================
@testset "GRMHD Con2Prim" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)

    @testset "Minkowski con2prim matches SRMHD" begin
        metric = MinkowskiMetric{2}()
        law = GRMHDEquations{2}(eos, metric)
        w = SVector(1.0, 0.3, 0.2, 0.0, 2.0, 0.5, 0.8, 0.3)
        u = primitive_to_conserved(law, w)
        w2 = conserved_to_primitive(law, u)
        @test w2 ≈ w atol = 1.0e-9
    end

    @testset "Schwarzschild con2prim roundtrip" begin
        metric = SchwarzschildMetric(1.0)
        law = GRMHDEquations{2}(eos, metric)
        # State far from BH
        w = SVector(1.0, 0.1, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0)
        u = primitive_to_conserved(law, w)
        w2 = conserved_to_primitive(law, u)
        @test w2[1] ≈ w[1] rtol = 1.0e-6
        @test w2[5] ≈ w[5] rtol = 1.0e-6
    end
end
