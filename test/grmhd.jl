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

        # Position independence
        for (x, y) in [(0.0, 0.0), (-5.0, 3.0), (100.0, -200.0)]
            @test lapse(m, x, y) == 1.0
            @test shift(m, x, y) == SVector(0.0, 0.0)
            @test sqrt_gamma(m, x, y) == 1.0
        end
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

        # Kerr-Schild identity: alpha = 1/sqrt(1 + 2H) where H = M/r
        for r in [3.0, 5.0, 10.0, 50.0]
            H = M / r
            @test lapse(m, r, 0.0) ≈ 1 / sqrt(1 + 2 * H) atol = 1.0e-14
            @test sqrt_gamma(m, r, 0.0) ≈ sqrt(1 + 2 * H) atol = 1.0e-14
        end

        # Metric determinant consistency: det(gamma_ij) = (sqrt_gamma)^2
        for (x, y) in [(3.0, 0.0), (4.0, 3.0), (0.0, 5.0)]
            g = spatial_metric(m, x, y)
            sg = sqrt_gamma(m, x, y)
            @test det(g) ≈ sg^2 atol = 1.0e-12
        end

        # Inverse metric: gamma^ij * gamma_jk = delta_ik
        for (x, y) in [(3.0, 0.0), (4.0, 3.0), (0.0, 5.0), (7.0, -2.0)]
            g = spatial_metric(m, x, y)
            gi = inv_spatial_metric(m, x, y)
            prod = g * gi
            @test prod[1, 1] ≈ 1.0 atol = 1.0e-12
            @test prod[2, 2] ≈ 1.0 atol = 1.0e-12
            @test prod[1, 2] ≈ 0.0 atol = 1.0e-12
            @test prod[2, 1] ≈ 0.0 atol = 1.0e-12
        end

        # Shift vector direction: should be radial and point inward
        for (x, y) in [(5.0, 0.0), (0.0, 5.0), (3.0, 4.0)]
            β = shift(m, x, y)
            r = sqrt(x^2 + y^2)
            # Shift should be parallel to radial direction
            lx, ly = x / r, y / r
            @test β[1] / norm(β) ≈ lx atol = 1.0e-12
            @test β[2] / norm(β) ≈ ly atol = 1.0e-12
        end

        # r_min floor test
        m_floor = SchwarzschildMetric(1.0; r_min = 2.0)
        # At origin, should use r_min
        α_origin = lapse(m_floor, 0.0, 0.0)
        @test isfinite(α_origin)
        @test α_origin > 0.0
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
            gi_k = inv_spatial_metric(m_kerr0, x, y)
            gi_s = inv_spatial_metric(m_sch, x, y)
            @test gi_k ≈ gi_s atol = 1.0e-10
            @test sqrt_gamma(m_kerr0, x, y) ≈ sqrt_gamma(m_sch, x, y) atol = 1.0e-10
        end

        # Inverse metric consistency for Kerr
        for (x, y) in [(5.0, 0.0), (3.0, 4.0), (0.0, 6.0)]
            g = spatial_metric(m, x, y)
            gi = inv_spatial_metric(m, x, y)
            prod = g * gi
            @test prod[1, 1] ≈ 1.0 atol = 1.0e-12
            @test prod[2, 2] ≈ 1.0 atol = 1.0e-12
            @test prod[1, 2] ≈ 0.0 atol = 1.0e-12
        end

        # Determinant consistency
        for (x, y) in [(5.0, 0.0), (3.0, 4.0), (0.0, 6.0)]
            g = spatial_metric(m, x, y)
            sg = sqrt_gamma(m, x, y)
            @test det(g) ≈ sg^2 atol = 1.0e-12
        end

        # Frame dragging: off-diagonal shift should be nonzero for a ≠ 0
        β_kerr = shift(m, 5.0, 0.0)
        @test norm(β_kerr) > 0.0
        # For a > 0, shift should have y-component at (x, 0)
        # (frame dragging in the phi direction)
        β_at_x = shift(m, 5.0, 0.0)
        @test abs(β_at_x[2]) > 0.0 || abs(β_at_x[1]) > 0.0

        # Different spin values
        for a_val in [0.0, 0.1, 0.3, 0.5, 0.9, 0.99]
            m_a = KerrMetric(M, a_val)
            α = lapse(m_a, 10.0, 0.0)
            @test 0.0 < α < 1.0
            @test isfinite(α)
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

        # Schwarzschild metric data consistency
        m_sch = SchwarzschildMetric(1.0)
        mesh_sch = StructuredMesh2D(3.0, 10.0, -3.0, 3.0, 8, 8)
        md_sch = FiniteVolumeMethod.precompute_metric(m_sch, mesh_sch)
        # All lapse values should be in (0, 1) for r > 2M
        @test all(0 .< md_sch.alpha .< 1)
        @test all(md_sch.sqrtg .> 1.0)
        # gamma_xx >= 1 (spatial metric enhanced by gravity)
        @test all(md_sch.gamma_xx .>= 1.0)
        @test all(md_sch.gamma_yy .>= 1.0)
    end

    @testset "MetricData2D face precomputation" begin
        m_sch = SchwarzschildMetric(1.0)
        mesh = StructuredMesh2D(3.0, 10.0, -3.0, 3.0, 10, 10)
        face_data = FiniteVolumeMethod.precompute_metric_at_faces(m_sch, mesh)
        alpha_xf, alpha_yf, betax_xf, betay_xf, betax_yf, betay_yf, sqrtg_xf, sqrtg_yf = face_data
        # x-faces: (nx+1) x ny
        @test size(alpha_xf) == (11, 10)
        # y-faces: nx x (ny+1)
        @test size(alpha_yf) == (10, 11)
        # All alpha values should be positive and finite
        @test all(isfinite, alpha_xf)
        @test all(isfinite, alpha_yf)
        @test all(alpha_xf .> 0)
        @test all(alpha_yf .> 0)
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
    @test law.con2prim_tol == 1.0e-12
    @test law.con2prim_maxiter == 50

    # Custom con2prim params
    law2 = GRMHDEquations{2}(eos, metric; con2prim_tol = 1.0e-10, con2prim_maxiter = 100)
    @test law2.con2prim_tol == 1.0e-10
    @test law2.con2prim_maxiter == 100

    # Different metrics
    for m in [MinkowskiMetric{2}(), SchwarzschildMetric(1.0), KerrMetric(1.0, 0.5)]
        law_m = GRMHDEquations{2}(eos, m)
        @test nvariables(law_m) == 8
        @test law_m.metric === m
    end
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

    @testset "Physical flux matches SRMHD in Minkowski" begin
        states = [
            SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0),
            SVector(1.0, 0.3, 0.2, 0.0, 2.0, 0.5, 0.8, 0.3),
            SVector(0.5, -0.4, 0.1, 0.2, 3.0, 1.0, -0.5, 0.5),
        ]
        for w in states
            for dir in [1, 2]
                f_gr = physical_flux(law_gr, w, dir)
                f_sr = physical_flux(law_sr, w, dir)
                @test f_gr ≈ f_sr atol = 1.0e-12
            end
        end
    end

    @testset "Wave speeds match SRMHD in Minkowski" begin
        states = [
            SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0),
            SVector(1.0, 0.3, 0.2, 0.0, 2.0, 0.5, 0.8, 0.3),
            SVector(0.5, -0.4, 0.1, 0.2, 3.0, 1.0, -0.5, 0.5),
        ]
        for w in states
            for dir in [1, 2]
                λm_gr, λp_gr = wave_speeds(law_gr, w, dir)
                λm_sr, λp_sr = wave_speeds(law_sr, w, dir)
                @test λm_gr ≈ λm_sr atol = 1.0e-12
                @test λp_gr ≈ λp_sr atol = 1.0e-12
            end
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

    @testset "Source terms vanish for all Minkowski states" begin
        mesh_src = StructuredMesh2D(0.0, 1.0, 0.0, 1.0, 6, 6)
        md_src = FiniteVolumeMethod.precompute_metric(metric, mesh_src)
        states = [
            SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            SVector(2.0, 0.5, 0.3, 0.0, 5.0, 1.0, 2.0, 0.0),
            SVector(0.1, -0.2, 0.1, 0.3, 0.5, 0.5, -0.3, 0.2),
        ]
        for w in states
            u = primitive_to_conserved(law_gr, w)
            for ix in 2:5, iy in 2:5
                src = FiniteVolumeMethod.grmhd_source_terms(law_gr, w, u, md_src, mesh_src, ix, iy)
                @test all(abs.(src) .< 1.0e-13)
            end
        end
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

    @testset "Schwarzschild densitized con2prim roundtrip" begin
        metric = SchwarzschildMetric(1.0)
        law = GRMHDEquations{2}(eos, metric)
        states = [
            (SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0), 5.0, 0.0),
            (SVector(1.0, 0.1, 0.0, 0.0, 2.0, 0.3, 0.5, 0.0), 4.0, 3.0),
            (SVector(0.5, 0.0, 0.1, 0.0, 0.5, 0.0, 0.2, 0.0), 0.0, 6.0),
        ]
        for (w, x, y) in states
            u_d = FiniteVolumeMethod.grmhd_primitive_to_conserved_densitized(law, w, x, y)
            w2, result = FiniteVolumeMethod.grmhd_con2prim(law, u_d, x, y)
            @test result.converged
            @test w2[1] ≈ w[1] rtol = 1.0e-6
            @test w2[5] ≈ w[5] rtol = 1.0e-6
        end
    end

    @testset "Kerr con2prim roundtrip" begin
        metric = KerrMetric(1.0, 0.5)
        law = GRMHDEquations{2}(eos, metric)
        states = [
            (SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0), 5.0, 0.0),
            (SVector(1.0, 0.1, 0.05, 0.0, 2.0, 0.2, 0.5, 0.0), 4.0, 3.0),
            (SVector(0.5, 0.0, 0.0, 0.0, 0.5, 0.1, 0.1, 0.0), 0.0, 6.0),
        ]
        for (w, x, y) in states
            u_d = FiniteVolumeMethod.grmhd_primitive_to_conserved_densitized(law, w, x, y)
            w2, result = FiniteVolumeMethod.grmhd_con2prim(law, u_d, x, y)
            @test result.converged
            @test w2[1] ≈ w[1] rtol = 1.0e-5
            @test w2[5] ≈ w[5] rtol = 1.0e-5
        end
    end

    @testset "Cached con2prim consistency" begin
        metric = SchwarzschildMetric(1.0)
        law = GRMHDEquations{2}(eos, metric)
        x, y = 5.0, 0.0
        w = SVector(1.0, 0.1, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0)
        u_d = FiniteVolumeMethod.grmhd_primitive_to_conserved_densitized(law, w, x, y)

        # Full con2prim
        w_full, res_full = FiniteVolumeMethod.grmhd_con2prim(law, u_d, x, y)

        # Cached con2prim
        sg = sqrt_gamma(metric, x, y)
        gi = inv_spatial_metric(metric, x, y)
        gm = spatial_metric(metric, x, y)
        w_cached, res_cached = FiniteVolumeMethod.grmhd_con2prim_cached(
            law, u_d, sg, gi[1, 1], gi[1, 2], gi[2, 2], gm[1, 1], gm[1, 2], gm[2, 2]
        )

        @test w_full ≈ w_cached atol = 1.0e-12
        @test res_full.converged == res_cached.converged
    end

    @testset "Static states (v=0) in curved spacetime" begin
        for metric in [SchwarzschildMetric(1.0), KerrMetric(1.0, 0.3)]
            law = GRMHDEquations{2}(eos, metric)
            states = [
                (SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0), 5.0, 0.0),
                (SVector(0.5, 0.0, 0.0, 0.0, 2.0, 0.0, 0.3, 0.0), 0.0, 5.0),
                (SVector(2.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0), 4.0, 3.0),
            ]
            for (w, x, y) in states
                u_d = FiniteVolumeMethod.grmhd_primitive_to_conserved_densitized(law, w, x, y)
                w2, result = FiniteVolumeMethod.grmhd_con2prim(law, u_d, x, y)
                @test result.converged
                @test w2[1] ≈ w[1] rtol = 1.0e-8
                @test abs(w2[2]) < 1.0e-8  # velocity should be zero
                @test abs(w2[3]) < 1.0e-8
                @test w2[5] ≈ w[5] rtol = 1.0e-6
            end
        end
    end

    @testset "Random states in Schwarzschild" begin
        metric = SchwarzschildMetric(1.0)
        law = GRMHDEquations{2}(eos, metric)
        for _ in 1:50
            ρ = 0.1 + 9.9 * rand()
            v_mag = 0.3 * rand()  # modest velocity
            θ = 2π * rand()
            vx = v_mag * cos(θ)
            vy = v_mag * sin(θ)
            P = 0.01 + 9.99 * rand()
            Bx = rand() - 0.5
            By = rand() - 0.5
            w = SVector(ρ, vx, vy, 0.0, P, Bx, By, 0.0)
            x = 4.0 + 6.0 * rand()
            y = -3.0 + 6.0 * rand()
            u_d = FiniteVolumeMethod.grmhd_primitive_to_conserved_densitized(law, w, x, y)
            w2, result = FiniteVolumeMethod.grmhd_con2prim(law, u_d, x, y)
            @test result.converged
            @test w2[1] ≈ w[1] rtol = 1.0e-6
            @test w2[5] ≈ w[5] rtol = 1.0e-6
        end
    end
end

# ============================================================
# Physical Flux Tests
# ============================================================
@testset "GRMHD Physical Flux" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = MinkowskiMetric{2}()
    law = GRMHDEquations{2}(eos, metric)

    @testset "Normal B-field flux = 0" begin
        w = SVector(1.0, 0.3, 0.2, 0.0, 2.0, 0.5, 0.8, 0.3)
        f1 = physical_flux(law, w, 1)
        @test f1[6] ≈ 0.0 atol = 1.0e-14  # Bx flux = 0 for x-direction
        f2 = physical_flux(law, w, 2)
        @test f2[7] ≈ 0.0 atol = 1.0e-14  # By flux = 0 for y-direction
    end

    @testset "B=0 reduces to SR hydro flux" begin
        w = SVector(1.0, 0.3, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0)
        f = physical_flux(law, w, 1)
        γ_eos = eos.gamma
        ρ, vx = w[1], w[2]
        P = w[5]
        W = 1 / sqrt(1 - vx^2)
        ε = P / ((γ_eos - 1) * ρ)
        h = 1 + ε + P / ρ
        D = ρ * W
        @test f[1] ≈ D * vx atol = 1.0e-14
    end

    @testset "Static state has zero mass flux" begin
        w = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0)
        f1 = physical_flux(law, w, 1)
        f2 = physical_flux(law, w, 2)
        @test f1[1] ≈ 0.0 atol = 1.0e-14  # D * vx = 0
        @test f2[1] ≈ 0.0 atol = 1.0e-14  # D * vy = 0
    end
end

# ============================================================
# Wave Speed Tests
# ============================================================
@testset "GRMHD Wave Speeds" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)
    metric = MinkowskiMetric{2}()
    law = GRMHDEquations{2}(eos, metric)

    @testset "Symmetric for v=0" begin
        w = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0)
        λm, λp = wave_speeds(law, w, 1)
        @test λp > 0
        @test λm < 0
        @test λp ≈ -λm atol = 1.0e-14
    end

    @testset "Bounded by speed of light" begin
        states = [
            SVector(1.0, 0.9, 0.0, 0.0, 100.0, 10.0, 10.0, 0.0),
            SVector(0.01, 0.0, 0.0, 0.0, 0.001, 5.0, 5.0, 0.0),
            SVector(10.0, 0.5, 0.3, 0.0, 50.0, 2.0, 3.0, 0.0),
        ]
        for w in states
            for dir in [1, 2]
                λm, λp = wave_speeds(law, w, dir)
                @test abs(λm) < 1.0
                @test abs(λp) < 1.0
            end
        end
    end

    @testset "max_wave_speed consistency" begin
        w = SVector(1.0, 0.3, 0.2, 0.0, 2.0, 0.5, 0.8, 0.3)
        for dir in [1, 2]
            λmax = max_wave_speed(law, w, dir)
            λm, λp = wave_speeds(law, w, dir)
            @test λmax ≈ max(abs(λm), abs(λp))
        end
    end

    @testset "Metric-corrected wave speeds" begin
        metric_sch = SchwarzschildMetric(1.0)
        law_sch = GRMHDEquations{2}(eos, metric_sch)
        w = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0)

        # At large r, metric correction should be small
        alpha_far = lapse(metric_sch, 100.0, 0.0)
        beta_far = shift(metric_sch, 100.0, 0.0)
        lam_coord = FiniteVolumeMethod.grmhd_max_wave_speed_coord(law_sch, w, 1, alpha_far, beta_far[1])
        lam_flat = max_wave_speed(law_sch, w, 1)
        @test lam_coord ≈ lam_flat atol = 0.05

        # Closer to BH, correction should be larger
        alpha_near = lapse(metric_sch, 3.0, 0.0)
        beta_near = shift(metric_sch, 3.0, 0.0)
        lam_coord_near = FiniteVolumeMethod.grmhd_max_wave_speed_coord(law_sch, w, 1, alpha_near, beta_near[1])
        @test lam_coord_near != lam_flat
    end
end

# ============================================================
# Geometric Source Terms
# ============================================================
@testset "GRMHD Source Terms" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)

    @testset "Source terms nonzero in Schwarzschild" begin
        metric = SchwarzschildMetric(1.0)
        law = GRMHDEquations{2}(eos, metric)
        mesh = StructuredMesh2D(3.0, 10.0, -3.0, 3.0, 8, 8)
        md = FiniteVolumeMethod.precompute_metric(metric, mesh)

        w = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0)
        u = primitive_to_conserved(law, w)

        # At interior cell (4, 4), source should be nonzero
        src = FiniteVolumeMethod.grmhd_source_terms(law, w, u, md, mesh, 4, 4)
        # D source should always be zero
        @test src[1] ≈ 0.0 atol = 1.0e-14
        # B sources should be zero
        @test src[6] ≈ 0.0 atol = 1.0e-14
        @test src[7] ≈ 0.0 atol = 1.0e-14
        @test src[8] ≈ 0.0 atol = 1.0e-14
        # Momentum/energy sources should be nonzero (gravity!)
        # At least some of them must be nonzero
        momentum_src = abs(src[2]) + abs(src[3]) + abs(src[5])
        @test momentum_src > 1.0e-10
    end

    @testset "Source terms structure" begin
        metric = SchwarzschildMetric(1.0)
        law = GRMHDEquations{2}(eos, metric)
        mesh = StructuredMesh2D(3.0, 10.0, -3.0, 3.0, 10, 10)
        md = FiniteVolumeMethod.precompute_metric(metric, mesh)

        w = SVector(1.0, 0.1, 0.0, 0.0, 1.0, 0.3, 0.2, 0.0)
        u = primitive_to_conserved(law, w)

        for ix in 2:9, iy in 2:9
            src = FiniteVolumeMethod.grmhd_source_terms(law, w, u, md, mesh, ix, iy)
            # D and B sources always zero
            @test src[1] ≈ 0.0 atol = 1.0e-14
            @test src[6] ≈ 0.0 atol = 1.0e-14
            @test src[7] ≈ 0.0 atol = 1.0e-14
            @test src[8] ≈ 0.0 atol = 1.0e-14
            # All components finite
            @test all(isfinite, src)
        end
    end

    @testset "Source terms decrease with distance" begin
        metric = SchwarzschildMetric(1.0)
        law = GRMHDEquations{2}(eos, metric)
        # Use a large mesh far from BH
        mesh_near = StructuredMesh2D(3.0, 6.0, -1.0, 1.0, 10, 4)
        mesh_far = StructuredMesh2D(50.0, 53.0, -1.0, 1.0, 10, 4)
        md_near = FiniteVolumeMethod.precompute_metric(metric, mesh_near)
        md_far = FiniteVolumeMethod.precompute_metric(metric, mesh_far)

        w = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0)
        u = primitive_to_conserved(law, w)

        src_near = FiniteVolumeMethod.grmhd_source_terms(law, w, u, md_near, mesh_near, 5, 2)
        src_far = FiniteVolumeMethod.grmhd_source_terms(law, w, u, md_far, mesh_far, 5, 2)

        # Gravitational source should be stronger near the BH
        @test norm(src_near) > norm(src_far)
    end
end

# ============================================================
# Densitized Prim2Con
# ============================================================
@testset "Densitized Prim2Con" begin
    eos = IdealGasEOS(gamma = 5.0 / 3.0)

    @testset "Minkowski: densitized = undensitized" begin
        metric = MinkowskiMetric{2}()
        law = GRMHDEquations{2}(eos, metric)
        w = SVector(1.0, 0.3, 0.2, 0.0, 2.0, 0.5, 0.8, 0.3)
        u_und = primitive_to_conserved(law, w)
        u_den = FiniteVolumeMethod.grmhd_primitive_to_conserved_densitized(law, w, 0.5, 0.5)
        # In Minkowski, sqrt(gamma) = 1, so densitized = undensitized
        @test u_und ≈ u_den atol = 1.0e-12
    end

    @testset "Schwarzschild: densitized includes sqrt(gamma)" begin
        metric = SchwarzschildMetric(1.0)
        law = GRMHDEquations{2}(eos, metric)
        w = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0)
        x, y = 5.0, 0.0
        u_den = FiniteVolumeMethod.grmhd_primitive_to_conserved_densitized(law, w, x, y)
        sg = sqrt_gamma(metric, x, y)
        # D component should be sqrt(gamma) * rho * W
        @test u_den[1] ≈ sg * w[1] atol = 1.0e-12  # v=0, so W=1
    end

    @testset "Cached prim2con consistency" begin
        metric = SchwarzschildMetric(1.0)
        law = GRMHDEquations{2}(eos, metric)
        w = SVector(1.0, 0.1, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0)
        x, y = 5.0, 0.0
        u_full = FiniteVolumeMethod.grmhd_primitive_to_conserved_densitized(law, w, x, y)
        sg = sqrt_gamma(metric, x, y)
        gm = spatial_metric(metric, x, y)
        u_cached = FiniteVolumeMethod.grmhd_prim2con_densitized_cached(
            law, w, sg, gm[1, 1], gm[1, 2], gm[2, 2]
        )
        @test u_full ≈ u_cached atol = 1.0e-14
    end
end
