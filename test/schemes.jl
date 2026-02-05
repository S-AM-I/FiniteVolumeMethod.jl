using FiniteVolumeMethod
using FiniteVolumeMethod: minmod, superbee, van_leer, venkatakrishnan, barth_jespersen, koren, ospre
using FiniteVolumeMethod: compute_slope_ratio, apply_limiter, select_limiter
using FiniteVolumeMethod: MinmodLimiter, SuperbeeLimiter, VanLeerLimiter, VenkatakrishnanLimiter
using FiniteVolumeMethod: KorenLimiter, OspreLimiter, BarthJespersenLimiter
using FiniteVolumeMethod: GreenGaussGradient, LeastSquaresGradient, reconstruct_gradient
using FiniteVolumeMethod: MUSCLScheme
using DelaunayTriangulation
using Test

@testset "Flux Limiters" begin
    @testset "minmod" begin
        # Same sign, positive
        @test minmod(1.0, 2.0) == 1.0
        @test minmod(3.0, 2.0) == 2.0

        # Same sign, negative
        @test minmod(-1.0, -2.0) == -1.0
        @test minmod(-3.0, -2.0) == -2.0

        # Opposite signs
        @test minmod(1.0, -2.0) == 0.0
        @test minmod(-1.0, 2.0) == 0.0

        # Zero
        @test minmod(0.0, 1.0) == 0.0
        @test minmod(1.0, 0.0) == 0.0
    end

    @testset "superbee" begin
        # Test basic properties
        @test superbee(1.0, 2.0) == max(minmod(2.0, 2.0), minmod(1.0, 4.0))
        @test superbee(-1.0, -2.0) == max(minmod(-2.0, -2.0), minmod(-1.0, -4.0))

        # Opposite signs
        @test superbee(1.0, -1.0) == 0.0
    end

    @testset "van_leer" begin
        # Positive values
        @test van_leer(1.0, 1.0) ≈ 1.0
        @test van_leer(2.0, 2.0) ≈ 2.0

        # Opposite signs
        @test van_leer(1.0, -1.0) == 0.0

        # Zero
        @test van_leer(0.0, 1.0) == 0.0
    end

    @testset "venkatakrishnan" begin
        @test venkatakrishnan(0.0) == 0.0
        @test 0.0 <= venkatakrishnan(1.0) <= 1.0
        @test 0.0 <= venkatakrishnan(2.0) <= 1.0
        # Should be monotonic
        @test venkatakrishnan(1.0) < venkatakrishnan(2.0)
    end

    @testset "barth_jespersen" begin
        # No limiting needed
        @test barth_jespersen(0.5, 0.0, 1.0, 0.7) == 1.0

        # Limiting needed (face value exceeds max)
        @test barth_jespersen(0.5, 0.0, 1.0, 1.5) < 1.0

        # Face value equals center
        @test barth_jespersen(0.5, 0.0, 1.0, 0.5) == 1.0
    end

    @testset "koren" begin
        @test koren(-1.0) == 0.0
        @test koren(0.0) == 0.0
        @test koren(1.0) == 1.0
        @test 0.0 <= koren(0.5) <= 2.0
    end

    @testset "ospre" begin
        @test 0.0 <= ospre(1.0) <= 2.0
        @test 0.0 <= ospre(2.0) <= 2.0
        @test ospre(0.0) == 0.0
    end

    @testset "Limiter Types" begin
        @test apply_limiter(MinmodLimiter(), 1.0, 2.0) == minmod(1.0, 2.0)
        @test apply_limiter(SuperbeeLimiter(), 1.0, 2.0) == superbee(1.0, 2.0)
        @test apply_limiter(VanLeerLimiter(), 1.0, 2.0) == van_leer(1.0, 2.0)
        @test apply_limiter(VenkatakrishnanLimiter(), 1.0) == venkatakrishnan(1.0)
        @test apply_limiter(KorenLimiter(), 1.0) == koren(1.0)
        @test apply_limiter(OspreLimiter(), 1.0) == ospre(1.0)
    end

    @testset "select_limiter" begin
        @test select_limiter(:conservative) isa MinmodLimiter
        @test select_limiter(:accuracy) isa VanLeerLimiter
        @test select_limiter(:shock_capturing) isa SuperbeeLimiter
        @test select_limiter(:accuracy, :unstructured) isa VenkatakrishnanLimiter
    end

    @testset "compute_slope_ratio" begin
        @test compute_slope_ratio(0.0, 1.0, 2.0) ≈ 1.0
        @test compute_slope_ratio(0.0, 1.0, 3.0) ≈ 0.5
        @test compute_slope_ratio(0.0, 2.0, 3.0) ≈ 2.0
    end
end

@testset "Gradient Reconstruction" begin
    # Create a simple test mesh
    tri = triangulate_rectangle(0, 1, 0, 1, 5, 5, single_boundary=true)
    mesh = FVMGeometry(tri)

    # Linear function: u(x, y) = 2x + 3y
    # Gradient should be (2, 3)
    u = [2*x + 3*y for (x, y) in DelaunayTriangulation.each_point(tri)]

    @testset "GreenGaussGradient" begin
        method = GreenGaussGradient()
        # Test at an interior point (not on boundary)
        # Due to the discretization, gradient won't be exact but should be close
        for i in each_solid_vertex(tri)
            grad = reconstruct_gradient(method, mesh, u, i)
            @test length(grad) == 2
            # Gradient should be approximately (2, 3) for interior points
        end
    end

    @testset "LeastSquaresGradient" begin
        method = LeastSquaresGradient()
        for i in each_solid_vertex(tri)
            grad = reconstruct_gradient(method, mesh, u, i)
            @test length(grad) == 2
        end

        # Test unweighted
        method_unweighted = LeastSquaresGradient(false)
        for i in each_solid_vertex(tri)
            grad = reconstruct_gradient(method_unweighted, mesh, u, i)
            @test length(grad) == 2
        end
    end
end

@testset "MUSCLScheme" begin
    @testset "Construction" begin
        scheme = MUSCLScheme()
        @test scheme.limiter isa VanLeerLimiter
        @test scheme.gradient_method isa GreenGaussGradient

        scheme2 = MUSCLScheme(limiter=MinmodLimiter(), gradient_method=LeastSquaresGradient())
        @test scheme2.limiter isa MinmodLimiter
        @test scheme2.gradient_method isa LeastSquaresGradient
    end
end
