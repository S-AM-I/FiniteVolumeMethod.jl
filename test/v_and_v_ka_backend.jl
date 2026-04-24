# test/v_and_v_ka_backend.jl — CPU-path V&V for the KA-dispatched kernels.

using FiniteVolumeMethod
using Test

@testset "V&V: CPUBackend — interpolate_face_ka! matches hand-rolled loop" begin
    nf = 16
    out = zeros(nf)
    P = rand(nf)
    N = rand(nf)
    w = fill(0.5, nf)
    FiniteVolumeMethod.interpolate_face_ka!(out, P, N, w)
    expected = 0.5 .* P .+ 0.5 .* N
    for f in 1:nf
        @test out[f] ≈ expected[f] rtol = 1.0e-14
    end
end

@testset "V&V: CPUBackend — elementwise_sum_ka! closed form" begin
    n = 8
    a = collect(1.0:n)
    b = collect((2.0 * n):-1.0:(n + 1.0))
    out = zeros(n)
    FiniteVolumeMethod.elementwise_sum_ka!(out, a, b)
    for i in 1:n
        @test out[i] == a[i] + b[i]
    end
end

@testset "V&V: kernel_backend — default is CPUBackend without KA" begin
    @test FiniteVolumeMethod.kernel_backend(nothing) isa FiniteVolumeMethod.CPUBackend
end

@testset "V&V: per_term_ad — finite-difference of x^2 recovers 2x" begin
    x = [1.5]
    d = [1.0]
    grad = FiniteVolumeMethod.per_term_ad(v -> v[1]^2, x, d; epsilon = 1.0e-5)
    @test isapprox(grad, 2 * 1.5; rtol = 1.0e-6)
end
