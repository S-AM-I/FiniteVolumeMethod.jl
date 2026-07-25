# test/experimental/aeroacoustics_surface.jl — Ffowcs Williams-Hawkings and
# Curle surface integration to a far-field observer.

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: FWHObserver, FWHSurface, curle_dipole_pressure, fwh_monopole_pressure
using Test
using StaticArrays: SVector

@testset "FW-H / Curle surface integration" begin
    # Unit-radius spherical "body" with two antipodal panels, far-field
    # observer at 10 units. Curle sum should linearly depend on dp.
    faces = [1, 2]
    centers = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0)]
    normals = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0)]
    areas = [1.0, 1.0]
    surface = FWHSurface{3, Float64}(faces, centers, normals, areas)

    observer = FWHObserver(SVector(10.0, 0.0, 0.0))

    # Equal pressure on both sides: dipole cancels by symmetry.
    p_equal = [1.0e5, 1.0e5]
    @test curle_dipole_pressure(observer, surface, p_equal, 1.0e5) ≈ 0.0 atol = 1.0e-12

    # Asymmetric surface pressures: dipole non-zero.
    p_asym = [2.0e5, 1.0e5]
    p_asym_larger = [3.0e5, 1.0e5]
    p1 = curle_dipole_pressure(observer, surface, p_asym, 1.0e5)
    p2 = curle_dipole_pressure(observer, surface, p_asym_larger, 1.0e5)
    @test abs(p2) > abs(p1)   # doubling the asymmetry doubles the magnitude (linearity)

    # Monopole: uniform time-derivative of mass flux on both faces, but with
    # opposite normals, should average distances equally.
    dmass = [1.0, 1.0]
    mono = fwh_monopole_pressure(observer, surface, dmass)
    # Both faces contribute; face at (1,0,0) has r=9, face at (-1,0,0) has r=11.
    # sum = 1·1/9 + 1·1/11 = 0.2020..., divide by 4π.
    @test mono ≈ (1 / 9 + 1 / 11) / (4π) atol = 1.0e-10
end
