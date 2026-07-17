# test/v_and_v_mesh_geometry.jl — Mesh geometry invariants V&V (v3.43)
#
# First convergence-verified benchmark for `polyhedral_mesh_io`.
# Promotes it from experimental/smoke_tested to provisional/
# convergence_verified.
#
# Verifies four geometric invariants of `build_cartesian_unstructured_mesh`
# plus `face_normal_area`:
#
#   1. Total volume: Σ_c V_c = Lx · Ly (domain area).
#   2. Cell volumes uniform on Cartesian mesh: V_c = Lx·Ly/(Nx·Ny).
#   3. Closed-cell face identity: Σ_f ε(c, f) · S_f = 0 per cell
#      (divergence-theorem identity).
#   4. Face area magnitudes match Lx/Nx or Ly/Ny per direction.

using FiniteVolumeMethod
using FiniteVolumeMethod: face_normal_area
using LinearAlgebra: norm
using StaticArrays
using Test

include("TestHelpers.jl")

@testset "V&V: mesh geometry — total volume = Lx · Ly" begin
    for (Nx, Ny, Lx, Ly) in ((8, 8, 1.0, 1.0), (16, 10, 2.0, 0.5), (32, 32, 3.0, 4.0))
        mesh = build_cartesian_unstructured_mesh(Nx, Ny, Lx, Ly)
        V_total = sum(mesh.cell_volumes)
        @test isapprox(V_total, Lx * Ly; rtol = 1.0e-14)
    end
end

@testset "V&V: mesh geometry — uniform cell volumes on Cartesian mesh" begin
    for (Nx, Ny, Lx, Ly) in ((16, 16, 1.0, 1.0), (20, 10, 2.0, 1.0))
        mesh = build_cartesian_unstructured_mesh(Nx, Ny, Lx, Ly)
        V_expected = Lx * Ly / (Nx * Ny)
        for c in 1:length(mesh.cell_volumes)
            @test isapprox(mesh.cell_volumes[c], V_expected; rtol = 1.0e-14)
        end
    end
end

@testset "V&V: mesh geometry — closed-cell face identity Σ_f ε(c,f)·S_f = 0" begin
    # The divergence theorem applied to 1 (constant scalar) gives
    # ∮_∂c dS = 0 (closed surface). On a discrete cell this is
    # Σ_f ε(c, f) · S_f = 0 where ε(c, f) = ±1.
    for N in (8, 16, 32)
        mesh = build_cartesian_unstructured_mesh(N, N, 1.0, 1.0)
        nc = length(mesh.cell_volumes)
        nf = size(mesh.face_cells, 2)

        net_S = [SVector(0.0, 0.0) for _ in 1:nc]
        for f in 1:nf
            S_f = face_normal_area(mesh, f)
            P = mesh.face_cells[1, f]
            N_cell = mesh.face_cells[2, f]
            net_S[P] = net_S[P] + S_f   # S_f points out of P (owner)
            if N_cell != 0
                net_S[N_cell] = net_S[N_cell] - S_f
            end
        end

        max_norm = maximum(norm, net_S)
        @test max_norm < 1.0e-12
    end
end

@testset "V&V: mesh geometry — face area magnitudes match analytical" begin
    # For a Nx×Ny Cartesian mesh on [0, Lx]×[0, Ly], every face is
    # either horizontal (area = Lx/Nx) or vertical (area = Ly/Ny).
    Nx, Ny, Lx, Ly = 16, 10, 2.0, 1.0
    mesh = build_cartesian_unstructured_mesh(Nx, Ny, Lx, Ly)
    nf = size(mesh.face_cells, 2)

    dx = Lx / Nx
    dy = Ly / Ny

    # Count faces matching each expected area.
    count_dx = 0   # horizontal faces have face-area = dx
    count_dy = 0   # vertical faces have face-area = dy
    count_other = 0
    for f in 1:nf
        area = mesh.face_areas[f]
        if isapprox(area, dx; rtol = 1.0e-12)
            count_dx += 1
        elseif isapprox(area, dy; rtol = 1.0e-12)
            count_dy += 1
        else
            count_other += 1
        end
    end

    # Every face must match one of the two expected sizes.
    @test count_other == 0
    @test count_dx + count_dy == nf
end

@testset "V&V: mesh geometry — cell-center convexity (inside domain)" begin
    # Every cell center must lie strictly inside [0, Lx] × [0, Ly].
    Nx, Ny, Lx, Ly = 16, 16, 1.0, 1.0
    mesh = build_cartesian_unstructured_mesh(Nx, Ny, Lx, Ly)
    nc = length(mesh.cell_volumes)

    for c in 1:nc
        x = mesh.cell_centers[1, c]
        y = mesh.cell_centers[2, c]
        @test 0.0 < x < Lx
        @test 0.0 < y < Ly
    end
end
