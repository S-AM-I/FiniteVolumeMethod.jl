# test/v_and_v_primary_breakup_fsi.jl — Primary-breakup ↔ ALE-FSI
# handshake V&V (v3.1 / Agent D)
#
# Invariants for `couple_primary_breakup_fsi!`:
#
#   - Empty tracker + sub-critical slip ⇒ no particles injected, zero
#     mass source, empty trigger list.
#   - High slip + small drop diameter ⇒ KH-ACT trigger fires; particle
#     count increases, mass source is populated.
#   - Injected particle position matches the face centre; velocity
#     matches the reconstructed interface velocity
#     (no "teleport to origin with zero velocity" regression).
#   - Total mass released = n_injected · m_per_drop (exact conservation
#     at handshake level).
#   - Only faces tagged in `interface_patches` are considered
#     (selectivity invariant).
#   - LISA model dispatch works end-to-end.

using FiniteVolumeMethod
using LinearAlgebra: norm
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "TestHelpers.jl"))

# If the main-thread layer files do not yet include the FSI coupling
# module, load it directly into the FiniteVolumeMethod module so the
# symbol `couple_primary_breakup_fsi!` is visible. Safe no-op once the
# main thread wires in the include.
if !isdefined(FiniteVolumeMethod, :couple_primary_breakup_fsi!)
    fsi_path = joinpath(
        dirname(pathof(FiniteVolumeMethod)),
        "lagrangian", "primary_breakup_fsi.jl",
    )
    @eval FiniteVolumeMethod Base.include($FiniteVolumeMethod, $fsi_path)
end

const _couple = FiniteVolumeMethod.couple_primary_breakup_fsi!
const _ParticleTracker = FiniteVolumeMethod.ParticleTracker
const _MeshMotionState = FiniteVolumeMethod.MeshMotionState

# Helper: tag a strip of interior-bottom faces as the liquid interface.
# The Cartesian mesh helper tags its boundary faces as :left, :right,
# :bottom, :top and internal faces as :internal; we build a fresh mesh
# with :interface tags on the bottom row for these tests.
function build_interface_tagged_mesh(nx::Int, ny::Int, Lx::Float64, Ly::Float64)
    m = build_cartesian_unstructured_mesh(nx, ny, Lx, Ly)
    # Retag every :bottom boundary face as :interface so the FSI walker
    # iterates them.
    new_tags = copy(m.face_tags)
    for f in 1:length(new_tags)
        if new_tags[f] === :bottom
            new_tags[f] = :interface
        end
    end
    return FiniteVolumeMethod.UnstructuredFVMMesh{2, Float64}(
        m.cell_centers, m.cell_volumes, m.face_cells,
        m.face_centers, m.face_areas, m.face_normals,
        new_tags, m.face_velocity, m.cell_faces,
    )
end

# Count of interface faces on a tagged mesh.
function n_interface_faces(mesh)
    return count(t -> t === :interface, mesh.face_tags)
end

@testset "V&V FSI — no trigger when tracker empty and slip is zero" begin
    mesh = build_interface_tagged_mesh(4, 4, 1.0, 1.0)
    tracker = _ParticleTracker{2, Float64}()
    ale = _MeshMotionState(mesh)
    # Leave phi_mesh at zero — no slip velocity anywhere.
    breakup = FiniteVolumeMethod.KHACTBreakup()

    res = _couple(
        tracker, breakup, ale, mesh,
        1.2, 0.072, 1.0e-3;  # rho_f, sigma, dt
        interface_patches = Symbol[:interface],
    )

    @test res.n_injected == 0
    @test length(tracker.particles) == 0
    @test isempty(res.triggered_faces)
    @test all(iszero, res.mass_source)
    @test res.total_mass_released == 0.0
end

@testset "V&V FSI — sub-critical slip (below U_crit) does not trigger" begin
    mesh = build_interface_tagged_mesh(4, 4, 1.0, 1.0)
    tracker = _ParticleTracker{2, Float64}()
    ale = _MeshMotionState(mesh)
    # Set a very small slip on every interface face — below U_crit.
    for f in 1:length(mesh.face_tags)
        if mesh.face_tags[f] === :interface
            A = mesh.face_areas[f]
            ale.phi_mesh[f] = 0.01 * A  # 1 cm/s slip
        end
    end
    breakup = FiniteVolumeMethod.KHACTBreakup()

    res = _couple(
        tracker, breakup, ale, mesh,
        1.2, 0.072, 1.0e-3;
        interface_patches = Symbol[:interface],
        U_crit = 1.0,  # require ≥ 1 m/s
    )
    @test res.n_injected == 0
    @test res.total_mass_released == 0.0
end

@testset "V&V FSI — high slip + small drop ⇒ KH-ACT triggers ⇒ particles seeded" begin
    mesh = build_interface_tagged_mesh(4, 4, 1.0, 1.0)
    tracker = _ParticleTracker{2, Float64}()
    ale = _MeshMotionState(mesh)
    # Large slip on every interface face.
    U_slip = 150.0  # 150 m/s — deep in atomisation regime
    for f in 1:length(mesh.face_tags)
        if mesh.face_tags[f] === :interface
            A = mesh.face_areas[f]
            # phi_mesh = u_n · A, where u_n is face-normal velocity
            # pointing along n (the outward boundary normal).
            ale.phi_mesh[f] = U_slip * A
        end
    end
    breakup = FiniteVolumeMethod.KHACTBreakup()
    rho_l = 1000.0
    d_parent = 1.0e-4
    dt = 1.0  # huge dt ⇒ τ_b < dt always

    res = _couple(
        tracker, breakup, ale, mesh,
        1.2, 0.072, dt;
        interface_patches = Symbol[:interface],
        rho_l = rho_l,
        d_parent = d_parent,
        U_crit = 0.0,
    )

    nface = n_interface_faces(mesh)
    @test nface == 4  # the bottom row has nx=4 boundary faces
    @test res.n_injected == nface
    @test length(tracker.particles) == nface
    @test length(res.triggered_faces) == nface
    @test res.total_mass_released > 0.0
    # Exact conservation at handshake level.
    m_drop = rho_l * (pi / 6) * d_parent^3
    @test isapprox(res.total_mass_released, nface * m_drop; rtol = 1.0e-14)
    @test isapprox(sum(res.mass_source), res.total_mass_released; rtol = 1.0e-14)
end

@testset "V&V FSI — injected particle velocity matches interface flow (no teleport)" begin
    # After trigger, particles must carry the face-normal velocity used
    # to evaluate the criterion — they are not reset to zero.
    mesh = build_interface_tagged_mesh(4, 4, 1.0, 1.0)
    tracker = _ParticleTracker{2, Float64}()
    ale = _MeshMotionState(mesh)
    U_slip = 200.0
    for f in 1:length(mesh.face_tags)
        if mesh.face_tags[f] === :interface
            A = mesh.face_areas[f]
            ale.phi_mesh[f] = U_slip * A
        end
    end
    breakup = FiniteVolumeMethod.KHACTBreakup()

    _couple(
        tracker, breakup, ale, mesh,
        1.2, 0.072, 1.0;
        interface_patches = Symbol[:interface],
        d_parent = 1.0e-4,
    )

    @test length(tracker.particles) > 0
    for p in tracker.particles
        @test norm(p.velocity) > 0.0
        # All interface faces on this mesh point outward in -y (bottom
        # boundary has face_normals = (0, -1)). U_slip = 200 · (-1) so
        # expected magnitude is 200.
        @test isapprox(norm(p.velocity), U_slip; rtol = 1.0e-10)
        # Particle y-position must equal the face centre y = 0 on the
        # bottom row; x lies on a face centre in (0, 1).
        @test isapprox(p.position[2], 0.0; atol = 1.0e-12)
        @test 0.0 < p.position[1] < 1.0
    end
end

@testset "V&V FSI — mass deficit upper-bounded by total breakup mass" begin
    # Physical sanity: mass_source ≤ n_injected · m_drop (equality under
    # the handshake stub, but the inequality is the generic invariant).
    mesh = build_interface_tagged_mesh(4, 4, 1.0, 1.0)
    tracker = _ParticleTracker{2, Float64}()
    ale = _MeshMotionState(mesh)
    U_slip = 250.0
    for f in 1:length(mesh.face_tags)
        if mesh.face_tags[f] === :interface
            A = mesh.face_areas[f]
            ale.phi_mesh[f] = U_slip * A
        end
    end
    breakup = FiniteVolumeMethod.KHACTBreakup()
    rho_l = 800.0
    d_parent = 2.0e-4

    res = _couple(
        tracker, breakup, ale, mesh,
        1.2, 0.072, 1.0;
        interface_patches = Symbol[:interface],
        rho_l = rho_l,
        d_parent = d_parent,
    )

    m_drop = rho_l * (pi / 6) * d_parent^3
    total_possible = res.n_injected * m_drop
    @test res.total_mass_released <= total_possible + 1.0e-14
    @test all(ms -> ms >= 0.0, res.mass_source)
end

@testset "V&V FSI — selectivity: only interface-tagged faces participate" begin
    # Rename the tag to :liquid so none of the faces match the default
    # :interface — must produce zero activity even with strong slip.
    m = build_cartesian_unstructured_mesh(4, 4, 1.0, 1.0)
    new_tags = copy(m.face_tags)
    for f in 1:length(new_tags)
        if new_tags[f] === :bottom
            new_tags[f] = :liquid
        end
    end
    mesh = FiniteVolumeMethod.UnstructuredFVMMesh{2, Float64}(
        m.cell_centers, m.cell_volumes, m.face_cells,
        m.face_centers, m.face_areas, m.face_normals,
        new_tags, m.face_velocity, m.cell_faces,
    )
    tracker = _ParticleTracker{2, Float64}()
    ale = _MeshMotionState(mesh)
    for f in 1:length(mesh.face_tags)
        if mesh.face_tags[f] === :liquid
            A = mesh.face_areas[f]
            ale.phi_mesh[f] = 200.0 * A
        end
    end
    breakup = FiniteVolumeMethod.KHACTBreakup()

    # First: default patch list [:interface] — should not match any face.
    res_default = _couple(
        tracker, breakup, ale, mesh,
        1.2, 0.072, 1.0
    )
    @test res_default.n_injected == 0
    @test length(tracker.particles) == 0

    # Now pass the matching tag list.
    res_match = _couple(
        tracker, breakup, ale, mesh,
        1.2, 0.072, 1.0;
        interface_patches = Symbol[:liquid],
    )
    @test res_match.n_injected == 4
    @test length(tracker.particles) == 4
end

@testset "V&V FSI — LISA breakup dispatch works end-to-end" begin
    mesh = build_interface_tagged_mesh(4, 4, 1.0, 1.0)
    tracker = _ParticleTracker{2, Float64}()
    ale = _MeshMotionState(mesh)
    for f in 1:length(mesh.face_tags)
        if mesh.face_tags[f] === :interface
            A = mesh.face_areas[f]
            ale.phi_mesh[f] = 100.0 * A
        end
    end
    lisa = FiniteVolumeMethod.LISABreakup()

    res = _couple(
        tracker, lisa, ale, mesh,
        1.2, 0.03, 1.0;
        interface_patches = Symbol[:interface],
        h_sheet = 1.0e-5,
        d_parent = 1.0e-4,
    )
    @test res.n_injected >= 0  # no regression: dispatch runs
    @test length(res.mass_source) == length(mesh.cell_volumes)
end

@testset "V&V FSI — short dt (< τ_b) suppresses trigger" begin
    # Make dt smaller than τ_b so the timescale criterion fails.
    mesh = build_interface_tagged_mesh(4, 4, 1.0, 1.0)
    tracker = _ParticleTracker{2, Float64}()
    ale = _MeshMotionState(mesh)
    for f in 1:length(mesh.face_tags)
        if mesh.face_tags[f] === :interface
            A = mesh.face_areas[f]
            ale.phi_mesh[f] = 150.0 * A
        end
    end
    breakup = FiniteVolumeMethod.KHACTBreakup()
    # τ_b for d=1e-4, U=150, ρ_g=1.2, ρ_l=1000, μ=1e-3, σ=0.072 is
    # roughly O(1e-6) s. Pick dt one order of magnitude smaller.
    dt_short = 1.0e-9

    res = _couple(
        tracker, breakup, ale, mesh,
        1.2, 0.072, dt_short;
        interface_patches = Symbol[:interface],
        d_parent = 1.0e-4,
    )
    @test res.n_injected == 0
    @test length(tracker.particles) == 0
end

@testset "V&V FSI — repeated calls accumulate in the tracker monotonically" begin
    mesh = build_interface_tagged_mesh(4, 4, 1.0, 1.0)
    tracker = _ParticleTracker{2, Float64}()
    ale = _MeshMotionState(mesh)
    for f in 1:length(mesh.face_tags)
        if mesh.face_tags[f] === :interface
            A = mesh.face_areas[f]
            ale.phi_mesh[f] = 200.0 * A
        end
    end
    breakup = FiniteVolumeMethod.KHACTBreakup()

    res1 = _couple(
        tracker, breakup, ale, mesh,
        1.2, 0.072, 1.0;
        interface_patches = Symbol[:interface],
    )
    n1 = length(tracker.particles)
    @test n1 == res1.n_injected

    res2 = _couple(
        tracker, breakup, ale, mesh,
        1.2, 0.072, 1.0;
        interface_patches = Symbol[:interface],
    )
    n2 = length(tracker.particles)
    @test n2 == n1 + res2.n_injected
    @test n2 > n1  # new particles were added
end

@testset "V&V FSI — mass source is localised to owner cells" begin
    mesh = build_interface_tagged_mesh(4, 4, 1.0, 1.0)
    tracker = _ParticleTracker{2, Float64}()
    ale = _MeshMotionState(mesh)
    for f in 1:length(mesh.face_tags)
        if mesh.face_tags[f] === :interface
            A = mesh.face_areas[f]
            ale.phi_mesh[f] = 200.0 * A
        end
    end
    breakup = FiniteVolumeMethod.KHACTBreakup()

    res = _couple(
        tracker, breakup, ale, mesh,
        1.2, 0.072, 1.0;
        interface_patches = Symbol[:interface],
    )
    # Exactly 4 interface faces on bottom row; mass source must be
    # non-zero on the 4 owner cells (bottom row, indices 1..4) and zero
    # elsewhere.
    @test count(!iszero, res.mass_source) == 4
    @test all(!iszero, res.mass_source[1:4])
    @test all(iszero, res.mass_source[5:end])
end
