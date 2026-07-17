# # Tutorial 12 — Ffowcs Williams-Hawkings Surface Integration
#
# Demonstrates the v3 aeroacoustics postprocessing: given a collection
# of surface panels carrying a pressure fluctuation, compute the
# far-field observer pressure via the Curle dipole (compact-surface
# subset of the full FW-H formulation). Uses a synthetic two-panel
# configuration so no upstream CFD solve is needed.
#
# Runtime budget: < 1 s on a laptop.
#
# Run with:
#
# ```bash
# julia --project=docs docs/src/literate_v3/12_aeroacoustics_fwh.jl
# ```
#
# What it demonstrates:
# - `FWHSurface{Dim, T}` construction from face indices / centers /
#   normals / areas
# - `FWHObserver` at a far-field position
# - `curle_dipole_pressure(observer, surface, p_surface, p_inf)` for
#   the dipole contribution, and `fwh_monopole_pressure` for
#   thickness / mass-flux terms

using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: FWHObserver, FWHSurface, curle_dipole_pressure, fwh_monopole_pressure
using StaticArrays
using Printf

# Synthetic two-panel body: antipodal normals on the unit sphere.
faces = [1, 2]
centers = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0)]
normals = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0)]
areas = [1.0, 1.0]

surface = FWHSurface{3, Float64}(faces, centers, normals, areas)
observer = FWHObserver(SVector(10.0, 0.0, 0.0))

# Case A — symmetric pressure loading: dipole cancels exactly.
p_equal = [1.0e5, 1.0e5]
p_symmetric = curle_dipole_pressure(observer, surface, p_equal, 1.0e5)

# Case B — asymmetric loading: the upstream-facing face sees higher
# pressure.
p_asym = [2.0e5, 1.0e5]
p_dipole = curle_dipole_pressure(observer, surface, p_asym, 1.0e5)

# Case C — monopole / thickness contribution from a uniform mass-flux
# time-derivative on each face.
dmass_dt = [1.0, 1.0]
p_monopole = fwh_monopole_pressure(observer, surface, dmass_dt)

# Analytical sanity check for the monopole case: r1 = 9, r2 = 11
p_monopole_exact = (1 / 9 + 1 / 11) / (4π)

println("=== FW-H / Curle surface integration ===")
@printf "observer position   : (%.1f, %.1f, %.1f) m\n" observer.position[1] observer.position[2] observer.position[3]
@printf "c_inf (sound speed) : %.1f m/s\n" observer.c_inf
@printf "—— Case A (symmetric p) ——\n"
@printf "Curle dipole        : %+.3e Pa (expected 0)\n" p_symmetric
@printf "—— Case B (asymmetric p) ——\n"
@printf "Curle dipole        : %+.3e Pa\n" p_dipole
@printf "—— Case C (monopole from dmass/dt) ——\n"
@printf "Monopole pressure   : %+.6e Pa\n" p_monopole
@printf "Analytical exact    : %+.6e Pa\n" p_monopole_exact

# Manifest feature  : stage6f.fwh_aeroacoustics (experimental)
# V&V tests         : test/v_and_v_fwh.jl, test/stage6_physics.jl
