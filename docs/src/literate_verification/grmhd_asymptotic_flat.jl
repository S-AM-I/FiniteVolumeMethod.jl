using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # GRMHD Asymptotic Flat-Space Verification
# This example verifies the asymptotic reduction from GRMHD to SRMHD in
# flat spacetime (Minkowski metric). Two checks are performed:
#
# 1. **Flux consistency**: The GRMHD physical flux with `MinkowskiMetric`
#    must match the SRMHD physical flux to machine precision for a
#    comprehensive set of primitive states.
# 2. **Con2Prim round-trip**: Converting primitive -> conserved -> primitive
#    must recover the original state for both GRMHD and SRMHD.
#
# No PDE is solved -- this is a pure algebraic verification of the
# GRMHD -> SRMHD asymptotic limit (tier-1 reduction in the V&V hierarchy).
#
# ## Reference
# - Del Zanna, L. et al. (2007). ECHO: a Eulerian conservative high-order
#   scheme for general relativistic MHD. A&A, 473, 11-30.
# - Rezzolla, L. & Zanotti, O. (2013). Relativistic Hydrodynamics. Oxford
#   University Press.

using FiniteVolumeMethod
using StaticArrays
using Test #src
using CairoMakie

gamma = 5.0 / 3.0
eos = IdealGasEOS(gamma)
metric = MinkowskiMetric{2}()
grmhd = GRMHDEquations{2}(eos, metric)
srmhd = SRMHDEquations{1}(eos)

# ## Comprehensive State Set
# We test across a wide range of physical regimes: low/high velocity,
# low/high magnetic field, and combinations thereof.
test_states = [
    ## (label, [rho, vx, vy, vz, P, Bx, By, Bz])
    ("rest", SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    ("low-v", SVector(1.0, 0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    ("moderate-v", SVector(1.0, 0.5, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0)),
    ("high-v", SVector(1.0, 0.9, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    ("low-B", SVector(1.0, 0.3, 0.0, 0.0, 1.0, 0.01, 0.01, 0.0)),
    ("moderate-B", SVector(1.0, 0.3, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0)),
    ("high-B", SVector(1.0, 0.3, 0.0, 0.0, 1.0, 5.0, 3.0, 0.0)),
    ("high-v high-B", SVector(1.0, 0.8, 0.3, 0.0, 1.0, 2.0, 1.0, 0.5)),
    ("low-rho", SVector(0.001, 0.5, 0.0, 0.0, 0.01, 0.1, 0.0, 0.0)),
    ("high-P", SVector(1.0, 0.3, 0.1, 0.0, 100.0, 1.0, 1.0, 0.0)),
]

# ## Check 1 --- Flux Consistency
# The GRMHD physical_flux with Minkowski metric must match SRMHD flux.
max_flux_diff = 0.0
for (label, w) in test_states
    for dir in 1:2
        F_gr = physical_flux(grmhd, w, dir)
        F_sr = physical_flux(srmhd, w, dir)
        diff = maximum(abs.(F_gr .- F_sr))
        global max_flux_diff = max(max_flux_diff, diff)
    end
end

# ## Check 2 --- Con2Prim Round-Trip Consistency
max_roundtrip_diff = 0.0
for (label, w) in test_states
    ## GRMHD round-trip
    u_gr = primitive_to_conserved(grmhd, w)
    w_gr = conserved_to_primitive(grmhd, u_gr)
    ## SRMHD round-trip
    u_sr = primitive_to_conserved(srmhd, w)
    w_sr = conserved_to_primitive(srmhd, u_sr)
    ## Check both recover original
    diff_gr = maximum(abs.(w_gr .- w))
    diff_sr = maximum(abs.(w_sr .- w))
    global max_roundtrip_diff = max(max_roundtrip_diff, diff_gr, diff_sr)
    ## Check GRMHD and SRMHD conserved vectors match (in Minkowski)
    u_diff = maximum(abs.(u_gr .- u_sr))
    global max_roundtrip_diff = max(max_roundtrip_diff, u_diff)
end

# ## Visualisation --- Error Summary
labels = [s[1] for s in test_states]
flux_diffs = Float64[]
for (label, w) in test_states
    d = 0.0
    for dir in 1:2
        F_gr = physical_flux(grmhd, w, dir)
        F_sr = physical_flux(srmhd, w, dir)
        d = max(d, maximum(abs.(F_gr .- F_sr)))
    end
    push!(flux_diffs, max(d, 1.0e-16))
end

fig = Figure(fontsize = 20, size = (900, 500))
ax = Axis(
    fig[1, 1]; xlabel = "Test State", ylabel = "Max |F_GR - F_SR|",
    yscale = log10, title = "GRMHD -> SRMHD Flux Consistency (Minkowski)",
    xticks = (1:length(labels), labels), xticklabelrotation = pi / 4,
)
barplot!(ax, 1:length(labels), flux_diffs; color = :steelblue)
hlines!(ax, [1.0e-13]; color = :red, linestyle = :dash, linewidth = 2, label = "Threshold (1e-13)")
axislegend(ax; position = :rt)
resize_to_layout!(fig)
fig

# ## Test Assertions
@test max_flux_diff < 1.0e-12 #src
@test max_roundtrip_diff < 1.0e-10 #src
@assert max_flux_diff < 1.0e-12 #hide
@assert max_roundtrip_diff < 1.0e-10 #hide
