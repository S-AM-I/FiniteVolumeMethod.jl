# # Polyhedral Volume Primitives
# This case verifies the polyhedral cell-volume primitives used by the
# mesh reader (`volume_tet`, `volume_hex`, `volume_pyramid`,
# `volume_prism`) against closed-form geometry:
# - Unit tetrahedron $V = 1/6$; degenerate (coplanar) $V = 0$
# - Axis-aligned hexahedron $V = L_x L_y L_z$
# - Square-based pyramid $V = \tfrac{1}{3} A_{\text{base}} h$
# - Right-triangle prism $V = \tfrac{1}{2} h$
#
# plus the invariances any volume functional must satisfy: $\alpha^3$
# scaling, translation invariance, and vertex-permutation (reflection)
# insensitivity of the unsigned volume.
#
# ## Acceptance Gates
# - All closed-form volumes to relative $10^{-14}$
# - Scaling, translation, and permutation invariances to $10^{-14}$
# - Wrong node counts throw

using FiniteVolumeMethod
using FiniteVolumeMethod: volume_hex, volume_prism, volume_pyramid, volume_tet
using CairoMakie
using Test #src

# The Cartesian unstructured-mesh helper ships with the test suite; locate it
# relative to the installed package so the path resolves from both the docs
# build and the evidence runner.
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

Node3D = FiniteVolumeMethod.Node3D

unit_tet = [
    Node3D(0.0, 0.0, 0.0), Node3D(1.0, 0.0, 0.0),
    Node3D(0.0, 1.0, 0.0), Node3D(0.0, 0.0, 1.0),
]
unit_hex = [
    Node3D(0.0, 0.0, 0.0), Node3D(1.0, 0.0, 0.0),
    Node3D(1.0, 1.0, 0.0), Node3D(0.0, 1.0, 0.0),
    Node3D(0.0, 0.0, 1.0), Node3D(1.0, 0.0, 1.0),
    Node3D(1.0, 1.0, 1.0), Node3D(0.0, 1.0, 1.0),
]
unit_pyramid = [
    Node3D(0.0, 0.0, 0.0), Node3D(1.0, 0.0, 0.0),
    Node3D(1.0, 1.0, 0.0), Node3D(0.0, 1.0, 0.0),
    Node3D(0.5, 0.5, 1.0),
]
unit_prism = [
    Node3D(0.0, 0.0, 0.0), Node3D(1.0, 0.0, 0.0), Node3D(0.0, 1.0, 0.0),
    Node3D(0.0, 0.0, 1.0), Node3D(1.0, 0.0, 1.0), Node3D(0.0, 1.0, 1.0),
]

# ## Closed-Form Volumes
closed_form_errors = [
    abs(volume_tet(unit_tet) - 1.0 / 6.0) / (1.0 / 6.0),
    abs(volume_hex(unit_hex) - 1.0),
    abs(volume_pyramid(unit_pyramid) - 1.0 / 3.0) / (1.0 / 3.0),
    abs(volume_prism(unit_prism) - 0.5) / 0.5,
]

degenerate_tet = volume_tet(
    [
        Node3D(0.0, 0.0, 0.0), Node3D(1.0, 0.0, 0.0),
        Node3D(0.0, 1.0, 0.0), Node3D(1.0, 1.0, 0.0),
    ]
)

box_errors = map(((2.0, 3.0, 5.0), (0.5, 1.0, 0.25), (10.0, 1.0, 0.1))) do (Lx, Ly, Lz)
    nodes = [
        Node3D(0.0, 0.0, 0.0), Node3D(Lx, 0.0, 0.0),
        Node3D(Lx, Ly, 0.0), Node3D(0.0, Ly, 0.0),
        Node3D(0.0, 0.0, Lz), Node3D(Lx, 0.0, Lz),
        Node3D(Lx, Ly, Lz), Node3D(0.0, Ly, Lz),
    ]
    abs(volume_hex(nodes) - Lx * Ly * Lz) / (Lx * Ly * Lz)
end

pyramid_height_errors = map((0.5, 1.0, 2.5, 5.0)) do h
    nodes = [
        Node3D(0.0, 0.0, 0.0), Node3D(1.0, 0.0, 0.0),
        Node3D(1.0, 1.0, 0.0), Node3D(0.0, 1.0, 0.0),
        Node3D(0.5, 0.5, h),
    ]
    abs(volume_pyramid(nodes) - h / 3.0) / (h / 3.0)
end

# ## Invariances
scaling_errors = map((0.5, 2.0, 3.0)) do alpha
    scaled_tet = [Node3D(n.x * alpha, n.y * alpha, n.z * alpha) for n in unit_tet]
    scaled_hex = [Node3D(n.x * alpha, n.y * alpha, n.z * alpha) for n in unit_hex]
    max(
        abs(volume_tet(scaled_tet) - alpha^3 * volume_tet(unit_tet)) / (alpha^3 / 6),
        abs(volume_hex(scaled_hex) - alpha^3 * volume_hex(unit_hex)) / alpha^3,
    )
end

translated_tet = [Node3D(n.x + 5.0, n.y - 3.0, n.z + 7.2) for n in unit_tet]
translation_error = abs(volume_tet(translated_tet) - volume_tet(unit_tet)) / (1.0 / 6.0)
permutation_exact = volume_tet([unit_tet[2], unit_tet[1], unit_tet[3], unit_tet[4]]) ==
    volume_tet(unit_tet)

wrong_count_throws = true
for f in (volume_hex, volume_pyramid, volume_prism)
    try
        f([Node3D(0.0, 0.0, 0.0), Node3D(1.0, 0.0, 0.0)])
        global wrong_count_throws = false
    catch err
        err isa ErrorException || (global wrong_count_throws = false)
    end
end

# ## Visualisation — Computed vs Exact
labels = ["tet 1/6", "hex 1", "pyramid 1/3", "prism 1/2"]
computed = [
    volume_tet(unit_tet), volume_hex(unit_hex),
    volume_pyramid(unit_pyramid), volume_prism(unit_prism),
]
exact = [1.0 / 6.0, 1.0, 1.0 / 3.0, 0.5]

fig1 = Figure(fontsize = 24, size = (700, 450))
ax1 = Axis(
    fig1[1, 1], ylabel = "volume", xticks = (1:4, labels),
    title = "Polyhedral primitives vs closed form"
)
barplot!(ax1, 1:4, computed, color = :steelblue, label = "computed")
scatter!(ax1, 1:4, exact, color = :black, marker = :hline, markersize = 24, label = "exact")
axislegend(ax1, position = :rt)
resize_to_layout!(fig1)
fig1
if isdefined(@__MODULE__, :evidence_artifact_path)
    save(evidence_artifact_path("polyhedral_volumes.png"), fig1)
end

# ## Acceptance
@test all(e -> e < 1.0e-14, closed_form_errors) #src
@test degenerate_tet < 1.0e-14 #src
@test all(e -> e < 1.0e-14, box_errors) #src
@test all(e -> e < 1.0e-14, pyramid_height_errors) #src
@test all(e -> e < 1.0e-13, scaling_errors) #src
@test translation_error < 1.0e-13 #src
@test permutation_exact #src
@test wrong_count_throws #src
@assert all(e -> e < 1.0e-14, closed_form_errors) #hide
@assert degenerate_tet < 1.0e-14 #hide
@assert all(e -> e < 1.0e-14, box_errors) #hide
@assert all(e -> e < 1.0e-14, pyramid_height_errors) #hide
@assert all(e -> e < 1.0e-13, scaling_errors) #hide
@assert translation_error < 1.0e-13 #hide
@assert permutation_exact #hide
@assert wrong_count_throws #hide

if isdefined(@__MODULE__, :record_evidence_result)
    record_evidence_result(
        metrics = Dict(
            "closed_form_errors" => closed_form_errors,
            "degenerate_tet" => degenerate_tet,
            "max_scaling_error" => maximum(scaling_errors),
            "translation_error" => translation_error,
        ),
        artifacts = ["polyhedral_volumes.png"],
        notes = [
            "Verification-stage exact-geometry evidence for polyhedral_mesh_io: the tet/hex/pyramid/prism volume primitives match closed forms to 1e-14 and satisfy scaling, translation, and permutation invariances.",
        ],
        summary = Dict(
            "primitives" => labels,
        ),
    )
end
