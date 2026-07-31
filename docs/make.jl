include(joinpath(@__DIR__, "..", "validation", "manifest.jl"))
using .RepoValidationManifest

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const VALIDATION_MANIFEST = RepoValidationManifest.load_manifest(joinpath(REPO_ROOT, "validation", "manifest.toml"))

IS_CI = get(ENV, "CI", "false") == "true"
IS_LIVESERVER = get(ENV, "LIVESERVER_ACTIVE", "false") == "true"
DOCS_EXECUTION_MODE = get(ENV, "FVM_DOCS_EXECUTION", IS_CI ? "subset" : "all")
DOCS_EXECUTION_MODE in ("all", "subset", "none") || error("FVM_DOCS_EXECUTION must be one of: all, subset, none")
USE_PRETTY_URLS = IS_CI || IS_LIVESERVER
DOCS_PRETTYURLS = get(ENV, "FVM_DOCS_PRETTYURLS", USE_PRETTY_URLS ? "true" : "false") == "true"

should_execute_example(entry) = !IS_LIVESERVER && (
    DOCS_EXECUTION_MODE == "all" ? entry.run_locally :
        DOCS_EXECUTION_MODE == "subset" ? entry.run_in_ci :
        false
)

if any(should_execute_example(entry) for entry in VALIDATION_MANIFEST.generated_pages)
    using CairoMakie
    CairoMakie.activate!()
end

using Literate
using Dates
ct() = Dates.format(now(), "HH:MM:SS")

function update_edit_url(content, source_relpath)
    content = replace(content, "<unknown>" => "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main")
    content = replace(content, "temp/" => "") # as of Literate 2.14.1
    content = replace(
        content,
        r"EditURL\s*=\s*\"[^\"]*\"" => "EditURL = \"https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/$source_relpath\""
    )
    return content
end

function add_just_the_code_section(file_path, source_relpath)
    file_name, file_ext = splitext(basename(file_path))
    new_file_path = joinpath(session_tmp, file_name * "_just_the_code" * file_ext)
    cp(file_path, new_file_path, force = true)
    open(new_file_path, "a") do io
        write(io, "\n")
        write(io, "# ## Just the code\n")
        write(io, "# An uncommented version of this example is given below.\n")
        write(io, "# You can view the source code for this file [here](<unknown>/$source_relpath).\n")
        write(io, "\n")
        write(io, "# ```julia\n")
        write(io, "# @__CODE__\n")
        write(io, "# ```\n")
    end
    return new_file_path
end

session_tmp = mktempdir()

for entry in VALIDATION_MANIFEST.generated_pages
    file_path = joinpath(REPO_ROOT, entry.source)
    outputdir = joinpath(@__DIR__, "src", dirname(entry.page))
    file = basename(file_path)
    mkpath(outputdir)
    new_file_path = add_just_the_code_section(file_path, entry.source)
    script = Literate.script(
        file_path,
        session_tmp;
        name = splitext(file)[1] * "_just_the_code_cleaned",
    )
    code = strip(read(script, String))
    @info "[$(ct())] Processing $file: Converting markdown script"
    line_ending_symbol = occursin(code, "\r\n") ? "\r\n" : "\n"
    code_clean = join(filter(x -> !endswith(x, "#hide"), split(code, r"\n|\r\n")), line_ending_symbol)
    code_clean = replace(code_clean, r"^# This file was generated .*$"m => "")
    code_clean = strip(code_clean)
    post_strip = content -> replace(content, "@__CODE__" => code_clean)
    editurl_update = content -> update_edit_url(content, entry.source)
    Literate.markdown(
        new_file_path,
        outputdir;
        documenter = true,
        postprocess = editurl_update ∘ post_strip,
        credit = true,
        execute = should_execute_example(entry),
        flavor = Literate.DocumenterFlavor(),
        name = splitext(file)[1],
        # Pages excluded from this execution mode must emit plain fences —
        # otherwise Documenter executes their `@example` blocks anyway and the
        # run_locally/run_in_ci manifest flags have no effect.
        (
            should_execute_example(entry) ? () :
                (codefence = "````julia" => "````",)
        )...,
    )
end

using FiniteVolumeMethod
using CommonSolve  # `solve` is CommonSolve.solve; makes the canonical name resolvable for @ref targets
using Documenter
using DocumenterVitepress
using Literate
using Dates

# All the pages to be included
_PAGES = [
    "Getting Started" => [
        "Introduction" => "index.md",
        "Capability Matrix" => "capability_matrix.md",
    ],
    "Solver Families" => [
        "Parabolic (Cell-Vertex)" => [
            "Overview" => "tutorials/overview.md",
            "Interface" => "interface.md",
            "Mathematical Details" => "math.md",
            "Diffusion Equation on a Square Plate" => "tutorials/diffusion_equation_on_a_square_plate.md",
            "Diffusion Equation in a Wedge with Mixed Boundary Conditions" => "tutorials/diffusion_equation_in_a_wedge_with_mixed_boundary_conditions.md",
            "Reaction-Diffusion Equation with a Time-dependent Dirichlet Boundary Condition on a Disk" => "tutorials/reaction_diffusion_equation_with_a_time_dependent_dirichlet_boundary_condition_on_a_disk.md",
            "Porous-Medium Equation" => "tutorials/porous_medium_equation.md",
            "Porous-Fisher Equation and Travelling Waves" => "tutorials/porous_fisher_equation_and_travelling_waves.md",
            "Piecewise Linear and Natural Neighbour Interpolation for an Advection-Diffusion Equation" => "tutorials/piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation.md",
            "Helmholtz Equation with Inhomogeneous Boundary Conditions" => "tutorials/helmholtz_equation_with_inhomogeneous_boundary_conditions.md",
            "Laplace's Equation with Internal Dirichlet Conditions" => "tutorials/laplaces_equation_with_internal_dirichlet_conditions.md",
            "Equilibrium Temperature Distribution with Mixed Boundary Conditions and using EnsembleProblems" => "tutorials/equilibrium_temperature_distribution_with_mixed_boundary_conditions_and_using_ensembleproblems.md",
            "A Reaction-Diffusion Brusselator System of PDEs" => "tutorials/reaction_diffusion_brusselator_system_of_pdes.md",
            "Gray-Scott Model: Turing Patterns from a Coupled Reaction-Diffusion System" => "tutorials/gray_scott_model_turing_patterns_from_a_coupled_reaction_diffusion_system.md",
            "Diffusion Equation on an Annulus" => "tutorials/diffusion_equation_on_an_annulus.md",
            "Mean Exit Time" => "tutorials/mean_exit_time.md",
            "Solving Mazes with Laplace's Equation" => "tutorials/solving_mazes_with_laplaces_equation.md",
            "Keller-Segel Model of Chemotaxis" => "tutorials/keller_segel_chemotaxis.md",
        ],
        "Hyperbolic (Cell-Centered)" => [
            "Overview" => "hyperbolic/overview.md",
            "Interface" => "hyperbolic/interface.md",
            "Mathematical Details" => "hyperbolic/math.md",
            "Sod Shock Tube" => "hyperbolic/tutorials/sod_shock_tube.md",
            "Sedov Blast Wave" => "hyperbolic/tutorials/sedov_blast_wave.md",
            "Brio-Wu MHD Shock Tube" => "hyperbolic/tutorials/brio_wu_shock_tube.md",
            "Orszag-Tang Vortex" => "hyperbolic/tutorials/orszag_tang_vortex.md",
            "Taylor-Green Vortex Decay" => "hyperbolic/tutorials/taylor_green_vortex.md",
            "Field Loop Advection" => "hyperbolic/tutorials/field_loop_advection.md",
            "Kelvin-Helmholtz Instability" => "hyperbolic/tutorials/kelvin_helmholtz_instability.md",
            "Balsara SRMHD Shock Tube" => "hyperbolic/tutorials/balsara_srmhd_shock_tube.md",
            "WENO Convergence Study" => "hyperbolic/tutorials/weno_convergence.md",
            "Couette Flow" => "hyperbolic/tutorials/couette_flow.md",
            "IMEX Stiff Relaxation" => "hyperbolic/tutorials/imex_stiff_relaxation.md",
            "AMR Sedov Blast" => "hyperbolic/tutorials/amr_sedov_blast.md",
            "Limiter Comparison" => "hyperbolic/tutorials/limiter_comparison.md",
            "MHD Rotor" => "hyperbolic/tutorials/mhd_rotor.md",
            "GRMHD in Flat Spacetime" => "hyperbolic/tutorials/grmhd_flat_space_shock.md",
            "SRMHD Cylindrical Blast" => "hyperbolic/tutorials/srmhd_cylindrical_blast.md",
            "Shallow Water Dam Break" => "hyperbolic/tutorials/shallow_water_dam_break.md",
            "SR Hydro Blast Wave" => "hyperbolic/tutorials/srhydro_blast_wave.md",
            "Resistive MHD Current Sheet" => "hyperbolic/tutorials/resistive_mhd_current_sheet.md",
            "Hall MHD Whistler Waves" => "hyperbolic/tutorials/hall_mhd_whistler.md",
            "Two-Fluid Plasma Sod" => "hyperbolic/tutorials/two_fluid_sod.md",
        ],
        "Collocated Incompressible" => [
            "Overview" => "literate_v3/README.md",
            "Lid-Driven Cavity (SIMPLE)" => "collocated/tutorials/01_lid_driven_cavity.md",
            "k-epsilon Turbulent Channel" => "collocated/tutorials/03_kepsilon_channel.md",
            "Rayleigh-Benard Convection" => "collocated/tutorials/04_rayleigh_benard.md",
            "Dam Break (VOF)" => "collocated/tutorials/05_dam_break.md",
            "One-Step Combustion" => "collocated/tutorials/06_combustion_one_step.md",
            "P1 Radiation" => "collocated/tutorials/07_radiation_p1.md",
            "Lagrangian Particles (Stokes Drag)" => "collocated/tutorials/08_dpm_stokes.md",
            "Dynamic Mesh Oscillator (ALE)" => "collocated/tutorials/09_dynamic_mesh_oscillator.md",
            "Two-Fluid Bubble Column" => "collocated/tutorials/10_two_fluid_bubble_column.md",
        ],
    ],
    "Solvers for Specific Problems, and Writing Your Own" => [
        "Section Overview" => "wyos/overview.md",
        "Diffusion Equations" => "wyos/diffusion_equations.md",
        "Mean Exit Time Problems" => "wyos/mean_exit_time.md",
        "Linear Reaction-Diffusion Equations" => "wyos/linear_reaction_diffusion_equations.md",
        "Poisson's Equation" => "wyos/poissons_equation.md",
        "Laplace's Equation" => "wyos/laplaces_equation.md",
    ],
    "Validation & Evidence" => [
        "V&V Status" => "research_governance.md",
        "Algorithm Provenance" => "provenance.md",
        "Overview" => "verification/overview.md",
        "Code Verification" => [
            "MMS Convergence (Parabolic)" => "verification/mms_convergence.md",
            "Decoupled MMS Convergence" => "verification/mms_spatial_temporal_decoupled.md",
            "Euler MMS Convergence" => "verification/euler_mms_convergence.md",
            "Poisson Equation Convergence" => "verification/poisson_convergence.md",
            "Smooth Advection Order of Accuracy" => "verification/smooth_advection_convergence.md",
            "Source Term Convergence" => "verification/source_term_convergence.md",
            "Flux Balance Verification" => "verification/flux_balance_verification.md",
            "Conservation Verification" => "verification/conservation_verification.md",
            "Species Conservation" => "verification/species_conservation.md",
            "Coupling Null-Source Identity" => "verification/coupling_nullsource_identity.md",
            "Coupled Mass Conservation" => "verification/coupling_mass_conservation.md",
            "Passive Scalar Convergence" => "verification/passive_scalar_convergence.md",
            "GRMHD Flat-Space Reduction" => "verification/grmhd_asymptotic_flat.md",
            "GRMHD Newtonian Limit" => "verification/grmhd_newtonian_limit.md",
            "MHD div(B) Preservation" => "verification/mhd_divb_verification.md",
            "AMR Smooth-Pulse Convergence" => "verification/amr_convergence.md",
            "AMR Regridding Conservation" => "verification/amr_regridding_conservation.md",
            "Poiseuille Convergence (Collocated)" => "verification/poiseuille_convergence_collocated.md",
            "DHIT Decay (Standard k-ε)" => "verification/kepsilon_dhit_decay.md",
            "Solid Conduction Convergence" => "verification/solid_conduction_convergence.md",
            "Smagorinsky Shear Verification" => "verification/smagorinsky_shear_verification.md",
            "VOF Disc Translation" => "verification/vof_disc_translation.md",
            "P1 Slab Attenuation" => "verification/p1_slab_attenuation.md",
            "Laplacian Operator MMS" => "verification/laplacian_operator_mms.md",
            "Stokes Terminal Velocity" => "verification/stokes_terminal_velocity.md",
            "ALE GCL Invariants" => "verification/ale_gcl_invariants.md",
            "Linear-Solver Backend Parity" => "verification/linear_solver_backend_parity.md",
            "Derived-Field Invariants" => "verification/derived_field_invariants.md",
        ],
        "Analytical Benchmarks" => [
            "Sod Shock Tube Grid Convergence" => "verification/sod_grid_convergence.md",
            "Toro Riemann Tests" => "verification/toro_riemann_tests.md",
            "Brio-Wu Shock Tube" => "verification/brio_wu_verification.md",
            "MHD Convergence" => "verification/mhd_convergence.md",
            "Navier-Stokes Convergence" => "verification/ns_convergence.md",
            "Taylor-Green KE Decay" => "verification/tgv_kinetic_energy_decay.md",
            "Porous Medium (Barenblatt)" => "verification/porous_medium_barenblatt.md",
            "Coupled Cooling Reference" => "verification/coupling_cooling_reference.md",
            "AMR Reference Tracking" => "verification/amr_reference_tracking.md",
            "SRMHD Convergence" => "verification/srmhd_convergence.md",
            "SRMHD Eigenmode Convergence" => "verification/srmhd_eigenmode_convergence.md",
            "GRMHD Convergence" => "verification/grmhd_convergence.md",
            "Ghia Lid-Driven Cavity (Re = 100)" => "verification/ghia_cavity_re100.md",
            "Log-Layer Equilibrium (Standard k-ε)" => "verification/kepsilon_loglayer_equilibrium.md",
            "Unsteady Heat Decay" => "verification/unsteady_heat_decay.md",
            "WALE Operator Invariants" => "verification/wale_invariants.md",
            "MULES Limiter Invariants" => "verification/mules_limiter_invariants.md",
            "Gradient & Divergence MMS" => "verification/gradient_divergence_mms.md",
        ],
    ],
    "Experimental" => [
        "Scope and Caveats" => "experimental/overview.md",
        "Low-Mach Compressible Channel" => "experimental/tutorials/02_compressible_channel.md",
        "Linear Elasticity Beam" => "experimental/tutorials/11_solid_mechanics_beam.md",
        "Ffowcs Williams-Hawkings" => "experimental/tutorials/12_aeroacoustics_fwh.md",
    ],
    "Mathematical Foundations" => [
        "General FVM Theory" => "finite-volume-method.md",
    ],
    "API Reference" => [
        "Overview" => "api/overview.md",
        "Collocated" => "api/collocated.md",
        "I/O and Session Tooling" => "api/io.md",
        "Experimental" => "api/experimental.md",
    ],
    "Migration" => [
        "v4.0" => "migration/v4.md",
    ],
    "Contributing" => [
        "Julia & dependency compat policy" => "contributing/compat.md",
    ],
]

# Make sure we haven't forgotten any files
set = Set{String}()
function _collect_pages!(set, pages)
    for page in pages
        if page[2] isa String
            push!(set, normpath(page[2]))
        else
            _collect_pages!(set, page[2])
        end
    end
    return
end
_collect_pages!(set, _PAGES)
missing_generated_pages = RepoValidationManifest.missing_generated_pages(VALIDATION_MANIFEST, set)
!isempty(missing_generated_pages) && error("Generated pages missing from docs navigation: $missing_generated_pages")
unexpected_generated_pages = RepoValidationManifest.unexpected_generated_navigation_pages(VALIDATION_MANIFEST, set)
!isempty(unexpected_generated_pages) && error("Docs navigation contains generated pages with no manifest entry: $unexpected_generated_pages")
missing_set = String[]
doc_dir = joinpath(@__DIR__, "src", "")
for (root, dir, files) in walkdir(doc_dir)
    for file in files
        filename = normpath(replace(joinpath(root, file), doc_dir => ""))
        if endswith(filename, ".md") && filename ∉ set
            push!(missing_set, filename)
        end
    end
end
!isempty(missing_set) && error("Missing files: $missing_set")

# Make and deploy
DocMeta.setdocmeta!(
    FiniteVolumeMethod, :DocTestSetup, :(using FiniteVolumeMethod, Test);
    recursive = true
)
# In Docker containers, git may fail due to ownership mismatch on bind mounts.
# Detect this and disable remote source links gracefully.
_git_works = try
    success(`git -C $(joinpath(@__DIR__, "..")) rev-parse --show-toplevel`)
catch
    false
end
makedocs(;
    modules = [FiniteVolumeMethod],
    # The repo exposes a much broader research/development surface than the
    # narrative manual aims to cover. Governance tests and the validation
    # manifest enforce the authoritative claim surface separately, so
    # Documenter's export-coverage check is intentionally disabled here.
    checkdocs = :none,
    authors = "Daniel VandenHeuvel <danj.vandenheuvel@gmail.com>, cx-xd contributors",
    sitename = "FiniteVolumeMethod.jl",
    (_git_works ? () : (remotes = nothing,))...,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/cx-xd/FiniteVolumeMethod.jl",
        devbranch = "main",
        devurl = "dev",
        deploy_url = "https://fvm.cx-xd.org",
    ),
    draft = IS_LIVESERVER,
    pages = _PAGES,
    warnonly = IS_LIVESERVER
)

# Flatten DocumenterVitepress's single-base `build/1/` into `build/`.
# DocumenterVitepress v0.2+ writes build output to `build/<i>/` where `<i>`
# is the 1-indexed position in the bases array (see DocumenterVitepress
# README §"Sub-URLs and multi-base deployments"). For single-base local
# and production builds the only base is `""` and the output always lives
# at `build/1/`. Move the contents up one level so `build/` behaves like
# a normal Documenter output and the GitHub Pages deploy in
# .github/workflows/CI.yml can point at `docs/build` directly. Guard
# skips when more than one base is present.
let build_root = joinpath(@__DIR__, "build")
    single_base = joinpath(build_root, "1")
    if isdir(single_base)
        numeric_subdirs = filter(readdir(build_root)) do d
            isdir(joinpath(build_root, d)) && !isempty(d) && all(isdigit, d)
        end
        if numeric_subdirs == ["1"]
            for entry in readdir(single_base)
                mv(
                    joinpath(single_base, entry),
                    joinpath(build_root, entry); force = true
                )
            end
            rm(single_base)
            # Drop DocumenterVitepress book-keeping so build/ contains
            # only the deployable site.
            for artifact in (".documenter", "bases.txt")
                path = joinpath(build_root, artifact)
                ispath(path) && rm(path; recursive = true, force = true)
            end
            @info "Flattened DocumenterVitepress build/1/ → build/"
        end
    end
end

# Docs deployment is handled exclusively by actions/deploy-pages in
# .github/workflows/CI.yml (docs job). The former !IS_CI deploydocs
# branch was removed: outside Actions, deploydocs has no deploy keys /
# DOCUMENTER_KEY and could never fire meaningfully, and the gh-pages
# branch is not the deployment mechanism for this repo.
