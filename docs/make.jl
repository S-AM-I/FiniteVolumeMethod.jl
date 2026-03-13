include(joinpath(@__DIR__, "..", "validation", "manifest.jl"))
using .RepoValidationManifest

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const VALIDATION_MANIFEST = RepoValidationManifest.load_manifest(joinpath(REPO_ROOT, "validation", "manifest.toml"))

IS_CI = get(ENV, "CI", "false") == "true"
IS_LIVESERVER = get(ENV, "LIVESERVER_ACTIVE", "false") == "true"
DOCS_EXECUTION_MODE = get(ENV, "FVM_DOCS_EXECUTION", IS_CI ? "subset" : "all")
DOCS_EXECUTION_MODE in ("all", "subset", "none") || error("FVM_DOCS_EXECUTION must be one of: all, subset, none")

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
    )
end

using FiniteVolumeMethod
using Documenter
using Literate
using Dates

# All the pages to be included
_PAGES = [
    "Introduction" => "index.md",
    "Scientific Governance" => [
        "Capability Matrix" => "capability_matrix.md",
    ],
    "Tutorials" => [
        "Parabolic and Elliptic PDEs" => [
            "Overview" => "tutorials/overview.md",
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
        "Hyperbolic Conservation Laws" => [
            "Overview" => "hyperbolic/overview.md",
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
    ],
    "Solvers for Specific Problems, and Writing Your Own" => [
        "Section Overview" => "wyos/overview.md",
        "Diffusion Equations" => "wyos/diffusion_equations.md",
        "Mean Exit Time Problems" => "wyos/mean_exit_time.md",
        "Linear Reaction-Diffusion Equations" => "wyos/linear_reaction_diffusion_equations.md",
        "Poisson's Equation" => "wyos/poissons_equation.md",
        "Laplace's Equation" => "wyos/laplaces_equation.md",
    ],
    "Verification & Validation" => [
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
            "Passive Scalar Convergence" => "verification/passive_scalar_convergence.md",
            "MHD Solver Comparison (HLL vs HLLD)" => "verification/mhd_solver_comparison.md",
            "GRMHD Flat-Space Reduction" => "verification/grmhd_asymptotic_flat.md",
            "GRMHD Newtonian Limit" => "verification/grmhd_newtonian_limit.md",
        ],
        "Analytical Benchmarks" => [
            "Sod Shock Tube Grid Convergence" => "verification/sod_grid_convergence.md",
            "Toro Riemann Tests" => "verification/toro_riemann_tests.md",
            "Balsara MHD Suite" => "verification/balsara_mhd_suite.md",
            "Brio-Wu Verification" => "verification/brio_wu_verification.md",
            "Orszag-Tang Verification" => "verification/orszag_tang_verification.md",
            "MHD div(B) Preservation" => "verification/mhd_divb_verification.md",
            "MHD Convergence" => "verification/mhd_convergence.md",
            "AMR Convergence" => "verification/amr_convergence.md",
            "Navier-Stokes Convergence" => "verification/ns_convergence.md",
            "Taylor-Green KE Decay" => "verification/tgv_kinetic_energy_decay.md",
            "Porous Medium (Barenblatt)" => "verification/porous_medium_barenblatt.md",
            "Premixed Flame 1D" => "verification/premixed_flame_1d.md",
            "SRMHD Convergence" => "verification/srmhd_convergence.md",
            "SRMHD Eigenmode Convergence" => "verification/srmhd_eigenmode_convergence.md",
            "GRMHD Convergence" => "verification/grmhd_convergence.md",
            "Bondi Accretion (Schwarzschild)" => "verification/bondi_accretion_schwarzschild.md",
        ],
        "Experimental Validation" => [
            "Lid-Driven Cavity" => "verification/lid_driven_cavity.md",
            "Fishbone-Moncrief Torus" => "verification/fishbone_moncrief_torus.md",
            "Heated Cavity" => "verification/heated_cavity.md",
        ],
    ],
    "Mathematical Details" => [
        "General FVM Theory" => "finite-volume-method.md",
        "Parabolic Solver (Cell-Vertex)" => "math.md",
        "Hyperbolic Solver (Cell-Centered)" => "hyperbolic/math.md",
    ],
    "Interface" => [
        "Parabolic Solver" => "interface.md",
        "Hyperbolic Solver" => "hyperbolic/interface.md",
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
    authors = "Daniel VandenHeuvel <danj.vandenheuvel@gmail.com>",
    sitename = "FiniteVolumeMethod.jl",
    (_git_works ? () : (remotes = nothing,))...,
    format = Documenter.HTML(;
        canonical = "https://cx-xd.github.io/FiniteVolumeMethod.jl",
        edit_link = "main",
        collapselevel = 2,
        assets = String[],
        mathengine = MathJax3(
            Dict(
                :loader => Dict("load" => ["[tex]/physics"]),
                :tex => Dict(
                    "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                    "tags" => "ams",
                    "packages" => ["base", "ams", "autoload", "physics"]
                )
            )
        )
    ),
    draft = IS_LIVESERVER,
    pages = _PAGES,
    warnonly = IS_LIVESERVER
)

# Only run deploydocs for local/non-Actions deployment (e.g. TagBot).
# GitHub Pages deployment is handled by actions/deploy-pages in CI.
if !IS_CI
    deploydocs(;
        repo = "github.com/cx-xd/FiniteVolumeMethod.jl",
        devbranch = "main",
        push_preview = true
    )
end
