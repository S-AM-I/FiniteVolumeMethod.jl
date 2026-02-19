using FiniteVolumeMethod
using Test
using Dates
using Aqua

ct() = Dates.format(now(), "HH:MM:SS")
function safe_include(filename; name = filename) # Workaround for not being able to interpolate into SafeTestset test names
    mod = @eval module $(gensym()) end
    @info "[$(ct())] Testing $name"
    return @testset verbose = true "Example: $name" begin
        Base.include(mod, filename)
    end
end

@testset verbose = true "FiniteVolumeMethod.jl" begin
    @testset verbose = true "Geometry" begin
        safe_include("geometry.jl")
    end
    @testset verbose = true "Conditions" begin
        safe_include("conditions.jl")
    end
    @testset verbose = true "Robin BCs" begin
        safe_include("robin.jl")
    end
    @testset verbose = true "Problem" begin
        safe_include("problem.jl")
    end
    @testset verbose = true "Equations" begin
        safe_include("equations.jl")
    end
    @testset verbose = true "Schemes" begin
        safe_include("schemes.jl")
    end
    @testset verbose = true "Advanced BCs" begin
        safe_include("advanced_bcs.jl")
    end
    @testset verbose = true "Physics Models" begin
        safe_include("physics.jl")
    end
    @testset verbose = true "Hyperbolic Solver" begin
        safe_include("hyperbolic.jl")
    end
    @testset verbose = true "Hyperbolic 2D + HLLC" begin
        safe_include("hyperbolic_2d.jl")
    end
    @testset verbose = true "MHD + HLLD" begin
        safe_include("mhd.jl")
    end
    @testset verbose = true "MHD 2D + CT" begin
        safe_include("mhd_2d.jl")
    end
    @testset verbose = true "Navier-Stokes" begin
        safe_include("navier_stokes.jl")
    end
    @testset verbose = true "SRMHD" begin
        safe_include("srmhd.jl")
    end
    @testset verbose = true "SRMHD 2D" begin
        safe_include("srmhd_2d.jl")
    end
    @testset verbose = true "GRMHD" begin
        safe_include("grmhd.jl")
    end
    @testset verbose = true "GRMHD 2D" begin
        safe_include("grmhd_2d.jl")
    end
    @testset verbose = true "Hyperbolic 3D" begin
        safe_include("hyperbolic_3d.jl")
    end
    @testset verbose = true "MHD 3D" begin
        safe_include("mhd_3d.jl")
    end
    @testset verbose = true "AMR" begin
        safe_include("amr.jl")
    end
    @testset verbose = true "WENO" begin
        safe_include("weno.jl")
    end
    @testset verbose = true "IMEX" begin
        safe_include("imex.jl")
    end
    @testset verbose = true "Unstructured Hyperbolic" begin
        safe_include("unstructured_hyperbolic.jl")
    end
    @testset verbose = true "Multi-Physics Coupling" begin
        safe_include("coupling.jl")
    end
    @testset verbose = true "Performance & Threading" begin
        safe_include("performance.jl")
    end
    @testset verbose = true "Advanced Numerics" begin
        safe_include("advanced_numerics.jl")
    end
    @testset verbose = true "Extended Physics" begin
        safe_include("extended_physics.jl")
    end
    @testset verbose = true "Reactive Euler" begin
        safe_include("reactive_euler.jl")
    end
    @testset verbose = true "README" begin
        safe_include("README.jl")
    end

    @testset verbose = true "Coordinate Systems" begin
        safe_include("test_coordinate_systems.jl")
    end

    @testset verbose = true "Dashboard" begin
        safe_include("test_dashboard.jl")
    end

    @testset verbose = true "Tutorials" begin
        dir = joinpath(dirname(@__DIR__), "docs", "src", "literate_tutorials")
        files = readdir(dir)
        file_names = [
            "diffusion_equation_in_a_wedge_with_mixed_boundary_conditions.jl",
            "diffusion_equation_on_a_square_plate.jl",
            "diffusion_equation_on_an_annulus.jl",
            "equilibrium_temperature_distribution_with_mixed_boundary_conditions_and_using_ensembleproblems.jl",
            "helmholtz_equation_with_inhomogeneous_boundary_conditions.jl",
            "laplaces_equation_with_internal_dirichlet_conditions.jl",
            "mean_exit_time.jl",
            "piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation.jl",
            "porous_fisher_equation_and_travelling_waves.jl",
            "porous_medium_equation.jl",
            "reaction_diffusion_brusselator_system_of_pdes.jl",
            "reaction_diffusion_equation_with_a_time_dependent_dirichlet_boundary_condition_on_a_disk.jl",
            "solving_mazes_with_laplaces_equation.jl",
            "gray_scott_model_turing_patterns_from_a_coupled_reaction_diffusion_system.jl",
        ] # do it manually just to make it easier for testing individual files rather than in a loop, e.g. one like
        #=
        for file in files
            @testset "Example: $file" begin
                safe_include(joinpath(dir, file))
            end
        end
        =#
        @test length(files) == length(file_names) # make sure we didn't miss any
        safe_include(joinpath(dir, file_names[1]); name = file_names[1]) # diffusion_equation_in_a_wedge_with_mixed_boundary_conditions
        safe_include(joinpath(dir, file_names[2]); name = file_names[2]) # diffusion_equation_on_a_square_plate
        safe_include(joinpath(dir, file_names[3]); name = file_names[3]) # diffusion_equation_on_an_annulus
        safe_include(joinpath(dir, file_names[4]); name = file_names[4]) # equilibrium_temperature_distribution_with_mixed_boundary_conditions_and_using_ensembleproblems
        safe_include(joinpath(dir, file_names[5]); name = file_names[5]) # helmholtz_equation_with_inhomogeneous_boundary_conditions
        safe_include(joinpath(dir, file_names[6]); name = file_names[6]) # laplaces_equation_with_internal_dirichlet_conditions
        safe_include(joinpath(dir, file_names[7]); name = file_names[7]) # mean_exit_time
        safe_include(joinpath(dir, file_names[8]); name = file_names[8]) # piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation
        safe_include(joinpath(dir, file_names[9]); name = file_names[9]) # porous_fisher_equation_and_travelling_waves
        safe_include(joinpath(dir, file_names[10]); name = file_names[10]) # porous_medium_equation
        safe_include(joinpath(dir, file_names[11]); name = file_names[11]) # reaction_diffusion_brusselator_system_of_pdes
        safe_include(joinpath(dir, file_names[12]); name = file_names[12]) # reaction_diffusion_equation_with_a_time_dependent_dirichlet_boundary_condition_on_a_disk
        safe_include(joinpath(dir, file_names[13]); name = file_names[13]) # solving_mazes_with_laplaces_equation
        safe_include(joinpath(dir, file_names[14]); name = file_names[14]) # gray_scott_model_turing_patterns_from_a_coupled_reaction_diffusion_system
        # safe_include(joinpath(dir, file_names[15]); name=file_names[15]) # keller_segel_chemotaxis
    end

    @testset verbose = true "Custom Templates" begin
        dir = joinpath(dirname(@__DIR__), "docs", "src", "literate_wyos")
        files = readdir(dir)
        file_names = [
            "diffusion_equations.jl",
            "mean_exit_time.jl",
            "linear_reaction_diffusion_equations.jl",
            "poissons_equation.jl",
            "laplaces_equation.jl",
        ]
        @test length(files) == length(file_names) # make sure we didn't miss any
        safe_include(joinpath(dir, file_names[1]); name = file_names[1]) # diffusion_equations
        safe_include(joinpath(dir, file_names[2]); name = file_names[2]) # mean_exit_time
        safe_include(joinpath(dir, file_names[3]); name = file_names[3]) # linear_reaction_diffusion_equations
        safe_include(joinpath(dir, file_names[4]); name = file_names[4]) # poissons_equation
        safe_include(joinpath(dir, file_names[5]); name = file_names[5]) # laplaces_equation
    end

    @testset verbose = true "Verification" begin
        dir = joinpath(dirname(@__DIR__), "docs", "src", "literate_verification")
        files = readdir(dir)
        file_names = [
            "amr_convergence.jl",
            "brio_wu_verification.jl",
            "conservation_verification.jl",
            "flux_balance_verification.jl",
            "grmhd_convergence.jl",
            "mhd_convergence.jl",
            "mhd_divb_verification.jl",
            "mms_convergence.jl",
            "ns_convergence.jl",
            "orszag_tang_verification.jl",
            "passive_scalar_convergence.jl",
            "poisson_convergence.jl",
            "premixed_flame_1d.jl",
            "smooth_advection_convergence.jl",
            "source_term_convergence.jl",
            "species_conservation.jl",
            "sod_grid_convergence.jl",
            "srmhd_convergence.jl",
            "toro_riemann_tests.jl",
        ]
        @test length(files) == length(file_names)
        for i in eachindex(file_names)
            safe_include(joinpath(dir, file_names[i]); name = file_names[i])
        end
    end

    @testset verbose = true "Aqua" begin
        Aqua.test_all(FiniteVolumeMethod; ambiguities = false, project_extras = false, unbound_args = false) # don't care about julia < 1.2
        Aqua.test_unbound_args(FiniteVolumeMethod; broken = true) # Val{N} pattern in AMR constructors is a known false positive
        Aqua.test_ambiguities(FiniteVolumeMethod) # don't pick up Base and Core...
    end

    @testset verbose = true "Explicit Imports" begin
        safe_include("explicit_imports.jl")
    end
end
