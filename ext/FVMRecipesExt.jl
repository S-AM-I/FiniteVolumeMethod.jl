module FVMRecipesExt

using FiniteVolumeMethod
using RecipesBase

"""
    plot(sol, prob::HyperbolicProblem; vars=nothing, tidx=length(sol.t))

Plot 1D hyperbolic solution fields.  By default plots all conserved
variables at the final time step.  Pass `vars=[:rho, :E]` to select
specific fields and `tidx=1` for the initial condition.
"""
@recipe function f(
        sol::AbstractVector, prob::FiniteVolumeMethod.HyperbolicProblem;
        vars = nothing, tidx = nothing,
    )
    law = prob.law
    mesh = prob.mesh
    N = FiniteVolumeMethod.nvariables(law)
    names = Symbol.(FiniteVolumeMethod.variable_names(law))
    xs = mesh.cell_centers

    idx = tidx === nothing ? length(sol) : tidx
    u = sol isa AbstractVector{<:AbstractVector} ? sol[idx] : sol

    selected = vars === nothing ? names : [v isa Symbol ? v : Symbol(v) for v in vars]

    layout --> (1, length(selected))
    for (k, name) in enumerate(selected)
        vi = findfirst(==(name), names)
        vi === nothing && error("Unknown variable: $name. Available: $names")
        field = @view u[vi:N:end]
        @series begin
            subplot := k
            label --> String(name)
            xlabel --> "x"
            ylabel --> String(name)
            seriestype --> :line
            xs, field
        end
    end
end

"""
    plot(sol, prob::HyperbolicProblem2D; var=:rho, tidx=length(sol.t))

Plot a 2D hyperbolic solution field as a heatmap.  Defaults to density
at the final time step.
"""
@recipe function f(
        sol::AbstractVector, prob::FiniteVolumeMethod.HyperbolicProblem2D;
        var = :rho, tidx = nothing,
    )
    law = prob.law
    mesh = prob.mesh
    N = FiniteVolumeMethod.nvariables(law)
    names = Symbol.(FiniteVolumeMethod.variable_names(law))
    nx, ny = mesh.nx, mesh.ny

    idx = tidx === nothing ? length(sol) : tidx
    u = sol isa AbstractVector{<:AbstractVector} ? sol[idx] : sol

    name = var isa Symbol ? var : Symbol(var)
    vi = findfirst(==(name), names)
    vi === nothing && error("Unknown variable: $name. Available: $names")

    field_flat = @view u[vi:N:(nx * ny * N)]
    field_2d = reshape(field_flat, nx, ny)

    # Cell center coordinates
    dx, dy = mesh.dx, mesh.dy
    xs = range(mesh.xmin + dx / 2; step = dx, length = nx)
    ys = range(mesh.ymin + dy / 2; step = dy, length = ny)

    seriestype --> :heatmap
    xlabel --> "x"
    ylabel --> "y"
    title --> String(name)
    xs, ys, permutedims(field_2d)
end

end # module
