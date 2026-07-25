```@meta
EditURL = "https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_v3/11_solid_mechanics_beam.jl"
```

# Tutorial 11 — Linear Elasticity (Compressed Block)

Demonstrates the v3 small-strain linear-elasticity solver on a
2D block. We clamp the left edge, prescribe a uniform horizontal
displacement on the right, and verify the resulting displacement
field is (nearly) affine.

Runtime budget: ~2 s on a laptop (16×16 mesh, 80 Gauss-Seidel sweeps).

Run with:

```bash
julia --project=docs docs/src/literate_v3/11_solid_mechanics_beam.jl
```

What it demonstrates:
- `IsotropicElastic(; E, nu)` for the material law
- Dirichlet displacement BCs as a `Dict{Symbol, SVector{2, Float64}}`
- `solve_linear_elasticity` and reading out `result.displacement`
- The Euler-Bernoulli tip-deflection helper
  `cantilever_tip_deflection(E, I, L, P)` for a sanity check

````julia
using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: IsotropicElastic, cantilever_tip_deflection
using LinearSolve
using StaticArrays
using Printf
````

Located relative to the installed package rather than to this file, so the
path resolves both when run as a script and when Literate executes it from
the generated-docs directory.

````julia
include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))
````

`solve_linear_elasticity` and `SolidProperties` are not exported.

````julia
const solve_linear_elasticity = FiniteVolumeMethod.solve_linear_elasticity
const SolidProperties = FiniteVolumeMethod.SolidProperties

const Lx = 1.0
const Ly = 1.0
mesh = build_cartesian_unstructured_mesh(16, 16, Lx, Ly)
````

Steel-like material. λ and μ are derived internally.

````julia
props = SolidProperties(; rho = 1.0, E = 1.0e6, nu = 0.3)
````

Pre-strain block: left edge fixed at u = 0, right edge pushed by a.
Top and bottom get the mean-axial displacement so the affine field
u = (a · x, 0) is the exact solution.

````julia
const a = 0.01
bcs = Dict{Symbol, SVector{2, Float64}}(
    :left => SVector(0.0, 0.0),
    :right => SVector(a * Lx, 0.0),
    :bottom => SVector(a * Lx / 2, 0.0),
    :top => SVector(a * Lx / 2, 0.0),
)

result = solve_linear_elasticity(
    mesh, props, bcs; max_iterations = 80, tolerance = 1.0e-10,
)
````

Compute the maximum departure from the analytical affine solution
u_exact(x) = (a · x, 0).

````julia
function affine_displacement_error(mesh, displacement, a)
    err = 0.0
    for c in 1:length(mesh.cell_volumes)
        x_c = mesh.cell_centers[1, c]
        u_ex = SVector(a * x_c, 0.0)
        u_num = displacement[c]
        err = max(err, abs(u_num[1] - u_ex[1]), abs(u_num[2] - u_ex[2]))
    end
    return err
end

max_err = affine_displacement_error(mesh, result.displacement, a)
````

Euler-Bernoulli cantilever tip deflection helper (symbolic, not used in
the above solve): δ = P L³ / (3 E I).

````julia
E = 2.1e11
I = 1.0e-6
L = 1.0
P = 100.0
delta_tip = cantilever_tip_deflection(E, I, L, P)

println("=== Linear elasticity — affine displacement verification ===")
@printf "iterations          : %d\n" result.iterations
@printf "converged           : %s\n" result.converged
@printf "prescribed strain a : %.4f\n" a
@printf "max abs error u     : %.2e  (analytical = a·x)\n" max_err
@printf "—— Euler-Bernoulli cantilever tip deflection helper ——\n"
@printf "δ = P L³ / (3 E I)  : %.4e m\n" delta_tip
````

Manifest feature  : stage7a.linear_elasticity (experimental)
V&V tests         : test/v_and_v_linear_elasticity.jl, test/stage7_coupled.jl

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/cx-xd/FiniteVolumeMethod.jl/tree/main/docs/src/literate_v3/11_solid_mechanics_beam.jl).

```julia
using FiniteVolumeMethod
using FiniteVolumeMethod.Experimental: IsotropicElastic, cantilever_tip_deflection
using LinearSolve
using StaticArrays
using Printf

include(joinpath(dirname(dirname(pathof(FiniteVolumeMethod))), "test", "TestHelpers.jl"))

const solve_linear_elasticity = FiniteVolumeMethod.solve_linear_elasticity
const SolidProperties = FiniteVolumeMethod.SolidProperties

const Lx = 1.0
const Ly = 1.0
mesh = build_cartesian_unstructured_mesh(16, 16, Lx, Ly)

props = SolidProperties(; rho = 1.0, E = 1.0e6, nu = 0.3)

const a = 0.01
bcs = Dict{Symbol, SVector{2, Float64}}(
    :left => SVector(0.0, 0.0),
    :right => SVector(a * Lx, 0.0),
    :bottom => SVector(a * Lx / 2, 0.0),
    :top => SVector(a * Lx / 2, 0.0),
)

result = solve_linear_elasticity(
    mesh, props, bcs; max_iterations = 80, tolerance = 1.0e-10,
)

function affine_displacement_error(mesh, displacement, a)
    err = 0.0
    for c in 1:length(mesh.cell_volumes)
        x_c = mesh.cell_centers[1, c]
        u_ex = SVector(a * x_c, 0.0)
        u_num = displacement[c]
        err = max(err, abs(u_num[1] - u_ex[1]), abs(u_num[2] - u_ex[2]))
    end
    return err
end

max_err = affine_displacement_error(mesh, result.displacement, a)

E = 2.1e11
I = 1.0e-6
L = 1.0
P = 100.0
delta_tip = cantilever_tip_deflection(E, I, L, P)

println("=== Linear elasticity — affine displacement verification ===")
@printf "iterations          : %d\n" result.iterations
@printf "converged           : %s\n" result.converged
@printf "prescribed strain a : %.4f\n" a
@printf "max abs error u     : %.2e  (analytical = a·x)\n" max_err
@printf "—— Euler-Bernoulli cantilever tip deflection helper ——\n"
@printf "δ = P L³ / (3 E I)  : %.4e m\n" delta_tip
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

