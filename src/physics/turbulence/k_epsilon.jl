@doc raw"""
    k-ε Turbulence Model Module

This module provides the standard k-ε turbulence model for use with `FVMSystem`.
The model solves two coupled transport equations for turbulent kinetic energy (k)
and its dissipation rate (ε).

## Transport Equations

```math
\pdv{k}{t} + \vb u \cdot \grad k = \div\left[\left(\nu + \frac{\nu_t}{\sigma_k}\right)\grad k\right] + P_k - \varepsilon
```

```math
\pdv{\varepsilon}{t} + \vb u \cdot \grad \varepsilon = \div\left[\left(\nu + \frac{\nu_t}{\sigma_\varepsilon}\right)\grad \varepsilon\right] + C_{1\varepsilon}\frac{\varepsilon}{k}P_k - C_{2\varepsilon}\frac{\varepsilon^2}{k}
```

where:
- $k$ = turbulent kinetic energy
- $\varepsilon$ = turbulent dissipation rate
- $\nu_t = C_\mu k^2/\varepsilon$ = turbulent (eddy) viscosity
- $P_k$ = production of turbulent kinetic energy

## Model Constants (Standard k-ε)

| Constant | Value | Description |
|----------|-------|-------------|
| $C_\mu$ | 0.09 | Eddy viscosity coefficient |
| $\sigma_k$ | 1.0 | Turbulent Prandtl number for k |
| $\sigma_\varepsilon$ | 1.3 | Turbulent Prandtl number for ε |
| $C_{1\varepsilon}$ | 1.44 | Production coefficient |
| $C_{2\varepsilon}$ | 1.92 | Destruction coefficient |

## Usage with FVMSystem

The k-ε model is implemented as utilities that help construct an `FVMSystem` with
the appropriate flux functions, source terms, and boundary conditions.

## Common Applications

1. **Pipe and channel flows**: Turbulent boundary layers
2. **External aerodynamics**: Wakes and mixing layers
3. **Industrial flows**: Heat exchangers, combustion
4. **Environmental flows**: Atmospheric boundary layers, ocean mixing
"""

"""
    StandardKEpsilon{T<:Real}

Standard k-ε turbulence model with default or custom coefficients.

# Fields
- `C_mu::T`: Eddy viscosity coefficient (default: 0.09)
- `sigma_k::T`: Turbulent Prandtl number for k (default: 1.0)
- `sigma_epsilon::T`: Turbulent Prandtl number for ε (default: 1.3)
- `C1_epsilon::T`: Production coefficient (default: 1.44)
- `C2_epsilon::T`: Destruction coefficient (default: 1.92)
- `kappa::T`: von Kármán constant (default: 0.41)

# Example
```julia
# Default model
model = StandardKEpsilon()

# Custom coefficients for realizable k-ε
model = StandardKEpsilon(;
    C_mu = 0.09,
    C1_epsilon = 1.44,
    C2_epsilon = 1.9
)
```
"""
struct StandardKEpsilon{T<:Real}
    C_mu::T
    sigma_k::T
    sigma_epsilon::T
    C1_epsilon::T
    C2_epsilon::T
    kappa::T
end

function StandardKEpsilon(;
        C_mu::Real = 0.09,
        sigma_k::Real = 1.0,
        sigma_epsilon::Real = 1.3,
        C1_epsilon::Real = 1.44,
        C2_epsilon::Real = 1.92,
        kappa::Real = 0.41
    )
    T = promote_type(
        typeof(C_mu), typeof(sigma_k), typeof(sigma_epsilon),
        typeof(C1_epsilon), typeof(C2_epsilon), typeof(kappa)
    )
    return StandardKEpsilon{T}(T(C_mu), T(sigma_k), T(sigma_epsilon), T(C1_epsilon), T(C2_epsilon), T(kappa))
end

@doc raw"""
    compute_turbulent_viscosity(model::StandardKEpsilon, k, epsilon; rho=1.0)

Compute turbulent (eddy) viscosity from k and ε.

```math
\nu_t = C_\mu \frac{k^2}{\varepsilon}
```

For dimensional problems with density:
```math
\mu_t = \rho C_\mu \frac{k^2}{\varepsilon}
```

# Arguments
- `model`: The k-ε model
- `k`: Turbulent kinetic energy (scalar or array)
- `epsilon`: Turbulent dissipation rate (scalar or array)
- `rho`: Density (default: 1.0 for kinematic viscosity)

# Returns
Turbulent viscosity (same shape as inputs).
"""
function compute_turbulent_viscosity(model::StandardKEpsilon, k::Real, epsilon::Real; rho::Real=1.0)
    # Safeguard against division by zero
    eps_safe = max(epsilon, 1e-10)
    k_safe = max(k, 0.0)
    return rho * model.C_mu * k_safe^2 / eps_safe
end

function compute_turbulent_viscosity(model::StandardKEpsilon, k::AbstractVector, epsilon::AbstractVector; rho=1.0)
    nu_t = similar(k)
    rho_vec = rho isa Real ? fill(rho, length(k)) : rho
    @inbounds for i in eachindex(k)
        eps_safe = max(epsilon[i], 1e-10)
        k_safe = max(k[i], 0.0)
        nu_t[i] = rho_vec[i] * model.C_mu * k_safe^2 / eps_safe
    end
    return nu_t
end

@doc raw"""
    compute_strain_rate_magnitude(grad_u, grad_v)

Compute the magnitude of the strain rate tensor in 2D.

```math
|S| = \sqrt{2 S_{ij} S_{ij}} = \sqrt{2(S_{xx}^2 + S_{yy}^2 + 2S_{xy}^2)}
```

where $S_{ij} = \frac{1}{2}\left(\pdv{u_i}{x_j} + \pdv{u_j}{x_i}\right)$.

# Arguments
- `grad_u`: Tuple `(∂u/∂x, ∂u/∂y)` - gradient of x-velocity
- `grad_v`: Tuple `(∂v/∂x, ∂v/∂y)` - gradient of y-velocity

# Returns
Strain rate magnitude `|S|`.
"""
function compute_strain_rate_magnitude(grad_u::Tuple, grad_v::Tuple)
    dudx, dudy = grad_u
    dvdx, dvdy = grad_v

    S_xx = dudx
    S_yy = dvdy
    S_xy = 0.5 * (dudy + dvdx)

    # |S|² = 2 * S_ij * S_ij
    S_sq = 2.0 * (S_xx^2 + S_yy^2 + 2.0 * S_xy^2)
    return sqrt(max(S_sq, 0.0))
end

@doc raw"""
    compute_production(nu_t, strain_magnitude)

Compute the production term for the k equation.

```math
P_k = \nu_t |S|^2 = 2 \nu_t S_{ij} S_{ij}
```

# Arguments
- `nu_t`: Turbulent viscosity
- `strain_magnitude`: Strain rate magnitude `|S|`

# Returns
Production term `P_k`.
"""
function compute_production(nu_t::Real, strain_magnitude::Real)
    return nu_t * strain_magnitude^2
end

@doc raw"""
    k_source_function(x, y, t, (k, epsilon), p)

Source function for the k equation in an `FVMSystem`.

The source term is:
```math
S_k = P_k - \varepsilon
```

where `P_k` is provided through the parameters.

This should be used with linearization for stability:
- Linearized: `S_k = P_k - (ε/k)·k`

# Parameters struct
- `p.Pk`: Production field (vector, one per node)
- `p.node_index`: Current node index (set dynamically)
"""
function k_source_function(x, y, t, u_tuple, p)
    k, epsilon = u_tuple

    # Get production at this point (from pre-computed field)
    Pk = haskey(p, :Pk) && !isnothing(p.Pk) ? p.Pk : 0.0

    # Source: Pk - epsilon
    # For stability, treat -epsilon as implicit: S = Pk - (epsilon/k)*k
    # But for explicit evaluation, just return Pk - epsilon
    return Pk - epsilon
end

@doc raw"""
    epsilon_source_function(x, y, t, (k, epsilon), p)

Source function for the ε equation in an `FVMSystem`.

The source term is:
```math
S_\varepsilon = C_{1\varepsilon} \frac{\varepsilon}{k} P_k - C_{2\varepsilon} \frac{\varepsilon^2}{k}
```

# Parameters struct
- `p.model`: StandardKEpsilon model
- `p.Pk`: Production field
"""
function epsilon_source_function(x, y, t, u_tuple, p)
    k, epsilon = u_tuple

    model = p.model
    Pk = haskey(p, :Pk) && !isnothing(p.Pk) ? p.Pk : 0.0

    # Safeguards
    k_safe = max(k, 1e-10)
    eps_by_k = epsilon / k_safe

    # Source terms
    S_C = model.C1_epsilon * eps_by_k * Pk           # Production
    S_P = -model.C2_epsilon * eps_by_k * epsilon     # Destruction

    return S_C + S_P
end

@doc raw"""
    make_k_diffusion_function(model::StandardKEpsilon, nu, nu_t_field)

Create a diffusion function for the k equation.

```math
\Gamma_k = \nu + \frac{\nu_t}{\sigma_k}
```

# Arguments
- `model`: The k-ε model
- `nu`: Laminar kinematic viscosity (scalar or function)
- `nu_t_field`: Turbulent viscosity field (vector, indexed by node)

# Returns
A diffusion function `(x, y, t, u, p) -> Γk` suitable for `FVMProblem`.
"""
function make_k_diffusion_function(model::StandardKEpsilon, nu::Real, nu_t_field::AbstractVector)
    return function(x, y, t, u, p)
        # Get node index from parameters
        i = haskey(p, :node_index) ? p.node_index : 1
        nu_t = i <= length(nu_t_field) ? nu_t_field[i] : 0.0
        return nu + nu_t / model.sigma_k
    end
end

@doc raw"""
    make_epsilon_diffusion_function(model::StandardKEpsilon, nu, nu_t_field)

Create a diffusion function for the ε equation.

```math
\Gamma_\varepsilon = \nu + \frac{\nu_t}{\sigma_\varepsilon}
```

# Arguments
- `model`: The k-ε model
- `nu`: Laminar kinematic viscosity
- `nu_t_field`: Turbulent viscosity field (vector)

# Returns
A diffusion function `(x, y, t, u, p) -> Γε` suitable for `FVMProblem`.
"""
function make_epsilon_diffusion_function(model::StandardKEpsilon, nu::Real, nu_t_field::AbstractVector)
    return function(x, y, t, u, p)
        i = haskey(p, :node_index) ? p.node_index : 1
        nu_t = i <= length(nu_t_field) ? nu_t_field[i] : 0.0
        return nu + nu_t / model.sigma_epsilon
    end
end

@doc raw"""
    compute_friction_velocity(u_tan, y, nu; kappa=0.41, E=9.8, max_iter=10, tol=1e-5)

Compute friction velocity `u_τ` using the law of the wall.

For the log layer:
```math
u^+ = \frac{1}{\kappa} \ln(E y^+)
```

where $u^+ = u/u_\tau$ and $y^+ = y u_\tau / \nu$.

# Arguments
- `u_tan`: Tangential velocity magnitude
- `y`: Distance from wall
- `nu`: Kinematic viscosity
- `kappa`: von Kármán constant (default: 0.41)
- `E`: Wall function constant (default: 9.8)

# Returns
Friction velocity `u_τ`.
"""
function compute_friction_velocity(u_tan::Real, y::Real, nu::Real;
        kappa::Real = 0.41, E::Real = 9.8, max_iter::Int = 10, tol::Real = 1e-5)

    # Initial guess from viscous sublayer
    u_tau = sqrt(abs(nu * u_tan / max(y, 1e-10)))

    for _ in 1:max_iter
        y_plus = y * u_tau / nu

        if y_plus < 11.225
            # Viscous sublayer - linear profile
            break
        else
            # Log layer - Newton-Raphson iteration
            f = u_tan / u_tau - (1.0 / kappa) * log(E * y_plus)
            df = -u_tan / u_tau^2 - (1.0 / kappa) / u_tau

            delta = f / df
            u_tau = max(u_tau - delta, 1e-10)

            if abs(delta) < tol
                break
            end
        end
    end

    return u_tau
end

@doc raw"""
    k_wall_value(u_tau, C_mu=0.09)

Compute equilibrium k value at the wall for wall functions.

```math
k_w = \frac{u_\tau^2}{\sqrt{C_\mu}}
```

# Arguments
- `u_tau`: Friction velocity
- `C_mu`: Model constant (default: 0.09)

# Returns
Wall value of k.
"""
function k_wall_value(u_tau::Real; C_mu::Real = 0.09)
    return u_tau^2 / sqrt(C_mu)
end

@doc raw"""
    epsilon_wall_value(u_tau, y, kappa=0.41, C_mu=0.09)

Compute equilibrium ε value at the wall for wall functions.

```math
\varepsilon_w = \frac{C_\mu^{3/4} k^{3/2}}{\kappa y} = \frac{u_\tau^3}{\kappa y}
```

# Arguments
- `u_tau`: Friction velocity
- `y`: Distance from wall
- `kappa`: von Kármán constant (default: 0.41)
- `C_mu`: Model constant (default: 0.09)

# Returns
Wall value of ε.
"""
function epsilon_wall_value(u_tau::Real, y::Real; kappa::Real = 0.41, C_mu::Real = 0.09)
    return C_mu^0.75 * k_wall_value(u_tau; C_mu=C_mu)^1.5 / (kappa * max(y, 1e-10))
end

@doc raw"""
    TurbulentWallBC{T}

Parameters for turbulent wall boundary conditions.

# Fields
- `roughness::T`: Surface roughness (default: 0.0 for smooth walls)
- `C_mu::T`: Model constant
- `kappa::T`: von Kármán constant
"""
struct TurbulentWallBC{T<:Real}
    roughness::T
    C_mu::T
    kappa::T
end

TurbulentWallBC(; roughness::Real = 0.0, C_mu::Real = 0.09, kappa::Real = 0.41) =
    TurbulentWallBC(promote(roughness, C_mu, kappa)...)

@doc raw"""
    KappaOmegaSST{T}

k-ω SST turbulence model (Shear Stress Transport) coefficients.

This is a more advanced model that blends k-ε and k-ω models.
Only the coefficients are provided here; implementation is deferred.

# Fields
- `a1::T`: Coefficient for SST limiter (default: 0.31)
- `beta_star::T`: k equation destruction coefficient (default: 0.09)
- `sigma_k1::T`, `sigma_k2::T`: Blending coefficients for k diffusion
- `sigma_omega1::T`, `sigma_omega2::T`: Blending coefficients for ω diffusion
"""
struct KappaOmegaSST{T<:Real}
    a1::T
    beta_star::T
    sigma_k1::T
    sigma_k2::T
    sigma_omega1::T
    sigma_omega2::T
    beta1::T
    beta2::T
    kappa::T
end

function KappaOmegaSST(;
        a1::Real = 0.31,
        beta_star::Real = 0.09,
        sigma_k1::Real = 0.85,
        sigma_k2::Real = 1.0,
        sigma_omega1::Real = 0.5,
        sigma_omega2::Real = 0.856,
        beta1::Real = 0.075,
        beta2::Real = 0.0828,
        kappa::Real = 0.41
    )
    T = promote_type(
        typeof(a1), typeof(beta_star), typeof(sigma_k1), typeof(sigma_k2),
        typeof(sigma_omega1), typeof(sigma_omega2), typeof(beta1), typeof(beta2), typeof(kappa)
    )
    return KappaOmegaSST{T}(
        T(a1), T(beta_star), T(sigma_k1), T(sigma_k2),
        T(sigma_omega1), T(sigma_omega2), T(beta1), T(beta2), T(kappa)
    )
end
