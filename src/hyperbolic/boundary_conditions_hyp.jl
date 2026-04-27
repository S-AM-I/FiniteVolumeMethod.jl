using StaticArrays: SVector

"""
    AbstractHyperbolicBC

Abstract supertype for ghost-cell boundary conditions in the hyperbolic solver.
Subtypes the Stage 1d umbrella `AbstractFVMBoundaryCondition`.
"""
abstract type AbstractHyperbolicBC <: AbstractFVMBoundaryCondition end

"""
    apply_bc!(U_ghost, bc, law, U_interior, t) -> Nothing

Fill the ghost cells `U_ghost` based on the boundary condition, the conservation law,
and the interior cell values.

For 1D problems, `U` is padded as:
  `U[1:ng]` = left ghost, `U[ng+1:ncells+ng]` = interior, `U[ncells+ng+1:ncells+2*ng]` = right ghost.

Each BC fills its side's ghost cells.
"""
function apply_bc! end

# ============================================================
# Transmissive (Outflow / Zero-Gradient) BC
# ============================================================

"""
    TransmissiveBC <: AbstractHyperbolicBC

Zero-gradient (outflow) boundary condition. Ghost cell values are copied from
the nearest interior cells (extrapolation of order 0).
"""
struct TransmissiveBC <: AbstractHyperbolicBC end

function apply_bc_left!(U::AbstractVector, ::TransmissiveBC, law, ncells::Int, ng::Int, t)
    first_interior = ng + 1
    for g in 1:ng
        U[ng + 1 - g] = U[first_interior]
    end
    return nothing
end

function apply_bc_right!(U::AbstractVector, ::TransmissiveBC, law, ncells::Int, ng::Int, t)
    last_interior = ncells + ng
    for g in 1:ng
        U[last_interior + g] = U[last_interior]
    end
    return nothing
end

# ============================================================
# Reflective (Slip Wall) BC
# ============================================================

"""
    ReflectiveBC <: AbstractHyperbolicBC

Reflective (slip wall) boundary condition. The normal velocity component
is negated in the ghost cells while density, pressure, and tangential
velocities are copied.
"""
struct ReflectiveBC <: AbstractHyperbolicBC end

function apply_bc_left!(U::AbstractVector, ::ReflectiveBC, law::EulerEquations{1}, ncells::Int, ng::Int, t)
    # Reflect: negate velocity
    for g in 1:ng
        u_int = U[ng + g]  # interior cell g (1st, 2nd, etc. from boundary)
        w = conserved_to_primitive(law, u_int)
        # Mirror: rho same, v negated, P same
        w_ghost = SVector(w[1], -w[2], w[3])
        U[ng + 1 - g] = primitive_to_conserved(law, w_ghost)
    end
    return nothing
end

function apply_bc_right!(U::AbstractVector, ::ReflectiveBC, law::EulerEquations{1}, ncells::Int, ng::Int, t)
    last_interior = ncells + ng
    for g in 1:ng
        u_int = U[last_interior + 1 - g]  # interior cell g from boundary
        w = conserved_to_primitive(law, u_int)
        w_ghost = SVector(w[1], -w[2], w[3])
        U[last_interior + g] = primitive_to_conserved(law, w_ghost)
    end
    return nothing
end

# ============================================================
# Inflow BC
# ============================================================

"""
    InflowBC{N, FT} <: AbstractHyperbolicBC

Prescribes all primitive variables at the boundary.

# Fields
- `state::SVector{N, FT}`: Prescribed primitive state `[rho, v, P]` (1D) or `[rho, vx, vy, P]` (2D).
"""
struct InflowBC{N, FT} <: AbstractHyperbolicBC
    state::SVector{N, FT}
end

function apply_bc_left!(U::AbstractVector, bc::InflowBC, law, ncells::Int, ng::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for g in 1:ng
        U[ng + 1 - g] = u_bc
    end
    return nothing
end

function apply_bc_right!(U::AbstractVector, bc::InflowBC, law, ncells::Int, ng::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    last_interior = ncells + ng
    for g in 1:ng
        U[last_interior + g] = u_bc
    end
    return nothing
end

# ============================================================
# Periodic BC
# ============================================================

"""
    PeriodicHyperbolicBC <: AbstractHyperbolicBC

Periodic boundary condition: the left ghost cells are filled from the right
interior cells and vice versa.
"""
struct PeriodicHyperbolicBC <: AbstractHyperbolicBC end

function apply_periodic_bcs!(U::AbstractVector, law, ncells::Int, ng::Int, t)
    first_interior = ng + 1
    last_interior = ncells + ng
    # Left ghosts from right interior
    for g in 1:ng
        U[ng + 1 - g] = U[last_interior + 1 - g]
    end
    # Right ghosts from left interior
    for g in 1:ng
        U[last_interior + g] = U[first_interior + g - 1]
    end
    return nothing
end

# ============================================================
# Dirichlet (fixed state) BC -- for Sod-like problems
# ============================================================

"""
    DirichletHyperbolicBC{N, FT} <: AbstractHyperbolicBC

Fixed-state boundary condition for hyperbolic problems. The ghost cells
are set to maintain the prescribed primitive state at the boundary.

# Fields
- `state::SVector{N, FT}`: Prescribed primitive state.
"""
struct DirichletHyperbolicBC{N, FT} <: AbstractHyperbolicBC
    state::SVector{N, FT}
end

function apply_bc_left!(U::AbstractVector, bc::DirichletHyperbolicBC, law, ncells::Int, ng::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    # Set ghost cells to reflect the boundary state
    for g in 1:ng
        U[ng + 1 - g] = u_bc
    end
    return nothing
end

function apply_bc_right!(U::AbstractVector, bc::DirichletHyperbolicBC, law, ncells::Int, ng::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    last_interior = ncells + ng
    for g in 1:ng
        U[last_interior + g] = u_bc
    end
    return nothing
end
