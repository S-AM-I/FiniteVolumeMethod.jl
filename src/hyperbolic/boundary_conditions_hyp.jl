using StaticArrays: SVector

"""
    AbstractHyperbolicBC

Abstract supertype for ghost-cell boundary conditions in the hyperbolic solver.
"""
abstract type AbstractHyperbolicBC end

"""
    apply_bc!(U_ghost, bc, law, U_interior, t) -> Nothing

Fill the ghost cells `U_ghost` based on the boundary condition, the conservation law,
and the interior cell values.

For 1D problems, `U` is padded as:
  `U[1:2]` = left ghost, `U[3:ncells+2]` = interior, `U[ncells+3:ncells+4]` = right ghost.

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

function apply_bc_left!(U::AbstractVector, ::TransmissiveBC, law, ncells::Int, t)
    U[2] = U[3]      # first ghost = first interior
    U[1] = U[3]      # second ghost = first interior
    return nothing
end

function apply_bc_right!(U::AbstractVector, ::TransmissiveBC, law, ncells::Int, t)
    U[ncells + 3] = U[ncells + 2]  # first ghost = last interior
    U[ncells + 4] = U[ncells + 2]  # second ghost = last interior
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

function apply_bc_left!(U::AbstractVector, ::ReflectiveBC, law::EulerEquations{1}, ncells::Int, t)
    # Reflect: negate velocity
    u1 = U[3]  # first interior cell
    u2 = U[4]  # second interior cell
    w1 = conserved_to_primitive(law, u1)
    w2 = conserved_to_primitive(law, u2)
    # Mirror: ρ same, v negated, P same
    w1_ghost = SVector(w1[1], -w1[2], w1[3])
    w2_ghost = SVector(w2[1], -w2[2], w2[3])
    U[2] = primitive_to_conserved(law, w1_ghost)
    U[1] = primitive_to_conserved(law, w2_ghost)
    return nothing
end

function apply_bc_right!(U::AbstractVector, ::ReflectiveBC, law::EulerEquations{1}, ncells::Int, t)
    u1 = U[ncells + 2]  # last interior cell
    u2 = U[ncells + 1]  # second-to-last interior cell
    w1 = conserved_to_primitive(law, u1)
    w2 = conserved_to_primitive(law, u2)
    w1_ghost = SVector(w1[1], -w1[2], w1[3])
    w2_ghost = SVector(w2[1], -w2[2], w2[3])
    U[ncells + 3] = primitive_to_conserved(law, w1_ghost)
    U[ncells + 4] = primitive_to_conserved(law, w2_ghost)
    return nothing
end

# ============================================================
# Inflow BC
# ============================================================

"""
    InflowBC{N, FT} <: AbstractHyperbolicBC

Prescribes all primitive variables at the boundary.

# Fields
- `state::SVector{N, FT}`: Prescribed primitive state `[ρ, v, P]` (1D) or `[ρ, vx, vy, P]` (2D).
"""
struct InflowBC{N, FT} <: AbstractHyperbolicBC
    state::SVector{N, FT}
end

function apply_bc_left!(U::AbstractVector, bc::InflowBC, law, ncells::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    U[2] = u_bc
    U[1] = u_bc
    return nothing
end

function apply_bc_right!(U::AbstractVector, bc::InflowBC, law, ncells::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    U[ncells + 3] = u_bc
    U[ncells + 4] = u_bc
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

function apply_periodic_bcs!(U::AbstractVector, law, ncells::Int, t)
    # Left ghosts from right interior
    U[2] = U[ncells + 2]  # ghost 1 = last interior
    U[1] = U[ncells + 1]  # ghost 2 = second-to-last interior
    # Right ghosts from left interior
    U[ncells + 3] = U[3]  # ghost 1 = first interior
    U[ncells + 4] = U[4]  # ghost 2 = second interior
    return nothing
end

# ============================================================
# Dirichlet (fixed state) BC — for Sod-like problems
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

function apply_bc_left!(U::AbstractVector, bc::DirichletHyperbolicBC, law, ncells::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    # Set ghost cells to reflect the boundary state
    U[2] = u_bc
    U[1] = u_bc
    return nothing
end

function apply_bc_right!(U::AbstractVector, bc::DirichletHyperbolicBC, law, ncells::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    U[ncells + 3] = u_bc
    U[ncells + 4] = u_bc
    return nothing
end
