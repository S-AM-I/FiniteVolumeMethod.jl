# ============================================================
# 2D Ghost-Cell Boundary Conditions
# ============================================================
#
# The 2D solution is stored in a padded matrix U[i, j] where:
#   i = 1:2           -> left ghost columns
#   i = 3:nx+2        -> interior columns
#   i = nx+3:nx+4     -> right ghost columns
#   j = 1:2           -> bottom ghost rows
#   j = 3:ny+2        -> interior rows
#   j = ny+3:ny+4     -> top ghost rows
#
# Interior cell (ix, iy) (1-based) maps to U[ix+2, iy+2].

# ============================================================
# TransmissiveBC (2D)
# ============================================================

function apply_bc_2d_left!(U::AbstractMatrix, ::TransmissiveBC, law, nx::Int, ny::Int, t)
    for j in 1:(ny + 4)
        U[2, j] = U[3, j]
        U[1, j] = U[3, j]
    end
    return nothing
end

function apply_bc_2d_right!(U::AbstractMatrix, ::TransmissiveBC, law, nx::Int, ny::Int, t)
    for j in 1:(ny + 4)
        U[nx + 3, j] = U[nx + 2, j]
        U[nx + 4, j] = U[nx + 2, j]
    end
    return nothing
end

function apply_bc_2d_bottom!(U::AbstractMatrix, ::TransmissiveBC, law, nx::Int, ny::Int, t)
    for i in 1:(nx + 4)
        U[i, 2] = U[i, 3]
        U[i, 1] = U[i, 3]
    end
    return nothing
end

function apply_bc_2d_top!(U::AbstractMatrix, ::TransmissiveBC, law, nx::Int, ny::Int, t)
    for i in 1:(nx + 4)
        U[i, ny + 3] = U[i, ny + 2]
        U[i, ny + 4] = U[i, ny + 2]
    end
    return nothing
end

# ============================================================
# ReflectiveBC (2D Euler)
# ============================================================

# Left wall: negate vx
function apply_bc_2d_left!(U::AbstractMatrix, ::ReflectiveBC, law::EulerEquations{2}, nx::Int, ny::Int, t)
    for j in 1:(ny + 4)
        w1 = conserved_to_primitive(law, U[3, j])
        w2 = conserved_to_primitive(law, U[4, j])
        # Negate vx (component 2), keep vy (component 3)
        U[2, j] = primitive_to_conserved(law, SVector(w1[1], -w1[2], w1[3], w1[4]))
        U[1, j] = primitive_to_conserved(law, SVector(w2[1], -w2[2], w2[3], w2[4]))
    end
    return nothing
end

# Right wall: negate vx
function apply_bc_2d_right!(U::AbstractMatrix, ::ReflectiveBC, law::EulerEquations{2}, nx::Int, ny::Int, t)
    for j in 1:(ny + 4)
        w1 = conserved_to_primitive(law, U[nx + 2, j])
        w2 = conserved_to_primitive(law, U[nx + 1, j])
        U[nx + 3, j] = primitive_to_conserved(law, SVector(w1[1], -w1[2], w1[3], w1[4]))
        U[nx + 4, j] = primitive_to_conserved(law, SVector(w2[1], -w2[2], w2[3], w2[4]))
    end
    return nothing
end

# Bottom wall: negate vy
function apply_bc_2d_bottom!(U::AbstractMatrix, ::ReflectiveBC, law::EulerEquations{2}, nx::Int, ny::Int, t)
    for i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, 3])
        w2 = conserved_to_primitive(law, U[i, 4])
        # Negate vy (component 3), keep vx (component 2)
        U[i, 2] = primitive_to_conserved(law, SVector(w1[1], w1[2], -w1[3], w1[4]))
        U[i, 1] = primitive_to_conserved(law, SVector(w2[1], w2[2], -w2[3], w2[4]))
    end
    return nothing
end

# Top wall: negate vy
function apply_bc_2d_top!(U::AbstractMatrix, ::ReflectiveBC, law::EulerEquations{2}, nx::Int, ny::Int, t)
    for i in 1:(nx + 4)
        w1 = conserved_to_primitive(law, U[i, ny + 2])
        w2 = conserved_to_primitive(law, U[i, ny + 1])
        U[i, ny + 3] = primitive_to_conserved(law, SVector(w1[1], w1[2], -w1[3], w1[4]))
        U[i, ny + 4] = primitive_to_conserved(law, SVector(w2[1], w2[2], -w2[3], w2[4]))
    end
    return nothing
end

# ============================================================
# DirichletHyperbolicBC (2D)
# ============================================================

function apply_bc_2d_left!(U::AbstractMatrix, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 4)
        U[2, j] = u_bc
        U[1, j] = u_bc
    end
    return nothing
end

function apply_bc_2d_right!(U::AbstractMatrix, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 4)
        U[nx + 3, j] = u_bc
        U[nx + 4, j] = u_bc
    end
    return nothing
end

function apply_bc_2d_bottom!(U::AbstractMatrix, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for i in 1:(nx + 4)
        U[i, 2] = u_bc
        U[i, 1] = u_bc
    end
    return nothing
end

function apply_bc_2d_top!(U::AbstractMatrix, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for i in 1:(nx + 4)
        U[i, ny + 3] = u_bc
        U[i, ny + 4] = u_bc
    end
    return nothing
end

# ============================================================
# InflowBC (2D)
# ============================================================

function apply_bc_2d_left!(U::AbstractMatrix, bc::InflowBC, law, nx::Int, ny::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 4)
        U[2, j] = u_bc
        U[1, j] = u_bc
    end
    return nothing
end

function apply_bc_2d_right!(U::AbstractMatrix, bc::InflowBC, law, nx::Int, ny::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 4)
        U[nx + 3, j] = u_bc
        U[nx + 4, j] = u_bc
    end
    return nothing
end

function apply_bc_2d_bottom!(U::AbstractMatrix, bc::InflowBC, law, nx::Int, ny::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for i in 1:(nx + 4)
        U[i, 2] = u_bc
        U[i, 1] = u_bc
    end
    return nothing
end

function apply_bc_2d_top!(U::AbstractMatrix, bc::InflowBC, law, nx::Int, ny::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for i in 1:(nx + 4)
        U[i, ny + 3] = u_bc
        U[i, ny + 4] = u_bc
    end
    return nothing
end

# ============================================================
# PeriodicHyperbolicBC (2D)
# ============================================================

function apply_bc_2d_periodic_x!(U::AbstractMatrix, law, nx::Int, ny::Int, t)
    for j in 1:(ny + 4)
        U[2, j] = U[nx + 2, j]
        U[1, j] = U[nx + 1, j]
        U[nx + 3, j] = U[3, j]
        U[nx + 4, j] = U[4, j]
    end
    return nothing
end

function apply_bc_2d_periodic_y!(U::AbstractMatrix, law, nx::Int, ny::Int, t)
    for i in 1:(nx + 4)
        U[i, 2] = U[i, ny + 2]
        U[i, 1] = U[i, ny + 1]
        U[i, ny + 3] = U[i, 3]
        U[i, ny + 4] = U[i, 4]
    end
    return nothing
end

# ============================================================
# Apply all 2D boundary conditions
# ============================================================

"""
    apply_boundary_conditions_2d!(U, prob, t)

Apply boundary conditions on all 4 sides of the 2D domain.
Order: left, right, bottom, top. Periodic BCs are handled specially.
"""
function apply_boundary_conditions_2d!(U::AbstractMatrix, prob::HyperbolicProblem2D, t)
    nx = prob.mesh.nx
    ny = prob.mesh.ny
    law = prob.law

    # Handle periodic BCs in x
    if prob.bc_left isa PeriodicHyperbolicBC && prob.bc_right isa PeriodicHyperbolicBC
        apply_bc_2d_periodic_x!(U, law, nx, ny, t)
    else
        apply_bc_2d_left!(U, prob.bc_left, law, nx, ny, t)
        apply_bc_2d_right!(U, prob.bc_right, law, nx, ny, t)
    end

    # Handle periodic BCs in y
    if prob.bc_bottom isa PeriodicHyperbolicBC && prob.bc_top isa PeriodicHyperbolicBC
        apply_bc_2d_periodic_y!(U, law, nx, ny, t)
    else
        apply_bc_2d_bottom!(U, prob.bc_bottom, law, nx, ny, t)
        apply_bc_2d_top!(U, prob.bc_top, law, nx, ny, t)
    end

    return nothing
end
