# ============================================================
# 2D Ghost-Cell Boundary Conditions
# ============================================================
#
# The 2D solution is stored in a padded matrix U[i, j] where:
#   i = 1:ng              -> left ghost columns
#   i = ng+1:nx+ng        -> interior columns
#   i = nx+ng+1:nx+2*ng   -> right ghost columns
#   j = 1:ng              -> bottom ghost rows
#   j = ng+1:ny+ng        -> interior rows
#   j = ny+ng+1:ny+2*ng   -> top ghost rows
#
# Interior cell (ix, iy) (1-based) maps to U[ix+ng, iy+ng].

# ============================================================
# TransmissiveBC (2D)
# ============================================================

function apply_bc_2d_left!(U::AbstractMatrix, ::TransmissiveBC, law, nx::Int, ny::Int, ng::Int, t)
    for j in 1:(ny + 2 * ng)
        for g in 1:ng
            U[ng + 1 - g, j] = U[ng + 1, j]
        end
    end
    return nothing
end

function apply_bc_2d_right!(U::AbstractMatrix, ::TransmissiveBC, law, nx::Int, ny::Int, ng::Int, t)
    for j in 1:(ny + 2 * ng)
        for g in 1:ng
            U[nx + ng + g, j] = U[nx + ng, j]
        end
    end
    return nothing
end

function apply_bc_2d_bottom!(U::AbstractMatrix, ::TransmissiveBC, law, nx::Int, ny::Int, ng::Int, t)
    for i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, ng + 1 - g] = U[i, ng + 1]
        end
    end
    return nothing
end

function apply_bc_2d_top!(U::AbstractMatrix, ::TransmissiveBC, law, nx::Int, ny::Int, ng::Int, t)
    for i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, ny + ng + g] = U[i, ny + ng]
        end
    end
    return nothing
end

# ============================================================
# ReflectiveBC (2D Euler)
# ============================================================

# Left wall: negate vx
function apply_bc_2d_left!(U::AbstractMatrix, ::ReflectiveBC, law::EulerEquations{2}, nx::Int, ny::Int, ng::Int, t)
    for j in 1:(ny + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[ng + g, j])
            # Negate vx (component 2), keep vy (component 3)
            U[ng + 1 - g, j] = primitive_to_conserved(law, SVector(w[1], -w[2], w[3], w[4]))
        end
    end
    return nothing
end

# Right wall: negate vx
function apply_bc_2d_right!(U::AbstractMatrix, ::ReflectiveBC, law::EulerEquations{2}, nx::Int, ny::Int, ng::Int, t)
    for j in 1:(ny + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[nx + ng + 1 - g, j])
            U[nx + ng + g, j] = primitive_to_conserved(law, SVector(w[1], -w[2], w[3], w[4]))
        end
    end
    return nothing
end

# Bottom wall: negate vy
function apply_bc_2d_bottom!(U::AbstractMatrix, ::ReflectiveBC, law::EulerEquations{2}, nx::Int, ny::Int, ng::Int, t)
    for i in 1:(nx + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[i, ng + g])
            # Negate vy (component 3), keep vx (component 2)
            U[i, ng + 1 - g] = primitive_to_conserved(law, SVector(w[1], w[2], -w[3], w[4]))
        end
    end
    return nothing
end

# Top wall: negate vy
function apply_bc_2d_top!(U::AbstractMatrix, ::ReflectiveBC, law::EulerEquations{2}, nx::Int, ny::Int, ng::Int, t)
    for i in 1:(nx + 2 * ng)
        for g in 1:ng
            w = conserved_to_primitive(law, U[i, ny + ng + 1 - g])
            U[i, ny + ng + g] = primitive_to_conserved(law, SVector(w[1], w[2], -w[3], w[4]))
        end
    end
    return nothing
end

# ============================================================
# DirichletHyperbolicBC (2D)
# ============================================================

function apply_bc_2d_left!(U::AbstractMatrix, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, ng::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 2 * ng)
        for g in 1:ng
            U[ng + 1 - g, j] = u_bc
        end
    end
    return nothing
end

function apply_bc_2d_right!(U::AbstractMatrix, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, ng::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 2 * ng)
        for g in 1:ng
            U[nx + ng + g, j] = u_bc
        end
    end
    return nothing
end

function apply_bc_2d_bottom!(U::AbstractMatrix, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, ng::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, ng + 1 - g] = u_bc
        end
    end
    return nothing
end

function apply_bc_2d_top!(U::AbstractMatrix, bc::DirichletHyperbolicBC, law, nx::Int, ny::Int, ng::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, ny + ng + g] = u_bc
        end
    end
    return nothing
end

# ============================================================
# InflowBC (2D)
# ============================================================

function apply_bc_2d_left!(U::AbstractMatrix, bc::InflowBC, law, nx::Int, ny::Int, ng::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 2 * ng)
        for g in 1:ng
            U[ng + 1 - g, j] = u_bc
        end
    end
    return nothing
end

function apply_bc_2d_right!(U::AbstractMatrix, bc::InflowBC, law, nx::Int, ny::Int, ng::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for j in 1:(ny + 2 * ng)
        for g in 1:ng
            U[nx + ng + g, j] = u_bc
        end
    end
    return nothing
end

function apply_bc_2d_bottom!(U::AbstractMatrix, bc::InflowBC, law, nx::Int, ny::Int, ng::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, ng + 1 - g] = u_bc
        end
    end
    return nothing
end

function apply_bc_2d_top!(U::AbstractMatrix, bc::InflowBC, law, nx::Int, ny::Int, ng::Int, t)
    u_bc = primitive_to_conserved(law, bc.state)
    for i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, ny + ng + g] = u_bc
        end
    end
    return nothing
end

# ============================================================
# PeriodicHyperbolicBC (2D)
# ============================================================

function apply_bc_2d_periodic_x!(U::AbstractMatrix, law, nx::Int, ny::Int, ng::Int, t)
    for j in 1:(ny + 2 * ng)
        for g in 1:ng
            U[ng + 1 - g, j] = U[nx + ng + 1 - g, j]  # left ghost from right interior
            U[nx + ng + g, j] = U[ng + g, j]            # right ghost from left interior
        end
    end
    return nothing
end

function apply_bc_2d_periodic_y!(U::AbstractMatrix, law, nx::Int, ny::Int, ng::Int, t)
    for i in 1:(nx + 2 * ng)
        for g in 1:ng
            U[i, ng + 1 - g] = U[i, ny + ng + 1 - g]  # bottom ghost from top interior
            U[i, ny + ng + g] = U[i, ng + g]            # top ghost from bottom interior
        end
    end
    return nothing
end

# ============================================================
# Apply all 2D boundary conditions
# ============================================================

"""
    apply_boundary_conditions_2d!(U, prob, ng, t)

Apply boundary conditions on all 4 sides of the 2D domain.
Order: left, right, bottom, top. Periodic BCs are handled specially.
"""
function apply_boundary_conditions_2d!(U::AbstractMatrix, prob::HyperbolicProblem2D, ng::Int, t)
    nx = prob.mesh.nx
    ny = prob.mesh.ny
    law = prob.law

    # Handle periodic BCs in x
    if prob.bc_left isa PeriodicHyperbolicBC && prob.bc_right isa PeriodicHyperbolicBC
        apply_bc_2d_periodic_x!(U, law, nx, ny, ng, t)
    else
        apply_bc_2d_left!(U, prob.bc_left, law, nx, ny, ng, t)
        apply_bc_2d_right!(U, prob.bc_right, law, nx, ny, ng, t)
    end

    # Handle periodic BCs in y
    if prob.bc_bottom isa PeriodicHyperbolicBC && prob.bc_top isa PeriodicHyperbolicBC
        apply_bc_2d_periodic_y!(U, law, nx, ny, ng, t)
    else
        apply_bc_2d_bottom!(U, prob.bc_bottom, law, nx, ny, ng, t)
        apply_bc_2d_top!(U, prob.bc_top, law, nx, ny, ng, t)
    end

    return nothing
end
