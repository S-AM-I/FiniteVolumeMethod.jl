# ============================================================
# Constrained Transport Update
# ============================================================
#
# Updates the face-centered magnetic field using the corner EMF,
# guaranteeing ∇·B = 0 to machine precision via Faraday's law
# in integral (Stokes) form.
#
# Bx_face[i,j] -= dt/dy * (emf_z[i,j+1] - emf_z[i,j])
# By_face[i,j] += dt/dx * (emf_z[i+1,j] - emf_z[i,j])
#
# This is a discrete form of:
#   ∂Bx/∂t = -∂Ez/∂y
#   ∂By/∂t = +∂Ez/∂x

"""
    ct_update!(ct::CTData2D, dt, dx, dy, nx::Int, ny::Int)

Update face-centered B-fields using the EMF at cell corners.
This guarantees ∇·B = 0 to machine precision.

  `Bx_face[i,j] -= dt/dy * (emf_z[i,j+1] - emf_z[i,j])`
  `By_face[i,j] += dt/dx * (emf_z[i+1,j] - emf_z[i,j])`
"""
function ct_update!(ct::CTData2D, dt, dx, dy, nx::Int, ny::Int)
    Bx = ct.Bx_face
    By = ct.By_face
    Ez = ct.emf_z

    dt_dy = dt / dy
    dt_dx = dt / dx

    # Update Bx at x-faces: ∂Bx/∂t = -∂Ez/∂y
    for j in 1:ny, i in 1:(nx + 1)
        Bx[i, j] -= dt_dy * (Ez[i, j + 1] - Ez[i, j])
    end

    # Update By at y-faces: ∂By/∂t = +∂Ez/∂x
    for j in 1:(ny + 1), i in 1:nx
        By[i, j] += dt_dx * (Ez[i + 1, j] - Ez[i, j])
    end

    return nothing
end

"""
    ct_weighted_update!(ct_out::CTData2D, ct_old::CTData2D, ct_stage::CTData2D,
                        α_old, α_stage, dt, dx, dy, nx::Int, ny::Int)

Perform a weighted CT update for SSP-RK stages:
  `ct_out.B = α_old * ct_old.B + α_stage * (ct_stage.B + dt * ΔB)`

where `ΔB` is the CT update computed from `ct_stage.emf_z`.
"""
function ct_weighted_update!(ct_out::CTData2D, ct_old::CTData2D, ct_stage::CTData2D,
        α_old, α_stage, dt, dx, dy, nx::Int, ny::Int)
    Bx_out = ct_out.Bx_face
    By_out = ct_out.By_face
    Bx_old = ct_old.Bx_face
    By_old = ct_old.By_face
    Bx_stage = ct_stage.Bx_face
    By_stage = ct_stage.By_face
    Ez = ct_stage.emf_z

    dt_dy = dt / dy
    dt_dx = dt / dx

    for j in 1:ny, i in 1:(nx + 1)
        dBx = -dt_dy * (Ez[i, j + 1] - Ez[i, j])
        Bx_out[i, j] = α_old * Bx_old[i, j] + α_stage * (Bx_stage[i, j] + dBx)
    end

    for j in 1:(ny + 1), i in 1:nx
        dBy = dt_dx * (Ez[i + 1, j] - Ez[i, j])
        By_out[i, j] = α_old * By_old[i, j] + α_stage * (By_stage[i, j] + dBy)
    end

    return nothing
end
