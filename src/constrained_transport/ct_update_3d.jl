# ============================================================
# 3D Constrained Transport Update
# ============================================================
#
# Updates the face-centered magnetic field using the edge EMFs,
# guaranteeing div(B) = 0 to machine precision via Faraday's law
# in integral (Stokes) form.
#
# dBx/dt = -dEz/dy + dEy/dz
# dBy/dt =  dEz/dx - dEx/dz
# dBz/dt = -dEy/dx + dEx/dy

"""
    ct_update_3d!(ct::CTData3D, dt, dx, dy, dz, nx, ny, nz)

Update face-centered B-fields using the edge EMFs.
This guarantees div(B) = 0 to machine precision.

  Bx_face[i,j,k] -= dt/dy*(Ez[i,j+1,k] - Ez[i,j,k]) - dt/dz*(Ey[i,j,k+1] - Ey[i,j,k])
  By_face[i,j,k] += dt/dx*(Ez[i+1,j,k] - Ez[i,j,k]) - dt/dz*(Ex[i,j,k+1] - Ex[i,j,k])
  Bz_face[i,j,k] -= dt/dx*(Ey[i+1,j,k] - Ey[i,j,k]) - dt/dy*(Ex[i,j+1,k] - Ex[i,j,k])
"""
function ct_update_3d!(ct::CTData3D, dt, dx, dy, dz, nx::Int, ny::Int, nz::Int)
    Bx = ct.Bx_face
    By = ct.By_face
    Bz = ct.Bz_face
    Ex = ct.emf_x
    Ey = ct.emf_y
    Ez = ct.emf_z

    dt_dx = dt / dx
    dt_dy = dt / dy
    dt_dz = dt / dz

    # Update Bx at x-faces: dBx/dt = -dEz/dy + dEy/dz
    for k in 1:nz, j in 1:ny, i in 1:(nx + 1)
        Bx[i, j, k] -= dt_dy * (Ez[i, j + 1, k] - Ez[i, j, k])
        Bx[i, j, k] += dt_dz * (Ey[i, j, k + 1] - Ey[i, j, k])
    end

    # Update By at y-faces: dBy/dt = dEz/dx - dEx/dz
    for k in 1:nz, j in 1:(ny + 1), i in 1:nx
        By[i, j, k] += dt_dx * (Ez[i + 1, j, k] - Ez[i, j, k])
        By[i, j, k] -= dt_dz * (Ex[i, j, k + 1] - Ex[i, j, k])
    end

    # Update Bz at z-faces: dBz/dt = -dEy/dx + dEx/dy
    for k in 1:(nz + 1), j in 1:ny, i in 1:nx
        Bz[i, j, k] -= dt_dx * (Ey[i + 1, j, k] - Ey[i, j, k])
        Bz[i, j, k] += dt_dy * (Ex[i, j + 1, k] - Ex[i, j, k])
    end

    return nothing
end

"""
    ct_weighted_update_3d!(ct_out, ct_old, ct_stage, a_old, a_stage, dt, dx, dy, dz, nx, ny, nz)

Perform a weighted CT update for SSP-RK stages:
  `ct_out.B = a_old * ct_old.B + a_stage * (ct_stage.B + dt * dB)`

where `dB` is the CT update computed from `ct_stage`'s EMFs.
"""
function ct_weighted_update_3d!(
        ct_out::CTData3D, ct_old::CTData3D, ct_stage::CTData3D,
        a_old, a_stage, dt, dx, dy, dz, nx::Int, ny::Int, nz::Int
    )
    Bx_out = ct_out.Bx_face
    By_out = ct_out.By_face
    Bz_out = ct_out.Bz_face
    Bx_old = ct_old.Bx_face
    By_old = ct_old.By_face
    Bz_old = ct_old.Bz_face
    Bx_stg = ct_stage.Bx_face
    By_stg = ct_stage.By_face
    Bz_stg = ct_stage.Bz_face
    Ex = ct_stage.emf_x
    Ey = ct_stage.emf_y
    Ez = ct_stage.emf_z

    dt_dx = dt / dx
    dt_dy = dt / dy
    dt_dz = dt / dz

    # Bx update
    for k in 1:nz, j in 1:ny, i in 1:(nx + 1)
        dBx = -dt_dy * (Ez[i, j + 1, k] - Ez[i, j, k]) + dt_dz * (Ey[i, j, k + 1] - Ey[i, j, k])
        Bx_out[i, j, k] = a_old * Bx_old[i, j, k] + a_stage * (Bx_stg[i, j, k] + dBx)
    end

    # By update
    for k in 1:nz, j in 1:(ny + 1), i in 1:nx
        dBy = dt_dx * (Ez[i + 1, j, k] - Ez[i, j, k]) - dt_dz * (Ex[i, j, k + 1] - Ex[i, j, k])
        By_out[i, j, k] = a_old * By_old[i, j, k] + a_stage * (By_stg[i, j, k] + dBy)
    end

    # Bz update
    for k in 1:(nz + 1), j in 1:ny, i in 1:nx
        dBz = -dt_dx * (Ey[i + 1, j, k] - Ey[i, j, k]) + dt_dy * (Ex[i, j + 1, k] - Ex[i, j, k])
        Bz_out[i, j, k] = a_old * Bz_old[i, j, k] + a_stage * (Bz_stg[i, j, k] + dBz)
    end

    return nothing
end
