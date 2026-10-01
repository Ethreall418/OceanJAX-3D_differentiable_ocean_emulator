"""
OceanJAX Physics – Tracers Module
==================================
Flux-form advection and horizontal diffusion for an arbitrary tracer C.

Responsibility contract
-----------------------
Every function in this module returns a **tendency** [tracer s-1] only.
No time integration, no state mutation, and no implicit vertical
diffusion are performed here; those belong in the time stepper and
mixing module respectively.  The caller is responsible for combining
explicit and implicit pieces without double-counting.

Vertical-face / surface-forcing division of labour
---------------------------------------------------
Tracer advection uses ``grid.mask_w``, whose surface face (k=0) is open
wherever the top cell is wet.  With a linear free surface the layer
thicknesses are fixed, so the kinematic flux w[0] * C[0] through the
surface is what lets a column exchange volume with the (unresolved)
eta-layer: it makes every cell exactly non-divergent and keeps a uniform
tracer uniform.  The conserved quantity is  sum(C*V) + sum(eta*C_surf*A).

This is a volume flux, not an air-sea exchange: all physical surface
fluxes (heat, virtual salt) still enter only through the explicit forcing
tendencies at the bottom of this module, and implicit vertical diffusion
keeps the surface face closed via ``grid.mask_w_adv``.

Governing equation (Boussinesq, per unit volume):

  dC/dt = adv(C)  +  div_h(kappa_h * grad_h C)
        + div_z(kappa_v * grad_z C)   ← implicit, not here
        + Q_C                          ← surface forcing tendencies below

Flux-form advection:

  adv(C) = -(1/V) [ (Fu_e - Fu_w) + (Fv_n - Fv_s) + (Fw_b - Fw_t) ]

  Fu = u * C_face * (dy_c  * dz_c)   east-face area
  Fv = v * C_face * (dx_v  * dz_c)   north-face area  (cos(lat_v) metric)
  Fw = w * C_face *  area_c           horizontal cell area

All fluxes are gated by the appropriate face mask
(mask_u, mask_v, mask_w).
"""

from __future__ import annotations

import jax.numpy as jnp

from OceanJAX.grid import OceanGrid
from OceanJAX.operators import interp_c_to_u, interp_c_to_v

# Specific heat capacity of seawater [J kg-1 K-1]
CP_SEAWATER: float = 3996.0


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _vertical_face_values(phi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build per-face arrays for the cell above and below each w-face.

      above_w[:,:,k] = phi[:,:,k-1]   for k = 1..Nz  (cell above face k)
                     = phi[:,:,0]     for k = 0       (surface: top-cell value)
      below_w[:,:,k] = phi[:,:,k]     for k = 0..Nz-1 (cell below face k)
                     = phi[:,:,Nz-1]  for k = Nz      (masked out by mask_w)

    Returns (above_w, below_w), each shape (Nx, Ny, Nz+1).
    """
    above_w = jnp.concatenate([phi[:, :, :1],   phi            ], axis=-1)
    below_w = jnp.concatenate([phi,              phi[:, :, -1:] ], axis=-1)
    return above_w, below_w


# ---------------------------------------------------------------------------
# Advection
# ---------------------------------------------------------------------------

def upwind_advection(
    phi:  jnp.ndarray,
    u:    jnp.ndarray,
    v:    jnp.ndarray,
    w:    jnp.ndarray,
    grid: OceanGrid,
) -> jnp.ndarray:
    """
    1st-order upwind flux-form advective tendency for tracer phi.

    Face value selection (all gated by the face mask):

      east  face (i+1/2): phi[i]   if u >= 0  (eastward),  else phi[i+1]
      north face (j+1/2): phi[j]   if v >= 0  (northward), else phi[j+1]
      top   face (k):     phi[k-1] if w >= 0  (downward),  else phi[k]

    The surface vertical face (k=0) carries the linear free-surface
    kinematic flux w[0] * phi[0] (see module docstring); the seafloor
    face is closed via ``grid.mask_w``.

    Args:
        phi : (Nx, Ny, Nz)    tracer at cell centres
        u   : (Nx, Ny, Nz)    zonal velocity at east faces
        v   : (Nx, Ny, Nz)    meridional velocity at north faces
        w   : (Nx, Ny, Nz+1)  vertical velocity (positive downward)
        grid: OceanGrid

    Returns:
        dC/dt|_adv : (Nx, Ny, Nz), zeroed at dry cells
    """
    dz = grid.dz_c   # (Nz,)

    # ---- Zonal flux (east face, i+1/2) -------------------------------------
    phi_e = jnp.roll(phi, -1, axis=0)
    C_e   = jnp.where(u >= 0, phi, phi_e)
    Fu_e  = u * grid.mask_u * C_e * (grid.dy_c[:, :, jnp.newaxis] * dz)
    Fu_w  = jnp.roll(Fu_e, 1, axis=0)

    # ---- Meridional flux (north face, j+1/2) --------------------------------
    phi_n = jnp.roll(phi, -1, axis=1)
    # Northern wall handled entirely by mask_v = 0; no phi override needed.
    C_v   = jnp.where(v >= 0, phi, phi_n)
    Fv_n  = v * grid.mask_v * C_v * (grid.dx_v[:, :, jnp.newaxis] * dz)
    Fv_s  = jnp.concatenate(
        [jnp.zeros((grid.Nx, 1, grid.Nz), dtype=phi.dtype), Fv_n[:, :-1, :]], axis=1
    )

    # ---- Vertical flux (top face k, using mask_w) ---------------------------
    # w > 0 (downward): upwind source is the cell above (above_w)
    # w < 0 (upward):   upwind source is the cell below (below_w)
    # Surface face (k=0) uses the top-cell value either way.
    above_w, below_w = _vertical_face_values(phi)
    C_w  = jnp.where(w >= 0, above_w, below_w)
    Fw   = w * grid.mask_w * C_w * grid.area_c[:, :, jnp.newaxis]
    Fw_t = Fw[:, :, :-1]   # top-face flux of cell k  (w-face k)
    Fw_b = Fw[:, :, 1:]    # bottom-face flux of cell k (w-face k+1)

    # ---- Flux divergence -> tendency ---------------------------------------
    tend = -((Fu_e - Fu_w) + (Fv_n - Fv_s) + (Fw_b - Fw_t)) / grid.volume_c
    return tend * grid.mask_c


def centered_advection(
    phi:  jnp.ndarray,
    u:    jnp.ndarray,
    v:    jnp.ndarray,
    w:    jnp.ndarray,
    grid: OceanGrid,
) -> jnp.ndarray:
    """
    2nd-order centered flux-form advective tendency for tracer phi.

    **Experimental / diagnostic use only.**  Face values are arithmetic
    averages of adjacent cell centres.  The scheme is non-dissipative and
    does not guarantee boundedness even when paired with explicit horizontal
    diffusion.  For production runs requiring low numerical diffusion, a
    limiter-based scheme (e.g. TVD or PPM) is the appropriate next step.

    Args / Returns: same convention as ``upwind_advection``.
    """
    dz = grid.dz_c

    # ---- Zonal ----
    phi_e = jnp.roll(phi, -1, axis=0)
    Fu_e  = u * grid.mask_u * 0.5 * (phi + phi_e) * (grid.dy_c[:, :, jnp.newaxis] * dz)
    Fu_w  = jnp.roll(Fu_e, 1, axis=0)

    # ---- Meridional ----
    phi_n = jnp.roll(phi, -1, axis=1)
    Fv_n  = v * grid.mask_v * 0.5 * (phi + phi_n) * (grid.dx_v[:, :, jnp.newaxis] * dz)
    Fv_s  = jnp.concatenate(
        [jnp.zeros((grid.Nx, 1, grid.Nz), dtype=phi.dtype), Fv_n[:, :-1, :]], axis=1
    )

    # ---- Vertical (surface face carries the kinematic flux) ----
    above_w, below_w = _vertical_face_values(phi)
    Fw    = w * grid.mask_w * 0.5 * (above_w + below_w) * grid.area_c[:, :, jnp.newaxis]
    Fw_t  = Fw[:, :, :-1]
    Fw_b  = Fw[:, :, 1:]

    tend = -((Fu_e - Fu_w) + (Fv_n - Fv_s) + (Fw_b - Fw_t)) / grid.volume_c
    return tend * grid.mask_c


# ---------------------------------------------------------------------------
# Horizontal diffusion with spatially varying kappa
# ---------------------------------------------------------------------------

def kappa_laplacian_h(
    phi:     jnp.ndarray,
    kappa_h: jnp.ndarray,
    grid:    OceanGrid,
) -> jnp.ndarray:
    """
    Flux-form horizontal Laplacian with scalar or spatially varying kappa_h.

      div_h(kappa * grad_h C) = (Fx_e - Fx_w + Fy_n - Fy_s) / area

      Fx_e = kappa_u * dy_c * (phi[i+1] - phi[i]) / dx_u * mask_u
      Fy_n = kappa_v * dx_v * (phi[j+1] - phi[j]) / dy_v * mask_v

    When kappa_h is a spatial array, it is interpolated arithmetically to
    face centres before forming fluxes.  Negative values are clamped to
    zero to prevent anti-diffusion.

    Args:
        phi     : (Nx, Ny, Nz)
        kappa_h : scalar  or  (Nx, Ny, Nz) [m² s-1]
        grid    : OceanGrid

    Returns:
        (Nx, Ny, Nz), zeroed at dry cells
    """
    if jnp.ndim(kappa_h) > 0:
        # Clamp to ensure non-negativity before face interpolation
        kappa_h = jnp.maximum(kappa_h, 0.0)
        kappa_u = interp_c_to_u(kappa_h, grid)
        kappa_v = interp_c_to_v(kappa_h, grid)
    else:
        kappa_u = kappa_h
        kappa_v = kappa_h

    dy_c = grid.dy_c[:, :, jnp.newaxis]
    dx_u = grid.dx_u[:, :, jnp.newaxis]
    dx_v = grid.dx_v[:, :, jnp.newaxis]
    dy_v = grid.dy_v[:, :, jnp.newaxis]

    # East-face diffusive flux
    phi_e = jnp.roll(phi, -1, axis=0)
    Fx_e  = kappa_u * dy_c * (phi_e - phi) / dx_u * grid.mask_u
    Fx_w  = jnp.roll(Fx_e, 1, axis=0)

    # North-face diffusive flux (north Neumann BC; dx_v metric)
    phi_n = jnp.roll(phi, -1, axis=1)
    phi_n = phi_n.at[:, -1, :].set(phi[:, -1, :])
    Fy_n  = kappa_v * dx_v * (phi_n - phi) / dy_v * grid.mask_v
    Fy_s  = jnp.concatenate(
        [jnp.zeros(phi.shape[:1] + (1,) + phi.shape[2:], dtype=phi.dtype),
         Fy_n[:, :-1, :]], axis=1
    )

    return ((Fx_e - Fx_w + Fy_n - Fy_s) / grid.area_c[:, :, jnp.newaxis]) * grid.mask_c


# ---------------------------------------------------------------------------
# Combined explicit tendency (advection + horizontal diffusion)
# ---------------------------------------------------------------------------

def tracer_tendency(
    phi:     jnp.ndarray,
    u:       jnp.ndarray,
    v:       jnp.ndarray,
    w:       jnp.ndarray,
    kappa_h: jnp.ndarray,
    grid:    OceanGrid,
) -> jnp.ndarray:
    """
    Explicit tracer tendency: 1st-order upwind advection + horizontal diffusion.

    Vertical diffusion is excluded; it is treated implicitly by the
    tridiagonal solver in OceanJAX.Physics.mixing and combined by the time
    stepper.  Surface forcing is also excluded; apply the functions below
    separately and add the tendencies before time-integrating.

    Args:
        phi     : (Nx, Ny, Nz)    tracer at cell centres
        u, v    : (Nx, Ny, Nz)    horizontal velocities
        w       : (Nx, Ny, Nz+1)  vertical velocity
        kappa_h : scalar or (Nx, Ny, Nz) [m² s-1]
        grid    : OceanGrid

    Returns:
        dC/dt|_explicit : (Nx, Ny, Nz)
    """
    adv    = upwind_advection(phi, u, v, w, grid)
    diff_h = kappa_laplacian_h(phi, kappa_h, grid)
    return (adv + diff_h) * grid.mask_c


# ---------------------------------------------------------------------------
# Surface forcing tendencies (generic + T/S-specific)
# ---------------------------------------------------------------------------

def surface_layer_tendency(
    flux_per_area: jnp.ndarray,
    grid:          OceanGrid,
) -> jnp.ndarray:
    """
    Generic surface flux tendency applied to the top tracer layer (k=0).

    This is the only sanctioned pathway for air-sea tracer exchange.  The
    advective surface flux w[0]*C[0] is a free-surface volume flux, not an
    air-sea flux, so adding a tendency here does not double-count it.

    Args:
        flux_per_area : (Nx, Ny) [tracer · m · s-1]
                        Positive = tracer entering the ocean.
                        Must already carry the correct physical units
                        (e.g. K·m·s-1 for temperature, psu·m·s-1 for
                        salinity).  The division by layer thickness dz_c[0]
                        is performed here.
        grid          : OceanGrid

    Returns:
        (Nx, Ny, Nz) tendency, nonzero only at k=0
    """
    tend_surf = flux_per_area / grid.dz_c[0] * grid.mask_c[:, :, 0]
    return jnp.zeros(
        (grid.Nx, grid.Ny, grid.Nz), dtype=flux_per_area.dtype
    ).at[:, :, 0].set(tend_surf)


def heat_surface_tendency(
    heat_flux: jnp.ndarray,
    grid:      OceanGrid,
    params,
) -> jnp.ndarray:
    """
    Temperature tendency from net surface heat flux.

    Args:
        heat_flux : (Nx, Ny) [W m-2], positive = downward into ocean
        grid      : OceanGrid
        params    : ModelParams  (uses rho0)

    Returns:
        dT/dt|_surf : (Nx, Ny, Nz) [K s-1]
    """
    # Convert: W m-2 -> K m s-1  via  Q / (rho * cp)
    flux_per_area = heat_flux / (params.rho0 * CP_SEAWATER)
    return surface_layer_tendency(flux_per_area, grid)


def salt_surface_tendency(
    fw_flux: jnp.ndarray,
    grid:    OceanGrid,
    params,
    sss:     jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Salinity tendency from net surface freshwater flux (virtual salt flux).

    With a linear free surface the freshwater volume is not added to the
    ocean; its diluting / concentrating effect on the top layer is applied
    as a salt flux

      dS/dt|_surf = SSS * (E - P) / dz_c[0]

    using the local sea-surface salinity (as NEMO does for a linear free
    surface).  Dilution is then proportional to the salinity itself, so a
    column under strong freshwater input (e.g. a river mouth, whose runoff
    ORAS5 includes in E - P) freshens exponentially towards 0 and can never
    become negative.  A fixed reference salinity S_ref in place of SSS
    removes the same salt however fresh the water already is; in a
    one-layer Amazon-mouth column that drove SSS to -28 psu in a 2026
    hindcast.

    Args:
        fw_flux : (Nx, Ny) [m s-1]  E - P, positive = net evaporation
                  (freshwater loss → salinity increase)
        grid    : OceanGrid
        params  : ModelParams  (uses S_ref when sss is None)
        sss     : (Nx, Ny) [psu] top-layer salinity, or None to use the
                  constant params.S_ref (the original formulation)

    Returns:
        dS/dt|_surf : (Nx, Ny, Nz) [psu s-1]
    """
    # fw_flux > 0 (evaporation) => ocean loses freshwater => salinity increases
    s_surf = params.S_ref if sss is None else jnp.maximum(sss, 0.0)
    flux_per_area = s_surf * fw_flux
    return surface_layer_tendency(flux_per_area, grid)
