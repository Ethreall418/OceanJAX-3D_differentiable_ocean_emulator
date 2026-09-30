"""
OceanJAX Grid Module
====================
Defines OceanGrid using an Arakawa C-grid in spherical coordinates.
All arrays are JAX arrays. Grid is an equinox Module with static integer dims.

Staggering convention (Arakawa C-grid):
  - Tracers (T, S, rho): cell centres     (Nx, Ny, Nz)
  - u (zonal velocity):  east  faces       (Nx, Ny, Nz)
  - v (meridional vel.): north faces       (Nx, Ny, Nz)
  - w (vertical vel.):   top   faces       (Nx, Ny, Nz+1)
  - eta (SSH):           surface centers   (Nx, Ny)

Indexing: [i, j, k] = [lon, lat, depth]
  i=0 is westernmost, j=0 is southernmost, k=0 is shallowest.
"""

from __future__ import annotations

import math
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
EARTH_RADIUS: float = 6.371e6   # metres
OMEGA: float = 7.2921e-5        # rad s-1
DEG2RAD: float = math.pi / 180.0


# ---------------------------------------------------------------------------
# Vertical levels
# ---------------------------------------------------------------------------

def face_depths(centres: np.ndarray) -> np.ndarray:
    """
    Cell-face depths (Nz+1,) [m] from ascending cell-centre depths (Nz,).

    This is the rule ``OceanGrid.create`` uses: the surface face is at 0,
    interior faces sit midway between neighbouring centres, and the bottom
    face lies half a cell below the last centre, i.e. as far below it as
    the last interior face lies above it (so uniform centres give uniform
    cells).  Before 2026-09-30 the bottom face was placed only a quarter
    cell below the last centre, making the bottom cell 25 % too thin.
    """
    c = np.asarray(centres, dtype=np.float64)
    n = len(c)
    faces = np.empty(n + 1, dtype=np.float64)
    faces[0] = 0.0
    faces[1:n] = 0.5 * (c[:-1] + c[1:])
    faces[n] = c[-1] + (c[-1] - faces[n - 1]) if n > 1 else 2.0 * c[-1]
    return faces


def stretched_levels(depth_max: float, nz: int, dz_top: float) -> np.ndarray:
    """
    Cell-centre depths (nz,) [m] with thicknesses growing geometrically with
    depth, fine near the surface and coarse in the deep ocean.

    Thicknesses follow dz_k = dz_top * r**k; the stretching factor r >= 1 is
    found by bisection so that the grid's bottom face (``face_depths``, the
    rule OceanGrid.create applies to the returned centres) lies exactly at
    ``depth_max``.  Because OceanGrid places faces midway between centres,
    the model's top cell comes out slightly thicker than ``dz_top``
    (by ~(r - 1)/4, a few per cent).

    Args:
        depth_max : model bottom depth [m]
        nz        : number of levels
        dz_top    : target thickness of the top cell [m]

    Returns:
        (nz,) float64 ascending centre depths, ready for OceanGrid.create.

    Raises:
        ValueError if nz * dz_top > depth_max (even uniform levels would be
        too deep) or an argument is not positive.
    """
    if nz < 1 or dz_top <= 0 or depth_max <= 0:
        raise ValueError(f"need nz >= 1, dz_top > 0, depth_max > 0; got "
                         f"nz={nz}, dz_top={dz_top}, depth_max={depth_max}")
    if nz * dz_top > depth_max:
        raise ValueError(
            f"nz * dz_top = {nz * dz_top:g} m exceeds depth_max = {depth_max:g} m; "
            f"reduce nz or dz_top."
        )

    def centres(r: float) -> np.ndarray:
        dz = dz_top * r ** np.arange(nz)
        f = np.concatenate([[0.0], np.cumsum(dz)])
        return 0.5 * (f[:-1] + f[1:])

    def bottom(r: float) -> float:
        return face_depths(centres(r))[-1]

    lo, hi = 1.0, 1.5
    if bottom(lo) >= depth_max:            # uniform already reaches depth_max
        return centres(lo) * depth_max / bottom(lo)
    while bottom(hi) < depth_max:
        hi = 1.0 + 2.0 * (hi - 1.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if bottom(mid) < depth_max else (lo, mid)
    c = centres(hi)
    return c * depth_max / face_depths(c)[-1]           # remove residual rounding


#: Empirical barotropic CFL stability limit of the time stepping (exp. 5.1.4).
CFL_LIMIT: float = 0.33


def barotropic_cfl(grid: "OceanGrid", dt: float, g: float = 9.81) -> tuple[float, float]:
    """
    Barotropic (surface gravity wave) CFL number and the largest dt that
    keeps it at the empirical stability limit.

      CFL = sqrt(g * H) * dt / dx_min

    H is the deepest wet column the model resolves (bathymetry capped at the
    model bottom), dx_min the smallest horizontal spacing over wet columns.
    Experiment 5.1.4 found the leapfrog / Asselin scheme stable up to
    CFL ~ 0.33 and unstable at ~0.45.

    Returns:
        (cfl, dt_max) with dt_max = 0.33 * dx_min / sqrt(g * H)  [s].
    """
    wet = np.asarray(grid.mask_c)[:, :, 0] > 0
    z_bottom = float(np.asarray(grid.z_w)[-1])
    H  = min(float(np.asarray(grid.H)[wet].max()), z_bottom) if wet.any() else z_bottom
    dx = np.minimum(np.asarray(grid.dx_c), np.asarray(grid.dy_c))
    dx_min = float(dx[wet].min() if wet.any() else dx.min())
    c = np.sqrt(g * H)
    return c * dt / dx_min, CFL_LIMIT * dx_min / c


# ---------------------------------------------------------------------------
# OceanGrid
# ---------------------------------------------------------------------------
class OceanGrid(eqx.Module):
    """
    Arakawa C-grid in spherical coordinates.

    Static fields (traced as compile-time constants):
        Nx, Ny, Nz

    Array fields (all jnp.ndarray):
        Coordinate arrays, spacing arrays, metric factors, masks, bathymetry.
    """

    # ---- static integer dimensions (compile-time constants) ---------------
    Nx: int = eqx.field(static=True)
    Ny: int = eqx.field(static=True)
    Nz: int = eqx.field(static=True)

    # ---- 1-D coordinate arrays at cell centres ----------------------------
    lon_c: jnp.ndarray   # (Nx,)  degrees east
    lat_c: jnp.ndarray   # (Ny,)  degrees north
    z_c:   jnp.ndarray   # (Nz,)  metres (positive downward)

    # ---- 1-D coordinate arrays at cell faces ------------------------------
    lon_u: jnp.ndarray   # (Nx,)  east face longitudes
    lat_v: jnp.ndarray   # (Ny,)  north face latitudes
    z_w:   jnp.ndarray   # (Nz+1,) top faces of cells (z_w[0]=0 = surface)

    # ---- horizontal grid spacings in metres -------------------------------
    # At tracer points
    dx_c: jnp.ndarray    # (Nx, Ny) spacing in x at tracer center
    dy_c: jnp.ndarray    # (Nx, Ny) spacing in y at tracer center
    # At velocity points
    dx_u: jnp.ndarray    # (Nx, Ny) spacing in x at u-point (east face, uses cos(lat_c))
    dy_v: jnp.ndarray    # (Nx, Ny) spacing in y at v-point (north face)
    dx_v: jnp.ndarray    # (Nx, Ny) zonal width at v-point (north face, uses cos(lat_v))

    # ---- vertical spacings in metres --------------------------------------
    dz_c: jnp.ndarray    # (Nz,)   cell thickness
    dz_w: jnp.ndarray    # (Nz+1,) distance between cell centres

    # ---- Coriolis parameter -----------------------------------------------
    f_c: jnp.ndarray     # (Nx, Ny) at tracer centres

    # ---- derived area / volume --------------------------------------------
    area_c:   jnp.ndarray  # (Nx, Ny)       m^2
    volume_c: jnp.ndarray  # (Nx, Ny, Nz)   m^3

    # ---- land / ocean masks (1 = ocean, 0 = land) -------------------------
    mask_c: jnp.ndarray   # (Nx, Ny, Nz)
    mask_u: jnp.ndarray   # (Nx, Ny, Nz)
    mask_v: jnp.ndarray   # (Nx, Ny, Nz)
    mask_w: jnp.ndarray       # (Nx, Ny, Nz+1)
    mask_w_adv: jnp.ndarray   # (Nx, Ny, Nz+1)  same as mask_w but k=0 always 0
    #   The surface face (k=0) is open in mask_w: it carries the linear
    #   free-surface kinematic flux w[0] = -deta/dt, used both by w and by
    #   tracer advection.  mask_w_adv closes it for implicit vertical
    #   diffusion, so no diffusive flux crosses the sea surface; air-sea
    #   exchange goes only through the explicit surface-forcing tendencies.

    # ---- bathymetry -------------------------------------------------------
    H: jnp.ndarray        # (Nx, Ny)  total water column depth [m]

    # ---- zonal boundary type (compile-time constant) ----------------------
    # True : x is periodic (east face of i=Nx-1 connects to i=0).
    # False: closed east/west walls (mask_u[Nx-1] = 0), for regional domains.
    periodic_x: bool = eqx.field(static=True, default=True)

    # ------------------------------------------------------------------
    # Factory method
    # ------------------------------------------------------------------
    @staticmethod
    def create(
        lon_bounds: tuple,
        lat_bounds: tuple,
        depth_levels: np.ndarray,
        Nx: int,
        Ny: int,
        bathymetry: Optional[np.ndarray] = None,
        lon_spacing: Optional[np.ndarray] = None,
        lat_spacing: Optional[np.ndarray] = None,
        periodic_x: bool = True,
    ) -> "OceanGrid":
        """
        Build an OceanGrid.

        Parameters
        ----------
        lon_bounds  : (lon_min, lon_max) in degrees east
        lat_bounds  : (lat_min, lat_max) in degrees north
        depth_levels: 1-D array of cell-center depths [m, positive down].
                      Length = Nz.  Cell faces are computed as midpoints
                      between consecutive centres (plus surface=0 and a
                      bottom face half a cell below the last center; see face_depths).
        Nx, Ny      : number of tracer cells in x and y
        bathymetry  : (Nx, Ny) array of total depth [m].  If None, flat
                      bottom at depth_levels[-1] + dz/2.
        lon_spacing : optional weight array (length Nx) for variable resolution
                      in x.  Weights are normalized to produce the correct
                      total span.
        lat_spacing : optional weight array (length Ny) for variable resolution
                      in y.
        periodic_x  : True (default) for a zonally periodic domain.  False
                      closes the east and west boundaries with solid walls
                      by zeroing mask_u on the east face of i = Nx-1 (the
                      face that would otherwise wrap around to i = 0).
                      Zonal fluxes at tracer and u points are gated by
                      mask_u, so this closes the boundary for the free
                      surface, advection, diffusion and u; the v-point
                      viscosity flux checks ``periodic_x`` explicitly.
        """
        lon_min, lon_max = lon_bounds
        lat_min, lat_max = lat_bounds
        depth_levels = np.asarray(depth_levels, dtype=np.float64)
        Nz = len(depth_levels)

        # --- build 1-D longitude at tracer centres -------------------------
        if lon_spacing is not None:
            w = np.asarray(lon_spacing, dtype=np.float64)
            assert len(w) == Nx
            w = w / w.sum()
            lon_edges = np.concatenate([[lon_min], lon_min + np.cumsum(w) * (lon_max - lon_min)])
        else:
            lon_edges = np.linspace(lon_min, lon_max, Nx + 1)
        lon_c_np = 0.5 * (lon_edges[:-1] + lon_edges[1:])

        # u-faces sit at the east edge of each tracer cell
        lon_u_np = lon_edges[1:].copy()

        # --- build 1-D latitude at tracer centres --------------------------
        if lat_spacing is not None:
            w = np.asarray(lat_spacing, dtype=np.float64)
            assert len(w) == Ny
            w = w / w.sum()
            lat_edges = np.concatenate([[lat_min], lat_min + np.cumsum(w) * (lat_max - lat_min)])
        else:
            lat_edges = np.linspace(lat_min, lat_max, Ny + 1)
        lat_c_np = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        lat_v_np = lat_edges[1:].copy()

        # --- vertical coordinates ------------------------------------------
        z_c_np = depth_levels.copy()
        z_w_np = face_depths(z_c_np)

        # cell thicknesses
        dz_c_np = np.diff(z_w_np)              # (Nz,)  positive downward
        # distance between consecutive cell centres (for gradient in z)
        dz_w_np = np.empty(Nz + 1, dtype=np.float64)
        dz_w_np[0] = z_c_np[0]                 # surface to first center
        dz_w_np[1:Nz] = np.diff(z_c_np)
        dz_w_np[Nz] = z_w_np[Nz] - z_c_np[-1] # last center to bottom

        # --- 2-D horizontal spacings in metres ------------------------------
        d_lambda = np.diff(lon_edges) * DEG2RAD           # (Nx,) in radians
        d_phi_c  = (lat_edges[1:] - lat_edges[:-1]) * DEG2RAD  # (Ny,) at tracer centre

        lat_c_rad = lat_c_np * DEG2RAD   # (Ny,)
        lat_v_rad = lat_v_np * DEG2RAD   # (Ny,) at v-face (north edge)

        # dx_c[i,j] = R * cos(lat_c[j]) * dlambda[i]
        dx_c_np = EARTH_RADIUS * np.outer(d_lambda, np.cos(lat_c_rad))      # (Nx, Ny)
        dy_c_np = EARTH_RADIUS * np.tile(d_phi_c, (Nx, 1))                  # (Nx, Ny)

        # dx_u: u sits on the east face at the same latitude row as the tracer cell
        dx_u_np = dx_c_np.copy()
        # dy_v: meridional distance between consecutive tracer centres (used for grad_y)
        dy_v_np = EARTH_RADIUS * np.tile(
            (lat_edges[1:] - lat_edges[:-1]) * DEG2RAD, (Nx, 1)
        )                                                                     # (Nx, Ny)
        # dx_v: zonal width of the north face, evaluated at lat_v (not lat_c)
        dx_v_np = EARTH_RADIUS * np.outer(d_lambda, np.cos(lat_v_rad))       # (Nx, Ny)

        # --- Coriolis -------------------------------------------------------
        f_c_np = 2.0 * OMEGA * np.sin(lat_c_rad)          # (Ny,)
        f_c_np = np.tile(f_c_np, (Nx, 1))                  # (Nx, Ny)

        # --- area / volume --------------------------------------------------
        area_c_np   = dx_c_np * dy_c_np                    # (Nx, Ny)
        volume_c_np = area_c_np[:, :, np.newaxis] * dz_c_np[np.newaxis, np.newaxis, :]  # (Nx,Ny,Nz)

        # --- bathymetry & masks --------------------------------------------
        if bathymetry is None:
            H_np = np.full((Nx, Ny), z_w_np[-1], dtype=np.float64)
        else:
            H_np = np.asarray(bathymetry, dtype=np.float64)
            assert H_np.shape == (Nx, Ny), f"bathymetry shape {H_np.shape} != ({Nx},{Ny})"

        # mask_c: ocean if cell top face (z_w[k]) < H(i,j)
        z_w_top = z_w_np[:-1]                              # top face of each cell (Nz,)
        mask_c_np = (z_w_top[np.newaxis, np.newaxis, :] < H_np[:, :, np.newaxis]).astype(np.float32)

        # u-mask: ocean if both i and i+1 tracer cells are ocean (periodic in x)
        mask_u_np = mask_c_np * np.roll(mask_c_np, -1, axis=0)
        if not periodic_x:
            mask_u_np[-1, :, :] = 0.0  # east/west walls: no wrap-around face
        # v-mask: ocean if both j and j+1 tracer cells are ocean
        mask_v_np = mask_c_np * np.roll(mask_c_np, -1, axis=1)
        mask_v_np[:, -1, :] = 0.0  # no flow at northern boundary

        # w-mask: (Nx, Ny, Nz+1)
        # Surface face (k=0): open wherever the surface tracer cell is ocean.
        # Internal face (1 <= k <= Nz-1): open only when BOTH adjacent tracer cells are wet.
        # Seafloor face (k=Nz): always closed (no-normal-flow bottom BC).
        mask_w_np = np.zeros((Nx, Ny, Nz + 1), dtype=np.float32)
        mask_w_np[:, :, 0]       = mask_c_np[:, :, 0]
        mask_w_np[:, :, 1:Nz]   = mask_c_np[:, :, :-1] * mask_c_np[:, :, 1:]
        mask_w_np[:, :, Nz]      = 0.0

        # mask_w_adv: surface face (k=0) hard-walled for tracer advection
        mask_w_adv_np = mask_w_np.copy()
        mask_w_adv_np[:, :, 0] = 0.0

        # --- assemble -------------------------------------------------------
        return OceanGrid(
            Nx=Nx, Ny=Ny, Nz=Nz,
            lon_c=jnp.array(lon_c_np, dtype=jnp.float32),
            lat_c=jnp.array(lat_c_np, dtype=jnp.float32),
            z_c  =jnp.array(z_c_np,   dtype=jnp.float32),
            lon_u=jnp.array(lon_u_np, dtype=jnp.float32),
            lat_v=jnp.array(lat_v_np, dtype=jnp.float32),
            z_w  =jnp.array(z_w_np,   dtype=jnp.float32),
            dx_c =jnp.array(dx_c_np,  dtype=jnp.float32),
            dy_c =jnp.array(dy_c_np,  dtype=jnp.float32),
            dx_u =jnp.array(dx_u_np,  dtype=jnp.float32),
            dy_v =jnp.array(dy_v_np,  dtype=jnp.float32),
            dx_v =jnp.array(dx_v_np,  dtype=jnp.float32),
            dz_c =jnp.array(dz_c_np,  dtype=jnp.float32),
            dz_w =jnp.array(dz_w_np,  dtype=jnp.float32),
            f_c  =jnp.array(f_c_np,   dtype=jnp.float32),
            area_c  =jnp.array(area_c_np,   dtype=jnp.float32),
            volume_c=jnp.array(volume_c_np, dtype=jnp.float32),
            mask_c=jnp.array(mask_c_np, dtype=jnp.float32),
            mask_u=jnp.array(mask_u_np, dtype=jnp.float32),
            mask_v=jnp.array(mask_v_np, dtype=jnp.float32),
            mask_w    =jnp.array(mask_w_np,     dtype=jnp.float32),
            mask_w_adv=jnp.array(mask_w_adv_np, dtype=jnp.float32),
            H=jnp.array(H_np, dtype=jnp.float32),
            periodic_x=periodic_x,
        )
