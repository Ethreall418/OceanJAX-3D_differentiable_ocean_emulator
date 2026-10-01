"""
OceanJAX Physics – Mixing Module
==================================
Vertical and horizontal mixing parameterisations.

Responsibility contract
-----------------------
Every function returns either a **tendency** [tracer s⁻¹ or m s⁻²] or a
**diffusivity field** [m² s⁻¹].  No time integration is performed here.

Vertical diffusion is treated **implicitly**: the time stepper calls
``implicit_vertical_mix`` which solves the tridiagonal system

  (I - dt * L_v) phi^{n+1} = phi^n + dt * explicit_tend

where L_v is the vertical diffusion operator, and returns phi^{n+1}
directly.  This avoids the severe stability constraint that would arise
from an explicit vertical  diffusion step.

Horizontal viscosity and diffusion are treated **explicitly** and return
tendencies that the time stepper adds before advancing.

Contents
--------
thomas_algorithm          – differentiable tridiagonal solver via lax.scan
_solve_increment          – solves for the increment (no float32 bias on
                            uniform columns)
implicit_vertical_mix     – implicit vertical diffusion for tracers
implicit_vertical_visc    – implicit vertical diffusion for velocities (u or v)
                            uses velocity-consistent vertical face masks;
                            optional implicit bottom drag
bottom_cell_mask          – indicator of each column's deepest wet cell
bottom_drag_velocity      – quadratic drag velocity Cd*|u_b| at u/v points
_laplacian_u / _v         – scalar Laplacian at u- / v-points with correct metrics
horizontal_viscosity      – Laplacian viscosity tendency for (u, v)
munk_viscosity            – resolution-dependent nu_h resolving the Munk layer
buoyancy_and_shear        – N² and S² aligned at tracer w-faces
richardson_number         – gradient Richardson number (diagnostic, unclipped)
pp81_coefficients         – Pacanowski–Philander (1981) nu, kappa + convection
kpp_coefficients          – KPP surface boundary layer (LMD94) on top of PP81,
                            with nonlocal tracer transport
kpp_boundary_layer_depth  – diagnostic KPP boundary-layer depth
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from OceanJAX.grid import OceanGrid


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def _diff_w(phi: jnp.ndarray) -> jnp.ndarray:
    """
    Centred vertical difference of phi at w-faces, shape (Nx, Ny, Nz+1).
    Interior faces k=1..Nz-1 get phi[k] - phi[k-1]; boundary faces are zero.
    """
    interior = phi[..., 1:] - phi[..., :-1]
    zeros    = jnp.zeros(phi.shape[:2] + (1,), dtype=phi.dtype)
    return jnp.concatenate([zeros, interior, zeros], axis=-1)


# ---------------------------------------------------------------------------
# Tridiagonal solver (Thomas algorithm) – differentiable via lax.scan
# ---------------------------------------------------------------------------

def thomas_algorithm(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray,
) -> jnp.ndarray:
    """
    Solve the tridiagonal system  A x = d  using the Thomas algorithm.

    The system has the form:

      b[0] x[0] + c[0] x[1]                          = d[0]
      a[k] x[k-1] + b[k] x[k] + c[k] x[k+1]         = d[k]   1 ≤ k ≤ N-2
                    a[N-1] x[N-2] + b[N-1] x[N-1]    = d[N-1]

    Args:
        a : (N,) lower diagonal  (a[0] is unused)
        b : (N,) main  diagonal
        c : (N,) upper diagonal  (c[N-1] is unused)
        d : (N,) right-hand side

    Returns:
        x : (N,) solution

    Implementation uses ``jax.lax.scan`` so the solver is fully
    differentiable via reverse-mode AD and JIT-compilable.
    No Python loops over array values.
    """
    N = b.shape[0]

    # ---- Forward sweep: eliminate lower diagonal ---------------------------
    def fwd_step(carry, k):
        b_prev, d_prev = carry          # modified b and d from previous row
        w   = a[k] / b_prev            # elimination factor
        b_k = b[k] - w * c[k - 1]     # modified main diagonal
        d_k = d[k] - w * d_prev       # modified RHS
        return (b_k, d_k), (b_k, d_k)

    # Initialise with row 0
    init    = (b[0], d[0])
    # Scan over rows 1..N-1
    _, (b_mod, d_mod) = jax.lax.scan(fwd_step, init, jnp.arange(1, N))

    # Concatenate row 0 back
    b_all = jnp.concatenate([b[:1], b_mod])   # (N,)
    d_all = jnp.concatenate([d[:1], d_mod])   # (N,)

    # ---- Back substitution -------------------------------------------------
    def bwd_step(x_next, k):
        x_k = (d_all[k] - c[k] * x_next) / b_all[k]
        return x_k, x_k

    # Initialise with last row
    x_last = d_all[-1] / b_all[-1]
    _, x_interior = jax.lax.scan(
        bwd_step, x_last, jnp.arange(N - 2, -1, -1)
    )
    # x_interior is reversed (N-1 entries); append x_last and flip
    x = jnp.concatenate([x_interior[::-1], jnp.array([x_last])])
    return x


def _vmap_columns(solve_column):
    """
    Map a single-column solver over the (i, j) axes of (Nx, Ny, ...) inputs.

    Nested vmaps (rather than reshaping to (Nx*Ny, Nz)) keep x and y as
    separate array axes, so a field sharded over a device mesh in x and y
    stays sharded: merging two sharded axes in a reshape would force XLA
    to all-gather the whole field onto every device.  The per-column
    arithmetic is identical either way.
    """
    return jax.vmap(jax.vmap(solve_column))


def _build_tridiag_implicit(kappa: jnp.ndarray, dz_c: jnp.ndarray,
                             dz_w: jnp.ndarray, dt: float,
                             mask_w_col: jnp.ndarray) -> tuple:
    """
    Build the tridiagonal coefficients for implicit vertical diffusion
    of a single water column.

    Discretisation of  -d/dz(kappa * dC/dz) at cell centres:

      flux_k   = kappa[k] / dz_w[k]    (flux coefficient at w-face k)

    The implicit system for column update C^{n+1}:

      C^{n+1}[k] - dt * (flux_{k+1} * C^{n+1}[k+1]
                        - (flux_k + flux_{k+1}) * C^{n+1}[k]
                        + flux_k * C^{n+1}[k-1]) / dz_c[k]
      = C^n[k]

    Boundary conditions (via mask_w_col):
      surface face k=0  : flux = 0 (Neumann, surface forcing handled separately)
      bottom  face k=Nz : flux = 0 (Neumann, no-flux seafloor)

    Args:
        kappa      : (Nz+1,) diffusivity at w-faces [m² s⁻¹]
        dz_c       : (Nz,)   cell thicknesses [m]
        dz_w       : (Nz+1,) distances between cell centres [m]
        dt         : timestep [s]
        mask_w_col : (Nz+1,) w-face mask for this column

    Returns:
        (a, b, c) tridiagonal coefficients, each (Nz,)
    """
    Nz = dz_c.shape[0]

    # Flux coefficients at each w-face [s⁻¹ equivalent]
    # Safe division: dz_w > 0 everywhere except possibly boundary
    safe_dz_w = jnp.where(dz_w > 0, dz_w, 1.0)
    flux = kappa * mask_w_col / safe_dz_w          # (Nz+1,)

    flux_top = flux[:Nz]    # face k   (top of cell k)
    flux_bot = flux[1:]     # face k+1 (bottom of cell k)

    # Lower diagonal: a[k] = -dt * flux_top[k] / dz_c[k]   (k=1..Nz-1)
    a = -dt * flux_top / dz_c                      # (Nz,)  a[0] unused
    # Upper diagonal: c[k] = -dt * flux_bot[k] / dz_c[k]   (k=0..Nz-2)
    c = -dt * flux_bot / dz_c                      # (Nz,)  c[Nz-1] unused
    # Main diagonal: b[k] = 1 - a[k] - c[k]
    b = 1.0 - a - c                                # (Nz,)

    return a, b, c


def _solve_increment(
    a:  jnp.ndarray,
    c:  jnp.ndarray,
    e:  jnp.ndarray,
    x0: jnp.ndarray,
) -> jnp.ndarray:
    """
    Solve  A x = x0  for one column, where A = tridiag(a, 1 - a - c + e, c)
    is an implicit diffusion operator (a, c from ``_build_tridiag_implicit``)
    plus a non-negative extra diagonal e (e.g. implicit bottom drag; pass
    zeros for pure diffusion).

    Solving for the full field accumulates a systematic float32 bias: for a
    uniform column the forward sweep rounds the bottom row so that x[-1]
    comes out one ulp high on every call (+3.8e-6 psu per step at S=35,
    +0.033 psu in 30 days).  Instead we solve for the increment
    delta = x - x0, whose right-hand side is formed from differences only:

      A delta = x0 - A x0 = -( a (x0[k-1] - x0[k]) + c (x0[k+1] - x0[k]) + e x0[k] )

    A uniform column (with e = 0) gives a right-hand side that is exactly
    zero, hence delta = 0 exactly, and rounding errors elsewhere scale with
    the increment rather than with the field itself.
    """
    b      = 1.0 - a - c + e
    x_up   = jnp.concatenate([x0[:1],  x0[:-1]])            # x0[k-1] (a[0] = 0)
    x_dn   = jnp.concatenate([x0[1:],  x0[-1:]])            # x0[k+1] (c[-1] = 0)
    rhs    = -(a * (x_up - x0) + c * (x_dn - x0) + e * x0)
    return x0 + thomas_algorithm(a, b, c, rhs)


# ---------------------------------------------------------------------------
# Implicit vertical diffusion for tracers
# ---------------------------------------------------------------------------

def implicit_vertical_mix(
    phi:   jnp.ndarray,
    kappa: jnp.ndarray,
    dt:    float,
    grid:  OceanGrid,
    rhs_explicit: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Implicitly mix a tracer field in the vertical direction.

    Solves column-by-column:

      (I - dt * L_v) phi^{n+1} = phi^n + dt * rhs_explicit

    where L_v is the vertical diffusion operator.  A separate call to
    ``thomas_algorithm`` is made for each (i, j) column via nested
    ``jax.vmap`` (see ``_vmap_columns``).

    Args:
        phi          : (Nx, Ny, Nz)    tracer field at time n
        kappa        : (Nx, Ny, Nz+1)  vertical diffusivity at w-faces [m² s⁻¹]
        dt           : timestep [s]
        grid         : OceanGrid
        rhs_explicit : (Nx, Ny, Nz) or None
                       Explicit tendency already accumulated for this timestep.
                       If None, treated as zero.

    Returns:
        phi^{n+1} : (Nx, Ny, Nz), masked to zero on dry cells
    """
    if rhs_explicit is None:
        rhs = phi
    else:
        rhs = phi + dt * rhs_explicit

    # Accept scalar kappa (e.g. params.kappa_v) and broadcast to (Nx, Ny, Nz+1)
    Nx, Ny, Nz = phi.shape
    if jnp.ndim(kappa) == 0:
        kappa = jnp.full((Nx, Ny, Nz + 1), kappa)

    def solve_column(phi_col, kappa_col, rhs_col, mask_w_col, mask_c_col):
        """Solve one (i,j) column. All inputs are 1-D in z."""
        a, _, c = _build_tridiag_implicit(
            kappa_col, grid.dz_c, grid.dz_w, dt, mask_w_col
        )
        # For dry cells, the system degenerates; keep phi = 0 there.
        # The mask on the diagonal (b=1 for dry cells, a=c=0) achieves this
        # naturally since rhs = 0 for dry cells.
        phi_new = _solve_increment(a, c, jnp.zeros_like(a), rhs_col * mask_c_col)
        return phi_new * mask_c_col

    # Use mask_w_adv (surface face k=0 always closed) so that the implicit
    # diffusion operator has no flux through the sea surface.  The surface
    # tracer exchange is handled exclusively by the explicit forcing tendencies
    # in tracers.py, exactly as for tracer advection.  Using mask_w here would
    # open a spurious diffusive flux at k=0 (a[0] ≠ 0 in the tridiagonal
    # system), corrupting the top-layer temperature even in a resting ocean.
    return _vmap_columns(solve_column)(
        phi, kappa, rhs, grid.mask_w_adv, grid.mask_c
    )


# ---------------------------------------------------------------------------
# Implicit vertical diffusion for velocity (u or v)
# ---------------------------------------------------------------------------

def implicit_vertical_visc(
    vel:   jnp.ndarray,
    nu_v:  jnp.ndarray,
    dt:    float,
    grid:  OceanGrid,
    mask:  jnp.ndarray,
    drag:  jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Implicitly mix a horizontal velocity component in the vertical direction,
    optionally with an implicit bottom drag.

    Uses a velocity-consistent vertical face mask: internal face k is active
    only when both ``mask[..., k-1]`` and ``mask[..., k]`` are 1, matching the
    actual grid locations of u or v rather than the tracer mask_w.  Using the
    tracer mask_w would incorrectly couple layers at seamount edges where a
    u- or v-column is entirely dry despite adjacent tracer columns being wet.

    Bottom drag enters as a linear damping rate on the deepest wet cell of
    each column,  d(vel)/dt = -r * vel  with  r = drag / dz_c[k_bot]  [s⁻¹],
    and is added to the diagonal of the implicit system, so it is
    unconditionally stable for any drag coefficient.

    Args:
        vel   : (Nx, Ny, Nz)    velocity component at time n
        nu_v  : (Nx, Ny, Nz+1) vertical viscosity at w-faces [m² s⁻¹]
        dt    : timestep [s]
        grid  : OceanGrid
        mask  : (Nx, Ny, Nz)   wet mask for this velocity component
                                (mask_u for u, mask_v for v)
        drag  : (Nx, Ny) bottom drag velocity [m s⁻¹] (e.g. Cd*|u_bot| for
                quadratic drag), or None for a free-slip bottom.

    Returns:
        vel^{n+1} : (Nx, Ny, Nz)
    """
    Nx, Ny, Nz = vel.shape

    # Accept scalar nu_v (e.g. params.nu_v) and broadcast to (Nx, Ny, Nz+1)
    if jnp.ndim(nu_v) == 0:
        nu_v = jnp.full((Nx, Ny, Nz + 1), nu_v)

    # Build the vertical face mask consistent with this velocity component.
    # Face k is open only when both adjacent velocity layers are wet.
    # Surface (k=0) and bottom (k=Nz) are always closed (no-flux BC).
    mask_w_vel = jnp.zeros((Nx, Ny, Nz + 1), dtype=mask.dtype)
    mask_w_vel = mask_w_vel.at[:, :, 1:Nz].set(mask[:, :, :-1] * mask[:, :, 1:])

    # Extra diagonal: dt * drag / dz on the deepest wet cell of each column
    if drag is None:
        e = jnp.zeros((Nx, Ny, Nz), dtype=vel.dtype)
    else:
        e = (dt * drag[:, :, jnp.newaxis] / grid.dz_c) * bottom_cell_mask(mask)

    def solve_column(vel_col, nu_col, e_col, mask_w_col, mask_col):
        a, _, c = _build_tridiag_implicit(
            nu_col, grid.dz_c, grid.dz_w, dt, mask_w_col
        )
        return _solve_increment(a, c, e_col, vel_col * mask_col) * mask_col

    return _vmap_columns(solve_column)(vel, nu_v, e, mask_w_vel, mask)


def bottom_cell_mask(mask: jnp.ndarray) -> jnp.ndarray:
    """
    (Nx, Ny, Nz) indicator of the deepest wet cell of each column: 1 where
    ``mask[..., k] = 1`` and the cell below is dry (or k = Nz-1).
    """
    below = jnp.concatenate(
        [mask[..., 1:], jnp.zeros_like(mask[..., :1])], axis=-1
    )
    return mask * (1.0 - below)


def bottom_drag_velocity(
    u:      jnp.ndarray,
    v:      jnp.ndarray,
    grid:   OceanGrid,
    params,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Quadratic bottom-drag velocity  Cd * sqrt(|u_b|^2 + u_bg^2)  at u- and
    v-points, (Nx, Ny) each [m s⁻¹].

    u_b is the velocity of the deepest wet cell.  The cross component is
    interpolated with the same 4-point averages as the Coriolis term, at
    the same level.  u_bg (``params.bottom_drag_ubg``) is a background
    speed standing in for unresolved tides and eddies, so the drag stays
    active for weak flows.  With ``params.bottom_drag_cd = 0`` the result
    is exactly zero.
    """
    from OceanJAX.Physics.dynamics import v_at_u_points, u_at_v_points   # deferred

    bot_u = bottom_cell_mask(grid.mask_u)
    bot_v = bottom_cell_mask(grid.mask_v)
    ub      = jnp.sum(u * bot_u, axis=-1)                           # (Nx, Ny)
    vb      = jnp.sum(v * bot_v, axis=-1)
    vb_at_u = jnp.sum(v_at_u_points(v, grid) * bot_u, axis=-1)
    ub_at_v = jnp.sum(u_at_v_points(u, grid) * bot_v, axis=-1)

    ubg2   = params.bottom_drag_ubg ** 2
    drag_u = params.bottom_drag_cd * jnp.sqrt(ub ** 2 + vb_at_u ** 2 + ubg2)
    drag_v = params.bottom_drag_cd * jnp.sqrt(ub_at_v ** 2 + vb ** 2 + ubg2)
    return drag_u, drag_v


# ---------------------------------------------------------------------------
# Horizontal viscosity (explicit Laplacian at velocity points)
# ---------------------------------------------------------------------------

def _laplacian_u(
    u:    jnp.ndarray,
    nu_h: float | jnp.ndarray,
    grid: OceanGrid,
) -> jnp.ndarray:
    """
    Scalar horizontal Laplacian of u at u-points (east faces).

    Uses the geometry of the u-point grid cell:

      x-direction
        Neighbours : u[i-1] and u[i+1]  (adjacent east faces)
        Spacing    : dx_c[i]   (west gap) and dx_c[i+1] (east gap)
        Face height: dy_c[j]
        Cell area  : 0.5 * (dx_c[i] + dx_c[i+1]) * dy_c[j]

      y-direction
        Neighbours : u[i,j-1] and u[i,j+1]  (same east face, adjacent rows)
        Spacing    : dy_v[j]  (= dy_c[j], distance between tracer-centre rows)
        Face width : dx_v[j]  (zonal width at north-face latitude lat_v[j])

    Face masks gate fluxes through faces that adjoin a dry u-point.
    Output is zeroed at dry u-points via mask_u.

    This is a scalar approximation; the full vector Laplacian on a spherical
    C-grid also includes off-diagonal stress terms that are neglected here.
    """
    Nx, Ny, Nz = u.shape

    # ---- x-direction -------------------------------------------------------
    u_e      = jnp.roll(u, -1, axis=0)                            # u[i+1]
    # Distance from u[i] to u[i+1] = width of tracer cell i+1
    dist_e   = jnp.roll(grid.dx_c, -1, axis=0)[:, :, jnp.newaxis]
    # Gate: both u[i] and u[i+1] must be active (periodic in x)
    mu_ee    = grid.mask_u * jnp.roll(grid.mask_u, -1, axis=0)
    fx_e     = nu_h * grid.dy_c[:, :, jnp.newaxis] * (u_e - u) / dist_e * mu_ee
    fx_w     = jnp.roll(fx_e, 1, axis=0)

    # ---- y-direction -------------------------------------------------------
    u_n      = jnp.roll(u, -1, axis=1)                            # u[i,j+1]
    # Gate: both u-rows active; north wall explicitly closed
    mu_nn    = (grid.mask_u * jnp.roll(grid.mask_u, -1, axis=1)
                ).at[:, -1, :].set(0.0)
    fy_n     = nu_h * grid.dx_v[:, :, jnp.newaxis] * (u_n - u) / grid.dy_v[:, :, jnp.newaxis] * mu_nn
    fy_s     = jnp.concatenate(
        [jnp.zeros((Nx, 1, Nz), dtype=u.dtype), fy_n[:, :-1, :]], axis=1
    )

    # ---- u-cell area -------------------------------------------------------
    area_u = 0.5 * (grid.dx_c + jnp.roll(grid.dx_c, -1, axis=0)) * grid.dy_c

    return ((fx_e - fx_w + fy_n - fy_s) / area_u[:, :, jnp.newaxis]) * grid.mask_u


def _laplacian_v(
    v:    jnp.ndarray,
    nu_h: float | jnp.ndarray,
    grid: OceanGrid,
) -> jnp.ndarray:
    """
    Scalar horizontal Laplacian of v at v-points (north faces).

    Uses the geometry of the v-point grid cell:

      x-direction
        Neighbours : v[i-1,j] and v[i+1,j]  (adjacent north faces)
        Spacing    : 0.5*(dx_v[i] + dx_v[i+1])  (distance between v-columns)
        Face height: dy_v[j]
        Cell area  : dx_v[j] * dy_v[j]

      y-direction
        Neighbours : v[i,j-1] and v[i,j+1]  (same north face, adjacent rows)
        v[j] at lat_v[j]; v[j+1] at lat_v[j+1]
        Spacing    : dy_c[j+1]  (height of tracer cell j+1)
        Face width : dx_c[j+1]  (zonal width at lat_c[j+1], midpoint of v-gap)

    Face masks gate fluxes through faces that adjoin a dry v-point.
    Output is zeroed at dry v-points via mask_v.
    """
    Nx, Ny, Nz = v.shape

    # ---- x-direction -------------------------------------------------------
    v_e      = jnp.roll(v, -1, axis=0)
    # Distance between v[i] and v[i+1] ≈ 0.5*(dx_v[i] + dx_v[i+1])
    dist_e   = 0.5 * (grid.dx_v + jnp.roll(grid.dx_v, -1, axis=0))[:, :, jnp.newaxis]
    mv_ee    = grid.mask_v * jnp.roll(grid.mask_v, -1, axis=0)
    if not grid.periodic_x:
        # This flux is gated by mask_v, not mask_u, so the east/west wall
        # must be closed explicitly: no v[Nx-1] <-> v[0] wrap-around.
        mv_ee = mv_ee.at[-1, :, :].set(0.0)
    fx_e     = nu_h * grid.dy_v[:, :, jnp.newaxis] * (v_e - v) / dist_e * mv_ee
    fx_w     = jnp.roll(fx_e, 1, axis=0)

    # ---- y-direction -------------------------------------------------------
    v_n      = jnp.roll(v, -1, axis=1)                            # v[i,j+1]
    # Distance from v[j] (lat_v[j]) to v[j+1] (lat_v[j+1]) = dy_c[j+1]
    dist_n   = jnp.roll(grid.dy_c, -1, axis=1)[:, :, jnp.newaxis]
    # Face width at midpoint between v-rows ≈ dx_c[j+1]
    width_n  = jnp.roll(grid.dx_c, -1, axis=1)[:, :, jnp.newaxis]
    mv_nn    = (grid.mask_v * jnp.roll(grid.mask_v, -1, axis=1)
                ).at[:, -1, :].set(0.0)
    fy_n     = nu_h * width_n * (v_n - v) / dist_n * mv_nn
    fy_s     = jnp.concatenate(
        [jnp.zeros((Nx, 1, Nz), dtype=v.dtype), fy_n[:, :-1, :]], axis=1
    )

    # ---- v-cell area -------------------------------------------------------
    area_v = grid.dx_v * grid.dy_v   # dx_v[j] * dy_c[j]

    return ((fx_e - fx_w + fy_n - fy_s) / area_v[:, :, jnp.newaxis]) * grid.mask_v


def horizontal_viscosity(
    u:    jnp.ndarray,
    v:    jnp.ndarray,
    nu_h: float | jnp.ndarray,
    grid: OceanGrid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Explicit horizontal Laplacian viscosity tendencies for (u, v).

    Delegates to ``_laplacian_u`` and ``_laplacian_v``, which use the correct
    grid-cell geometry and masks for each velocity point rather than the
    tracer-cell metrics used by ``kappa_laplacian_h``.

    This is a scalar Laplacian approximation.  The full vector Laplacian on
    a spherical C-grid includes off-diagonal stress terms that are omitted
    here; they are second-order corrections relevant mainly at very high
    resolution or when modelling viscous boundary layers.

    Args:
        u    : (Nx, Ny, Nz) zonal velocity at east faces
        v    : (Nx, Ny, Nz) meridional velocity at north faces
        nu_h : scalar or (Nx, Ny, Nz) horizontal viscosity [m² s⁻¹]
        grid : OceanGrid

    Returns:
        (du_dt_visc, dv_dt_visc) : each (Nx, Ny, Nz)
    """
    return _laplacian_u(u, nu_h, grid), _laplacian_v(v, nu_h, grid)


def munk_viscosity(grid: OceanGrid, n_points: float = 1.0) -> float:
    """
    Horizontal eddy viscosity [m² s⁻¹] that resolves the Munk western
    boundary layer on this grid.

    nu_h in a coarse model is not the molecular viscosity of seawater
    (~1e-6 m² s⁻¹) but a closure for momentum mixing by unresolved motions,
    so it must scale with resolution.  The Munk layer width is

        delta_M = (nu_h / beta)^(1/3),

    and if delta_M is narrower than the grid spacing the discrete western
    boundary current degenerates into 2-dx noise that grows without bound
    (Bryan, Manabe & Pacanowski 1975).  Requiring delta_M >= n_points * dx
    gives

        nu_h >= beta(phi) * (n_points * dx(phi))^3.

    Because beta ∝ cos(phi) and dx ∝ cos(phi), the bound ∝ cos^4(phi) is
    largest at the latitude closest to the equator; the maximum over all
    wet columns is returned as a single domain-wide value.

    Args:
        grid     : OceanGrid (uses lat_c, dx_c, mask_c)
        n_points : number of grid points across the Munk layer (default 1)

    Returns:
        nu_h as a Python float, ready for ``ModelParams(nu_h=...)``.
    """
    from OceanJAX.grid import EARTH_RADIUS, OMEGA, DEG2RAD

    lat  = np.asarray(grid.lat_c, dtype=np.float64) * DEG2RAD             # (Ny,)
    beta = 2.0 * OMEGA * np.cos(lat) / EARTH_RADIUS                       # (Ny,)
    dx   = np.asarray(grid.dx_c, dtype=np.float64)                        # (Nx, Ny)
    wet  = np.asarray(grid.mask_c)[:, :, 0] > 0
    bound = beta[np.newaxis, :] * (n_points * dx) ** 3
    return float(bound[wet].max()) if np.any(wet) else float(bound.max())


# ---------------------------------------------------------------------------
# Richardson number and Pacanowski-Philander (1981) vertical mixing
# ---------------------------------------------------------------------------

# Floor on S² [s⁻²]: with no shear, Ri = N² / floor is huge and PP81 falls
# back to its background values.
_S2_FLOOR: float = 1e-12


def _face_mask(mask: jnp.ndarray) -> jnp.ndarray:
    """(Nx, Ny, Nz+1): interior face k open when layers k-1 and k are both
    wet in ``mask``; surface and seafloor faces closed."""
    interior = mask[..., :-1] * mask[..., 1:]
    zeros    = jnp.zeros(mask.shape[:2] + (1,), dtype=mask.dtype)
    return jnp.concatenate([zeros, interior, zeros], axis=-1)


def buoyancy_and_shear(
    T:    jnp.ndarray,
    S:    jnp.ndarray,
    u:    jnp.ndarray,
    v:    jnp.ndarray,
    grid: OceanGrid,
    params,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    N² and S² [s⁻²] at the w-faces of tracer columns, each (Nx, Ny, Nz+1).

    z is positive downward (k increases with depth), so a stably stratified
    column has rho[k] > rho[k-1] and

      N²[k] = (g / rho0) * (rho[k] - rho[k-1]) / dz_w[k]      (> 0 stable)

    For the linear equation of state (``dynamics.equation_of_state``)
    rho[k] - rho[k-1] = rho0 * (-alpha_T dT + beta_S dS) exactly, so N² is
    evaluated as

      N²[k] = g * (-alpha_T (T[k] - T[k-1]) + beta_S (S[k] - S[k-1])) / dz_w[k]

    Differencing T and S directly avoids subtracting two ~1025 kg m⁻³
    densities in float32, which leaves a round-off noise of ~1e-8 s⁻² in N²
    (at dz = 50 m); here it is ~1e-10 s⁻².  This must be revisited if the
    equation of state becomes nonlinear.

    Shear is formed where u and v live, squared, and averaged onto the
    tracer column over its wet neighbouring faces, so that N² and S² sit at
    the same point:

      S²_c = mean_{i±1/2} (du/dz)²  +  mean_{j±1/2} (dv/dz)²

    Averaging over wet faces only keeps coastal columns from being diluted
    by the zero velocity of land faces.  Surface, seafloor and dry faces
    are zero.
    """
    safe_dz_w = jnp.where(grid.dz_w > 0, grid.dz_w, 1.0)
    db  = params.g * (params.beta_S * _diff_w(S) - params.alpha_T * _diff_w(T))
    n2  = db / safe_dz_w * grid.mask_w

    mu = _face_mask(grid.mask_u)
    mv = _face_mask(grid.mask_v)
    su2 = (_diff_w(u * grid.mask_u) / safe_dz_w) ** 2 * mu
    sv2 = (_diff_w(v * grid.mask_v) / safe_dz_w) ** 2 * mv

    # u faces of tracer column i: i+1/2 (index i) and i-1/2 (index i-1)
    su2_w, mu_w = jnp.roll(su2, 1, axis=0), jnp.roll(mu, 1, axis=0)
    # v faces of tracer column j: j+1/2 (index j) and j-1/2 (index j-1, wall at j=0)
    pad = lambda a: jnp.concatenate([jnp.zeros_like(a[:, :1]), a[:, :-1]], axis=1)
    sv2_s, mv_s = pad(sv2), pad(mv)

    s2 = ((su2 + su2_w) / jnp.maximum(mu + mu_w, 1.0)
          + (sv2 + sv2_s) / jnp.maximum(mv + mv_s, 1.0))
    return n2, s2 * grid.mask_w


def richardson_number(
    T:    jnp.ndarray,
    S:    jnp.ndarray,
    u:    jnp.ndarray,
    v:    jnp.ndarray,
    grid: OceanGrid,
    params,
) -> jnp.ndarray:
    """
    Gradient Richardson number Ri = N² / S² at tracer w-faces (Nx, Ny, Nz+1).

    Diagnostic, unclipped: negative Ri marks static instability.  S² is
    floored at 1e-12 s⁻² (see ``buoyancy_and_shear`` for the staggering).
    Dry and boundary faces are zero.
    """
    n2, s2 = buoyancy_and_shear(T, S, u, v, grid, params)
    return n2 / jnp.maximum(s2, _S2_FLOOR) * grid.mask_w


def pp81_coefficients(
    T:      jnp.ndarray,
    S:      jnp.ndarray,
    u:      jnp.ndarray,
    v:      jnp.ndarray,
    grid:   OceanGrid,
    params,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Pacanowski & Philander (1981) vertical viscosity and diffusivity.

    For a stably stratified face (N² >= 0), with Ri = N² / S²:

      nu_pp    = nu0 / (1 + alpha Ri)^n            + nu_b
      kappa_pp = nu0 / (1 + alpha Ri)^(n+1)        + kappa_b

    Statically unstable faces (N² < 0) get convective adjustment, blended
    in continuously over  -N²_c <= N² <= 0  (N²_c = ``params.vmix_n2_ramp``):

      x = clip(-N² / N²_c, 0, 1),   w = x² (3 - 2x)        (smoothstep)
      nu    = (1 - w) nu_pp(Ri=0)    + w nu_conv
      kappa = (1 - w) kappa_pp(Ri=0) + w nu_conv

    so nu and kappa are continuous with a continuous derivative in N².
    A hard switch at N² = 0 (the original PP81 convective adjustment) jumps
    from nu0 + nu_b to nu_conv, and round-off in N² of a nearly neutral
    column then flips the branch and changes the mixing by O(0.1 m² s⁻¹).
    Stable faces (w = 0) are unaffected; below -N²_c the full convective
    value applies (stable for any value because vertical mixing is
    implicit).  vmix_n2_ramp = 0 recovers the hard switch.

    On the stable side PP81 itself is steep when there is no shear:
    Ri = N² / max(S², 1e-12) is 0/0 at N² = S² = 0, so kappa falls from
    nu0 + kappa_b to the background within N² ~ 1e-11 s⁻².  That is the
    closure's own Ri dependence and is left unchanged.

    Parameters come from ModelParams: pp81_nu0, pp81_alpha, pp81_n,
    vmix_convective, vmix_n2_ramp, with the constant-mixing values
    nu_v / kappa_v as backgrounds nu_b / kappa_b.

    PP81 is a local shear/stratification closure for the stratified
    interior (designed for the tropical ocean).  It has no surface
    boundary-layer physics (wind-driven deepening, nonlocal convection),
    so mid/high-latitude mixed layers come out too shallow; it depends on
    vertical, not horizontal, resolution.

    Returns
    -------
    kappa : (Nx, Ny, Nz+1) tracer diffusivity at tracer w-faces [m² s⁻¹]
    nu_u  : (Nx, Ny, Nz+1) viscosity at u-column w-faces
    nu_v  : (Nx, Ny, Nz+1) viscosity at v-column w-faces
    """
    n2, s2 = buoyancy_and_shear(T, S, u, v, grid, params)
    kappa, nu = _pp81_tracer_faces(n2, s2, grid, params)
    nu_u, nu_v = _tracer_to_velocity_faces(nu, grid)
    return kappa, nu_u, nu_v


def _pp81_tracer_faces(
    n2:   jnp.ndarray,
    s2:   jnp.ndarray,
    grid: OceanGrid,
    params,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """PP81 + convective blend (see ``pp81_coefficients``): kappa and nu at
    tracer w-faces, each (Nx, Ny, Nz+1)."""
    stable = n2 >= 0.0
    ri     = jnp.where(stable, n2 / jnp.maximum(s2, _S2_FLOOR), 0.0)
    f      = 1.0 / (1.0 + params.pp81_alpha * ri)
    shear  = params.pp81_nu0 * f ** params.pp81_n
    nu_pp    = shear + params.nu_v
    kappa_pp = shear * f + params.kappa_v

    # Convective weight: 0 for N² >= 0, smoothstep up to 1 at N² = -N²_c.
    # The floor on the ramp width turns vmix_n2_ramp = 0 into a hard switch.
    x    = jnp.clip(-n2 / jnp.maximum(params.vmix_n2_ramp, 1e-30), 0.0, 1.0)
    w    = x * x * (3.0 - 2.0 * x)
    conv = params.vmix_convective
    nu    = ((1.0 - w) * nu_pp    + w * conv) * grid.mask_w
    kappa = ((1.0 - w) * kappa_pp + w * conv) * grid.mask_w
    return kappa, nu


def _tracer_to_velocity_faces(
    nu:   jnp.ndarray,
    grid: OceanGrid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Viscosity at u / v columns: mean of the two adjacent tracer columns over
    their wet faces (the velocity solver applies its own face mask).
    """
    mw   = grid.mask_w
    nu_e, mw_e = jnp.roll(nu, -1, axis=0), jnp.roll(mw, -1, axis=0)
    nu_n = jnp.concatenate([nu[:, 1:], nu[:, -1:]], axis=1)
    mw_n = jnp.concatenate([mw[:, 1:], mw[:, -1:]], axis=1)
    nu_u = (nu + nu_e) / jnp.maximum(mw + mw_e, 1.0)
    nu_v = (nu + nu_n) / jnp.maximum(mw + mw_n, 1.0)
    return nu_u, nu_v


# ---------------------------------------------------------------------------
# K-profile parameterisation (Large, McWilliams & Doney 1994, "LMD94")
# ---------------------------------------------------------------------------

_VON_KARMAN = 0.4
_KPP_EPS    = 0.1                   # surface layer = top eps*h of the boundary layer
_KPP_CV     = 1.8                   # N_entrainment / N for the unresolved shear
_KPP_BETA_T = -0.2                  # entrainment / surface buoyancy flux ratio
_KPP_ZETA_M, _KPP_A_M, _KPP_C_M = -0.2, 1.26,   8.38    # momentum, unstable
_KPP_ZETA_S, _KPP_A_S, _KPP_C_S = -1.0, -28.86, 98.96   # scalars,  unstable
_KPP_CSTAR  = 10.0                  # nonlocal transport constant C*
_KPP_EKMAN  = 0.7                   # stable limit h <= 0.7 u*/|f|
_USTAR_MIN  = 1e-5                  # [m s⁻¹] floor on u*, keeps zeta finite

# Nonlocal coefficient C_s = C* kappa (c_s kappa eps)^(1/3)   (~6.33)
_KPP_CS_NL = _KPP_CSTAR * _VON_KARMAN * (_KPP_C_S * _VON_KARMAN * _KPP_EPS) ** (1.0 / 3.0)
# Unresolved turbulent shear  V_t² = coef * d * N * w_s   (LMD94 eq. 23)
_KPP_VT2 = float(_KPP_CV * np.sqrt(-_KPP_BETA_T) / _VON_KARMAN ** 2
                 / np.sqrt(_KPP_C_S * _KPP_EPS))


def kpp_velocity_scales(
    d:     jnp.ndarray,
    ustar: jnp.ndarray,
    bflux: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    KPP turbulent velocity scales w_m, w_s = kappa u* / phi(zeta) [m s⁻¹].

    zeta = d / L with the Monin-Obukhov length L = u*³ / (kappa B0), where
    B0 [m² s⁻³] is the surface buoyancy flux into the ocean (> 0 stable,
    < 0 convective).  The caller limits d to eps*h in unstable conditions.

      stable   (zeta >= 0) : phi_m = phi_s = 1 + 5 zeta
      unstable (zeta <  0) : phi_m = (1 - 16 zeta)^(-1/4)        zeta >= -0.2
                                     (1.26 - 8.38 zeta)^(-1/3)    otherwise
                             phi_s = (1 - 16 zeta)^(-1/2)        zeta >= -1.0
                                     (-28.86 - 98.96 zeta)^(-1/3) otherwise

    The branches join continuously.  In the free-convection limit u* -> 0,
    w_s -> kappa (c_s kappa d |B0|)^(1/3) (finite; u* is floored at 1e-5).
    Each branch is evaluated on a clipped argument so that the unused
    branch of ``jnp.where`` never produces NaN (safe for gradients).
    """
    zeta = _VON_KARMAN * d * bflux / ustar ** 3
    zs   = jnp.maximum(zeta, 0.0)
    zu   = jnp.minimum(zeta, 0.0)
    inv_phi_st = 1.0 / (1.0 + 5.0 * zs)
    inv_phi_m = jnp.where(
        zu >= _KPP_ZETA_M,
        (1.0 - 16.0 * zu) ** 0.25,
        (_KPP_A_M - _KPP_C_M * zu) ** (1.0 / 3.0))
    inv_phi_s = jnp.where(
        zu >= _KPP_ZETA_S,
        (1.0 - 16.0 * zu) ** 0.5,
        (_KPP_A_S - _KPP_C_S * jnp.minimum(zu, _KPP_ZETA_S)) ** (1.0 / 3.0))
    stable = zeta >= 0.0
    w_m = _VON_KARMAN * ustar * jnp.where(stable, inv_phi_st, inv_phi_m)
    w_s = _VON_KARMAN * ustar * jnp.where(stable, inv_phi_st, inv_phi_s)
    return w_m, w_s


def kpp_surface_fluxes(
    forcing,
    sss:    jnp.ndarray,
    params,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Surface quantities for KPP, each (Nx, Ny):

      ustar = (|tau| / rho0)^(1/2)          friction velocity, floored at 1e-5
      F_T   = Q / (rho0 cp)                 downward kinematic heat flux [K m s⁻¹]
      F_S   = SSS (E - P)                   downward kinematic salt flux [psu m s⁻¹]
      B0    = g (alpha_T F_T - beta_S F_S)  buoyancy flux into the ocean [m² s⁻³]

    F_T and F_S are exactly the fluxes the surface forcing applies to the
    top layer (``tracers.heat_surface_tendency`` / ``salt_surface_tendency``
    with the local SSS).  ``forcing=None`` gives u* = floor and zero fluxes.
    """
    from OceanJAX.Physics.tracers import CP_SEAWATER    # deferred

    if forcing is None:
        zeros = jnp.zeros_like(sss)
        return jnp.full_like(sss, _USTAR_MIN), zeros, zeros, zeros
    tau2  = forcing.tau_x ** 2 + forcing.tau_y ** 2
    # (|tau| / rho0)^(1/2) = (tau² / rho0²)^(1/4); the floor also avoids the
    # infinite derivative of the root at zero stress.
    ustar = jnp.maximum(tau2 / params.rho0 ** 2, _USTAR_MIN ** 4) ** 0.25
    f_t   = forcing.heat_flux / (params.rho0 * CP_SEAWATER)
    f_s   = jnp.maximum(sss, 0.0) * forcing.fw_flux
    b0    = params.g * (params.alpha_T * f_t - params.beta_S * f_s)
    return ustar, f_t, f_s, b0


def _centre_velocity(vel: jnp.ndarray, mask: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Mean of the two faces of each tracer cell along ``axis`` (0: u, 1: v),
    over wet faces only (same staggering as ``buoyancy_and_shear``)."""
    vm = vel * mask
    if axis == 0:
        vm_b, m_b = jnp.roll(vm, 1, axis=0), jnp.roll(mask, 1, axis=0)
    else:
        pad = lambda a: jnp.concatenate([jnp.zeros_like(a[:, :1]), a[:, :-1]], axis=1)
        vm_b, m_b = pad(vm), pad(mask)
    return (vm + vm_b) / jnp.maximum(mask + m_b, 1.0)


def _kpp_core(T, S, u, v, forcing, grid: OceanGrid, params) -> dict:
    """Shared KPP computation; see ``kpp_coefficients``."""
    Nz = grid.Nz
    mc = grid.mask_c
    n2, s2 = buoyancy_and_shear(T, S, u, v, grid, params)
    kappa_int, nu_int = _pp81_tracer_faces(n2, s2, grid, params)
    ustar, f_t, f_s, b0 = kpp_surface_fluxes(forcing, S[:, :, 0], params)
    us3, b03 = ustar[..., None], b0[..., None]

    # ---- bulk Richardson number at cell centres (reference = top cell) ----
    u_c = _centre_velocity(u, grid.mask_u, 0)
    v_c = _centre_velocity(v, grid.mask_v, 1)
    d_b = params.g * (params.alpha_T * (T[..., :1] - T)
                      - params.beta_S * (S[..., :1] - S))        # B_r - B(d)
    dv2 = (u_c[..., :1] - u_c) ** 2 + (v_c[..., :1] - v_c) ** 2
    n2c = 0.5 * (n2[..., :-1] + n2[..., 1:])
    n_c = jnp.sqrt(jnp.maximum(n2c, 1e-20))
    d   = jnp.broadcast_to(grid.z_c, T.shape)
    d_eval = jnp.where(b03 < 0.0, _KPP_EPS * d, d)
    _, ws_d = kpp_velocity_scales(d_eval, us3, b03)
    vt2  = _KPP_VT2 * d * n_c * ws_d
    ri_b = d * d_b / (dv2 + vt2 + 1e-10)

    # ---- boundary-layer depth: first centre where Ri_b > Ri_c -------------
    # Dry cells count as a crossing and a sentinel at k = Nz closes full
    # columns; a crossing at a dry index kc means h = its top face, i.e.
    # the column depth.
    ones   = jnp.ones(T.shape[:2] + (1,), dtype=bool)
    exceed = jnp.concatenate([(ri_b > params.kpp_ri_crit) | (mc == 0), ones], axis=-1)
    kc     = jnp.argmax(exceed, axis=-1)                           # (Nx, Ny)
    k_hi   = jnp.minimum(kc, Nz - 1)
    k_lo   = jnp.maximum(kc - 1, 0)
    rb_hi  = jnp.take_along_axis(ri_b, k_hi[..., None], axis=-1)[..., 0]
    rb_lo  = jnp.take_along_axis(ri_b, k_lo[..., None], axis=-1)[..., 0]
    z_hi, z_lo = grid.z_c[k_hi], grid.z_c[k_lo]
    den    = rb_hi - rb_lo
    frac   = jnp.clip((params.kpp_ri_crit - rb_lo)
                      / jnp.where(den > 0.0, den, 1.0), 0.0, 1.0)
    mc_ext = jnp.concatenate([mc, jnp.zeros_like(mc[..., :1])], axis=-1)
    wet_kc = jnp.take_along_axis(mc_ext, kc[..., None], axis=-1)[..., 0] > 0
    h = jnp.where(wet_kc, z_lo + frac * (z_hi - z_lo), grid.z_w[kc])

    # Neutral / stabilising forcing (B0 >= 0): h <= Ekman depth 0.7 u*/|f|,
    # and h <= Monin-Obukhov length L = u*³ / (kappa B0) when B0 > 0.
    # (L's denominator is replaced, not floored, where B0 <= 0: a floor of
    # 1e-30 gives an inf * 0 = NaN derivative in float32.)
    h_ek = _KPP_EKMAN * ustar / jnp.maximum(jnp.abs(grid.f_c), 1e-10)
    pos  = b0 > 0.0
    h_mo = jnp.where(pos, ustar ** 3 / (_VON_KARMAN * jnp.where(pos, b0, 1.0)), 1e10)
    h = jnp.where(b0 >= 0.0, jnp.minimum(h, jnp.minimum(h_ek, h_mo)), h)
    h = jnp.maximum(h, grid.z_c[0])        # at least the top half-cell
    h3 = h[..., None]

    # ---- K profile K = h w(sigma) G(sigma), G = sigma (1 - sigma)² --------
    z_w    = jnp.broadcast_to(grid.z_w, grid.mask_w.shape)
    sigma  = z_w / h3
    shape  = jnp.where(sigma < 1.0, sigma * (1.0 - sigma) ** 2, 0.0)
    d_face = jnp.where(b03 < 0.0, jnp.minimum(z_w, _KPP_EPS * h3), z_w)
    w_m, w_s = kpp_velocity_scales(d_face, us3, b03)
    kappa = jnp.maximum(kappa_int, h3 * w_s * shape) * grid.mask_w
    nu    = jnp.maximum(nu_int,    h3 * w_m * shape) * grid.mask_w

    # ---- nonlocal tracer transport (convective forcing only) --------------
    # Downward flux C_s G(sigma) F0 at each face: zero at the surface
    # (G(0) = 0) and at and below h (G = 0), so it only redistributes the
    # surface flux within the column and conserves heat and salt exactly.
    nl_shape = jnp.where(b03 < 0.0, _KPP_CS_NL * shape, 0.0) * grid.mask_w_adv

    def _nonlocal(f0):
        flux = nl_shape * f0[..., None]
        return (flux[..., :-1] - flux[..., 1:]) / grid.dz_c * mc

    return dict(kappa=kappa, nu=nu, nl_T=_nonlocal(f_t), nl_S=_nonlocal(f_s),
                hbl=h * mc[..., 0])


def kpp_coefficients(
    T:       jnp.ndarray,
    S:       jnp.ndarray,
    u:       jnp.ndarray,
    v:       jnp.ndarray,
    forcing,
    grid:    OceanGrid,
    params,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    K-profile parameterisation (Large, McWilliams & Doney 1994) for the
    surface boundary layer, on top of PP81 + convection in the interior.

    1. Surface forcing: friction velocity u* from the wind stress and the
       buoyancy flux B0 from the heat and freshwater fluxes
       (``kpp_surface_fluxes``); ``forcing`` is a ``SurfaceForcing`` or None.
    2. Boundary-layer depth h: the shallowest depth d at which the bulk
       Richardson number

         Ri_b(d) = d (B_r - B(d)) / (|V_r - V(d)|² + V_t²(d))

       reaches ``params.kpp_ri_crit`` (0.3), linearly interpolated between
       cell centres.  The reference B_r, V_r is the top cell; V_t² is the
       unresolved turbulent shear of LMD94 eq. (23).  Wind (through V_r - V
       and u* in V_t) and convection (through w_s in V_t) deepen h.  Under
       neutral or stabilising forcing (B0 >= 0) h is limited by the Ekman
       depth 0.7 u*/|f|, and for B0 > 0 by the Monin-Obukhov length.  If
       Ri_b stays below critical, h is the column depth.  h is at least
       the top half-cell.  Without forcing (u* at its 1e-5 floor, B0 = 0)
       h is the top half-cell and the scheme reduces exactly to PP81.
    3. Inside the boundary layer (sigma = depth / h < 1)

         kappa = max(PP81, h w_s(sigma) G(sigma)),   nu = max(PP81, h w_m G)

       with G = sigma (1 - sigma)² ("simple shapes", no matching to the
       interior; continuous at h since G(1) = 0).  Below h the PP81 values
       (with the convective blend) are used unchanged.
    4. Convective forcing (B0 < 0) adds the nonlocal tracer flux
       C_s G(sigma) F0 (C_s ~ 6.33), which carries part of the surface
       flux F0 into the boundary layer; it is returned as tendencies
       [tracer s⁻¹] with zero column integral.

    All heat is applied at the surface (no shortwave penetration yet), so
    B0 uses the total net heat flux.  Everything is column-local (z is
    never sharded), so the scheme runs unchanged under domain decomposition.

    Returns
    -------
    kappa      : (Nx, Ny, Nz+1) tracer diffusivity at tracer w-faces [m² s⁻¹]
    nu_u, nu_v : (Nx, Ny, Nz+1) viscosity at u- / v-column w-faces
    nonlocal_T : (Nx, Ny, Nz)   nonlocal T tendency [K s⁻¹]
    nonlocal_S : (Nx, Ny, Nz)   nonlocal S tendency [psu s⁻¹]
    """
    out = _kpp_core(T, S, u, v, forcing, grid, params)
    nu_u, nu_v = _tracer_to_velocity_faces(out["nu"], grid)
    return out["kappa"], nu_u, nu_v, out["nl_T"], out["nl_S"]


def kpp_boundary_layer_depth(
    T:       jnp.ndarray,
    S:       jnp.ndarray,
    u:       jnp.ndarray,
    v:       jnp.ndarray,
    forcing,
    grid:    OceanGrid,
    params,
) -> jnp.ndarray:
    """Diagnostic KPP boundary-layer depth h (Nx, Ny) [m]; 0 on land."""
    return _kpp_core(T, S, u, v, forcing, grid, params)["hbl"]
