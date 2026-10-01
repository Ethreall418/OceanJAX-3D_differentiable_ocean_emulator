"""
OceanJAX Time Stepper
=====================
Coordinates one model time step (and multi-step runs) by orchestrating the
Physics modules in the correct order.

Splitting strategy
------------------
Momentum  — Leapfrog with Asselin-Robert filter (suppresses the computational
             mode that leapfrog generates).  Vertical viscosity is applied
             implicitly after the leapfrog advance.

Tracers   — Adams-Bashforth 3rd order (AB3) for the explicit part; vertical
             diffusion is applied implicitly afterwards.  The AB order is
             selected explicitly by ``state.step_count``:
               step_count == 0  →  AB1  (Forward Euler; coefficients 1, 0, 0)
               step_count == 1  →  AB2  (coefficients 3/2, -1/2, 0)
               step_count >= 2  →  AB3  (coefficients 23/12, -16/12, 5/12)
             This replaces the older "zero-history" bootstrap (which produced
             the same first step but could silently mis-apply AB2 coefficients
             if ``T_tend_prev`` was ever non-zero at step 0).

Free surface — Leapfrog update (eta_prev + 2*dt * deta_dt), consistent with
               the momentum leapfrog, with an Asselin-Robert filter on eta.
               Bootstrap at step_count == 0: uses 1*dt (Forward Euler start),
               identical to the momentum bootstrap.
               w is diagnosed upward from the seafloor, so its surface face
               carries the kinematic signal w[0] = -deta/dt (w positive
               downward) and tracers are advected with a non-divergent
               (u, v, w).

Division of labour with Physics modules
----------------------------------------
  dynamics.py  : equation_of_state, hydrostatic_pressure computed ONCE per
                 step and passed to both momentum tendency functions.
  mixing.py    : implicit_vertical_visc / implicit_vertical_mix applied AFTER
                 explicit advances; not called inside tendency functions.
  tracers.py   : tracer_tendency (advection + horizontal diffusion) and the
                 surface forcing tendencies.

Surface forcing
---------------
All surface fluxes are packaged in ``SurfaceForcing``.  Wind stress enters
the momentum equations as a surface-layer tendency analogous to the tracer
heat / freshwater fluxes.  Passing ``forcing=None`` disables all surface
inputs (useful for spin-up or idealised experiments).

ML closure hook
---------------
An optional ``closure`` argument (``OceanJAX.ml.AbstractClosure``) can be
passed to ``step()`` and ``run()``.  When supplied, the closure is called
once per step after the explicit tracer tendencies are computed:

  G_T  +=  corr.dT_tend             (extra T tendency [K s-1])
  G_S  +=  corr.dS_tend             (extra S tendency [psu s-1])
  kappa_v_eff = kappa_v * corr.kappa_v_scale   (vertical diffusivity scaling)

Passing ``closure=None`` (the default) skips the hook entirely; the
numerical path is bit-identical to the pre-ML pure-physics version.

Contents
--------
  SurfaceForcing    – equinox Module holding surface flux fields
  step              – advance OceanState by one dt
  run               – lax.scan wrapper for multi-step integrations
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx

from OceanJAX.grid import OceanGrid
from OceanJAX.state import OceanState, ModelParams
from OceanJAX.ml.closure import AbstractClosure
from OceanJAX.Physics.dynamics import (
    equation_of_state,
    hydrostatic_pressure,
    momentum_tendency_u,
    momentum_tendency_v,
    free_surface_tendency,
    compute_w,
)
from OceanJAX.Physics.tracers import (
    tracer_tendency,
    heat_surface_tendency,
    salt_surface_tendency,
)
from OceanJAX.Physics.mixing import (
    implicit_vertical_visc,
    implicit_vertical_mix,
    bottom_drag_velocity,
    pp81_coefficients,
)


# ---------------------------------------------------------------------------
# Surface forcing container
# ---------------------------------------------------------------------------

class SurfaceForcing(eqx.Module):
    """
    Surface boundary conditions for one time step.

    All fields are (Nx, Ny) arrays in SI units.  Pass ``forcing=None``
    to ``step`` or ``run`` to disable all surface inputs.

    Fields
    ------
    heat_flux : (Nx, Ny) [W m-2]   Net downward heat flux into the ocean.
                                    Positive = ocean gains heat.
    fw_flux   : (Nx, Ny) [m s-1]   Net evaporation minus precipitation (E-P).
                                    Positive = net freshwater loss → salinity increase.
    tau_x     : (Nx, Ny) [N m-2]   Zonal wind stress at the sea surface.
                                    Positive eastward.
    tau_y     : (Nx, Ny) [N m-2]   Meridional wind stress at the sea surface.
                                    Positive northward.
    """
    heat_flux: jnp.ndarray
    fw_flux:   jnp.ndarray
    tau_x:     jnp.ndarray
    tau_y:     jnp.ndarray


# ---------------------------------------------------------------------------
# Internal: wind stress surface tendencies
# ---------------------------------------------------------------------------

def _wind_tendency_u(
    tau_x:  jnp.ndarray,
    grid:   OceanGrid,
    params: ModelParams,
) -> jnp.ndarray:
    """
    Zonal momentum tendency from surface wind stress [m s-2].

    Applies the stress to the top layer (k=0) only:

      du/dt|_wind = tau_x / (rho0 * dz_c[0])   at k = 0
                 = 0                             at k > 0

    The sign convention matches the momentum equation:
    positive tau_x (eastward stress) accelerates u.

    Args:
        tau_x  : (Nx, Ny) [N m-2]
        grid   : OceanGrid
        params : ModelParams  (uses rho0)

    Returns:
        (Nx, Ny, Nz) [m s-2]
    """
    tend_surf = tau_x / (params.rho0 * grid.dz_c[0]) * grid.mask_u[:, :, 0]
    return jnp.zeros(
        (grid.Nx, grid.Ny, grid.Nz), dtype=tau_x.dtype
    ).at[:, :, 0].set(tend_surf)


def _wind_tendency_v(
    tau_y:  jnp.ndarray,
    grid:   OceanGrid,
    params: ModelParams,
) -> jnp.ndarray:
    """
    Meridional momentum tendency from surface wind stress [m s-2].

    Mirrors ``_wind_tendency_u`` for the v-equation.

    Args:
        tau_y  : (Nx, Ny) [N m-2]
        grid   : OceanGrid
        params : ModelParams  (uses rho0)

    Returns:
        (Nx, Ny, Nz) [m s-2]
    """
    tend_surf = tau_y / (params.rho0 * grid.dz_c[0]) * grid.mask_v[:, :, 0]
    return jnp.zeros(
        (grid.Nx, grid.Ny, grid.Nz), dtype=tau_y.dtype
    ).at[:, :, 0].set(tend_surf)


# ---------------------------------------------------------------------------
# Single time step
# ---------------------------------------------------------------------------

def step(
    state:   OceanState,
    grid:    OceanGrid,
    params:  ModelParams,
    forcing: Optional[SurfaceForcing] = None,
    closure: Optional[AbstractClosure] = None,
) -> OceanState:
    """
    Advance the model state by one time step dt.

    Calling sequence
    ----------------
    1.  Shared diagnostics (EOS + hydrostatic pressure, computed once).
    2.  Explicit momentum tendencies (Coriolis + PGF + horiz. viscosity).
        + Wind stress surface forcing (if supplied).
    3.  Leapfrog momentum advance:  u_new = u_prev + 2*dt * G_u
    4.  Asselin-Robert filter on u(n):
          u_filt = u + alpha*(u_new - 2*u + u_prev)
    5.  Vertical mixing coefficients (constant or PP81); implicit vertical
        viscosity + quadratic bottom drag on u_new, v_new.
    6.  Diagnose w from (u_filt, v_filt), bottom-up; w[0] = -deta/dt.
    7.  Explicit tracer tendencies (advection with u_filt, v_filt, w
        + horiz. diffusion) + heat / freshwater surface forcing.
        [ML hook] closure corrections to G_T, G_S and kappa_v (if supplied).
    8.  Adams-Bashforth 3 tracer advance.
    9.  Implicit vertical diffusion applied to T_new, S_new; freezing-point
        limit T >= -freezing_slope * S (if params.limit_freezing).
    10. Free-surface leapfrog: eta_new = eta_prev + 2*dt * deta_dt,
        plus Asselin filter on eta(n).
    11. Apply all masks and assemble new OceanState.

    Args:
        state   : OceanState at time n
        grid    : OceanGrid (static)
        params  : ModelParams
        forcing : SurfaceForcing for this time step, or None
        closure : AbstractClosure instance, or None.
                  When supplied, called after the explicit tracer tendencies
                  to add learnable corrections (dT_tend, dS_tend) and to
                  scale the vertical diffusivity (kappa_v_scale).
                  Passing None (default) skips the hook entirely; the
                  numerical path is bit-identical to the pure-physics version.

    Returns:
        OceanState at time n+1
    """
    dt = params.dt

    # ------------------------------------------------------------------
    # 1. Shared diagnostics
    # ------------------------------------------------------------------
    rho         = equation_of_state(state.T, state.S, params)
    p_hyd_prime = hydrostatic_pressure(rho, grid, params)

    # ------------------------------------------------------------------
    # 2. Explicit momentum tendencies
    # ------------------------------------------------------------------
    G_u = momentum_tendency_u(state, p_hyd_prime, grid, params)
    G_v = momentum_tendency_v(state, p_hyd_prime, grid, params)

    if forcing is not None:
        G_u = G_u + _wind_tendency_u(forcing.tau_x, grid, params)
        G_v = G_v + _wind_tendency_v(forcing.tau_y, grid, params)

    # ------------------------------------------------------------------
    # 3. Leapfrog momentum advance
    #    Bootstrap (step_count == 0): use 1*dt (Forward Euler) so that
    #    u_prev = u gives u_new = u + dt*G — the correct first step.
    #    All subsequent steps use the standard 2*dt leapfrog.
    # ------------------------------------------------------------------
    leapfrog_dt = jnp.where(state.step_count == 0, dt, 2.0 * dt)
    u_new = (state.u_prev + leapfrog_dt * G_u) * grid.mask_u
    v_new = (state.v_prev + leapfrog_dt * G_v) * grid.mask_v

    # ------------------------------------------------------------------
    # 4. Asselin-Robert filter on current-time-level u(n), v(n)
    #    This damps the spurious computational mode of the leapfrog
    #    scheme; the filtered values become u_prev for the next step.
    # ------------------------------------------------------------------
    alpha  = params.asselin_coeff
    u_filt = (state.u + alpha * (u_new - 2.0 * state.u + state.u_prev)) * grid.mask_u
    v_filt = (state.v + alpha * (v_new - 2.0 * state.v + state.v_prev)) * grid.mask_v

    # ------------------------------------------------------------------
    # 5. Implicit vertical viscosity + quadratic bottom drag
    #    Vertical mixing coefficients are evaluated at time level n:
    #      "constant": params.nu_v / params.kappa_v everywhere;
    #      "pp81"    : Ri-dependent Pacanowski-Philander + convection
    #                  (kappa_vmix is reused for the tracers in step 9).
    #    Drag velocity Cd*|u_b| is also evaluated at time n (semi-implicit)
    #    and applied implicitly to the deepest wet cell of each column, so
    #    it is unconditionally stable.  bottom_drag_cd = 0 disables it.
    # ------------------------------------------------------------------
    if params.vertical_mixing == "pp81":
        kappa_vmix, nu_u, nu_v = pp81_coefficients(
            state.T, state.S, state.u, state.v, grid, params)
    else:
        kappa_vmix, nu_u, nu_v = params.kappa_v, params.nu_v, params.nu_v

    drag_u, drag_v = bottom_drag_velocity(state.u, state.v, grid, params)
    u_new = implicit_vertical_visc(u_new, nu_u, dt, grid, grid.mask_u, drag_u)
    v_new = implicit_vertical_visc(v_new, nu_v, dt, grid, grid.mask_v, drag_v)

    # ------------------------------------------------------------------
    # 6. Diagnose w from the Asselin-filtered velocities
    #    compute_w integrates upward from w = 0 at the seafloor, so its
    #    surface face is the kinematic value w[0] = -deta/dt of the same
    #    u_filt/v_filt that drive eta below.  Every cell of (u_filt,
    #    v_filt, w_new) is exactly non-divergent.
    # ------------------------------------------------------------------
    w_new = compute_w(u_filt, v_filt, grid)

    # ------------------------------------------------------------------
    # 7. Explicit tracer tendencies
    #    Advect with (u_filt, v_filt, w_new) — one continuity-consistent
    #    velocity state, so a uniform tracer stays exactly uniform.
    # ------------------------------------------------------------------
    G_T = tracer_tendency(state.T, u_filt, v_filt, w_new, params.kappa_h, grid)
    G_S = tracer_tendency(state.S, u_filt, v_filt, w_new, params.kappa_h, grid)

    if forcing is not None:
        G_T = G_T + heat_surface_tendency(forcing.heat_flux, grid, params)
        # Virtual salt flux with the local SSS (dilution proportional to S)
        G_S = G_S + salt_surface_tendency(forcing.fw_flux,   grid, params,
                                          sss=state.S[:, :, 0])

    # ------------------------------------------------------------------
    # [ML hook] Closure corrections to tracer tendencies and kappa_v.
    #   Activated only when closure is not None; closure=None (default)
    #   leaves G_T, G_S, and kappa_v completely unchanged.
    # ------------------------------------------------------------------
    if closure is not None:
        corr    = closure(state, grid, params)
        G_T     = (G_T + corr.dT_tend) * grid.mask_c
        G_S     = (G_S + corr.dS_tend) * grid.mask_c
        kappa_v = kappa_vmix * corr.kappa_v_scale
    else:
        kappa_v = kappa_vmix

    # ------------------------------------------------------------------
    # 8. Adams-Bashforth 3 tracer advance
    #    Select coefficients based on step_count for proper bootstrap:
    #      step_count == 0  →  AB1: (1,    0,    0  )
    #      step_count == 1  →  AB2: (3/2, -1/2,  0  )
    #      step_count >= 2  →  AB3: params.ab3_coeffs
    # ------------------------------------------------------------------
    ab3_0, ab3_1, ab3_2 = params.ab3_coeffs
    sc = state.step_count
    a0 = jnp.where(sc == 0, 1.0,       jnp.where(sc == 1,  3.0/2.0, ab3_0))
    a1 = jnp.where(sc == 0, 0.0,       jnp.where(sc == 1, -1.0/2.0, ab3_1))
    a2 = jnp.where(sc == 0, 0.0,       jnp.where(sc == 1,  0.0,     ab3_2))
    T_new = (state.T + dt * (a0 * G_T
                             + a1 * state.T_tend_prev
                             + a2 * state.T_tend_prev2)) * grid.mask_c
    S_new = (state.S + dt * (a0 * G_S
                             + a1 * state.S_tend_prev
                             + a2 * state.S_tend_prev2)) * grid.mask_c

    # ------------------------------------------------------------------
    # 9. Implicit vertical diffusion
    #    kappa_v is the scheme diffusivity (params.kappa_v, or the PP81
    #    field), times corr.kappa_v_scale when a closure is active.
    # ------------------------------------------------------------------
    T_new = implicit_vertical_mix(T_new, kappa_v, dt, grid,
                                  rhs_explicit=jnp.zeros_like(T_new))
    S_new = implicit_vertical_mix(S_new, kappa_v, dt, grid,
                                  rhs_explicit=jnp.zeros_like(S_new))

    # Freezing-point limit (stand-in for sea ice): T >= -freezing_slope * S.
    # Heat removed below T_f is discarded, as if it went into ice formation.
    if params.limit_freezing:
        T_new = jnp.maximum(T_new, -params.freezing_slope * S_new) * grid.mask_c

    # ------------------------------------------------------------------
    # 10. Free-surface update (leapfrog, consistent with momentum)
    #    Bootstrap (step_count == 0): 1*dt Forward Euler start so that
    #    eta_prev = eta gives eta_new = eta + dt*deta_dt.
    #    Subsequent steps: eta_new = eta_prev + 2*dt * deta_dt.
    #    Asselin-Robert filter applied to current eta (same alpha as u/v).
    # ------------------------------------------------------------------
    deta_dt   = free_surface_tendency(u_filt, v_filt, grid)
    surf_mask = grid.mask_c[:, :, 0]
    eta_new   = (state.eta_prev + leapfrog_dt * deta_dt) * surf_mask
    eta_filt  = (state.eta + alpha * (eta_new - 2.0 * state.eta + state.eta_prev)) * surf_mask

    # ------------------------------------------------------------------
    # 11. Assemble new state
    # ------------------------------------------------------------------
    return OceanState(
        u   = u_new,
        v   = v_new,
        w   = w_new,
        T   = T_new,
        S   = S_new,
        eta      = eta_new,
        # Asselin-filtered fields become the "previous" level for next step
        u_prev   = u_filt,
        v_prev   = v_filt,
        eta_prev = eta_filt,
        # Rotate tendency history for AB3
        T_tend_prev  = G_T,
        S_tend_prev  = G_S,
        T_tend_prev2 = state.T_tend_prev,
        S_tend_prev2 = state.S_tend_prev,
        time       = state.time + dt,
        step_count = state.step_count + 1,
    )


# ---------------------------------------------------------------------------
# Multi-step run via lax.scan
# ---------------------------------------------------------------------------

def run(
    state:            OceanState,
    grid:             OceanGrid,
    params:           ModelParams,
    n_steps:          int,
    forcing_sequence: Optional[SurfaceForcing] = None,
    save_history:     bool = False,
    closure:          Optional[AbstractClosure] = None,
) -> tuple[OceanState, Optional[OceanState]]:
    """
    Integrate the model for ``n_steps`` time steps using ``jax.lax.scan``.

    The scan loop is JIT-compilable and supports reverse-mode AD through
    the entire trajectory, enabling gradient-based optimisation over model
    parameters or initial conditions.

    Args:
        state            : Initial OceanState
        grid             : OceanGrid (static)
        params           : ModelParams
        n_steps          : Number of time steps to advance
        forcing_sequence : SurfaceForcing whose array fields have a leading
                           time axis of length n_steps, e.g.
                           ``heat_flux.shape == (n_steps, Nx, Ny)``.
                           Pass ``None`` for no surface forcing.
        save_history     : If True, return all intermediate OceanState
                           snapshots (memory-intensive for long runs).
                           If False, only the final state is returned and
                           history is None.
        closure          : AbstractClosure instance, or None.
                           Passed through to every call to ``step()``.
                           See ``step()`` for details.

    Returns:
        (final_state, history)
        final_state : OceanState after n_steps
        history     : OceanState with a leading time axis of length n_steps
                      if save_history=True, else None.
    """
    def _step_fn(carry: OceanState, forcing_t: Optional[SurfaceForcing]):
        new_state = step(carry, grid, params, forcing_t, closure)
        output    = new_state if save_history else None
        return new_state, output

    if forcing_sequence is None:
        # No time-varying forcing: scan over a dummy axis
        final_state, history = jax.lax.scan(
            lambda carry, _: _step_fn(carry, None),
            state,
            None,
            length=n_steps,
        )
    else:
        # Validate that the forcing leading axis matches n_steps so that a
        # shape mismatch is caught here rather than inside jax.lax.scan
        # (where the error message is harder to interpret).
        seq_len = jax.tree_util.tree_leaves(forcing_sequence)[0].shape[0]
        if seq_len != n_steps:
            raise ValueError(
                f"forcing_sequence leading axis length ({seq_len}) does not "
                f"match n_steps ({n_steps})."
            )
        final_state, history = jax.lax.scan(
            _step_fn,
            state,
            forcing_sequence,
        )

    return final_state, history
