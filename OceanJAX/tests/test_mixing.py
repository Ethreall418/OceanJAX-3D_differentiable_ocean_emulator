"""
Tests for OceanJAX.Physics.mixing (implicit solvers, bottom drag) and the
freezing-point limit in the time stepper
=========================================================================
Four groups of properties are verified:

  1. Increment-form implicit solve
       A uniform column is preserved *exactly* over many solves (the
       full-field Thomas solve added one float32 ulp to the bottom level on
       every call), and the solution matches a float64 dense solve.

  2. Bottom drag
       Cd = 0 is bit-identical to no drag; a single-layer column decays at
       the implicit rate u / (1 + dt*r); drag acts only on the deepest wet
       cell of each column.

  3. Freezing-point limit
       T is clamped to -freezing_slope * S when limit_freezing is on, and
       left alone when it is off.

  4. Munk-criterion viscosity
       munk_viscosity matches beta*dx^3 at the row nearest the equator,
       scales as dx^3 with resolution, and ignores land columns.

  5. Pacanowski-Philander (1981) vertical mixing
       N² > 0 for stable stratification (z positive down); background
       values without shear; exact coefficients at Ri = 1; convective
       values where N² < 0; no shear dilution at walls; nu0 = 0 reduces
       to constant mixing; convection overturns an unstable column.

Running
-------
    pytest OceanJAX/tests/test_mixing.py -v
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, create_rest_state
from OceanJAX.timeStepping import step, run
from OceanJAX.Physics.mixing import (
    implicit_vertical_mix,
    implicit_vertical_visc,
    bottom_cell_mask,
    bottom_drag_velocity,
)


def _stepped_grid() -> OceanGrid:
    z_levels = np.array([5.0, 20.0, 50.0, 100.0, 180.0])
    H = np.full((4, 3), 250.0)
    H[1, 1] = 30.0      # two wet levels
    H[2, 0] = 8.0       # one wet level
    H[3, 2] = 0.0       # land
    return OceanGrid.create((0.0, 20.0), (10.0, 25.0), z_levels, 4, 3, bathymetry=H)


# ---------------------------------------------------------------------------
# 1. Increment-form implicit solve
# ---------------------------------------------------------------------------

class TestImplicitSolve:

    def test_uniform_column_preserved_exactly(self):
        grid = _stepped_grid()
        S = jnp.full((grid.Nx, grid.Ny, grid.Nz), 35.0) * grid.mask_c
        for _ in range(200):
            S = implicit_vertical_mix(S, 1e-2, 300.0, grid)
        wet = np.asarray(grid.mask_c) > 0
        assert np.all(np.asarray(S)[wet] == 35.0), (
            f"max |S - 35| = {np.abs(np.asarray(S)[wet] - 35.0).max():.3e}"
        )

    def test_matches_dense_float64_solve(self):
        grid  = OceanGrid.create((0.0, 10.0), (0.0, 10.0),
                                 np.array([5.0, 20.0, 50.0, 100.0, 180.0]), 2, 2)
        rng   = np.random.default_rng(0)
        phi   = jnp.asarray(10.0 + rng.normal(size=(2, 2, grid.Nz)), dtype=jnp.float32)
        kappa, dt = 5e-2, 900.0
        out   = np.asarray(implicit_vertical_mix(phi, kappa, dt, grid), np.float64)

        # Dense reference: -d/dz(kappa dphi/dz), closed surface and bottom
        dz_c, dz_w = np.asarray(grid.dz_c, np.float64), np.asarray(grid.dz_w, np.float64)
        Nz = grid.Nz
        A  = np.eye(Nz)
        for k in range(Nz):
            if k > 0:
                f = kappa / dz_w[k]
                A[k, k] += dt * f / dz_c[k];  A[k, k - 1] -= dt * f / dz_c[k]
            if k < Nz - 1:
                f = kappa / dz_w[k + 1]
                A[k, k] += dt * f / dz_c[k];  A[k, k + 1] -= dt * f / dz_c[k]
        ref = np.linalg.solve(A, np.asarray(phi, np.float64).reshape(-1, Nz).T).T
        np.testing.assert_allclose(out.reshape(-1, Nz), ref, atol=1e-5)

    def test_column_content_conserved(self):
        """Implicit diffusion conserves sum(phi * dz) in every column."""
        grid = _stepped_grid()
        rng  = np.random.default_rng(1)
        phi  = jnp.asarray(20.0 + rng.normal(size=(grid.Nx, grid.Ny, grid.Nz)),
                           dtype=jnp.float32) * grid.mask_c
        out  = implicit_vertical_mix(phi, 1e-1, 900.0, grid)
        dz   = np.asarray(grid.dz_c, np.float64)
        before = (np.asarray(phi, np.float64) * dz).sum(-1)
        after  = (np.asarray(out, np.float64) * dz).sum(-1)
        np.testing.assert_allclose(after, before, rtol=1e-6)


# ---------------------------------------------------------------------------
# 2. Bottom drag
# ---------------------------------------------------------------------------

class TestBottomDrag:

    def test_bottom_cell_mask(self):
        grid = _stepped_grid()
        bot  = np.asarray(bottom_cell_mask(grid.mask_c))
        n    = np.asarray(grid.mask_c).sum(-1).astype(int)
        for i in range(grid.Nx):
            for j in range(grid.Ny):
                expected = np.zeros(grid.Nz)
                if n[i, j] > 0:
                    expected[n[i, j] - 1] = 1.0
                np.testing.assert_array_equal(bot[i, j], expected)

    def test_zero_cd_is_identical_to_no_drag(self):
        grid = _stepped_grid()
        rng  = np.random.default_rng(2)
        u    = jnp.asarray(rng.normal(size=(grid.Nx, grid.Ny, grid.Nz)),
                           dtype=jnp.float32) * grid.mask_u
        params  = ModelParams(bottom_drag_cd=0.0)
        drag, _ = bottom_drag_velocity(u, u, grid, params)
        a = implicit_vertical_visc(u, 1e-3, 300.0, grid, grid.mask_u)
        b = implicit_vertical_visc(u, 1e-3, 300.0, grid, grid.mask_u, drag)
        assert np.array_equal(np.asarray(a), np.asarray(b))

    def test_single_layer_decay_rate(self):
        """One wet level, uniform u, no v: u_new = u / (1 + dt*Cd*|u|_eff/dz)."""
        z_levels = np.array([25.0, 75.0])
        H    = np.full((4, 3), 40.0)                       # top level only
        grid = OceanGrid.create((0.0, 20.0), (10.0, 25.0), z_levels, 4, 3,
                                bathymetry=H)
        params = ModelParams(bottom_drag_cd=2e-3, bottom_drag_ubg=0.05)
        u0 = 0.5
        u  = jnp.full((grid.Nx, grid.Ny, grid.Nz), u0) * grid.mask_u
        v  = jnp.zeros_like(u)
        drag_u, _ = bottom_drag_velocity(u, v, grid, params)
        dt  = 300.0
        out = np.asarray(implicit_vertical_visc(u, 1e-4, dt, grid, grid.mask_u, drag_u))

        r        = 2e-3 * np.sqrt(u0 ** 2 + 0.05 ** 2) / float(grid.dz_c[0])
        expected = u0 / (1.0 + dt * r)
        wet      = np.asarray(grid.mask_u)[:, :, 0] > 0
        np.testing.assert_allclose(out[:, :, 0][wet], expected, rtol=1e-5)

    def test_drag_only_touches_bottom_cell(self):
        """With nu_v = 0 the layers are uncoupled: only the bottom cell changes."""
        grid   = _stepped_grid()
        params = ModelParams(bottom_drag_cd=1e-3)
        u      = jnp.full((grid.Nx, grid.Ny, grid.Nz), 0.3) * grid.mask_u
        drag_u, _ = bottom_drag_velocity(u, jnp.zeros_like(u), grid, params)
        out    = np.asarray(implicit_vertical_visc(u, 0.0, 300.0, grid, grid.mask_u, drag_u))
        bot    = np.asarray(bottom_cell_mask(grid.mask_u)) > 0
        wet    = np.asarray(grid.mask_u) > 0
        assert np.all(out[bot] < 0.3)
        np.testing.assert_array_equal(out[wet & ~bot], np.float32(0.3))


# ---------------------------------------------------------------------------
# 3. Freezing-point limit
# ---------------------------------------------------------------------------

class TestFreezingLimit:
    # Constant mixing with kappa_v = 0 isolates the limiter: under PP81 the
    # cold-over-warm column would convect.

    @staticmethod
    def _cold_state(grid):
        s = create_rest_state(grid, T_background=-5.0, S_background=35.0)
        return eqx.tree_at(lambda st: st.T, s, s.T.at[:, :, 1:].set(4.0) * grid.mask_c)

    def test_clamped_to_freezing_point(self):
        grid   = OceanGrid.create((0.0, 20.0), (50.0, 60.0), np.array([10.0, 30.0]), 3, 3)
        params = ModelParams(dt=300.0, kappa_v=0.0, kappa_h=0.0,
                             vertical_mixing="constant")
        new    = step(self._cold_state(grid), grid, params)
        T, S   = np.asarray(new.T), np.asarray(new.S)
        np.testing.assert_allclose(T[:, :, 0], -0.0575 * S[:, :, 0], rtol=1e-6)
        np.testing.assert_array_equal(T[:, :, 1], 4.0)

    def test_disabled(self):
        grid   = OceanGrid.create((0.0, 20.0), (50.0, 60.0), np.array([10.0, 30.0]), 3, 3)
        params = ModelParams(dt=300.0, kappa_v=0.0, kappa_h=0.0, limit_freezing=False,
                             vertical_mixing="constant")
        new    = step(self._cold_state(grid), grid, params)
        np.testing.assert_array_equal(np.asarray(new.T)[:, :, 0], -5.0)


# ---------------------------------------------------------------------------
# 4. Munk-criterion viscosity
# ---------------------------------------------------------------------------

class TestMunkViscosity:

    def test_analytic_value(self):
        """nu = beta(phi) * dx(phi)^3 at the row nearest the equator."""
        from OceanJAX.Physics.mixing import munk_viscosity
        from OceanJAX.grid import EARTH_RADIUS, OMEGA, DEG2RAD
        grid = OceanGrid.create((0.0, 20.0), (-10.0, 10.0), np.array([10.0, 30.0]), 10, 10)
        lat  = np.asarray(grid.lat_c, np.float64)
        j    = int(np.argmin(np.abs(lat)))
        beta = 2 * OMEGA * np.cos(lat[j] * DEG2RAD) / EARTH_RADIUS
        dx   = EARTH_RADIUS * np.cos(lat[j] * DEG2RAD) * 2.0 * DEG2RAD
        assert munk_viscosity(grid) == pytest.approx(beta * dx ** 3, rel=1e-5)
        assert munk_viscosity(grid, n_points=2) == pytest.approx(8 * beta * dx ** 3, rel=1e-5)

    def test_scales_with_dx_cubed(self):
        from OceanJAX.Physics.mixing import munk_viscosity
        z = np.array([10.0, 30.0])
        coarse = OceanGrid.create((0.0, 20.0), (20.0, 40.0), z, 10, 10)
        fine   = OceanGrid.create((0.0, 20.0), (20.0, 40.0), z, 20, 20)
        assert munk_viscosity(coarse) / munk_viscosity(fine) == pytest.approx(8.0, rel=0.05)

    def test_ignores_land_columns(self):
        """Land rows nearest the equator must not set the value."""
        from OceanJAX.Physics.mixing import munk_viscosity
        z = np.array([10.0, 30.0])
        H = np.full((10, 10), 100.0)
        H[:, :5] = 0.0                       # southern half (near equator) is land
        land = OceanGrid.create((0.0, 20.0), (0.0, 40.0), z, 10, 10, bathymetry=H)
        sea  = OceanGrid.create((0.0, 20.0), (0.0, 40.0), z, 10, 10)
        assert munk_viscosity(land) < munk_viscosity(sea)


# ---------------------------------------------------------------------------
# 5. Pacanowski-Philander (1981) vertical mixing
# ---------------------------------------------------------------------------

def _pp81_grid(periodic_x=True):
    z = np.array([10.0, 30.0, 50.0, 70.0, 90.0])          # uniform dz_w = 20 m
    return OceanGrid.create((0.0, 10.0), (10.0, 20.0), z, 4, 4, periodic_x=periodic_x)


def _column_fields(grid, dTdz, dudz):
    """T linear in depth (S const), u linear in depth, v = 0."""
    z  = np.asarray(grid.z_c, np.float64)
    sh = (grid.Nx, grid.Ny, grid.Nz)
    T  = jnp.asarray(np.broadcast_to(20.0 + dTdz * z, sh), jnp.float32)
    S  = jnp.full(sh, 35.0)
    u  = jnp.asarray(np.broadcast_to(dudz * z, sh), jnp.float32) * grid.mask_u
    return T, S, u, jnp.zeros(sh)


class TestPP81:

    PARAMS = ModelParams(vertical_mixing="pp81")

    def test_n2_positive_for_stable_stratification(self):
        """z is positive downward: T decreasing with depth is stable."""
        from OceanJAX.Physics.mixing import buoyancy_and_shear
        grid = _pp81_grid()
        T, S, u, v = _column_fields(grid, dTdz=-0.05, dudz=0.0)
        n2, _ = buoyancy_and_shear(T, S, u, v, grid, self.PARAMS)
        expected = self.PARAMS.g * self.PARAMS.alpha_T * 0.05       # -g α dT/dz
        # N² from T/S differences: float32 round-off only (no ~1025 cancellation)
        np.testing.assert_allclose(np.asarray(n2)[:, :, 1:-1], expected, rtol=1e-5)

    def test_background_without_shear(self):
        from OceanJAX.Physics.mixing import pp81_coefficients
        grid = _pp81_grid()
        T, S, u, v = _column_fields(grid, dTdz=-0.05, dudz=0.0)
        kappa, nu_u, _ = pp81_coefficients(T, S, u, v, grid, self.PARAMS)
        inner = (slice(None), slice(None), slice(1, -1))
        np.testing.assert_allclose(np.asarray(kappa)[inner], self.PARAMS.kappa_v, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(nu_u)[inner], self.PARAMS.nu_v, rtol=1e-6)

    def test_known_richardson_number(self):
        """Uniform N² and S² with Ri = 1: nu = nu0/36 + nu_b, kappa = nu0/216 + kappa_b."""
        from OceanJAX.Physics.mixing import pp81_coefficients, richardson_number
        p    = self.PARAMS
        grid = _pp81_grid()
        dTdz = -0.05
        n2   = p.g * p.alpha_T * 0.05
        T, S, u, v = _column_fields(grid, dTdz=dTdz, dudz=float(np.sqrt(n2)))
        ri = np.asarray(richardson_number(T, S, u, v, grid, p))[:, :, 1:-1]
        np.testing.assert_allclose(ri, 1.0, rtol=1e-3)
        kappa, nu_u, _ = pp81_coefficients(T, S, u, v, grid, p)
        np.testing.assert_allclose(np.asarray(nu_u)[:, :, 1:-1],
                                   p.pp81_nu0 / 36.0 + p.nu_v, rtol=2e-3)
        np.testing.assert_allclose(np.asarray(kappa)[:, :, 1:-1],
                                   p.pp81_nu0 / 216.0 + p.kappa_v, rtol=2e-3)

    def test_convective_when_unstable(self):
        from OceanJAX.Physics.mixing import pp81_coefficients
        grid = _pp81_grid()
        T, S, u, v = _column_fields(grid, dTdz=+0.05, dudz=0.0)     # warm below
        kappa, nu_u, nu_v = pp81_coefficients(T, S, u, v, grid, self.PARAMS)
        for a in (kappa, nu_u):
            np.testing.assert_allclose(np.asarray(a)[:, :, 1:-1], self.PARAMS.vmix_convective)

    def test_shear_not_diluted_at_walls(self):
        """Tracer columns next to a closed wall average only wet u-faces."""
        from OceanJAX.Physics.mixing import buoyancy_and_shear
        grid = _pp81_grid(periodic_x=False)
        T, S, u, v = _column_fields(grid, dTdz=-0.05, dudz=0.01)
        _, s2 = buoyancy_and_shear(T, S, u, v, grid, self.PARAMS)
        s2 = np.asarray(s2)[:, :, 1:-1]
        np.testing.assert_allclose(s2, 1e-4, rtol=1e-4)               # incl. i = 0, Nx-1

    def test_zero_nu0_matches_constant_scheme(self):
        """With nu0 = 0 on a stable column PP81 reduces to constant mixing."""
        grid = _pp81_grid()
        T, S, u, v = _column_fields(grid, dTdz=-0.05, dudz=0.02)
        from OceanJAX.state import create_from_arrays
        st = create_from_arrays(grid, u=u, v=v, T=T, S=S, eta=jnp.zeros((4, 4)))
        a = step(st, grid, ModelParams(dt=300.0, vertical_mixing="constant"))
        b = step(st, grid, ModelParams(dt=300.0, vertical_mixing="pp81", pp81_nu0=0.0))
        for f in ("u", "v", "T", "S", "eta"):
            np.testing.assert_allclose(np.asarray(getattr(a, f)),
                                       np.asarray(getattr(b, f)), rtol=1e-6, atol=1e-9)

    def test_convection_removes_static_instability(self):
        """Warm-below column: PP81 overturns it within a day; constant does not."""
        grid = _pp81_grid()
        T, S, u, v = _column_fields(grid, dTdz=+0.05, dudz=0.0)
        from OceanJAX.state import create_from_arrays
        st0 = create_from_arrays(grid, u=u, v=v, T=T, S=S, eta=jnp.zeros((4, 4)))

        def run_day(params):
            final, _ = run(st0, grid, params, 288)
            return np.asarray(final.T)[0, 0, :]

        mixed = run_day(ModelParams(dt=300.0, vertical_mixing="pp81"))
        const = run_day(ModelParams(dt=300.0, vertical_mixing="constant"))
        assert np.ptp(mixed) < 0.01 * np.ptp(np.asarray(T)[0, 0, :])
        assert np.all(np.diff(const) > 0), "constant mixing keeps the unstable profile"

    def test_invalid_scheme_rejected(self):
        with pytest.raises(ValueError, match="vertical_mixing"):
            ModelParams(vertical_mixing="kpp")

    def test_pp81_is_default(self):
        assert ModelParams().vertical_mixing == "pp81"


# ---------------------------------------------------------------------------
# 6. PP81 N² accuracy and continuous convective blend
# ---------------------------------------------------------------------------

def _coefficients_at_n2(n2_values, params, dudz=0.0):
    """
    PP81 (kappa, nu) with prescribed N² on single-interior-face columns.

    One column per N² value; T is set so that N² = -g alpha_T dT / dz, and
    u has the same shear dudz in every column (S² = dudz²).  kappa is
    column-local; nu is returned at u-points, which averages neighbouring
    columns, so it is only meaningful when all values are equal (use
    ``_nu_at_n2``).
    """
    from OceanJAX.Physics.mixing import pp81_coefficients
    n2_values = np.asarray(n2_values, np.float64)
    n = len(n2_values)
    z = np.array([10.0, 30.0])                           # one interior face, dz_w = 20 m
    grid = OceanGrid.create((0.0, float(n)), (10.0, 11.0), z, n, 1)
    dT = -n2_values * 20.0 / (params.g * params.alpha_T)
    T = np.stack([np.full(n, 15.0), 15.0 + dT], axis=-1)[:, None, :]
    sh = (n, 1, 2)
    u = jnp.asarray(np.broadcast_to(dudz * z, sh), jnp.float32)
    kappa, nu_u, _ = pp81_coefficients(
        jnp.asarray(T, jnp.float32), jnp.full(sh, 35.0),
        u, jnp.zeros(sh), grid, params)
    return np.asarray(kappa)[:, 0, 1], np.asarray(nu_u)[:, 0, 1]


def _nu_at_n2(n2, params, dudz=0.0):
    """nu at a u-point between two columns with the same N²."""
    return float(_coefficients_at_n2([n2, n2], params, dudz)[1][0])


class TestPP81Continuity:

    P = ModelParams(vertical_mixing="pp81")

    def test_n2_matches_float64_density_difference(self):
        """N² agrees with a float64 evaluation of (g/rho0) d(rho)/dz."""
        from OceanJAX.Physics.mixing import buoyancy_and_shear
        p = self.P
        grid = _pp81_grid()
        rng = np.random.default_rng(3)
        sh = (grid.Nx, grid.Ny, grid.Nz)
        T = (20.0 - 0.05 * np.asarray(grid.z_c) + rng.normal(0, 0.01, sh)).astype(np.float32)
        S = (35.0 + rng.normal(0, 0.01, sh)).astype(np.float32)
        n2, _ = buoyancy_and_shear(jnp.asarray(T), jnp.asarray(S), jnp.zeros(sh),
                                   jnp.zeros(sh), grid, p)
        T64, S64 = T.astype(np.float64), S.astype(np.float64)
        rho = p.rho0 * (1 - p.alpha_T * (T64 - p.T_ref) + p.beta_S * (S64 - p.S_ref))
        ref = p.g / p.rho0 * np.diff(rho, axis=-1) / 20.0
        err = np.abs(np.asarray(n2, np.float64)[:, :, 1:-1] - ref).max()
        assert err < 1e-9, f"N² error {err:.2e} s⁻² (density differencing gives ~1e-8)"

    def test_uniform_column_has_zero_n2(self):
        from OceanJAX.Physics.mixing import buoyancy_and_shear
        grid = _pp81_grid()
        sh = (grid.Nx, grid.Ny, grid.Nz)
        n2, _ = buoyancy_and_shear(jnp.full(sh, 17.3), jnp.full(sh, 34.7),
                                   jnp.zeros(sh), jnp.zeros(sh), grid, self.P)
        assert np.all(np.asarray(n2) == 0.0)

    def test_blend_end_points(self):
        p = self.P
        n2c = p.vmix_n2_ramp
        conv = p.vmix_convective
        kappa, _ = _coefficients_at_n2([1e-5, 0.0, -0.5 * n2c, -n2c, -10 * n2c], p)
        # stable, no shear: Ri huge, background value
        np.testing.assert_allclose(kappa[0], p.kappa_v, rtol=1e-3)
        # neutral: Ri = 0
        np.testing.assert_allclose(kappa[1], p.pp81_nu0 + p.kappa_v, rtol=1e-6)
        # half-way: smoothstep(0.5) = 0.5 (dT ~ 0.005 K on 15 K: float32 N² ~2e-4 rel.)
        np.testing.assert_allclose(kappa[2], 0.5 * (p.pp81_nu0 + p.kappa_v + conv), rtol=1e-3)
        np.testing.assert_allclose(kappa[3:], conv, rtol=1e-6)
        np.testing.assert_allclose(_nu_at_n2(0.0, p), p.pp81_nu0 + p.nu_v, rtol=1e-6)
        np.testing.assert_allclose(_nu_at_n2(-0.5 * n2c, p),
                                   0.5 * (p.pp81_nu0 + p.nu_v + conv), rtol=1e-3)
        np.testing.assert_allclose(_nu_at_n2(-n2c, p), conv, rtol=1e-6)

    def test_continuous_and_monotone_across_zero(self):
        """
        With shear present (S² = 1e-4 s⁻², so Ri = N²/S² is well defined
        around N² = 0) kappa has no jump anywhere: neighbouring samples
        differ by O(step).
        """
        p = self.P
        n2c = p.vmix_n2_ramp
        n2 = np.linspace(-1.5 * n2c, 0.5 * n2c, 401)
        kappa, _ = _coefficients_at_n2(n2, p, dudz=0.01)
        step = n2[1] - n2[0]
        # max slope: smoothstep 1.5 / N²_c on the unstable side; PP81 on the
        # stable side, d(kappa)/dN² = -3 alpha nu0 / S² at Ri = 0
        slope = max(1.5 * (p.vmix_convective - p.pp81_nu0) / n2c,
                    3 * p.pp81_alpha * p.pp81_nu0 / 1e-4)
        assert np.abs(np.diff(kappa)).max() <= 1.05 * slope * step
        assert np.all(np.diff(kappa) <= 1e-9), "mixing must not increase with N²"

    def test_no_convective_jump_without_shear(self):
        """
        Without shear the unstable side is still continuous up to N² = 0.
        (On the stable side Ri = N²/max(S², 1e-12) rises steeply, so kappa
        falls to the background within N² ~ 1e-11; that is PP81 itself.)
        """
        p = self.P
        n2 = np.linspace(-1.5 * p.vmix_n2_ramp, 0.0, 301)
        kappa, _ = _coefficients_at_n2(n2, p)
        max_jump = 1.5 * (p.vmix_convective - p.pp81_nu0) * (n2[1] - n2[0]) / p.vmix_n2_ramp
        assert np.abs(np.diff(kappa)).max() <= 1.05 * max_jump

    def test_zero_ramp_recovers_hard_switch(self):
        p = ModelParams(vertical_mixing="pp81", vmix_n2_ramp=0.0)
        kappa, _ = _coefficients_at_n2([-1e-9, 0.0], p)
        np.testing.assert_allclose(kappa[0], p.vmix_convective, rtol=1e-6)
        np.testing.assert_allclose(kappa[1], p.pp81_nu0 + p.kappa_v, rtol=1e-6)

    def test_stable_faces_bit_identical_to_hard_switch(self):
        """The blend only touches N² < 0: stable faces match vmix_n2_ramp = 0."""
        from OceanJAX.Physics.mixing import pp81_coefficients
        grid = _pp81_grid()
        T, S, u, v = _column_fields(grid, dTdz=-0.05, dudz=0.01)
        hard = ModelParams(vertical_mixing="pp81", vmix_n2_ramp=0.0)
        for a, b in zip(pp81_coefficients(T, S, u, v, grid, self.P),
                        pp81_coefficients(T, S, u, v, grid, hard)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_gradient_finite_near_neutral(self):
        """d(sum kappa)/dT is finite on neutral, blended and convective faces."""
        from OceanJAX.Physics.mixing import pp81_coefficients
        p = self.P
        grid = _pp81_grid()
        sh = (grid.Nx, grid.Ny, grid.Nz)
        dz = np.diff(np.asarray(grid.z_c))[0]
        # warm-below increments spanning 0 .. 2 N²_c
        dT = np.linspace(0.0, 2.0, grid.Nx)[:, None, None] * p.vmix_n2_ramp * dz / (p.g * p.alpha_T)
        T0 = jnp.asarray(15.0 + dT * np.arange(grid.Nz)[None, None, :]
                         * np.ones(sh), jnp.float32)

        def total(T):
            kappa, _, _ = pp81_coefficients(T, jnp.full(sh, 35.0), jnp.zeros(sh),
                                            jnp.zeros(sh), grid, p)
            return jnp.sum(kappa)

        g = np.asarray(jax.grad(total)(T0))
        assert np.all(np.isfinite(g))
        assert np.abs(g).max() > 0
