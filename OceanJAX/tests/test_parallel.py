"""
Tests for OceanJAX.parallel (Phase 1 — ensemble parallelism)
=============================================================
Four groups of properties are verified:

  1. batch_step output shape and finiteness
       A batch of N states advanced one step must preserve the leading
       batch axis and contain only finite values.

  2. batch_run correctness
       N independent runs of n_steps must produce finite results, and
       identical initial conditions must yield identical trajectories
       (member 0 == member 1 when no perturbation is applied).

  3. sharded_ensemble_run vs batch_run agreement
       sharded_ensemble_run must produce bit-identical results to
       batch_run on a single device (no sharding overhead, no numerical
       difference).

  4. T-perturbation diversity
       When ENSEMBLE_PERTURB_T > 0 is applied, different members must
       start with different T fields; after integration the spread must
       remain non-zero (members diverge).

Running
-------
    pytest OceanJAX/tests/test_parallel.py -v
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, create_rest_state
from OceanJAX.parallel.ensemble import batch_step, batch_run, sharded_ensemble_run


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_grid():
    """8×6×4 flat-bottom open-ocean grid — fast enough for all tests."""
    z_levels = np.array([12.5, 37.5, 62.5, 87.5], dtype=np.float64)
    return OceanGrid.create(
        lon_bounds=(-10.0, 10.0),
        lat_bounds=(-5.0, 5.0),
        depth_levels=z_levels,
        Nx=8,
        Ny=6,
    )


@pytest.fixture(scope="module")
def default_params():
    return ModelParams(dt=300.0)


@pytest.fixture(scope="module")
def base_state(small_grid):
    return create_rest_state(small_grid, T_background=10.0, S_background=35.0)


def _make_batched(base, n: int):
    """Stack n copies of base_state into a batched OceanState."""
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n), base)


# ---------------------------------------------------------------------------
# Group 1 — batch_step shape and finiteness
# ---------------------------------------------------------------------------

class TestBatchStep:
    def test_output_shape_n2(self, base_state, small_grid, default_params):
        """batch_step preserves the leading batch axis (N=2)."""
        N = 2
        batched = _make_batched(base_state, N)
        out = batch_step(batched, small_grid, default_params, forcing=None)
        assert out.T.shape == (N, small_grid.Nx, small_grid.Ny, small_grid.Nz)
        assert out.eta.shape == (N, small_grid.Nx, small_grid.Ny)

    def test_output_finite_n3(self, base_state, small_grid, default_params):
        """All fields are finite after one batch step (N=3)."""
        N = 3
        batched = _make_batched(base_state, N)
        out = batch_step(batched, small_grid, default_params)
        assert np.all(np.isfinite(np.array(out.T))),   "T contains non-finite values"
        assert np.all(np.isfinite(np.array(out.S))),   "S contains non-finite values"
        assert np.all(np.isfinite(np.array(out.eta))), "eta contains non-finite values"
        assert np.all(np.isfinite(np.array(out.u))),   "u contains non-finite values"

    def test_time_advances(self, base_state, small_grid, default_params):
        """Each member's time counter advances by exactly dt."""
        N = 2
        batched = _make_batched(base_state, N)
        out = batch_step(batched, small_grid, default_params)
        times = np.array(out.time)       # shape (N,)
        assert times.shape == (N,)
        assert np.allclose(times, default_params.dt)


# ---------------------------------------------------------------------------
# Group 2 — batch_run correctness
# ---------------------------------------------------------------------------

N_STEPS = 288   # 1 simulated day at dt=300 s


class TestBatchRun:
    def test_finite_after_n_steps(self, base_state, small_grid, default_params):
        """batch_run produces finite fields after 288 steps (N=2)."""
        N = 2
        batched = _make_batched(base_state, N)
        final, hist = batch_run(batched, small_grid, default_params,
                                n_steps=N_STEPS, save_history=False)
        assert hist is None
        assert np.all(np.isfinite(np.array(final.T)))
        assert np.all(np.isfinite(np.array(final.S)))
        assert np.all(np.isfinite(np.array(final.eta)))

    def test_identical_members_identical_output(self, base_state, small_grid, default_params):
        """Identical initial conditions yield identical trajectories across members."""
        N = 3
        batched = _make_batched(base_state, N)
        final, _ = batch_run(batched, small_grid, default_params, n_steps=N_STEPS)
        T = np.array(final.T)
        for i in range(1, N):
            np.testing.assert_array_equal(
                T[0], T[i],
                err_msg=f"member 0 and member {i} diverged despite identical IC"
            )

    def test_output_shape(self, base_state, small_grid, default_params):
        """Output shape is (N, Nx, Ny, Nz) after batch_run."""
        N = 2
        batched = _make_batched(base_state, N)
        final, _ = batch_run(batched, small_grid, default_params, n_steps=N_STEPS)
        assert final.T.shape == (N, small_grid.Nx, small_grid.Ny, small_grid.Nz)

    def test_time_counter(self, base_state, small_grid, default_params):
        """Time counter equals n_steps * dt for every member."""
        N = 2
        batched = _make_batched(base_state, N)
        final, _ = batch_run(batched, small_grid, default_params, n_steps=N_STEPS)
        expected = N_STEPS * default_params.dt
        times = np.array(final.time)
        assert np.allclose(times, expected), f"expected {expected}, got {times}"


# ---------------------------------------------------------------------------
# Group 3 — sharded_ensemble_run vs batch_run agreement
# ---------------------------------------------------------------------------

class TestShardedEnsembleRun:
    def test_matches_batch_run(self, base_state, small_grid, default_params):
        """sharded_ensemble_run gives bit-identical results to batch_run (single device)."""
        N = 2
        batched = _make_batched(base_state, N)

        ref_final, _ = batch_run(batched, small_grid, default_params, n_steps=N_STEPS)
        she_final, _ = sharded_ensemble_run(batched, small_grid, default_params,
                                             n_steps=N_STEPS,
                                             devices=jax.devices()[:1])

        np.testing.assert_array_equal(
            np.array(ref_final.T),
            np.array(she_final.T),
            err_msg="T differs between batch_run and sharded_ensemble_run",
        )
        np.testing.assert_array_equal(
            np.array(ref_final.eta),
            np.array(she_final.eta),
            err_msg="eta differs between batch_run and sharded_ensemble_run",
        )

    def test_multi_device_matches_batch_run(self, base_state, small_grid, default_params):
        """Members spread over 2 devices match batch_run bit for bit (no cross-member ops)."""
        if len(jax.devices()) < 2:
            pytest.skip("needs 2 devices")
        N = 2
        batched = _make_batched(base_state, N)
        ref_final, _ = batch_run(batched, small_grid, default_params, n_steps=N_STEPS)
        she_final, _ = sharded_ensemble_run(batched, small_grid, default_params,
                                             n_steps=N_STEPS,
                                             devices=jax.devices()[:2])
        assert len(she_final.T.sharding.device_set) == 2
        np.testing.assert_array_equal(np.array(ref_final.T), np.array(she_final.T))
        np.testing.assert_array_equal(np.array(ref_final.eta), np.array(she_final.eta))

    def test_finite_output(self, base_state, small_grid, default_params):
        """sharded_ensemble_run produces finite fields."""
        N = 2
        batched = _make_batched(base_state, N)
        final, _ = sharded_ensemble_run(batched, small_grid, default_params,
                                         n_steps=N_STEPS,
                                         devices=jax.devices()[:1])
        assert np.all(np.isfinite(np.array(final.T)))
        assert np.all(np.isfinite(np.array(final.S)))

    def test_invalid_batch_raises(self, base_state, small_grid, default_params):
        """batch size not divisible by n_devices raises ValueError (multi-device only)."""
        n_devices = len(jax.devices())
        if n_devices < 2:
            pytest.skip("only one device available; multi-device check not applicable")
        N = n_devices + 1   # intentionally indivisible
        batched = _make_batched(base_state, N)
        with pytest.raises(ValueError, match="divisible"):
            sharded_ensemble_run(batched, small_grid, default_params, n_steps=10)


# ---------------------------------------------------------------------------
# Group 4 — T-perturbation diversity
# ---------------------------------------------------------------------------

class TestEnsemblePerturbation:
    def _make_perturbed(self, base, grid, n: int, perturb: float):
        batched = jax.tree_util.tree_map(lambda x: jnp.stack([x] * n), base)
        keys  = jax.random.split(jax.random.PRNGKey(0), n)
        noise = jax.vmap(
            lambda k: jax.random.normal(k, base.T.shape, dtype=jnp.float32)
        )(keys) * perturb
        T_new = (batched.T + noise) * grid.mask_c[None]
        return eqx.tree_at(lambda s: s.T, batched, T_new)

    def test_members_start_different(self, base_state, small_grid, default_params):
        """After perturbation, member T fields must differ."""
        N, PERTURB = 4, 0.1
        batched = self._make_perturbed(base_state, small_grid, N, PERTURB)
        T0 = np.array(batched.T)
        assert T0.std(axis=0).mean() > 0, "All members are identical after perturbation"

    def test_spread_survives_integration(self, base_state, small_grid, default_params):
        """Ensemble spread must remain non-zero after 288 steps."""
        N, PERTURB = 4, 0.1
        batched = self._make_perturbed(base_state, small_grid, N, PERTURB)
        final, _ = batch_run(batched, small_grid, default_params, n_steps=N_STEPS)
        Tf = np.array(final.T)
        assert np.all(np.isfinite(Tf)), "Non-finite values after perturbed run"
        assert Tf.std(axis=0).mean() > 0, "Ensemble spread collapsed to zero"

    def test_perturb_zero_gives_identical_members(self, base_state, small_grid, default_params):
        """Zero perturbation must give identical members (regression guard)."""
        N = 3
        batched = _make_batched(base_state, N)
        final, _ = batch_run(batched, small_grid, default_params, n_steps=N_STEPS)
        T = np.array(final.T)
        assert T.std(axis=0).max() == 0.0, "Members diverged with zero perturbation"
