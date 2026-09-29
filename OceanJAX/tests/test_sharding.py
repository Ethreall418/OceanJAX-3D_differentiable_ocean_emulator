"""
Tests for OceanJAX.parallel.sharding (domain decomposition, phase A)
=====================================================================
Run on the 8 simulated CPU devices set up in conftest.py.

  1. Mesh and partition specs
       make_mesh shapes / errors; which arrays are split over x / y and
       which are replicated; divisibility checks.

  2. Correctness
       * 1 x 1 mesh is bit-identical to jax.jit(run).
       * 2x4, 4x2, 8x1, 1x4 meshes agree with the single-device run to
         round-off (~1 ulp per operation from different XLA fusion), on a
         domain with bathymetry, land, closed walls, PP81, bottom drag and
         surface forcing.
       * A resting uniform ocean stays exactly at rest and exactly uniform.
       * Ensemble x domain mesh (batch=2, x=2, y=2) matches batch_run.
       * Gradients through sharded_run match the single-device gradients.

  3. Communication
       * The implicit vertical solvers are column-local: no collectives at
         all when x / y are sharded (a reshape to (Nx*Ny, Nz) would
         all-gather the whole field).
       * A full sharded run contains no all-gather / all-to-all, and every
         collective moves less than one local block (halo exchanges only).

Running
-------
    pytest OceanJAX/tests/test_sharding.py -v
"""

from __future__ import annotations

import re

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from jax.sharding import PartitionSpec as P

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, create_from_arrays, create_rest_state
from OceanJAX.timeStepping import run, SurfaceForcing
from OceanJAX.Physics.mixing import implicit_vertical_mix, implicit_vertical_visc
from OceanJAX.parallel.ensemble import batch_run
from OceanJAX.parallel.sharding import (
    make_mesh,
    field_spec,
    shard_grid,
    shard_state,
    shard_forcing,
    sharded_run,
    gather_to_host,
    init_distributed,
    _sharded_run_jit,
    _traced_params,
)

pytestmark = pytest.mark.skipif(
    len(jax.devices()) < 8,
    reason="needs 8 devices (conftest sets --xla_force_host_platform_device_count=8)",
)

NX, NY, NZ = 16, 12, 8
N_STEPS    = 40
FIELDS     = ("u", "v", "w", "T", "S", "eta")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid():
    """Regional domain with random bathymetry, a land block and closed walls."""
    rng = np.random.default_rng(0)
    H = rng.uniform(50.0, 420.0, (NX, NY))
    H[:3, :4] = 0.0
    z = (np.arange(NZ) + 0.5) * 50.0
    return OceanGrid.create((-40.0, -10.0), (-10.0, 15.0), z, NX, NY,
                            bathymetry=H, periodic_x=False)


@pytest.fixture(scope="module")
def params():
    return ModelParams(dt=600.0, nu_h=2e4, vertical_mixing="pp81")


@pytest.fixture(scope="module")
def state(grid):
    rng = np.random.default_rng(1)
    z = np.asarray(grid.z_c)
    T = 20.0 - 0.03 * z[None, None, :] + rng.normal(0.0, 0.3, (NX, NY, NZ))
    S = 35.0 + rng.normal(0.0, 0.05, (NX, NY, NZ))
    u = rng.normal(0.0, 0.05, (NX, NY, NZ))
    v = rng.normal(0.0, 0.05, (NX, NY, NZ))
    return create_from_arrays(grid, u, v, T, S, np.zeros((NX, NY)))


@pytest.fixture(scope="module")
def forcing():
    rng = np.random.default_rng(2)
    return SurfaceForcing(*(
        jnp.asarray(rng.normal(0.0, s, (N_STEPS, NX, NY)), jnp.float32)
        for s in (100.0, 1e-7, 0.1, 0.1)
    ))


@pytest.fixture(scope="module")
def reference(state, grid, params, forcing):
    """Single-device run, the path used by experiment.py."""
    final, _ = jax.jit(run, static_argnames=("n_steps", "save_history"))(
        state, grid, params, n_steps=N_STEPS, forcing_sequence=forcing)
    return final


def _max_abs_diff(a, b, name):
    return float(np.max(np.abs(np.asarray(getattr(a, name))
                               - np.asarray(getattr(b, name)))))


def _collectives(fn, *args):
    """(op, element count) for every collective in the compiled HLO."""
    txt = fn.lower(*args).compile()
    txt = (txt.compiled if hasattr(txt, "compiled") else txt).as_text()
    found = re.findall(
        r"= \w+\[([\d,]*)\]\S* "
        r"(all-gather|all-reduce|all-to-all|collective-permute|reduce-scatter)"
        r"(?:-start)?\(", txt)
    return [(op, int(np.prod([int(d) for d in dims.split(",") if d])))
            for dims, op in found]


# ---------------------------------------------------------------------------
# 1. Mesh and partition specs
# ---------------------------------------------------------------------------

class TestMeshAndSpecs:
    def test_mesh_shape(self):
        mesh = make_mesh(n_x=2, n_y=4)
        assert mesh.axis_names == ("batch", "x", "y")
        assert dict(mesh.shape) == {"batch": 1, "x": 2, "y": 4}

    def test_mesh_too_many_devices(self):
        with pytest.raises(ValueError, match="needs 16 devices"):
            make_mesh(n_x=4, n_y=4)

    def test_mesh_invalid_size(self):
        with pytest.raises(ValueError, match=">= 1"):
            make_mesh(n_x=0)

    def test_field_spec(self):
        assert field_spec((NX, NY, NZ), NX, NY) == P("x", "y")
        assert field_spec((NX, NY), NX, NY) == P("x", "y")
        assert field_spec((NX,), NX, NY) == P()                    # lon_c
        assert field_spec((NZ + 1,), NX, NY) == P()                # z_w
        assert field_spec((), NX, NY) == P()                       # time
        assert field_spec((N_STEPS, NX, NY), NX, NY, n_lead=1) == P(None, "x", "y")
        assert field_spec((4, NX, NY, NZ), NX, NY, 1, batch=True) == P("batch", "x", "y")
        assert field_spec((4,), NX, NY, 1, batch=True) == P("batch")
        assert (field_spec((4, N_STEPS, NX, NY), NX, NY, 2, batch=True)
                == P("batch", None, "x", "y"))

    def test_shard_grid_layout(self, grid):
        gs = shard_grid(grid, make_mesh(2, 4))
        for name in ("dx_c", "f_c", "mask_c", "mask_w", "H", "volume_c"):
            assert getattr(gs, name).sharding.spec == P("x", "y"), name
            shard = getattr(gs, name).addressable_shards[0].data
            assert shard.shape[:2] == (NX // 2, NY // 4), name
        for name in ("lon_c", "lat_c", "z_c", "dz_c", "dz_w"):
            assert getattr(gs, name).sharding.is_fully_replicated, name
        assert gs.periodic_x is False and gs.Nx == NX

    def test_shard_state_layout(self, state):
        ss = shard_state(state, make_mesh(4, 2))
        assert ss.T.sharding.spec == P("x", "y")
        assert ss.w.addressable_shards[0].data.shape == (NX // 4, NY // 2, NZ + 1)
        assert ss.time.sharding.is_fully_replicated

    def test_indivisible_grid_raises(self, grid):
        with pytest.raises(ValueError, match="not divisible"):
            shard_grid(grid, make_mesh(1, 8))       # Ny = 12

    def test_state_without_batch_axis_on_batch_mesh_raises(self, state):
        with pytest.raises(ValueError, match="no ensemble axis"):
            shard_state(state, make_mesh(2, 2, n_batch=2))

    def test_forcing_layout(self, forcing, grid):
        fs = shard_forcing(forcing, make_mesh(2, 4), grid)
        assert fs.heat_flux.sharding.spec == P(None, "x", "y")


# ---------------------------------------------------------------------------
# 2. Correctness
# ---------------------------------------------------------------------------

class TestCorrectness:
    def test_single_device_mesh_bit_identical(self, state, grid, params, forcing,
                                              reference):
        final, _ = sharded_run(state, grid, params, N_STEPS, make_mesh(1, 1),
                               forcing_sequence=forcing)
        for name in FIELDS:
            np.testing.assert_array_equal(
                np.asarray(getattr(final, name)), np.asarray(getattr(reference, name)),
                err_msg=name)

    @pytest.mark.parametrize("n_x,n_y", [(2, 4), (4, 2), (8, 1), (1, 4)])
    def test_matches_single_device(self, state, grid, params, forcing, reference,
                                   n_x, n_y):
        final, _ = sharded_run(state, grid, params, N_STEPS, make_mesh(n_x, n_y),
                               forcing_sequence=forcing)
        assert final.T.sharding.spec == P("x", "y")
        assert len(final.T.sharding.device_set) == n_x * n_y
        # Round-off level: a few float32 ulp of each field's magnitude.
        tol = {"u": 1e-6, "v": 1e-6, "w": 1e-9, "T": 2e-5, "S": 2e-5, "eta": 2e-6}
        for name in FIELDS:
            d = _max_abs_diff(final, reference, name)
            assert d <= tol[name], f"{name}: max |diff| = {d:.3e} > {tol[name]:.0e}"
        assert int(final.step_count) == N_STEPS

    def test_rest_state_exactly_preserved(self, grid, params):
        """Uniform T/S at rest, no forcing: stays exactly uniform and at rest."""
        rest = create_rest_state(grid, T_background=12.0, S_background=35.0)
        final, _ = sharded_run(rest, grid, params, N_STEPS, make_mesh(2, 4))
        wet = np.asarray(grid.mask_c) > 0
        T, S = np.asarray(final.T), np.asarray(final.S)
        assert np.all(T[wet] == np.float32(12.0))
        assert np.all(S[wet] == np.float32(35.0))
        for name in ("u", "v", "w", "eta"):
            assert np.all(np.asarray(getattr(final, name)) == 0.0), name

    def test_pp81_near_neutral_insensitive_to_roundoff(self):
        """
        Nearly neutral column (uniform T + 0.05 K noise, surface cooling,
        wind): the ~1 ulp differences of the partitioned programme must not
        be amplified by PP81.  With the old hard convective switch at N² = 0
        (vmix_n2_ramp = 0) u differs by ~1.5e-5 after 144 steps; with the
        continuous blend by ~1e-6.
        """
        nx, ny, nz, n = 16, 12, 8, 144
        g = OceanGrid.create((-40.0, -10.0), (-10.0, 15.0),
                             (np.arange(nz) + 0.5) * 25.0, nx, ny)
        rng = np.random.default_rng(5)
        st = create_rest_state(g, 10.0, 35.0)
        noise = jnp.asarray(rng.normal(0.0, 0.05, st.T.shape), jnp.float32)
        st = eqx.tree_at(lambda s: s.T, st, (st.T + noise) * g.mask_c)
        f = SurfaceForcing(jnp.full((n, nx, ny), -50.0), jnp.zeros((n, nx, ny)),
                           jnp.full((n, nx, ny), 0.05), jnp.zeros((n, nx, ny)))
        p = ModelParams(dt=600.0, nu_h=2e4, vertical_mixing="pp81")
        ref, _ = jax.jit(run, static_argnums=(3,))(st, g, p, n, f)
        out, _ = sharded_run(st, g, p, n, make_mesh(2, 4), forcing_sequence=f)
        assert _max_abs_diff(out, ref, "u") < 3e-6
        assert _max_abs_diff(out, ref, "T") < 1e-5

    def test_chunks_reuse_sharded_state(self, state, grid, params, reference, forcing):
        """Two chunks of N/2 steps, feeding the sharded state back in."""
        mesh = make_mesh(2, 4)
        half = N_STEPS // 2
        f1 = jax.tree_util.tree_map(lambda x: x[:half], forcing)
        f2 = jax.tree_util.tree_map(lambda x: x[half:], forcing)
        s1, _ = sharded_run(state, grid, params, half, mesh, forcing_sequence=f1)
        assert s1.T.sharding.spec == P("x", "y")
        s2, _ = sharded_run(s1, grid, params, half, mesh, forcing_sequence=f2)
        assert _max_abs_diff(s2, reference, "T") <= 2e-5

    def test_ensemble_times_domain(self, state, grid, params, forcing):
        """batch=2 x (x=2, y=2): 2 perturbed members, shared forcing."""
        members = [state, eqx.tree_at(lambda s: s.T, state,
                                      (state.T + 0.1) * grid.mask_c)]
        batched = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *members)
        f_batch = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x[None], (2,) + x.shape), forcing)

        ref, _ = jax.jit(batch_run, static_argnames=("n_steps", "save_history"))(
            batched, grid, params, n_steps=N_STEPS, forcing_sequence=f_batch)

        mesh = make_mesh(n_x=2, n_y=2, n_batch=2)
        shared, _ = sharded_run(batched, grid, params, N_STEPS, mesh,
                                forcing_sequence=forcing)
        per_member, _ = sharded_run(batched, grid, params, N_STEPS, mesh,
                                    forcing_sequence=f_batch)
        assert shared.T.sharding.spec == P("batch", "x", "y")
        assert shared.time.shape == (2,)
        for out in (shared, per_member):
            assert _max_abs_diff(out, ref, "T") <= 2e-5
            assert _max_abs_diff(out, ref, "u") <= 1e-6
        assert _max_abs_diff(shared, per_member, "T") == 0.0
        # Members really differ (the perturbation was not lost).
        T = np.asarray(shared.T)
        assert np.abs(T[0] - T[1]).max() > 0.05

    def test_gradient_matches_single_device(self, state, grid, params):
        """d(mean SST)/d(T0) through a short sharded run (adjoint path)."""
        n = 5
        wet_top = grid.mask_c[:, :, 0]

        def loss_single(T0):
            s = eqx.tree_at(lambda s: s.T, state, T0)
            final, _ = run(s, grid, params, n)
            return jnp.sum(final.T[:, :, 0] * wet_top) / jnp.sum(wet_top)

        mesh = make_mesh(2, 4)
        grid_s = shard_grid(grid, mesh)

        def loss_sharded(T0):
            s = eqx.tree_at(lambda s: s.T, state, T0)
            final, _ = sharded_run(s, grid_s, params, n, mesh)
            return jnp.sum(final.T[:, :, 0] * wet_top) / jnp.sum(wet_top)

        g_ref = np.asarray(jax.jit(jax.grad(loss_single))(state.T))
        g_shd = np.asarray(jax.grad(loss_sharded)(state.T))
        assert np.abs(g_ref).max() > 0
        np.testing.assert_allclose(g_shd, g_ref, rtol=1e-4,
                                   atol=1e-6 * np.abs(g_ref).max())


# ---------------------------------------------------------------------------
# 3. Communication
# ---------------------------------------------------------------------------

class TestCommunication:
    def test_vertical_solvers_are_column_local(self, grid, state):
        mesh = make_mesh(2, 4)
        gs, ss = shard_grid(grid, mesh), shard_state(state, mesh)
        mix = jax.jit(lambda T, g: implicit_vertical_mix(T, 1e-2, 600.0, g))
        visc = jax.jit(lambda u, g: implicit_vertical_visc(u, 1e-2, 600.0, g, g.mask_u))
        assert _collectives(mix, ss.T, gs) == []
        assert _collectives(visc, ss.u, gs) == []
        out = mix(ss.T, gs)
        assert out.sharding.spec == P("x", "y")

    def test_run_has_only_halo_exchanges(self, state, grid, params, forcing):
        mesh = make_mesh(2, 4)
        args = (shard_state(state, mesh), shard_grid(grid, mesh),
                _traced_params(params), shard_forcing(forcing, mesh, grid),
                None, N_STEPS, False, mesh)
        ops = _collectives(_sharded_run_jit, *args)
        assert ops, "expected halo exchanges between neighbouring devices"
        kinds = {op for op, _ in ops}
        assert "all-gather" not in kinds and "all-to-all" not in kinds, kinds
        local_block = (NX // 2) * (NY // 4) * NZ
        biggest = max(n for _, n in ops)
        assert biggest < local_block, (
            f"a collective moves {biggest} elements >= local block {local_block}")


# ---------------------------------------------------------------------------
# 4. Host transfer and distributed init
# ---------------------------------------------------------------------------

class TestHostAndDistributed:
    def test_gather_to_host(self, state):
        ss = shard_state(state, make_mesh(2, 4))
        host = gather_to_host(ss)
        assert isinstance(host.T, np.ndarray)
        np.testing.assert_array_equal(host.T, np.asarray(state.T))

    def test_init_distributed_single_process_is_noop(self, monkeypatch):
        monkeypatch.delenv("SLURM_NTASKS", raising=False)
        assert init_distributed() is False
        assert jax.process_count() == 1
