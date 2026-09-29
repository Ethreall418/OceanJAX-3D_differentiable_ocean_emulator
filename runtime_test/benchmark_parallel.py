"""
benchmark_parallel.py
=====================
Wall-time benchmarks for OceanJAX.parallel.

Ensemble mode (default) — three execution modes for identical workloads:

  single_run         — jax.jit(run())  for a single OceanState
  batch_run          — eqx.filter_vmap over N ensemble members (1 GPU)
  sharded_ensemble   — NamedSharding across all available GPUs

Domain mode — one large domain split over x / y (sharding.sharded_run):
  times the same run on meshes of 1, 2, 4, ... devices and reports the
  speed-up and parallel efficiency relative to one device.

Usage
-----
    python runtime_test/benchmark_parallel.py [N_ENSEMBLE] [N_STEPS]
    python runtime_test/benchmark_parallel.py domain [N_STEPS] [NX NY NZ]

Defaults: N_ENSEMBLE=4, N_STEPS=288 (one simulated day at dt=300 s);
domain mode NX, NY, NZ = 128, 96, 20 and N_STEPS=96.

To try domain mode on a CPU-only machine, simulate devices with
    XLA_FLAGS=--xla_force_host_platform_device_count=8
(simulated devices share the same cores, so this checks that the
decomposition runs, not that it is faster; measure speed-up on GPUs).

Output
------
Prints a table of wall times (compilation excluded) and throughput.
In ensemble mode on a single-device machine, sharded_ensemble_run falls
back to batch_run, so the two should match.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

# Allow running from repo root or from runtime_test/
sys.path.insert(0, str(Path(__file__).parent.parent))

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, create_rest_state
from OceanJAX.timeStepping import run as ocean_run
from OceanJAX.parallel.ensemble import batch_run, sharded_ensemble_run
from OceanJAX.parallel.sharding import make_mesh, shard_grid, shard_state, sharded_run


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOMAIN_MODE = len(sys.argv) > 1 and sys.argv[1] == "domain"
_args       = sys.argv[2:] if DOMAIN_MODE else sys.argv[1:]

if DOMAIN_MODE:
    N_ENSEMBLE = 1
    N_STEPS    = int(_args[0]) if len(_args) > 0 else 96
else:
    N_ENSEMBLE = int(_args[0]) if len(_args) > 0 else 4
    N_STEPS    = int(_args[1]) if len(_args) > 1 else 288

NX, NY, NZ  = 20, 15, 10       # matches experiment.py default
if DOMAIN_MODE:
    NX, NY, NZ = (int(a) for a in _args[1:4]) if len(_args) >= 4 else (128, 96, 20)
DT          = 300.0


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_grid():
    dz = 500.0 / NZ
    z  = (np.arange(NZ) + 0.5) * dz
    return OceanGrid.create(
        lon_bounds=(-40.0, -5.0),
        lat_bounds=(-15.0, 15.0),
        depth_levels=z,
        Nx=NX, Ny=NY,
    )


def timed(fn, *args, n_runs: int = 3):
    """
    Run fn(*args) n_runs times (after the first call which is used only to
    ensure compilation is done) and return (mean_wall_seconds, last_result).
    """
    # ensure compiled
    result = fn(*args)
    jax.block_until_ready(result[0].T if isinstance(result, tuple) else result.T)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result[0].T if isinstance(result, tuple) else result.T)
        times.append(time.perf_counter() - t0)

    return float(np.mean(times)), result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("OceanJAX parallel benchmark")
    print(f"  grid      : {NX}x{NY}x{NZ}  dt={DT} s")
    print(f"  steps     : {N_STEPS}  ({N_STEPS*DT/86400:.1f} sim-days)")
    print(f"  ensemble  : {N_ENSEMBLE} members")
    print(f"  devices   : {jax.devices()}")
    print("=" * 64)

    grid   = build_grid()
    params = ModelParams(dt=DT)
    base   = create_rest_state(grid, T_background=10.0, S_background=35.0)

    # Batched state: leading axis of size N_ENSEMBLE
    batched = jax.tree_util.tree_map(lambda x: jnp.stack([x] * N_ENSEMBLE), base)

    # JIT-compiled single-run function
    run_jit = jax.jit(ocean_run, static_argnames=("n_steps", "save_history"))

    # ------------------------------------------------------------------
    # [A] single_run: run each member sequentially (baseline)
    # ------------------------------------------------------------------
    print("\n[1/3] Warming up single_run ...")

    def single_sequential(n_steps):
        """Run N_ENSEMBLE members back-to-back (sequential baseline)."""
        s = base
        for _ in range(N_ENSEMBLE):
            s, _ = run_jit(s, grid, params, n_steps=n_steps, save_history=False)
        return s, None

    wall_single, _ = timed(single_sequential, N_STEPS)
    tput_single     = N_ENSEMBLE * N_STEPS / wall_single

    # ------------------------------------------------------------------
    # [B] batch_run
    # ------------------------------------------------------------------
    print("[2/3] Warming up batch_run ...")

    def run_batch(n_steps):
        return batch_run(batched, grid, params, n_steps=n_steps, save_history=False)

    wall_batch, _ = timed(run_batch, N_STEPS)
    tput_batch     = N_ENSEMBLE * N_STEPS / wall_batch

    # ------------------------------------------------------------------
    # [C] sharded_ensemble_run
    # ------------------------------------------------------------------
    print("[3/3] Warming up sharded_ensemble_run ...")

    def run_sharded(n_steps):
        return sharded_ensemble_run(batched, grid, params, n_steps=n_steps,
                                    save_history=False)

    wall_sharded, _ = timed(run_sharded, N_STEPS)
    tput_sharded     = N_ENSEMBLE * N_STEPS / wall_sharded

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    speedup_batch   = wall_single / wall_batch
    speedup_sharded = wall_single / wall_sharded

    print()
    print(f"{'Mode':<25}  {'Wall time (s)':>13}  {'member-steps/s':>15}  {'Speedup':>8}")
    print("-" * 68)
    print(f"{'single_run (sequential)':<25}  {wall_single:13.3f}  {tput_single:15.1f}  {'1.00x':>8}")
    print(f"{'batch_run (vmap)':<25}  {wall_batch:13.3f}  {tput_batch:15.1f}  {speedup_batch:>7.2f}x")
    print(f"{'sharded_ensemble_run':<25}  {wall_sharded:13.3f}  {tput_sharded:15.1f}  {speedup_sharded:>7.2f}x")
    print()

    n_devices = len(jax.devices())
    if n_devices == 1:
        print("Note: only 1 device available — sharded_ensemble_run falls back "
              "to batch_run (identical performance expected).")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Domain decomposition mode
# ---------------------------------------------------------------------------

def _layouts(n_devices: int):
    """
    One (n_x, n_y) mesh per device count 1, 2, 4, ... <= n_devices: the
    most nearly square layout that divides NX x NY.
    """
    out, n = [], 1
    while n <= n_devices:
        best = None
        for n_x in range(1, n + 1):
            if n % n_x:
                continue
            n_y = n // n_x
            if NX % n_x or NY % n_y:
                continue
            # prefer square local blocks
            score = abs(np.log((NX / n_x) / (NY / n_y)))
            if best is None or score < best[0]:
                best = (score, n_x, n_y)
        if best is not None:
            out.append(best[1:])
        n *= 2
    return out


def main_domain():
    n_devices = len(jax.devices())
    print("=" * 64)
    print("OceanJAX domain-decomposition benchmark")
    print(f"  grid      : {NX}x{NY}x{NZ}  dt={DT} s")
    print(f"  steps     : {N_STEPS}")
    print(f"  devices   : {n_devices} x {jax.devices()[0].platform}")
    print("=" * 64)

    grid   = build_grid()
    params = ModelParams(dt=DT, nu_h=2e4, vertical_mixing="pp81")
    base   = create_rest_state(grid, T_background=10.0, S_background=35.0)
    # A little structure so the solvers do real work.
    rng    = np.random.default_rng(0)
    noise  = jnp.asarray(rng.normal(0.0, 0.1, base.T.shape), jnp.float32)
    base   = eqx.tree_at(lambda s: s.T, base, base.T + noise * grid.mask_c)

    rows = []
    for n_x, n_y in _layouts(n_devices):
        mesh   = make_mesh(n_x, n_y)
        grid_s = shard_grid(grid, mesh)
        state  = shard_state(base, mesh)
        print(f"  mesh {n_x}x{n_y} ...", flush=True)
        wall, _ = timed(lambda: sharded_run(state, grid_s, params, N_STEPS, mesh))
        rows.append((n_x, n_y, wall))

    cells = NX * NY * NZ
    wall1 = rows[0][2]
    print()
    print(f"{'Mesh':<8}  {'Devices':>7}  {'Local block':>12}  {'Wall (s)':>9}  "
          f"{'Mcell-steps/s':>13}  {'Speedup':>8}  {'Efficiency':>10}")
    print("-" * 80)
    for n_x, n_y, wall in rows:
        n = n_x * n_y
        print(f"{f'{n_x}x{n_y}':<8}  {n:7d}  {f'{NX//n_x}x{NY//n_y}':>12}  "
              f"{wall:9.3f}  {cells * N_STEPS / wall / 1e6:13.2f}  "
              f"{wall1 / wall:7.2f}x  {wall1 / wall / n:9.0%}")
    if jax.devices()[0].platform == "cpu" and n_devices > 1:
        print("\nNote: CPU devices share the same cores; speed-up is only "
              "meaningful on separate GPUs.")
    print("\nDone.")


if __name__ == "__main__":
    main_domain() if DOMAIN_MODE else main()
