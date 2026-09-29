# Parallel execution in OceanJAX

OceanJAX runs on several devices in two ways. Both are in `OceanJAX/parallel/`.

| Mode | Module | What is split | Communication |
|---|---|---|---|
| Ensemble | `ensemble.py` (`sharded_ensemble_run`) | independent members | none |
| Domain decomposition (phase A) | `sharding.py` (`sharded_run`) | the x / y axes of one domain, optionally combined with ensemble members | halo exchanges inserted by XLA (GSPMD) |

Phase B (`shard_map` with hand-written halo exchange, `halo.py`) is only
planned if phase A turns out to scale poorly on a real cluster.

## How domain decomposition works

`make_mesh(n_x, n_y, n_batch)` builds a device mesh with axes
`("batch", "x", "y")`. `shard_grid`, `shard_state` and `shard_forcing`
place every array whose axes include the horizontal `(Nx, Ny)` pair so that
it is split over `x` and `y`. Everything else is replicated: 1-D
coordinates, `dz`, scalars and `ModelParams`. The model code is unchanged.

- **Horizontal stencils.** `jnp.roll` and shifted concatenates compile to
  collective-permutes of one-cell halos between neighbouring devices.
- **Column operations.** Hydrostatic pressure, `compute_w` and the implicit
  vertical solvers are local, because z is never sharded. The tridiagonal
  solvers map over columns with nested `vmap`s. A reshape to
  `(Nx*Ny, Nz)` would all-gather the whole field.
- **Tests.** `tests/test_sharding.py` checks the compiled programme: it
  contains no all-gather, and no collective moves more than a halo.

Constraints: `Nx % n_x == 0`, `Ny % n_y == 0`, and the ensemble size must be
divisible by `n_batch`.

### Numerical agreement

- A `1 x 1` mesh is bit-identical to `jax.jit(run)`.
- **Larger meshes** agree to round-off, about 1 ulp per operation. XLA
  fuses the partitioned stencils differently, which is the same kind of
  difference as CPU vs GPU. A uniform resting ocean stays exactly uniform.
- **PP81 caveat.** The PP81 closure switches discontinuously at N² = 0 (shear
  mixing vs convective 0.1 m²/s). Where N² ≈ 0, for example with grid-scale
  random T perturbations, a 1-ulp difference can flip the branch, and
  results then differ at O(1e-3). This is a property of the closure, not of
  the decomposition: the same happens between CPU and GPU. With constant
  mixing the difference stays at round-off.

## Python API

```python
from OceanJAX.parallel.sharding import (
    make_mesh, shard_grid, shard_state, sharded_run, gather_to_host)

mesh   = make_mesh(n_x=4, n_y=2)            # 8 devices
grid_s = shard_grid(grid, mesh)             # once
state  = shard_state(state, mesh)

for chunk in range(n_chunks):
    forcing = ...                           # (n_steps, Nx, Ny) host arrays
    state, _ = sharded_run(state, grid_s, params, n_steps, mesh,
                           forcing_sequence=forcing)
    host = gather_to_host(state)            # numpy, full field (all processes)
```

- **Ensembles.** Pass a batched state `(B, Nx, Ny, Nz)` and a mesh with
  `n_batch > 1`. A forcing sequence `(n_steps, Nx, Ny)` is shared by all
  members. A sequence of shape `(B, n_steps, Nx, Ny)` gives per-member
  forcing.
- **Gradients.** `jax.grad` works through `sharded_run`.
- **Host-side setup.** Compute whole-grid quantities such as
  `munk_viscosity(grid)` on the host grid, before sharding.

In `experiment.py`, set `N_DEVICES_X` and `N_DEVICES_Y` in the CONFIG block.

## Testing on one machine

Simulated CPU devices exercise the multi-device code paths. The test suite
sets this up in `OceanJAX/tests/conftest.py`:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 python experiment.py
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    python runtime_test/benchmark_parallel.py domain 96 128 96 20
```

Simulated devices share the same CPU cores. They show that the
decomposition is correct, not that it is faster. Measure speed-up on real
GPUs.

## Multi-node GPU cluster (SLURM)

JAX runs one process per GPU (or per node), and all processes execute the
same script. `init_distributed()` (called at the start of
`experiment.py`) reads the SLURM environment that `srun` provides and
connects the processes. After that, `jax.devices()` lists the GPUs of all
nodes. Only process 0 prints and writes NetCDF.

This setup has not been tested on a real cluster yet. The code paths it
relies on are the standard JAX multi-controller APIs:
`jax.distributed.initialize`, `jax.make_array_from_callback` and
`multihost_utils.process_allgather`.

Example job: 2 nodes × 4 GPUs, domain split 4 × 2.

```bash
#!/bin/bash
#SBATCH --job-name=oceanjax
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4          # one process per GPU
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

module load cuda                     # site specific
source .venv/bin/activate            # jax[cuda12] installed here

# experiment.py: N_DEVICES_X = 4, N_DEVICES_Y = 2
srun python experiment.py
```

Notes:

- **Installation.** Install `jax[cuda12]` (or `jax[cuda13]`) in the
  cluster's Python environment. JAX on native Windows has no CUDA support;
  use Linux or WSL2.
- **Explicit initialisation.** If auto-detection fails, pass the settings
  directly:
  `init_distributed(coordinator_address="node0:12345", num_processes=8,
  process_id=int(os.environ["SLURM_PROCID"]))`.
- **Local test.** `init_distributed()` is a no-op when `SLURM_NTASKS` is
  unset or 1, so the same script runs unchanged on a workstation.
- **Mesh layout.** Keep each node's GPUs adjacent along one mesh axis, so
  that most halo traffic stays inside a node. `make_mesh` uses the device
  order from `jax.devices()`, which lists devices process by process.
