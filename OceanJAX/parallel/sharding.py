"""
OceanJAX Parallel – Domain Decomposition (GSPMD)
==================================================
Split one ocean domain (and optionally an ensemble) across devices by
sharding the horizontal x / y axes of every field over a device mesh and
letting XLA's GSPMD partitioner insert the communication.

Public interfaces
-----------------
make_mesh()          — build a ("batch", "x", "y") device mesh.
field_spec()         — PartitionSpec for one array, inferred from its shape.
shard_grid()         — place an OceanGrid on the mesh.
shard_state()        — place an OceanState (single or ensemble) on the mesh.
shard_forcing()      — place a SurfaceForcing (single step or sequence).
sharded_run()        — jit-compiled run() / batch_run() on sharded inputs.
gather_to_host()     — bring a sharded pytree back to host numpy arrays.
init_distributed()   — multi-process (e.g. SLURM) initialisation.

How it works
------------
Every array whose axes include the horizontal (Nx, Ny) pair is sharded
along x and y; everything else (1-D coordinates, dz, scalars, params) is
replicated.  The model code itself is unchanged: stencils written with
``jnp.roll`` / shifted concatenates become halo exchanges
(collective-permutes) between neighbouring devices, and column operations
(hydrostatic pressure, w, the implicit vertical solvers) stay local
because z is never sharded.  The implicit solvers map over columns with
nested vmaps, not a reshape to (Nx*Ny, Nz), which would merge the two
sharded axes and force an all-gather of the whole field.

On a 1 x 1 mesh the result is bit-identical to ``jax.jit(run)``.  On a
larger mesh XLA fuses the partitioned stencils differently, which changes
the rounding of individual operations by about one ulp (the same kind of
difference as CPU vs GPU); a field that is exactly uniform stays exactly
uniform, so the rest-state conservation properties are unchanged.

Mesh layout
-----------
Axes ("batch", "x", "y") with sizes (n_batch, n_x, n_y):
  * single domain, domain decomposition : n_batch = 1
  * ensemble only                        : n_x = n_y = 1
  * both                                 : ensemble members on "batch",
                                           each member split over x × y.
Nx must be divisible by n_x, Ny by n_y, and the ensemble size by n_batch.

Multi-process (cluster) use
---------------------------
Call ``init_distributed()`` once, before any other JAX call, in every
process (one process per node or per GPU, as launched by ``srun``).
All processes then run the same script; ``shard_*`` builds global arrays
from the host copies each process holds, and ``gather_to_host`` returns
the full field on every process (write output from process 0 only).
See ``docs/parallel.md`` for a SLURM example.

Phase B (shard_map with explicit halo exchange) is only planned if the
automatic partitioning turns out to be too slow on a real cluster.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from OceanJAX.grid import OceanGrid
from OceanJAX.state import OceanState, ModelParams
from OceanJAX.timeStepping import SurfaceForcing, run
from OceanJAX.ml.closure import AbstractClosure

MESH_AXES = ("batch", "x", "y")


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def make_mesh(
    n_x:     int = 1,
    n_y:     int = 1,
    n_batch: int = 1,
    devices: Optional[Sequence] = None,
) -> Mesh:
    """
    Build a device mesh with axes ("batch", "x", "y").

    Args:
        n_x, n_y : number of devices along the zonal / meridional axis.
        n_batch  : number of devices along the ensemble axis.
        devices  : devices to use (default ``jax.devices()``, i.e. all
                   devices of all processes).  The first
                   n_batch * n_x * n_y are used.

    Returns:
        jax.sharding.Mesh of shape (n_batch, n_x, n_y).
    """
    devices = list(jax.devices() if devices is None else devices)
    n_needed = n_batch * n_x * n_y
    if min(n_batch, n_x, n_y) < 1:
        raise ValueError(f"mesh sizes must be >= 1; got batch={n_batch}, "
                         f"x={n_x}, y={n_y}")
    if n_needed > len(devices):
        raise ValueError(
            f"mesh batch={n_batch} x={n_x} y={n_y} needs {n_needed} devices; "
            f"only {len(devices)} available."
        )
    dev = np.array(devices[:n_needed], dtype=object).reshape(n_batch, n_x, n_y)
    return Mesh(dev, MESH_AXES)


def _mesh_sizes(mesh: Mesh) -> tuple[int, int, int]:
    return mesh.shape["batch"], mesh.shape["x"], mesh.shape["y"]


# ---------------------------------------------------------------------------
# Partition specs
# ---------------------------------------------------------------------------

def field_spec(shape: tuple, Nx: int, Ny: int, n_lead: int = 0,
               batch: bool = False) -> P:
    """
    PartitionSpec for one array, inferred from its shape.

    An array is a horizontal field if its axes ``n_lead`` and ``n_lead+1``
    are (Nx, Ny); those axes are sharded over "x" and "y" and the leading
    ``n_lead`` (time) axes are replicated.  With ``batch=True`` the first
    axis of every non-scalar array is the ensemble axis and is sharded
    over "batch" (per-member scalars such as ``time`` become (B,) vectors).
    Anything else (coordinates, dz, scalars) is replicated.

    Args:
        shape  : array shape
        Nx, Ny : horizontal grid size
        n_lead : number of leading non-spatial axes (0 for a grid or a
                 single state; 1 for an ensemble state or a forcing
                 sequence; 2 for an ensemble forcing sequence)
        batch  : True if the first axis is the ensemble axis
    """
    if len(shape) >= n_lead + 2 and tuple(shape[n_lead:n_lead + 2]) == (Nx, Ny):
        spec = [None] * n_lead + ["x", "y"]
    else:
        spec = []
    if batch and len(shape) >= 1:
        spec = ["batch"] + spec[1:]
    return P(*spec)


def _put(x, sharding: NamedSharding):
    """
    Place one array according to ``sharding``.

    Inside a trace this is a sharding constraint.  In a single process it
    is ``jax.device_put``.  With several processes each one holds the full
    host array, and the global array is assembled from the shards each
    process owns.
    """
    if isinstance(x, jax.core.Tracer):
        return jax.lax.with_sharding_constraint(x, sharding)
    if jax.process_count() == 1:
        return jax.device_put(x, sharding)
    host = np.asarray(x)
    return jax.make_array_from_callback(host.shape, sharding, lambda idx: host[idx])


def _shard_tree(tree, mesh: Mesh, Nx: int, Ny: int, n_lead: int,
                batch_axis: bool):
    """Shard every array leaf of an equinox pytree; static fields untouched."""
    arrays, static = eqx.partition(tree, eqx.is_array)
    placed = jax.tree_util.tree_map(
        lambda x: _put(x, NamedSharding(
            mesh, field_spec(x.shape, Nx, Ny, n_lead, batch_axis))),
        arrays,
    )
    return eqx.combine(placed, static)


def _check_divisible(mesh: Mesh, Nx: int, Ny: int, batch: Optional[int] = None):
    """Raise if the grid (and ensemble size, if given) does not split evenly."""
    n_batch, n_x, n_y = _mesh_sizes(mesh)
    if Nx % n_x or Ny % n_y:
        raise ValueError(
            f"Grid {Nx}x{Ny} is not divisible by the device mesh "
            f"x={n_x}, y={n_y}.  Choose N_DEVICES_X | Nx and N_DEVICES_Y | Ny."
        )
    if batch is not None and batch % n_batch:
        raise ValueError(
            f"Ensemble size ({batch}) must be divisible by the mesh batch "
            f"axis ({n_batch})."
        )


def _is_batched(state: OceanState) -> bool:
    return state.T.ndim == 4


# ---------------------------------------------------------------------------
# Placement of grid / state / forcing
# ---------------------------------------------------------------------------

def shard_grid(grid: OceanGrid, mesh: Mesh) -> OceanGrid:
    """
    Place an OceanGrid on the mesh: (Nx, Ny[, ...]) arrays (metrics, f,
    masks, H) are split over x / y; 1-D coordinate and dz arrays are
    replicated.  The grid is replicated over the "batch" axis.

    Compute anything that needs the whole grid on the host (e.g.
    ``munk_viscosity``) before sharding, or use ``gather_to_host``.
    """
    _check_divisible(mesh, grid.Nx, grid.Ny)
    return _shard_tree(grid, mesh, grid.Nx, grid.Ny, n_lead=0, batch_axis=False)


def shard_state(state: OceanState, mesh: Mesh) -> OceanState:
    """
    Place an OceanState on the mesh.

    A single state, T.shape == (Nx, Ny, Nz), is split over x / y and must
    be used with a mesh whose batch axis is 1.  An ensemble state,
    T.shape == (B, Nx, Ny, Nz), is additionally split over "batch".
    """
    if _is_batched(state):
        B, Nx, Ny = state.T.shape[:3]
        _check_divisible(mesh, Nx, Ny, batch=B)
        return _shard_tree(state, mesh, Nx, Ny, n_lead=1, batch_axis=True)
    Nx, Ny = state.T.shape[:2]
    _check_divisible(mesh, Nx, Ny)
    if _mesh_sizes(mesh)[0] != 1:
        raise ValueError(
            f"Mesh has a batch axis of {_mesh_sizes(mesh)[0]} devices but the "
            f"state has no ensemble axis; use make_mesh(n_batch=1, ...)."
        )
    return _shard_tree(state, mesh, Nx, Ny, n_lead=0, batch_axis=False)


def shard_forcing(
    forcing: Optional[SurfaceForcing],
    mesh:    Mesh,
    grid:    OceanGrid,
    batched: bool = False,
) -> Optional[SurfaceForcing]:
    """
    Place a SurfaceForcing on the mesh.

    Accepted field shapes:
      (Nx, Ny)              one step
      (T, Nx, Ny)           a forcing sequence for run()
      (B, T, Nx, Ny)        a per-member sequence (``batched=True``)

    x / y are sharded; the time axis is replicated (lax.scan slices it on
    every device); the ensemble axis, if any, is sharded over "batch".
    """
    if forcing is None:
        return None
    ndim   = forcing.heat_flux.ndim
    n_lead = ndim - 2
    if batched and n_lead < 1:
        raise ValueError("batched=True needs a leading ensemble axis on forcing")
    _check_divisible(mesh, grid.Nx, grid.Ny,
                     batch=forcing.heat_flux.shape[0] if batched else None)
    return _shard_tree(forcing, mesh, grid.Nx, grid.Ny, n_lead=n_lead,
                       batch_axis=batched)


def state_shardings(state: OceanState, mesh: Mesh) -> OceanState:
    """Pytree of NamedShardings matching ``shard_state(state, mesh)``."""
    batched = _is_batched(state)
    Nx, Ny  = state.T.shape[1:3] if batched else state.T.shape[:2]
    n_lead  = 1 if batched else 0
    return jax.tree_util.tree_map(
        lambda x: NamedSharding(mesh, field_spec(x.shape, Nx, Ny, n_lead, batched)),
        state,
    )


# ---------------------------------------------------------------------------
# Sharded run
# ---------------------------------------------------------------------------

def sharded_run(
    state:            OceanState,
    grid:             OceanGrid,
    params:           ModelParams,
    n_steps:          int,
    mesh:             Mesh,
    forcing_sequence: Optional[SurfaceForcing] = None,
    save_history:     bool = False,
    closure:          Optional[AbstractClosure] = None,
) -> tuple[OceanState, Optional[OceanState]]:
    """
    ``run()`` (single state) or ``batch_run()`` (ensemble state) with the
    domain split over ``mesh``.

    Inputs may be host arrays or already sharded (``shard_state`` etc.):
    they are placed on the mesh here if needed, which is a no-op when the
    placement already matches.  For long integrations in chunks, shard the
    grid once and pass the returned state back in; it keeps its sharding.

    Args:
        state            : OceanState, (Nx, Ny, Nz) or (B, Nx, Ny, Nz).
        grid, params     : as for run().  params is replicated.
        n_steps          : number of steps (static).
        mesh             : device mesh from ``make_mesh``.
        forcing_sequence : SurfaceForcing with fields (n_steps, Nx, Ny) —
                           shared by all members — or (B, n_steps, Nx, Ny)
                           for an ensemble; or None.
        save_history     : as for run().
        closure          : as for run(); replicated on every device.

    Returns:
        (final_state, history) with final_state sharded like the input.
    """
    batched = _is_batched(state)
    grid_s  = shard_grid(grid, mesh)
    state_s = shard_state(state, mesh)

    f_batched = (forcing_sequence is not None and batched
                 and forcing_sequence.heat_flux.ndim == 4)
    forcing_s = shard_forcing(forcing_sequence, mesh, grid, batched=f_batched)
    closure_s = _replicate(closure, mesh)

    return _sharded_run_jit(state_s, grid_s, _traced_params(params), forcing_s,
                            closure_s, n_steps, save_history, mesh)


def _traced_params(params: ModelParams) -> ModelParams:
    """
    Turn the Python-number fields of ``params`` into 0-d arrays so that
    ``eqx.filter_jit`` traces them (as ``jax.jit(run)`` does) instead of
    baking them in as constants.  Constant folding changes the rounding
    of the compiled arithmetic, and traced params can be differentiated.
    """
    def to_array(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, float):
            return jnp.asarray(x, dtype=jnp.float32)
        if isinstance(x, int):
            return jnp.asarray(x, dtype=jnp.int32)
        return x

    return jax.tree_util.tree_map(to_array, params)


def _replicate(tree, mesh: Mesh):
    """Replicate every array leaf of ``tree`` (e.g. closure weights)."""
    if tree is None:
        return None
    arrays, static = eqx.partition(tree, eqx.is_array)
    arrays = jax.tree_util.tree_map(
        lambda x: _put(x, NamedSharding(mesh, P())), arrays)
    return eqx.combine(arrays, static)


@eqx.filter_jit
def _sharded_run_jit(state, grid, params, forcing, closure,
                     n_steps, save_history, mesh):
    """
    Compiled once per (shapes, n_steps, save_history, mesh, static params);
    later chunks of a long run reuse the executable.  The final state is
    constrained to the input layout so it can be fed straight back in.
    """
    from OceanJAX.parallel.ensemble import batch_run

    batched = _is_batched(state)
    if batched and forcing is not None and forcing.heat_flux.ndim == 3:
        # One forcing sequence shared by every member.
        final, hist = eqx.filter_vmap(
            lambda s: run(s, grid, params, n_steps, forcing, save_history, closure)
        )(state)
    elif batched:
        final, hist = batch_run(state, grid, params, n_steps, forcing,
                                save_history, closure)
    else:
        final, hist = run(state, grid, params, n_steps, forcing,
                          save_history, closure)
    final = jax.lax.with_sharding_constraint(final, state_shardings(final, mesh))
    return final, hist


# ---------------------------------------------------------------------------
# Host transfer
# ---------------------------------------------------------------------------

def gather_to_host(tree):
    """
    Return a copy of ``tree`` with every array as a full host numpy array.

    Works in a single process (plain ``jax.device_get``) and with several
    processes, where each process receives the full field.
    """
    if jax.process_count() == 1:
        return jax.device_get(tree)
    from jax.experimental import multihost_utils
    arrays, static = eqx.partition(tree, eqx.is_array)
    arrays = multihost_utils.process_allgather(arrays, tiled=True)
    return eqx.combine(arrays, static)


# ---------------------------------------------------------------------------
# Multi-process initialisation
# ---------------------------------------------------------------------------

def init_distributed(
    coordinator_address: Optional[str] = None,
    num_processes:       Optional[int] = None,
    process_id:          Optional[int] = None,
    local_device_ids:    Optional[Sequence[int]] = None,
) -> bool:
    """
    Initialise multi-process JAX (one process per node or per GPU).

    Must be called once in every process before any other JAX call that
    touches devices.  With no arguments, the cluster is detected from the
    environment (SLURM: ``srun`` sets SLURM_NTASKS, SLURM_PROCID, ...;
    JAX's own auto-detection also covers Open MPI and cloud TPU).  In a
    plain single-process run (no cluster environment, no arguments) this
    is a no-op, so scripts can call it unconditionally.

    Returns:
        True if multi-process JAX is (now) initialised, False for a
        single-process run.
    """
    is_init = getattr(jax.distributed, "is_initialized", None)
    if is_init is not None and is_init():
        return True

    explicit = any(a is not None for a in
                   (coordinator_address, num_processes, process_id))
    n_tasks  = int(os.environ.get("SLURM_NTASKS", "1"))
    if not explicit and n_tasks <= 1:
        return False

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=local_device_ids,
    )
    return True
