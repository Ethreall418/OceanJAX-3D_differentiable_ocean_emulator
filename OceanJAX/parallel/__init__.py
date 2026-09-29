"""
OceanJAX Parallel Package
=========================
Batch / multi-GPU execution utilities.

Ensemble parallelism (ensemble.py):
    Distribute independent model instances across devices via vmap +
    NamedSharding.  No inter-device communication; grid and params are
    replicated.

Domain decomposition, phase A (sharding.py):
    Split one domain over a ("batch", "x", "y") device mesh by sharding the
    horizontal axes of every field; XLA's GSPMD partitioner inserts the
    halo exchanges.  Ensembles can be combined with it on the "batch" axis.

Phase B (future, only if phase A is too slow on a cluster):
    shard_map + explicit halo exchange (halo.py).
"""

from OceanJAX.parallel.ensemble import batch_step, batch_run, sharded_ensemble_run
from OceanJAX.parallel.sharding import (
    make_mesh,
    shard_grid,
    shard_state,
    shard_forcing,
    sharded_run,
    gather_to_host,
    init_distributed,
)

__all__ = [
    "batch_step", "batch_run", "sharded_ensemble_run",
    "make_mesh", "shard_grid", "shard_state", "shard_forcing",
    "sharded_run", "gather_to_host", "init_distributed",
]
