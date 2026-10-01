"""
experiment.py
=============
OceanJAX experiment template.  Edit the CONFIG block below, then run:

    python experiment.py

Nothing else needs to change.
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import netCDF4 as nc_lib

# ==============================================================================
# EXPERIMENT CONFIGURATION — only edit this section
# ==============================================================================

# --- Domain & resolution ------------------------------------------------------
LON       = (-40.0, -5.0)    # (lon_min, lon_max) degrees east
LAT       = (-15.0, 15.0)    # (lat_min, lat_max) degrees north
DEPTH_MAX = 500.0             # m (model bottom)
NX, NY, NZ = 20, 15, 10      # grid cells in x, y, z (NZ ignored for "oras5" levels)
DT        = 300.0             # time step [s]; checked against the CFL limit at start-up

# --- Vertical levels ----------------------------------------------------------
#   "uniform"   — NZ equal layers of DEPTH_MAX / NZ
#   "stretched" — NZ layers, DZ_TOP thick at the surface and growing
#                 geometrically so that the bottom is exactly DEPTH_MAX
#   "oras5"     — ORAS5's own levels down to DEPTH_MAX, thinned so that every
#                 layer is at least DZ_TOP thick (number of levels follows)
# Keep the top layer >= ~10 m (linear free surface: |eta| << top layer) and
# DEPTH_MAX within the ORAS5 data (~5900 m).
VERTICAL_LEVELS = "uniform"
DZ_TOP          = 10.0        # m

# --- Horizontal viscosity -----------------------------------------------------
#   "munk" — nu_h from the Munk criterion for this grid (resolves the western
#            boundary layer; scales as dx^3, e.g. ~1.7e5 m2/s at 1.75°)
#   float  — fixed nu_h [m2 s-1]
NU_H = "munk"

# --- Vertical mixing ----------------------------------------------------------
#   "constant" — nu_v = 1e-4, kappa_v = 1e-5 m2/s everywhere
#   "pp81"     — Pacanowski & Philander (1981) Richardson-number mixing for
#                momentum and tracers, with convective adjustment (0.1 m2/s
#                where statically unstable, blended in continuously over
#                -1e-6 < N2 < 0 s-2); constants above as backgrounds.
#                ModelParams default.
#   "kpp"      — KPP surface boundary layer (Large et al. 1994: depth from the
#                bulk Richardson number, deepened by wind and convection,
#                nonlocal tracer flux) over PP81 in the interior.
VERTICAL_MIXING = "pp81"

# --- Initial conditions -------------------------------------------------------
#   "rest"       — uniform T_BG / S_BG, zero velocity
#   "oras5_cold" — T/S from ORAS5, u = v = eta = 0  (recommended: most stable)
#   "oras5_full" — full ORAS5 state (T, S, u, v, eta)
INIT_MODE  = "oras5_cold"
ORAS5_PATH = "OceanJAX/data/data_oras5/oras5_2026_01_native_merged.nc"
ORAS5_TIME_INDEX = 0          # time slice index in the ORAS5 file
T_BG, S_BG = 10.0, 35.0      # background T [°C] and S [psu] for "rest" mode

# --- Integration length -------------------------------------------------------
N_DAYS = 30                   # total simulation length in days

# --- Surface forcing ----------------------------------------------------------
#
# FORCING_DIR: folder with ORAS5 monthly 2-D forcing files
#   (<var>_control_monthly_highres_2D_<YYYYMM>_OPER_v0.1.nc, var in sohefldo,
#   sowaflup, sozotaux, sometauy), relative to this script or absolute.
#   None — use only the constant values below (HEAT_FLUX etc.).
#
#   The month in effect follows the model calendar starting at START_DATE.
#   Every complete month in the folder is used; a missing month falls back to
#   the same calendar month of another year, then to the nearest month (with
#   only January on disk the run is forced by perpetual January).
#
# FORCING_INTERP:
#   "linear"  — monthly means at mid-month, linearly interpolated (smooth)
#   "monthly" — each month's mean held constant, switching on the 1st
#
# FORCING_FIELDS: which fields to take from the files.  Any subset of:
#   {"heat_flux", "fw_flux", "tau_x", "tau_y"}
#
# Constant fallback values (used when FORCING_DIR is None, or for fields
# not listed in FORCING_FIELDS):
#   HEAT_FLUX  [W m-2]  net downward heat flux   (positive = warming ocean)
#   FW_FLUX    [m s-1]  net E-P freshwater flux  (positive = net evaporation)
#   TAU_X      [N m-2]  zonal wind stress        (positive = eastward)
#   TAU_Y      [N m-2]  meridional wind stress   (positive = northward)

FORCING_DIR    = "OceanJAX/data/data_oras5"
START_DATE     = "2026-01-01"      # calendar date of model time 0
FORCING_INTERP = "linear"
FORCING_FIELDS = {"heat_flux", "fw_flux", "tau_x", "tau_y"}
HEAT_FLUX = 0.0
FW_FLUX   = 0.0
TAU_X     = 0.0
TAU_Y     = 0.0

# --- Ensemble / multi-GPU -----------------------------------------------------
#
# N_ENSEMBLE = 1  → single run (default, identical to previous behaviour).
# N_ENSEMBLE > 1  → batch / ensemble run using OceanJAX.parallel.
#
#   Each member starts from the same base initial condition, optionally with
#   independent Gaussian T perturbations (std = ENSEMBLE_PERTURB_T).
#   Members share the same grid, params, and surface forcing.
#
#   Execution:
#     - Single GPU  : vmap over the batch dimension (batch_run).
#     - Multiple GPU: batch dimension sharded across devices via
#                     NamedSharding (sharded_ensemble_run).
#
#   Output: NetCDF gains a "member" dimension;
#           diagnostics print ensemble mean ± spread.
#
# Note: N_ENSEMBLE must be divisible by the number of available GPUs.

N_ENSEMBLE        = 1      # number of ensemble members (1 = single run)
ENSEMBLE_PERTURB_T = 0.0   # Gaussian T perturbation std [°C] for each member

# --- Domain decomposition -----------------------------------------------------
#
# N_DEVICES_X x N_DEVICES_Y > 1 splits the domain over that many devices
# (OceanJAX.parallel.sharding; XLA inserts the halo exchanges).  Requires
# NX % N_DEVICES_X == 0 and NY % N_DEVICES_Y == 0.  1 x 1 = off.
#
# Combined with N_ENSEMBLE > 1, members are also spread over the remaining
# devices: batch axis = largest divisor of N_ENSEMBLE that fits in
# n_devices // (N_DEVICES_X * N_DEVICES_Y).
#
# On a cluster, start one process per GPU (or node) with srun; the SLURM
# environment is picked up automatically (see docs/parallel.md).  Only
# process 0 prints and writes the NetCDF output.

N_DEVICES_X = 1
N_DEVICES_Y = 1

# --- Output -------------------------------------------------------------------
OUTPUT_NC     = "output_cold_full_forcing.nc"
SAVE_INTERVAL = 288   # steps between NetCDF snapshots  (288 × 300 s = 1 day)
CHUNK_SIZE    = 288   # steps per JIT-compiled scan call
#   SAVE_DAILY_3D = False writes only surface fields (SST, SSS, eta) per
#   snapshot, which keeps long high-resolution runs small.
SAVE_DAILY_3D = True
#   Calendar-month means of T, S, u, v, eta (sampled at every snapshot) are
#   written to MONTHLY_NC; None disables them.  Used by
#   verification_experiments/hindcast_eval.py.
MONTHLY_NC    = None

# ==============================================================================
# END OF CONFIGURATION
# ==============================================================================


_SCRIPT_DIR = Path(__file__).parent
_DEPTH_LEVELS = None      # exact float64 centre depths, set by _build_grid
_ORAS5_FILE = _SCRIPT_DIR / ORAS5_PATH


# ---------------------------------------------------------------------------
# Grid & state builders
# ---------------------------------------------------------------------------

def _read_raw():
    """Read the ORAS5 slice once (None for INIT_MODE == "rest")."""
    from OceanJAX.data.oras5 import read_oras5

    if INIT_MODE == "rest":
        return None
    if not _ORAS5_FILE.exists():
        print(f"ERROR: ORAS5 file not found: {_ORAS5_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading ORAS5 from {_ORAS5_FILE} (time_index={ORAS5_TIME_INDEX}) ...")
    t0  = _time.perf_counter()
    raw = read_oras5(_ORAS5_FILE, time_index=ORAS5_TIME_INDEX)
    print(f"  done in {_time.perf_counter() - t0:.1f} s")
    return raw


def _build_grid(raw):
    """
    ORAS5 runs: land mask + bathymetry from ORAS5, closed east/west walls.
    "rest" runs: flat-bottom, zonally periodic ocean.
    """
    from OceanJAX.grid import OceanGrid, face_depths
    from OceanJAX.data.oras5 import oras5_grid

    global NZ, _DEPTH_LEVELS
    depth_levels = _depth_levels(raw)
    _DEPTH_LEVELS = np.asarray(depth_levels, dtype=np.float64)
    NZ    = len(depth_levels)
    dz    = np.diff(face_depths(depth_levels))
    print(f"  levels    : {VERTICAL_LEVELS}, {NZ} layers, top {dz[0]:.1f} m, "
          f"bottom {dz[-1]:.0f} m, model bottom {face_depths(depth_levels)[-1]:.0f} m")
    if raw is None:
        return OceanGrid.create(LON, LAT, depth_levels, NX, NY)
    grid = oras5_grid(raw, LON, LAT, depth_levels, NX, NY, periodic_x=False)
    wet_cols = int(np.asarray(grid.mask_c)[:, :, 0].sum())
    print(f"  ORAS5 land mask: {wet_cols}/{NX * NY} wet columns")
    return grid


def _depth_levels(raw):
    """Cell-centre depths for VERTICAL_LEVELS (see the CONFIG block)."""
    from OceanJAX.grid import stretched_levels
    from OceanJAX.data.oras5 import oras5_levels

    if VERTICAL_LEVELS == "uniform":
        return (np.arange(NZ) + 0.5) * (DEPTH_MAX / NZ)
    if VERTICAL_LEVELS == "stretched":
        return stretched_levels(DEPTH_MAX, NZ, DZ_TOP)
    if VERTICAL_LEVELS == "oras5":
        if raw is not None:
            return oras5_levels(raw, DEPTH_MAX, DZ_TOP)
        import xarray as xr                       # "rest" runs: read the depth axis only
        with xr.open_dataset(_ORAS5_FILE) as ds:
            return oras5_levels(np.asarray(ds["deptht"].values), DEPTH_MAX, DZ_TOP)
    raise ValueError(f"VERTICAL_LEVELS must be 'uniform', 'stretched' or 'oras5'; "
                     f"got {VERTICAL_LEVELS!r}")


def _build_state(grid, raw):
    import warnings
    from OceanJAX.state import create_rest_state, create_from_arrays
    from OceanJAX.data.oras5 import regrid_to_model

    if INIT_MODE == "rest":
        print(f"Init: rest  T={T_BG} °C  S={S_BG} psu")
        return create_rest_state(grid, T_background=T_BG, S_background=S_BG)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        full_state = regrid_to_model(raw, grid)

    if INIT_MODE == "oras5_full":
        state = full_state
        u_a, eta_a = np.array(state.u), np.array(state.eta)
        print(f"Init: oras5_full  "
              f"u=[{u_a.min():.3f}, {u_a.max():.3f}] m/s  "
              f"eta=[{eta_a.min():.3f}, {eta_a.max():.3f}] m")
    else:  # oras5_cold
        zeros3 = jnp.zeros((NX, NY, NZ), dtype=jnp.float32)
        zeros2 = jnp.zeros((NX, NY),     dtype=jnp.float32)
        state  = create_from_arrays(
            grid, u=zeros3, v=zeros3,
            T=full_state.T, S=full_state.S, eta=zeros2,
        )
        print("Init: oras5_cold  u=v=eta=0")

    T_a, S_a = np.array(state.T), np.array(state.S)
    wet = np.asarray(grid.mask_c) > 0
    print(f"  T_wet=[{T_a[wet].min():.2f}, {T_a[wet].max():.2f}] °C  "
          f"S_wet=[{S_a[wet].min():.2f}, {S_a[wet].max():.2f}] psu")
    return state


def _build_ensemble_states(base_state, grid):
    """
    Stack N_ENSEMBLE copies of base_state into a batched OceanState.
    Optionally adds independent Gaussian T perturbations (std = ENSEMBLE_PERTURB_T).

    Returns an OceanState whose array fields carry a leading axis of size N_ENSEMBLE,
    e.g. T.shape == (N_ENSEMBLE, NX, NY, NZ).
    """
    import equinox as eqx

    # Add leading batch axis by stacking N_ENSEMBLE copies
    batched = jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * N_ENSEMBLE), base_state
    )

    if ENSEMBLE_PERTURB_T > 0.0:
        key   = jax.random.PRNGKey(0)
        keys  = jax.random.split(key, N_ENSEMBLE)
        noise = jax.vmap(
            lambda k: jax.random.normal(k, base_state.T.shape, dtype=jnp.float32)
        )(keys) * ENSEMBLE_PERTURB_T                           # (N_ENSEMBLE, NX, NY, NZ)
        T_new = (batched.T + noise) * grid.mask_c[None]
        batched = eqx.tree_at(lambda s: s.T, batched, T_new)

    return batched


# ---------------------------------------------------------------------------
# Forcing builder
# ---------------------------------------------------------------------------

def _make_forcing_provider(grid):
    """
    Return ``provider(t_start, n_steps) -> SurfaceForcing | None`` for one chunk.

    ORAS5 months are read and regridded once here (MonthlyForcing); each call
    then only interpolates in time.  Fields not in FORCING_FIELDS use the
    constants (HEAT_FLUX, FW_FLUX, TAU_X, TAU_Y).  With FORCING_DIR None and
    all constants zero, the provider returns None (no surface forcing).
    """
    from OceanJAX.timeStepping import SurfaceForcing
    from OceanJAX.data.monthly_forcing import MonthlyForcing

    const = {
        "heat_flux": float(HEAT_FLUX),
        "fw_flux":   float(FW_FLUX),
        "tau_x":     float(TAU_X),
        "tau_y":     float(TAU_Y),
    }
    if FORCING_DIR is None and not any(const.values()):
        return lambda t_start, n_steps: None

    monthly = None
    if FORCING_DIR is not None:
        forcing_dir = Path(FORCING_DIR)
        if not forcing_dir.is_absolute():
            forcing_dir = _SCRIPT_DIR / forcing_dir
        monthly = MonthlyForcing(forcing_dir, grid, START_DATE,
                                 interp=FORCING_INTERP, use_fields=FORCING_FIELDS)
        print(f"  forcing months on disk: "
              f"{', '.join(f'{y}-{m:02d}' for y, m in monthly.months)}  "
              f"(interp={FORCING_INTERP}, start={START_DATE})")

    def provider(t_start: float, n_steps: int):
        const_fields = {k: jnp.full((n_steps, NX, NY), v, dtype=jnp.float32)
                        for k, v in const.items()}
        if monthly is None:
            return SurfaceForcing(**const_fields)
        sf = monthly.chunk(t_start, n_steps, DT)
        return SurfaceForcing(**{
            k: getattr(sf, k) if k in FORCING_FIELDS else const_fields[k]
            for k in const
        })

    return provider


def _broadcast_forcing_to_ensemble(forcing, n_members: int):
    """
    Add a leading ensemble axis to a SurfaceForcing chunk.
    Shape: (n_steps, NX, NY) → (n_members, n_steps, NX, NY).
    Returns None if forcing is None.
    """
    if forcing is None:
        return None
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x[None], (n_members,) + x.shape),
        forcing,
    )


# ---------------------------------------------------------------------------
# NetCDF helpers
# ---------------------------------------------------------------------------

def _write_grid_info(ds: nc_lib.Dataset, grid) -> None:
    """
    Store what is needed to rebuild the model grid exactly,
    OceanGrid.create(lon_bounds, lat_bounds, z_exact, Nx, Ny, bathymetry=H,
    periodic_x=periodic_x), plus the resulting wet mask for convenience.
    Assumes the "x", "y", "z" dimensions already exist.
    """
    ds.lon_bounds      = np.array(LON, dtype=np.float64)
    ds.lat_bounds      = np.array(LAT, dtype=np.float64)
    ds.periodic_x      = int(grid.periodic_x)
    ds.vertical_levels = VERTICAL_LEVELS
    ds.depth_max       = DEPTH_MAX
    ds.dz_top          = DZ_TOP
    ds.init_mode       = INIT_MODE
    ds.start_date      = START_DATE
    v = ds.createVariable("H", "f4", ("x", "y")); v.units = "m"; v[:] = np.array(grid.H)
    ds.createVariable("mask_c", "i1", ("x", "y", "z"))[:] = np.array(grid.mask_c).astype(np.int8)
    v = ds.createVariable("z_exact", "f8", ("z",)); v.units = "m"
    v.long_name = "cell-centre depths (float64) for OceanGrid.create"
    v[:] = (np.asarray(grid.z_c, dtype=np.float64) if _DEPTH_LEVELS is None
            else _DEPTH_LEVELS)


def _create_nc(path: str, grid) -> nc_lib.Dataset:
    ds = nc_lib.Dataset(path, mode="w", format="NETCDF4")
    ds.description = f"OceanJAX experiment  init={INIT_MODE}  N_ensemble={N_ENSEMBLE}"
    ds.domain      = f"lon={LON} lat={LAT}"
    ds.dt          = DT
    ds.n_days      = N_DAYS
    ds.heat_flux   = HEAT_FLUX
    ds.fw_flux     = FW_FLUX
    ds.tau_x       = TAU_X
    ds.tau_y       = TAU_Y
    ds.forcing_dir    = str(FORCING_DIR)
    ds.start_date     = START_DATE
    ds.forcing_interp = FORCING_INTERP
    ds.forcing_fields = ",".join(sorted(FORCING_FIELDS))
    ds.n_ensemble  = N_ENSEMBLE

    ds.createDimension("time",   None)
    ds.createDimension("x",      grid.Nx)
    ds.createDimension("y",      grid.Ny)
    ds.createDimension("z",      grid.Nz)
    ds.createDimension("zw",     grid.Nz + 1)

    v = ds.createVariable("time", "f4", ("time",));  v.units = "s"
    v = ds.createVariable("x",    "f4", ("x",));     v.units = "degrees_east";  v[:] = np.array(grid.lon_c)
    v = ds.createVariable("y",    "f4", ("y",));     v.units = "degrees_north"; v[:] = np.array(grid.lat_c)
    v = ds.createVariable("z",    "f4", ("z",));     v.units = "m";             v[:] = np.array(grid.z_c)
    v = ds.createVariable("zw",   "f4", ("zw",));    v.units = "m";             v[:] = np.array(grid.z_w)

    _write_grid_info(ds, grid)
    ds.save_daily_3d = int(SAVE_DAILY_3D)

    lead = ("time", "member") if N_ENSEMBLE > 1 else ("time",)
    if N_ENSEMBLE > 1:
        ds.createDimension("member", N_ENSEMBLE)
    units = {"T": "degC", "S": "psu", "u": "m s-1", "v": "m s-1", "w": "m s-1",
             "eta": "m", "sst": "degC", "sss": "psu"}
    if SAVE_DAILY_3D:
        spec = {"T": "z", "S": "z", "u": "z", "v": "z", "w": "zw", "eta": None}
    else:
        spec = {"sst": None, "sss": None, "eta": None}
    for name, zdim in spec.items():
        dims = lead + ("x", "y") + ((zdim,) if zdim else ())
        ds.createVariable(name, "f4", dims, fill_value=np.float32(np.nan)).units = units[name]
    return ds


def _write_snapshot(ds: nc_lib.Dataset, state) -> None:
    """Write one time record.  Handles both single and ensemble states."""
    i = len(ds.variables["time"])
    # For ensemble state, state.time has shape (N_ENSEMBLE,); use member 0's time.
    t = float(state.time) if state.time.ndim == 0 else float(state.time[0])
    ds.variables["time"][i] = t
    fields = {"eta": np.array(state.eta)}
    if SAVE_DAILY_3D:
        fields.update(T=np.array(state.T), S=np.array(state.S), u=np.array(state.u),
                      v=np.array(state.v), w=np.array(state.w))
    else:
        fields.update(sst=np.array(state.T)[..., 0], sss=np.array(state.S)[..., 0])
    for name, arr in fields.items():
        ds.variables[name][i, ...] = arr
    ds.sync()


class _MonthlyMeans:
    """
    Running calendar-month means of T, S, u, v, eta, written to MONTHLY_NC.

    ``add(state, t_seconds)`` accumulates a host state sampled at model time
    t (seconds since START_DATE) into the calendar month containing t; when
    a sample falls into a new month the previous month is written out.
    ``close()`` writes the last month.  Each record stores the month
    (YYYYMM), the number of samples and the first / last sample time, so a
    partial first or last month is visible.
    """
    FIELDS = ("T", "S", "u", "v", "eta")

    def __init__(self, path: str, grid):
        self.start = np.datetime64(START_DATE, "s")
        self.ds = ds = nc_lib.Dataset(path, mode="w", format="NETCDF4")
        ds.description = f"OceanJAX calendar-month means  init={INIT_MODE}"
        ds.domain = f"lon={LON} lat={LAT}"
        ds.dt = DT
        for name, n in (("time", None), ("x", grid.Nx), ("y", grid.Ny), ("z", grid.Nz)):
            ds.createDimension(name, n)
        ds.createVariable("x", "f4", ("x",))[:] = np.array(grid.lon_c)
        ds.createVariable("y", "f4", ("y",))[:] = np.array(grid.lat_c)
        ds.createVariable("z", "f4", ("z",))[:] = np.array(grid.z_c)
        _write_grid_info(ds, grid)
        lead = ("time",)
        if N_ENSEMBLE > 1:
            ds.createDimension("member", N_ENSEMBLE)
            lead = ("time", "member")
        ds.createVariable("month", "i4", ("time",)).long_name = "YYYYMM"
        ds.createVariable("n_samples", "i4", ("time",))
        for name in ("t_first", "t_last"):
            ds.createVariable(name, "f8", ("time",)).units = f"seconds since {START_DATE}"
        units = {"T": "degC", "S": "psu", "u": "m s-1", "v": "m s-1", "eta": "m"}
        for name in self.FIELDS:
            dims = lead + (("x", "y") if name == "eta" else ("x", "y", "z"))
            ds.createVariable(name, "f4", dims, fill_value=np.float32(np.nan)).units = units[name]
        self.month, self.sums, self.n, self.t0, self.t1 = None, None, 0, None, None

    def add(self, state, t_seconds: float) -> None:
        month = (self.start + np.timedelta64(int(round(t_seconds)), "s")).astype("datetime64[M]")
        if self.month is not None and month != self.month:
            self._flush()
        if self.month is None:
            self.month, self.n, self.t0 = month, 0, t_seconds
            self.sums = {f: np.zeros(np.shape(getattr(state, f)), np.float64) for f in self.FIELDS}
        for f in self.FIELDS:
            self.sums[f] += np.asarray(getattr(state, f), np.float64)
        self.n += 1
        self.t1 = t_seconds

    def _flush(self) -> None:
        ds, i = self.ds, len(self.ds.variables["month"])
        m = int(self.month.astype(np.int64))
        ds.variables["month"][i] = (1970 + m // 12) * 100 + m % 12 + 1
        ds.variables["n_samples"][i] = self.n
        ds.variables["t_first"][i] = self.t0
        ds.variables["t_last"][i] = self.t1
        for f in self.FIELDS:
            ds.variables[f][i, ...] = (self.sums[f] / self.n).astype(np.float32)
        ds.sync()
        self.month = None

    def close(self) -> None:
        if self.month is not None:
            self._flush()
        self.ds.close()


def _diag_line(state, sim_day: float, steps_done: int, wall: float, grid) -> bool:
    """
    Print one diagnostic line.  Handles both single and ensemble states.
    Ranges are taken over wet cells only (land cells hold zeros).
    Returns True if any non-finite value is detected.
    """
    wet  = np.asarray(grid.mask_c) > 0         # (NX, NY, NZ)
    wet2 = wet[:, :, 0]                        # (NX, NY)
    if N_ENSEMBLE > 1:
        # Ensemble: report mean ± std across members
        T_all   = np.array(state.T)    # (B, NX, NY, NZ)
        S_all   = np.array(state.S)
        eta_all = np.array(state.eta)
        bad = (not np.all(np.isfinite(T_all)) or
               not np.all(np.isfinite(S_all)) or
               not np.all(np.isfinite(eta_all)))
        T_mean   = T_all.mean(axis=0)[wet];    T_std   = T_all.std(axis=0)[wet]
        eta_mean = eta_all.mean(axis=0)[wet2]; eta_std = eta_all.std(axis=0)[wet2]
        print(f"{sim_day:5.1f}  {steps_done:6d}  "
              f"T_mean=[{T_mean.min():.3f},{T_mean.max():.3f}] "
              f"±{T_std.max():.4f}  "
              f"eta_mean=[{eta_mean.min():.4f},{eta_mean.max():.4f}] "
              f"±{eta_std.max():.5f}  "
              f"{'NON-FINITE!' if bad else 'ok':>10}  {wall:5.1f}s",
              flush=True)
    else:
        T_a   = np.array(state.T)
        S_a   = np.array(state.S)
        eta_a = np.array(state.eta)
        bad   = (not np.all(np.isfinite(T_a)) or
                 not np.all(np.isfinite(S_a)) or
                 not np.all(np.isfinite(eta_a)))
        print(f"{sim_day:5.1f}  {steps_done:6d}  "
              f"{T_a[wet].min():7.3f} {T_a[wet].max():7.3f}  "
              f"{S_a[wet].min():6.3f} {S_a[wet].max():6.3f}  "
              f"{eta_a[wet2].min():8.4f} {eta_a[wet2].max():8.4f}  "
              f"{'NON-FINITE!' if bad else 'ok':>10}  {wall:5.1f}s",
              flush=True)
    return bad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _make_domain_mesh(n_devices: int):
    """("batch", "x", "y") mesh for N_DEVICES_X x N_DEVICES_Y (+ ensemble)."""
    from OceanJAX.parallel.sharding import make_mesh

    n_domain = N_DEVICES_X * N_DEVICES_Y
    if n_domain > n_devices:
        print(f"ERROR: N_DEVICES_X * N_DEVICES_Y = {n_domain} but only "
              f"{n_devices} devices available.", file=sys.stderr)
        sys.exit(1)
    n_batch = 1
    if N_ENSEMBLE > 1:
        n_batch = max(d for d in range(1, n_devices // n_domain + 1)
                      if N_ENSEMBLE % d == 0)
    return make_mesh(N_DEVICES_X, N_DEVICES_Y, n_batch=n_batch)


def main() -> None:
    import os
    from OceanJAX.state import ModelParams
    from OceanJAX.timeStepping import run as ocean_run
    from OceanJAX.parallel.sharding import init_distributed, gather_to_host

    # Multi-process (SLURM) setup; a no-op for an ordinary single process.
    init_distributed()
    is_main = jax.process_index() == 0
    if not is_main:
        sys.stdout = open(os.devnull, "w")

    n_steps    = round(N_DAYS * 86400 / DT)
    ensemble   = N_ENSEMBLE > 1
    n_devices  = len(jax.devices())
    decomposed = N_DEVICES_X * N_DEVICES_Y > 1

    print("=" * 62)
    print(f"OceanJAX experiment")
    print(f"  domain    : lon={LON}  lat={LAT}  depth={DEPTH_MAX} m")
    print(f"  grid      : {NX}x{NY} horizontal, {VERTICAL_LEVELS} levels  dt={DT} s")
    print(f"  run       : {N_DAYS} days  ({n_steps} steps)")
    if ensemble:
        print(f"  ensemble  : {N_ENSEMBLE} members  "
              f"perturb_T={ENSEMBLE_PERTURB_T} °C  "
              f"devices={n_devices}")
    if decomposed:
        print(f"  domain    : split over {N_DEVICES_X} x {N_DEVICES_Y} devices "
              f"({jax.process_count()} process(es), {n_devices} devices)")
    print(f"  output    : {OUTPUT_NC}  save_every={SAVE_INTERVAL} steps")
    print("=" * 62)

    from OceanJAX.Physics.mixing import munk_viscosity

    raw    = _read_raw()
    grid   = _build_grid(raw)
    nu_h   = munk_viscosity(grid) if NU_H == "munk" else float(NU_H)
    params = ModelParams(dt=DT, nu_h=nu_h, vertical_mixing=VERTICAL_MIXING)
    print(f"  nu_h = {nu_h:.3g} m2/s  ({'Munk criterion' if NU_H == 'munk' else 'fixed'})"
          f"  vertical mixing: {VERTICAL_MIXING}")

    # Barotropic CFL (surface gravity waves); exp. 5.1.4: stable <= 0.33, NaN at 0.45
    from OceanJAX.grid import barotropic_cfl, CFL_LIMIT
    cfl, dt_max = barotropic_cfl(grid, DT)
    print(f"  CFL       : {cfl:.3f} (limit {CFL_LIMIT})  largest safe dt ~ {dt_max:.0f} s")
    if cfl > CFL_LIMIT:
        print(f"WARNING: DT = {DT} s exceeds the barotropic CFL limit; "
              f"use DT <= {dt_max:.0f} s or the run may blow up.", file=sys.stderr)
    forcing_for = _make_forcing_provider(grid)

    # Build initial state(s)
    base_state = _build_state(grid, raw)
    if ensemble:
        state = _build_ensemble_states(base_state, grid)
        print(f"  Ensemble of {N_ENSEMBLE} members created.")
    else:
        state = base_state

    # Compile run function
    if decomposed:
        from OceanJAX.parallel.sharding import (
            shard_grid, shard_state, sharded_run)
        mesh   = _make_domain_mesh(n_devices)
        grid_s = shard_grid(grid, mesh)          # placed once, reused by every chunk
        state  = shard_state(state, mesh)
        print(f"  mesh      : batch={mesh.shape['batch']} x={mesh.shape['x']} "
              f"y={mesh.shape['y']}  local block "
              f"{NX // N_DEVICES_X}x{NY // N_DEVICES_Y}x{NZ}")

        def _run_chunk(s, chunk, forcing):
            # One forcing sequence (chunk, NX, NY) is shared by all members.
            return sharded_run(s, grid_s, params, chunk, mesh,
                               forcing_sequence=forcing, save_history=False)
    elif ensemble:
        from OceanJAX.parallel.ensemble import sharded_ensemble_run
        # sharded_ensemble_run handles jit internally
        def _run_chunk(s, chunk, forcing):
            f_batch = _broadcast_forcing_to_ensemble(forcing, N_ENSEMBLE)
            return sharded_ensemble_run(s, grid, params, chunk,
                                        forcing_sequence=f_batch,
                                        save_history=False)
    else:
        run_jit = jax.jit(ocean_run, static_argnames=("n_steps", "save_history"))
        def _run_chunk(s, chunk, forcing):
            return run_jit(s, grid, params, n_steps=chunk,
                           forcing_sequence=forcing, save_history=False)

    # Host copy of the (possibly sharded) state for output and diagnostics
    to_host = gather_to_host if decomposed else (lambda s: s)

    # Open output file and save t=0 (process 0 only)
    ds = None
    if is_main:
        ds = _create_nc(OUTPUT_NC, grid)
        ds.nu_h = params.nu_h
        ds.vertical_mixing = params.vertical_mixing
        ds.n_devices_x = N_DEVICES_X
        ds.n_devices_y = N_DEVICES_Y
    host_state = to_host(state)
    if is_main:
        _write_snapshot(ds, host_state)
    print(f"\nOutput: {OUTPUT_NC}  (t=0 saved)")
    monthly = None
    if is_main and MONTHLY_NC is not None:
        monthly = _MonthlyMeans(MONTHLY_NC, grid)
        print(f"Monthly means: {MONTHLY_NC}")
    print()

    if ensemble:
        print(f"{'Day':>5}  {'Step':>6}  "
              f"{'T_mean range':^25}  {'eta_mean range':^22}  "
              f"{'status':>10}  {'wall':>6}")
    else:
        print(f"{'Day':>5}  {'Step':>6}  {'T_min':>7} {'T_max':>7}  "
              f"{'S_min':>6} {'S_max':>6}  {'eta_min':>8} {'eta_max':>8}  "
              f"{'status':>10}  {'wall':>6}")
    print("-" * 90)

    steps_done     = 0
    next_save_step = SAVE_INTERVAL
    all_ok         = True

    try:
        while steps_done < n_steps:
            steps_remaining  = n_steps - steps_done
            steps_until_save = next_save_step - steps_done
            chunk = min(CHUNK_SIZE, steps_remaining, steps_until_save)

            # Model time from the step counter (exact; state.time is float32)
            forcing = forcing_for(steps_done * DT, chunk)

            t0 = _time.perf_counter()
            state, _ = _run_chunk(state, chunk, forcing)
            jax.block_until_ready(state.T)
            wall = _time.perf_counter() - t0

            steps_done += chunk
            host_state = to_host(state)
            # For ensemble, each member has its own time scalar; use member 0.
            sim_day = float(host_state.time) / 86400.0 if host_state.time.ndim == 0 \
                      else float(host_state.time[0])   / 86400.0

            bad = _diag_line(host_state, sim_day, steps_done, wall, grid)

            if steps_done == next_save_step:
                if is_main:
                    _write_snapshot(ds, host_state)
                    if monthly is not None:
                        monthly.add(host_state, steps_done * DT)
                next_save_step += SAVE_INTERVAL

            if bad:
                print("\nABORTED: non-finite values detected.", file=sys.stderr)
                all_ok = False
                break

    finally:
        if ds is not None:
            ds.close()
        if monthly is not None:
            monthly.close()

    print("-" * 90)
    label = f"stable for {N_DAYS} days" if all_ok else "model blew up"
    if ensemble:
        label += f"  [{N_ENSEMBLE} members]"
    print(f"\n{'PASS' if all_ok else 'FAIL'} — {label}")
    print(f"Output: {OUTPUT_NC}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
