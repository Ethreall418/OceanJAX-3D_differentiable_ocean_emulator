"""
OceanJAX data — Monthly ORAS5 surface forcing with calendar-aware switching
===========================================================================
Drives long integrations with a sequence of monthly-mean ORAS5 surface
fluxes.  Each available month is read and regridded once and cached; the
forcing for any stretch of model time is then assembled on demand, one
chunk at a time, so memory stays at a single chunk (a one-year sequence for
a 50x40 grid at dt = 300 s would otherwise need ~3.4 GB).

File layout
-----------
ORAS5 monthly 2-D forcing files are named

    <var>_control_monthly_highres_2D_<YYYYMM>_OPER_v0.1.nc

with var in {sohefldo, sowaflup, sozotaux, sometauy}.  A month is usable
when all four files are present.  Their ``time_counter`` is the mid-month
instant (e.g. 2026-01-16T12:00), which is exactly the calendar midpoint
used as the interpolation node below.

Time interpolation
------------------
``interp="linear"`` (default)
    Monthly means are placed at mid-month and interpolated linearly in
    between, so the forcing varies smoothly and equals the monthly mean at
    mid-month.  Switching abruptly on the 1st would inject flux jumps of
    O(50-100 W m-2) and spurious adjustment.
``interp="monthly"``
    Piecewise constant: the value of the calendar month containing t.

Missing months
--------------
A node month without data is substituted by, in order:
  1. the same calendar month of the nearest available year (climatology);
  2. the nearest available month in time.
With a single month on disk this reduces to perpetual forcing from that
month.  Each substitution is reported once via ``warnings.warn``.

Typical use (see experiment.py)
-------------------------------
    mf = MonthlyForcing("OceanJAX/data/data_oras5", grid, "2026-01-01")
    for each chunk:
        forcing = mf.chunk(t_start=steps_done * dt, n_steps=chunk, dt=dt)
        state, _ = run(state, grid, params, chunk, forcing_sequence=forcing)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

from OceanJAX.grid import OceanGrid

#: model forcing field -> ORAS5 variable (file-name prefix)
ORAS5_FORCING_VARS: dict[str, str] = {
    "heat_flux": "sohefldo",
    "fw_flux":   "sowaflup",
    "tau_x":     "sozotaux",
    "tau_y":     "sometauy",
}
_FIELDS  = tuple(ORAS5_FORCING_VARS)
_MONTH_RE = re.compile(r"_(\d{4})(\d{2})_")


# ---------------------------------------------------------------------------
# Calendar helpers (numpy datetime64, proleptic Gregorian)
# ---------------------------------------------------------------------------

def _month_mid(months: np.ndarray) -> np.ndarray:
    """Mid-point instant [datetime64[s]] of each calendar month (datetime64[M])."""
    start = months.astype("datetime64[s]")
    end   = (months + 1).astype("datetime64[s]")
    return start + (end - start) // 2


def _ym(month: np.datetime64) -> tuple[int, int]:
    n = int(month.astype("datetime64[M]").astype(np.int64))
    return 1970 + n // 12, n % 12 + 1


def _month64(year: int, month: int) -> np.datetime64:
    return np.datetime64(f"{year:04d}-{month:02d}", "M")


# ---------------------------------------------------------------------------
# Public: discover_oras5_forcing
# ---------------------------------------------------------------------------

def discover_oras5_forcing(directory: str | Path) -> dict[tuple[int, int], dict[str, Path]]:
    """
    Group the ORAS5 2-D forcing files in ``directory`` by month.

    Returns ``{(year, month): {field: path}}`` for months that have all four
    forcing variables.  Incomplete months are skipped with a warning.
    """
    prefix_to_field = {v: k for k, v in ORAS5_FORCING_VARS.items()}
    found: dict[tuple[int, int], dict[str, Path]] = {}
    for p in sorted(Path(directory).glob("*.nc")):
        field = prefix_to_field.get(p.name.split("_")[0])
        m = _MONTH_RE.search(p.name)
        if field is None or m is None:
            continue
        found.setdefault((int(m.group(1)), int(m.group(2))), {})[field] = p

    complete = {}
    for ym, files in sorted(found.items()):
        missing = [ORAS5_FORCING_VARS[f] for f in _FIELDS if f not in files]
        if missing:
            warnings.warn(f"ORAS5 forcing {ym[0]}-{ym[1]:02d} skipped: missing {missing}",
                          UserWarning, stacklevel=2)
        else:
            complete[ym] = files
    return complete


# ---------------------------------------------------------------------------
# Public: MonthlyForcing
# ---------------------------------------------------------------------------

class MonthlyForcing:
    """
    Calendar-aware monthly surface forcing on an OceanJAX grid.

    Parameters
    ----------
    directory  : folder with ORAS5 monthly 2-D forcing files.
    grid       : target OceanGrid.
    start_date : calendar date of model time 0, e.g. "2026-01-01".
    interp     : "linear" (mid-month nodes, default) or "monthly" (step).
    use_fields : forcing fields to take from the files; the others are 0.
    """

    def __init__(
        self,
        directory:  str | Path,
        grid:       OceanGrid,
        start_date: str,
        interp:     str = "linear",
        use_fields=frozenset(_FIELDS),
    ):
        from OceanJAX.data.oras5 import read_oras5_forcing, regrid_forcing

        if interp not in ("linear", "monthly"):
            raise ValueError(f"interp must be 'linear' or 'monthly'; got {interp!r}")
        files = discover_oras5_forcing(directory)
        if not files:
            raise FileNotFoundError(f"no complete ORAS5 forcing month in {directory}")

        self.interp = interp
        self.start  = np.datetime64(start_date, "s")
        self.months = sorted(files)                               # [(y, m), ...]
        self._index = {ym: i for i, ym in enumerate(self.months)}
        self._mids  = _month_mid(np.array([_month64(*ym) for ym in self.months]))
        self._reported: set[tuple[tuple[int, int], tuple[int, int]]] = set()

        stacks = {f: [] for f in _FIELDS}
        for ym in self.months:
            raw = read_oras5_forcing([files[ym][f] for f in _FIELDS])
            sf  = regrid_forcing(raw, grid, use_fields=use_fields)
            for f in _FIELDS:
                stacks[f].append(np.asarray(getattr(sf, f), dtype=np.float32))
        self._data = {f: np.stack(v) for f, v in stacks.items()}   # (n_months, Nx, Ny)

    # ------------------------------------------------------------------
    def _source(self, ym: tuple[int, int]) -> int:
        """Index of the cached month used for calendar month ``ym``."""
        if ym in self._index:
            return self._index[ym]
        same = [m for m in self.months if m[1] == ym[1]]
        if same:
            src = min(same, key=lambda m: abs(m[0] - ym[0]))
        else:
            mid = _month_mid(np.array([_month64(*ym)]))[0]
            src = self.months[int(np.argmin(np.abs(self._mids - mid)))]
        if (ym, src) not in self._reported:
            self._reported.add((ym, src))
            warnings.warn(f"ORAS5 forcing {ym[0]}-{ym[1]:02d} not available; "
                          f"using {src[0]}-{src[1]:02d}", UserWarning, stacklevel=3)
        return self._index[src]

    def _weights(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(ia, ib, wb): value = (1 - wb) * month[ia] + wb * month[ib]."""
        secs = np.round(np.asarray(times, np.float64)).astype(np.int64)
        tau  = self.start + secs.astype("timedelta64[s]")
        mon = tau.astype("datetime64[M]")
        if self.interp == "monthly":
            left, wb = mon, np.zeros(len(tau))
        else:
            left  = np.where(tau >= _month_mid(mon), mon, mon - 1)
            m0, m1 = _month_mid(left), _month_mid(left + 1)
            wb = (tau - m0).astype(np.float64) / (m1 - m0).astype(np.float64)
        src = {m: self._source(_ym(m)) for m in np.unique(np.concatenate([left, left + 1]))}
        ia = np.array([src[m] for m in left])
        ib = np.array([src[m] for m in left + 1])
        return ia, ib, wb

    # ------------------------------------------------------------------
    def at(self, times: np.ndarray):
        """SurfaceForcing with fields (len(times), Nx, Ny) at model times [s]."""
        from OceanJAX.timeStepping import SurfaceForcing
        import jax.numpy as jnp

        ia, ib, wb = self._weights(np.atleast_1d(times))
        w = wb.astype(np.float32)[:, None, None]
        out = {f: (1.0 - w) * d[ia] + w * d[ib] for f, d in self._data.items()}
        return SurfaceForcing(**{f: jnp.asarray(v) for f, v in out.items()})

    def chunk(self, t_start: float, n_steps: int, dt: float):
        """Forcing for steps t_start + (i + 1/2) dt, i = 0 .. n_steps-1."""
        return self.at(t_start + (np.arange(n_steps) + 0.5) * dt)
