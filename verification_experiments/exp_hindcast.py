"""
Hindcast experiment: ORAS5-initialised run, evaluated against ORAS5
===================================================================
Runs experiment.py with the configuration below and then evaluates the
calendar-month means with hindcast_eval.py.  One integration covers both
planned experiments:

  Experiment 1 — one month ahead:  model February mean vs ORAS5 February
  Experiment 2 — five months ahead: model June mean vs ORAS5 June
  (March–May give the error growth in between.)

Setup
-----
  - Atlantic 80W–20E, 50S–60N, 1 deg (100 x 80), ORAS5 levels to 4000 m
    thinned to >= 10 m thickness, land mask / bathymetry from ORAS5,
    closed east/west walls.
  - Initial state: full ORAS5 January 2026 (T, S, u, v, eta), taken to
    represent 2026-01-16 12:00 (the time stamp of the monthly mean).
  - Forcing: monthly ORAS5 fluxes, January–June, linearly interpolated
    between mid-months (after 16 June the June values are held).
  - Physics: PP81 vertical mixing (--mixing kpp adds the KPP surface
    boundary layer), Munk nu_h, quadratic bottom drag, freezing limit.  No SST/SSS restoring (the evaluation target is ORAS5
    SST itself).
  - dt = 90 s (checked against the barotropic CFL limit at start-up).
  - Output: daily surface fields and calendar-month means of the 3-D state.

usage:
    python verification_experiments/exp_hindcast.py [--out DIR] [--days N]
                                                     [--mixing pp81|kpp|constant] [--eval-only]

Outputs (default hindcast_output/, not tracked by git):
    hindcast_daily.nc, hindcast_monthly.nc, hindcast_monthly_eval/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verification_experiments"))

START = "2026-01-16T12:00"
END   = "2026-07-01T00:00"


def configure(e, out: Path, days: float, mixing: str = "pp81"):
    """Set experiment.py's CONFIG block for the hindcast."""
    e.LON, e.LAT = (-80.0, 20.0), (-50.0, 60.0)
    e.NX, e.NY = 100, 80
    e.VERTICAL_LEVELS, e.DEPTH_MAX, e.DZ_TOP = "oras5", 4000.0, 10.0
    e.DT = 90.0
    e.INIT_MODE = "oras5_full"
    e.ORAS5_PATH = "OceanJAX/data/data_oras5/oras5_2026_01_native_merged.nc"
    e.START_DATE = START
    e.N_DAYS = days
    e.FORCING_DIR, e.FORCING_INTERP = "OceanJAX/data/data_oras5", "linear"
    e.NU_H, e.VERTICAL_MIXING = "munk", mixing
    e.N_ENSEMBLE, e.N_DEVICES_X, e.N_DEVICES_Y = 1, 1, 1
    steps_per_day = int(round(86400 / e.DT))
    e.SAVE_INTERVAL = e.CHUNK_SIZE = steps_per_day
    e.SAVE_DAILY_3D = False
    e.OUTPUT_NC = str(out / "hindcast_daily.nc")
    e.MONTHLY_NC = str(out / "hindcast_monthly.nc")
    e._ORAS5_FILE = e._SCRIPT_DIR / e.ORAS5_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "hindcast_output"))
    ap.add_argument("--days", type=float, default=None,
                    help=f"integration length (default: {START} to {END})")
    ap.add_argument("--mixing", default="pp81", choices=["pp81", "kpp", "constant"],
                    help="vertical mixing scheme (default: pp81)")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    days = a.days if a.days is not None else float(
        (np.datetime64(END, "s") - np.datetime64(START, "s")).astype(np.int64)) / 86400.0

    if not a.eval_only:
        import experiment as e
        configure(e, out, days, a.mixing)
        try:
            e.main()
        except SystemExit as exc:
            if exc.code not in (0, None):
                sys.exit(f"model run failed (exit {exc.code}); evaluation skipped")

    import hindcast_eval
    hindcast_eval.main([str(out / "hindcast_monthly.nc"), "--init", "2026-01"])


if __name__ == "__main__":
    main()
