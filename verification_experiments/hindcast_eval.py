"""
Hindcast evaluation against ORAS5
=================================
Compares the calendar-month means written by experiment.py (MONTHLY_NC)
with the ORAS5 monthly means of the same months, on the model grid and
model levels, and against a persistence forecast (the ORAS5 state of the
initial month carried forward unchanged).

usage:
    python verification_experiments/hindcast_eval.py MONTHLY_NC
        [--oras5-dir OceanJAX/data/data_oras5] [--init 2026-01] [--out DIR]

Method
------
* The model grid is rebuilt from the grid description stored in the
  monthly file (lon/lat bounds, exact level depths, bathymetry H,
  periodic_x), so the wet mask is identical to the run's.
* Each ORAS5 month is read (``oras5_month_files`` + ``read_oras5``) and
  interpolated onto that grid with ``regrid_to_model`` (horizontally and to
  the model level depths); only wet model cells are compared.
* Variables: SST and SSS (top model level), T and S at the model levels
  nearest 100, 300 and 1000 m, SSH (both fields minus their area mean;
  a linear free-surface eta and ORAS5's SSH differ by a domain offset),
  and surface current speed (qualitative at coarse resolution).
* Scores (area weighted, see OceanJAX.diagnostics): bias, RMSE, pattern
  correlation; RMSE of persistence; skill = 1 - RMSE_model / RMSE_pers;
  change correlation = corr(model - init, ORAS5 - init), i.e. whether the
  model predicts the observed evolution since the initial month.
* Regions: all, tropics (|lat| < 20), north (lat >= 20), south (lat <= -20).
* The initial month is partial (the run starts mid-month) and is only
  listed for reference, not scored.

Outputs (in --out, default <MONTHLY_NC stem>_eval/)
-------
  metrics.csv            every (month, variable, region) score
  summary.md             headline tables for the whole domain
  rmse_by_month.png      model vs persistence RMSE per month
  maps_<var>_<YYYYMM>.png  ORAS5, model error, ORAS5 change, model change
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netCDF4 as nc_lib
from OceanJAX.grid import OceanGrid, face_depths
from OceanJAX.data.oras5 import read_oras5, regrid_to_model, oras5_month_files
from OceanJAX.diagnostics import compare_fields, skill_score, weighted_mean

TARGET_DEPTHS = (100.0, 300.0, 1000.0)
REGIONS = {
    "all":     lambda lat: np.ones_like(lat, dtype=bool),
    "tropics": lambda lat: np.abs(lat) < 20.0,
    "north":   lambda lat: lat >= 20.0,
    "south":   lambda lat: lat <= -20.0,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(path: Path):
    """Monthly means + the grid rebuilt from the stored description."""
    ds = nc_lib.Dataset(path)
    # Rebuild H from the stored wet mask (bottom face of each column's deepest
    # wet cell) rather than from the stored H, which is float32 and can round
    # just below a face and drop a cell.
    z = np.asarray(ds["z_exact"][:], np.float64)
    stored = np.asarray(ds["mask_c"][:]).astype(bool)
    n_wet = stored.sum(axis=-1)
    H = np.where(n_wet > 0, face_depths(z)[n_wet], 0.0)
    grid = OceanGrid.create(
        tuple(np.asarray(ds.lon_bounds, np.float64)),
        tuple(np.asarray(ds.lat_bounds, np.float64)),
        z, len(ds.dimensions["x"]), len(ds.dimensions["y"]),
        bathymetry=H, periodic_x=bool(ds.periodic_x),
    )
    if not np.array_equal(stored, np.asarray(grid.mask_c) > 0):
        raise RuntimeError("rebuilt grid mask differs from the mask stored in the run")
    months = [int(m) for m in ds["month"][:]]
    fields = {m: {f: np.asarray(ds[f][i], np.float64) for f in ("T", "S", "u", "v", "eta")}
              for i, m in enumerate(months)}
    info = dict(n_samples={m: int(n) for m, n in zip(months, ds["n_samples"][:])},
                start_date=ds.start_date, init_mode=ds.init_mode)
    if "member" in ds.dimensions:
        raise NotImplementedError("ensemble monthly files: evaluate one member at a time")
    ds.close()
    return grid, fields, info


def load_oras5_month(oras5_dir: Path, yyyymm: int, grid, cache: dict):
    """ORAS5 month interpolated to the model grid (host numpy arrays)."""
    if yyyymm not in cache:
        y, m = divmod(yyyymm, 100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = regrid_to_model(read_oras5(oras5_month_files(oras5_dir, y, m)), grid)
        cache[yyyymm] = {f: np.asarray(getattr(st, f), np.float64)
                         for f in ("T", "S", "u", "v", "eta")}
    return cache[yyyymm]


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

def _speed(u, v):
    """Surface speed at tracer points from face velocities (C grid)."""
    uc = 0.5 * (u + np.roll(u, 1, axis=0))
    vc = 0.5 * (v + np.concatenate([np.zeros_like(v[:, :1]), v[:, :-1]], axis=1))
    return np.sqrt(uc ** 2 + vc ** 2)


def variables(grid):
    """name -> (extract(fields) -> 2-D array, wet mask (Nx, Ny), unit)."""
    z = np.asarray(grid.z_c, np.float64)
    m = np.asarray(grid.mask_c) > 0
    out = {
        "SST": (lambda F: F["T"][:, :, 0], m[:, :, 0], "degC"),
        "SSS": (lambda F: F["S"][:, :, 0], m[:, :, 0], "psu"),
    }
    for d in TARGET_DEPTHS:
        if d > z[-1]:
            continue
        k = int(np.argmin(np.abs(z - d)))
        tag = f"{z[k]:.0f}m"
        out[f"T{tag}"] = (lambda F, k=k: F["T"][:, :, k], m[:, :, k], "degC")
        out[f"S{tag}"] = (lambda F, k=k: F["S"][:, :, k], m[:, :, k], "psu")
    area = np.asarray(grid.area_c, np.float64)

    def ssh(F):
        e = F["eta"].copy()
        return e - weighted_mean(e, area, m[:, :, 0])

    out["SSH"] = (ssh, m[:, :, 0], "m (minus area mean)")
    out["speed0"] = (lambda F: _speed(F["u"][:, :, 0], F["v"][:, :, 0]), m[:, :, 0], "m/s")
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(grid, model, oras5_dir: Path, init: int):
    area = np.asarray(grid.area_c, np.float64)
    lat = np.broadcast_to(np.asarray(grid.lat_c, np.float64)[None, :], area.shape)
    cache: dict = {}
    o_init = load_oras5_month(oras5_dir, init, grid, cache)
    rows, maps = [], {}
    for month in sorted(model):
        if month == init:
            continue
        print(f"  evaluating {month} ...", flush=True)
        o_mon = load_oras5_month(oras5_dir, month, grid, cache)
        for name, (get, wet, unit) in variables(grid).items():
            mf, of, pf = get(model[month]), get(o_mon), get(o_init)
            maps[(name, month)] = (of, mf, pf)
            for region, sel in REGIONS.items():
                mask = wet & sel(lat)
                s_mod = compare_fields(mf, of, area, mask)
                s_per = compare_fields(pf, of, area, mask)
                tend = compare_fields(mf - pf, of - pf, area, mask)["corr"]
                rows.append(dict(month=month, variable=name, unit=unit, region=region,
                                 n=s_mod["n"], bias=s_mod["bias"], rmse=s_mod["rmse"],
                                 corr=s_mod["corr"], rmse_persistence=s_per["rmse"],
                                 skill=skill_score(s_mod["rmse"], s_per["rmse"]),
                                 change_corr=tend))
    return rows, maps


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(rows, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_summary(rows, info, path: Path):
    months = sorted({r["month"] for r in rows})
    names = list(dict.fromkeys(r["variable"] for r in rows))
    lines = [f"# Hindcast evaluation vs ORAS5", "",
             f"Run start: {info['start_date']}, init mode: {info['init_mode']}. "
             f"Samples per month: {info['n_samples']}.", "",
             "Whole domain, area weighted. skill = 1 - RMSE_model / RMSE_persistence "
             "(> 0: better than persistence). change corr = corr(model - init, ORAS5 - init).", ""]
    for name in names:
        unit = next(r["unit"] for r in rows if r["variable"] == name)
        lines += [f"## {name} [{unit}]", "",
                  "| month | bias | RMSE | RMSE persistence | skill | pattern corr | change corr |",
                  "|---|---|---|---|---|---|---|"]
        for m in months:
            r = next(x for x in rows if x["variable"] == name and x["month"] == m
                     and x["region"] == "all")
            lines.append(f"| {m} | {r['bias']:+.3f} | {r['rmse']:.3f} | "
                         f"{r['rmse_persistence']:.3f} | {r['skill']:+.2f} | "
                         f"{r['corr']:.2f} | {r['change_corr']:.2f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(rows, maps, grid, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    months = sorted({r["month"] for r in rows})
    labels = [f"{m % 100:02d}" for m in months]
    names = [n for n in ("SST", "SSS", "SSH") if any(r["variable"] == n for r in rows)]
    names += [n for n in dict.fromkeys(r["variable"] for r in rows)
              if n.startswith("T") and n != "T" and n not in names][:2]
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.2))
    for ax, name in zip(np.atleast_1d(axes), names):
        sel = [next(r for r in rows if r["variable"] == name and r["month"] == m
                    and r["region"] == "all") for m in months]
        ax.plot(labels, [r["rmse"] for r in sel], "o-", label="model")
        ax.plot(labels, [r["rmse_persistence"] for r in sel], "s--", label="persistence")
        ax.set_title(f"{name} RMSE"); ax.set_xlabel("month"); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout(); fig.savefig(out / "rmse_by_month.png", dpi=120); plt.close(fig)

    lon = np.asarray(grid.lon_c); lat = np.asarray(grid.lat_c)
    for (name, month), (of, mf, pf) in maps.items():
        if name not in ("SST", "SSS", "SSH") and not name.startswith("T"):
            continue
        wet = np.isfinite(of) & (np.abs(of) > 0) if name != "SSH" else np.isfinite(of)
        wet &= (np.asarray(grid.mask_c)[:, :, 0] > 0)
        panels = [("ORAS5", of, "viridis", None), ("model - ORAS5", mf - of, "RdBu_r", True),
                  ("ORAS5 change since init", of - pf, "RdBu_r", True),
                  ("model change since init", mf - pf, "RdBu_r", True)]
        lim = np.nanmax(np.abs(np.where(wet, of - pf, np.nan))) or 1.0
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
        for ax, (title, f, cmap, sym) in zip(axes, panels):
            data = np.ma.masked_where(~wet, f).T
            kw = dict(vmin=-lim, vmax=lim) if sym else {}
            im = ax.pcolormesh(lon, lat, data, cmap=cmap, shading="auto", **kw)
            ax.set_title(f"{name} {month}: {title}", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout(); fig.savefig(out / f"maps_{name}_{month}.png", dpi=110); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("monthly_nc")
    ap.add_argument("--oras5-dir", default=str(Path(__file__).resolve().parents[1]
                                              / "OceanJAX/data/data_oras5"))
    ap.add_argument("--init", default=None, help="initial month YYYY-MM (default: first record)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    path = Path(a.monthly_nc)
    out = Path(a.out) if a.out else path.with_name(path.stem + "_eval")
    out.mkdir(parents=True, exist_ok=True)
    grid, model, info = load_model(path)
    init = (int(a.init.replace("-", "")) if a.init else min(model))
    print(f"Evaluating {path.name}: months {sorted(model)}, init {init}, "
          f"grid {grid.Nx}x{grid.Ny}x{grid.Nz}", flush=True)
    rows, maps = evaluate(grid, model, Path(a.oras5_dir), init)
    write_csv(rows, out / "metrics.csv")
    write_summary(rows, info, out / "summary.md")
    plot(rows, maps, grid, out)
    print(f"Results in {out}")
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
