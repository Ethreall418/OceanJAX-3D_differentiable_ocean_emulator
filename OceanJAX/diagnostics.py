"""
OceanJAX Diagnostics – verification metrics
===========================================
Area-weighted scores for comparing a model field with a reference
(e.g. ORAS5) on the model grid, as used by
``verification_experiments/hindcast_eval.py``.

All functions take plain numpy arrays: a 2-D field (Nx, Ny), the cell area
(Nx, Ny) and a boolean wet mask (Nx, Ny); points outside the mask or with a
non-finite value in either field are ignored.

Contents
--------
  weighted_mean        – area-weighted mean
  compare_fields       – bias, RMSE and pattern correlation of model vs reference
  skill_score          – 1 - RMSE_model / RMSE_baseline
"""

from __future__ import annotations

import numpy as np


def _valid(mask: np.ndarray, *fields: np.ndarray) -> np.ndarray:
    ok = np.asarray(mask, dtype=bool)
    for f in fields:
        ok = ok & np.isfinite(f)
    return ok


def weighted_mean(field: np.ndarray, area: np.ndarray, mask: np.ndarray) -> float:
    """Area-weighted mean of ``field`` over valid wet points (nan if none)."""
    ok = _valid(mask, field)
    w = np.asarray(area, np.float64)[ok]
    if w.sum() <= 0:
        return float("nan")
    return float((np.asarray(field, np.float64)[ok] * w).sum() / w.sum())


def compare_fields(
    model: np.ndarray,
    ref:   np.ndarray,
    area:  np.ndarray,
    mask:  np.ndarray,
) -> dict[str, float]:
    """
    Area-weighted comparison of ``model`` with ``ref``.

    Returns a dict with
      n     : number of valid points
      bias  : mean(model - ref)
      rmse  : sqrt(mean((model - ref)^2))
      corr  : Pearson pattern correlation (weighted, anomalies about the
              weighted means); nan if either field is constant
    """
    ok = _valid(mask, model, ref)
    w = np.asarray(area, np.float64)[ok]
    if w.sum() <= 0:
        nan = float("nan")
        return dict(n=0, bias=nan, rmse=nan, corr=nan)
    m = np.asarray(model, np.float64)[ok]
    r = np.asarray(ref, np.float64)[ok]
    w = w / w.sum()
    d = m - r
    ma, ra = m - (w * m).sum(), r - (w * r).sum()
    denom = np.sqrt((w * ma ** 2).sum() * (w * ra ** 2).sum())
    corr = float((w * ma * ra).sum() / denom) if denom > 0 else float("nan")
    return dict(n=int(ok.sum()), bias=float((w * d).sum()),
                rmse=float(np.sqrt((w * d ** 2).sum())), corr=corr)


def skill_score(rmse_model: float, rmse_baseline: float) -> float:
    """
    RMSE skill score relative to a baseline forecast (e.g. persistence):
    1 = perfect, 0 = no better than the baseline, < 0 = worse.
    """
    if not np.isfinite(rmse_baseline) or rmse_baseline <= 0:
        return float("nan")
    return 1.0 - rmse_model / rmse_baseline
