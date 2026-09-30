"""
Tests for OceanJAX.diagnostics (verification metrics) and the monthly-mean
output of experiment.py
=========================================================================
  1. compare_fields: identical fields, constant offset, mask, area weights.
  2. skill_score.
  3. experiment._MonthlyMeans: samples are averaged per calendar month,
     the month boundary is respected and the grid description round-trips.

Running
-------
    pytest OceanJAX/tests/test_diagnostics.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from OceanJAX.diagnostics import compare_fields, skill_score, weighted_mean
from OceanJAX.grid import OceanGrid

_RNG = np.random.default_rng(1)
_A = _RNG.normal(size=(6, 5))
_AREA = np.ones((6, 5))
_MASK = np.ones((6, 5), dtype=bool)


class TestCompareFields:

    def test_identical(self):
        s = compare_fields(_A, _A, _AREA, _MASK)
        assert s["n"] == 30 and s["bias"] == 0.0 and s["rmse"] == 0.0
        assert s["corr"] == pytest.approx(1.0)

    def test_constant_offset(self):
        s = compare_fields(_A + 0.7, _A, _AREA, _MASK)
        assert s["bias"] == pytest.approx(0.7)
        assert s["rmse"] == pytest.approx(0.7)
        assert s["corr"] == pytest.approx(1.0)

    def test_mask_and_nan_ignored(self):
        m, r = _A.copy(), _A.copy()
        m[0, 0] = 100.0                        # masked out
        r[1, 1] = np.nan                       # non-finite
        mask = _MASK.copy(); mask[0, 0] = False
        s = compare_fields(m, r, _AREA, mask)
        assert s["n"] == 28 and s["rmse"] == 0.0

    def test_area_weighting(self):
        m = np.zeros((2, 1)); r = np.array([[1.0], [0.0]])
        area = np.array([[3.0], [1.0]])
        s = compare_fields(m, r, area, np.ones((2, 1), bool))
        assert s["bias"] == pytest.approx(-0.75)
        assert s["rmse"] == pytest.approx(np.sqrt(0.75))
        assert weighted_mean(r, area, np.ones((2, 1), bool)) == pytest.approx(0.75)


def test_skill_score():
    assert skill_score(0.0, 2.0) == 1.0
    assert skill_score(2.0, 2.0) == 0.0
    assert skill_score(4.0, 2.0) == -1.0
    assert np.isnan(skill_score(1.0, 0.0))


class TestMonthlyMeans:

    def test_calendar_month_means(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import experiment as e
        import netCDF4 as nc

        grid = OceanGrid.create((0.0, 10.0), (0.0, 8.0), np.array([5.0, 20.0]), 4, 3,
                                periodic_x=False)
        e.LON, e.LAT, e.START_DATE, e.N_ENSEMBLE = (0.0, 10.0), (0.0, 8.0), "2026-01-30T00:00", 1
        e._DEPTH_LEVELS = np.array([5.0, 20.0])
        mm = e._MonthlyMeans(str(tmp_path / "m.nc"), grid)

        def state(val):
            f3 = np.full((4, 3, 2), val); f2 = np.full((4, 3), val)
            return SimpleNamespace(T=f3, S=f3 + 30, u=f3 * 0.1, v=-f3, eta=f2)

        day = 86400.0
        for d, val in ((1, 1.0), (2, 3.0), (3, 10.0), (4, 20.0), (5, 30.0)):
            mm.add(state(val), d * day)        # Jan 31, Feb 1 .. Feb 4
        mm.close()

        ds = nc.Dataset(tmp_path / "m.nc")
        assert list(ds["month"][:]) == [202601, 202602]
        assert list(ds["n_samples"][:]) == [1, 4]
        np.testing.assert_allclose(ds["T"][0], 1.0)
        np.testing.assert_allclose(ds["T"][1], (3 + 10 + 20 + 30) / 4)
        np.testing.assert_allclose(ds["S"][1], 30 + 15.75)
        np.testing.assert_allclose(ds["eta"][1], 15.75)
        np.testing.assert_allclose(ds["z_exact"][:], [5.0, 20.0])
        np.testing.assert_array_equal(ds["mask_c"][:], np.asarray(grid.mask_c).astype(np.int8))
        assert tuple(ds.lon_bounds) == (0.0, 10.0) and ds.periodic_x == 0
        ds.close()
