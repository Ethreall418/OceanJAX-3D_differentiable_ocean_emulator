"""
Tests for OceanJAX.data.monthly_forcing
=======================================
Synthetic ORAS5-style monthly files (regular 1° grid) whose fields are
spatially constant and encode the month, e.g. heat_flux = 10 * month, so
every interpolated value can be checked exactly.

  1. File discovery: grouping by YYYYMM, incomplete months skipped.
  2. "linear": mid-month equals the monthly mean; halfway between two
     mid-months equals their average; smooth across month boundaries.
  3. "monthly": piecewise constant, switching on the 1st.
  4. Missing months: single month -> perpetual; same calendar month of
     another year; nearest month in time.
  5. Calendar: year crossing, leap-year February, chunk shape and times.
  6. fw_flux unit conversion kg m-2 s-1 -> m s-1.

Running
-------
    pytest OceanJAX/tests/test_monthly_forcing.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from OceanJAX.grid import OceanGrid
from OceanJAX.data.monthly_forcing import (
    MonthlyForcing, discover_oras5_forcing, ORAS5_FORCING_VARS,
)

_DAY = 86400.0


def _write_month(directory, year, month, fields=tuple(ORAS5_FORCING_VARS)):
    lon = np.arange(0.5, 10.0, 1.0)
    lat = np.arange(0.5, 10.0, 1.0)
    values = {"heat_flux": 10.0 * month, "fw_flux": 1e-3 * month,
              "tau_x": 0.01 * month, "tau_y": -0.01 * month}
    units  = {"heat_flux": "W/m2", "fw_flux": "Kg/m2/s", "tau_x": "N/m2", "tau_y": "N/m2"}
    for f in fields:
        var = ORAS5_FORCING_VARS[f]
        da  = xr.DataArray(np.full((1, len(lat), len(lon)), values[f], np.float32),
                           dims=("time_counter", "lat", "lon"),
                           coords={"lat": lat, "lon": lon}, attrs={"units": units[f]})
        xr.Dataset({var: da}).to_netcdf(
            directory / f"{var}_control_monthly_highres_2D_{year}{month:02d}_OPER_v0.1.nc")


def _grid():
    return OceanGrid.create((2.0, 8.0), (2.0, 8.0), np.array([10.0, 30.0]), 3, 3)


def _secs(start, when):
    """Model seconds from ``start`` to the datetime string ``when``."""
    return float((np.datetime64(when, "s") - np.datetime64(start, "s")).astype(np.int64))


def _heat(mf, start, when):
    return float(np.asarray(mf.at(np.array([_secs(start, when)])).heat_flux).mean())


@pytest.fixture
def three_months(tmp_path):
    for m in (1, 2, 3):
        _write_month(tmp_path, 2026, m)
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Discovery
# ---------------------------------------------------------------------------

class TestDiscovery:

    def test_groups_by_month_and_skips_incomplete(self, three_months):
        _write_month(three_months, 2026, 4, fields=("heat_flux",))
        (three_months / "votemper_control_monthly_highres_3D_202601_OPER_v0.1.nc").write_bytes(b"")
        with pytest.warns(UserWarning, match="2026-04 skipped"):
            months = discover_oras5_forcing(three_months)
        assert sorted(months) == [(2026, 1), (2026, 2), (2026, 3)]
        assert set(months[(2026, 1)]) == set(ORAS5_FORCING_VARS)


# ---------------------------------------------------------------------------
# 2-3. Interpolation modes
# ---------------------------------------------------------------------------

class TestInterpolation:

    def test_linear_mid_month_and_halfway(self, three_months):
        s  = "2026-01-01"
        mf = MonthlyForcing(three_months, _grid(), s)
        assert _heat(mf, s, "2026-01-16T12:00") == pytest.approx(10.0)     # mid-Jan
        assert _heat(mf, s, "2026-02-15T00:00") == pytest.approx(20.0)     # mid-Feb (28 d)
        # halfway between mid-Jan (Jan 16 12:00) and mid-Feb (Feb 15 00:00)
        half = np.datetime64("2026-01-16T12:00") + (
            np.datetime64("2026-02-15T00:00") - np.datetime64("2026-01-16T12:00")) // 2
        assert _heat(mf, s, str(half)) == pytest.approx(15.0, abs=1e-4)

    def test_linear_is_continuous_across_month_start(self, three_months):
        s  = "2026-01-01"
        mf = MonthlyForcing(three_months, _grid(), s)
        before = _heat(mf, s, "2026-01-31T23:00")
        after  = _heat(mf, s, "2026-02-01T01:00")
        assert abs(after - before) < 0.05

    def test_monthly_switches_on_the_first(self, three_months):
        s  = "2026-01-01"
        mf = MonthlyForcing(three_months, _grid(), s, interp="monthly")
        assert _heat(mf, s, "2026-01-31T23:59") == pytest.approx(10.0)
        assert _heat(mf, s, "2026-02-01T00:00") == pytest.approx(20.0)
        assert _heat(mf, s, "2026-02-27T00:00") == pytest.approx(20.0)

    def test_all_fields_follow_the_same_weights(self, three_months):
        s  = "2026-01-01"
        mf = MonthlyForcing(three_months, _grid(), s)
        f  = mf.at(np.array([_secs(s, "2026-02-15T00:00")]))
        assert float(np.asarray(f.tau_x).mean()) == pytest.approx(0.02)
        assert float(np.asarray(f.tau_y).mean()) == pytest.approx(-0.02)


# ---------------------------------------------------------------------------
# 4. Missing months
# ---------------------------------------------------------------------------

class TestMissingMonths:

    def test_single_month_is_perpetual(self, tmp_path):
        _write_month(tmp_path, 2026, 1)
        s = "2026-01-01"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mf = MonthlyForcing(tmp_path, _grid(), s)
            vals = [_heat(mf, s, d) for d in ("2026-01-02", "2026-06-15", "2027-11-30")]
        np.testing.assert_allclose(vals, 10.0)

    def test_same_calendar_month_of_other_year(self, three_months):
        s  = "2027-01-01"
        mf = MonthlyForcing(three_months, _grid(), s)
        with pytest.warns(UserWarning, match="2027-02 not available; using 2026-02"):
            v = _heat(mf, s, "2027-02-15T00:00")
        assert v == pytest.approx(20.0)

    def test_nearest_month_in_time(self, tmp_path):
        _write_month(tmp_path, 2026, 1)
        _write_month(tmp_path, 2026, 2)
        s  = "2026-01-01"
        mf = MonthlyForcing(tmp_path, _grid(), s)
        with pytest.warns(UserWarning, match="2026-03 not available; using 2026-02"):
            v = _heat(mf, s, "2026-03-16T12:00")
        assert v == pytest.approx(20.0)

    def test_substitution_reported_once(self, tmp_path):
        _write_month(tmp_path, 2026, 1)
        s  = "2026-01-01"
        mf = MonthlyForcing(tmp_path, _grid(), s)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mf.chunk(_secs(s, "2026-05-01"), 10, 300.0)
            mf.chunk(_secs(s, "2026-05-02"), 10, 300.0)
        assert len([x for x in w if "2026-05" in str(x.message)]) == 1


# ---------------------------------------------------------------------------
# 5. Calendar
# ---------------------------------------------------------------------------

class TestCalendar:

    def test_year_crossing(self, tmp_path):
        _write_month(tmp_path, 2026, 12)
        _write_month(tmp_path, 2027, 1)
        s  = "2026-12-01"
        mf = MonthlyForcing(tmp_path, _grid(), s)
        assert _heat(mf, s, "2026-12-16T12:00") == pytest.approx(120.0)
        assert _heat(mf, s, "2027-01-16T12:00") == pytest.approx(10.0)
        mid = _heat(mf, s, "2027-01-01T00:00")         # exactly halfway
        assert mid == pytest.approx(65.0, abs=1e-3)

    def test_leap_year_february(self, tmp_path):
        for m in (2, 3):
            _write_month(tmp_path, 2028, m)
        s  = "2028-02-01"
        mf = MonthlyForcing(tmp_path, _grid(), s)
        # Feb 2028 has 29 days: mid-month is Feb 15 12:00
        assert _heat(mf, s, "2028-02-15T12:00") == pytest.approx(20.0)
        assert np.isfinite(_heat(mf, s, "2028-02-29T12:00"))

    def test_chunk_shape_and_step_times(self, three_months):
        s  = "2026-01-01"
        mf = MonthlyForcing(three_months, _grid(), s)
        dt = 3600.0
        f  = mf.chunk(t_start=_secs(s, "2026-01-16T11:00"), n_steps=4, dt=dt)
        assert f.heat_flux.shape == (4, 3, 3)
        # step 1 is centred on 12:30... step 0 on 11:30 -> just before mid-Jan
        hf = np.asarray(f.heat_flux)[:, 0, 0]
        assert hf[0] < 10.0 + 1e-3 and np.all(np.diff(hf[1:]) > 0)


# ---------------------------------------------------------------------------
# 6. Units
# ---------------------------------------------------------------------------

def test_fw_flux_converted_to_m_per_s(three_months):
    s  = "2026-01-01"
    mf = MonthlyForcing(three_months, _grid(), s)
    f  = mf.at(np.array([_secs(s, "2026-01-16T12:00")]))
    assert float(np.asarray(f.fw_flux).mean()) == pytest.approx(1e-3 / 1000.0)
