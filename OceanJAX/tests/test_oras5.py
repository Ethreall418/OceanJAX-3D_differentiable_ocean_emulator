"""
Tests for OceanJAX.data.oras5
==============================
All tests use synthetic in-memory NetCDF files; no real ORAS5 download required.

Coverage:
  TestReadOras5      — file I/O, alias resolution, coordinate normalisation,
                       optional-variable contract, error paths.
  TestRegridToModel  — interpolation shapes, NaN-fill strategy,
                       None u/v/eta → zero fields, resolution warning,
                       OceanGrid mask application.

Running
-------
    pytest OceanJAX/tests/test_oras5.py -v
"""

from __future__ import annotations

import warnings
import numpy as np
import pytest
import xarray as xr

from OceanJAX.grid import OceanGrid
from OceanJAX.data.oras5 import read_oras5, regrid_to_model, load_oras5, oras5_bathymetry


# ---------------------------------------------------------------------------
# Shared synthetic-data helpers
# ---------------------------------------------------------------------------

# Source grid dimensions
_SRC_LON   = np.array([0.0, 1.0, 2.0, 3.0, 4.0],         dtype=np.float64)
_SRC_LAT   = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)
_SRC_DEPTH = np.array([10.0, 50.0, 200.0],                  dtype=np.float64)

_NX_SRC = len(_SRC_LON)
_NY_SRC = len(_SRC_LAT)
_NZ_SRC = len(_SRC_DEPTH)


def _make_T() -> np.ndarray:
    """Synthetic temperature: linearly varies with depth (Nz, Ny, Nx)."""
    T = np.zeros((_NZ_SRC, _NY_SRC, _NX_SRC), dtype=np.float32)
    for k in range(_NZ_SRC):
        T[k] = 20.0 - k * 5.0   # 20, 15, 10 °C
    return T


def _make_S() -> np.ndarray:
    S = np.full((_NZ_SRC, _NY_SRC, _NX_SRC), 35.0, dtype=np.float32)
    return S


def _write_nc(
    path,
    T_name:   str = "thetao",
    S_name:   str = "so",
    u_name:   str | None = None,
    v_name:   str | None = None,
    eta_name: str | None = None,
    lat:      np.ndarray | None = None,
    add_time: bool = True,
    nan_mask: np.ndarray | None = None,  # (Nz, Ny, Nx) bool — set these T cells to NaN
) -> None:
    """Write a minimal synthetic ORAS5-like NetCDF to *path*."""
    if lat is None:
        lat = _SRC_LAT

    T = _make_T()
    S = _make_S()

    if nan_mask is not None:
        T = T.copy()
        T[nan_mask] = np.nan

    if add_time:
        dims_3d  = ("time", "depth", "lat", "lon")
        dims_2d  = ("time", "lat",   "lon")
        T_data   = T[np.newaxis]
        S_data   = S[np.newaxis]
        u_data   = np.zeros_like(T)[np.newaxis]
        v_data   = np.zeros_like(T)[np.newaxis]
        eta_data = np.zeros((_NY_SRC, _NX_SRC), dtype=np.float32)[np.newaxis]
    else:
        dims_3d  = ("depth", "lat", "lon")
        dims_2d  = ("lat",   "lon")
        T_data   = T
        S_data   = S
        u_data   = np.zeros_like(T)
        v_data   = np.zeros_like(T)
        eta_data = np.zeros((_NY_SRC, _NX_SRC), dtype=np.float32)

    coords = {
        "lon":   ("lon",   _SRC_LON),
        "lat":   ("lat",   lat),
        "depth": ("depth", _SRC_DEPTH),
    }
    if add_time:
        coords["time"] = ("time", np.array([0.0]))

    data_vars = {
        T_name: (dims_3d, T_data),
        S_name: (dims_3d, S_data),
    }
    if u_name is not None:
        data_vars[u_name] = (dims_3d, u_data)
    if v_name is not None:
        data_vars[v_name] = (dims_3d, v_data)
    if eta_name is not None:
        data_vars[eta_name] = (dims_2d, eta_data)

    ds = xr.Dataset(data_vars, coords=coords)
    ds.to_netcdf(path)


def _target_grid() -> OceanGrid:
    """Small OceanJAX grid entirely within the source domain."""
    depth_levels = np.array([15.0, 100.0], dtype=np.float64)
    return OceanGrid.create(
        lon_bounds=(1.0, 3.0),
        lat_bounds=(11.0, 14.0),
        depth_levels=depth_levels,
        Nx=3,
        Ny=4,
    )


# ---------------------------------------------------------------------------
# TestReadOras5
# ---------------------------------------------------------------------------

class TestReadOras5:

    def test_required_only_uveta_are_none(self, tmp_path):
        """File with only T and S: u, v, eta must be None (not absent keys)."""
        nc = tmp_path / "req_only.nc"
        _write_nc(nc)
        raw = read_oras5(nc)

        assert raw["u"]   is None
        assert raw["v"]   is None
        assert raw["eta"] is None

    def test_all_variables_non_none(self, tmp_path):
        """File with all 5 variables: none should be None."""
        nc = tmp_path / "all_vars.nc"
        _write_nc(nc, u_name="uo", v_name="vo", eta_name="zos")
        raw = read_oras5(nc)

        assert raw["u"]   is not None
        assert raw["v"]   is not None
        assert raw["eta"] is not None

    def test_alias_thetao_oras(self, tmp_path):
        """thetao_oras / so_oras aliases are resolved correctly."""
        nc = tmp_path / "alias_oras.nc"
        _write_nc(nc, T_name="thetao_oras", S_name="so_oras")
        raw = read_oras5(nc)
        assert raw["T"] is not None
        assert raw["T"].shape == (_NZ_SRC, _NY_SRC, _NX_SRC)

    def test_alias_votemper(self, tmp_path):
        """votemper / vosaline (NEMO-style) aliases are resolved."""
        nc = tmp_path / "alias_nemo.nc"
        _write_nc(nc, T_name="votemper", S_name="vosaline")
        raw = read_oras5(nc)
        assert raw["T"] is not None

    def test_negative_depth_flipped(self, tmp_path):
        """Negative depth coordinates are flipped to positive-downward."""
        # Write a file then patch the depth coordinate to negative
        nc = tmp_path / "neg_depth.nc"
        _write_nc(nc)
        ds = xr.open_dataset(nc).load()
        ds = ds.assign_coords(depth=-ds["depth"])
        ds.to_netcdf(tmp_path / "neg_depth2.nc")

        raw = read_oras5(tmp_path / "neg_depth2.nc")
        assert np.all(raw["depth"] > 0), "depth should be positive downward"

    def test_descending_lat_sorted(self, tmp_path):
        """Descending latitude coordinates are sorted to ascending."""
        nc = tmp_path / "desc_lat.nc"
        _write_nc(nc, lat=_SRC_LAT[::-1])   # descending
        raw = read_oras5(nc)

        assert np.all(np.diff(raw["lat"]) > 0), "lat should be ascending"
        assert raw["T"].shape == (_NZ_SRC, _NY_SRC, _NX_SRC)

    def test_descending_lat_values_consistent(self, tmp_path):
        """After sorting, T values must match those from an ascending-lat file."""
        nc_asc  = tmp_path / "asc.nc"
        nc_desc = tmp_path / "desc.nc"
        _write_nc(nc_asc,  lat=_SRC_LAT)
        _write_nc(nc_desc, lat=_SRC_LAT[::-1])
        raw_asc  = read_oras5(nc_asc)
        raw_desc = read_oras5(nc_desc)

        np.testing.assert_allclose(raw_asc["T"], raw_desc["T"], atol=1e-5)

    def test_2d_latlon_accepted(self, tmp_path):
        """2-D nav_lat/nav_lon (NEMO curvilinear) must NOT raise — they are accepted."""
        nc = tmp_path / "curv_ok.nc"
        lon2d = np.tile(_SRC_LON, (_NY_SRC, 1)).astype(np.float64)
        lat2d = np.tile(_SRC_LAT[:, np.newaxis], (1, _NX_SRC)).astype(np.float64)
        T_data = _make_T()[np.newaxis]
        ds = xr.Dataset(
            {"thetao": (("time_counter", "deptht", "y", "x"), T_data),
             "so":     (("time_counter", "deptht", "y", "x"), T_data)},
            coords={
                "nav_lon":  (("y", "x"), lon2d),
                "nav_lat":  (("y", "x"), lat2d),
                "deptht":   ("deptht",   _SRC_DEPTH),
                "time_counter": ("time_counter", [0.0]),
            },
        )
        ds.to_netcdf(nc)
        raw = read_oras5(nc)   # must not raise
        assert raw["lat"].ndim == 2
        assert raw["lon"].ndim == 2
        assert raw["T"].shape == (_NZ_SRC, _NY_SRC, _NX_SRC)

    def test_missing_required_raises(self, tmp_path):
        """File with no T-like variable must raise KeyError."""
        nc = tmp_path / "no_T.nc"
        ds = xr.Dataset(
            {"so": (("time", "depth", "lat", "lon"), _make_S()[np.newaxis])},
            coords={
                "lon": ("lon", _SRC_LON),
                "lat": ("lat", _SRC_LAT),
                "depth": ("depth", _SRC_DEPTH),
                "time":  ("time",  [0.0]),
            },
        )
        ds.to_netcdf(nc)
        with pytest.raises(KeyError):
            read_oras5(nc)

    def test_return_keys_always_present(self, tmp_path):
        """All fourteen dict keys are present regardless of which vars exist."""
        nc = tmp_path / "keys.nc"
        _write_nc(nc)
        raw = read_oras5(nc)
        for key in ("T", "S", "u", "v", "eta",
                    "lon", "lat", "depth",
                    "depth_u", "depth_v",
                    "lon_u", "lat_u", "lon_v", "lat_v"):
            assert key in raw, f"Key '{key}' missing from read_oras5 output"

    def test_coordinate_shapes(self, tmp_path):
        """lon/lat/depth are 1-D with the expected lengths."""
        nc = tmp_path / "coords.nc"
        _write_nc(nc)
        raw = read_oras5(nc)
        assert raw["lon"].shape   == (_NX_SRC,)
        assert raw["lat"].shape   == (_NY_SRC,)
        assert raw["depth"].shape == (_NZ_SRC,)


# ---------------------------------------------------------------------------
# TestRegridToModel
# ---------------------------------------------------------------------------

class TestRegridToModel:

    def test_output_shapes(self, tmp_path):
        """T, S, u, v have shape (Nx, Ny, Nz); eta has shape (Nx, Ny)."""
        nc = tmp_path / "shapes.nc"
        _write_nc(nc, u_name="uo", v_name="vo", eta_name="zos")
        grid = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)

        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        assert state.T.shape   == (Nx, Ny, Nz)
        assert state.S.shape   == (Nx, Ny, Nz)
        assert state.u.shape   == (Nx, Ny, Nz)
        assert state.v.shape   == (Nx, Ny, Nz)
        assert state.eta.shape == (Nx, Ny)

    def test_no_nan_in_output(self, tmp_path):
        """Regridded state must contain no NaN."""
        nc = tmp_path / "nonan.nc"
        _write_nc(nc)
        grid = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)

        assert not np.any(np.isnan(np.array(state.T))), "NaN in T"
        assert not np.any(np.isnan(np.array(state.S))), "NaN in S"

    def test_nan_fill_linear_then_nn(self, tmp_path):
        """NaN cells in the source (coast mask) must be filled by the three-tier strategy."""
        # Set all source T values at the top two cells of j=0 to NaN
        nan_mask = np.zeros((_NZ_SRC, _NY_SRC, _NX_SRC), dtype=bool)
        nan_mask[:, 0, :] = True   # entire southern row is NaN

        nc = tmp_path / "nan_src.nc"
        _write_nc(nc, nan_mask=nan_mask)
        grid = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid, T_fill=5.0)

        T = np.array(state.T)
        assert not np.any(np.isnan(T)), "NaN survived three-tier fill strategy"
        # All ocean cells should have T > 0 (fill=5°, real data ≥ 10°)
        ocean = np.array(grid.mask_c, dtype=bool)
        assert np.all(T[ocean] > 0.0)

    def test_none_uveta_gives_zero_fields(self, tmp_path):
        """raw u/v/eta = None must produce zero u, v, eta in the OceanState."""
        nc = tmp_path / "ts_only.nc"
        _write_nc(nc)   # no u/v/eta
        grid = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)

        assert np.allclose(np.array(state.u),   0.0), "u should be zero"
        assert np.allclose(np.array(state.v),   0.0), "v should be zero"
        assert np.allclose(np.array(state.eta), 0.0), "eta should be zero"

    def test_resolution_warning(self, tmp_path):
        """Finer target than source grid must trigger UserWarning."""
        nc = tmp_path / "fine.nc"
        _write_nc(nc)
        raw = read_oras5(nc)

        # Target grid at ~0.1° — finer than the 1° synthetic source
        depth_levels = np.array([15.0, 100.0], dtype=np.float64)
        fine_grid = OceanGrid.create(
            lon_bounds=(1.0, 2.5),
            lat_bounds=(11.0, 13.5),
            depth_levels=depth_levels,
            Nx=15,
            Ny=25,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            regrid_to_model(raw, fine_grid)

        categories = [str(w.category) for w in caught]
        assert any("UserWarning" in c for c in categories), (
            "Expected UserWarning for upscaling; got: " + str(categories)
        )

    def test_masks_applied(self, tmp_path):
        """Land points (mask_c == 0) must be zero in regridded T."""
        nc = tmp_path / "mask.nc"
        _write_nc(nc)

        depth_levels = np.array([15.0, 100.0, 300.0], dtype=np.float64)
        bathy = np.full((3, 4), 200.0)   # most cells: 200 m → only k=0,1 are wet
        grid = OceanGrid.create(
            lon_bounds=(1.0, 3.0),
            lat_bounds=(11.0, 14.0),
            depth_levels=depth_levels,
            Nx=3,
            Ny=4,
            bathymetry=bathy,
        )
        state = regrid_to_model(read_oras5(nc), grid)

        T = np.array(state.T)
        mask = np.array(grid.mask_c)
        # T must be zero wherever mask_c == 0
        assert np.allclose(T[mask == 0], 0.0), (
            "Non-zero T values in land cells after regridding"
        )

    def test_u_without_v_does_not_crash(self, tmp_path):
        """u present but v absent: regrid must succeed and v must be zero."""
        nc = tmp_path / "u_only.nc"
        _write_nc(nc, u_name="uo")   # v_name=None by default
        grid = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)
        assert np.allclose(np.array(state.v), 0.0), "v should be zero when absent"

    def test_v_without_u_does_not_crash(self, tmp_path):
        """v present but u absent: regrid must succeed and u must be zero."""
        nc = tmp_path / "v_only.nc"
        _write_nc(nc, v_name="vo")   # u_name=None by default
        grid = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)
        assert np.allclose(np.array(state.u), 0.0), "u should be zero when absent"

    def test_no_coastal_contamination(self, tmp_path):
        """
        NaN-aware fill must not bleed fill_value into coast-adjacent ocean cells.

        Setup: all source T values are 10.0 except one coastal column (i=0)
        which is entirely NaN (land).  A large fill_value=999 is passed.
        The target grid sits well inside the ocean region (lon >= 1.0) so
        none of its cells should receive a value blended with 999.
        With the old fill-then-interpolate approach, coast-adjacent target
        points would come out ≈ (10*w + 999*(1-w)) >> 10.  With normalised
        convolution the answer should be ≈ 10.
        """
        nan_mask = np.zeros((_NZ_SRC, _NY_SRC, _NX_SRC), dtype=bool)
        nan_mask[:, :, 0] = True    # entire western column is NaN / land

        nc = tmp_path / "coastal.nc"
        _write_nc(nc, nan_mask=nan_mask)
        grid = _target_grid()   # lon in [1, 3], safely away from lon=0 coast

        state = regrid_to_model(read_oras5(nc), grid, T_fill=999.0)
        T = np.array(state.T)
        ocean = np.array(grid.mask_c, dtype=bool)

        # All ocean cells should be close to real ocean values (10–20 °C),
        # NOT contaminated by the fill_value of 999.
        assert np.all(T[ocean] < 50.0), (
            f"Coastal contamination detected: max T = {T[ocean].max():.1f} "
            "(expected ≈ 10–20, fill=999)"
        )

    def test_uv_zero_below_source_depth(self, tmp_path):
        """u and v must be exactly zero at target levels below src_depth.max()."""
        nc = tmp_path / "deep_uv.nc"
        _write_nc(nc, u_name="uo", v_name="vo", eta_name="zos")

        # Target grid with a level (300 m) deeper than _SRC_DEPTH.max() = 200 m
        depth_levels = np.array([15.0, 300.0], dtype=np.float64)
        grid = OceanGrid.create(
            lon_bounds=(1.0, 3.0),
            lat_bounds=(11.0, 14.0),
            depth_levels=depth_levels,
            Nx=3,
            Ny=4,
        )
        state = regrid_to_model(read_oras5(nc), grid)
        u = np.array(state.u)
        v = np.array(state.v)

        # k=1 corresponds to 300 m, which is below the source max of 200 m
        assert np.allclose(u[:, :, 1], 0.0), "u at 300 m (below src) should be zero"
        assert np.allclose(v[:, :, 1], 0.0), "v at 300 m (below src) should be zero"

    def test_load_oras5_wrapper(self, tmp_path):
        """load_oras5 convenience wrapper produces same result as two-layer call."""
        nc = tmp_path / "wrap.nc"
        _write_nc(nc)
        grid = _target_grid()

        state_direct = regrid_to_model(read_oras5(nc), grid)
        state_wrap   = load_oras5(nc, grid)

        np.testing.assert_allclose(np.array(state_wrap.T), np.array(state_direct.T))
        np.testing.assert_allclose(np.array(state_wrap.S), np.array(state_direct.S))


# ---------------------------------------------------------------------------
# TestCurvilinearFormat  — NEMO/ORCA-style inputs
# ---------------------------------------------------------------------------

def _write_nemo_nc(
    path,
    u_name: str | None = None,
    v_name: str | None = None,
    eta_name: str | None = None,
) -> None:
    """Write a minimal NEMO/ORCA-style NetCDF (2-D nav_lat/nav_lon, time_counter)."""
    lon2d = np.tile(_SRC_LON, (_NY_SRC, 1)).astype(np.float64)
    lat2d = np.tile(_SRC_LAT[:, np.newaxis], (1, _NX_SRC)).astype(np.float64)

    T = _make_T()[np.newaxis]           # (1, Nz, Ny, Nx)
    S = _make_S()[np.newaxis]
    uv = np.zeros_like(T)
    eta_data = np.zeros((1, _NY_SRC, _NX_SRC), dtype=np.float32)

    dims3 = ("time_counter", "deptht", "y", "x")
    dims2 = ("time_counter", "y", "x")
    data_vars = {"votemper": (dims3, T), "vosaline": (dims3, S)}
    if u_name is not None:
        data_vars[u_name] = (dims3, uv)
    if v_name is not None:
        data_vars[v_name] = (dims3, uv)
    if eta_name is not None:
        data_vars[eta_name] = (dims2, eta_data)

    ds = xr.Dataset(data_vars, coords={
        "nav_lon":      (("y", "x"), lon2d),
        "nav_lat":      (("y", "x"), lat2d),
        "deptht":       ("deptht",   _SRC_DEPTH),
        "time_counter": ("time_counter", np.array([0.0])),
    })
    ds.to_netcdf(path)


class TestCurvilinearFormat:
    """End-to-end tests for NEMO/ORCA curvilinear-grid ORAS5 files."""

    def test_read_shapes(self, tmp_path):
        """read_oras5 must return correct array shapes for a NEMO file."""
        nc = tmp_path / "nemo.nc"
        _write_nemo_nc(nc)
        raw = read_oras5(nc)
        assert raw["T"].shape   == (_NZ_SRC, _NY_SRC, _NX_SRC)
        assert raw["S"].shape   == (_NZ_SRC, _NY_SRC, _NX_SRC)
        assert raw["lat"].shape == (_NY_SRC, _NX_SRC)
        assert raw["lon"].shape == (_NY_SRC, _NX_SRC)
        assert raw["depth"].shape == (_NZ_SRC,)

    def test_read_optional_none(self, tmp_path):
        """Without u/v/eta in file, raw dict entries must be None."""
        nc = tmp_path / "nemo_ts.nc"
        _write_nemo_nc(nc)
        raw = read_oras5(nc)
        assert raw["u"]   is None
        assert raw["v"]   is None
        assert raw["eta"] is None

    def test_read_full_vars(self, tmp_path):
        """With all five variables, none should be None."""
        nc = tmp_path / "nemo_full.nc"
        _write_nemo_nc(nc, u_name="vozocrtx", v_name="vomecrty", eta_name="sossheig")
        raw = read_oras5(nc)
        assert raw["u"]   is not None
        assert raw["v"]   is not None
        assert raw["eta"] is not None

    def test_regrid_output_shapes(self, tmp_path):
        """regrid_to_model must produce (Nx, Ny, Nz) T/S and (Nx, Ny) eta."""
        nc = tmp_path / "nemo_r.nc"
        _write_nemo_nc(nc, u_name="vozocrtx", v_name="vomecrty", eta_name="sossheig")
        grid  = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        assert state.T.shape   == (Nx, Ny, Nz)
        assert state.S.shape   == (Nx, Ny, Nz)
        assert state.eta.shape == (Nx, Ny)

    def test_regrid_no_nan(self, tmp_path):
        """Regridded fields from NEMO file must contain no NaN."""
        nc = tmp_path / "nemo_nan.nc"
        _write_nemo_nc(nc)
        grid  = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)
        assert not np.any(np.isnan(np.array(state.T))), "NaN in T"
        assert not np.any(np.isnan(np.array(state.S))), "NaN in S"

    def test_regrid_ts_values_plausible(self, tmp_path):
        """Regridded T must stay within the synthetic source range [10, 20]."""
        nc = tmp_path / "nemo_vals.nc"
        _write_nemo_nc(nc)
        grid  = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)
        T = np.array(state.T)
        ocean = np.array(grid.mask_c, dtype=bool)
        assert float(T[ocean].min()) > 5.0,  "T too low"
        assert float(T[ocean].max()) < 25.0, "T too high"


# ---------------------------------------------------------------------------
# TestFieldSpecificDepth — depthu/depthv distinct from deptht
# ---------------------------------------------------------------------------

# Depth levels that intentionally differ for T vs u
_DEPTH_T = np.array([10.0, 50.0, 200.0], dtype=np.float64)
_DEPTH_U = np.array([20.0, 100.0, 300.0], dtype=np.float64)   # different from _DEPTH_T


def _write_nemo_diff_depth_nc(path) -> None:
    """
    NEMO-style file where ``vozocrtx`` (u) lives on ``depthu`` while
    ``votemper``/``vosaline`` live on ``deptht``.  The two depth axes have
    different values so ``read_oras5`` must return non-None ``depth_u``.
    """
    Nzt = len(_DEPTH_T)
    Nzu = len(_DEPTH_U)
    Ny, Nx = _NY_SRC, _NX_SRC

    lon2d = np.tile(_SRC_LON, (Ny, 1)).astype(np.float64)
    lat2d = np.tile(_SRC_LAT[:, np.newaxis], (1, Nx)).astype(np.float64)

    T_data = np.zeros((1, Nzt, Ny, Nx), dtype=np.float32)
    for k in range(Nzt):
        T_data[0, k] = 20.0 - k * 5.0
    S_data = np.full((1, Nzt, Ny, Nx), 35.0, dtype=np.float32)
    u_data = np.zeros((1, Nzu, Ny, Nx), dtype=np.float32)

    ds = xr.Dataset(
        {
            "votemper": (("time_counter", "deptht", "y", "x"), T_data),
            "vosaline": (("time_counter", "deptht", "y", "x"), S_data),
            "vozocrtx": (("time_counter", "depthu", "y", "x"), u_data),
        },
        coords={
            "nav_lon":      (("y", "x"), lon2d),
            "nav_lat":      (("y", "x"), lat2d),
            "deptht":       ("deptht",   _DEPTH_T),
            "depthu":       ("depthu",   _DEPTH_U),
            "time_counter": ("time_counter", np.array([0.0])),
        },
    )
    ds.to_netcdf(path)


class TestFieldSpecificDepth:
    """
    ``read_oras5`` / ``regrid_to_model`` when u uses a different depth axis
    (``depthu``) than T/S (``deptht``).
    """

    def test_depth_u_extracted(self, tmp_path):
        """``depth_u`` must be non-None and equal to _DEPTH_U."""
        nc = tmp_path / "diff_depth.nc"
        _write_nemo_diff_depth_nc(nc)
        raw = read_oras5(nc)
        assert raw["depth_u"] is not None, "depth_u should be extracted from depthu"
        np.testing.assert_allclose(raw["depth_u"], np.sort(_DEPTH_U))

    def test_depth_t_unchanged(self, tmp_path):
        """T-grid depth (``deptht``) must appear in ``raw['depth']``."""
        nc = tmp_path / "diff_depth_t.nc"
        _write_nemo_diff_depth_nc(nc)
        raw = read_oras5(nc)
        np.testing.assert_allclose(raw["depth"], np.sort(_DEPTH_T))

    def test_depth_v_none_when_absent(self, tmp_path):
        """No ``depthv`` variable in the file → ``depth_v`` must be None."""
        nc = tmp_path / "diff_depth_nv.nc"
        _write_nemo_diff_depth_nc(nc)
        raw = read_oras5(nc)
        assert raw["depth_v"] is None, "depth_v should be None when v is absent"

    def test_depth_u_none_for_regular_grid(self, tmp_path):
        """Regular (non-NEMO) file with shared depth dim → ``depth_u`` must be None."""
        nc = tmp_path / "reg_depth.nc"
        _write_nc(nc, u_name="uo")
        raw = read_oras5(nc)
        assert raw["depth_u"] is None, "depth_u should be None when depth dim is shared"

    def test_regrid_no_nan(self, tmp_path):
        """``regrid_to_model`` must produce no NaN when u uses a different depth."""
        nc = tmp_path / "diff_depth_rg.nc"
        _write_nemo_diff_depth_nc(nc)
        grid  = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)
        assert not np.any(np.isnan(np.array(state.T))), "NaN in T"
        assert not np.any(np.isnan(np.array(state.u))), "NaN in u"

    def test_regrid_u_shape(self, tmp_path):
        """Output u shape must match the target grid even when src u has Nzu levels."""
        nc = tmp_path / "diff_depth_sh.nc"
        _write_nemo_diff_depth_nc(nc)
        grid  = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)
        assert state.u.shape == (grid.Nx, grid.Ny, grid.Nz)


# ---------------------------------------------------------------------------
# TestFieldSpecificHoriz — staggered nav_lon_u / nav_lat_u
# ---------------------------------------------------------------------------

_STAGGER_DEG = 0.125   # u-grid stagger offset in longitude


def _write_nemo_stag_horiz_nc(path) -> None:
    """
    NEMO-style file that contains explicit ``nav_lon_u`` / ``nav_lat_u``
    staggered coordinates (+0.125° in longitude relative to T-grid).
    The u variable itself is stored on the same (y, x) spatial dimensions
    as T; the staggered coords are metadata about where those u-points sit.
    """
    Ny, Nx = _NY_SRC, _NX_SRC

    lon2d   = np.tile(_SRC_LON, (Ny, 1)).astype(np.float64)
    lat2d   = np.tile(_SRC_LAT[:, np.newaxis], (1, Nx)).astype(np.float64)
    lon_u2d = lon2d + _STAGGER_DEG    # u-grid: staggered +0.125° in lon
    lat_u2d = lat2d                   # u-grid: same lat as T

    T_data = _make_T()[np.newaxis]
    S_data = _make_S()[np.newaxis]
    u_data = np.zeros_like(T_data)

    ds = xr.Dataset(
        {
            "votemper": (("time_counter", "deptht", "y", "x"), T_data),
            "vosaline": (("time_counter", "deptht", "y", "x"), S_data),
            "vozocrtx": (("time_counter", "deptht", "y", "x"), u_data),
        },
        coords={
            "nav_lon":      (("y", "x"), lon2d),
            "nav_lat":      (("y", "x"), lat2d),
            "nav_lon_u":    (("y", "x"), lon_u2d),
            "nav_lat_u":    (("y", "x"), lat_u2d),
            "deptht":       ("deptht",   _SRC_DEPTH),
            "time_counter": ("time_counter", np.array([0.0])),
        },
    )
    ds.to_netcdf(path)


class TestFieldSpecificHoriz:
    """
    ``read_oras5`` / ``regrid_to_model`` when explicit staggered horizontal
    coordinate arrays (``nav_lon_u`` / ``nav_lat_u``) are present.
    """

    def test_lon_u_extracted(self, tmp_path):
        """``lon_u`` must be non-None and differ from ``lon`` by ~_STAGGER_DEG."""
        nc = tmp_path / "stag_horiz.nc"
        _write_nemo_stag_horiz_nc(nc)
        raw = read_oras5(nc)
        assert raw["lon_u"] is not None, "lon_u should be extracted from nav_lon_u"
        delta = raw["lon_u"] - raw["lon"]
        np.testing.assert_allclose(delta, _STAGGER_DEG, atol=1e-9)

    def test_lat_u_same_as_t(self, tmp_path):
        """``lat_u`` must equal the T-grid lat (no lat stagger in this fixture)."""
        nc = tmp_path / "stag_horiz_lat.nc"
        _write_nemo_stag_horiz_nc(nc)
        raw = read_oras5(nc)
        assert raw["lat_u"] is not None
        np.testing.assert_allclose(raw["lat_u"], raw["lat"], atol=1e-9)

    def test_lon_u_none_regular_grid(self, tmp_path):
        """Regular-grid file has no nav_lon_u → ``lon_u`` must be None."""
        nc = tmp_path / "reg_stag.nc"
        _write_nc(nc, u_name="uo")
        raw = read_oras5(nc)
        assert raw["lon_u"] is None, "lon_u should be None for regular-grid file"

    def test_lon_v_lat_v_none_when_absent(self, tmp_path):
        """File has nav_lon_u/nav_lat_u but not nav_lon_v/nav_lat_v → v coords None."""
        nc = tmp_path / "stag_uonly.nc"
        _write_nemo_stag_horiz_nc(nc)
        raw = read_oras5(nc)
        assert raw["lon_v"] is None
        assert raw["lat_v"] is None

    def test_regrid_no_nan_staggered(self, tmp_path):
        """``regrid_to_model`` must produce no NaN when u uses staggered coords."""
        nc = tmp_path / "stag_rg.nc"
        _write_nemo_stag_horiz_nc(nc)
        grid  = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)
        assert not np.any(np.isnan(np.array(state.T))), "NaN in T"
        assert not np.any(np.isnan(np.array(state.u))), "NaN in u"

    def test_regrid_output_shape_staggered(self, tmp_path):
        """Output shape must match the target grid when staggered u coords are used."""
        nc = tmp_path / "stag_shape.nc"
        _write_nemo_stag_horiz_nc(nc)
        grid  = _target_grid()
        state = regrid_to_model(read_oras5(nc), grid)
        assert state.u.shape == (grid.Nx, grid.Ny, grid.Nz)


# ---------------------------------------------------------------------------
# oras5_bathymetry — land/sea mask and water depth from ORAS5 NaN pattern
# ---------------------------------------------------------------------------

_BATHY_DEPTH = np.array([5.0, 15.0, 30.0, 60.0, 100.0])
# Faces by the OceanGrid.create rule: 0, 10, 22.5, 45, 80, 120


def _bathy_raw(curvilinear: bool = False, lon0: float = 0.0) -> dict:
    """
    Synthetic 0.25° source on lon [lon0, lon0+10], lat [0, 8]:
      lon < lon0+3.5           : land (all NaN)
      lat < 4                  : wet in the top 3 levels  -> depth 45 m
      lat >= 4                 : wet in all 5 levels      -> depth 120 m
    One column also has a wet level *below* a dry one, which must not
    count (wetness is taken contiguously from the surface).
    """
    lon = lon0 + np.arange(0.125, 10.0, 0.25)
    lat = np.arange(0.125, 8.0, 0.25)
    T = np.full((len(_BATHY_DEPTH), len(lat), len(lon)), 20.0, dtype=np.float32)
    T[:, :, lon < lon0 + 3.5] = np.nan
    T[3:, lat < 4.0, :] = np.nan
    T[4, lat < 4.0, -1] = 20.0          # isolated wet cell under a dry level
    raw = {"T": T, "depth": _BATHY_DEPTH.copy()}
    if curvilinear:
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        raw.update(lon=lon2d, lat=lat2d)
    else:
        raw.update(lon=lon, lat=lat)
    return raw


def _bathy_grid(Nx=5, Ny=4, lon=(0.0, 10.0)) -> OceanGrid:
    return OceanGrid.create(lon, (0.0, 8.0), np.array([5.0, 30.0, 70.0]), Nx, Ny)


class TestOras5Bathymetry:

    @pytest.mark.parametrize("curvilinear", [False, True])
    def test_land_and_depths(self, curvilinear):
        """2°×2° cells: majority-land cells are land, depths follow ORAS5."""
        H = oras5_bathymetry(_bathy_raw(curvilinear), _bathy_grid())
        assert H.shape == (5, 4)
        # i=0 (lon 0-2) all land; i=1 (lon 2-4) 6/8 land -> land
        assert np.all(H[:2] == 0.0)
        # ocean columns: lat 0-4 -> 45 m, lat 4-8 -> 120 m
        np.testing.assert_allclose(H[2:, :2], 45.0)
        np.testing.assert_allclose(H[2:, 2:], 120.0)

    def test_grid_mask_from_bathymetry(self):
        """Feeding H back into OceanGrid.create gives the expected mask."""
        H    = oras5_bathymetry(_bathy_raw(), _bathy_grid())
        grid = OceanGrid.create((0.0, 10.0), (0.0, 8.0), np.array([5.0, 30.0, 70.0]),
                                5, 4, bathymetry=H, periodic_x=False)
        m = np.asarray(grid.mask_c)
        assert np.all(m[:2] == 0.0)                        # land columns
        # model faces 0, 17.5, 50, 90: H=45 -> 2 wet levels, H=120 -> 3
        np.testing.assert_array_equal(m[2:, :2].sum(axis=-1), 2)
        np.testing.assert_array_equal(m[2:, 2:].sum(axis=-1), 3)

    def test_finer_target_uses_nearest(self):
        """Target cells smaller than the source spacing get no NaN."""
        H = oras5_bathymetry(_bathy_raw(), _bathy_grid(Nx=80, Ny=64))
        assert np.all(np.isfinite(H))
        assert H[0, 0] == 0.0 and H[-1, -1] == 120.0

    def test_longitude_convention_mismatch(self):
        """Source in 0-360, target in -180-180 (same physical region)."""
        H = oras5_bathymetry(_bathy_raw(lon0=340.0), _bathy_grid(lon=(-20.0, -10.0)))
        assert np.all(H[:2] == 0.0)
        np.testing.assert_allclose(H[2:, 2:], 120.0)


# ---------------------------------------------------------------------------
# oras5_levels — model levels taken from the ORAS5 vertical grid
# ---------------------------------------------------------------------------

class TestOras5Levels:

    # Roughly ORAS5-like: ~1 m spacing at the top, stretching with depth
    DEPTHS = np.concatenate([np.arange(0.5, 10.0, 1.0),
                             10.0 * 1.15 ** np.arange(1, 40)])

    def test_min_thickness_and_bottom(self):
        from OceanJAX.data.oras5 import oras5_levels
        from OceanJAX.grid import face_depths
        c  = oras5_levels(self.DEPTHS, depth_max=2000.0, dz_min=10.0)
        f  = face_depths(c)
        assert np.diff(f).min() >= 10.0 - 1e-9
        assert f[-1] <= 2000.0
        assert set(c).issubset(set(self.DEPTHS)), "levels are ORAS5 levels"
        # the next deeper ORAS5 level would have exceeded depth_max
        deeper = self.DEPTHS[self.DEPTHS > c[-1]]
        assert face_depths(np.append(c, deeper[0]))[-1] > 2000.0

    def test_accepts_read_oras5_dict(self):
        from OceanJAX.data.oras5 import oras5_levels
        a = oras5_levels(self.DEPTHS, 1000.0, 10.0)
        b = oras5_levels({"depth": self.DEPTHS}, 1000.0, 10.0)
        np.testing.assert_array_equal(a, b)

    def test_nothing_fits(self):
        from OceanJAX.data.oras5 import oras5_levels
        with pytest.raises(ValueError):
            oras5_levels(self.DEPTHS, depth_max=5.0, dz_min=10.0)


# ---------------------------------------------------------------------------
# Per-variable monthly files: read_oras5(list) and oras5_month_files
# ---------------------------------------------------------------------------

def _write_month_split(directory, ym="202602", merged_name=None):
    """A merged NEMO file with non-trivial u/v/eta, optionally split per variable."""
    merged = directory / "tmp_merged.nc"
    _write_nemo_nc(merged, u_name="vozocrtx", v_name="vomecrty", eta_name="sossheig")
    with xr.open_dataset(merged) as src:           # close before unlink (Windows)
        ds = src.load()
    rng = np.random.default_rng(0)
    for v in ("vozocrtx", "vomecrty", "sossheig"):
        ds[v] = ds[v] + rng.normal(size=ds[v].shape).astype(np.float32)
    ds.to_netcdf(directory / (merged_name or "merged.nc"))
    for v in ("votemper", "vosaline", "vozocrtx", "vomecrty", "sossheig"):
        kind = "2D" if v == "sossheig" else "3D"
        ds[[v]].to_netcdf(directory / f"{v}_control_monthly_highres_{kind}_{ym}_OPER_v0.1.nc")
    merged.unlink()
    return directory / (merged_name or "merged.nc")


class TestMonthFiles:

    def test_split_files_read_like_merged(self, tmp_path):
        from OceanJAX.data.oras5 import oras5_month_files
        merged = _write_month_split(tmp_path)
        a = read_oras5(merged)
        b = read_oras5(oras5_month_files(tmp_path, 2026, 2))
        for k in ("T", "S", "u", "v", "eta", "lon", "lat", "depth"):
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)

    def test_prefers_merged_file(self, tmp_path):
        from OceanJAX.data.oras5 import oras5_month_files
        _write_month_split(tmp_path, merged_name="oras5_2026_02_native_merged.nc")
        files = oras5_month_files(tmp_path, 2026, 2)
        assert [f.name for f in files] == ["oras5_2026_02_native_merged.nc"]

    def test_missing_velocity_warns(self, tmp_path):
        from OceanJAX.data.oras5 import oras5_month_files
        _write_month_split(tmp_path)
        next(tmp_path.glob("vomecrty_*")).unlink()
        with pytest.warns(UserWarning, match="vomecrty"):
            files = oras5_month_files(tmp_path, 2026, 2)
        assert len(files) == 4
        assert read_oras5(files)["v"] is None

    def test_missing_temperature_raises(self, tmp_path):
        from OceanJAX.data.oras5 import oras5_month_files
        _write_month_split(tmp_path)
        next(tmp_path.glob("votemper_*")).unlink()
        with pytest.raises(FileNotFoundError):
            oras5_month_files(tmp_path, 2026, 2)
