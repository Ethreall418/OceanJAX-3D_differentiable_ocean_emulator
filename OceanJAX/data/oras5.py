"""
OceanJAX ORAS5 data loader
===========================
Loads the Copernicus Marine Service ORAS5 ocean reanalysis product
(GLOBAL_MULTIYEAR_PHY_001_030) and regrids it onto an OceanJAX model grid.

Supported formats
-----------------
1. **Regular 0.25° product** (Copernicus ``GLOBAL_MULTIYEAR_PHY_001_030``):
   1-D ``longitude`` / ``latitude`` / ``depth`` coordinate variables.
2. **Native NEMO/ORCA curvilinear grid**: 2-D ``nav_lon`` / ``nav_lat``
   coordinates on ``(y, x)`` dimensions, ``time_counter`` time axis,
   ``deptht`` depth axis.  Both formats are detected automatically.

For curvilinear grids the horizontal step uses ``LinearNDInterpolator``
on a Delaunay triangulation built once per call (expensive for large
domains; subsetting source data to the target region is recommended).

Variable-name aliasing
----------------------
Copernicus variable names vary slightly across ORAS5 product versions.
The reader resolves names through a priority-ordered alias table so that
``thetao`` / ``thetao_oras`` / ``votemper`` are all accepted transparently.

Two-layer design
----------------
``read_oras5()``      — pure I/O: opens the file, extracts one time slice,
                        returns a plain dict of numpy arrays.  No JAX
                        dependency; easy to test with synthetic NetCDF files.
``regrid_to_model()`` — interpolation and masking: takes that dict and an
                        OceanGrid, returns a masked OceanState.
``load_oras5()``      — convenience wrapper that calls both layers.

Target resolution
-----------------
ORAS5 source resolution is ~0.25°.  Regridding onto a *finer* target grid
invents spurious fine-scale structure; a ``UserWarning`` is issued but
execution is not blocked.

Missing-value strategy (three-tier, NaN-aware)
-----------------------------------------------
1. Normalised convolution — data and validity mask are interpolated
   separately (data_sum / mask_sum), so coast-adjacent ocean points
   are never contaminated by fill values.
2. ``NearestNDInterpolator`` on valid source points only — fills any
   target point whose entire linear stencil is NaN (e.g. isolated
   deep cells below the bathymetry).
3. Constant ``T_fill`` / ``S_fill`` / ``0`` as unconditional fallback.

Vertical extrapolation
-----------------------
  T, S — nearest valid source level (handled automatically by tier 2).
  u, v — explicitly zeroed below the deepest ORAS5 level (extrapolating
          bottom velocities downward is physically wrong).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import xarray as xr
from scipy.interpolate import (
    RegularGridInterpolator,
    NearestNDInterpolator,
    LinearNDInterpolator,
)
from scipy.spatial import Delaunay

from OceanJAX.grid import OceanGrid, face_depths
from OceanJAX.state import OceanState, create_from_arrays


# ---------------------------------------------------------------------------
# Alias tables
# ---------------------------------------------------------------------------

_VAR_ALIASES: dict[str, list[str]] = {
    # Ocean state
    "T":        ["thetao", "thetao_oras", "votemper", "temperature"],
    "S":        ["so",     "so_oras",     "vosaline",  "salinity"],
    "u":        ["uo",     "uo_oras",     "vozocrtx",  "u"],
    "v":        ["vo",     "vo_oras",     "vomecrty",  "v"],
    "eta":      ["zos",    "zos_oras",    "sossheig",  "ssh"],
    # Surface forcing (2-D, no depth axis)
    "heat_flux": ["sohefldo", "qnet", "netheatflux", "hfds"],
    "fw_flux":   ["sowaflup", "wfo",  "freshwaterflux", "emp"],
    "tau_x":     ["sozotaux", "tauuo", "utau", "tauu"],
    "tau_y":     ["sometauy", "tauvo", "vtau", "tauv"],
}

_COORD_ALIASES: dict[str, list[str]] = {
    "lon":   ["longitude", "lon", "x", "nav_lon"],
    "lat":   ["latitude",  "lat", "y", "nav_lat"],
    "depth": ["depth", "deptht", "level", "lev", "z"],
}

# Dimension names recognised as depth axes when checking per-field coordinates.
# Wider than _COORD_ALIASES["depth"] to include depthu/depthv/depthw used by
# NEMO velocity components.
_DEPTH_DIM_NAMES: frozenset[str] = frozenset(
    {"depth", "deptht", "depthu", "depthv", "depthw", "level", "lev", "z"}
)


def _find_var(ds: xr.Dataset, key: str, optional: bool = False) -> Optional[str]:
    """Return the first matching variable name, or None if optional and absent."""
    for name in _VAR_ALIASES[key]:
        if name in ds:
            return name
    if optional:
        return None
    raise KeyError(
        f"Cannot find variable '{key}' in dataset.  "
        f"Tried: {_VAR_ALIASES[key]}.  "
        f"Available data variables: {sorted(ds.data_vars)}"
    )


def _find_coord(ds: xr.Dataset, key: str) -> str:
    """Return the first matching coordinate name (checks both coords and data_vars)."""
    for name in _COORD_ALIASES[key]:
        if name in ds.coords or name in ds:
            return name
    raise KeyError(
        f"Cannot find coordinate '{key}' in dataset.  "
        f"Tried: {_COORD_ALIASES[key]}.  "
        f"Available coords: {sorted(ds.coords)}"
    )


def _ensure_ascending(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_arr, index_order) such that sorted_arr is strictly ascending."""
    order = np.argsort(arr)
    return arr[order], order


# ---------------------------------------------------------------------------
# Public: read_oras5
# ---------------------------------------------------------------------------

def read_oras5(
    path: str | Path | list,
    time_index: int = 0,
) -> dict[str, Optional[np.ndarray]]:
    """
    Read one time slice from a Copernicus Marine ORAS5 NetCDF file.

    Both the regular 0.25° lat-lon product (1-D coordinate variables) and the
    native NEMO/ORCA curvilinear grid (2-D ``nav_lon`` / ``nav_lat``,
    ``time_counter`` time axis, per-field ``deptht`` / ``depthu`` / ``depthv``
    depth axes) are accepted and detected automatically.

    Parameters
    ----------
    path : str, Path, or list of them
        NetCDF file containing at least temperature and salinity, or several
        files of the same month that together do (e.g. the per-variable
        ORAS5 files votemper_*, vosaline_*, vozocrtx_*, vomecrty_*,
        sossheig_*; see ``oras5_month_files``).  Several files are merged
        lazily in memory, exactly as ``data_merger.py`` merges them on disk.
    time_index : int
        Index along the time dimension to read (default 0).

    Returns
    -------
    dict with the following fixed keys.  Optional fields are **always**
    ``None`` when absent — never a missing key or an empty array:

    ``"T"``       : ``(Nz_src, Ny_src, Nx_src)`` float32 — potential temperature [°C]
    ``"S"``       : ``(Nz_src, Ny_src, Nx_src)`` float32 — practical salinity [psu]
    ``"u"``       : ``(Nz_src, Ny_src, Nx_src)`` float32, or ``None``
    ``"v"``       : ``(Nz_src, Ny_src, Nx_src)`` float32, or ``None``
    ``"eta"``     : ``(Ny_src, Nx_src)``          float32, or ``None``
    ``"lon"``     : ``(Nx_src,)`` float64 — regular grid (degrees east, ascending).
                    ``(Ny_src, Nx_src)`` float64 — curvilinear T-grid.
    ``"lat"``     : ``(Ny_src,)`` float64 — regular grid (degrees north, ascending).
                    ``(Ny_src, Nx_src)`` float64 — curvilinear T-grid.
    ``"depth"``   : ``(Nz_src,)`` float64 — T-grid depth, metres positive downward,
                    ascending.

    Per-field coordinate keys (all ``None`` when absent or identical to T-grid):

    ``"depth_u"`` : ``(Nz_u,)`` float64 — u-field depth levels, or ``None`` if
                    identical to ``"depth"`` (e.g. Copernicus regular product,
                    or merged NEMO files with equal ``deptht``/``depthu``).
    ``"depth_v"`` : ``(Nz_v,)`` float64 — v-field depth levels, or ``None``.
    ``"lon_u"``   : ``(Ny_src, Nx_src)`` float64 — u horizontal positions, or ``None``
                    (present only when the file contains an explicit staggered
                    coordinate such as ``nav_lon_u`` / ``glamu``).
    ``"lat_u"``   : ``(Ny_src, Nx_src)`` float64 — u latitude positions, or ``None``.
    ``"lon_v"``   : ``(Ny_src, Nx_src)`` float64 — v horizontal positions, or ``None``.
    ``"lat_v"``   : ``(Ny_src, Nx_src)`` float64 — v latitude positions, or ``None``.

    Land / fill-value cells are returned as ``np.nan`` (xarray's
    ``mask_and_scale=True`` handles ``_FillValue`` / ``missing_value``
    automatically).
    """
    paths = list(path) if isinstance(path, (list, tuple)) else [path]
    parts = [xr.open_dataset(p, mask_and_scale=True) for p in paths]
    ds = parts[0] if len(parts) == 1 else xr.merge(
        [d.drop_vars("time_counter_bnds", errors="ignore") for d in parts],
        compat="override",
    )
    try:
        # --- coordinates ---
        lon_name   = _find_coord(ds, "lon")
        lat_name   = _find_coord(ds, "lat")
        depth_name = _find_coord(ds, "depth")

        lon   = np.asarray(ds[lon_name].values,   dtype=np.float64)
        lat   = np.asarray(ds[lat_name].values,   dtype=np.float64)
        depth = np.asarray(ds[depth_name].values, dtype=np.float64)

        if depth.ndim != 1:
            raise ValueError(
                f"depth coordinate must be 1-D; found shape {depth.shape}."
            )

        # Detect curvilinear (NEMO/ORCA) vs. regular grid
        curvilinear = lon.ndim != 1 or lat.ndim != 1

        # Ensure depth is positive downward
        if np.any(depth < 0):
            depth = -depth

        # Always sort depth to strictly ascending order
        depth, dep_idx = _ensure_ascending(depth)

        # For regular grids also sort lat/lon
        lon_idx = lat_idx = None
        if not curvilinear:
            lon, lon_idx = _ensure_ascending(lon)
            lat, lat_idx = _ensure_ascending(lat)

        # --- time-dimension helper (accepts both "time" and "time_counter") ---
        def _drop_time(da):
            time_dim = next(
                (d for d in da.dims if "time" in d.lower()), None
            )
            if time_dim is not None:
                da = da.isel({time_dim: time_index})
            return da

        # --- load and reorder a 3-D variable ---
        # Axis order is determined explicitly from da.dims so that files whose
        # dimensions arrive in non-canonical order are handled correctly.
        _lat_dim_names = frozenset(_COORD_ALIASES["lat"])
        _lon_dim_names = frozenset(_COORD_ALIASES["lon"])

        def _load_3d(var_name: str, field_dep_idx=None) -> np.ndarray:
            _di = dep_idx if field_dep_idx is None else field_dep_idx
            da = _drop_time(ds[var_name])
            if da.ndim != 3:
                raise ValueError(
                    f"Variable '{var_name}' has {da.ndim} dimensions after "
                    f"time selection; expected 3 (depth, spatial1, spatial2). "
                    f"Dims: {da.dims}"
                )

            # Identify the depth axis for this variable by dim name
            var_depth_dim = next(
                (d for d in da.dims
                 if d.lower() in _DEPTH_DIM_NAMES and "time" not in d.lower()),
                None,
            )

            if curvilinear:
                # Reorder to (depth_dim, y, x); keep y/x in their original order.
                if var_depth_dim is not None and da.dims[0] != var_depth_dim:
                    spatial = [d for d in da.dims if d != var_depth_dim]
                    da = da.transpose(var_depth_dim, *spatial)
                arr = np.asarray(da.values, dtype=np.float32)
                return arr[_di, :, :]
            else:
                # Reorder to (depth_dim, lat_dim, lon_dim) by name.
                lat_dim = next((d for d in da.dims if d in _lat_dim_names), None)
                lon_dim = next((d for d in da.dims if d in _lon_dim_names), None)
                if var_depth_dim is not None and lat_dim is not None and lon_dim is not None:
                    da = da.transpose(var_depth_dim, lat_dim, lon_dim)
                elif var_depth_dim is not None and da.dims[0] != var_depth_dim:
                    # Depth is not first but lat/lon dims are unrecognised — move depth to front
                    spatial = [d for d in da.dims if d != var_depth_dim]
                    da = da.transpose(var_depth_dim, *spatial)
                arr = np.asarray(da.values, dtype=np.float32)
                return arr[np.ix_(_di, lat_idx, lon_idx)]

        # --- load and reorder a 2-D variable ---
        def _load_2d(var_name: str) -> np.ndarray:
            da = _drop_time(ds[var_name])
            if da.ndim != 2:
                raise ValueError(
                    f"Variable '{var_name}' has {da.ndim} dimensions after "
                    f"time selection; expected 2 (lat/y, lon/x). "
                    f"Dims: {da.dims}"
                )

            if curvilinear:
                # Spatial dims are named y/x — keep original order (y, x).
                arr = np.asarray(da.values, dtype=np.float32)
                return arr
            else:
                # Reorder to (lat_dim, lon_dim) by name.
                lat_dim = next((d for d in da.dims if d in _lat_dim_names), None)
                lon_dim = next((d for d in da.dims if d in _lon_dim_names), None)
                if lat_dim is not None and lon_dim is not None:
                    da = da.transpose(lat_dim, lon_dim)
                arr = np.asarray(da.values, dtype=np.float32)
                return arr[np.ix_(lat_idx, lon_idx)]

        T = _load_3d(_find_var(ds, "T"))
        S = _load_3d(_find_var(ds, "S"))

        u_name   = _find_var(ds, "u",   optional=True)
        v_name   = _find_var(ds, "v",   optional=True)
        eta_name = _find_var(ds, "eta", optional=True)

        # --- per-field depth (e.g. depthu/depthv may differ from deptht) ---
        def _field_depth(var_name_f):
            """Return (depth_arr, dep_idx_f) for var_name_f, or (None, dep_idx)."""
            if var_name_f is None or var_name_f not in ds:
                return None, dep_idx
            da_f = ds[var_name_f]
            field_depth_dim = next(
                (d for d in da_f.dims
                 if d.lower() in _DEPTH_DIM_NAMES and "time" not in d.lower()),
                None,
            )
            if field_depth_dim is None or field_depth_dim == depth_name:
                return None, dep_idx
            arr = np.asarray(ds[field_depth_dim].values, dtype=np.float64)
            if np.any(arr < 0):
                arr = -arr
            arr_sorted, order = _ensure_ascending(arr)
            if arr_sorted.shape == depth.shape and np.allclose(arr_sorted, depth):
                return None, dep_idx
            return arr_sorted, order

        depth_u, u_dep_idx = _field_depth(u_name)
        depth_v, v_dep_idx = _field_depth(v_name)

        # --- per-field horizontal coordinates (staggered grids in separate files) ---
        _STAG_ALIASES = {
            "lon_u": ["nav_lon_u", "glamu"],
            "lat_u": ["nav_lat_u", "gphiu"],
            "lon_v": ["nav_lon_v", "glamv"],
            "lat_v": ["nav_lat_v", "gphiv"],
        }

        def _find_stag_coord(aliases):
            for a in aliases:
                if a in ds.coords or a in ds:
                    return np.asarray(ds[a].values, dtype=np.float64)
            return None

        lon_u = _find_stag_coord(_STAG_ALIASES["lon_u"])
        lat_u = _find_stag_coord(_STAG_ALIASES["lat_u"])
        lon_v = _find_stag_coord(_STAG_ALIASES["lon_v"])
        lat_v = _find_stag_coord(_STAG_ALIASES["lat_v"])

        u   = _load_3d(u_name,   u_dep_idx) if u_name   is not None else None
        v   = _load_3d(v_name,   v_dep_idx) if v_name   is not None else None
        eta = _load_2d(eta_name)            if eta_name is not None else None

    finally:
        ds.close()
        for d in parts:
            d.close()

    return {
        "T": T, "S": S, "u": u, "v": v, "eta": eta,
        "lon": lon, "lat": lat, "depth": depth,
        "depth_u": depth_u, "depth_v": depth_v,
        "lon_u": lon_u, "lat_u": lat_u,
        "lon_v": lon_v, "lat_v": lat_v,
    }


# ---------------------------------------------------------------------------
# Private: interpolation helpers
# ---------------------------------------------------------------------------

def _normalized_conv_3d(
    src:       np.ndarray,   # (Nz, Ny, Nx), may contain NaN
    src_depth: np.ndarray,
    src_lat:   np.ndarray,
    src_lon:   np.ndarray,
    tgt_depth: np.ndarray,
    tgt_lat:   np.ndarray,
    tgt_lon:   np.ndarray,
) -> np.ndarray:
    """
    NaN-aware linear interpolation via normalised convolution.

    Instead of replacing NaN with fill_value (which contaminates coast-adjacent
    ocean points), we interpolate the data and the validity mask separately:

        data_sum  = RGI_linear(NaN → 0)
        mask_sum  = RGI_linear(valid=1, NaN=0)
        out       = data_sum / mask_sum   where mask_sum > eps

    Points where mask_sum ≤ eps receive NaN and are handled by the caller.
    """
    valid      = (~np.isnan(src)).astype(np.float64)
    src_filled = np.where(np.isnan(src), 0.0, src).astype(np.float64)

    dz, dy, dx = np.meshgrid(tgt_depth, tgt_lat, tgt_lon, indexing="ij")
    pts = np.stack([dz.ravel(), dy.ravel(), dx.ravel()], axis=-1)
    out_shape  = (len(tgt_depth), len(tgt_lat), len(tgt_lon))

    rgi_data = RegularGridInterpolator(
        (src_depth, src_lat, src_lon), src_filled,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    rgi_mask = RegularGridInterpolator(
        (src_depth, src_lat, src_lon), valid,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    data_sum = rgi_data(pts).reshape(out_shape)
    mask_sum = rgi_mask(pts).reshape(out_shape)

    eps = 1e-6
    out = np.where(mask_sum > eps, data_sum / np.maximum(mask_sum, eps), np.nan)
    return out


def _normalized_conv_2d(
    src:     np.ndarray,   # (Ny, Nx), may contain NaN
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
) -> np.ndarray:
    """2-D version of ``_normalized_conv_3d``."""
    valid      = (~np.isnan(src)).astype(np.float64)
    src_filled = np.where(np.isnan(src), 0.0, src).astype(np.float64)

    dy, dx = np.meshgrid(tgt_lat, tgt_lon, indexing="ij")
    pts = np.stack([dy.ravel(), dx.ravel()], axis=-1)
    out_shape = (len(tgt_lat), len(tgt_lon))

    rgi_data = RegularGridInterpolator(
        (src_lat, src_lon), src_filled,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    rgi_mask = RegularGridInterpolator(
        (src_lat, src_lon), valid,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    data_sum = rgi_data(pts).reshape(out_shape)
    mask_sum = rgi_mask(pts).reshape(out_shape)

    eps = 1e-6
    out = np.where(mask_sum > eps, data_sum / np.maximum(mask_sum, eps), np.nan)
    return out


def _interp_3d(
    src:        np.ndarray,
    src_lon:    np.ndarray,
    src_lat:    np.ndarray,
    src_depth:  np.ndarray,
    tgt_lon:    np.ndarray,
    tgt_lat:    np.ndarray,
    tgt_depth:  np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """
    Interpolate ``(Nz_src, Ny_src, Nx_src)`` → ``(Nz_tgt, Ny_tgt, Nx_tgt)``.

    Three-tier NaN strategy:
    1. Normalised convolution (NaN-aware linear) — no coastal contamination.
    2. Per-level 2-D ``NearestNDInterpolator`` in (lat, lon) only, querying
       the nearest source depth level — fills points where the linear stencil
       is entirely NaN without mixing depth (m) with lat/lon (°) in the
       distance metric.
    3. Constant ``fill_value`` as final fallback.
    """
    out = _normalized_conv_3d(
        src, src_depth, src_lat, src_lon,
        tgt_depth, tgt_lat, tgt_lon,
    )

    needs_nn = np.isnan(out)
    if np.any(needs_nn):
        # Per-level 2-D nearest-neighbour in (lat, lon) only.
        # A 3-D NN mixing depth (metres) with lat/lon (degrees) would produce
        # meaningless distances and borrow values from the wrong depth level.
        # Instead, for each target level we find the nearest *source* depth
        # level and run a 2-D NearestNDInterpolator on its valid ocean points.
        # For target levels below src_depth.max() this naturally selects the
        # deepest source level, giving T/S extrapolation by the bottom value.
        dy_src, dx_src = np.meshgrid(src_lat, src_lon, indexing="ij")
        dy_tgt, dx_tgt = np.meshgrid(tgt_lat, tgt_lon, indexing="ij")
        tgt_pts_2d = np.stack([dy_tgt.ravel(), dx_tgt.ravel()], axis=-1)

        nn_fill = np.empty_like(out)
        for k in range(len(tgt_depth)):
            k_src   = int(np.argmin(np.abs(src_depth - tgt_depth[k])))
            valid_k = ~np.isnan(src[k_src])          # (Ny_src, Nx_src)

            if not np.any(valid_k):
                nn_fill[k] = np.nan      # tier-3 constant fallback handles this
                continue

            interp_nn = NearestNDInterpolator(
                np.stack([dy_src[valid_k].ravel(), dx_src[valid_k].ravel()], axis=-1),
                src[k_src][valid_k].ravel(),
            )
            nn_fill[k] = interp_nn(tgt_pts_2d).reshape(len(tgt_lat), len(tgt_lon))

        out = np.where(needs_nn, nn_fill, out)

    still_nan = np.isnan(out)
    if np.any(still_nan):
        out = np.where(still_nan, fill_value, out)

    return out.astype(np.float32)


def _interp_2d(
    src:        np.ndarray,
    src_lon:    np.ndarray,
    src_lat:    np.ndarray,
    tgt_lon:    np.ndarray,
    tgt_lat:    np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """
    Interpolate ``(Ny_src, Nx_src)`` → ``(Ny_tgt, Nx_tgt)``.
    Same three-tier NaN strategy as ``_interp_3d``.
    """
    out = _normalized_conv_2d(src, src_lat, src_lon, tgt_lat, tgt_lon)

    needs_nn = np.isnan(out)
    if np.any(needs_nn):
        dy_src, dx_src = np.meshgrid(src_lat, src_lon, indexing="ij")
        valid_mask = ~np.isnan(src)
        pts_valid  = np.stack([dy_src[valid_mask].ravel(),
                               dx_src[valid_mask].ravel()], axis=-1)
        vals_valid = src[valid_mask].ravel()

        if len(vals_valid) == 0:
            out = np.where(needs_nn, fill_value, out)
        else:
            interp_nn  = NearestNDInterpolator(pts_valid, vals_valid)
            dy, dx = np.meshgrid(tgt_lat, tgt_lon, indexing="ij")
            tgt_pts = np.stack([dy.ravel(), dx.ravel()], axis=-1)
            nn_fill = interp_nn(tgt_pts).reshape(len(tgt_lat), len(tgt_lon))
            out = np.where(needs_nn, nn_fill, out)

    still_nan = np.isnan(out)
    if np.any(still_nan):
        out = np.where(still_nan, fill_value, out)

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Private: curvilinear (NEMO/ORCA) interpolation helpers
# ---------------------------------------------------------------------------

_CURV_BUFFER_DEG = 5.0  # degrees of padding around the target domain


def _curv_build_tri(
    src_lat2d: np.ndarray,
    src_lon2d: np.ndarray,
    tgt_lat:   np.ndarray,
    tgt_lon:   np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subset source points to a buffer around the target domain and build a
    Delaunay triangulation.  Subsetting reduces the triangulation cost from
    O(N_global) to O(N_region), which is critical for global ORCA grids.

    Returns
    -------
    tri       : ``scipy.spatial.Delaunay`` object
    sub_lat   : 1-D lat of subset source points
    sub_lon   : 1-D lon of subset source points
    in_region : bool mask ``(Ny_src, Nx_src)`` — True for included points
    """
    lat_lo = tgt_lat.min() - _CURV_BUFFER_DEG
    lat_hi = tgt_lat.max() + _CURV_BUFFER_DEG

    # Unify tgt_lon to match the source convention before computing the
    # subsetting window (handles 0–360 vs −180–180 mismatch).
    tgt_lon_u = _unify_lon(src_lon2d.ravel(), tgt_lon)
    lon_lo = tgt_lon_u.min() - _CURV_BUFFER_DEG
    lon_hi = tgt_lon_u.max() + _CURV_BUFFER_DEG

    in_region = (
        (src_lat2d >= lat_lo) & (src_lat2d <= lat_hi) &
        (src_lon2d >= lon_lo) & (src_lon2d <= lon_hi)
    )

    # Fall back to the full grid if the buffer contains too few points
    if in_region.sum() < 10:
        in_region = np.ones_like(src_lat2d, dtype=bool)

    sub_pts = np.stack(
        [src_lat2d[in_region].ravel(), src_lon2d[in_region].ravel()],
        axis=-1,
    )
    return Delaunay(sub_pts), src_lat2d[in_region].ravel(), src_lon2d[in_region].ravel(), in_region


def _curv_interp_level(
    src_k_flat: np.ndarray,   # (N_sub,) values at one depth level, may contain NaN
    tri:        "Delaunay",
    tgt_pts:    np.ndarray,   # (N_tgt, 2) query points (lat, lon)
    Ny_tgt:     int,
    Nx_tgt:     int,
    fill_value: float,
) -> np.ndarray:
    """
    Interpolate one horizontal level from a curvilinear source to a regular
    target grid.  Three-tier NaN-aware strategy:
    1. Normalised convolution via ``LinearNDInterpolator`` (no coastal contamination).
    2. ``NearestNDInterpolator`` on valid source points only.
    3. Constant ``fill_value`` fallback.
    """
    valid = ~np.isnan(src_k_flat)
    if not np.any(valid):
        return np.full((Ny_tgt, Nx_tgt), fill_value, dtype=np.float32)

    src_filled = np.where(valid, src_k_flat, 0.0)
    src_mask   = valid.astype(np.float64)

    lin_data = LinearNDInterpolator(tri, src_filled, fill_value=0.0)
    lin_mask = LinearNDInterpolator(tri, src_mask,   fill_value=0.0)
    data_sum = lin_data(tgt_pts).reshape(Ny_tgt, Nx_tgt)
    mask_sum = lin_mask(tgt_pts).reshape(Ny_tgt, Nx_tgt)

    eps    = 1e-6
    result = np.where(mask_sum > eps, data_sum / np.maximum(mask_sum, eps), np.nan)

    needs_nn = np.isnan(result)
    if np.any(needs_nn):
        # Build NN interpolator using only valid-point coordinates
        src_pts_valid = np.stack(
            [tri.points[valid, 0], tri.points[valid, 1]], axis=-1
        )
        nn = NearestNDInterpolator(src_pts_valid, src_k_flat[valid])
        result = np.where(needs_nn, nn(tgt_pts).reshape(Ny_tgt, Nx_tgt), result)

    still_nan = np.isnan(result)
    if np.any(still_nan):
        result = np.where(still_nan, fill_value, result)

    return result.astype(np.float32)


def _interp_curvilinear_3d(
    src:        np.ndarray,   # (Nz_src, Ny_src, Nx_src)
    src_lat2d:  np.ndarray,   # (Ny_src, Nx_src)
    src_lon2d:  np.ndarray,   # (Ny_src, Nx_src)
    src_depth:  np.ndarray,   # (Nz_src,) ascending
    tgt_lon:    np.ndarray,   # (Nx_tgt,)
    tgt_lat:    np.ndarray,   # (Ny_tgt,)
    tgt_depth:  np.ndarray,   # (Nz_tgt,) ascending
    fill_value: float,
) -> np.ndarray:
    """
    Interpolate a curvilinear 3-D field onto a regular target grid.

    Two-step approach:
    1. Per-source-level horizontal interpolation using normalised convolution
       (``LinearNDInterpolator``) + NN fallback in (lat, lon) only.
    2. Vertical linear interpolation from ``src_depth`` to ``tgt_depth``;
       nearest-level clamping for depths outside the source range.

    A Delaunay triangulation is built once on the subset of source points
    within ``_CURV_BUFFER_DEG`` of the target domain to keep runtime
    manageable for global ORCA grids.
    """
    Nz_src = len(src_depth)
    Ny_tgt, Nx_tgt = len(tgt_lat), len(tgt_lon)

    tri, _, _, in_region = _curv_build_tri(src_lat2d, src_lon2d, tgt_lat, tgt_lon)

    dy_tgt, dx_tgt = np.meshgrid(tgt_lat, tgt_lon, indexing="ij")
    tgt_pts = np.stack([dy_tgt.ravel(), dx_tgt.ravel()], axis=-1)

    # Step 1: horizontal interpolation at every source depth level
    horiz = np.full((Nz_src, Ny_tgt, Nx_tgt), fill_value, dtype=np.float64)
    for k in range(Nz_src):
        horiz[k] = _curv_interp_level(
            src[k][in_region].ravel().astype(np.float64),
            tri, tgt_pts, Ny_tgt, Nx_tgt, fill_value,
        )

    # Step 2: vertical linear interpolation (nearest-level extrapolation)
    Nz_tgt = len(tgt_depth)
    out = np.empty((Nz_tgt, Ny_tgt, Nx_tgt), dtype=np.float32)
    for k_tgt, z in enumerate(tgt_depth):
        idx = int(np.searchsorted(src_depth, z, side="right")) - 1
        idx = max(0, min(idx, Nz_src - 2))
        z0, z1 = src_depth[idx], src_depth[idx + 1]
        w1 = float(np.clip((z - z0) / (z1 - z0), 0.0, 1.0))
        out[k_tgt] = ((1.0 - w1) * horiz[idx] + w1 * horiz[idx + 1]).astype(np.float32)

    return out


def _interp_curvilinear_2d(
    src:        np.ndarray,   # (Ny_src, Nx_src)
    src_lat2d:  np.ndarray,
    src_lon2d:  np.ndarray,
    tgt_lon:    np.ndarray,
    tgt_lat:    np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """Horizontal-only curvilinear interpolation for 2-D fields (e.g. eta)."""
    Ny_tgt, Nx_tgt = len(tgt_lat), len(tgt_lon)

    tri, _, _, in_region = _curv_build_tri(src_lat2d, src_lon2d, tgt_lat, tgt_lon)
    dy_tgt, dx_tgt = np.meshgrid(tgt_lat, tgt_lon, indexing="ij")
    tgt_pts = np.stack([dy_tgt.ravel(), dx_tgt.ravel()], axis=-1)

    return _curv_interp_level(
        src[in_region].ravel().astype(np.float64),
        tri, tgt_pts, Ny_tgt, Nx_tgt, fill_value,
    )


def _unify_lon(
    src_lon: np.ndarray,
    tgt_lon: np.ndarray,
) -> np.ndarray:
    """
    Shift ``tgt_lon`` values by ±360° so that they fall inside the longitude
    window spanned by ``src_lon``.

    This resolves 0–360 vs −180–180 convention mismatches (the most common
    case when the Copernicus regular product and a model grid use different
    conventions).

    **Limitation**: a target domain that straddles the dateline (e.g., a
    region spanning both sides of ±180°) is *not* handled.  For such domains
    the caller should split the request into two sub-domains or re-project
    the coordinates before calling ``regrid_to_model``.
    """
    src_min = float(np.nanmin(src_lon))
    src_max = float(np.nanmax(src_lon))
    # Give a 10° margin so that values just outside the source domain are
    # still shifted correctly before the out-of-domain warning triggers.
    out = tgt_lon.copy()
    out = np.where(out < src_min - 10.0, out + 360.0, out)
    out = np.where(out > src_max + 10.0, out - 360.0, out)
    return out


def _check_resolution(
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    tgt_lon: np.ndarray,
    tgt_lat: np.ndarray,
) -> None:
    if src_lon.ndim == 1:
        src_dx = float(np.median(np.diff(src_lon)))
        src_dy = float(np.median(np.diff(src_lat)))
    else:
        # Curvilinear: estimate grid spacing from adjacent-cell differences
        src_dx = float(np.median(np.abs(np.diff(src_lon, axis=1))))
        src_dy = float(np.median(np.abs(np.diff(src_lat, axis=0))))
    tgt_dx = float(np.median(np.diff(tgt_lon))) if len(tgt_lon) > 1 else src_dx
    tgt_dy = float(np.median(np.diff(tgt_lat))) if len(tgt_lat) > 1 else src_dy
    if tgt_dx < src_dx * 0.9 or tgt_dy < src_dy * 0.9:
        warnings.warn(
            f"Target grid ({tgt_dx:.3f}° × {tgt_dy:.3f}°) is finer than the "
            f"ORAS5 source ({src_dx:.3f}° × {src_dy:.3f}°).  "
            "Upscaling invents spurious fine-scale structure; "
            "consider a coarser target grid (≥ 0.25°).",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Public: regrid_to_model
# ---------------------------------------------------------------------------

def regrid_to_model(
    raw:    dict[str, Optional[np.ndarray]],
    grid:   OceanGrid,
    T_fill: float = 10.0,
    S_fill: float = 35.0,
) -> OceanState:
    """
    Interpolate raw ORAS5 fields onto the OceanJAX model grid.

    Parameters
    ----------
    raw    : dict returned by ``read_oras5()``.
    grid   : Target OceanGrid.
    T_fill : Temperature fallback [°C] for points outside ORAS5 coverage.
    S_fill : Salinity fallback [psu] for points outside ORAS5 coverage.

    Returns
    -------
    OceanState with masks applied, ready for ``step()`` / ``run()``.

    Notes
    -----
    Array layout: ORAS5 stores data as ``(Nz, Ny, Nx)``; OceanJAX uses
    ``(Nx, Ny, Nz)``.  Transposition is applied internally.

    Vertical extrapolation:
      T, S — nearest-neighbour (deepest valid ORAS5 value).
      u, v — zero below ORAS5 deepest level.
    """
    src_lon   = raw["lon"]
    src_lat   = raw["lat"]
    src_depth = raw["depth"]

    # Per-field source depths — fall back to T-grid if not explicitly provided.
    u_src_depth = raw["depth_u"] if raw.get("depth_u") is not None else src_depth
    v_src_depth = raw["depth_v"] if raw.get("depth_v") is not None else src_depth

    tgt_lon   = np.asarray(grid.lon_c)
    tgt_lat   = np.asarray(grid.lat_c)
    tgt_depth = np.asarray(grid.z_c)

    # Align tgt_lon to the same 0–360 / −180–180 window as the source so
    # that interpolation queries land inside the source domain.
    tgt_lon = _unify_lon(src_lon if src_lon.ndim == 1 else src_lon.ravel(),
                         tgt_lon)

    _check_resolution(src_lon, src_lat, tgt_lon, tgt_lat)

    if (tgt_lon.min() < src_lon.min() - 1e-6 or
            tgt_lon.max() > src_lon.max() + 1e-6 or
            tgt_lat.min() < src_lat.min() - 1e-6 or
            tgt_lat.max() > src_lat.max() + 1e-6):
        warnings.warn(
            "Target grid extends outside ORAS5 source domain.  "
            "Out-of-domain points will be filled by nearest-neighbour "
            "then constant fallback.",
            UserWarning,
            stacklevel=2,
        )

    Nz = len(tgt_depth)
    Ny = len(tgt_lat)
    Nx = len(tgt_lon)

    curvilinear = src_lat.ndim == 2

    if curvilinear:
        # ---- NEMO/ORCA curvilinear grid pipeline ----
        # Per-field horizontal coords — fall back to T-grid when absent
        # (merged files lose per-variable nav_lon/nav_lat stagger).
        u_src_lat2d = raw["lat_u"] if raw.get("lat_u") is not None else src_lat
        u_src_lon2d = raw["lon_u"] if raw.get("lon_u") is not None else src_lon
        v_src_lat2d = raw["lat_v"] if raw.get("lat_v") is not None else src_lat
        v_src_lon2d = raw["lon_v"] if raw.get("lon_v") is not None else src_lon

        kw3c   = dict(src_lat2d=src_lat,    src_lon2d=src_lon,    src_depth=src_depth,
                      tgt_lon=tgt_lon, tgt_lat=tgt_lat, tgt_depth=tgt_depth)
        kw3c_u = dict(src_lat2d=u_src_lat2d, src_lon2d=u_src_lon2d, src_depth=u_src_depth,
                      tgt_lon=tgt_lon, tgt_lat=tgt_lat, tgt_depth=tgt_depth)
        kw3c_v = dict(src_lat2d=v_src_lat2d, src_lon2d=v_src_lon2d, src_depth=v_src_depth,
                      tgt_lon=tgt_lon, tgt_lat=tgt_lat, tgt_depth=tgt_depth)
        kw2c   = dict(src_lat2d=src_lat,    src_lon2d=src_lon,
                      tgt_lon=tgt_lon, tgt_lat=tgt_lat)

        T_zyx = _interp_curvilinear_3d(raw["T"], **kw3c, fill_value=T_fill)
        S_zyx = _interp_curvilinear_3d(raw["S"], **kw3c, fill_value=S_fill)

        u_zyx = (_interp_curvilinear_3d(raw["u"], **kw3c_u, fill_value=0.0)
                  if raw["u"] is not None
                  else np.zeros((Nz, Ny, Nx), dtype=np.float32))
        v_zyx = (_interp_curvilinear_3d(raw["v"], **kw3c_v, fill_value=0.0)
                  if raw["v"] is not None
                  else np.zeros((Nz, Ny, Nx), dtype=np.float32))
        eta_yx = (_interp_curvilinear_2d(raw["eta"], **kw2c, fill_value=0.0)
                  if raw["eta"] is not None
                  else np.zeros((Ny, Nx), dtype=np.float32))
    else:
        # ---- Regular (Copernicus) grid pipeline ----
        # Regular ORAS5 product shares the same 1-D lat/lon for all variables;
        # only depth can differ per field.
        kw3   = dict(src_lon=src_lon, src_lat=src_lat, src_depth=src_depth,
                     tgt_lon=tgt_lon, tgt_lat=tgt_lat, tgt_depth=tgt_depth)
        kw3_u = dict(src_lon=src_lon, src_lat=src_lat, src_depth=u_src_depth,
                     tgt_lon=tgt_lon, tgt_lat=tgt_lat, tgt_depth=tgt_depth)
        kw3_v = dict(src_lon=src_lon, src_lat=src_lat, src_depth=v_src_depth,
                     tgt_lon=tgt_lon, tgt_lat=tgt_lat, tgt_depth=tgt_depth)
        kw2   = dict(src_lon=src_lon, src_lat=src_lat,
                     tgt_lon=tgt_lon, tgt_lat=tgt_lat)

        T_zyx = _interp_3d(raw["T"], **kw3, fill_value=T_fill)
        S_zyx = _interp_3d(raw["S"], **kw3, fill_value=S_fill)

        u_zyx = (_interp_3d(raw["u"], **kw3_u, fill_value=0.0)
                  if raw["u"] is not None
                  else np.zeros((Nz, Ny, Nx), dtype=np.float32))
        v_zyx = (_interp_3d(raw["v"], **kw3_v, fill_value=0.0)
                  if raw["v"] is not None
                  else np.zeros((Nz, Ny, Nx), dtype=np.float32))
        eta_yx = (_interp_2d(raw["eta"], **kw2, fill_value=0.0)
                  if raw["eta"] is not None
                  else np.zeros((Ny, Nx), dtype=np.float32))

    # Zero out u/v levels below each field's deepest ORAS5 level.
    # Extrapolating bottom velocities downward is physically wrong.
    if raw["u"] is not None:
        below_u = tgt_depth > u_src_depth.max()
        if np.any(below_u):
            u_zyx[below_u] = 0.0
    if raw["v"] is not None:
        below_v = tgt_depth > v_src_depth.max()
        if np.any(below_v):
            v_zyx[below_v] = 0.0

    # Transpose (Nz, Ny, Nx) → (Nx, Ny, Nz) for OceanJAX convention
    T   = T_zyx.transpose(2, 1, 0)
    S   = S_zyx.transpose(2, 1, 0)
    u   = u_zyx.transpose(2, 1, 0)
    v   = v_zyx.transpose(2, 1, 0)
    eta = eta_yx.T                    # (Ny, Nx) → (Nx, Ny)

    return create_from_arrays(grid, u=u, v=v, T=T, S=S, eta=eta)


# ---------------------------------------------------------------------------
# Public: load_oras5
# ---------------------------------------------------------------------------

def load_oras5(
    path:       str | Path,
    grid:       OceanGrid,
    time_index: int   = 0,
    T_fill:     float = 10.0,
    S_fill:     float = 35.0,
) -> OceanState:
    """
    Convenience wrapper: read and regrid in one call.

    Equivalent to::

        regrid_to_model(read_oras5(path, time_index), grid, T_fill, S_fill)

    Parameters
    ----------
    path       : Copernicus Marine ORAS5 NetCDF file.
    grid       : Target OceanGrid.
    time_index : Time slice index within the file (default 0).
    T_fill     : Fallback temperature [°C] for uncovered ocean points.
    S_fill     : Fallback salinity [psu] for uncovered ocean points.

    Returns
    -------
    OceanState ready to pass to ``step()`` or ``run()``.
    """
    return regrid_to_model(
        read_oras5(path, time_index),
        grid,
        T_fill=T_fill,
        S_fill=S_fill,
    )


# ---------------------------------------------------------------------------
# Public: oras5_month_files
# ---------------------------------------------------------------------------

#: ORAS5 variables that make up one month of ocean state (T, S required).
ORAS5_STATE_VARS: tuple[str, ...] = ("votemper", "vosaline", "vozocrtx", "vomecrty", "sossheig")


def oras5_month_files(directory: str | Path, year: int, month: int) -> list[Path]:
    """
    Files holding the ORAS5 ocean state of one month, for ``read_oras5``.

    A merged file ``oras5_<YYYY>_<MM>_native_merged.nc`` (``data_merger.py``)
    is used if present; otherwise the per-variable files
    ``<var>_control_monthly_highres_<2D|3D>_<YYYYMM>_*.nc`` for
    ``ORAS5_STATE_VARS``.  Temperature and salinity are required; missing
    velocity / SSH files only trigger a warning (``read_oras5`` then returns
    None for them).

    Raises:
        FileNotFoundError if temperature or salinity is missing.
    """
    directory = Path(directory)
    merged = directory / f"oras5_{year:04d}_{month:02d}_native_merged.nc"
    if merged.exists():
        return [merged]
    tag = f"_{year:04d}{month:02d}_"
    found = {}
    for var in ORAS5_STATE_VARS:
        hits = sorted(p for p in directory.glob(f"{var}_*.nc") if tag in p.name)
        if hits:
            found[var] = hits[0]
    missing = [v for v in ORAS5_STATE_VARS if v not in found]
    if "votemper" in missing or "vosaline" in missing:
        raise FileNotFoundError(
            f"ORAS5 {year}-{month:02d}: temperature/salinity files not found in {directory}")
    if missing:
        warnings.warn(f"ORAS5 {year}-{month:02d}: no file for {missing}", UserWarning,
                      stacklevel=2)
    return list(found.values())


# ---------------------------------------------------------------------------
# Public: oras5_bathymetry
# ---------------------------------------------------------------------------

def oras5_levels(
    depths:    "np.ndarray | dict",
    depth_max: float,
    dz_min:    float = 10.0,
) -> np.ndarray:
    """
    Model cell-centre depths (Nz,) [m] taken from the ORAS5 vertical grid.

    Uses ORAS5's own stretched levels (about 1 m thick at the surface,
    ~200 m at 5000 m), so model and data levels coincide and vertical
    interpolation error is minimal, with two adjustments:

    * Levels are thinned from the top so that consecutive centres are at
      least ``dz_min`` apart, which makes every model cell (in particular
      the top one) at least ``dz_min`` thick.  The linear free surface
      needs |eta| << top-cell thickness; ORAS5's ~1 m top levels are too
      thin for that.
    * Deep levels are dropped until the model bottom face (``face_depths``)
      lies at or above ``depth_max``.  Keep depth_max within the ORAS5 data
      (~5900 m): below it T/S would only be extrapolated.

    Args:
        depths    : ORAS5 T-level depths (ascending, metres), or the dict
                    returned by ``read_oras5`` (its ``"depth"`` entry).
        depth_max : maximum model depth [m].
        dz_min    : minimum cell thickness [m] (default 10).

    Returns:
        (Nz,) float64 ascending centre depths for OceanGrid.create.
    """
    d = np.asarray(depths["depth"] if isinstance(depths, dict) else depths,
                   dtype=np.float64)
    d = np.sort(d[np.isfinite(d)])
    picks = []
    for z in d:
        if not picks:
            if z >= 0.5 * dz_min:
                picks.append(z)
        elif z - picks[-1] >= dz_min:
            picks.append(z)
    while picks and face_depths(np.array(picks))[-1] > depth_max:
        picks.pop()
    if not picks:
        raise ValueError(f"no ORAS5 level fits dz_min={dz_min} m within "
                         f"depth_max={depth_max} m")
    return np.array(picks)


def _cell_edges(centres: np.ndarray, upper_faces: np.ndarray) -> np.ndarray:
    """
    Cell edges (N+1,) from centres and the upper (east / north) face of
    each cell, as stored on OceanGrid (lon_c + lon_u, or lat_c + lat_v).
    """
    lower0 = 2.0 * centres[0] - upper_faces[0]
    return np.concatenate([[lower0], upper_faces])


def oras5_bathymetry(
    raw:  dict[str, Optional[np.ndarray]],
    grid: OceanGrid,
) -> np.ndarray:
    """
    Model bathymetry H (Nx, Ny) [m] derived from the ORAS5 land/sea mask.

    ORAS5 marks land and sub-bottom cells as NaN.  For every source column
    the water depth is taken as the bottom face of its deepest wet level
    (counting only levels that are wet contiguously from the surface), and
    0 for land.  Each model column then receives the **median** of the
    source depths that fall inside its horizontal cell, i.e. a majority
    vote: a model cell that is more than half land becomes land (H = 0).
    Model cells that contain no source point (target finer than ORAS5)
    fall back to the nearest source column.

    Only ``grid.lon_c/lon_u/lat_c/lat_v`` are used, so a flat-bottom grid
    built with the same horizontal layout can be passed in::

        raw       = read_oras5(path)
        grid_flat = OceanGrid.create(LON, LAT, depth_levels, Nx, Ny)
        H         = oras5_bathymetry(raw, grid_flat)
        grid      = OceanGrid.create(LON, LAT, depth_levels, Nx, Ny,
                                     bathymetry=H, periodic_x=False)

    A model cell is then wet where its top face lies above H, so H values
    deeper than the model's bottom simply give a full-depth column.

    Parameters
    ----------
    raw  : dict returned by ``read_oras5()`` (uses ``T``, ``depth``,
           ``lon``, ``lat``).  Regular and curvilinear grids are supported.
    grid : target OceanGrid (horizontal layout only).

    Returns
    -------
    H : (Nx, Ny) float64 water depth [m]; 0 on land columns.
    """
    T     = raw["T"]                               # (Nz_src, Ny_src, Nx_src)
    depth = np.asarray(raw["depth"], dtype=np.float64)

    # --- source water depth per column -----------------------------------
    wet_contig = np.cumprod(~np.isnan(T), axis=0)  # 1 while wet from surface
    n_wet      = wet_contig.sum(axis=0)            # (Ny_src, Nx_src)
    src_H      = np.where(n_wet > 0, face_depths(depth)[n_wet], 0.0)

    src_lon, src_lat = raw["lon"], raw["lat"]
    if src_lon.ndim == 1:
        src_lat, src_lon = np.meshgrid(src_lat, src_lon, indexing="ij")
    src_lon, src_lat, src_H = src_lon.ravel(), src_lat.ravel(), src_H.ravel()

    # --- target cell edges, in the source longitude convention -------------
    lon_c = np.asarray(grid.lon_c, dtype=np.float64)
    lat_c = np.asarray(grid.lat_c, dtype=np.float64)
    lon_edges = _unify_lon(
        src_lon, _cell_edges(lon_c, np.asarray(grid.lon_u, dtype=np.float64))
    )
    lat_edges = _cell_edges(lat_c, np.asarray(grid.lat_v, dtype=np.float64))
    lon_c = _unify_lon(src_lon, lon_c)
    Nx, Ny = len(lon_c), len(lat_c)

    # --- bin source columns into model cells, take the median --------------
    ix = np.searchsorted(lon_edges, src_lon, side="right") - 1
    iy = np.searchsorted(lat_edges, src_lat, side="right") - 1
    inside = (ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny) & np.isfinite(src_H)

    H      = np.full((Nx, Ny), np.nan, dtype=np.float64)
    cell   = ix[inside] * Ny + iy[inside]
    vals   = src_H[inside]
    order  = np.argsort(cell, kind="stable")
    cell, vals = cell[order], vals[order]
    ids, starts = np.unique(cell, return_index=True)
    for cid, chunk in zip(ids, np.split(vals, starts[1:])):
        H[cid // Ny, cid % Ny] = np.median(chunk)

    # --- empty cells: nearest source column --------------------------------
    empty = np.isnan(H)
    if np.any(empty):
        nn = NearestNDInterpolator(np.stack([src_lat, src_lon], axis=-1), src_H)
        lat2d, lon2d = np.meshgrid(lat_c, lon_c)   # (Nx, Ny)
        H[empty] = nn(np.stack([lat2d[empty], lon2d[empty]], axis=-1))

    return H


def oras5_grid(
    raw:          dict[str, Optional[np.ndarray]],
    lon_bounds:   tuple,
    lat_bounds:   tuple,
    depth_levels: np.ndarray,
    Nx:           int,
    Ny:           int,
    periodic_x:   bool = False,
) -> OceanGrid:
    """
    OceanGrid whose land mask and bathymetry come from ORAS5.

    Convenience wrapper around ``oras5_bathymetry``: builds a flat-bottom
    grid for the horizontal layout, derives H from ``raw`` and rebuilds the
    grid with it.  ``periodic_x`` defaults to False (closed east/west walls)
    because ORAS5-initialised runs are regional.
    """
    flat = OceanGrid.create(lon_bounds, lat_bounds, depth_levels, Nx, Ny)
    H    = oras5_bathymetry(raw, flat)
    return OceanGrid.create(lon_bounds, lat_bounds, depth_levels, Nx, Ny,
                            bathymetry=H, periodic_x=periodic_x)


# ---------------------------------------------------------------------------
# Public: read_oras5_forcing
# ---------------------------------------------------------------------------

#: All surface-forcing field keys recognised by :func:`read_oras5_forcing`.
FORCING_FIELDS: frozenset[str] = frozenset({"heat_flux", "fw_flux", "tau_x", "tau_y"})


def read_oras5_forcing(
    path:       str | Path | list,
    time_index: int = 0,
) -> dict[str, Optional[np.ndarray]]:
    """
    Read surface-forcing fields from one or more ORAS5 NetCDF files.

    All four forcing fields are optional.  If a variable is absent from
    every supplied file its key is set to ``None`` rather than raising an
    error — the caller can decide how to handle missing fields.

    Parameters
    ----------
    path       : A single NetCDF file path, or a list of paths.
                 When a list is given every file is searched; later files
                 in the list take precedence if the same field appears in
                 multiple files.  Coordinates (lon/lat) are taken from the
                 first file that contains them.
    time_index : Index along the time dimension to read (default 0).

    Returns
    -------
    dict with keys ``"heat_flux"``, ``"fw_flux"``, ``"tau_x"``, ``"tau_y"``
    (each a ``(Ny_src, Nx_src)`` float32 array or ``None``), plus the
    horizontal coordinate keys ``"lon"`` and ``"lat"``.

    Sign conventions
    ----------------
    ``heat_flux`` : positive = net downward into the ocean  [W m-2]
    ``fw_flux``   : positive = net evaporation (E - P)      [m s-1]
    ``tau_x``     : positive = eastward                     [N m-2]
    ``tau_y``     : positive = northward                    [N m-2]

    Notes
    -----
    ``sohefldo`` in ORAS5 is defined as positive-downward, consistent with the
    model convention.  ``sowaflup`` is positive upward (evaporation > 0), also
    consistent.  Wind-stress sign conventions vary by product — check the
    source file metadata if in doubt.

    Unit conversion
    ---------------
    ``fw_flux`` is expected by the model in **m s-1** (volume flux per unit
    area).  ORAS5 ``sowaflup`` is stored in **kg m-2 s-1** (mass flux).
    This function detects the mass-flux unit and converts automatically by
    dividing by the freshwater density 1000 kg m-3.
    ``heat_flux`` (W m-2), ``tau_x``, and ``tau_y`` (N m-2) require no
    conversion and are returned as-is.
    """
    # Freshwater density for mass-flux → volume-flux conversion [kg m-3]
    _RHO_FRESHWATER: float = 1000.0
    # Unit strings that indicate kg/(m^2 s); case-insensitive comparison below
    _MASS_FLUX_UNITS = {"kg/m2/s", "kg m-2 s-1", "kg/(m2 s)", "kg/m**2/s"}

    paths = path if isinstance(path, list) else [path]

    result: dict[str, Optional[np.ndarray]] = {
        "lon": None, "lat": None,
        "heat_flux": None, "fw_flux": None, "tau_x": None, "tau_y": None,
    }

    for p in paths:
        ds = xr.open_dataset(p, mask_and_scale=True)
        try:
            # Coordinates: take from the first file that has them
            if result["lon"] is None:
                try:
                    lon_name = _find_coord(ds, "lon")
                    lat_name = _find_coord(ds, "lat")
                    result["lon"] = np.asarray(ds[lon_name].values, dtype=np.float64)
                    result["lat"] = np.asarray(ds[lat_name].values, dtype=np.float64)
                except KeyError:
                    pass

            for key in ("heat_flux", "fw_flux", "tau_x", "tau_y"):
                vname = _find_var(ds, key, optional=True)
                if vname is None:
                    continue

                da = ds[vname]
                time_dims = [d for d in da.dims
                             if d in {"time", "time_counter", "t"}]
                if time_dims:
                    da = da.isel({time_dims[0]: time_index})

                arr = np.asarray(da.values, dtype=np.float32)
                arr = np.where(np.isfinite(arr), arr, np.nan)

                # Unit conversion for freshwater flux: kg/(m^2 s) → m/s
                if key == "fw_flux":
                    units_str = str(da.attrs.get("units", "")).lower().replace(" ", "")
                    # Normalise common separators for robust matching
                    units_norm = units_str.replace("**", "").replace("^", "")
                    if any(u.replace(" ", "") in units_norm
                           for u in _MASS_FLUX_UNITS):
                        arr = arr / _RHO_FRESHWATER

                result[key] = arr  # later file overrides earlier
        finally:
            ds.close()

    return result


# ---------------------------------------------------------------------------
# Public: regrid_forcing
# ---------------------------------------------------------------------------

def regrid_forcing(
    raw_forcing: dict[str, Optional[np.ndarray]],
    grid:        OceanGrid,
    use_fields:  set[str] | frozenset[str] = FORCING_FIELDS,
) -> "SurfaceForcing":
    """
    Interpolate surface-forcing fields onto the OceanJAX model grid.

    Parameters
    ----------
    raw_forcing : dict returned by :func:`read_oras5_forcing`.
    grid        : Target OceanGrid.
    use_fields  : Which forcing fields to interpolate.  Fields not in this
                  set, or absent from ``raw_forcing``, are set to zero.
                  Defaults to all four: ``{"heat_flux", "fw_flux", "tau_x", "tau_y"}``.

    Returns
    -------
    :class:`~OceanJAX.timeStepping.SurfaceForcing` with shape ``(Nx, Ny)``
    per field, ready to broadcast into a ``forcing_sequence``.
    """
    from OceanJAX.timeStepping import SurfaceForcing
    import jax.numpy as jnp

    src_lon = raw_forcing["lon"]
    src_lat = raw_forcing["lat"]

    tgt_lon = np.asarray(grid.lon_c)
    tgt_lat = np.asarray(grid.lat_c)

    tgt_lon_adj = _unify_lon(
        src_lon if src_lon.ndim == 1 else src_lon.ravel(), tgt_lon
    )

    curvilinear = src_lat.ndim == 2
    Ny = len(tgt_lat)
    Nx = len(tgt_lon)

    def _regrid_field(arr: np.ndarray) -> np.ndarray:
        """Regrid one (Ny_src, Nx_src) forcing field → (Nx, Ny) model grid."""
        if curvilinear:
            yx = _interp_curvilinear_2d(
                arr,
                src_lat2d=src_lat, src_lon2d=src_lon,
                tgt_lon=tgt_lon_adj, tgt_lat=tgt_lat,
                fill_value=0.0,
            )
        else:
            yx = _interp_2d(
                arr,
                src_lon=src_lon, src_lat=src_lat,
                tgt_lon=tgt_lon_adj, tgt_lat=tgt_lat,
                fill_value=0.0,
            )
        # (Ny, Nx) → (Nx, Ny)
        return yx.T.astype(np.float32)

    zeros = np.zeros((Nx, Ny), dtype=np.float32)

    def _get(key: str) -> np.ndarray:
        if key not in use_fields:
            return zeros
        arr = raw_forcing.get(key)
        if arr is None:
            return zeros
        return _regrid_field(arr)

    return SurfaceForcing(
        heat_flux = jnp.asarray(_get("heat_flux")),
        fw_flux   = jnp.asarray(_get("fw_flux")),
        tau_x     = jnp.asarray(_get("tau_x")),
        tau_y     = jnp.asarray(_get("tau_y")),
    )
