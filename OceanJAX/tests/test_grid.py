"""
Tests for OceanJAX.grid vertical-level and CFL helpers
======================================================
  1. face_depths reproduces the faces OceanGrid.create builds.
  2. stretched_levels: bottom face exactly at depth_max, thicknesses
     growing monotonically from ~dz_top, uniform limit, invalid input.
  3. barotropic_cfl: matches sqrt(gH) dt / dx_min on a simple grid, caps H
     at the model bottom, and dt_max sits at the CFL limit.

Running
-------
    pytest OceanJAX/tests/test_grid.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from OceanJAX.grid import (
    OceanGrid, face_depths, stretched_levels, barotropic_cfl, CFL_LIMIT,
)


class TestFaceDepths:

    def test_matches_ocean_grid(self):
        c = stretched_levels(4000.0, 30, 10.0)
        g = OceanGrid.create((0.0, 10.0), (0.0, 10.0), c, 2, 2)
        np.testing.assert_allclose(np.asarray(g.z_w), face_depths(c), rtol=1e-6)

    def test_uniform(self):
        np.testing.assert_allclose(face_depths(np.array([5.0, 15.0, 25.0])),
                                   [0.0, 10.0, 20.0, 30.0])


class TestStretchedLevels:

    @pytest.mark.parametrize("depth_max, nz, dz_top",
                             [(500.0, 10, 10.0), (4000.0, 30, 10.0), (4000.0, 40, 5.0)])
    def test_bottom_and_monotone(self, depth_max, nz, dz_top):
        c  = stretched_levels(depth_max, nz, dz_top)
        f  = face_depths(c)
        dz = np.diff(f)
        assert len(c) == nz
        assert f[-1] == pytest.approx(depth_max, rel=1e-12)
        assert np.all(np.diff(dz) > 0), "thickness must grow with depth"
        # top cell within a few per cent of dz_top (faces sit between centres)
        assert dz_top <= dz[0] <= 1.1 * dz_top

    def test_uniform_limit(self):
        c = stretched_levels(100.0, 10, 10.0)          # nz * dz_top == depth_max
        np.testing.assert_allclose(np.diff(face_depths(c)), 10.0, rtol=1e-12)

    @pytest.mark.parametrize("args", [(100.0, 20, 10.0), (100.0, 0, 10.0), (100.0, 5, -1.0)])
    def test_invalid(self, args):
        with pytest.raises(ValueError):
            stretched_levels(*args)


class TestBarotropicCFL:

    def test_value_and_dt_max(self):
        c = (np.arange(5) + 0.5) * 100.0                  # flat bottom at 500 m
        g = OceanGrid.create((0.0, 10.0), (-5.0, 5.0), c, 5, 5)
        cfl, dt_max = barotropic_cfl(g, 300.0)
        dx_min = min(float(np.asarray(g.dx_c).min()), float(np.asarray(g.dy_c).min()))
        expected = np.sqrt(9.81 * 500.0) * 300.0 / dx_min
        assert cfl == pytest.approx(expected, rel=1e-5)
        assert barotropic_cfl(g, dt_max)[0] == pytest.approx(CFL_LIMIT, rel=1e-6)

    def test_depth_capped_at_model_bottom(self):
        """Bathymetry deeper than the model bottom does not raise the CFL."""
        c = (np.arange(5) + 0.5) * 100.0
        deep = OceanGrid.create((0.0, 10.0), (-5.0, 5.0), c, 5, 5,
                                bathymetry=np.full((5, 5), 5000.0))
        flat = OceanGrid.create((0.0, 10.0), (-5.0, 5.0), c, 5, 5)
        assert barotropic_cfl(deep, 300.0)[0] == pytest.approx(barotropic_cfl(flat, 300.0)[0])
