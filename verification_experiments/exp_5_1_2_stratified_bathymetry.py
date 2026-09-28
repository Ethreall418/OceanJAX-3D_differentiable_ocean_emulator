"""
Experiment 5.1.2 — Static Stratification Test on Real Bathymetry
=================================================================
Setup
-----
  - Real ORAS5 land mask and bathymetry via oras5_grid (mask_c reflects
    actual land-sea boundaries and stepped bottom depths); closed E/W walls
  - Initial T/S: horizontally uniform, vertically stratified profile
    taken as the domain-averaged ORAS5 T/S column broadcast to all (i,j)
  - u = v = w = eta = 0  (no initial motion)
  - No surface forcing
  - Integration: 10 days  (dt = 300 s → 2880 steps), three cases:
      A  kappa_v = kappa_h = 0      numerical-consistency test
      B  kappa_v = 1e-5 (default)   physical response, recorded
      C  kappa_v = 2e-5             attribution check against B

Rationale
---------
A horizontally uniform, vertically stratified field has no horizontal
pressure gradient on any level, so without diffusion it must stay exactly
at rest.  On stepped bathymetry this exercises the mask-consistency of
the pressure-gradient, Coriolis and continuity operators at land
boundaries and partial-depth columns (case A).

With vertical diffusion the state is NOT a steady solution: at a given
depth, a column whose seafloor lies just below that level cannot pass
heat downward (no-flux bottom), while a deeper neighbour can.  The two
columns' T diverge on the same level, creating a horizontal density
gradient and a genuine, diffusion-driven flow (cf. Phillips 1970, Wunsch
1970, boundary mixing over slopes).  Horizontal diffusion of a
horizontally uniform field is zero, so kappa_v alone drives it.  Case B
records this response; case C doubles kappa_v, and if the flow is
diffusion-driven it must grow monotonically with kappa_v: A = 0 < B < C.

Proportionality is NOT required.  Doubling kappa_v doubles the T anomaly
at the steps, but it also changes the anomaly's vertical structure
(penetration ~ sqrt(kappa_v t)), and with a Munk-criterion nu_h the
velocity is set by a viscous balance; so the speed/eta/KE ratios C/B are
printed for reference only (e.g. T 2.00, eta 1.84, KE 1.33, speed 1.27
at nu_h = 1.7e5 m² s⁻¹).

Metrics recorded every SAVE_INTERVAL steps
-------------------------------------------
  max_u, max_v : max |u|, |v|  [m/s]
  max_eta      : max |eta|  [m]
  KE           : domain kinetic energy  0.5·rho0·∫(u²+v²)dV  [J]
  max_T_drift  : max pointwise |T(t) - T(0)| over wet cells  [°C]

Pass criteria
-------------
  A: u, v, eta exactly zero and T exactly unchanged
  B, C: monotonic response  0 = A < B < C  for max speed, max |eta| and
        max KE (magnitudes recorded, no thresholds)
  No non-finite values
"""

from __future__ import annotations

import sys
import time as _time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import equinox as eqx
from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, OceanState, create_from_arrays
from OceanJAX.data.oras5 import read_oras5, regrid_to_model, oras5_grid
from OceanJAX.Physics.mixing import munk_viscosity
from OceanJAX.timeStepping import run

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORAS5_PATH   = "OceanJAX/data/data_oras5/oras5_2026_01_native_merged.nc"
LON          = (-40.0, -5.0)
LAT          = (-15.0, 15.0)
DEPTH_LEVELS = np.array([25., 75., 150., 250., 375., 500.], dtype=np.float64)
NX, NY       = 20, 15
DT           = 300.0
N_DAYS       = 10
RHO0         = 1025.0

STEPS_PER_DAY = int(86400 / DT)   # 288
TOTAL_STEPS   = N_DAYS * STEPS_PER_DAY
SAVE_INTERVAL = STEPS_PER_DAY

# ---------------------------------------------------------------------------
# Load ORAS5 → get bathymetry and real T/S structure
# ---------------------------------------------------------------------------
print("Loading ORAS5 ...", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw = read_oras5(ORAS5_PATH, time_index=0)

# Grid with the real ORAS5 land mask and bathymetry (median of ORAS5 water
# depths within each model cell); closed east/west walls.
grid = oras5_grid(raw, LON, LAT, DEPTH_LEVELS, NX, NY, periodic_x=False)
# Horizontal eddy viscosity from the Munk criterion for this grid
NU_H = munk_viscosity(grid)
print(f"  nu_h = {NU_H:.3g} m2/s (Munk criterion)")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    oras5_state = regrid_to_model(raw, grid)

H_np = np.array(grid.H)
n_levels = np.array(grid.mask_c).sum(axis=-1)
print(f"  Grid: {NX}x{NY}x{len(DEPTH_LEVELS)}  "
      f"wet columns: {int((H_np > 0).sum())}/{NX*NY}  "
      f"partial-depth columns: {int(((n_levels > 0) & (n_levels < len(DEPTH_LEVELS))).sum())}")

# ---------------------------------------------------------------------------
# Build horizontally uniform, vertically stratified initial state
# ---------------------------------------------------------------------------
# Domain-average T/S profile over wet cells
T_oras5_r = np.array(oras5_state.T)   # (Nx, Ny, Nz)
S_oras5_r = np.array(oras5_state.S)

mask_np = np.array(grid.mask_c)       # (Nx, Ny, Nz) — real bathymetry mask

T_profile = np.zeros(len(DEPTH_LEVELS), dtype=np.float32)
S_profile = np.zeros(len(DEPTH_LEVELS), dtype=np.float32)
for k in range(len(DEPTH_LEVELS)):
    wet = mask_np[:, :, k] > 0
    if wet.sum() > 0:
        T_profile[k] = T_oras5_r[:, :, k][wet].mean()
        S_profile[k] = S_oras5_r[:, :, k][wet].mean()
    else:
        T_profile[k] = T_profile[k-1] if k > 0 else 10.0
        S_profile[k] = S_profile[k-1] if k > 0 else 35.0

print(f"  Domain-mean T profile: {T_profile}")
print(f"  Domain-mean S profile: {S_profile}")

# Broadcast profile to all (i,j), zero at land
T_init = (np.ones((NX, NY, len(DEPTH_LEVELS)), dtype=np.float32)
          * T_profile[np.newaxis, np.newaxis, :]) * mask_np
S_init = (np.ones((NX, NY, len(DEPTH_LEVELS)), dtype=np.float32)
          * S_profile[np.newaxis, np.newaxis, :]) * mask_np
u_init   = np.zeros((NX, NY, len(DEPTH_LEVELS)), dtype=np.float32)
v_init   = np.zeros((NX, NY, len(DEPTH_LEVELS)), dtype=np.float32)
eta_init = np.zeros((NX, NY), dtype=np.float32)

T_init_jnp = jnp.array(T_init)   # reference for drift calculation

# Domain volume
vol = np.array(grid.volume_c)
RHO0_val = 1025.0
wet = mask_np > 0


# ---------------------------------------------------------------------------
# One 10-day integration from the stratified rest state
# ---------------------------------------------------------------------------
def run_case(label: str, params: ModelParams) -> dict:
    state = create_from_arrays(grid, u_init, v_init, T_init, S_init, eta_init)
    records = []

    print()
    print(f"--- {label} ---")
    print(f"{'Day':>5}  {'max|u|':>12}  {'max|v|':>12}  {'max|eta|':>10}  "
          f"{'KE [J]':>12}  {'max_T_drift':>13}  {'status':>10}")
    print("-" * 88)

    for chunk_idx in range(N_DAYS):
        state, _ = run(state, grid, params, n_steps=SAVE_INTERVAL,
                       forcing_sequence=None, save_history=False)

        u_arr   = np.array(state.u)
        v_arr   = np.array(state.v)
        eta_arr = np.array(state.eta)
        T_arr   = np.array(state.T)

        bad = not (np.all(np.isfinite(u_arr)) and np.all(np.isfinite(v_arr))
                   and np.all(np.isfinite(eta_arr)) and np.all(np.isfinite(T_arr)))

        rec = dict(
            day         = float(state.time) / 86400.0,
            max_u       = float(np.max(np.abs(u_arr))),
            max_v       = float(np.max(np.abs(v_arr))),
            max_eta     = float(np.max(np.abs(eta_arr))),
            KE          = 0.5 * RHO0_val * float(np.sum((u_arr**2 + v_arr**2) * vol)),
            max_T_drift = float(np.max(np.abs(T_arr - T_init)[wet])),
            bad         = bad,
        )
        records.append(rec)
        print(f"{rec['day']:5.0f}  {rec['max_u']:12.3e}  {rec['max_v']:12.3e}  "
              f"{rec['max_eta']:10.3e}  {rec['KE']:12.3e}  {rec['max_T_drift']:13.3e}  "
              f"{'NON-FINITE' if bad else 'ok':>10}")
        if bad:
            print("  [ABORT] Non-finite values detected.")
            break

    return dict(
        max_u   = max(r["max_u"]   for r in records),
        max_v   = max(r["max_v"]   for r in records),
        max_eta = max(r["max_eta"] for r in records),
        KE      = max(r["KE"]      for r in records),
        T_drift = max(r["max_T_drift"] for r in records),
        speed   = max(max(r["max_u"], r["max_v"]) for r in records),
        bad     = any(r["bad"] for r in records),
    )


t0_wall = _time.time()
KAPPA_V = 1e-5
res_A = run_case("A: no diffusion (kappa_v = kappa_h = 0) — numerical consistency",
                 ModelParams(nu_h=NU_H, dt=DT, kappa_v=0.0, kappa_h=0.0))
res_B = run_case(f"B: kappa_v = {KAPPA_V:g} — physical response (recorded)",
                 ModelParams(nu_h=NU_H, dt=DT, kappa_v=KAPPA_V))
res_C = run_case(f"C: kappa_v = {2*KAPPA_V:g} — attribution check",
                 ModelParams(nu_h=NU_H, dt=DT, kappa_v=2 * KAPPA_V))
wall = _time.time() - t0_wall

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 88)
print(f"Wall time: {wall:.1f} s")
print()
print(f"{'case':>4}  {'max|u|':>10}  {'max|v|':>10}  {'max|eta|':>10}  {'KE [J]':>10}  {'max T drift':>11}")
for name, r in (("A", res_A), ("B", res_B), ("C", res_C)):
    print(f"{name:>4}  {r['max_u']:10.3e}  {r['max_v']:10.3e}  {r['max_eta']:10.3e}  "
          f"{r['KE']:10.3e}  {r['T_drift']:11.3e}")
def _ratio(key):
    return res_C[key] / res_B[key] if res_B[key] > 0 else float("nan")

print(f"\nRatios C/B (kappa_v x2, reference only): "
      f"T drift {_ratio('T_drift'):.2f}  eta {_ratio('max_eta'):.2f}  "
      f"KE {_ratio('KE'):.2f}  speed {_ratio('speed'):.2f}")
print()


def _monotonic(key):
    return res_A[key] == 0.0 < res_B[key] < res_C[key]


results = {
    "A: u exactly zero"                         : res_A["max_u"]   == 0.0,
    "A: v exactly zero"                         : res_A["max_v"]   == 0.0,
    "A: eta exactly zero"                       : res_A["max_eta"] == 0.0,
    "A: T exactly unchanged"                    : res_A["T_drift"] == 0.0,
    "0 = A < B < C : max speed"                 : _monotonic("speed"),
    "0 = A < B < C : max |eta|"                 : _monotonic("max_eta"),
    "0 = A < B < C : max KE"                    : _monotonic("KE"),
    "no non-finite values"                      : not (res_A["bad"] or res_B["bad"] or res_C["bad"]),
}

all_pass = all(results.values())
for name, passed in results.items():
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")

print()
print("Overall:", "PASS" if all_pass else "FAIL")
