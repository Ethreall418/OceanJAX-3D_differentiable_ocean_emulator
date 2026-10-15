# OceanJAX

A fully differentiable 3-D ocean model built in JAX, implementing Boussinesq hydrostatic primitive equations on an Arakawa C-grid in spherical coordinates. Designed for ML-based parameterization and adjoint-based optimization.

**Features**

- **Dynamics:** linear free surface; leapfrog + Asselin–Robert filter for momentum and sea level, Adams–Bashforth 3 for tracers; linear equation of state with the hydrostatic pressure integrated from the density anomaly.
- **Vertical mixing:** constant, Pacanowski & Philander (1981) Richardson-number mixing with continuous convective adjustment, or the KPP surface boundary layer (Large, McWilliams & Doney 1994); implicit column solves.
- **Other physics:** implicit quadratic bottom drag, resolution-aware horizontal viscosity (Munk criterion), freezing-point limit, virtual salt flux based on the local surface salinity.
- **Grids:** variable horizontal resolution, uniform / geometrically stretched / ORAS5 vertical levels, land mask and bathymetry from ORAS5, periodic or closed east–west boundaries, barotropic CFL check.
- **Data:** ORAS5 initialisation (cold or full state) and calendar-aware monthly ORAS5 surface forcing.
- **Machine learning:** closure hook for learned tracer tendencies and diffusivity scaling; `jax.grad` through the time stepping.
- **Parallelism:** `vmap` ensembles and domain decomposition across devices (GSPMD).
- **Evaluation:** verification experiments and a hindcast evaluation against ORAS5 with persistence skill scores.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Project Structure](#3-project-structure)
4. [Quick Start](#4-quick-start)
5. [Step-by-Step Usage Guide](#5-step-by-step-usage-guide)
   - 5.1 [Build a Grid](#51-build-a-grid)
   - 5.2 [Initialize Model State](#52-initialize-model-state)
   - 5.3 [Set Model Parameters](#53-set-model-parameters)
   - 5.4 [Prepare Surface Forcing](#54-prepare-surface-forcing)
   - 5.5 [Run the Model](#55-run-the-model)
   - 5.6 [Save Output](#56-save-output)
6. [Vertical Mixing Schemes](#6-vertical-mixing-schemes)
7. [Using ORAS5 Reanalysis Data](#7-using-oras5-reanalysis-data)
8. [Time-Varying Forcing](#8-time-varying-forcing)
9. [ML Closure Interface](#9-ml-closure-interface)
10. [Ensemble / Multi-Device Runs](#10-ensemble--multi-device-runs)
11. [Hindcast Evaluation against ORAS5](#11-hindcast-evaluation-against-oras5)
12. [Running Verification Tests](#12-running-verification-tests)
13. [Stability Guidelines](#13-stability-guidelines)
14. [ModelParams Reference](#14-modelparams-reference)

---

## 1. Requirements

| Package | Minimum version |
|---|---|
| Python | 3.10 |
| jax | 0.4.25 |
| jaxlib | 0.4.25 |
| equinox | 0.11.0 |
| numpy | 1.26 |
| xarray | 2024.1 |
| netCDF4 | 1.6 |
| scipy | 1.12 |
| pytest | 8.0 |

**GPU (optional):** a CUDA-enabled JAX build (`jax[cuda12]` or `jax[cuda13]`) on Linux or WSL2; JAX on native Windows is CPU-only. The model runs correctly on CPU. The GPU pays off on larger grids: for a 0.5° Atlantic domain (200 × 160 × 33) one model day takes 22.5 s on an RTX 5090 versus 125 s on the CPU of the same machine, while at 1° (100 × 80 × 33) the gain is only about 1.15× because each step launches many small kernels.

Under WSL2, set

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false          # GPU memory on demand
export XLA_FLAGS=--xla_gpu_enable_command_buffer=   # disable CUDA graphs
```

With CUDA graphs enabled, the compiled time-stepping loop crashes (segmentation fault) on its second execution under WSL2.

---

## 2. Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ethreall418/OceanJAX-3D_differentiable_ocean_emulator.git
cd OceanJAX-3D_differentiable_ocean_emulator

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (GPU users only, Linux / WSL2) Install CUDA-enabled JAX
pip install --upgrade "jax[cuda12]"

# 5. Verify installation
python -m pytest OceanJAX/tests/ -q
# Expected: all tests pass (244 on CPU at the time of writing)
```

`OceanJAX/tests/conftest.py` sets up 8 simulated CPU devices for the domain-decomposition tests; single-device code is unaffected. On a single GPU the multi-device tests are skipped; run `JAX_PLATFORMS=cpu python -m pytest OceanJAX/tests/test_sharding.py OceanJAX/tests/test_parallel.py` to cover them.

---

## 3. Project Structure

```
OceanJAX/
├── grid.py                 # OceanGrid — Arakawa C-grid, masks, metrics;
│                           #   face_depths, stretched_levels, barotropic_cfl
├── state.py                # OceanState, ModelParams, factory functions
├── operators.py            # grad_x, grad_y, div_h, laplacian
├── timeStepping.py         # step(), run(), SurfaceForcing
├── diagnostics.py          # area-weighted bias / RMSE / correlation, skill scores
├── Physics/
│   ├── dynamics.py         # EOS, hydrostatic pressure, PGF, Coriolis, free surface, w
│   ├── tracers.py          # tracer advection/diffusion, surface heat and salt fluxes
│   └── mixing.py           # implicit vertical mixing/viscosity, PP81, KPP,
│                           #   bottom drag, horizontal viscosity, munk_viscosity
├── data/
│   ├── oras5.py            # ORAS5 reading and regridding, bathymetry, levels,
│   │                       #   oras5_grid, oras5_month_files, forcing
│   ├── forcing.py          # make_synthetic_forcing, make_forcing_sequence
│   └── monthly_forcing.py  # MonthlyForcing: calendar-aware monthly ORAS5 forcing
├── ml/
│   └── closure.py          # AbstractClosure, NullClosure, ClosureOutput
├── parallel/
│   ├── ensemble.py         # batch_run, sharded_ensemble_run
│   └── sharding.py         # domain decomposition: make_mesh, shard_*, sharded_run
└── tests/                  # pytest test suite

experiment.py               # Full experiment driver (CONFIG block, NetCDF output, diagnostics)
plot_output.py              # Standard figures from NetCDF output
validate_oras5.py           # ORAS5 file validation utility
verification_experiments/   # Verification scripts and the ORAS5 hindcast (Section 11)
docs/parallel.md            # Domain decomposition: design, accuracy, SLURM example
runtime_test/               # Parallel benchmark
```

---

## 4. Quick Start

The following runs a 30-day integration on a small idealised domain in under one minute.

```python
import numpy as np
import jax
import jax.numpy as jnp

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, create_rest_state
from OceanJAX.timeStepping import run
from OceanJAX.data.forcing import make_synthetic_forcing

# 1. Grid
grid = OceanGrid.create(
    lon_bounds   = (-20.0, 20.0),
    lat_bounds   = (-10.0, 10.0),
    depth_levels = np.linspace(25, 475, 10),
    Nx=8, Ny=6,
)

# 2. Initial state (uniform T=20 C, S=35 psu, no motion)
state  = create_rest_state(grid, T_background=20.0, S_background=35.0)
params = ModelParams(dt=300.0)

# 3. Surface forcing: seasonal heat flux + constant wind stress
N_STEPS = 30 * 288   # 30 days at dt=300 s
forcing = make_synthetic_forcing(
    grid, N_STEPS, dt=300.0,
    heat_flux = {"mean": -50.0, "amplitude": 100.0, "period": 365 * 86400},
    tau_x     = -0.05,
)

# 4. Run (first call compiles ~20-60 s; subsequent calls are fast)
run_jit  = jax.jit(run, static_argnames=("n_steps", "save_history"))
final, _ = run_jit(state, grid, params, n_steps=N_STEPS,
                   forcing_sequence=forcing, save_history=False)

# 5. Inspect results
T_surf = np.array(final.T[:, :, 0])
print(f"Final SST range: [{T_surf.min():.2f}, {T_surf.max():.2f}] C")
```

---

## 5. Step-by-Step Usage Guide

### 5.1 Build a Grid

```python
from OceanJAX.grid import OceanGrid
import numpy as np

grid = OceanGrid.create(
    lon_bounds   = (-40.0, -5.0),    # (lon_min, lon_max) degrees east
    lat_bounds   = (-15.0, 15.0),    # (lat_min, lat_max) degrees north
    depth_levels = np.array([25., 75., 150., 250., 375., 500.]),  # cell centres [m]
    Nx           = 20,               # cells in x (longitude)
    Ny           = 15,               # cells in y (latitude)
    bathymetry   = None,             # optional (Nx, Ny) array of water depths [m];
                                     # None -> flat bottom half a cell below the deepest centre
    periodic_x   = True,             # False closes the east and west boundaries with walls
)
```

**Key grid attributes:**

| Attribute | Shape | Description |
|---|---|---|
| `grid.Nx, Ny, Nz` | scalar | Grid dimensions |
| `grid.lon_c, lat_c, z_c` | 1-D | Cell-centre coordinates |
| `grid.mask_c` | `(Nx, Ny, Nz)` | 1 = ocean cell, 0 = land or below bathymetry |
| `grid.volume_c` | `(Nx, Ny, Nz)` | Cell volumes [m^3] |
| `grid.H` | `(Nx, Ny)` | Water depth per column [m] |

**Vertical levels.** `depth_levels` are cell centres; `face_depths(centres)` gives the interfaces (midway between centres, with the bottom face half a cell below the last centre). Helpers for non-uniform levels:

```python
from OceanJAX.grid import stretched_levels, barotropic_cfl, CFL_LIMIT

# Geometrically stretched: ~10 m at the surface, bottom face exactly at 4000 m
z_c = stretched_levels(depth_max=4000.0, nz=30, dz_top=10.0)

# ORAS5's own levels, thinned to >= 10 m per cell (see Section 7)
# z_c = oras5_levels(raw, depth_max=4000.0, dz_min=10.0)

grid = OceanGrid.create((-80.0, 20.0), (-50.0, 60.0), z_c, Nx=100, Ny=80,
                        periodic_x=False)

cfl, dt_max = barotropic_cfl(grid, dt=90.0)   # keep cfl <= CFL_LIMIT (0.33)
```

Keep the top cell at least ~10 m thick (the linear free surface requires |eta| much smaller than the top cell) and the model bottom within the ORAS5 data (~5900 m) when initialising from ORAS5.

**Recommended grid sizes:**

| Application | Nx x Ny x Nz | Recommended dt |
|---|---|---|
| Testing / verification | 5-10 x 5-8 x 6 | 300 s |
| Regional experiment | 20 x 15 x 10 | 300 s |
| Basin-scale, 1°, 4000 m | 100 x 80 x 30-50 | 90 s |

---

### 5.2 Initialize Model State

```python
from OceanJAX.state import create_rest_state, create_from_arrays
import jax.numpy as jnp

# Option A — Idealised uniform state (no data needed)
state = create_rest_state(grid, T_background=15.0, S_background=35.0)

# Option B — Custom arrays
T_init   = ...   # (Nx, Ny, Nz) float32  [C]
S_init   = ...   # (Nx, Ny, Nz) float32  [psu]
u_init   = jnp.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=jnp.float32)
v_init   = jnp.zeros_like(u_init)
eta_init = jnp.zeros((grid.Nx, grid.Ny), dtype=jnp.float32)

state = create_from_arrays(grid, u=u_init, v=v_init,
                            T=T_init, S=S_init, eta=eta_init)

# Option C — ORAS5 cold or full start (see Section 7)
```

**OceanState fields:**

| Field | Shape | Description |
|---|---|---|
| `state.T` | `(Nx, Ny, Nz)` | Temperature [C] |
| `state.S` | `(Nx, Ny, Nz)` | Salinity [psu] |
| `state.u`, `state.v` | `(Nx, Ny, Nz)` | Horizontal velocities [m/s] |
| `state.w` | `(Nx, Ny, Nz+1)` | Vertical velocity [m/s], positive downward |
| `state.eta` | `(Nx, Ny)` | Free surface height [m] |
| `state.time` | scalar | Elapsed model time [s] |
| `state.step_count` | scalar | Number of steps taken |

---

### 5.3 Set Model Parameters

```python
from OceanJAX.state import ModelParams
from OceanJAX.Physics.mixing import munk_viscosity

params = ModelParams(
    dt      = 300.0,                 # time step [s] — check with barotropic_cfl (Section 13)
    kappa_h = 100.0,                 # horizontal tracer diffusivity [m^2/s]
    kappa_v = 1e-5,                  # background vertical tracer diffusivity [m^2/s]
    nu_h    = munk_viscosity(grid),  # horizontal viscosity [m^2/s], resolution dependent
    nu_v    = 1e-4,                  # background vertical viscosity [m^2/s]
    vertical_mixing = "pp81",        # "constant" | "pp81" (default) | "kpp" (Section 6)
    # All other parameters have physically validated defaults (see Section 14).
)
```

`nu_h` is a closure for unresolved momentum mixing, not a property of seawater, and must grow with grid spacing. `munk_viscosity(grid)` returns the smallest value that resolves the Munk western boundary layer, max(β Δx³) over wet columns. The `ModelParams` default of 200 m²/s is meant for tests; on a 2° grid it lets 2Δx noise grow at the western boundary.

---

### 5.4 Prepare Surface Forcing

For a multi-step run, each forcing field must have shape `(n_steps, Nx, Ny)` so that every time step can receive a different value.

**Option A — No forcing**
```python
forcing = None
```

**Option B — Constant forcing**
```python
from OceanJAX.timeStepping import SurfaceForcing
import jax.numpy as jnp, numpy as np

hf   = np.full((grid.Nx, grid.Ny), -80.0, dtype=np.float32)  # W/m^2
ones = np.ones((N_STEPS, 1, 1), dtype=np.float32)

forcing = SurfaceForcing(
    heat_flux = jnp.asarray(hf * ones),
    fw_flux   = jnp.zeros((N_STEPS, grid.Nx, grid.Ny), dtype=jnp.float32),
    tau_x     = jnp.asarray(np.full((N_STEPS, grid.Nx, grid.Ny), -0.05, dtype=np.float32)),
    tau_y     = jnp.zeros((N_STEPS, grid.Nx, grid.Ny), dtype=jnp.float32),
)
```

**Option C — Synthetic time-varying forcing**
```python
from OceanJAX.data.forcing import make_synthetic_forcing

forcing = make_synthetic_forcing(
    grid, N_STEPS, dt=300.0,
    heat_flux = {"mean": -50.0, "amplitude": 150.0, "period": 365*86400},
    tau_x     = -0.05,         # constant scalar
    tau_y     = wind_pattern,  # constant (Nx, Ny) array
    fw_flux   = 0.0,
)
```

**Option D — Monthly ORAS5 forcing** (recommended for realistic runs; see Section 8)
```python
from OceanJAX.data.monthly_forcing import MonthlyForcing

mf = MonthlyForcing("OceanJAX/data/data_oras5", grid, start_date="2026-01-16T12:00")
forcing = mf.chunk(t_start=0.0, n_steps=N_STEPS, dt=params.dt)
```

**Sign conventions for SurfaceForcing:**

| Field | Unit | Positive = |
|---|---|---|
| `heat_flux` | W m^-2 | Ocean gains heat |
| `fw_flux` | m s^-1 | Net evaporation (salinity increases) |
| `tau_x` | N m^-2 | Eastward wind stress |
| `tau_y` | N m^-2 | Northward wind stress |

The heat flux is applied to the top cell. The freshwater flux enters as a virtual salt flux scaled by the local surface salinity, dS/dt = SSS · (E − P) / Δz_top (the linear-free-surface form used in NEMO). Dilution is therefore proportional to salinity, and strong freshwater input (river runoff is included in the ORAS5 E − P) cannot drive salinity negative.

---

### 5.5 Run the Model

```python
import jax
from OceanJAX.timeStepping import run

# JIT-compile once; reuse for all calls with the same n_steps
run_jit = jax.jit(run, static_argnames=("n_steps", "save_history"))

final_state, history = run_jit(
    state,
    grid,
    params,
    n_steps          = N_STEPS,
    forcing_sequence = forcing,    # SurfaceForcing or None
    save_history     = False,      # True -> history has a leading time axis
)
```

**Chunked runs (recommended for long integrations and per-day diagnostics):**

```python
CHUNK = 288   # 1 day at dt=300 s

for day in range(N_DAYS):
    forcing = mf.chunk(t_start=day * CHUNK * params.dt, n_steps=CHUNK, dt=params.dt)
    state, _ = run_jit(state, grid, params,
                       n_steps=CHUNK,
                       forcing_sequence=forcing,
                       save_history=False)
    T_arr = np.array(state.T)
    print(f"Day {day+1:3d}: SST = {T_arr[:,:,0].mean():.3f} C  "
          f"eta_max = {np.abs(np.array(state.eta)).max():.4f} m")
```

---

### 5.6 Save Output

```python
import netCDF4 as nc
import numpy as np

ds = nc.Dataset("output.nc", "w", format="NETCDF4")
ds.createDimension("time", None)
ds.createDimension("x", grid.Nx)
ds.createDimension("y", grid.Ny)
ds.createDimension("z", grid.Nz)

ds.createVariable("time", "f4", ("time",));       ds["time"].units = "days"
ds.createVariable("T",    "f4", ("time","x","y","z")); ds["T"].units = "degC"
ds.createVariable("S",    "f4", ("time","x","y","z")); ds["S"].units = "psu"
ds.createVariable("eta",  "f4", ("time","x","y"));     ds["eta"].units = "m"

i = len(ds["time"])
ds["time"][i] = float(state.time) / 86400.0
ds["T"][i]    = np.array(state.T)
ds["S"][i]    = np.array(state.S)
ds["eta"][i]  = np.array(state.eta)
ds.sync()
ds.close()
```

A complete experiment driver with per-day diagnostics and full NetCDF output is available in `experiment.py`: edit its CONFIG block (domain, levels, mixing, initial state, forcing, output) and run `python experiment.py`. With `SAVE_DAILY_3D = False` it writes only surface fields per snapshot, and `MONTHLY_NC` adds calendar-month means of T, S, u, v and eta together with a description of the grid. Standard figures can be generated with:

```bash
python plot_output.py   # reads output_*.nc from the working directory
```

---

## 6. Vertical Mixing Schemes

Selected with `ModelParams(vertical_mixing=...)`. All schemes are solved implicitly, column by column, for both momentum and tracers.

| Scheme | Description |
|---|---|
| `"constant"` | `nu_v`, `kappa_v` everywhere. Use when a test needs exactly constant diffusivity (e.g. an analytical diffusion solution or κ = 0). |
| `"pp81"` (default) | Pacanowski & Philander (1981): ν = ν₀/(1+αRi)ⁿ + ν_b, κ = ν₀/(1+αRi)ⁿ⁺¹ + κ_b, with `nu_v` / `kappa_v` as backgrounds. Where the column is statically unstable the coefficients blend smoothly into a convective value of 0.1 m²/s over −10⁻⁶ < N² < 0 s⁻², so round-off-level changes in N² cannot flip the scheme between branches. |
| `"kpp"` | KPP surface boundary layer (Large, McWilliams & Doney 1994) on top of PP81 + convection. |

**KPP in OceanJAX**

- **Surface forcing.** The friction velocity u\* comes from the wind stress, and the surface buoyancy flux B₀ from the same heat and freshwater fluxes that force the tracers.
- **Boundary-layer depth h.** h is the shallowest depth where the bulk Richardson number reaches `kpp_ri_crit` (0.3), interpolated between cell centres and including the unresolved turbulent shear of LMD94 eq. (23). Under stabilising forcing, h is further limited by the Ekman depth 0.7 u\*/|f| and the Monin–Obukhov length.
- **Diffusivity inside h.** Within the boundary layer, K = max(PP81, h · w(σ) · G(σ)), with the shape function G(σ) = σ(1 − σ)² and the turbulent velocity scales of LMD94.
- **Non-local transport.** Under convective forcing (B₀ < 0), a non-local heat and salt flux is added inside h. It integrates to zero over each column, so heat and salt are conserved exactly.
- **No forcing.** Without surface forcing, KPP reduces bit-for-bit to PP81.
- **Gradients.** All branches are written so that gradients stay finite, including in float32 reverse mode.
- **Domain decomposition.** Only column operations and nearest-neighbour averages are used, so no extra communication is needed.

`mixing.kpp_boundary_layer_depth(...)` returns h for diagnostics.

---

## 7. Using ORAS5 Reanalysis Data

### 7.1 Download Data

ORAS5 monthly means on the native ORCA025 grid are distributed by the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu) (dataset `reanalysis-oras5`). Files are named `<var>_control_monthly_highres_<2D|3D>_<YYYYMM>_*.nc`.

| Variable | File name pattern | Used for |
|---|---|---|
| Temperature | `votemper_*.nc` | Initial T |
| Salinity | `vosaline_*.nc` | Initial S |
| Zonal velocity | `vozocrtx_*.nc` | Full-state init only |
| Meridional velocity | `vomecrty_*.nc` | Full-state init only |
| Sea surface height | `sossheig_*.nc` | Full-state init only |
| Net heat flux | `sohefldo_*.nc` | Surface forcing |
| Freshwater flux | `sowaflup_*.nc` | Surface forcing |
| Zonal wind stress | `sozotaux_*.nc` | Surface forcing |
| Meridional wind stress | `sometauy_*.nc` | Surface forcing |

Place all files in `OceanJAX/data/data_oras5/` (NetCDF files there are ignored by git).

### 7.2 Grid, Levels and Initial Conditions

```python
import warnings
import jax.numpy as jnp
from OceanJAX.data.oras5 import (oras5_month_files, read_oras5, oras5_levels,
                                 oras5_grid, regrid_to_model)
from OceanJAX.state import create_from_arrays

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # A merged oras5_YYYY_MM_native_merged.nc is used if present,
    # otherwise the per-variable files of that month are merged in memory.
    files = oras5_month_files("OceanJAX/data/data_oras5", 2026, 1)
    raw   = read_oras5(files)

    levels = oras5_levels(raw, depth_max=4000.0, dz_min=10.0)   # ORAS5 levels, >= 10 m thick
    grid   = oras5_grid(raw, (-80.0, 20.0), (-50.0, 60.0), levels,
                        Nx=100, Ny=80)                          # mask + bathymetry from ORAS5
    full_state = regrid_to_model(raw, grid)

# Cold start — T/S from ORAS5, zero velocity and sea level (most stable)
zeros3 = jnp.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=jnp.float32)
zeros2 = jnp.zeros((grid.Nx, grid.Ny),           dtype=jnp.float32)
state  = create_from_arrays(grid, u=zeros3, v=zeros3,
                             T=full_state.T, S=full_state.S, eta=zeros2)

# Full-state start — T/S/u/v/eta all from ORAS5
state = full_state
```

`oras5_grid` takes the land mask and bathymetry (the median ORAS5 depth within each model cell) from the data and closes the east and west boundaries by default. To merge the per-variable files on disk instead, run `python OceanJAX/data/data_oras5/data_merger.py`.

### 7.3 Surface Forcing

Use `MonthlyForcing` (Section 8) for runs longer than a few days. For a single fixed snapshot:

```python
from OceanJAX.data.oras5 import read_oras5_forcing, regrid_forcing
from OceanJAX.timeStepping import SurfaceForcing
import numpy as np

FORCING_FILES = [
    "OceanJAX/data/data_oras5/sohefldo_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sowaflup_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sozotaux_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sometauy_control_monthly_highres_2D_202601_OPER_v0.1.nc",
]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw_f = read_oras5_forcing(FORCING_FILES, time_index=0)
    sf    = regrid_forcing(raw_f, grid,
                           use_fields={"heat_flux", "fw_flux", "tau_x", "tau_y"})

# Broadcast the single snapshot to all N_STEPS
ones = np.ones((N_STEPS, 1, 1), dtype=np.float32)
forcing = SurfaceForcing(
    heat_flux = jnp.asarray(np.array(sf.heat_flux) * ones),
    fw_flux   = jnp.asarray(np.array(sf.fw_flux)   * ones),
    tau_x     = jnp.asarray(np.array(sf.tau_x)     * ones),
    tau_y     = jnp.asarray(np.array(sf.tau_y)     * ones),
)
```

---

## 8. Time-Varying Forcing

### `MonthlyForcing` — calendar-aware ORAS5 forcing

```python
from OceanJAX.data.monthly_forcing import MonthlyForcing

mf = MonthlyForcing(
    "OceanJAX/data/data_oras5", grid,
    start_date = "2026-01-16T12:00",   # calendar date of model time 0
    interp     = "linear",             # or "monthly" (piecewise constant)
)
forcing = mf.chunk(t_start=0.0, n_steps=CHUNK, dt=params.dt)   # SurfaceForcing (CHUNK, Nx, Ny)
```

- **Loading.** Every month that has all four forcing files in the folder is read and regridded once, then cached.
- **Interpolation.** With `"linear"`, the monthly means sit at mid-month and are interpolated linearly in between. This avoids flux jumps of O(50–100 W m⁻²) at month boundaries.
- **Missing months.** A missing month falls back to the same calendar month of the nearest available year, and then to the nearest available month in time.
- **Memory.** Forcing is assembled one chunk at a time, so memory stays at a single chunk.

### `make_synthetic_forcing` — analytical specification

```python
from OceanJAX.data.forcing import make_synthetic_forcing

forcing = make_synthetic_forcing(
    grid, n_steps, dt,
    heat_flux = {"mean": -50.0, "amplitude": 150.0,
                 "period": 365*86400, "phase": 0.0},
    tau_x     = -0.05,           # constant scalar
    tau_y     = wind_pattern,    # constant (Nx, Ny) spatial array
    fw_flux   = 0.0,
)
```

Each field parameter accepts:

| Type | Behaviour |
|---|---|
| `float` | Spatially and temporally constant |
| `np.ndarray (Nx, Ny)` | Spatially varying, constant in time |
| `dict` | `mean + amplitude * sin(2*pi*t/period + phase)`, each can be scalar or `(Nx, Ny)` |

### `make_forcing_sequence` — interpolation from arbitrary snapshots

```python
from OceanJAX.data.forcing import make_forcing_sequence

snapshots = [(0.0, sf_january), (30*86400, sf_february)]   # (time [s], SurfaceForcing)
forcing = make_forcing_sequence(
    snapshots, n_steps=N_STEPS, dt=300.0,
    interp  = "cyclic",   # repeat the cycle indefinitely
    t_start = 0.0,
)
```

| `interp` mode | Behaviour |
|---|---|
| `"linear"` | Piecewise-linear between snapshots; clamps at edges |
| `"nearest"` | Step function; uses the nearest snapshot value |
| `"cyclic"` | Linear + periodic wrapping; ideal for repeating annual cycles |

---

## 9. ML Closure Interface

The model exposes a single hook per time step for injecting learnable parameterizations. The hook fires after the explicit tracer tendencies are computed and before the implicit vertical diffusion.

### Define a closure

```python
import equinox as eqx
import jax.numpy as jnp
from OceanJAX.ml.closure import AbstractClosure, ClosureOutput

class MyClosure(AbstractClosure):
    weights: jnp.ndarray   # any equinox Module fields

    def __call__(self, state, grid, params) -> ClosureOutput:
        dT_tend = self.weights * state.T
        return ClosureOutput(
            dT_tend       = dT_tend,
            dS_tend       = jnp.zeros_like(dT_tend),
            kappa_v_scale = jnp.array(1.0),
        )
```

### Use a closure

```python
closure = MyClosure(weights=jnp.zeros((grid.Nx, grid.Ny, grid.Nz)))

final, _ = run_jit(state, grid, params, n_steps=N_STEPS,
                   forcing_sequence=forcing, closure=closure)
```

### ClosureOutput fields

| Field | Shape | Description |
|---|---|---|
| `dT_tend` | `(Nx, Ny, Nz)` [K/s] | Added to explicit T tendency before AB3 advance |
| `dS_tend` | `(Nx, Ny, Nz)` [psu/s] | Added to explicit S tendency before AB3 advance |
| `kappa_v_scale` | scalar or `(Nx, Ny, Nz+1)` | Multiplies `params.kappa_v` before implicit diffusion |

### No-op closure (for testing)

```python
from OceanJAX.ml.closure import NullClosure

closure = NullClosure()
# Produces bit-identical results to closure=None.
```

### Training with gradients

```python
import optax, equinox as eqx

@eqx.filter_jit
@eqx.filter_grad
def loss_fn(closure, state, target_T):
    final, _ = run(state, grid, params, n_steps=N_STEPS,
                   forcing_sequence=forcing, closure=closure)
    return jnp.mean((final.T - target_T) ** 2)

grads     = loss_fn(closure, state, target_T)
optim     = optax.adam(1e-4)
opt_state = optim.init(eqx.filter(closure, eqx.is_array))
updates, opt_state = optim.update(grads, opt_state)
closure   = eqx.apply_updates(closure, updates)
```

The test suite checks gradients through short multi-step runs (including on a sharded domain). For long integrations, wrap the step in `jax.checkpoint` to bound memory.

---

## 10. Ensemble / Multi-Device Runs

```python
from OceanJAX.parallel.ensemble import batch_run, sharded_ensemble_run
import jax

N_MEMBERS = 8

# Stack N_MEMBERS copies of the initial state along a leading batch axis
batched_state = jax.tree_util.tree_map(
    lambda x: jnp.stack([x] * N_MEMBERS), state
)

# Option A: single device (vmap over the batch axis)
final_batch, _ = batch_run(
    batched_state, grid, params,
    n_steps=N_STEPS, forcing_sequence=None, save_history=False,
)

# Option B: multiple devices (NamedSharding; N_MEMBERS must be divisible by n_devices)
final_batch, _ = sharded_ensemble_run(
    batched_state, grid, params,
    n_steps=N_STEPS, forcing_sequence=None, save_history=False,
)

# final_batch.T has shape (N_MEMBERS, Nx, Ny, Nz)
T_mean = jnp.mean(final_batch.T, axis=0)
T_std  = jnp.std( final_batch.T, axis=0)
```

### Domain decomposition (one domain over several devices)

```python
from OceanJAX.parallel.sharding import make_mesh, shard_grid, shard_state, sharded_run

mesh   = make_mesh(n_x=4, n_y=2)            # Nx % 4 == 0, Ny % 2 == 0
grid_s = shard_grid(grid, mesh)
state  = shard_state(state, mesh)
final, _ = sharded_run(state, grid_s, params, N_STEPS, mesh,
                       forcing_sequence=forcing)
```

In `experiment.py` set `N_DEVICES_X` / `N_DEVICES_Y`. A 1 × 1 mesh is bit-identical to `jax.jit(run)`; on more devices results differ only at round-off level, and the implicit vertical solves need no communication. See [docs/parallel.md](docs/parallel.md) for how it works, numerical agreement, and running on a multi-node SLURM cluster.

---

## 11. Hindcast Evaluation against ORAS5

`verification_experiments/exp_hindcast.py` runs an ORAS5-initialised hindcast and scores its monthly means against ORAS5.

```bash
python verification_experiments/exp_hindcast.py                  # PP81, 16 Jan - 1 Jul 2026
python verification_experiments/exp_hindcast.py --mixing kpp     # with the KPP boundary layer
python verification_experiments/exp_hindcast.py --days 20        # short smoke test
python verification_experiments/exp_hindcast.py --eval-only      # re-score an existing run
```

**Setup**

| Item | Setting |
|---|---|
| Domain and grid | Atlantic 80°W–20°E, 50°S–60°N at 1° (100 × 80); closed east and west walls |
| Vertical levels | ORAS5 levels to 4000 m, thinned to cells ≥ 10 m thick (49 levels) |
| Land mask and bathymetry | from ORAS5 |
| Time step | 90 s (barotropic CFL 0.31) |
| Initial state | full ORAS5 January 2026 state, taken as 16 January 12:00 |
| Forcing | monthly ORAS5 fluxes for January–June, interpolated between mid-months |
| Physics | PP81 (or KPP), Munk ν_h, bottom drag, freezing limit |
| Restoring | none, because the evaluation target is ORAS5 SST itself |

**Evaluation (`hindcast_eval.py`)**

- **Regridding.** The grid is rebuilt from the description stored in the monthly-mean file, and ORAS5 monthly means are interpolated onto the model grid and levels.
- **Variables.** SST, SSS, T and S near 100, 300 and 1000 m, SSH (minus its area mean) and surface current speed.
- **Scores.** All scores are area-weighted (`OceanJAX.diagnostics`): bias, RMSE and pattern correlation; skill = 1 − RMSE_model / RMSE_persistence, where persistence is the January ORAS5 state carried forward; and change correlation, i.e. the correlation between the model's and ORAS5's change since January.
- **Regions.** The whole domain, the tropics, north of 20°N and south of 20°S.
- **Outputs.** `metrics.csv`, `summary.md`, `rmse_by_month.png` and maps of ORAS5, model error, ORAS5 change and model change for each month.

**Current results (PP81, whole domain, area weighted)**

| Variable | Feb RMSE (persistence) | Feb change corr | Jun RMSE (persistence) | Jun change corr |
|---|---|---|---|---|
| SST [°C] | 1.69 (0.94) | 0.65 | 3.77 (2.81) | 0.81 |
| SSS [psu] | 0.49 (0.34) | 0.28 | 0.69 (0.64) | 0.52 |
| T ~100 m [°C] | 1.01 (0.87) | 0.50 | 1.69 (1.42) | 0.47 |
| T ~300 m [°C] | 0.65 (0.45) | 0.36 | 0.94 (0.66) | 0.38 |
| SSH [m] | 0.17 (0.05) | 0.00 | 0.15 (0.07) | 0.28 |

The run is stable for the full 165 days (about 52 min on CPU). The model gets the **pattern** of the seasonal change right (June SST change correlation 0.81) but not its **amplitude**, and it does not yet beat persistence:

- **Summer warm bias.** North of 20°N, June SST is too warm by +4.3 °C on average and by 5–15 °C locally. All surface heating, including shortwave, is deposited in the ~10 m top cell, and the fixed fluxes provide no SST feedback.
- **KPP.** KPP helps only marginally (June SST RMSE 3.77 → 3.75 °C). Under strong summer heating and weak monthly-mean winds, the Monin–Obukhov length confines the boundary layer to about one cell.
- **SSH.** SSH adjusts to the model's own balance within the first days and then stays put.

The planned next steps are shortwave penetration (two-band Jerlov profile) and an SST feedback term.

---

## 12. Running Verification Tests

```bash
# Full test suite
python -m pytest OceanJAX/tests/ -v

# Individual modules
python -m pytest OceanJAX/tests/test_grid.py            -v
python -m pytest OceanJAX/tests/test_dynamics.py        -v
python -m pytest OceanJAX/tests/test_tracers.py         -v
python -m pytest OceanJAX/tests/test_mixing.py          -v   # implicit solve, PP81, KPP, bottom drag
python -m pytest OceanJAX/tests/test_timeStepping.py    -v
python -m pytest OceanJAX/tests/test_closure.py         -v
python -m pytest OceanJAX/tests/test_forcing.py         -v
python -m pytest OceanJAX/tests/test_monthly_forcing.py -v
python -m pytest OceanJAX/tests/test_oras5.py           -v
python -m pytest OceanJAX/tests/test_diagnostics.py     -v
python -m pytest OceanJAX/tests/test_parallel.py        -v
python -m pytest OceanJAX/tests/test_sharding.py        -v   # 8 simulated CPU devices

# Verification experiments
python verification_experiments/exp_5_1_1_rest_conservation.py
python verification_experiments/exp_5_1_2_stratified_bathymetry.py
python verification_experiments/exp_5_1_3_null_closure.py
python verification_experiments/exp_5_1_4_cfl_scan.py
python verification_experiments/exp_5_1_5_vertical_mixing.py
python verification_experiments/exp_5_2_1_cold_full_forcing.py
python verification_experiments/exp_5_2_2_cold_no_forcing.py
python verification_experiments/exp_5_2_3_full_no_forcing.py
python verification_experiments/exp_extra_atlantic_180d.py
python verification_experiments/exp_hindcast.py           # Section 11
```

The experiments that read ORAS5 need the data in `OceanJAX/data/data_oras5/` (Section 7); the unit tests use synthetic data only.

---

## 13. Stability Guidelines

The model uses leapfrog + Asselin-Robert for momentum and free surface, and Adams-Bashforth 3 for tracers. The critical constraint is the barotropic CFL number:

```
CFL = c_bt * dt / dx_min

where  c_bt    = sqrt(g * H_max)    barotropic wave speed [m/s]
       dx_min  = minimum horizontal grid spacing [m]
```

`grid.barotropic_cfl(grid, dt)` returns the CFL number and the largest stable dt, and `experiment.py` prints it at start-up and warns above the limit.

| CFL range | Outcome |
|---|---|
| <= 0.33 | Stable |
| 0.33 – 0.44 | Use with caution |
| > 0.44 | Unstable — NaN within first day |

**Validated configurations:**

| DEPTH_MAX | Approx. dx_min | dt | Notes |
|---|---|---|---|
| 500 m | 200 km | 900 s | Stability limit (1200 s fails) |
| 1000 m | 200 km | 600 s | |
| 500 m | 200 km | **300 s** | 3x safety margin |
| 4000 m | 57 km (1° at 60°N) | 90 s | Atlantic hindcast, CFL 0.31 |
| 4000 m | 28 km (0.5° at 60°N) | 45 s | GPU benchmark runs (short) |

**Rules of thumb:**
- Choose dt from `barotropic_cfl` rather than a fixed value; deep, fine grids need much shorter steps.
- Cold-start initial conditions (`u = v = eta = 0`) are more forgiving than full-state ORAS5 initialization.
- Keep the top cell at least ~10 m thick and the model bottom within the ORAS5 data; deeper levels are filled by extrapolation and can drive spurious currents.

---

## 14. ModelParams Reference

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `dt` | 900.0 | s | Time step (choose with `barotropic_cfl`) |
| `rho0` | 1025.0 | kg/m^3 | Reference density (Boussinesq) |
| `g` | 9.81 | m/s^2 | Gravitational acceleration |
| `alpha_T` | 2e-4 | 1/K | Thermal expansion coefficient |
| `beta_S` | 7.4e-4 | 1/psu | Haline contraction coefficient |
| `T_ref` | 10.0 | C | EOS reference temperature |
| `S_ref` | 35.0 | psu | EOS reference salinity |
| `nu_h` | 200.0 | m^2/s | Horizontal viscosity (tests only; use `munk_viscosity(grid)`) |
| `nu_v` | 1e-4 | m^2/s | Vertical viscosity (background for PP81 / KPP) |
| `kappa_h` | 100.0 | m^2/s | Horizontal tracer diffusivity |
| `kappa_v` | 1e-5 | m^2/s | Vertical tracer diffusivity (background for PP81 / KPP) |
| `vertical_mixing` | "pp81" | — | "constant", "pp81" or "kpp" (Section 6) |
| `pp81_nu0`, `pp81_alpha`, `pp81_n` | 1e-2, 5, 2 | m^2/s, —, — | PP81 shear-mixing amplitude, Ri coefficient, exponent |
| `vmix_convective` | 0.1 | m^2/s | Convective nu = kappa where statically unstable |
| `vmix_n2_ramp` | 1e-6 | s^-2 | N² width of the continuous blend into convection (0 = hard switch) |
| `kpp_ri_crit` | 0.3 | — | Critical bulk Richardson number for the KPP boundary-layer depth |
| `bottom_drag_cd` | 1e-3 | — | Quadratic bottom drag coefficient (0 = free slip) |
| `bottom_drag_ubg` | 0.05 | m/s | Background speed for unresolved tides and eddies |
| `freezing_slope` | 0.0575 | K/psu | Freezing point T_f = −slope · S |
| `limit_freezing` | True | — | Keep T ≥ T_f (stands in for sea ice) |
| `asselin_coeff` | 0.1 | — | Asselin-Robert filter coefficient |
| `ab3_coeffs` | (23/12, -16/12, 5/12) | — | Adams-Bashforth 3 coefficients |
