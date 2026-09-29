# CLAUDE.md — OceanJAX 项目记忆

本文件是跨会话（本地 / cloud session）共享的项目记忆。**每次会话结束前按第 9 节更新并推送。**
最后更新：2026-09-29（cloud 会话），PP81 连续过渡 + 默认 PP81（分支 `pp81-continuous-mixing`，基于 `ff3dde2` = PR #1 并行阶段 A 合并后的 main）。

---

## 1. 项目简介

OceanJAX：用 JAX 写的三维可微分海洋模式（Boussinesq 静力原始方程，球坐标 Arakawa C 网格），
目标是 ML 次网格参数化和基于伴随的参数优化。课程/研究项目 EAEE9280，论文 PDF 在仓库根目录。

- GitHub：`https://github.com/Ethreall418/OceanJAX-3D_differentiable_ocean_emulator`（旧名 EAEE9280_Personal_Research_Project 会自动跳转）
- 标签：`thesis-v1` = `c80fa76`（论文 PDF 对应状态，所有 2026-09-28 修复之前）；`numerics-fix-v2` = `22c9e50`
- 回到论文版本：`git switch -c thesis-check thesis-v1`

## 2. 协作约定（用户偏好，务必遵守）

- **用中文沟通**；代码、注释、docstring、commit message 用英文。
- **先提方案 → 等用户明确认可（"可以"/"我认可"）→ 再写代码**。小 bug 修复可直接做。不要追问可选参数的细节。
- 回复简洁，不要复述刚做过的事。
- 用户说"**汇报模型进度**"时：先重新读关键模块，再按固定格式回复：
  物理方程 → 模块结构 → 各模块职责 → 时间步顺序 → 数据层 → 能力表格（✅/❌）。
- **提交规则**：只提交必要文件（模式代码、正式单元测试、实验脚本、文档）。
  临时的诊断/排查脚本不进仓库（放系统临时目录）。`.nc` 数据和输出文件不提交（数 GB，超 GitHub 100 MB 限制）。
  commit message 用英文、详细（标题 + 分条说明做了什么/为什么/效果）。推送前需用户同意。
- 用户会仔细审查物理与数值细节；发现问题要如实说明，包括自己之前的判断错误。

## 3. 环境

- 本地：Windows 11，项目 venv `.venv`（Python 3.14，JAX 0.9.2，equinox 0.13.6），**仅 CPU**。
  原生 Windows 的 JAX 不支持 CUDA；WSL 已装平台但**没有 Linux 发行版**（用户需自行 `wsl --install -d Ubuntu-24.04`，GPU 暂缓）。
  硬件：RTX 5090 24 GB（目前用不上），24 逻辑核。
- **Cloud session 注意**：ORAS5 数据（`OceanJAX/data/data_oras5/*.nc`，约 3.5 GB）**不在仓库里**。
  因此 cloud 上：单元测试可跑（`test_oras5` / `test_monthly_forcing` 用合成数据）；
  `verification_experiments/*` 和 `experiment.py` 的 ORAS5 模式**跑不了**（缺数据）。
- Cloud 环境（2026-09-29）：Python 3.11、JAX 0.10.2、equinox 0.13.8，4 核 CPU，无 GPU。
  依赖需自己装：`pip install --ignore-installed packaging -r requirements.txt`（Debian 自带的 packaging 无法卸载）。
- 跑测试：`python -m pytest OceanJAX/tests -q`（当前 208 passed, 0 skipped；cloud 4 核约 6 min）。
  `tests/conftest.py` 设置 `XLA_FLAGS=--xla_force_host_platform_device_count=8`（8 个模拟 CPU 设备），
  单设备代码不受影响；`test_sharding.py` 需要这 8 个设备。
- 本地 `git stash@{0}`：旧的 OBC 半成品（`dynamics.py`/`tracers.py` 改动）；`OceanJAX/Physics/obc.py` 未跟踪。
  另：GitHub 上有远程分支 `feature/open-boundary-conditions`（`05a68fb`，基于 `thesis-v1`，8 个文件 +676 行），
  内容未检查，与 2026-09-28 之后的修复和并行/PP81 改动未合并。
- `OceanJAX/data/data_oras5/vomecrty_…_3D_202601_…nc` 单独文件已损坏（非本会话造成）；合并文件 `oras5_2026_01_native_merged.nc` 完好，模式只读合并文件。

## 4. 代码结构

```
OceanJAX/
├── grid.py          OceanGrid：C 网格、度量、f、掩膜；periodic_x（False = 东西墙）
├── state.py         ModelParams（含 vertical_mixing / PP81 / 底摩擦 / 冰点参数）、OceanState、构造函数
├── operators.py     通量形式 grad/div/interp/laplacian
├── timeStepping.py  SurfaceForcing、step()、run()（lax.scan）
├── Physics/
│   ├── dynamics.py  EOS、静水压力、PGF、Coriolis（v_at_u_points / u_at_v_points）、自由面、compute_w
│   ├── tracers.py   迎风/中心平流、水平扩散、表面热盐强迫
│   └── mixing.py    增量式隐式垂直求解、底摩擦、水平粘性、munk_viscosity、PP81
├── data/
│   ├── oras5.py            read_oras5、regrid_to_model、oras5_bathymetry、oras5_grid、read/regrid_forcing
│   ├── forcing.py          make_forcing_sequence、make_synthetic_forcing
│   └── monthly_forcing.py  MonthlyForcing：按日历切换月份强迫
├── ml/closure.py    AbstractClosure / NullClosure / ClosureOutput（dT、dS、kappa_v_scale）
├── parallel/        ensemble.py（vmap 集合 + NamedSharding）；
│                    sharding.py（区域分解阶段 A：make_mesh、shard_grid/state/forcing、sharded_run、
│                    gather_to_host、init_distributed）；halo.py 为空占位（阶段 B）
└── tests/           10 个测试文件 + conftest.py（8 模拟设备）
experiment.py        实验主程序（CONFIG 区：NU_H="munk"、VERTICAL_MIXING="pp81"、FORCING_DIR、START_DATE…）
verification_experiments/  论文第 5 章验证实验（5.1.1–5.2.3 + extra_atlantic_180d）
runtime_test/benchmark_parallel.py  集合模式 / `domain` 区域分解模式 benchmark
docs/parallel.md     并行说明：GSPMD 原理、数值一致性、SLURM 多节点示例
```

## 5. 关键设计约定（改代码前必读）

- **z / k 向下为正**（k=0 在海面）。w **向下为正**。
- `compute_w`：**从海底 w[Nz]=0 向上积分**，海面 w[0] = −∂η/∂t；每个单元严格无散。
- `step()`：先由 `u_filt/v_filt` 诊断 w，再用同一组 (u, v, w) 做温盐平流。
- 温盐平流用 `mask_w`（**海面面开放**，线性自由面运动学通量 w₀C₀）；守恒量是 ΣCV + ΣηC_sA。
  `mask_w_adv`（海面关闭）**只用于隐式垂直扩散**。
- 静水压力只积分密度异常 ρ' = ρ − ρ₀。
- 隐式垂直求解 `_solve_increment`：解增量（右端只含差分），均匀水柱逐位不变。不要改回全场 Thomas 求解（会有 +1 ulp/步 的底层偏差）。
- N² = +(g/ρ₀)(ρ[k] − ρ[k−1])/dz_w（稳定为正）。旧的 `ri_based_diffusivity` 符号错误，已删除。
- 底摩擦：二次、隐式，`bottom_drag_cd=1e-3`、`bottom_drag_ubg=0.05`，作用于每列最深湿层；Cd=0 与无摩擦逐位一致。
- 冰点限制：T ≥ −0.0575·S（`limit_freezing=True`）。
- **ν_h 是随分辨率变化的涡粘性闭合**（不是海水物性）：ORAS5 实验用 `munk_viscosity(grid)` = max(β·Δx³)。
  `ModelParams` 默认 nu_h=200 仅供测试。nu_h=200 在 2° 网格上会导致西边界 2Δ 噪声指数增长。
- 垂直混合：`ModelParams(vertical_mixing="constant"|"pp81")`，**默认 pp81**（2026-09-29 起）。
  PP81：ν = ν₀/(1+αRi)ⁿ + ν_b，κ = ν₀/(1+αRi)ⁿ⁺¹ + κ_b；动量和温盐都用。
  对流：N² < 0 时用 smoothstep 在 [−N²_c, 0] 内从 PP81(Ri=0) 连续过渡到 0.1，N²_c = `vmix_n2_ramp` = 1e-6 s⁻²
  （0 = 旧的硬开关）；N² ≥ 0 的面与硬开关逐位一致。
  N² 由温盐差分计算：N² = g(−αΔT + βΔS)/dz_w（线性 EOS 下与 ρ 差分等价，避免两个 ~1025 相减，
  舍入噪声 1e-8 → 1e-10）。**若改非线性 EOS，`buoyancy_and_shear` 必须同步改。**
  已知残留：无剪切时稳定侧 Ri = N²/max(S², 1e-12) 在 N² ~1e-11 内从 0 升到很大，κ 从 0.01 陡降到背景值（PP81 本身的 0/0，未改）。
  需要"只有常数 κ"的测试/实验（κ_v=0 隔离、解析扩散解）必须显式写 `vertical_mixing="constant"`。
  垂直混合依赖**垂直**分辨率与边界层物理，不依赖水平分辨率；中高纬需要时再上 KPP。
- ORAS5 实验：`oras5_grid(raw, LON, LAT, levels, Nx, Ny, periodic_x=False)`（格内 ORAS5 水深中位数）。
- 多月强迫：`MonthlyForcing(dir, grid, start_date, interp="linear")`，月中为节点线性插值；
  缺月份 → 其他年同月 → 最近月份（目前只有 2026-01 → 永久一月）。只需下载 4 个 2D 文件/月。
- 时间步：动量与 η 为 leapfrog + Asselin（α=0.1），首步 1·dt；温盐 AB3（首两步 AB1/AB2）。
- **并行 / 区域分解（阶段 A）**：mesh 轴 ("batch","x","y")；凡是含 (Nx, Ny) 轴对的数组按 x/y 分片，其余复制；
  GSPMD 把 roll/移位 concatenate 编译成单格 halo 的 collective-permute。
  **不要把 x、y 两个轴 reshape 合并**（如 `reshape(Nx*Ny, Nz)`）——会触发整场 all-gather；
  逐列求解用 `mixing._vmap_columns`（嵌套 vmap）。z 轴永远不分片（列运算本地）。
  `sharded_run` 用 eqx.filter_jit，params 的 Python 浮点先转 0 维数组（`_traced_params`），否则常量折叠改变舍入。
  1×1 mesh 与 `jax.jit(run)` 逐位一致；多设备约 1 ulp/运算差异（XLA 融合不同），均匀静止态仍严格不变。
  整格计算量（如 `munk_viscosity`）在分片前的主机网格上算。

## 6. step() 顺序

1 EOS + p' → 2 动量显式趋势 + 风 → 3 leapfrog → 4 Asselin 得 u_filt → 5 垂直混合系数（constant/PP81）+ 隐式粘性 + 底摩擦
→ 6 由 u_filt 诊断 w → 7 温盐显式趋势 + 表面强迫 + ML closure → 8 AB3 → 9 隐式垂直扩散 + 冰点限制 → 10 η leapfrog + Asselin → 11 组装

## 7. 验证与稳定性现状（2026-09-28，constant 混合，除非注明）

| 实验 | 结果 |
|---|---|
| 5.1.1 静止均匀 | T/S **严格守恒**（判定已改为 exact） PASS；2026-09-29 改用 PP81 后在 cloud 重跑仍 PASS（全部严格为 0） |
| 5.1.2 层结 + 真实地形 | A（κ=0）u/v/η/T 严格为 0；B/C 单调 0=A<B<C（比值仅记录）PASS |
| 5.1.3 NullClosure | 逐位一致 PASS |
| 5.1.4 CFL | dt≤900 s 稳定，1200 s NaN |
| 5.1.5 垂直混合 | PASS |
| 5.2.1 / 5.2.2 / 5.2.3（30 d） | ΔSST −0.56 / −0.40 / −0.64 °C（Munk ν_h 后；旧版约 −2 °C 大部分是网格噪声） |
| 大西洋 2 年（PP81） | 最大流速 0.28–0.33 m/s，稳定；热、盐收支与外加通量吻合 |
| 热带 2 年（PP81） | 最大流速 0.43 m/s，稳定 |

论文第 5 章：5.1.1、5.1.2 判定方法已改；5.2 节数值与"降温与热通量一致"的解释需按新结果重写。

并行阶段 A 验证（2026-09-29，8 个模拟 CPU 设备，`test_sharding.py`）：
- 1×1 mesh 与单设备逐位一致；2×4 / 4×2 / 8×1 / 1×4 与单设备差 ≤ 约 1 ulp 量级（40 步，含地形、陆地、东西墙、PP81、底摩擦、强迫）。
- 静止均匀态分片后严格守恒；集合×区域（batch=2, 2×2）与 batch_run 一致；梯度与单设备一致（rtol 1e-4）。
- 编译后 HLO：无 all-gather / all-to-all，只有 halo 大小的 collective-permute 和一个单行 all-reduce；
  隐式垂直求解分片后**零通信**（旧 reshape 版本会 all-gather 整个 y 轴）。
- experiment.py（rest 模式、2 天）2×3 分解与单设备差 ≤ 2e-8。
- **发现：PP81 在 N²=0 处不连续**（剪切混合 ↔ 对流 0.1），N²≈0 时 1 ulp 差异可翻转分支，
  集合成员加 0.05 °C 网格尺度随机扰动时 2 天后差到 u ~3e-3、T ~4e-4；constant 混合时仅 ~1e-6。
  **已修复（2026-09-29）**：同一实验（experiment.py rest 模式，2 成员 × 2×1 分解 vs 2 设备集合）day 2 的 u 差：
  旧 2.7e-3 → 只改 N²（硬开关）7.5e-6 且增长 → 连续过渡 + 新 N² 1.9e-7（与 constant 相同，舍入量级）。
  `test_sharding.py::test_pp81_near_neutral_insensitive_to_roundoff` 为回归测试（硬开关 u 差 1.5e-5，现 ~9e-7）。

## 8. 已知问题与待办

- **表层热点**：固定热通量无 SST 反馈，停滞副热带格点 2 年后 > 40 °C。方案：Haney 恢复项
  Q = Q_ORAS5 + γ(SST_ORAS5 − SST)，γ≈40 W/m²/K（用户已理解，**推迟实现**）。
- ORCA 风应力沿网格 i/j 方向，高纬北大西洋未旋转到东/北。
- 可变分辨率下 ν_h 应随空间变化（目前全域取最大值）。
- 中高纬混合层：以后考虑 KPP。
- OBC：新分支计划（模型完善后再做），半成品在本地 stash。
- **并行计算**：阶段 A **已完成**（2026-09-29，见第 5、7 节与 `docs/parallel.md`）。
  待办：在真实多 GPU / 多节点集群上实测（`init_distributed` + SLURM 路径尚未实测）与 benchmark 效率；
  阶段 B（shard_map + 显式 halo 交换，`parallel/halo.py`）仅在集群实测效率不足时再做。
  experiment.py：`N_DEVICES_X/Y` > 1 时走 `sharded_run`，只有进程 0 打印和写 NetCDF。
- PP81 连续过渡已完成（见第 5、7 节）。验证脚本已改：5.1.1/5.1.3/5.1.4/5.2.x/大西洋 180 d 用 PP81，
  5.1.2、5.1.5 **保持 constant**（需要 κ=0 严格为零 / 检验 κ_v 缩放）。
  **需要在本地用 ORAS5 重跑**：5.1.3、5.1.4（CFL 上限可能变）、5.2.1–5.2.3、大西洋 180 d；论文第 5 章数值随之更新。
- `runtime_test/benchmark_parallel.py` 集合模式里 `batch_run` 未 jit，每次重新 trace，计时偏慢（旧问题，未改）。

## 9. 会话结束前（每次）

1. 更新本文件：第 7 节（结果）、第 8 节（待办）、顶部"最后更新"和对应提交。
2. 与代码一起 commit + push（推送需用户同意）。Cloud session 若推到新分支/PR，用户需在 GitHub 合并进 `main` 后本地 `git pull`。
