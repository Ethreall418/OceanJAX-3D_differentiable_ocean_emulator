# CLAUDE.md — OceanJAX 项目记忆

本文件是跨会话（本地 / cloud session）共享的项目记忆。**每次会话结束前按第 9 节更新并推送。**
最后更新：2026-09-28（本地会话），对应提交 `22c9e50` 之后。

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
- 跑测试：`python -m pytest OceanJAX/tests -q`（当前 174 passed, 1 skipped，约 90 s）。
- 本地 `git stash@{0}`：旧的 OBC 半成品（`dynamics.py`/`tracers.py` 改动）；`OceanJAX/Physics/obc.py` 未跟踪。**这两者都只在本地，不在 GitHub。**
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
├── parallel/        ensemble.py（vmap 集合 + NamedSharding）；halo.py、sharding.py 为空占位
└── tests/           9 个测试文件
experiment.py        实验主程序（CONFIG 区：NU_H="munk"、VERTICAL_MIXING="pp81"、FORCING_DIR、START_DATE…）
verification_experiments/  论文第 5 章验证实验（5.1.1–5.2.3 + extra_atlantic_180d）
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
- 垂直混合：`ModelParams(vertical_mixing="constant"|"pp81")`，默认 constant；experiment.py 用 pp81。
  PP81：ν = ν₀/(1+αRi)ⁿ + ν_b，κ = ν₀/(1+αRi)ⁿ⁺¹ + κ_b，N²<0 时取对流值 0.1；动量和温盐都用。
  垂直混合依赖**垂直**分辨率与边界层物理，不依赖水平分辨率；中高纬需要时再上 KPP。
- ORAS5 实验：`oras5_grid(raw, LON, LAT, levels, Nx, Ny, periodic_x=False)`（格内 ORAS5 水深中位数）。
- 多月强迫：`MonthlyForcing(dir, grid, start_date, interp="linear")`，月中为节点线性插值；
  缺月份 → 其他年同月 → 最近月份（目前只有 2026-01 → 永久一月）。只需下载 4 个 2D 文件/月。
- 时间步：动量与 η 为 leapfrog + Asselin（α=0.1），首步 1·dt；温盐 AB3（首两步 AB1/AB2）。

## 6. step() 顺序

1 EOS + p' → 2 动量显式趋势 + 风 → 3 leapfrog → 4 Asselin 得 u_filt → 5 垂直混合系数（constant/PP81）+ 隐式粘性 + 底摩擦
→ 6 由 u_filt 诊断 w → 7 温盐显式趋势 + 表面强迫 + ML closure → 8 AB3 → 9 隐式垂直扩散 + 冰点限制 → 10 η leapfrog + Asselin → 11 组装

## 7. 验证与稳定性现状（2026-09-28，constant 混合，除非注明）

| 实验 | 结果 |
|---|---|
| 5.1.1 静止均匀 | T/S **严格守恒**（判定已改为 exact） PASS |
| 5.1.2 层结 + 真实地形 | A（κ=0）u/v/η/T 严格为 0；B/C 单调 0=A<B<C（比值仅记录）PASS |
| 5.1.3 NullClosure | 逐位一致 PASS |
| 5.1.4 CFL | dt≤900 s 稳定，1200 s NaN |
| 5.1.5 垂直混合 | PASS |
| 5.2.1 / 5.2.2 / 5.2.3（30 d） | ΔSST −0.56 / −0.40 / −0.64 °C（Munk ν_h 后；旧版约 −2 °C 大部分是网格噪声） |
| 大西洋 2 年（PP81） | 最大流速 0.28–0.33 m/s，稳定；热、盐收支与外加通量吻合 |
| 热带 2 年（PP81） | 最大流速 0.43 m/s，稳定 |

论文第 5 章：5.1.1、5.1.2 判定方法已改；5.2 节数值与"降温与热通量一致"的解释需按新结果重写。

## 8. 已知问题与待办

- **表层热点**：固定热通量无 SST 反馈，停滞副热带格点 2 年后 > 40 °C。方案：Haney 恢复项
  Q = Q_ORAS5 + γ(SST_ORAS5 − SST)，γ≈40 W/m²/K（用户已理解，**推迟实现**）。
- 验证实验脚本是否改用 PP81：**待用户决定**。
- ORCA 风应力沿网格 i/j 方向，高纬北大西洋未旋转到东/北。
- 可变分辨率下 ν_h 应随空间变化（目前全域取最大值）。
- 中高纬混合层：以后考虑 KPP。
- OBC：新分支计划（模型完善后再做），半成品在本地 stash。
- **并行计算（下一大任务，方案待用户确认）**：目标多 GPU 集群，本地只有单 GPU。
  阶段 A：`parallel/sharding.py` 用 GSPMD 自动分片（mesh 轴 batch/x/y，`shard_grid/state/forcing`、`sharded_run`）；
  Thomas 求解改为嵌套 vmap（避免 reshape 触发 all-gather）；在 8 个模拟 CPU 设备
  （`XLA_FLAGS=--xla_force_host_platform_device_count=8`，conftest 设置）上测正确性与"无全场 all-gather"；
  benchmark 加区域分解模式；experiment.py 加 N_DEVICES_X/Y；`init_distributed()` + SLURM 文档（本地不可测）。
  阶段 B（shard_map + halo 交换）仅在集群实测效率不足时再做。

## 9. 会话结束前（每次）

1. 更新本文件：第 7 节（结果）、第 8 节（待办）、顶部"最后更新"和对应提交。
2. 与代码一起 commit + push（推送需用户同意）。Cloud session 若推到新分支/PR，用户需在 GitHub 合并进 `main` 后本地 `git pull`。
