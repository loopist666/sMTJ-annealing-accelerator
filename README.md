# sMTJ Annealing Accelerator

这个仓库整理了基于 MTJ/sMTJ 随机物理信号的 simulated annealing、p-bit、K-state Ising solver 和组合优化应用 demo。当前代码的重点不是搭建通用框架，而是把实验曲线、硬件随机性建模、Ising/QUBO 优化和论文图表结果放在一个可以复现实验的轻量项目里，方便迁移到另一台设备继续开发。

## 当前工作摘要

- 已读取两组 MTJ Excel 数据：`simulated annealing.xlsx` 和 `simulated annealing_bad.xlsx`，字段主要为 `Time`、`AI`、`AV`。
- 已完成传统 simulated annealing 对齐 demo，用于比较 good/bad 曲线并输出残差与拟合指标。
- 已从 MTJ 时间序列构造硬件退火 profile、多维 p-bit profile、physical K-state profile，并将这些 profile 用于 Ising 随机优化。
- 已完成 benchmark、multistate sweep、K-state selected-curve 分析和对应图表。
- 最新增量是 `mtj_ising_application_demos.py`：把 TSP、Weighted Max-Cut、Portfolio Selection 编码为 QUBO，再转换为 Ising energy，用 MTJ physical K-state stochastic solver 求解，并输出 scaling CSV、PNG 和论文用总结。

## 文件结构

```text
.
├── simulated annealing.xlsx
├── simulated annealing_bad.xlsx
├── random number.xlsx
├── mtj_sa_demo.py
├── mtj_hardware_sa_demo.py
├── mtj_hardware_sa_multistate_demo.py
├── mtj_hardware_sa_benchmark.py
├── mtj_multistate_pbit_demo.py
├── mtj_multistate_pbit_sweep.py
├── mtj_physical_sa_demo.py
├── mtj_physical_kstate_demo.py
├── mtj_physical_kstate_selected_curve.py
├── mtj_ising_application_demos.py
├── plot_selected_curve_multistate_performance.py
├── plot_selected_curve_multistate_performance_k6.py
├── mtj_*_output/
├── selected_curve_*.png
├── multistate_pbit_best_kstate_performance.png
├── mtj_physical_sa_vs_multistate_pbit_performance.png
├── mtj_ising_application_results.md
└── mtj_ising_application_results_overleaf.tex
```

主要脚本与输出关系：

| 脚本 | 用途 | 主要输出 |
|---|---|---|
| `mtj_sa_demo.py` | 用 SA 对齐 good/bad MTJ 曲线 | `mtj_sa_output/` |
| `mtj_hardware_sa_demo.py` | 从 MTJ 信号构造硬件退火 profile 并跑 Ising demo | `mtj_hardware_sa_output/` |
| `mtj_hardware_sa_multistate_demo.py` | 多状态硬件 SA profile | `mtj_hardware_sa_multistate_output/` |
| `mtj_hardware_sa_benchmark.py` | 多个 Ising benchmark 实例统计 | `mtj_benchmark_output/` |
| `mtj_multistate_pbit_demo.py` | 多维 p-bit 随机更新 demo | `mtj_multistate_pbit_output/` |
| `mtj_multistate_pbit_sweep.py` | multistate p-bit 参数/实例 sweep | `mtj_multistate_sweep_output/` |
| `mtj_physical_sa_demo.py` | 物理随机退火 profile 分析 | `mtj_physical_sa_output/` |
| `mtj_physical_kstate_demo.py` | 从 MTJ 曲线提取 K-state 隐状态并用于 Ising | `mtj_physical_kstate_output/` |
| `mtj_physical_kstate_selected_curve.py` | 对选定曲线做 K-state 技术路线图和稳定性分析 | `mtj_physical_kstate_selected_curve_output/` |
| `mtj_ising_application_demos.py` | TSP、Max-Cut、Portfolio 的 QUBO/Ising 应用 demo | `mtj_ising_application_demos_output/` |

## 环境迁移

在新设备上克隆仓库后，建议使用 Python 3.10+。当前脚本只依赖常见科学计算包。

```powershell
git clone https://github.com/loopist666/sMTJ-annealing-accelerator.git
cd sMTJ-annealing-accelerator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS 激活虚拟环境时使用：

```bash
source .venv/bin/activate
```

## 推荐复现顺序

从基础数据对齐到应用 demo，可以按下面顺序运行：

```powershell
python mtj_sa_demo.py
python mtj_hardware_sa_demo.py
python mtj_hardware_sa_benchmark.py
python mtj_physical_kstate_demo.py
python mtj_physical_kstate_selected_curve.py
python mtj_multistate_pbit_demo.py
python mtj_multistate_pbit_sweep.py
python mtj_ising_application_demos.py
python plot_selected_curve_multistate_performance.py
python plot_selected_curve_multistate_performance_k6.py
```

每个脚本通常会在对应的 `*_output/` 目录写入：

- `*.json`：汇总配置、关键指标、参考解和运行统计。
- `*.csv`：逐实例或逐 run 的原始统计，适合后续制图和论文表格。
- `*.png`：论文或汇报用图。

## 最新 Ising 应用 demo

`mtj_ising_application_demos.py` 的核心流程：

1. 将实际问题写成 QUBO。
2. 使用 `x = (s + 1) / 2` 转换为 Ising energy。
3. 调用 `mtj_physical_kstate_demo.py` 中的 K-state profile，把 MTJ 隐状态、dwell/exit rate 和随机序列用于 spin proposal/acceptance。
4. 对 TSP、Max-Cut、Portfolio 分别解码并和 exact 或 heuristic reference 比较。

当前结果位置：

- 总结 JSON：`mtj_ising_application_demos_output/application_demo_summary.json`
- 主报告：`mtj_ising_application_results.md`
- Overleaf 版本：`mtj_ising_application_results_overleaf.tex`
- TSP scaling：`mtj_ising_application_demos_output/tsp_scaling_summary.csv`
- Max-Cut scaling：`mtj_ising_application_demos_output/maxcut_scaling_summary.csv`
- Portfolio scaling：`mtj_ising_application_demos_output/portfolio_scaling_summary.csv`

当前关键观察：

- K-state profile 使用 good dataset，当前应用 demo 中 selected K 为 4，p-bit dimensions 为 2，state entropy 约为 1.665 bits。
- TSP 的 one-hot permutation 约束随城市数按平方增长，raw Ising state 在 12 城市以后很难直接满足可行性；但作为 route seed，经 2-opt repair 后仍能得到接近参考路线的结果。
- Weighted Max-Cut 更贴近 Ising 原生二值结构，10/16/24 节点都达到 reference cut，32 节点 best cut 也达到 reference，median cut ratio 约为 0.987。
- Portfolio Selection 通过 cardinality penalty 控制选股数量，小规模可达到 exact reference，规模增大后仍保持可行 cardinality，但 objective gap 增大。

## 数据与版本管理说明

- 已提交的 `*_output/` 目录是当前实验快照，用于迁移后快速检查结果是否一致。
- `*.zip`、`__pycache__/`、`.pycache_check/`、`.venv/`、`pptx_extracted_images/` 属于本地缓存或可再生成文件，默认不进入 Git。
- 如果需要重新打包 Overleaf 材料，可用当前 `mtj_ising_application_results_overleaf.tex` 和 `mtj_ising_application_demos_output/` 中的图重新生成 zip。
- 若新设备没有图形后端，脚本仍可生成 CSV/JSON；安装 `matplotlib` 后会额外输出 PNG。

## GitHub 同步

默认远端：

```text
origin https://github.com/loopist666/sMTJ-annealing-accelerator.git
```

常用同步命令：

```powershell
git status --short --branch
git add README.md requirements.txt .gitignore mtj_ising_application_demos.py mtj_ising_application_demos_output mtj_ising_application_results.md mtj_ising_application_results_overleaf.tex
git commit -m "Document MTJ Ising application demos"
git push origin main
```
