# Diagnostic Only: Fixed-Encoding Ordered 1D Embedding Ablation

## 1. 目的

P1 器件分析表明，selected-curve 的 K=6 实测转移图更符合 ordered 1D
几何，而不是 Gray-coded binary 或 isotropic six-state 几何。本实验把该结论放入
`mtj_ising_application_demos.py` 的三个实际用例中，比较：

- `gray_binary_pbit`：六个实测状态映射到三位 Gray path；
- `ordered_1d_pdit`：六个实测状态映射到 `[-1, 1]` 上的有序标量。

仓库和 P1 分析使用的术语是 **ordered 1D P-dit**。如果把该有序离散变量称为
“1D p-int”，本文中的实现对应同一 K=6 ordered scalar state geometry。

## 2. 公平对比设计

两种方法固定使用相同的：

1. selected-curve K=6 实测状态、dwell/exit rate 和随机数流；
2. application QUBO/Ising 编码与 penalty；
3. 每个配对 run 的初始 Ising state；
4. 单 spin 更新次数；
5. coupling、data 和 hazard gain。

为了避免不同 embedding 的直流偏置和尺度影响结果，各 embedding channel 按实测
state occupancy 去均值并归一化到 unit RMS。这保留状态拓扑和相对距离，但让 data
drive 的数值尺度可比。

默认配置为每种方法、每个问题 24 个配对 run：

| problem | variables | updates/run | reference |
|---|---:|---:|---|
| 8-city TSP | 64 | 81,000 | exact route |
| 10-node Max-Cut | 10 | 54,000 | exact cut |
| 8-asset Portfolio | 8 | 45,000 | exact feasible portfolio |

这里保持原 application 的 one-hot TSP 编码，以隔离随机器件状态解释的影响；不与
`src/benchmarks/tsp_binary_pbit_vs_ordered_1d_mh.py` 中改变状态空间的 direct permutation
P-dit 混为一谈。本实验不作为当前 TSP 主性能对比。

## 3. 结果

| problem | metric | Gray binary p-bit | Ordered 1D P-dit |
|---|---|---:|---:|
| TSP | exact/feasible hit rate | 0.0% | 0.0% |
| TSP | mean best QUBO gap | 1206.29 | **1150.48** |
| TSP | median raw route/reference | **1.539** | 1.556 |
| TSP | median 2-opt route/reference | 1.246 | **1.078** |
| Max-Cut | exact hit rate | 100.0% | 100.0% |
| Max-Cut | median first-hit update | 22.5 | 22.5 |
| Portfolio | exact hit rate | 66.7% | **75.0%** |
| Portfolio | median first-hit update among successes | **405** | 831 |

配对结果：

- TSP 中 ordered 1D 的 best QUBO gap 在 70.8% 的配对 run 中更小，平均差
  `ordered - binary = -55.81`；bootstrap 95% CI 为 `[-188.85, 77.17]`，仍跨过 0。
- Max-Cut 在该小规模实例上已饱和，两种方法 24/24 全部命中，最终质量没有差异；
  早期 checkpoint 的 binary p-bit 略快。
- Portfolio 的 ordered 1D 最终命中率高 8.3 个百分点，但配对 bootstrap 95% CI
  为 `[-16.7, 33.3]` 个百分点；样本量不足以把该方向判断为显著优势。
- Portfolio 的 fixed-budget TTS99 为 ordered 180,000 updates、binary 225,000
  updates；若允许在最佳早停 checkpoint 重启，则 binary 为 4,500 updates、ordered
  为 8,000 updates。ordered 的最终覆盖率更高，但 binary 更早命中部分容易的初态。

## 4. 结论

ordered 1D 是更合理的器件状态解释，但本组 application 数据不支持“所有问题都更快”
的强结论：

- 它在 TSP 的平均 QUBO gap、经统一 2-opt 后的路线种子质量，以及 Portfolio 的最终
  exact hit rate 上表现更好；
- Max-Cut 已进入 ceiling regime，无法区分两种方法；
- binary p-bit 在 Max-Cut 的早期 checkpoint 和 Portfolio 的早停 TTS99 上更快；
- TSP 的根本瓶颈仍是 one-hot binary encoding。两种随机原语都没有产生 raw feasible
  optimum；要解决这个瓶颈，应使用 direct permutation P-dit，而不是仅替换 state
  embedding。

因此，更稳妥的表述是：**ordered 1D P-dit 改善了部分结构化问题的最终解质量或覆盖率，
并与实测器件转移图更一致，但其应用性能优势依赖问题结构和运行预算。**

## 5. 输出

- `application_comparison_runs.csv`：每个 paired run 的结果；
- `application_comparison_summary.csv`：各方法/问题汇总；
- `application_paired_comparison.csv`：配对差值和 bootstrap 95% CI；
- `application_success_curves.csv`：随 update budget 的成功率；
- `application_comparison_summary.json`：配置、器件 profile、几何指标和全部汇总；
- `application_method_comparison.png`、`application_success_curves.png`：对比图。

软件 wall time 只表示当前 Python 实现的运行开销，不能当作 MTJ 硬件 latency 或能效。
