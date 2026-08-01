# Binary P-bit + One-hot Ising vs Direct-Permutation Ordered 1D MH

## 1. 更新核

TSP 状态直接保存为固定 city 0 的排列：

\[
\pi=[0,\pi_1,\ldots,\pi_{N-1}].
\]

selected-curve 的 K=6 器件 profile 提供退出率 \(r_k\) 和转移矩阵 \(P_{kl}\)：

\[
q_{kl}=r_kP_{kl},\qquad k\ne l.
\]

离散 proposal clock 使用

\[
p_{\mathrm{event}}(k)=\Delta t\sum_{l\ne k}q_{kl},
\]

其中 \(\Delta t\) 选择为使最大 event probability 等于 0.5。发生事件后按
\(q_{kl}/\sum_lq_{kl}\) 选择目标状态 \(l\)。

有序距离 \(|l-k|\) 映射到固定起点之外的 cyclic transposition distance。选择位置
\(i\) 后，对 \(i\) 和 \(i+d\) 做交换。由于映射只依赖无向状态对 \(\{k,l\}\)，反向
\(l\rightarrow k\) 产生相同的交换分布。

联合状态 \((\pi,k)\) 的接受概率为：

\[
A[(\pi,k)\rightarrow(\pi',l)]
=\min\left(1,
e^{-\beta[L(\pi')-L(\pi)]}\frac{q_{lk}}{q_{kl}}
\right).
\]

接受时同时更新路线和器件状态；拒绝时两者均保持不变。

## 2. 合法性、可逆性和可达性

- 初始化状态是排列，transposition 后仍是排列，因此 raw feasible rate 恒为 100%。
- city 0 不参与交换，去除了旋转简并。
- 当前实测 \(q\) 的 30 条有向非对角边全部存在正的反向边。
- \(k\leftrightarrow k+1\) 的相邻器件边均为双向，因此 distance-1 route swap 有正概率。
- 相邻 transposition 可以生成所有固定起点排列，所以 route state space 可达。
- 脚本对不同 \(k,l\)、城市规模、位置随机数及一组 \(\beta,\Delta L\) 组合验证了
  move reversibility 和理论 detailed-balance flux identity。

## 3. 对比设置

| setting | value |
|---|---:|
| dataset | selected_curve |
| selected K | 6 |
| runs/method/size | 24 |
| update attempts/run | 121,500 |
| beta schedule | 0.08 → 4.0 |
| event probability cap | 0.5 |
| TSP sizes | 8, 12 |

这里采用端到端方案比较：二值 p-bit 基线使用 one-hot Ising/QUBO，P-dit 路径使用直接
排列编码与 ordered 1D empirical-transition MH。保存实验同时保持相同的 algorithmic
element-update budget，但性能差异允许包含编码、可行域和更新核的共同收益。one-hot 基线一次 profile clock 更新
三个 Gray p-bit channel，因此每条 run 为 40,500 profile clocks、121,500 spin attempts；
ordered 1D 路线每条 run 为 121,500 empirical proposal-clock ticks。

## 4. 结果

| cities | method | variables | raw feasible | median raw/ref | raw reference hit | median 2-opt/ref |
|---:|---|---:|---:|---:|---:|---:|
| 8 | binary p-bit + one-hot Ising | 64 | 100.0% | 1.263 | 0.0% | 1.025 |
| 8 | ordered 1D empirical-MH | 8 | 100.0% | **1.000** | **100.0%** | **1.000** |
| 12 | binary p-bit + one-hot Ising | 144 | 0.0% | 1.831 | 0.0% | 1.172 |
| 12 | ordered 1D empirical-MH | 12 | **100.0%** | **1.000** | **54.2%** | **1.000** |

12-city reference 由 nearest-neighbor all-start + 2-opt 得到，不保证是严格全局最优；
ratio 1.000 表示达到该 reference。

ordered 1D 的 proposal event rate 为 8 城市 0.348、12 城市 0.250；给定 event 的
MH acceptance 分别为 0.072 和 0.095。被接受的平均 1D move distance 约为 1.10，符合
实测 proposal 的局部性。

## 5. Max-Cut 与 Portfolio 发布视图

`results/benchmarks/tsp_binary_pbit_vs_ordered_1d_mh/` 同时保存 Max-Cut 和 Portfolio 的：

- `*_comparison_runs.csv`：每种方法 24 条逐 run 结果；
- `*_comparison_summary.csv`：成功率、首次命中、gap、接受率和 TTS；
- `*_paired_comparison.csv`：配对差值及 bootstrap 区间；
- `*_success_curves.csv` / `*_success_curves.png`：成功率随 update budget 的变化；
- `*_method_comparison.png`：成功率、首次命中和平均 best gap；
- `maxcut_portfolio_comparison_summary.json`：两类问题的严格 JSON 汇总。

这两类问题不使用 TSP 的直接排列编码。它们保持原生 QUBO/Ising 表示，只比较
Gray binary p-bit 与 ordered 1D P-dit 的器件状态解释。导出命令为：

```bash
python tools/results/export_maxcut_portfolio_views.py
```

当前结果：Max-Cut 两种方法均为 24/24 reference hit；Portfolio 的 ordered 1D 为
18/24（75.0%），Gray binary 为 16/24（66.7%）。Portfolio 中 ordered 1D 的最终覆盖率
更高，但成功 run 的 median first-hit update 更晚（831 vs 405），因此不能概括为全预算更快。

## 6. 结论与限制

替换后的 kernel 已同时解决表示和器件一致性问题：路线不再依赖 one-hot penalty，
器件状态也不再只是 hazard/random source，而是作为实际 \(q_{kl}\) proposal 使用。

需要注意：这里比较的是“binary p-bit + one-hot Ising 整体路线”与“direct permutation ordered 1D
路线”，因此性能差异同时包含表示法和 proposal kernel 的收益，不能只归因于状态几何。
固定编码的几何消融保留在 `src/diagnostics/p1_fixed_encoding_ablation.py`，仅用于诊断，
不作为本项目的 TSP 主性能对比。
