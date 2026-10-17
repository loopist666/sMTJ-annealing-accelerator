# sMTJ measured-noise Ising demo

基于 **R4C5 单通道实测电流回放** 的软件 Ising solver。新的主线不依赖可编程多状态、p-bit / P-dit、磁柱逐个寻址或分区读出。

**当前定位：器件测量信号参与的软件优化实验。** 自旋、耦合 `J`、局域场 `h`、节点调度和退火包络全部在软件中。器件时序作为连续扰动，控制阈值接受搜索可以跨过的能量障碍。它不代表已经实现物理 Ising 机、独立真随机源、Boltzmann 采样或硬件加速。

## 运行 demo

需要 Python 3.10+。从仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m smtj_ising --data-dir "/path/to/R4C5"
```

Windows 激活命令为 `.venv\Scripts\Activate.ps1`。

完成后打开 **`results/r4c5_local/demo.html`**。无需服务端或网络，可以改变问题、实测记录、算法、增益和随机种子，查看自旋分组、能量下降、实测残差与接受阈值。网页的浏览器 PRNG 与 Python 不同；批量报告以 Python 为准。

| 文件 | 内容 |
|---|---|
| `demo.html` | 包含留出段实测信号的离线交互 demo |
| `report_zh.md` | 数据结论、算法、对照、应用和下一轮实验建议 |
| `measurement_overview.png` | I–V、I–t、偏压依赖与残差 ACF |
| `benchmark.png` | 固定四个实例的精确解命中率与能量曲线 |
| `image_denoising.png` | 12×12 二值图像去噪，含精确图割参照 |
| `runs.csv` / `benchmark.json` | 每次软件运行结果、精确值与统计 |
| `measurement_summary.csv` / `filter_sensitivity.csv` | 全部记录统计及预处理敏感性 |
| `input_manifest.json` / `preprocessing.json` | 工作簿 SHA256、采集设置、标定范围 |

R4C5 原始工作簿和包含逐点测量信号的生成结果默认留在本地，不随 Git 提交。仓库中的 [参考结果](results/r4c5_reference/report_zh.md) 是汇总快照，复现时需提供 R4C5 目录。

快速检查、完整复现与预处理敏感性：

```bash
python -m unittest discover -s tests -v
python -m smtj_ising --data-dir "/path/to/R4C5" --analyze-only
python -m smtj_ising --data-dir "/path/to/R4C5" --runs 4 --steps 1024 --out results/quick_local
python -m smtj_ising --data-dir "/path/to/R4C5" --runs 32 --steps 2048 --gain 1.5 --seed 700
python -m smtj_ising --data-dir "/path/to/R4C5" --filter-tau 10 --out results/tau10_local
python -m smtj_ising.sensitivity --data-dir "/path/to/R4C5" --runs 8
```

超过留出序列长度会明确报错；程序不会静默循环测量序列或用插值制造新样本。输出目录不能放入原始测量目录。

## 算法与映射

采用统一能量约定：`E(s) = -sᵀJs/2 - hᵀs`，其中 `sᵢ∈{-1,+1}`、`J=Jᵀ`、`Jᵢᵢ=0`。

1. 保留实测时间顺序，用因果 EMA 去除慢基线。
2. 前 40% 记录用于标定均值、标准差；后 60% 留给回放。滤波启动段 `5τ` 不用于标定。
3. 每步软件选择一个节点，计算 `ΔEᵢ = 2sᵢ[(Js)ᵢ+hᵢ]`。
4. 若 `ΔEᵢ ≤ Aₖ |zₖ|`，接受翻转；`Aₖ` 按二次包络下降到零。
5. 保留历史最好解，无隐藏抛光、额外重启或提前终止。

`Aₖ = gain × 2 medianᵢ sqrt(ΣⱼJᵢⱼ²+hᵢ²) × (1−k/(M−1))²`。
该包络按问题耦合尺度归一化，不使用精确答案或测试成功率。它是**软件阈值**，没有把电压直接当作温度；有色回放也不保证详细平衡。

加权 Max-Cut 使用 `J=-W, h=0`，因此 `E = Σ(i<j)Wᵢⱼ − 2 Cut`。QUBO 使用完整矩阵约定 `xᵀQx`，`model.qubo()` 同时返回能量常数项，避免混淆上三角与对称矩阵的双计数。

## 对照设计

- 主基准预先选择全部固定 −1 V 记录，排除名称含 `dead` 的记录；未按求解成绩挑曲线。
- 原序回放、打乱回放、匹配 RMS 高斯、匹配平均幅度常阈值、无扰动贪心、标准软件模拟退火共六种方法。
- 每组共用初态、节点顺序、翻转尝试预算。原序与打乱使用完全相同的样本集合。
- 高斯与常量的匹配利用完整测试片段，只用于离线消融，不是在线标定。
- 三个 16 节点加权 Max-Cut、一个带局域场的 16 自旋 spin-glass，全枚举给出精确答案。
- 第二应用为合成二值图像去噪：144 个软件自旋，精确 s–t graph cut 验证模型能量。
- 报告逐记录结果、软件种子配对 bootstrap 区间和 0.5/2/10 s 滤波敏感性。不把同一记录的重放次数当作独立硬件样本。

当前是固定小规模验证套件，各算法没有分别调至最优，结果不用于证明普遍优势。

## R4C5 的重要事实

- 共 25 条 I–t 和 24 条 I–V。所有 Excel 按只读方式加载，忽略锁文件和 ZIP 重复件。
- `fast -1v` 的实际采样率约 9.27 Hz，普通采样约 7.97 Hz；文件名 “fast” 不代表已测得高速翻转。
- AV 列为设定电压。名称为 `160R4C5-0.25VA1V` 的工作表实际 AV 为 **+0.25 V**。
- 固定电压轨迹有明显时间相关与漂移，不能从不同直方图推出可保持、可寻址、独立可调的多状态。
- Settings 记录 105 mA compliance；没有独立仪器底噪、磁场、温度对照，不能把全部测量涨落直接归因为磁热随机性。
- 单路顺序采样足以驱动本 demo；软件自旋数量不等于器件内部可读出的磁单元数量。

## 后续研究方向

优先做固定偏置的实时**模拟扰动协处理器**：器件 → ADC → 因果基线去除 → CPU/FPGA 搜索。先通过原序/打乱/高斯/常阈值/SA 对照确定器件信号的贡献，再计入采集等待与外围能耗。

可探索事件触发的大邻域搜索，以及历史相关动态用于单节点 reservoir；两者仍需独立实验，不能将自发漂移直接解释为可编程记忆。详细判断与实验清单见 [分析报告](results/r4c5_reference/report_zh.md)。

## 代码组织与迁移

```text
smtj_ising/                 # 当前主线：数据读取、Ising、求解、基准、图表与 demo
tests/                     # 能量/映射、精确验证、因果性、读取与预算测试
results/r4c5_reference/     # 本次汇总结果，无逐点原始电流
results/*_local/            # 本地生成结果，Git 忽略
src/                       # 历史多状态 / p-bit / P-dit 实验，保持旧路径以便复现
docs/legacy_multistate_README.md
```

历史 `src/mtj_*`、P0/P1、旧结果和 PDF 保留，但它们的状态可调与器件映射假设不构成 R4C5 已验证能力，也不再是默认演示路线。旧说明见 [历史入口](docs/legacy_multistate_README.md)。

## 方法参考

- Dueck & Scheuer, threshold accepting, *J. Computational Physics* 90, 161–175 (1990), [DOI](https://doi.org/10.1016/0021-9991(90)90201-B)。这里实现的是由记录信号调制阈值的变体。
- Böhm et al., noise-injected analog Ising machines, *Nature Communications* 13, 5847 (2022), [原文](https://www.nature.com/articles/s41467-022-33441-3)。其硬件/采样结论不能转移到本软件回放。
- Boykov & Kolmogorov, graph-cut energy minimization (2004), [作者论文页](https://pub.ista.ac.at/~vnk/publications.html)。

仓库尚未声明开源许可证。
