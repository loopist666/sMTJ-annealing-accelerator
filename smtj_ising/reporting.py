"""Scientific figures, a Chinese report and a self-contained interactive demo."""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .measurements import causal_residual, acf

COLORS = dict(greedy="#737f8d", constant="#ab70b9", gaussian="#de923d", replay="#169a8b", shuffled="#568bd4", sa="#d35e70")
LABELS = dict(greedy="Greedy", constant="Constant TA", gaussian="Gaussian TA", replay="Measured replay", shuffled="Shuffled replay", sa="Software SA")


def figures(records, profiles, summary, result, image_result, out):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "figure.facecolor":"white", "savefig.dpi":180})
    fig, axes = plt.subplots(2, 2, figsize=(12,8), constrained_layout=True)
    for r in records:
        if r.time is None:
            axes[0,0].plot(r.voltage, r.current*1e3, alpha=.5, lw=.8)
    axes[0,0].set(title="24 bipolar sweeps (acquisition order retained)", xlabel="Programmed voltage (V)", ylabel="Current (mA)")
    for r in records:
        if r.time is not None and "IT 1-1V CYCLE" in r.source:
            color = "#d35e70" if np.median(r.voltage)>0 else "#169a8b"
            axes[0,1].plot(r.time-r.time[0], r.current*1e3, color=color, alpha=.65, lw=.7,
                linestyle="--" if "dead" in r.sheet else "-")
    axes[0,1].set(title="Repeated +1 V (red) / -1 V (green) traces", xlabel="Elapsed time (s)", ylabel="Current (mA)")
    bias = [r for r in summary if r["kind"]=="IT" and "0.01~2V IT" in r["source"]]
    axes[1,0].scatter([r["voltage_median_V"] for r in bias], [r["residual_std_uA"] for r in bias], color="#169a8b", s=45)
    axes[1,0].set(yscale="log",title="Variation after causal 2 s EMA removal", xlabel="Programmed voltage (V)", ylabel="Residual standard deviation (uA)")
    for p in profiles:
        rho=acf(p["test_z"],50)
        axes[1,1].plot(np.arange(len(rho))*p["dt_s"],rho,lw=1,alpha=.7)
    axes[1,1].axhline(0,color="#aaa",lw=.6)
    axes[1,1].set(title="Held-out -1 V residual autocorrelation",xlabel="Lag (s)",ylabel="Autocorrelation")
    fig.savefig(out/"measurement_overview.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12,4.5), constrained_layout=True)
    instances=[p["name"] for p in result["instances"]]
    modes=list(COLORS)
    for mi,mode in enumerate(modes):
        values=[next(r["hit_rate"] for r in result["summary"] if r["instance"]==name and r["mode"]==mode) for name in instances]
        axes[0].bar(np.arange(len(instances))+(mi-2.5)*.13,values,width=.12,label=LABELS[mode],color=COLORS[mode])
    axes[0].set(xticks=np.arange(len(instances)),xticklabels=["Cut / 101","Cut / 202","Cut / 303","Glass / 404"],
                ylabel="Exact optimum hit fraction",ylim=(0,1.05),title="Equal flip-attempt budgets; all traces included")
    axes[0].legend(fontsize=8,ncol=2)
    for mode in modes:
        gaps=np.mean([np.array(result["mean_paths"][p["name"]][mode])-p["exact_energy"] for p in result["instances"]],axis=0)
        axes[1].plot(np.arange(len(gaps)),gaps,label=LABELS[mode],color=COLORS[mode])
    axes[1].set(xlabel="Single-spin flip attempts",ylabel="Mean best-so-far energy gap",title="Fixed 4-instance suite; no hidden polishing")
    fig.savefig(out/"benchmark.png")
    plt.close(fig)

    fig,axes=plt.subplots(1,5,figsize=(13,3),constrained_layout=True)
    views=[("Synthetic clean",image_result["clean"]),("Noisy input",image_result["noisy"]),
           ("Measured replay",image_result["methods"]["replay"]["image"]),
           ("Software SA",image_result["methods"]["sa"]["image"]),("Exact graph cut",image_result["exact_image"])]
    for ax,(name,im) in zip(axes,views):
        ax.imshow(im,cmap="gray",vmin=-1,vmax=1,interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    fig.savefig(out/"image_denoising.png")
    plt.close(fig)


def dashboard(profiles, result, summary, out):
    # Full held-out normalized samples remain local unless explicitly published.
    # Script JSON is escaped against HTML termination even for custom sheet names.
    public_result={k:result[k] for k in ("steps","runs_per_trace","gain","summary","instances","paired_comparisons")}
    payload=dict(profiles=[{k:p[k] for k in ("key","sheet","dt_s","test_z","calibration_scale_A","split_index","filter_tau_s")} for p in profiles],
                 benchmark=public_result, measurements=summary)
    encoded=json.dumps(payload,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x,
                       allow_nan=False).replace("<","\\u003c").replace("&","\\u0026")
    template=Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
    (out/"demo.html").write_text(template.replace("__PAYLOAD__",encoded),encoding="utf-8")


def chinese_report(summary, profiles, result, image_result, out):
    it=[r for r in summary if r["kind"]=="IT"]
    fast=next((r for r in it if "fast" in r["sheet"].lower()),None)
    aggregate={mode:float(np.mean([r['hit_rate'] for r in result['summary'] if r['mode']==mode])) for mode in COLORS}
    lines=["# R4C5：单通道实测扰动驱动的软件 Ising demo", "",
           "## 结论与边界", "",
           "项目主线改为 **单通道电流回放 + 软件阈值接受退火**。自旋、J、h、节点调度、能量计算和阈值包络均在软件；器件记录只提供一维连续扰动。无需可编程多状态、p-bit 概率曲线、分区读出或物理耦合网络。",
           "本结果是有限记录的离线软件实验，不是硬件 Ising 机、独立真随机数认证、Boltzmann 采样或加速优势证明。", "",
           f"在固定四问题套件中，实测原序回放的平均精确解命中率为 **{aggregate['replay']:.1%}**，贪心为 {aggregate['greedy']:.1%}、打乱回放为 {aggregate['shuffled']:.1%}、软件 SA 为 **{aggregate['sa']:.1%}**。当前支持“实测扰动可以参与优化”，不支持“器件相关性或器件方案优于成熟软件算法”。", "",
           "## 数据核查", "",
           f"- 共 {len(summary)} 条测量记录：{len(it)} 条 I–t、{len(summary)-len(it)} 条 I–V，来自 5 个工作簿。锁文件和 ZIP 重复件未导入。",
           "- 电流 AI 按 A、AV 按 V、Time 按 s 读取；Settings 显示电压为 Programmed，因此 V×I 只是按设定电压估计的器件功耗。Settings 中的 Compliance 为 0.105 A。",
           "- 按原行顺序处理，依据 AV 识别偏压；`160R4C5-0.25VA1V` 名称带负号，但 AV 实为 +0.25 V。",
           "- 含 dead 的记录保留在数据诊断中，按名称预先排除求解器主基准；不根据解题成绩筛曲线。",
           f"- 主基准使用所有 {len(profiles)} 条合格的固定 −1 V 记录。它们来自同一个 R4C5，重复种子不是独立器件重复。",
           "- 没有磁场、温度、空载/等效电阻噪声对照，当前只能称为测量链路中的电流涨落，不能单独归因于磁热翻转。", ""]
    if fast:
        lines += [f"`fast -1v` 的 Δt 中位数为 {fast['dt_median_s']:.6f} s，采样率 {fast['sample_rate_Hz']:.3f} Hz；普通记录约 7.97 Hz。当前数据不能验证 ns/µs 翻转速率。", ""]
    lines += ["| 记录 | AV (V) | 平均 I (mA) | 原始 σ (µA) | 2 s 因果残差 σ (µA) | 残差 lag-1 | τeff (s) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in it:
        lines.append(f"| {r['sheet']} | {r['voltage_median_V']:.2f} | {r['current_mean_mA']:.4f} | {r['current_std_uA']:.3f} | {r['residual_std_uA']:.3f} | {r['residual_lag1']:.3f} | {r['tau_effective_s']:.3f} |")
    cycle=[r for r in it if 'IT 1-1V CYCLE' in r['source'] and not r['flagged_name']]
    positive=[r['residual_std_uA'] for r in cycle if r['voltage_median_V']>0]
    negative=[r['residual_std_uA'] for r in cycle if r['voltage_median_V']<0]
    if positive and negative:
        lines += ['', f"在循环记录中，+1 V 的 2 s 残差标准差范围为 {min(positive):.1f}–{max(positive):.1f} µA，−1 V 为 {min(negative):.1f}–{max(negative):.1f} µA。负偏置下更强的突发涨落是选择 −1 V 做扰动回放的依据。它提示极性与历史相关的传输机制，但测试顺序尚未随机化，不能据此宣称极性可独立控制随机性。"]
    lines += ["", "τeff 仅为给定采样与滤波条件下、ACF 首次非正前的积分摘要；不是本征驻留时间。`filter_sensitivity.csv` 同时报 0.5、2、10 s 基线常数下的结果。不同偏压记录的历史与时刻不同，不据此建立可编程温度码本。", "",
        "![测量概览](measurement_overview.png)", "", "## 新求解器", "",
        "```text", "E(s) = -Σ(i<j) J_ij s_i s_j - Σi h_i s_i", "ΔE_i = 2 s_i (Σj J_ij s_j + h_i)",
        "r_k = I_k - b_(k-1)", "b_k = b_(k-1) + [1-exp(-Δt_k/τ)] r_k",
        "z_k = (r_k - μ_train) / σ_train", "A_k = gain × 2 median_i sqrt(Σj J_ij² + h_i²) × (1-k/(M-1))²",
        "接受翻转当且仅当 ΔE_i ≤ A_k |z_k|", "```", "",
        f"求解器 EMA 时间常数为 {result['filter_tau_s']} s。前 40% 为标定段，丢弃前 5τ 的滤波启动段；后 60% 为回放段。μ、σ 只由标定段计算。滤波因果进行，不读取未来样本、不插值、不提取 K-state；每次翻转尝试消耗一个连续样本，不循环回放。不同软件 run 可以重用同一记录，必须按这一限制解释统计。", "",
        "A 的量纲是算法能量；gain 和退火包络是软件设置，未映射到实际电压或热力学温度。单路扰动按时间依次用于软件自旋，节点初态与选择由软件 PRNG 提供。此方法是优化启发式，相关阈值一般不满足 Boltzmann 详细平衡。", "",
        "## 对照与结果", "",
        f"固定 3 个 16 节点加权 Max-Cut 与 1 个带局域场的 16 自旋 spin-glass。每个实例完整枚举 65,536 个状态得到精确基准。每条记录 {result['runs_per_trace']} 个软件种子、每次 {result['steps']} 次翻转尝试、gain={result['gain']}。每组方法共用初态、节点顺序和原始片段，无提前结束或额外局部抛光。", "",
        "- Greedy：阈值为 0。",
        "- Constant TA：与该片段平均 |z| 匹配的恒定扰动因子，仍使用同一冷却包络。",
        "- Gaussian TA：与该片段 RMS 匹配的软件高斯扰动。",
        "- Measured replay / Shuffled：完全相同的样本集合，仅时间顺序不同。",
        "- Software SA：标准 Metropolis 接受，使用软件 Exp(1) 阈值；T=A。",
        "高斯和常量匹配使用完整测试片段，仅作离线消融，不能伪称为在线校准。当前没有为各方法分别调到最优，结果不构成算法排名。", "",
        "| 问题 | 方法 | 精确解命中 | 平均能量差 |", "|---|---|---:|---:|"]
    for row in result["summary"]:
        lines.append(f"| {row['instance']} | {LABELS[row['mode']]} | {row['hit_rate']:.1%} | {row['mean_gap']:.4f} |")
    lines += ["", "![基准](benchmark.png)", "", "配对差值以每个软件种子跨记录/问题平均后 bootstrap；区间只反映软件种子变动，不反映器件间可重复性：", "",
              "| 差值 | 平均 gap 差 | 软件种子 95% 区间 |", "|---|---:|---:|"]
    for c in result["paired_comparisons"]:
        lines.append(f"| {c['comparison']} | {c['estimate']:.4f} | [{c['software_seed_ci95_low']:.4f}, {c['software_seed_ci95_high']:.4f}] |")
    lines += ["", "负值有利于原序回放。即使优于某个对照，也只能说明这组记录、预算和超参数下的结果，不能推导普遍优势。", "",
        "## 第二应用：二值图像去噪", "",
        "用合成 12×12 图像建立铁磁近邻耦合与观测局域场；实测序列驱动同一个求解器。另用精确 s–t min-cut 求出该能量模型的最优解。该图不是器件显微图，也不是物理分区读出。", "",
        f"输入像素准确率 {image_result['input_accuracy']:.1%}；实测回放 {image_result['methods']['replay']['pixel_accuracy']:.1%}，能量 gap {image_result['methods']['replay']['gap']:.3f}；精确 MAP 的像素准确率 {image_result['exact_pixel_accuracy']:.1%}。降低模型能量不保证每个真实像素都正确。", "",
        "![图像去噪](image_denoising.png)", "",
        "## 器件结构与应用判断", "",
        "提供的元素叠图显示 Fe/Ni 富集区域以及 Ba/Sr 的空间分布；仅此不能确定化学相、隧穿势垒、电极连接、磁各向异性、10⁷ 可寻址自旋或耦合 J。本文只将其作为非均匀复合结构的线索，不将柱/颗粒直接当作 Ising 自旋。", "",
        "1. **优先验证：单通道模拟扰动协处理器。** 固定工作点采样，经 AC 耦合/数字基线去除后送入 CPU/FPGA 阈值接受搜索。新增价值应通过实测回放对比打乱、高斯、常阈值、SA 来检验；不依赖器件存储自旋。",
        "2. **事件驱动搜索。** 大电流突变可触发更大的软件邻域或重启。需把触发频率、计算量与 PRNG 事件流匹配，防止把更大搜索预算误当成器件优势。当前仅为待测方向。",
        "3. **历史相关动态 / 时序 reservoir。** 正负偏置下的漂移和路径依赖可启发单节点时序特征处理。需先做随机化脉冲序列、重复响应与 fading-memory 检验；不能由自发漂移直接声称可训练存储。",
        "4. **状态与健康监测。** 波动幅度、相关性和慢漂移可用于检测工作点变化；要有外部温度/磁场/损伤标签才能验证传感含义。", "",
        "## 下一轮最有辨别力的实验", "",
        "- 同一工作点至少多条独立重复，随机化电压测试顺序；记录预处理、恢复等待、采样积分时间、量程和实际端电压。",
        "- 空载、短路/等效电阻与仪器基线对照；磁场/温度扫描用于区分磁性、电阻通道和测量链路的贡献。",
        "- 提高采样带宽并测 anti-alias / ADC 链路；先定位 PSD 与相关时间，再决定合理采样率。",
        "- 固定偏置、仅软件改增益完成第一版实时闭环。比较相同翻转尝试数和相同墙钟下的解质量，计入 ADC、软件、数据传输和采样等待。",
        "- 分区读出不是此路线的前置条件；若将来主张物理自旋网络，仍必须补充寻址、J 可控性、读出和物理能量映射实验。", "",
        "## 时间与能量不能混淆", "",
        f"当前 {result['steps']} 个不同样本的采集跨度约 {min((result['steps']-1)*p['dt_s'] for p in profiles):.1f}–{max((result['steps']-1)*p['dt_s'] for p in profiles):.1f} s。Python 对已存数组的运行时间不是器件求解时间。`runs.csv` 分别记录两者，并记录设定 V×I 的片段积分；该积分未包含外围，也不是测得的闭环 energy-to-solution。另需计入前 40% 的标定采集与启动等待；打乱方法还需要离线缓冲。", "",
        "## 参考方法", "",
        "- Dueck & Scheuer, Threshold accepting, J. Computational Physics 90, 161–175 (1990): https://doi.org/10.1016/0021-9991(90)90201-B",
        "- Böhm et al., Noise-injected analog Ising machines, Nature Communications 13, 5847 (2022): https://www.nature.com/articles/s41467-022-33441-3 。仅作噪声参与计算的概念参照；其中的硬件条件和采样结论不转移到 R4C5。",
        "- Boykov & Kolmogorov, min-cut/max-flow energy minimization (2004), author publications: https://pub.ista.ac.at/~vnk/publications.html", ""]
    sensitivity_path=out/"solver_sensitivity.json"
    if sensitivity_path.exists():
        sensitivity=json.loads(sensitivity_path.read_text(encoding='utf-8'))
        lines += ['## 预处理与软件增益敏感性', '',
            f"附加固定网格：EMA τ∈{{0.5, 2, 10}} s，gain∈{{0.5, 1.5, 3}}，每条记录 {sensitivity['runs_per_trace']} 个种子，均为 2048 次尝试。下面报告跨固定四问题的平均 gap；没有用此网格挑选主结果。",
            '', '| τ (s) | gain | 原序 gap | 打乱 gap | 软件 SA gap |', '|---:|---:|---:|---:|---:|']
        rows=sensitivity['rows']
        for tau in (.5,2.,10.):
            for gain in (.5,1.5,3.):
                vals={r['mode']:r['mean_gap'] for r in rows if r['filter_tau_s']==tau and r['gain']==gain}
                lines.append(f"| {tau} | {gain} | {vals['replay']:.3f} | {vals['shuffled']:.3f} | {vals['sa']:.3f} |")
        lines += ['', '该网格说明预处理会改变算法表现；这里的 τ 是软件滤波旋钮，不是器件可编程的物理相关时间。完整六方法结果见 `solver_sensitivity.csv`。', '']
    (out/"report_zh.md").write_text("\n".join(lines),encoding="utf-8")
