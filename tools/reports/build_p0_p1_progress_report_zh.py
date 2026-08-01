from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/pdf/p0_p1_ordered_1d_empirical_mh_progress_zh.pdf"

P0 = json.loads(
    (ROOT / "results/mtj_p0_direct_pdit_output/p0_summary.json").read_text()
)
FOUR = json.loads(
    (ROOT / "results/mtj_p0_four_solver_output/four_solver_summary.json").read_text()
)
P1 = json.loads(
    (ROOT / "results/mtj_p1_empirical_transition_mh_output/p1_summary.json").read_text()
)
TSP = json.loads(
    (
        ROOT
        / "results/benchmarks/tsp_binary_pbit_vs_ordered_1d_mh/tsp_comparison_summary.json"
    ).read_text()
)

BODY_FONT = "ArialUnicode"
registerFont(
    TTFont(BODY_FONT, "/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
)

PAGE_W, PAGE_H = A4
MARGIN_X = 17 * mm
MARGIN_TOP = 17 * mm
MARGIN_BOTTOM = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2B6F9F")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
ORANGE = colors.HexColor("#D9772D")
LIGHT_ORANGE = colors.HexColor("#FCEFE4")
GREEN = colors.HexColor("#2F9E62")
LIGHT_GREEN = colors.HexColor("#E9F6EF")
GRAY = colors.HexColor("#667085")
LIGHT_GRAY = colors.HexColor("#F3F5F7")
DARK = colors.HexColor("#1C2430")
RED = colors.HexColor("#B53A3A")


def pstyle(name: str, **kwargs) -> ParagraphStyle:
    base = {
        "fontName": BODY_FONT,
        "fontSize": 9.2,
        "leading": 14,
        "textColor": DARK,
        "spaceAfter": 5,
        "wordWrap": "CJK",
    }
    base.update(kwargs)
    return ParagraphStyle(name, **base)


styles = getSampleStyleSheet()
S = {
    "cover_title": pstyle(
        "cover_title",
        fontSize=24,
        leading=34,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=10,
    ),
    "cover_sub": pstyle(
        "cover_sub", fontSize=13, leading=20, textColor=BLUE, spaceAfter=12
    ),
    "h1": pstyle(
        "h1", fontSize=18, leading=25, textColor=NAVY, spaceAfter=9
    ),
    "h2": pstyle(
        "h2", fontSize=12.5, leading=18, textColor=BLUE, spaceBefore=5, spaceAfter=6
    ),
    "body": pstyle("body"),
    "small": pstyle("small", fontSize=7.7, leading=11, textColor=GRAY),
    "caption": pstyle(
        "caption",
        fontSize=7.6,
        leading=10.5,
        textColor=GRAY,
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=6,
    ),
    "callout": pstyle(
        "callout",
        fontSize=10.2,
        leading=16,
        textColor=NAVY,
        leftIndent=7,
        rightIndent=7,
        spaceBefore=4,
        spaceAfter=4,
    ),
    "table": pstyle("table", fontSize=7.5, leading=10.5, spaceAfter=0),
    "table_head": pstyle(
        "table_head",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.white,
        alignment=TA_CENTER,
        spaceAfter=0,
    ),
    "metric": pstyle(
        "metric",
        fontSize=16,
        leading=20,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=2,
    ),
    "metric_label": pstyle(
        "metric_label",
        fontSize=7.6,
        leading=10,
        textColor=GRAY,
        alignment=TA_CENTER,
        spaceAfter=0,
    ),
    "timeline": pstyle(
        "timeline", fontSize=8.2, leading=12.5, textColor=DARK, spaceAfter=0
    ),
}


def para(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullet(text: str) -> Paragraph:
    return Paragraph("• " + text, S["body"])


def callout(text: str, background=LIGHT_BLUE) -> Table:
    t = Table([[para(text, "callout")]], colWidths=[CONTENT_W])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#C9D8E5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return t


def metric_cards(items: list[tuple[str, str]], background=LIGHT_BLUE) -> Table:
    cells = []
    inner_width = CONTENT_W / len(items) - 8
    for value, label in items:
        cell = Table(
            [[para(value, "metric")], [para(label, "metric_label")]],
            colWidths=[inner_width],
        )
        cell.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ]
            )
        )
        cells.append(cell)
    table = Table(
        [cells],
        colWidths=[CONTENT_W / len(items)] * len(items),
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CFD8E3")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def styled_table(
    rows: list[list[str]],
    widths: list[float],
    header: bool = True,
    font_size: float = 7.5,
) -> Table:
    prepared = []
    for ridx, row in enumerate(rows):
        style = "table_head" if header and ridx == 0 else "table"
        prepared.append([Paragraph(str(cell), S[style]) for cell in row])
    t = Table(prepared, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D0D9")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    for idx in range(1 if header else 0, len(rows)):
        if idx % 2 == 0:
            commands.append(("BACKGROUND", (0, idx), (-1, idx), LIGHT_GRAY))
    t.setStyle(TableStyle(commands))
    return t


def figure(path: Path, max_w: float, max_h: float) -> Image:
    reader = ImageReader(str(path))
    iw, ih = reader.getSize()
    scale = min(max_w / iw, max_h / ih)
    return Image(str(path), width=iw * scale, height=ih * scale)


def fig(path: str, caption: str, max_h: float) -> list:
    return [
        figure(ROOT / path, CONTENT_W, max_h),
        para(caption, "caption"),
    ]


def page_decor(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#D7DEE7"))
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN_X, PAGE_H - 12 * mm, PAGE_W - MARGIN_X, PAGE_H - 12 * mm)
        canvas.setFont(BODY_FONT, 7.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(MARGIN_X, PAGE_H - 9 * mm, "sMTJ P-dit 阶段进展")
        canvas.drawRightString(
            PAGE_W - MARGIN_X,
            PAGE_H - 9 * mm,
            "Ordered 1D empirical-transition MH",
        )
    canvas.setStrokeColor(colors.HexColor("#D7DEE7"))
    canvas.line(MARGIN_X, 11 * mm, PAGE_W - MARGIN_X, 11 * mm)
    canvas.setFont(BODY_FONT, 7.2)
    canvas.setFillColor(GRAY)
    canvas.drawString(MARGIN_X, 7 * mm, "结果来自仓库脚本的已保存输出 · 2026-07-23")
    canvas.drawRightString(PAGE_W - MARGIN_X, 7 * mm, f"{doc.page}")
    canvas.restoreState()


doc = BaseDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=MARGIN_X,
    rightMargin=MARGIN_X,
    topMargin=MARGIN_TOP,
    bottomMargin=MARGIN_BOTTOM,
    title="P0-P1进展：Ordered 1D empirical-transition MH 与 TSP 优化",
    author="sMTJ annealing accelerator project",
    subject="P-dit progress report and end-to-end TSP comparison",
)
frame = Frame(
    MARGIN_X,
    MARGIN_BOTTOM,
    CONTENT_W,
    PAGE_H - MARGIN_TOP - MARGIN_BOTTOM,
    leftPadding=0,
    rightPadding=0,
    topPadding=0,
    bottomPadding=0,
)
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=page_decor)])

story = []

# Cover
story += [
    Spacer(1, 20 * mm),
    para("P0 → P1 阶段进展", "cover_sub"),
    para("Ordered 1D empirical-transition MH<br/>与 TSP 直接排列优化", "cover_title"),
    Spacer(1, 5 * mm),
    callout(
        "<b>本报告采用端到端比较口径：</b>二值 p-bit 使用 one-hot QUBO/Ising 路径；"
        "P-dit 使用新的直接排列编码与 ordered 1D empirical-transition MH 更新核。"
        "因此对比目标是当前可实现方案的整体性能，不要求保持原算法编码不变。",
        LIGHT_ORANGE,
    ),
    Spacer(1, 9 * mm),
    metric_cards(
        [
            ("100%", "ordered 路线合法率"),
            ("24/24", "8 城市 exact 命中"),
            ("13/24", "12 城市 reference 命中"),
            ("1.37×10⁻¹⁶", "详细平衡最大相对误差"),
        ],
        LIGHT_GREEN,
    ),
    Spacer(1, 12 * mm),
    para(
        "范围：P0 直接 categorical P-dit、实测 sMTJ 转移驱动的 MH、P1 状态几何分析，"
        "以及新更新核在 8/12 城市 TSP 上相对二值 p-bit 的结果。",
        "body",
    ),
    Spacer(1, 4 * mm),
    para(
        "数据集：selected_curve · K=6 · 报告日期：2026-07-23",
        "small",
    ),
    PageBreak(),
]

# Executive summary and timeline
story += [
    para("1. 结论先行", "h1"),
    callout(
        "TSP 的关键改进不是把同一个 one-hot Ising 模型里的随机源从 Gray 改成 1D，"
        "而是同时改变<b>表示空间</b>与<b>更新核</b>：路线直接存为 permutation，"
        "器件跳变 k→l 通过 q<sub>kl</sub>=r<sub>k</sub>P<sub>kl</sub> 产生局部换位 proposal，"
        "再用 Hastings 比率纠正实测转移不对称。",
        LIGHT_BLUE,
    ),
    Spacer(1, 4 * mm),
    para("阶段路线", "h2"),
]

timeline_rows = [
    ["阶段", "解决的问题", "关键产出"],
    [
        "P0-A",
        "去除 Gray/one-hot 无效码与 penalty",
        "直接 categorical P-dit；所有状态天然合法",
    ],
    [
        "P0-B",
        "把实测器件动力学用于 proposal",
        "q<sub>kl</sub>=r<sub>k</sub>P<sub>kl</sub> + MH 校正",
    ],
    [
        "P1",
        "确定 K=6 器件最匹配的状态几何",
        "ordered 1D 的 locality=0.708，优于 Gray 的 0.363",
    ],
    [
        "TSP",
        "消除 one-hot 可行域瓶颈",
        "直接排列 + 可逆 cyclic transposition；raw feasible=100%",
    ],
]
story += [
    styled_table(timeline_rows, [28 * mm, 66 * mm, 77 * mm]),
    Spacer(1, 7 * mm),
    para("本报告如何使用“二值 p-bit 对比”", "h2"),
    bullet(
        "<b>主对比：</b>二值 p-bit/one-hot Ising 与 ordered 1D direct-permutation MH 的端到端结果。"
    ),
    bullet(
        "<b>不作为主结论：</b>固定 one-hot 编码、只替换 Gray/1D embedding 的 P1 公平消融。"
        "该实验说明“只换随机原语”无法解决 TSP 的表示瓶颈。"
    ),
    bullet(
        "<b>解释边界：</b>性能差异同时包含编码压缩、可行域重构、proposal locality 和 MH 更新核的收益，"
        "不能只归因于器件几何。"
    ),
    Spacer(1, 6 * mm),
    para("最终判断", "h2"),
    callout(
        "当前 ordered 1D empirical-transition MH 已经把器件物理、状态几何与 TSP 约束结构连成"
        "同一条算法路径：K=6 实测状态不再只是噪声源，而是实际 proposal state；"
        "每一步始终位于合法 permutation 空间内。",
        LIGHT_GREEN,
    ),
    PageBreak(),
]

# P0 direct categorical
p0_final = P0["final"]
enc = P0["encoding_comparison"]
story += [
    para("2. P0：从编码瓶颈到直接 categorical P-dit", "h1"),
    para(
        "P0 首先验证直接多值变量能够替代 Gray/one-hot 展开。14 个物品、3 个分区的验证问题中，"
        "每个物品只保存一个类别 z<sub>i</sub>∈{0,1,2}，更新使用 softmax 条件分布；"
        "不再需要无效码处理或 one-hot penalty。",
        "body",
    ),
    metric_cards(
        [
            (str(enc["direct_pdit_variables"]), "直接 P-dit 变量"),
            (str(enc["gray_binary_variables"]), "Gray 二值变量"),
            (str(enc["onehot_binary_variables"]), "one-hot 二值变量"),
            (f'{100*p0_final["ground_hit_rate"]:.0f}%', "2048 clocks ground hit"),
        ]
    ),
    Spacer(1, 6 * mm),
    *fig(
        "results/mtj_p0_direct_pdit_output/pdit_partition_performance.png",
        "图 1  P0 直接 P-dit 分区求解：随着物理时钟增加，ground hit rate 在 1024 次时达到 90%，"
        "median best error 在 512 次时已经降为 0。",
        88 * mm,
    ),
    para("P0 同时完成的物理数据处理修正", "h2"),
    bullet(
        "使用 duration-proportional integer apportionment 做 dwell-preserving resampling；"
        "136 个 dwell 的 135 条真实边界全部保留。"
    ),
    bullet(
        f'新方法 phantom transition 数为 {P0["resampling_validation"]["new_phantom_transition_count"]}；'
        "避免均匀索引重采样制造伪跳变。"
    ),
    bullet(
        "这一阶段证明“直接多值状态”能够消除编码无效区，但尚未决定 K=6 实测状态应采用何种几何解释。"
    ),
    PageBreak(),
]

# P0 four solver
summaries = {row["solver"]: row for row in FOUR["summaries"]}
story += [
    para("3. P0：从直接 P-dit 到实测转移 MH", "h1"),
    para(
        "随后用四类求解器区分“表示法收益”和“物理 proposal 收益”。其中 D 路径首次把实测"
        " q<sub>kl</sub>=r<sub>k</sub>P<sub>kl</sub> 作为直接 categorical proposal，"
        "并使用 Metropolis-Hastings 接受率修正正反向速率不对称。",
        "body",
    ),
    *fig(
        "results/mtj_p0_four_solver_output/four_solver_success.png",
        "图 2  P0 四求解器成功率曲线。C 是软件 direct P-dit，D 是 physical empirical-transition MH P-dit；"
        "A/B 为 Gray 二值路径。该图用于验证路线选择，不直接作为最终 TSP 性能结论。",
        86 * mm,
    ),
]

four_rows = [["求解器", "4000 updates 成功率", "median first hit", "有效状态率"]]
for key in [
    "physical_gray_pbit",
    "software_gray_glauber",
    "direct_pdit",
    "physical_mh_pdit",
]:
    row = summaries[key]
    four_rows.append(
        [
            row["label"],
            f'{100*row["success_probability"]:.1f}%',
            f'{row["median_first_hit_update_successes"]:.1f}',
            f'{100*row["valid_state_fraction"]:.0f}%',
        ]
    )
story += [
    styled_table(four_rows, [64 * mm, 37 * mm, 38 * mm, 32 * mm]),
    Spacer(1, 5 * mm),
    para("这一阶段得到的设计约束", "h2"),
    bullet(
        "MH 接受率必须包含 q<sub>lk</sub>/q<sub>kl</sub>，否则实测器件的非对称跳变会改变目标稳态分布。"
    ),
    bullet(
        "direct categorical 表示保证合法性，但 proposal 的几何仍需与器件最常见的跳变方向匹配。"
    ),
    bullet(
        "因此 P1 的任务从“比较更多离散状态”收敛为“识别 K=6 转移图实际支持的几何”。"
    ),
    PageBreak(),
]

# P1 geometry
geo = {row["geometry"]: row for row in P1["geometry_compatibility"]}
p1s = {row["geometry"]: row for row in P1["summaries"]}
story += [
    para("4. P1：K=6 实测转移更符合 ordered 1D", "h1"),
    para(
        "selected_curve 的 6 个状态按电阻/电导均值天然有序。P1 将实测 proposal rate 矩阵与"
        " Gray-coded binary、ordered 1D、isotropic six-state 三种几何比较。"
        "结果显示跳变强度主要集中在有序邻近状态，而不是各向同性全连接。",
        "body",
    ),
    *fig(
        "results/mtj_p1_empirical_transition_mh_output/p1_transition_geometry.png",
        "图 3  左：实测 q 矩阵；其余：三种几何的相对 affinity。ordered 1D 的带状局部结构"
        "与实测强跳变更一致；isotropic 把所有非对角状态视为等距，无法表达这种局部性。",
        58 * mm,
    ),
]

geo_rows = [
    ["几何", "proposal locality", "q 加权归一化距离", "rate vs -distance Pearson"],
    [
        "Gray binary",
        f'{geo["gray_binary"]["proposal_locality_score"]:.3f}',
        f'{geo["gray_binary"]["q_weighted_mean_normalized_distance"]:.3f}',
        f'{geo["gray_binary"]["symmetric_rate_vs_negative_distance_pearson"]:.3f}',
    ],
    [
        "Ordered 1D",
        f'<b>{geo["ordered_1d"]["proposal_locality_score"]:.3f}</b>',
        f'<b>{geo["ordered_1d"]["q_weighted_mean_normalized_distance"]:.3f}</b>',
        f'<b>{geo["ordered_1d"]["symmetric_rate_vs_negative_distance_pearson"]:.3f}</b>',
    ],
    [
        "Isotropic 6-state",
        "0.000",
        "1.000",
        "未定义（所有距离相同）",
    ],
]
story += [
    styled_table(geo_rows, [42 * mm, 39 * mm, 49 * mm, 41 * mm]),
    Spacer(1, 5 * mm),
    callout(
        "为什么 isotropic 能绕开 one-hot，而 ordered 1D 过去不行？问题不在“1D 本身”，"
        "而在旧实现仍把 TSP 保留为 N² 个 one-hot 二值变量。新的直接 permutation 映射"
        "让 ordered 1D 同样无需 one-hot，同时保留器件真实的局部有序跳变。",
        LIGHT_ORANGE,
    ),
    PageBreak(),
]

# P1 sampler
story += [
    para("5. P1：empirical-transition MH 的采样验证", "h1"),
    para(
        "在 4 变量、6 状态、每种几何 64 条轨迹、每条 8000 updates 的枚举验证问题上，"
        "三种方法使用相同 q<sub>kl</sub> proposal 与 MH 框架，仅能量几何不同。"
        "ordered 1D 同时获得更高的 event acceptance 与更短的能量相关时间。",
        "body",
    ),
    *fig(
        "results/mtj_p1_empirical_transition_mh_output/p1_sampling_metrics.png",
        "图 4  P1 采样指标。ordered 1D 的 MH acceptance/event 约 0.490，"
        "median energy IAT 约 52.3 updates；Gray 分别约 0.219 和 121.3。",
        108 * mm,
    ),
]
p1_rows = [
    ["几何", "stationary KL", "MH acceptance/event", "median IAT", "success"],
]
for key in ["gray_binary", "ordered_1d", "isotropic_6state"]:
    row = p1s[key]
    p1_rows.append(
        [
            row["label"],
            f'{row["stationary_kl_empirical_to_exact"]:.4f}',
            f'{row["mh_acceptance_given_event"]:.3f}',
            f'{row["median_energy_iat_updates"]:.1f}',
            f'{100*row["success_probability"]:.1f}%',
        ]
    )
story += [
    styled_table(p1_rows, [48 * mm, 31 * mm, 38 * mm, 29 * mm, 25 * mm]),
    Spacer(1, 5 * mm),
    para(
        "解释：ordered 1D 不一定在任意目标能量上都最快，但它最忠实地利用了当前 K=6 器件的"
        "局部跳变结构。该结论为 TSP 路线空间中的“跳变距离 → 换位距离”映射提供了物理依据。",
        "body",
    ),
    PageBreak(),
]

# Kernel design
validation = TSP["kernel_validation"]
story += [
    para("6. TSP 更新核：直接排列 + ordered 1D empirical-transition MH", "h1"),
    para("状态与 proposal", "h2"),
    callout(
        "路线状态：π=[0,π₁,…,π<sub>N-1</sub>]，city 0 固定以去除旋转简并。"
        "器件事件 k→l 的速率为 q<sub>kl</sub>=r<sub>k</sub>P<sub>kl</sub>；"
        "|l-k| 映射为固定起点之外的 cyclic transposition distance。",
        LIGHT_BLUE,
    ),
    Spacer(1, 5 * mm),
    para("接受率", "h2"),
    callout(
        "A[(π,k)→(π′,l)] = min{1, exp[-β(L(π′)-L(π))] · q<sub>lk</sub>/q<sub>kl</sub>}",
        LIGHT_ORANGE,
    ),
    Spacer(1, 5 * mm),
    para("为什么它消除了 one-hot 瓶颈", "h2"),
]
kernel_rows = [
    ["性质", "二值 p-bit / one-hot Ising", "新 ordered 1D permutation MH"],
    [
        "状态规模",
        "N² 个二值变量",
        "N 个 route entries；固定起点后仅搜索 (N-1)! 路线",
    ],
    [
        "合法性",
        "靠 row/column penalty 逼近",
        "任意 transposition 后仍是 permutation",
    ],
    [
        "器件作用",
        "随机更新/偏置源",
        "k→l 是真实 q<sub>kl</sub> proposal state",
    ],
    [
        "可逆性",
        "由 Ising 更新规则决定",
        "无向状态对产生相同 swap，MH 校正 q 不对称",
    ],
]
story += [
    styled_table(kernel_rows, [30 * mm, 67 * mm, 74 * mm]),
    Spacer(1, 7 * mm),
    para("核验证", "h2"),
    metric_cards(
        [
            ("630", "检查的详细平衡 flux cases"),
            (
                f'{validation["theoretical_db_max_relative_flux_error"]:.2e}',
                "最大相对 flux 误差",
            ),
            ("通过", "k↔l 下 route move 可逆"),
            ("通过", "相邻器件边双向"),
        ],
        LIGHT_GREEN,
    ),
    Spacer(1, 6 * mm),
    bullet(
        "distance-1 swap 具有正概率；相邻 transposition 可生成所有固定起点 permutation，因此路线空间可达。"
    ),
    bullet(
        "接受时同时更新路线和器件状态；拒绝时二者保持不变，联合链满足目标详细平衡。"
    ),
    PageBreak(),
]

# TSP results
tsp_rows_by = {(r["n_cities"], r["method"]): r for r in TSP["method_summary"]}
story += [
    para("7. TSP 端到端性能：新编码更新核 vs 二值 p-bit", "h1"),
    para(
        "这里采用用户指定的整体方案比较，不做“只改变一个变量”的公平消融。"
        "二值基线是 one-hot QUBO/Ising 路径；P-dit 路径同时采用直接 permutation 编码与"
        " ordered 1D empirical-transition MH。每个城市规模、每种方法各 24 runs。"
        "保存实验恰好使用相同的 121,500 algorithmic update attempts，但两类 update 的物理/软件含义不同。",
        "body",
    ),
    *fig(
        "results/benchmarks/tsp_binary_pbit_vs_ordered_1d_mh/tsp_method_comparison.png",
        "图 5  TSP 主比较。左：2-opt 前路线质量；中：raw assignment 可行率；右：表示变量数。"
        "图中的二值基线标记为 Binary p-bit + one-hot Ising。",
        67 * mm,
    ),
]

result_rows = [
    [
        "城市",
        "方法",
        "变量",
        "raw feasible",
        "median route/ref",
        "raw reference hit",
    ]
]
for n in [8, 12]:
    for method, label in [
        ("binary_pbit_onehot_ising", "Binary p-bit / one-hot"),
        ("ordered_1d_empirical_mh_pdit", "Ordered 1D permutation MH"),
    ]:
        row = tsp_rows_by[(n, method)]
        result_rows.append(
            [
                str(n),
                label,
                str(row["variables"]),
                f'{100*row["raw_feasible_rate"]:.1f}%',
                f'{row["median_route_ratio"]:.3f}',
                f'{100*row["raw_reference_hit_rate"]:.1f}%',
            ]
        )
story += [
    styled_table(
        result_rows,
        [16 * mm, 53 * mm, 19 * mm, 28 * mm, 30 * mm, 30 * mm],
    ),
    Spacer(1, 5 * mm),
    para("核心读数", "h2"),
    bullet(
        "<b>8 城市：</b>ordered MH 为 8 个变量，24/24 raw 命中 exact reference；"
        "二值 one-hot 为 64 个变量，raw reference hit 为 0%，median route/ref=1.263。"
    ),
    bullet(
        "<b>12 城市：</b>ordered MH 为 12 个变量，raw feasible=100%，13/24 达到 reference；"
        "二值 one-hot 为 144 个变量，raw feasible=0%，median one-hot violation=10.5。"
    ),
    bullet(
        "12 城市 reference 来自 nearest-neighbor all-start + 2-opt，不保证全局最优；"
        "ratio=1 表示达到该参考值。"
    ),
    PageBreak(),
]

# Route example and interpretation
story += [
    para("8. 路线实例、收益来源与边界", "h1"),
    *fig(
        "results/benchmarks/tsp_binary_pbit_vs_ordered_1d_mh/tsp_best_routes_comparison.png",
        "图 6  12 城市最佳保存路线。左：二值 one-hot 解码路线，长度 51.581；"
        "右：ordered 1D empirical-MH 路线，长度 34.645，达到当前 reference。",
        80 * mm,
    ),
    para("收益来自三层共同作用", "h2"),
]
benefit_rows = [
    ["层级", "机制", "观测效果"],
    [
        "表示",
        "N² one-hot → N-entry permutation",
        "8/12 城市变量减少 8×/12×；搜索始终合法",
    ],
    [
        "几何",
        "|l-k| 控制换位距离",
        "accepted 平均距离约 1.10，保持局部搜索",
    ],
    [
        "动力学",
        "实测 q<sub>kl</sub> proposal + Hastings 校正",
        "保留器件非对称跳变，同时维持目标详细平衡",
    ],
]
story += [
    styled_table(benefit_rows, [27 * mm, 65 * mm, 79 * mm]),
    Spacer(1, 6 * mm),
    para("限制与正确解读", "h2"),
    bullet(
        "这是端到端算法比较，不是硬件延迟或能效比较；Python wall time 不代表未来 sMTJ 阵列速度。"
    ),
    bullet(
        "两条路径的 update 语义不同：one-hot 路径更新单个 Ising spin，ordered 路径进行一次"
        " empirical proposal-clock tick，因此不应仅用软件耗时判断器件优劣。"
    ),
    bullet(
        "样本量为每格 24 runs；8 城市 reference 是 exact，12 城市 reference 是启发式。"
        "更大规模结论仍需增加实例、随机种子及 confidence interval。"
    ),
    bullet(
        "P1 固定编码实验仍有诊断价值：它说明 ordered 1D 更符合器件几何；"
        "但它不再用于本报告的 TSP 性能主张。"
    ),
    PageBreak(),
]

# Final
story += [
    para("9. 当前结论与下一步", "h1"),
    callout(
        "<b>当前结论：</b>ordered 1D empirical-transition MH 已完成从器件 K=6 转移统计到"
        " TSP 路线 proposal 的闭环。直接 permutation 编码从结构上消除了 one-hot 可行域瓶颈；"
        "实测 q 驱动与 Hastings 校正让更新核同时满足器件一致性和理论可逆性。",
        LIGHT_GREEN,
    ),
    Spacer(1, 7 * mm),
    para("建议的后续验证优先级", "h2"),
]
next_rows = [
    ["优先级", "工作", "目的"],
    [
        "1",
        "扩展 16/20/30 城市、多实例、多随机种子",
        "验证表示优势随规模的稳定性",
    ],
    [
        "2",
        "报告 success-vs-clock、proposal event 与 accepted swap 三种预算",
        "避免不同 update 语义造成误读",
    ],
    [
        "3",
        "加入 adaptive β 与 distance-aware proposal mixture",
        "平衡局部精修与长距离逃逸",
    ],
    [
        "4",
        "建立器件并行映射、latency/energy 模型",
        "把算法优势转化为硬件指标",
    ],
]
story += [
    styled_table(next_rows, [18 * mm, 83 * mm, 70 * mm]),
    Spacer(1, 8 * mm),
    para("复现实验与图像来源", "h2"),
    para(
        "P0：<font name='Helvetica'>src/mtj_p0_direct_pdit_partition.py</font>、"
        "<font name='Helvetica'>src/mtj_p0_four_solver_benchmark.py</font><br/>"
        "P1：<font name='Helvetica'>src/mtj_p1_empirical_transition_mh.py</font><br/>"
        "TSP：<font name='Helvetica'>src/benchmarks/tsp_binary_pbit_vs_ordered_1d_mh.py</font><br/>"
        "报告中的图 1-6 均直接来自上述脚本保存在 <font name='Helvetica'>results/</font> 下的 PNG，"
        "未重新绘制或修改数据。",
        "body",
    ),
    Spacer(1, 5 * mm),
    para("参考", "h2"),
    para(
        "P-dits: Probabilistic d-dimensional bits for extended-variable probabilistic computing. "
        "DOI: 10.1103/4ngx-cmz7",
        "small",
    ),
    Spacer(1, 7 * mm),
    callout(
        "一句话总结：TSP 的突破来自“ordered 1D 器件几何 + empirical-transition MH + "
        "permutation-preserving 编码”的组合，而不是在原 one-hot 模型中单独替换随机源。",
        LIGHT_ORANGE,
    ),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.build(story)
print(OUT)
